"""The resumable runner.

For each stage, in order: compute its units, skip the ones whose inputs hash
and output files are unchanged, and run the rest one at a time. The manifest
is saved before and after every unit, so the pipeline can be killed at any
moment and resumed with the same command.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from shortgen.domain.models import Job, JobStatus
from shortgen.errors import ProviderUnavailableError, StageError, StageFailedError, TransientError
from shortgen.pipeline.context import PipelineContext
from shortgen.pipeline.manifest import Manifest
from shortgen.pipeline.retry import call_with_retry
from shortgen.pipeline.stage import (
    STAGE_ORDER,
    ProviderHandle,
    Stage,
    StageName,
    Targets,
    UnitRun,
    WorkUnit,
)
from shortgen.providers.interfaces import Provider

log = logging.getLogger(__name__)
_OLD_PLAN_NOTE = "counted against the old plan; redone scenes show after planning"

AnyStage = Stage[Any]
AnyUnit = WorkUnit[Any]


@dataclass(frozen=True)
class StageReport:
    """What happened to one stage during a run."""

    stage: StageName
    total: int
    skipped: int
    completed: int
    failed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StagePreview:
    """A stage's state without running it, for ``shortgen status``."""

    stage: StageName
    total: int = 0
    fresh: int = 0
    failed: int = 0
    blocked_reason: str | None = None
    #: Set when the counts cannot be trusted yet, with the reason.
    note: str | None = None

    @property
    def pending(self) -> int:
        """Units that would run on the next invocation."""
        return self.total - self.fresh


class PipelineRunner:
    """Runs stages against one project, skipping everything already up to date."""

    def __init__(self, ctx: PipelineContext, stages: Sequence[AnyStage]) -> None:
        self._ctx = ctx
        self._stages = {stage.name: stage for stage in stages}

    def run(
        self,
        selected: Collection[StageName] | None = None,
        *,
        force: bool = False,
        scene_ids: Collection[str] | None = None,
    ) -> list[StageReport]:
        """Run the selected stages (default: all) in pipeline order.

        ``force`` re-runs the selected units even if they are up to date.
        ``scene_ids`` limits per-scene stages to those scenes. Raises
        :class:`StageFailedError` after a stage in which any unit failed;
        the other units of that stage are still attempted first.
        """
        order = [name for name in STAGE_ORDER if name in self._stages]
        chosen = [name for name in order if selected is None or name in selected]
        if not chosen:
            return []
        manifest = self._ctx.store.load()
        reports: list[StageReport] = []
        for name in order[: order.index(chosen[-1]) + 1]:
            stage = self._stages[name]
            if name not in chosen:
                self._warn_if_stale(stage, manifest)
                continue
            report = self._run_stage(stage, manifest, force, scene_ids)
            reports.append(report)
            if report.failed:
                raise StageFailedError(name.value, report.failed)
        return reports

    def preview(self) -> list[StagePreview]:
        """Describe every stage's freshness without doing any work."""
        manifest = self._ctx.store.load()
        previews: list[StagePreview] = []
        for name in STAGE_ORDER:
            stage = self._stages.get(name)
            if stage is None:
                continue
            try:
                units = stage.units(self._ctx, manifest)
            except StageError as blocked:
                previews.append(StagePreview(stage=name, blocked_reason=str(blocked)))
                continue
            failed = sum(1 for unit in units if self._status(unit, manifest) is JobStatus.FAILED)
            fresh = sum(1 for unit in units if self._is_fresh(unit, manifest))
            previews.append(StagePreview(stage=name, total=len(units), fresh=fresh, failed=failed))
        return self._flag_counts_from_an_old_plan(previews)

    @staticmethod
    def _flag_counts_from_an_old_plan(previews: list[StagePreview]) -> list[StagePreview]:
        """Scenes come from the plan, so a stale plan makes every later count provisional."""
        plan = next((p for p in previews if p.stage is StageName.PLAN), None)
        if plan is None or plan.pending == 0:
            return previews
        return [
            replace(preview, note=_OLD_PLAN_NOTE)
            if preview.stage is not StageName.PLAN and preview.blocked_reason is None
            else preview
            for preview in previews
        ]

    def _run_stage(
        self,
        stage: AnyStage,
        manifest: Manifest,
        force: bool,
        scene_ids: Collection[str] | None,
    ) -> StageReport:
        units = stage.units(self._ctx, manifest)
        if scene_ids is not None:
            units = [unit for unit in units if unit.scene_id is None or unit.scene_id in scene_ids]
        pending = [unit for unit in units if force or not self._is_fresh(unit, manifest)]
        reporter = self._ctx.reporter
        reporter.stage_started(stage.name.value, len(units), len(pending))
        handle = ProviderHandle(self._provider_factory(stage))
        failed: list[str] = []
        completed = 0
        try:
            for unit in pending:
                error = self._run_unit(stage, unit, manifest, handle)
                reporter.unit_finished(stage.name.value, unit.label, error is None)
                if error is None:
                    completed += 1
                    continue
                failed.append(unit.label)
                if isinstance(error, ProviderUnavailableError):
                    log.error("stopping stage %s: its provider cannot run here", stage.name.value)
                    break
        finally:
            handle.close()
            reporter.stage_finished(stage.name.value)
        return StageReport(stage.name, len(units), len(units) - len(pending), completed, failed)

    def _run_unit(
        self, stage: AnyStage, unit: AnyUnit, manifest: Manifest, handle: ProviderHandle
    ) -> Exception | None:
        """Run one unit to a recorded outcome. Returns the error if it failed."""
        job = Job(
            stage=unit.stage.value,
            scene_id=unit.scene_id,
            status=JobStatus.RUNNING,
            inputs_hash=unit.inputs_hash,
            started_at=datetime.now(UTC),
        )
        manifest.jobs[unit.key] = job
        self._ctx.store.save(manifest)
        targets = Targets(self._ctx.workspace, unit.outputs)
        run = UnitRun(ctx=self._ctx, manifest=manifest, provider=handle, targets=targets)

        def attempt() -> None:
            job.attempts += 1
            targets.discard()
            stage.execute(unit, run)

        failure: Exception | None = None
        try:
            call_with_retry(
                attempt, self._ctx.retry_policy, sleep=self._ctx.sleep, on_retry=self._log_retry
            )
            job.outputs = targets.commit(unit.scene_id)
            job.status = JobStatus.DONE
        except KeyboardInterrupt:
            job.status = JobStatus.INTERRUPTED
            raise
        except Exception as error:  # recorded in the manifest and reported, never dropped
            job.status = JobStatus.FAILED
            job.error = f"{type(error).__name__}: {error}"
            log.exception("%s failed for %s", unit.stage.value, unit.label)
            failure = error
        finally:
            targets.discard()
            job.finished_at = datetime.now(UTC)
            self._ctx.store.save(manifest)
        return failure

    def _provider_factory(self, stage: AnyStage) -> Callable[[], Provider] | None:
        kind = stage.provider_kind
        if kind is None:
            return None
        return lambda: self._ctx.create_provider(kind)

    def _is_fresh(self, unit: AnyUnit, manifest: Manifest) -> bool:
        """Done, with the same inputs, and every output still on disk."""
        job = manifest.jobs.get(unit.key)
        if job is None or job.status is not JobStatus.DONE or job.inputs_hash != unit.inputs_hash:
            return False
        return all(self._ctx.workspace.resolve(spec.path).is_file() for spec in unit.outputs)

    @staticmethod
    def _status(unit: AnyUnit, manifest: Manifest) -> JobStatus:
        job = manifest.jobs.get(unit.key)
        return job.status if job is not None else JobStatus.PENDING

    def _warn_if_stale(self, stage: AnyStage, manifest: Manifest) -> None:
        """A skipped upstream stage that is out of date would feed stale files downstream."""
        try:
            stale = [
                u.label for u in stage.units(self._ctx, manifest) if not self._is_fresh(u, manifest)
            ]
        except StageError:
            return  # the selected stage will raise the precise error itself
        if stale:
            log.warning(
                "stage %s is out of date for %s but was not selected; "
                "downstream stages will use its previous outputs",
                stage.name.value,
                ", ".join(stale),
            )

    @staticmethod
    def _log_retry(attempt: int, error: TransientError, wait: float) -> None:
        log.warning("attempt %d failed (%s); retrying in %.1fs", attempt, error, wait)
