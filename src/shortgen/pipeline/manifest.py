"""The project manifest: one JSON file that records the plan and every job."""

from __future__ import annotations

from pydantic import Field, ValidationError

from shortgen.domain.models import Asset, AssetKind, Job, JobStatus, Project, StrictModel
from shortgen.errors import ManifestError, StageError
from shortgen.pipeline.workspace import Workspace, atomic_write_text

SCHEMA_VERSION = 1
_PROJECT_SCOPE = "_"


def job_key(stage: str, scene_id: str | None) -> str:
    """Manifest key of a unit of work, e.g. ``image:s003`` or ``render:_``."""
    return f"{stage}:{scene_id or _PROJECT_SCOPE}"


class Manifest(StrictModel):
    """Everything shortgen knows about a project."""

    schema_version: int = SCHEMA_VERSION
    project: Project
    jobs: dict[str, Job] = Field(default_factory=dict)

    def job(self, stage: str, scene_id: str | None = None) -> Job | None:
        """The recorded job for a unit of work, if it ever ran."""
        return self.jobs.get(job_key(stage, scene_id))

    def asset(self, stage: str, scene_id: str | None, kind: AssetKind) -> Asset:
        """An output of a finished upstream job.

        Raises :class:`StageError` when the upstream stage has not completed,
        which is how a stage reports that its prerequisites are missing.
        """
        job = self.job(stage, scene_id)
        scope = scene_id or "the project"
        if job is None or job.status is not JobStatus.DONE:
            raise StageError(f"stage '{stage}' has not completed for {scope}; run it first")
        for asset in job.outputs:
            if asset.kind is kind:
                return asset
        raise ManifestError(f"job '{job_key(stage, scene_id)}' recorded no {kind.value} output")


class ManifestStore:
    """Loads and atomically saves the manifest of one workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self._path = workspace.manifest_path

    def exists(self) -> bool:
        """Whether this directory has been initialised as a project."""
        return self._path.is_file()

    def load(self) -> Manifest:
        """Read and validate the manifest."""
        if not self.exists():
            raise ManifestError(
                f"no project at {self._path.parent} (missing {self._path.name}); "
                "create one with `shortgen init`"
            )
        try:
            manifest = Manifest.model_validate_json(self._path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise ManifestError(f"cannot read manifest {self._path}: {exc}") from exc
        if manifest.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"manifest schema {manifest.schema_version} is not supported "
                f"(this version reads schema {SCHEMA_VERSION})"
            )
        return manifest

    def save(self, manifest: Manifest) -> None:
        """Persist the manifest; safe against interruption at any moment."""
        atomic_write_text(self._path, manifest.model_dump_json(indent=2) + "\n")
