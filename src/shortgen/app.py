"""Application services: the one API behind both the CLI and the MCP server.

Each function takes plain arguments and returns plain models, so front ends
stay thin and can never drift apart in behaviour.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path

from shortgen.config import PROJECT_CONFIG_NAME, AppConfig, load_config
from shortgen.device import detect_device
from shortgen.doctor import run_doctor
from shortgen.domain.models import AspectRatio, FrozenModel, Project
from shortgen.errors import ConfigError, ShortgenError
from shortgen.pipeline.catalog import default_stages
from shortgen.pipeline.context import PipelineContext
from shortgen.pipeline.manifest import Manifest, ManifestStore
from shortgen.pipeline.progress import LogReporter, Reporter
from shortgen.pipeline.runner import PipelineRunner, StagePreview, StageReport
from shortgen.pipeline.stage import StageName
from shortgen.pipeline.workspace import Workspace, validate_input_file
from shortgen.process import CommandRunner, SubprocessRunner
from shortgen.providers.health import HealthCheck
from shortgen.providers.spec import ProviderKind
from shortgen.registry import ModelRegistry, load_registry

log = logging.getLogger(__name__)

SCRIPT_SUFFIXES = frozenset({".md", ".txt"})
DEFAULT_PROJECTS_DIR = "projects"
_TOPIC_SLUG_CHARS = 40
_TORCH = "torch"
_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")
_PROJECT_CONFIG_HEADER = (
    "# Project-specific overrides. Any key from shortgen's configs/default.yaml\n"
    "# can be set here, for example:\n"
    "#\n"
    "# providers:\n"
    "#   planner: ollama\n"
    "# characters:\n"
    "#   - name: Mira\n"
    "#     description: a young keeper in a red scarf\n"
    "#     reference_image: refs/mira.png\n"
)


class StagePreviewModel(FrozenModel):
    """JSON-friendly view of one stage's freshness."""

    stage: str
    total: int
    fresh: int
    pending: int
    failed: int
    blocked_reason: str | None
    note: str | None

    @classmethod
    def from_preview(cls, preview: StagePreview) -> StagePreviewModel:
        """Convert the runner's preview into its serialisable form."""
        return cls(
            stage=preview.stage.value,
            total=preview.total,
            fresh=preview.fresh,
            pending=preview.pending,
            failed=preview.failed,
            blocked_reason=preview.blocked_reason,
            note=preview.note,
        )


class ProjectStatus(FrozenModel):
    """A snapshot of a project for ``shortgen status``."""

    name: str
    project_dir: str
    scenes: int
    stages: list[StagePreviewModel]
    final_video: str | None


class StageSummary(FrozenModel):
    """What one stage did during a run."""

    stage: str
    total: int
    skipped: int
    completed: int
    failed: list[str]


class RunSummary(FrozenModel):
    """The result of running some stages, for printing or returning over MCP."""

    project_dir: str
    stages: list[StageSummary]
    final_video: str | None


def slugify(text: str) -> str:
    """A filesystem-friendly project name."""
    return _SLUG_SEPARATORS.sub("-", text.lower()).strip("-") or "project"


def init_project(
    project_dir: Path,
    *,
    script_file: Path | None = None,
    topic: str | None = None,
    name: str | None = None,
    aspect: AspectRatio | None = None,
) -> Manifest:
    """Create a project directory with its manifest. Loads no models."""
    workspace = Workspace(project_dir)
    store = ManifestStore(workspace)
    if store.exists():
        raise ShortgenError(f"a project already exists at {workspace.root}")
    if script_file is None and topic is None and workspace.script_path.is_file():
        script_file = workspace.script_path  # a folder that already holds its script.md
    if (script_file is None) == (topic is None):
        raise ShortgenError("give exactly one of a script file or a topic")
    script = _read_script(script_file) if script_file else ""
    manifest = Manifest(
        project=Project(
            name=name or slugify(workspace.root.name),
            script=script,
            topic=topic,
            created_at=datetime.now(UTC),
        )
    )
    workspace.root.mkdir(parents=True, exist_ok=True)
    if script and script_file is not None and script_file.resolve() != workspace.script_path:
        workspace.script_path.write_text(script, encoding="utf-8")
    config_path = workspace.root / PROJECT_CONFIG_NAME
    if not config_path.exists():  # never clobber a config the folder came with
        overrides = f"\nproject:\n  aspect: '{aspect.value}'\n" if aspect else ""
        config_path.write_text(_PROJECT_CONFIG_HEADER + overrides, encoding="utf-8")
    store.save(manifest)
    log.info("created project '%s' at %s", manifest.project.name, workspace.root)
    return manifest


def sync_script(project_dir: Path, script_file: Path) -> bool:
    """Adopt an edited script file into an existing project. Returns True if it changed."""
    workspace = Workspace(project_dir)
    store = ManifestStore(workspace)
    manifest = store.load()
    script = _read_script(script_file)
    if script.strip() == manifest.project.script.strip():
        return False
    manifest.project.script = script
    manifest.project.topic = None
    if script_file.resolve() != workspace.script_path.resolve():
        workspace.script_path.write_text(script, encoding="utf-8")
    store.save(manifest)
    log.info("script changed; affected stages will be regenerated")
    return True


def adopt_script_edits(project_dir: Path) -> bool:
    """Take an edited ``script.md`` inside an existing project into its manifest.

    Every command opens a project through :func:`build_context`, which calls
    this, so editing the script and then running ``generate`` or ``status``
    sees the edit exactly as ``run`` does. Returns True if the script changed.
    """
    workspace = Workspace(project_dir)
    store = ManifestStore(workspace)
    if not store.exists() or not workspace.script_path.is_file():
        return False
    if not store.load().project.script.strip():
        return False  # a topic project: its script is written by the planner, not adopted
    return sync_script(project_dir, workspace.script_path)


def build_context(
    project_dir: Path,
    *,
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    reporter: Reporter | None = None,
    commands: CommandRunner | None = None,
) -> PipelineContext:
    """Wire configuration, registry, device and I/O into a pipeline context.

    A folder that holds a ``script.md`` is opened the way ``run`` opens it: set
    up in place if it is new, and its script adopted if it was edited. So
    ``generate`` and ``status`` work on a brand-new story folder too.
    """
    workspace = Workspace(project_dir)
    if workspace.script_path.is_file():
        _prepare_directory(project_dir)
    config = load_config(project_dir=workspace.root, config_file=config_file, env=env)
    registry = load_registry(config)
    device = detect_device(config.runtime.device)
    # A CPU fallback only matters if a configured model actually runs on torch.
    level = logging.WARNING if _uses_torch(config, registry) else logging.DEBUG
    for warning in device.warnings:
        log.log(level, warning)
    return PipelineContext(
        config=config,
        workspace=workspace,
        store=ManifestStore(workspace),
        registry=registry,
        device=device,
        commands=commands or SubprocessRunner(),
        reporter=reporter or LogReporter(),
        env=env or {},
    )


def run_pipeline(
    ctx: PipelineContext,
    stages: Collection[StageName] | None = None,
    *,
    force: bool = False,
    scene_ids: Collection[str] | None = None,
) -> list[StageReport]:
    """Run stages (default: the whole pipeline) against the context's project."""
    _validate_providers(ctx)
    return PipelineRunner(ctx, default_stages()).run(stages, force=force, scene_ids=scene_ids)


def execute(
    project_dir: Path,
    stages: Collection[StageName] | None = None,
    *,
    force: bool = False,
    scene_ids: Collection[str] | None = None,
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    reporter: Reporter | None = None,
) -> RunSummary:
    """Build a context for ``project_dir``, run the stages and summarise the outcome.

    A plan that is out of date (the script or the planning settings changed) is
    brought up to date first even when ``stages`` leaves it out. Otherwise
    ``generate --stage image`` after editing a picture description would draw
    the old description.
    """
    ctx = build_context(project_dir, config_file=config_file, env=env, reporter=reporter)
    reports: list[StageReport] = []
    if stages is not None and StageName.PLAN not in stages and _plan_is_stale(ctx):
        log.info("the script or planning settings changed; planning again first")
        reports = run_pipeline(ctx, [StageName.PLAN])
    reports += run_pipeline(ctx, stages, force=force, scene_ids=scene_ids)
    return RunSummary(
        project_dir=str(ctx.workspace.root),
        stages=[
            StageSummary(
                stage=report.stage.value,
                total=report.total,
                skipped=report.skipped,
                completed=report.completed,
                failed=report.failed,
            )
            for report in reports
        ],
        final_video=_final_video(ctx),
    )


def prepare_project(
    source: Path | None,
    *,
    project_dir: Path | None = None,
    topic: str | None = None,
    aspect: AspectRatio | None = None,
    projects_root: Path = Path(DEFAULT_PROJECTS_DIR),
) -> Path:
    """Resolve what ``shortgen run`` was pointed at into a ready project directory.

    ``source`` may be a project directory or a script file. A script (or a
    topic) creates the project on first use; on later runs an edited script is
    adopted so only the affected scenes regenerate. A directory that holds a
    ``script.md`` but no manifest yet (such as the bundled examples) becomes a
    project in place, and editing a project's own ``script.md`` works the same
    way as editing the original file.
    """
    if source is not None and source.is_dir():
        return _prepare_directory(source)
    if source is None and topic is None:
        if project_dir is None:
            raise ShortgenError("give a script file, a project directory or --topic")
        return project_dir
    label = source.stem if source is not None else (topic or "")[:_TOPIC_SLUG_CHARS]
    target = project_dir or projects_root / slugify(label)
    if not ManifestStore(Workspace(target)).exists():
        init_project(
            target, script_file=source, topic=topic if source is None else None, aspect=aspect
        )
    elif source is not None:
        sync_script(target, source)
    return target


def _prepare_directory(directory: Path) -> Path:
    workspace = Workspace(directory)
    if not ManifestStore(workspace).exists():
        if not workspace.script_path.is_file():
            raise ShortgenError(
                f"{directory} is not a project: it has neither {workspace.manifest_path.name} "
                f"nor {workspace.script_path.name}"
            )
        init_project(directory)
    else:
        adopt_script_edits(directory)
    return directory


def project_status(ctx: PipelineContext) -> ProjectStatus:
    """Summarise what is done, stale, failed or blocked. Runs nothing."""
    _validate_providers(ctx)
    manifest = ctx.store.load()
    previews = PipelineRunner(ctx, default_stages()).preview()
    return ProjectStatus(
        name=manifest.project.name,
        project_dir=str(ctx.workspace.root),
        scenes=len(manifest.project.scenes),
        stages=[StagePreviewModel.from_preview(preview) for preview in previews],
        final_video=_final_video(ctx),
    )


def doctor(
    *,
    project_dir: Path | None = None,
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    commands: CommandRunner | None = None,
) -> list[HealthCheck]:
    """Check ffmpeg, the device, memory and every configured model. Loads no weights."""
    config = load_config(project_dir=project_dir, config_file=config_file, env=env)
    device = detect_device(config.runtime.device)
    return run_doctor(
        config, load_registry(config), device, commands or SubprocessRunner(), env or {}
    )


def _plan_is_stale(ctx: PipelineContext) -> bool:
    """Whether the stored scene plan no longer matches the script and settings."""
    previews = PipelineRunner(ctx, default_stages()).preview()
    plan = next(preview for preview in previews if preview.stage is StageName.PLAN)
    return plan.fresh < plan.total


def _uses_torch(config: AppConfig, registry: ModelRegistry) -> bool:
    """Whether any configured provider is a PyTorch model (unknown names count as no)."""
    for kind in ProviderKind:
        name = str(getattr(config.providers, kind.value))
        spec = next((s for s in registry.specs(kind) if s.name == name), None)
        if spec is not None and _TORCH in spec.requires:
            return True
    return False


def _final_video(ctx: PipelineContext) -> str | None:
    final = ctx.workspace.resolve(Workspace.final_video)
    return str(final) if final.is_file() else None


def _validate_providers(ctx: PipelineContext) -> None:
    """Fail fast, before any work, if a configured provider name is wrong."""
    for kind in ProviderKind:
        ctx.fingerprint(kind)


def _read_script(script_file: Path) -> str:
    path = validate_input_file(script_file, SCRIPT_SUFFIXES)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"script file is empty: {script_file}")
    return text + "\n"
