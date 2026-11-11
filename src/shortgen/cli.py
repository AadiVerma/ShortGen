"""The ``shortgen`` command line. A thin layer over :mod:`shortgen.app`."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ParamSpec, TypeVar

import typer
from rich.console import Console

from shortgen import __version__, app
from shortgen.config import load_config, process_environment
from shortgen.console import RichReporter, print_doctor, print_status, print_summary
from shortgen.doctor import has_failures
from shortgen.domain.models import AspectRatio
from shortgen.errors import ShortgenError, StageFailedError
from shortgen.logging import configure_logging
from shortgen.pipeline.catalog import GENERATE_STAGES, RENDER_STAGES
from shortgen.pipeline.stage import StageName

P = ParamSpec("P")
R = TypeVar("R")

EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130

cli = typer.Typer(
    name="shortgen",
    help="Turn a short story or script into a narrated, captioned vertical video. Locally.",
    no_args_is_help=True,
    add_completion=False,
)

out = Console()  # results: stdout
err = Console(stderr=True)  # logs, progress and errors: stderr

ProjectArg = Annotated[Path, typer.Argument(help="Project directory.", show_default=False)]
ForceOpt = Annotated[bool, typer.Option("--force", help="Redo work even if it is up to date.")]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="Extra YAML config merged over the defaults.")
]


@dataclass(frozen=True)
class State:
    """Options shared by every command, carried on the Typer context."""

    config_file: Path | None


def handle_errors(command: Callable[P, R]) -> Callable[P, R]:
    """Turn expected failures into one-line messages and the right exit code."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except KeyboardInterrupt:
            err.print(
                "\n[yellow]Interrupted.[/yellow] Progress is saved; run the same command to resume."
            )
            raise typer.Exit(EXIT_INTERRUPTED) from None
        except StageFailedError as failure:
            err.print(f"[red]error:[/red] {failure}", soft_wrap=True)
            err.print(
                "Other scenes were still processed. Fix the cause and re-run to retry only these."
            )
            raise typer.Exit(EXIT_FAILURE) from None
        except ShortgenError as failure:
            err.print(f"[red]error:[/red] {failure}", soft_wrap=True)
            raise typer.Exit(EXIT_FAILURE) from None

    return wrapper


def _version(value: bool) -> None:
    if value:
        out.print(f"shortgen {__version__}")
        raise typer.Exit


@cli.callback()
def main(
    ctx: typer.Context,
    config: ConfigOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    """Global options, applied before any command runs."""
    ctx.obj = State(config_file=config)
    env = process_environment()
    try:
        settings = load_config(config_file=config, env=env).runtime
    except ShortgenError as failure:
        err.print(f"[red]error:[/red] {failure}", soft_wrap=True)
        raise typer.Exit(EXIT_FAILURE) from None
    configure_logging("DEBUG" if verbose else settings.log_level, settings.log_format, err)


def _config_file(ctx: typer.Context) -> Path | None:
    shared = ctx.obj
    return shared.config_file if isinstance(shared, State) else None


def _execute(
    ctx: typer.Context,
    project: Path,
    stages: tuple[StageName, ...] | None,
    force: bool,
    scenes: list[str] | None = None,
) -> None:
    summary = app.execute(
        project,
        stages,
        force=force,
        scene_ids=scenes or None,
        config_file=_config_file(ctx),
        env=process_environment(),
        reporter=RichReporter(err),
    )
    print_summary(out, summary)


@cli.command()
@handle_errors
def init(
    project: ProjectArg,
    script: Annotated[
        Path | None, typer.Option("--script", "-s", help="Script file (.md or .txt).")
    ] = None,
    topic: Annotated[
        str | None, typer.Option("--topic", "-t", help="Topic for an LLM planner to write about.")
    ] = None,
    name: Annotated[
        str | None, typer.Option(help="Project name (default: directory name).")
    ] = None,
    aspect: Annotated[
        AspectRatio | None, typer.Option(help="Frame shape for this project.")
    ] = None,
) -> None:
    """Create a project directory from a script file or a topic."""
    manifest = app.init_project(project, script_file=script, topic=topic, name=name, aspect=aspect)
    out.print(
        f"Created project [bold]{manifest.project.name}[/bold] at {project.resolve()}",
        soft_wrap=True,
    )


@cli.command()
@handle_errors
def plan(ctx: typer.Context, project: ProjectArg, force: ForceOpt = False) -> None:
    """Split the script into scenes with a shot list."""
    _execute(ctx, project, (StageName.PLAN,), force)


@cli.command()
@handle_errors
def generate(
    ctx: typer.Context,
    project: ProjectArg,
    stage: Annotated[
        list[StageName] | None,
        typer.Option(
            "--stage",
            help="Stage to run; repeat for several. Default: speech, align, image, video.",
        ),
    ] = None,
    scene: Annotated[
        list[str] | None,
        typer.Option("--scene", help="Limit to a scene id such as s003; repeatable."),
    ] = None,
    force: ForceOpt = False,
) -> None:
    """Generate narration, alignment, illustrations and clips."""
    _execute(ctx, project, tuple(stage) if stage else GENERATE_STAGES, force, scene)


@cli.command()
@handle_errors
def render(ctx: typer.Context, project: ProjectArg, force: ForceOpt = False) -> None:
    """Build subtitles, compose scene segments and render the final MP4."""
    _execute(ctx, project, RENDER_STAGES, force)


@cli.command()
@handle_errors
def run(
    ctx: typer.Context,
    source: Annotated[
        Path | None,
        typer.Argument(
            help="A script file (.md/.txt) or an existing project directory.", show_default=False
        ),
    ] = None,
    project: Annotated[
        Path | None,
        typer.Option("--project", "-p", help="Project directory (default: projects/<name>)."),
    ] = None,
    topic: Annotated[
        str | None,
        typer.Option("--topic", "-t", help="Write the script from a topic (needs an LLM planner)."),
    ] = None,
    aspect: Annotated[
        AspectRatio | None, typer.Option(help="Frame shape, for a new project.")
    ] = None,
    force: ForceOpt = False,
) -> None:
    """Run the whole pipeline: script in, video out. Safe to re-run; it resumes."""
    target = app.prepare_project(source, project_dir=project, topic=topic, aspect=aspect)
    _execute(ctx, target, None, force)


@cli.command()
@handle_errors
def status(
    ctx: typer.Context,
    project: ProjectArg,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Show what is done, out of date, failed or blocked. Runs nothing."""
    context = app.build_context(project, config_file=_config_file(ctx), env=process_environment())
    report = app.project_status(context)
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_status(out, report)


@cli.command()
@handle_errors
def doctor(
    ctx: typer.Context,
    project: Annotated[
        Path | None, typer.Option("--project", "-p", help="Also apply this project's config.yaml.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check ffmpeg, the compute device, memory and the configured models."""
    checks = app.doctor(
        project_dir=project, config_file=_config_file(ctx), env=process_environment()
    )
    if as_json:
        typer.echo(json.dumps([check.model_dump(mode="json") for check in checks], indent=2))
    else:
        print_doctor(out, checks)
    if has_failures(checks):
        raise typer.Exit(EXIT_FAILURE)
