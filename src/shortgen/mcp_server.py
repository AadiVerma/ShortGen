"""Optional MCP server: lets an AI assistant drive the pipeline.

Every tool is a few lines that call :mod:`shortgen.app`, the same functions
the CLI uses, so the two front ends cannot disagree. Requires the ``mcp``
extra (``pip install 'shortgen[mcp]'``, MCP Python SDK 2.x).

Tools run synchronously and generation is slow: an assistant should call
``status`` to see progress and simply call a tool again to resume after a
timeout, since finished work is never repeated.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shortgen import app
from shortgen.config import load_config, process_environment
from shortgen.doctor import has_failures
from shortgen.domain.models import AspectRatio
from shortgen.errors import ProviderUnavailableError, ShortgenError
from shortgen.logging import configure_logging
from shortgen.pipeline.catalog import GENERATE_STAGES, RENDER_STAGES
from shortgen.pipeline.stage import StageName

INSTRUCTIONS = (
    "shortgen turns a script into a narrated, captioned short video using local models. "
    "Typical flow: init_project, then run (or plan, generate, render separately). "
    "All tools are resumable: calling one again only redoes what changed or failed. "
    "Call doctor first if anything fails, and status to inspect progress."
)


def _execute(
    project_dir: str,
    stages: tuple[StageName, ...] | None,
    force: bool,
    scenes: list[str] | None = None,
) -> dict[str, Any]:
    summary = app.execute(
        Path(project_dir), stages, force=force, scene_ids=scenes or None, env=process_environment()
    )
    return summary.model_dump(mode="json")


def init_project(
    project_dir: str,
    script_path: str | None = None,
    topic: str | None = None,
    aspect: str | None = None,
) -> dict[str, Any]:
    """Create a project from a script file (.md/.txt) or a topic. aspect is "9:16" or "16:9"."""
    manifest = app.init_project(
        Path(project_dir),
        script_file=Path(script_path) if script_path else None,
        topic=topic,
        aspect=AspectRatio(aspect) if aspect else None,
    )
    return {"project_dir": str(Path(project_dir).resolve()), "name": manifest.project.name}


def plan(project_dir: str, force: bool = False) -> dict[str, Any]:
    """Split the project's script into scenes with a shot list."""
    return _execute(project_dir, (StageName.PLAN,), force)


def generate(
    project_dir: str,
    stages: list[str] | None = None,
    scenes: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate assets for a planned project.

    stages: any of speech, align, image, video (default: all four, in order).
    scenes: optional scene ids such as "s003" to limit the work.
    """
    selected = tuple(StageName(stage) for stage in stages) if stages else GENERATE_STAGES
    return _execute(project_dir, selected, force, scenes)


def render(project_dir: str, force: bool = False) -> dict[str, Any]:
    """Build subtitles, compose scene segments and render the final MP4."""
    return _execute(project_dir, RENDER_STAGES, force)


def run(project_dir: str, force: bool = False) -> dict[str, Any]:
    """Run the whole pipeline for an initialised project and return the video path."""
    return _execute(project_dir, None, force)


def status(project_dir: str) -> dict[str, Any]:
    """Report which stages are done, pending, failed or blocked. Runs nothing."""
    ctx = app.build_context(Path(project_dir), env=process_environment())
    return app.project_status(ctx).model_dump(mode="json")


def doctor(project_dir: str | None = None) -> dict[str, Any]:
    """Check ffmpeg, the compute device, memory and the configured models."""
    checks = app.doctor(
        project_dir=Path(project_dir) if project_dir else None, env=process_environment()
    )
    return {
        "ok": not has_failures(checks),
        "checks": [check.model_dump(mode="json") for check in checks],
    }


TOOLS = (init_project, plan, generate, render, run, status, doctor)


def create_server() -> Any:
    """Build the MCP server with one tool per pipeline command."""
    try:
        module = importlib.import_module("mcp.server.mcpserver")
        tool_error = importlib.import_module("mcp.server.mcpserver.exceptions").ToolError
    except ImportError as exc:
        raise ProviderUnavailableError(
            "the MCP server needs the MCP Python SDK 2.x: pip install 'shortgen[mcp]'"
        ) from exc
    server = module.MCPServer("shortgen", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.add_tool(_reporting_errors(tool, tool_error))
    return server


def _reporting_errors(
    tool: Callable[..., dict[str, Any]], tool_error: type[Exception]
) -> Callable[..., dict[str, Any]]:
    """Re-raise expected failures as ``ToolError`` so the assistant sees the reason.

    The SDK hides the message of any other exception type behind a generic
    "Error executing tool", which would leave an assistant unable to react.
    """

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return tool(*args, **kwargs)
        except (ShortgenError, ValueError) as failure:
            raise tool_error(str(failure)) from failure

    return wrapper


def main() -> None:
    """Console-script entry point: serve over stdio."""
    settings = load_config(env=process_environment()).runtime
    configure_logging(settings.log_level, settings.log_format)  # stderr; stdout is the transport
    create_server().run()
