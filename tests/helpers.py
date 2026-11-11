"""Test doubles and project builders shared by unit and integration tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from shortgen import app
from shortgen.pipeline.context import PipelineContext
from shortgen.process import CommandResult, CommandRunner

STORY = """# The Lighthouse

Mira climbed the stairs of the old lighthouse. The lamp had gone dark, and a storm was coming.

She lit the wick with shaking hands. Far out at sea, a small boat turned toward the light.

By morning the sea was calm.
"""

#: Fake models everywhere and small frames: fast, no GPU, no downloads.
FAKE_ENV: dict[str, str] = {
    "SHORTGEN__PROVIDERS__PLANNER": "fake_planner",
    "SHORTGEN__PROVIDERS__SPEECH": "fake_speech",
    "SHORTGEN__PROVIDERS__TRANSCRIBER": "proportional",
    "SHORTGEN__PROVIDERS__IMAGE": "fake_image",
    "SHORTGEN__PROVIDERS__VIDEO": "fake_video",
    "SHORTGEN__RENDER__RESOLUTIONS": (
        "{'9:16': {width: 180, height: 320}, '16:9': {width: 320, height: 180}}"
    ),
    "SHORTGEN__RUNTIME__BACKOFF_BASE_S": "0",
}

FAKE_CLIP_SECONDS = "2.000000"


class RecordingRunner:
    """Stands in for ffmpeg/ffprobe: records commands and touches output files."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
        command = list(args)
        self.commands.append(command)
        if "format=duration" in command:  # an ffprobe duration query
            return CommandResult(stdout=f"{FAKE_CLIP_SECONDS}\n", stderr="")
        Path(command[-1]).write_bytes(b"fake media written by RecordingRunner\n")
        return CommandResult(stdout="", stderr="")

    def ffmpeg_calls(self) -> list[list[str]]:
        """Recorded commands that would have produced a file."""
        return [command for command in self.commands if "format=duration" not in command]


def make_project(tmp_path: Path, script: str = STORY) -> Path:
    """Create an initialised project from ``script`` and return its directory."""
    script_file = tmp_path / "story.md"
    script_file.write_text(script, encoding="utf-8")
    project_dir = tmp_path / "project"
    app.init_project(project_dir, script_file=script_file)
    return project_dir


def make_context(
    project_dir: Path,
    env: Mapping[str, str] | None = None,
    commands: CommandRunner | None = None,
) -> PipelineContext:
    """A pipeline context using fake providers unless ``env`` says otherwise."""
    return app.build_context(
        project_dir, env={**FAKE_ENV, **(env or {})}, commands=commands or RecordingRunner()
    )
