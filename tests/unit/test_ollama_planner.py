"""The Ollama planner against a local stub HTTP server speaking Ollama's REST shapes."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from shortgen.adapters.ollama_planner import OllamaPlanner
from shortgen.config import AppConfig, default_config_dir
from shortgen.domain.models import CameraMove, Character
from shortgen.errors import ProviderUnavailableError, TransientError
from shortgen.providers.health import HealthStatus
from shortgen.providers.interfaces import PlanRequest, ProviderContext
from shortgen.providers.spec import ProviderKind
from shortgen.registry import ModelRegistry
from tests.helpers import RecordingRunner
from tests.unit.test_registry import CPU

SCRIPT = "Mira climbed the stairs.\n\nThe lamp was dark."


class StubOllama:
    """Scripted replies per path, plus a log of every request body received."""

    def __init__(self) -> None:
        self.replies: dict[str, tuple[int, Any]] = {}
        self.requests: list[tuple[str, Any]] = []

    def storyboard(self, scenes: list[dict[str, str]]) -> None:
        content = json.dumps({"scenes": scenes})
        self.replies["/api/chat"] = (200, {"message": {"role": "assistant", "content": content}})


@pytest.fixture
def ollama() -> Iterator[tuple[StubOllama, str]]:
    stub = StubOllama()

    class Handler(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            stub.requests.append((self.path, body))
            status, payload = stub.replies.get(self.path, (500, {"error": "unscripted"}))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _respond  # noqa: N815

        def log_message(self, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield stub, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def planner(host: str) -> OllamaPlanner:
    registry = ModelRegistry.from_files(default_config_dir() / "models.yaml")
    context = ProviderContext(device=CPU, commands=RecordingRunner(), ffmpeg_bin="ffmpeg")
    provider = registry.create("ollama", ProviderKind.PLANNER, context, {"host": host})
    assert isinstance(provider, OllamaPlanner)
    return provider


def request(config: AppConfig) -> PlanRequest:
    mira = Character(name="Mira", description="a keeper in a red scarf")
    return PlanRequest(
        script=SCRIPT, language="en", rules=config.planning, characters=[mira], seed=42
    )


def test_plan_keeps_narration_and_takes_visuals_from_the_model(
    ollama: tuple[StubOllama, str], config: AppConfig
) -> None:
    stub, host = ollama
    stub.storyboard(
        [
            {"image_prompt": "Mira on a spiral stair", "motion_prompt": "dust", "camera": "pan_up"},
            {"image_prompt": "a dark lamp room", "motion_prompt": "", "camera": "dolly_zoom"},
        ]
    )
    scenes = planner(host).plan(request(config))

    assert [scene.narration for scene in scenes] == SCRIPT.split("\n\n")  # never rewritten
    assert scenes[0].shot.image_prompt == "Mira on a spiral stair"
    assert scenes[0].shot.camera is CameraMove.PAN_UP
    assert scenes[0].characters == ["Mira"]
    assert scenes[1].shot.motion_prompt == config.planning.default_motion_prompt  # blank -> draft
    assert scenes[1].shot.camera is config.planning.camera_cycle[1]  # unknown move -> draft

    path, body = stub.requests[0]
    assert path == "/api/chat"
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.7, "seed": 42}
    assert body["format"]["properties"]["scenes"]["type"] == "array"
    brief = json.loads(body["messages"][1]["content"])
    assert [scene["narration"] for scene in brief["scenes"]] == SCRIPT.split("\n\n")
    assert brief["characters"] == [{"name": "Mira", "description": "a keeper in a red scarf"}]


@pytest.mark.parametrize(
    "content",
    ["not json at all", '{"scenes": [{"image_prompt": "only one"}]}', '{"wrong": []}', "[1, 2]"],
)
def test_unusable_storyboards_are_transient_so_the_runner_retries(
    ollama: tuple[StubOllama, str], config: AppConfig, content: str
) -> None:
    stub, host = ollama
    stub.replies["/api/chat"] = (200, {"message": {"content": content}})
    with pytest.raises(TransientError, match="unusable storyboard"):
        planner(host).plan(request(config))


def test_write_script_returns_the_model_text(ollama: tuple[StubOllama, str]) -> None:
    stub, host = ollama
    stub.replies["/api/chat"] = (200, {"message": {"content": "  Once upon a time.\n"}})
    assert planner(host).write_script("a lighthouse", 120, "en") == "Once upon a time."
    _, body = stub.requests[0]
    assert "format" not in body  # free text, not JSON
    assert "about 120 words" in body["messages"][1]["content"]

    stub.replies["/api/chat"] = (200, {"message": {"content": "   "}})
    with pytest.raises(TransientError, match="empty script"):
        planner(host).write_script("a lighthouse", 120, "en")


def test_http_errors_map_onto_the_error_hierarchy(
    ollama: tuple[StubOllama, str], config: AppConfig
) -> None:
    stub, host = ollama
    stub.replies["/api/chat"] = (404, {"error": "model not found"})
    with pytest.raises(ProviderUnavailableError, match=r"ollama pull qwen2\.5:7b"):
        planner(host).plan(request(config))
    stub.replies["/api/chat"] = (503, {"error": "busy"})
    with pytest.raises(TransientError, match="HTTP 503"):
        planner(host).plan(request(config))


def test_unreachable_server_is_unavailable_not_transient(config: AppConfig) -> None:
    with pytest.raises(ProviderUnavailableError, match="is `ollama serve` running"):
        planner("http://127.0.0.1:9").plan(request(config))


def test_unload_asks_ollama_to_drop_the_model(ollama: tuple[StubOllama, str]) -> None:
    stub, host = ollama
    stub.replies["/api/generate"] = (200, {"done": True})
    planner(host).unload()
    assert stub.requests == [("/api/generate", {"model": "qwen2.5:7b", "keep_alive": 0})]
    planner("http://127.0.0.1:9").unload()  # a dead server must not crash the pipeline


def test_health_reports_server_and_model(ollama: tuple[StubOllama, str]) -> None:
    stub, host = ollama
    stub.replies["/api/tags"] = (200, {"models": [{"name": "llama3.2:latest"}]})
    server, model = planner(host).health()
    assert server.status is HealthStatus.OK
    assert (model.status, model.hint) == (HealthStatus.FAIL, "ollama pull qwen2.5:7b")

    stub.replies["/api/tags"] = (200, {"models": [{"name": "qwen2.5:7b"}]})
    assert [check.status for check in planner(host).health()] == [HealthStatus.OK] * 2

    (down,) = planner("http://127.0.0.1:9").health()
    assert down.status is HealthStatus.FAIL
    assert "ollama serve" in (down.hint or "")
