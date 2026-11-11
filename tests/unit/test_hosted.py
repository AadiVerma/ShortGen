"""The optional Higgsfield adapters, run against the real client and a pretend server.

Nothing here touches the network: the client's HTTP layer is replaced by
``httpx.MockTransport``. That proves the adapters call the client the way its
source expects; it cannot prove the live service answers like the pretend one.
"""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from shortgen import app
from shortgen.adapters._higgsfield import HiggsfieldJobs, credentials
from shortgen.adapters.higgsfield_image import HiggsfieldImage, aspect_ratio
from shortgen.adapters.higgsfield_video import HiggsfieldVideo, pick_duration
from shortgen.config import default_config_dir
from shortgen.domain.models import CameraMove
from shortgen.errors import ProviderError, ProviderUnavailableError, TransientError
from shortgen.providers.health import HealthStatus
from shortgen.providers.interfaces import ImageRequest, ProviderContext, VideoRequest
from shortgen.providers.spec import ProviderKind
from shortgen.registry import ModelRegistry
from shortgen.render.layout import Size
from tests.helpers import RecordingRunner, make_context, make_project
from tests.unit.test_registry import CPU

httpx = pytest.importorskip("httpx")
pytest.importorskip("higgsfield_client")

REGISTRY = ModelRegistry.from_files(default_config_dir() / "models.yaml")
API = "https://api.higgsfield.ai"
KEY = {"HF_KEY": "my-key:my-secret"}
IMAGE_URL = "https://cdn.example/out.png"
VIDEO_URL = "https://cdn.example/out.mp4"
DONE_IMAGE = {"status": "completed", "images": [{"url": IMAGE_URL}]}
DONE_VIDEO = {"status": "completed", "video": {"url": VIDEO_URL}}


def png_bytes(color: str = "teal") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


class Server:
    """A pretend Higgsfield: accepts jobs, reports the given statuses, takes uploads."""

    def __init__(self, *statuses: dict[str, Any], submit: Any = None) -> None:
        self.statuses = list(statuses)
        self.submit = submit
        self.jobs: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[bytes] = []
        self.headers: dict[str, str] = {}

    def __call__(self, request: Any) -> Any:
        path = request.url.path
        if request.method == "PUT":
            self.uploads.append(request.content)
            return httpx.Response(200)
        if path == "/files/generate-upload-url":
            number = len(self.uploads) + 1
            return httpx.Response(
                200,
                json={
                    "public_url": f"https://cdn.example/upload-{number}",
                    "upload_url": "https://upload.example/put",
                },
            )
        if path.endswith("/status"):
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json=status)
        if isinstance(self.submit, Exception):
            raise self.submit
        if self.submit is not None:
            return self.submit
        self.headers = dict(request.headers)
        self.jobs.append((path.lstrip("/"), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "status_url": f"{API}/requests/req-1/status",
                "cancel_url": f"{API}/requests/req-1/cancel",
            },
        )


class Downloads:
    """Stands in for ``urlopen``: serves bytes, or raises the queued errors first."""

    def __init__(self, body: bytes | dict[str, bytes], *errors: Exception) -> None:
        self.body, self.errors = body, list(errors)
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.urls.append(url)
        if self.errors:
            raise self.errors.pop(0)
        return io.BytesIO(self.body[url] if isinstance(self.body, dict) else self.body)


def connect(
    provider: Any,
    server: Server,
    downloads: Downloads,
    clock: Callable[[], float] | None = None,
) -> None:
    """Load the adapter, then swap its HTTP layer and its downloader for the pretend ones."""
    provider.load()
    jobs: HiggsfieldJobs = provider._jobs
    sdk = jobs._client
    sdk._client = httpx.Client(
        transport=httpx.MockTransport(server),
        base_url=API,
        headers={"Authorization": f"Key {sdk.api_key}"},
    )
    sdk._upload_client = httpx.Client(transport=httpx.MockTransport(server))
    jobs._opener = downloads
    jobs._sleep = lambda seconds: None
    if clock is not None:
        jobs._clock = clock


def make(
    name: str,
    kind: ProviderKind,
    tmp_path: Path,
    env: Mapping[str, str] = KEY,
    **options: Any,
) -> Any:
    context = ProviderContext(
        device=CPU, commands=RecordingRunner(), ffmpeg_bin="ffmpeg", project_dir=tmp_path, env=env
    )
    overrides = {"job_cache": str(tmp_path / "jobs"), "poll_interval_s": 0.001, **options}
    return REGISTRY.create(name, kind, context, overrides)


def image_provider(tmp_path: Path, **options: Any) -> HiggsfieldImage:
    provider = make("higgsfield_image", ProviderKind.IMAGE, tmp_path, **options)
    assert isinstance(provider, HiggsfieldImage)
    return provider


def image_request(tmp_path: Path, references: list[Path] | None = None) -> ImageRequest:
    return ImageRequest(
        scene_id="s001",
        prompt="a lighthouse at dusk, ink wash",
        negative_prompt="text",
        target=Size(width=1080, height=1920),
        seed=7,
        reference_images=references or [],
        output_path=tmp_path / "s001.png",
    )


def video_request(tmp_path: Path, duration_s: float = 4.2) -> VideoRequest:
    still = tmp_path / "still.png"
    still.write_bytes(png_bytes())
    return VideoRequest(
        image_path=still,
        prompt="Waves roll in.",
        camera=CameraMove.ZOOM_IN,
        duration_s=duration_s,
        fps=30,
        target=Size(width=1080, height=1920),
        seed=7,
        output_path=tmp_path / "s001.mp4",
    )


def pending_jobs(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "jobs").glob("*.job"))


# ----------------------------------------------------------------------- credentials


def test_credentials_come_from_either_documented_form() -> None:
    assert credentials({"HF_KEY": "a:b", "HF_API_KEY": "x", "HF_API_SECRET": "y"}) == "a:b"
    assert credentials({"HF_API_KEY": "x", "HF_API_SECRET": "y"}) == "x:y"
    assert credentials({"HF_API_KEY": "x"}) is None
    assert credentials({"HF_HOME": "/models"}) is None  # a Hugging Face variable, unrelated


def test_missing_credentials_stop_the_stage_and_say_what_to_set(tmp_path: Path) -> None:
    provider = make("higgsfield_image", ProviderKind.IMAGE, tmp_path, env={})
    with pytest.raises(ProviderUnavailableError, match="HF_KEY"):
        provider.load()
    health = {check.name: check for check in provider.health()}
    assert health["higgsfield_image: credentials"].status is HealthStatus.FAIL


def test_health_reports_credentials_without_showing_them(tmp_path: Path) -> None:
    provider = make("higgsfield_video", ProviderKind.VIDEO, tmp_path)
    checks = provider.health()
    assert all(check.status is HealthStatus.OK for check in checks)
    assert "my-secret" not in " ".join(f"{check.detail} {check.hint}" for check in checks)


# ---------------------------------------------------------------------------- images


def test_aspect_ratio_is_reduced_to_whole_numbers() -> None:
    assert aspect_ratio(Size(width=1080, height=1920)) == "9:16"
    assert aspect_ratio(Size(width=1920, height=1080)) == "16:9"
    assert aspect_ratio(Size(width=512, height=512)) == "1:1"


def test_image_job_sends_the_documented_arguments_and_saves_a_png(tmp_path: Path) -> None:
    provider = image_provider(tmp_path, arguments={"camera_fixed": False})
    server = Server({"status": "queued"}, {"status": "in_progress"}, DONE_IMAGE)
    downloads = Downloads(png_bytes("teal"))
    connect(provider, server, downloads)
    request = image_request(tmp_path)

    provider.generate(request)

    assert server.jobs == [
        (
            "bytedance/seedream/v4/text-to-image",
            {
                "prompt": "a lighthouse at dusk, ink wash",
                "aspect_ratio": "9:16",
                "resolution": "2K",
                "camera_fixed": False,
            },
        )
    ]
    assert server.headers["authorization"] == "Key my-key:my-secret"
    assert downloads.urls == [IMAGE_URL]
    with Image.open(request.output_path) as saved:
        assert saved.format == "PNG"
        assert saved.getpixel((0, 0)) == (0, 128, 128)
    assert pending_jobs(tmp_path) == []
    assert sorted(path.name for path in tmp_path.glob("s001*")) == ["s001.png"]


def test_reference_images_are_uploaded_only_when_an_argument_name_is_configured(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "mira.png"
    reference.write_bytes(png_bytes("red"))
    ignoring = image_provider(tmp_path)
    server = Server(DONE_IMAGE)
    connect(ignoring, server, Downloads(png_bytes()))
    ignoring.generate(image_request(tmp_path, [reference]))
    assert server.uploads == []
    assert "image_urls" not in server.jobs[0][1]

    sending = image_provider(tmp_path, reference_argument="image_urls")
    server = Server(DONE_IMAGE)
    connect(sending, server, Downloads(png_bytes()))
    sending.generate(image_request(tmp_path, [reference]))
    assert server.uploads == [png_bytes("red")]
    assert server.jobs[0][1]["image_urls"] == ["https://cdn.example/upload-1"]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ({"status": "failed", "error": "model overloaded"}, "failed.*model overloaded"),
        ({"status": "nsfw"}, "content filter"),
        ({"status": "canceled"}, "cancelled"),
    ],
)
def test_a_job_that_ends_badly_is_an_error_and_is_not_resumed(
    tmp_path: Path, status: dict[str, Any], message: str
) -> None:
    provider = image_provider(tmp_path)
    connect(provider, Server(status), Downloads(png_bytes()))
    with pytest.raises(ProviderError, match=message) as caught:
        provider.generate(image_request(tmp_path))
    assert not isinstance(caught.value, TransientError)
    assert pending_jobs(tmp_path) == []


def test_a_result_without_an_https_url_is_rejected(tmp_path: Path) -> None:
    results: list[dict[str, Any]] = [
        {"status": "completed"},
        {"status": "completed", "images": []},
        {"status": "completed", "images": [{"url": "http://cdn.example/out.png"}]},
    ]
    for number, result in enumerate(results):
        cache = tmp_path / f"jobs-{number}"
        provider = image_provider(tmp_path, job_cache=str(cache))
        downloads = Downloads(png_bytes())
        connect(provider, Server(result), downloads)
        with pytest.raises(ProviderError, match=r"no https URL.*stays remembered"):
            provider.generate(image_request(tmp_path))
        assert downloads.urls == []
        assert len(list(cache.glob("*.job"))) == 1  # paid for: kept, not thrown away


def test_output_that_is_not_an_image_is_rejected(tmp_path: Path) -> None:
    provider = image_provider(tmp_path)
    connect(provider, Server(DONE_IMAGE), Downloads(b"<html>not an image</html>"))
    request = image_request(tmp_path)
    with pytest.raises(ProviderError, match=r"not a readable image.*stays remembered"):
        provider.generate(request)
    assert not request.output_path.exists()
    assert list(tmp_path.glob("s001*")) == []


# ------------------------------------------------------- never paying for a job twice


def test_a_job_that_times_out_is_picked_up_by_the_next_run_not_resubmitted(
    tmp_path: Path,
) -> None:
    ticks = iter(range(0, 10_000, 400))
    provider = image_provider(tmp_path, job_timeout_s=600)
    server = Server({"status": "in_progress"})
    connect(provider, server, Downloads(png_bytes()), clock=lambda: float(next(ticks)))
    with pytest.raises(ProviderError, match=r"did not finish within 600 s.*Run the same") as caught:
        provider.generate(image_request(tmp_path))
    assert not isinstance(caught.value, TransientError)  # a retry must not start a second job
    assert len(server.jobs) == 1
    assert len(pending_jobs(tmp_path)) == 1
    assert pending_jobs(tmp_path)[0].read_text(encoding="utf-8") == "req-1"

    resumed = image_provider(tmp_path)
    later = Server(DONE_IMAGE)
    connect(resumed, later, Downloads(png_bytes()))
    resumed.generate(image_request(tmp_path))
    assert later.jobs == []  # nothing new was submitted
    assert image_request(tmp_path).output_path.is_file()
    assert pending_jobs(tmp_path) == []


def test_changing_the_prompt_is_a_new_job(tmp_path: Path) -> None:
    ticks = iter(range(0, 10_000, 400))
    provider = image_provider(tmp_path)
    connect(provider, Server({"status": "queued"}), Downloads(b""), clock=lambda: next(ticks))
    with pytest.raises(ProviderError):
        provider.generate(image_request(tmp_path))

    edited = image_request(tmp_path).model_copy(update={"prompt": "a harbour at dawn"})
    server = Server(DONE_IMAGE)
    again = image_provider(tmp_path)
    connect(again, server, Downloads(png_bytes()))
    again.generate(edited)
    assert [arguments["prompt"] for _, arguments in server.jobs] == ["a harbour at dawn"]


def test_a_failed_download_is_retried_here_and_never_costs_a_second_job(tmp_path: Path) -> None:
    provider = image_provider(tmp_path)
    server = Server(DONE_IMAGE)
    flaky = Downloads(png_bytes(), urllib.error.URLError("reset"), TimeoutError("slow"))
    connect(provider, server, flaky)
    provider.generate(image_request(tmp_path))
    assert len(flaky.urls) == 3
    assert len(server.jobs) == 1

    broken = Downloads(png_bytes(), *[urllib.error.URLError("down")] * 3)
    unlucky = image_provider(tmp_path, job_cache=str(tmp_path / "jobs2"))
    server = Server(DONE_IMAGE)
    connect(unlucky, server, broken)
    with pytest.raises(ProviderError, match="will not be charged a second time") as caught:
        unlucky.generate(image_request(tmp_path))
    assert not isinstance(caught.value, TransientError)
    assert len(list((tmp_path / "jobs2").glob("*.job"))) == 1

    retry = image_provider(tmp_path, job_cache=str(tmp_path / "jobs2"))
    later = Server(DONE_IMAGE)
    connect(retry, later, Downloads(png_bytes()))
    retry.generate(image_request(tmp_path))
    assert later.jobs == []


def test_an_oversized_download_is_refused(tmp_path: Path) -> None:
    provider = make("higgsfield_video", ProviderKind.VIDEO, tmp_path, max_download_mb=1)
    connect(provider, Server(DONE_VIDEO), Downloads(b"x" * (2 << 20)))
    request = video_request(tmp_path)
    with pytest.raises(ProviderError, match="larger than max_download_mb"):
        provider.generate(request)
    assert not request.output_path.exists()
    assert list(tmp_path.glob("*.partial")) == []


# ------------------------------------------------------------------- submit failures


def test_only_a_failure_to_connect_is_worth_retrying(tmp_path: Path) -> None:
    for never_sent in (httpx.ConnectError("no route"), httpx.ProxyError("403 Forbidden")):
        unreachable = image_provider(tmp_path)
        connect(unreachable, Server(submit=never_sent), Downloads(b""))
        with pytest.raises(TransientError, match="could not reach Higgsfield"):
            unreachable.generate(image_request(tmp_path))

    cut_off = image_provider(tmp_path)
    connect(cut_off, Server(submit=httpx.ReadTimeout("no answer")), Downloads(b""))
    with pytest.raises(ProviderError, match="may have been accepted") as caught:
        cut_off.generate(image_request(tmp_path))
    assert not isinstance(caught.value, TransientError)
    assert pending_jobs(tmp_path) == []


def test_a_rejected_job_reports_what_the_service_said(tmp_path: Path) -> None:
    provider = image_provider(tmp_path)
    refusal = httpx.Response(402, json={"detail": "Not enough credits"})
    connect(provider, Server(submit=refusal), Downloads(b""))
    with pytest.raises(ProviderError, match=r"rejected the job.*Not enough credits") as caught:
        provider.generate(image_request(tmp_path))
    assert not isinstance(caught.value, TransientError)


# ----------------------------------------------------------------------------- video


def test_pick_duration_covers_the_scene_when_it_can() -> None:
    assert pick_duration(4.2, [5, 10]) == 5
    assert pick_duration(7.5, [10, 5]) == 10
    assert pick_duration(14.0, [5, 10]) == 10


def test_video_job_uploads_the_still_and_saves_the_clip(tmp_path: Path) -> None:
    provider = make("higgsfield_video", ProviderKind.VIDEO, tmp_path, durations_s=[5, 10])
    assert isinstance(provider, HiggsfieldVideo)
    server = Server({"status": "queued"}, DONE_VIDEO)
    downloads = Downloads(b"mp4-bytes")
    connect(provider, server, downloads)
    request = video_request(tmp_path, duration_s=7.5)

    provider.generate(request)

    assert server.uploads == [request.image_path.read_bytes()]
    assert server.jobs == [
        (
            "higgsfield-ai/dop/standard",
            {
                "image_url": "https://cdn.example/upload-1",
                "prompt": "Waves roll in, the camera slowly pushes in",
                "duration": 10,
            },
        )
    ]
    assert downloads.urls == [VIDEO_URL]
    assert request.output_path.read_bytes() == b"mp4-bytes"
    assert pending_jobs(tmp_path) == []


def test_video_argument_names_can_be_corrected_in_configuration(tmp_path: Path) -> None:
    provider = make(
        "higgsfield_video",
        ProviderKind.VIDEO,
        tmp_path,
        image_argument="input_image",
        duration_argument=None,
    )
    server = Server(DONE_VIDEO)
    connect(provider, server, Downloads(b"mp4"))
    provider.generate(video_request(tmp_path))
    assert set(server.jobs[0][1]) == {"input_image", "prompt"}


def test_a_resumed_video_job_does_not_upload_the_still_again(tmp_path: Path) -> None:
    ticks = iter(range(0, 100_000, 2000))
    provider = make("higgsfield_video", ProviderKind.VIDEO, tmp_path)
    first = Server({"status": "in_progress"})
    connect(provider, first, Downloads(b""), clock=lambda: float(next(ticks)))
    with pytest.raises(ProviderError, match="did not finish"):
        provider.generate(video_request(tmp_path))
    assert len(first.uploads) == 1

    resumed = make("higgsfield_video", ProviderKind.VIDEO, tmp_path)
    later = Server(DONE_VIDEO)
    connect(resumed, later, Downloads(b"mp4"))
    resumed.generate(video_request(tmp_path))
    assert later.uploads == []
    assert later.jobs == []


# ------------------------------------------------------------------ the whole pipeline


def test_a_whole_run_with_hosted_pictures_and_clips_pays_once_per_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import higgsfield_client

    server = Server({**DONE_IMAGE, **DONE_VIDEO})
    downloads = Downloads({IMAGE_URL: png_bytes(), VIDEO_URL: b"mp4-bytes"})
    real_client = higgsfield_client.SyncClient

    def pretend_client(**settings: Any) -> Any:
        client = real_client(**settings)
        client._client = httpx.Client(
            transport=httpx.MockTransport(server),
            base_url=API,
            headers={"Authorization": f"Key {settings['api_key']}"},
        )
        client._upload_client = httpx.Client(transport=httpx.MockTransport(server))
        return client

    monkeypatch.setattr(higgsfield_client, "SyncClient", pretend_client)
    monkeypatch.setattr("urllib.request.urlopen", downloads)
    cache = str(tmp_path / "jobs")
    env = {
        **KEY,
        "SHORTGEN__PROVIDERS__IMAGE": "higgsfield_image",
        "SHORTGEN__PROVIDERS__VIDEO": "higgsfield_video",
        "SHORTGEN__PROVIDER_OPTIONS": (
            f"{{higgsfield_image: {{job_cache: '{cache}', poll_interval_s: 0.001}}, "
            f"higgsfield_video: {{job_cache: '{cache}', poll_interval_s: 0.001}}}}"
        ),
    }
    project = make_project(tmp_path)

    reports = app.run_pipeline(make_context(project, env=env))

    assert all(not report.failed for report in reports)
    models = [model for model, _ in server.jobs]
    assert models.count("bytedance/seedream/v4/text-to-image") == 3  # one per scene
    assert models.count("higgsfield-ai/dop/standard") == 3
    assert server.headers["authorization"] == "Key my-key:my-secret"
    with Image.open(project / "images" / "s001.png") as picture:
        assert picture.format == "PNG"
    assert (project / "clips" / "s003.mp4").read_bytes() == b"mp4-bytes"
    assert pending_jobs(tmp_path) == []
    for written in project.rglob("*.json"):
        assert "my-secret" not in written.read_text(encoding="utf-8")

    again = app.run_pipeline(make_context(project, env=env))
    assert all(not report.failed and not report.completed for report in again)
    assert len(server.jobs) == 6  # a second run sends nothing and costs nothing
