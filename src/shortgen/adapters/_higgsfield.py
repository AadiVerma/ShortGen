"""Shared plumbing for the optional Higgsfield adapters: a HOSTED, PAID service.

Everything else in shortgen runs on your machine. The Higgsfield adapters do
not: they send each scene's prompt (and, for video, its picture) to
https://api.higgsfield.ai, and every job spends credits on your account. They
are never a default, and they need credentials that you create yourself.

NOT VERIFIED. This was written from the source of ``higgsfield-client`` 0.2.0
and the examples in its README. No request was ever sent: there was no
account to test with. What is certain and what is not:

* Read from the client's source: ``SyncClient(timeout=, api_key=)``,
  ``submit(application, arguments)`` returning a controller with
  ``request_id``, ``status()`` and ``get()``, ``get_request_controller(id)``,
  ``upload_file(path)`` returning a public URL, ``DONE_STATUSES``,
  ``HiggsfieldClientError``, and credentials in ``HF_KEY`` or ``HF_API_KEY``
  plus ``HF_API_SECRET``.
* From the README's one example only: the model id
  ``bytedance/seedream/v4/text-to-image``, its arguments (``prompt``,
  ``resolution``, ``aspect_ratio``) and ``result["images"][0]["url"]``.
* From examples in the hosted API's documentation, least certain: the
  image-to-video model ids, their argument names and
  ``result["video"]["url"]``. Model ids and argument names are registry
  options, so they can be corrected without touching this code.

Because every job costs money, a job that was accepted is never submitted
twice: its request id is written to a small ledger file until the output is
saved, and the next run picks that job up again instead of starting a new one.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import Field

from shortgen.adapters._torch import require
from shortgen.domain.models import FrozenModel
from shortgen.errors import ProviderError, ProviderUnavailableError, TransientError
from shortgen.pipeline.hashing import stable_hash
from shortgen.providers.health import HealthCheck, HealthStatus
from shortgen.providers.spec import ModelSpec

log = logging.getLogger(__name__)

KEY_VARIABLE = "HF_KEY"
PAIR_VARIABLES = ("HF_API_KEY", "HF_API_SECRET")
CREDENTIAL_HINT = (
    f"Create an API key at https://cloud.higgsfield.ai, then set {KEY_VARIABLE}='key:secret' "
    f"(or {PAIR_VARIABLES[0]} and {PAIR_VARIABLES[1]}) in your shell or in ./.env. "
    "These are Higgsfield variables; they have nothing to do with Hugging Face."
)
_COMPLETED = "completed"
_REASONS = {
    "failed": "the job failed",
    "nsfw": "the job was blocked by Higgsfield's content filter; change the prompt",
    "canceled": "the job was cancelled",
}
_CHUNK_BYTES = 1 << 20
_BYTES_PER_MB = 1 << 20
_PARTIAL_SUFFIX = ".partial"
_SECURE_SCHEME = "https://"
#: Appended when a finished job's output cannot be used, so nobody pays twice to find out.
KEPT = (
    "The job stays remembered, so running again does not pay for a new one; "
    "see docs/hosted.md for how to start over"
)


class JobOptions(FrozenModel):
    """How to wait for a job and fetch its output; values come from the registry entry."""

    arguments: dict[str, Any] = Field(description="extra arguments sent with every job")
    job_timeout_s: float = Field(gt=0)
    poll_interval_s: float = Field(gt=0)
    request_timeout_s: float = Field(gt=0)
    download_attempts: int = Field(ge=1)
    max_download_mb: int = Field(ge=1)
    job_cache: str = Field(min_length=1, description="where request ids of unfinished jobs go")


def credentials(env: Mapping[str, str]) -> str | None:
    """The ``key:secret`` string from the environment, or ``None`` if it is not set."""
    key = env.get(KEY_VARIABLE)
    if key:
        return key
    api_key, secret = (env.get(name) for name in PAIR_VARIABLES)
    return f"{api_key}:{secret}" if api_key and secret else None


def credentials_check(name: str, env: Mapping[str, str]) -> HealthCheck:
    """A doctor line saying whether credentials are set. Never shows them."""
    found = credentials(env) is not None
    return HealthCheck(
        name=f"{name}: credentials",
        status=HealthStatus.OK if found else HealthStatus.FAIL,
        detail="found in the environment" if found else "no Higgsfield API key is set",
        hint=None if found else CREDENTIAL_HINT,
    )


def image_url(result: Mapping[str, Any]) -> str:
    """URL of the first generated image in a finished job's result."""
    images = result.get("images")
    first = images[0] if isinstance(images, list) and images else None
    return _url(first, "images[0].url", result)


def video_url(result: Mapping[str, Any]) -> str:
    """URL of the generated clip in a finished job's result."""
    return _url(result.get("video"), "video.url", result)


def _url(holder: Any, where: str, result: Mapping[str, Any]) -> str:
    url = holder.get("url") if isinstance(holder, dict) else None
    if not isinstance(url, str) or not url.startswith(_SECURE_SCHEME):
        raise ProviderError(
            f"Higgsfield's result has no https URL at {where}; it contains: {sorted(result)}. "
            f"{KEPT}"
        )
    return url


class JobLedger:
    """Remembers the request id of each accepted job until its output is saved."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path(self, key: str) -> Path:
        """The file that holds the request id of the job identified by ``key``."""
        return self._directory / f"{stable_hash(key)}.job"

    def get(self, key: str) -> str | None:
        """The request id of an unfinished job with these inputs, if there is one."""
        path = self.path(key)
        return (path.read_text(encoding="utf-8").strip() or None) if path.is_file() else None

    def put(self, key: str, request_id: str) -> None:
        """Record that a job with these inputs was accepted."""
        self._directory.mkdir(parents=True, exist_ok=True)
        self.path(key).write_text(request_id, encoding="utf-8")

    def forget(self, key: str) -> None:
        """Drop a job: its output is saved, or it ended without one."""
        self.path(key).unlink(missing_ok=True)


class HiggsfieldJobs:
    """Runs one job at a time against one model, without ever paying for it twice."""

    def __init__(
        self,
        spec: ModelSpec,
        options: JobOptions,
        env: Mapping[str, str],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        key = credentials(env)
        if key is None:
            raise ProviderUnavailableError(f"no Higgsfield API key is set. {CREDENTIAL_HINT}")
        if not spec.model_id:
            raise ProviderUnavailableError(f"registry entry '{spec.name}' has no model_id")
        self._sdk = require("higgsfield_client", spec.install_hint)
        self._httpx = require("httpx", spec.install_hint)
        self._client = self._sdk.SyncClient(timeout=options.request_timeout_s, api_key=key)
        self._model = spec.model_id
        self._options = options
        self._ledger = JobLedger(Path(options.job_cache).expanduser())
        self._clock, self._sleep = clock, sleep
        self._opener = opener or urllib.request.urlopen

    def run(self, key: str, build: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        """Return the result of the job identified by ``key``, submitting it only if needed.

        ``build`` makes the job's arguments; it is not called when an earlier
        run already submitted this job, so nothing is uploaded or paid twice.
        Call :meth:`finish` once the output is safely on disk.
        """
        request_id = self._ledger.get(key)
        if request_id is None:
            controller = self._submit({**build(), **self._options.arguments})
            self._ledger.put(key, str(controller.request_id))
        else:
            log.info("picking up Higgsfield job %s from an earlier run", request_id)
            controller = self._client.get_request_controller(request_id)
        self._wait(controller, key)
        result = self._result(controller, key)
        status = result.get("status")
        if status != _COMPLETED:
            self._ledger.forget(key)
            reason = _REASONS.get(str(status), f"it ended with status {status!r}")
            detail = f" ({result['error']})" if result.get("error") else ""
            raise ProviderError(f"Higgsfield job {controller.request_id}: {reason}{detail}")
        return result

    def finish(self, key: str) -> None:
        """Forget a job whose output has been saved."""
        self._ledger.forget(key)

    def upload(self, path: Path) -> str:
        """Upload a local file and return the URL a job can read it from."""
        try:
            return str(self._client.upload_file(path))
        except self._httpx.HTTPError as exc:
            raise TransientError(f"could not upload {path.name} to Higgsfield: {exc}") from exc
        except self._sdk.HiggsfieldClientError as exc:
            raise ProviderError(f"Higgsfield refused the upload of {path.name}: {exc}") from exc

    def download(self, url: str, target: Path) -> None:
        """Save a finished output to ``target``.

        Retried here rather than by the pipeline: a pipeline retry would start
        over, and the job is already paid for.
        """
        if not url.startswith(_SECURE_SCHEME):
            raise ProviderError(f"refusing to download from a non-https URL: {url}")
        partial = target.with_name(target.name + _PARTIAL_SUFFIX)
        attempts = self._options.download_attempts
        last: Exception | None = None
        try:
            for attempt in range(1, attempts + 1):
                try:
                    self._fetch(url, partial)
                    partial.replace(target)
                    return
                except (urllib.error.URLError, OSError) as exc:
                    log.warning("download %d of %d failed: %s", attempt, attempts, exc)
                    last = exc
        finally:
            partial.unlink(missing_ok=True)
        raise ProviderError(
            f"could not download the finished output ({last}). Run the same command again: "
            "the job is remembered and will not be charged a second time"
        )

    def _submit(self, arguments: Mapping[str, Any]) -> Any:
        """Start a job. Only a failure to connect is retried: nothing was accepted yet."""
        httpx = self._httpx
        try:
            return self._client.submit(self._model, dict(arguments))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError) as exc:
            raise TransientError(f"could not reach Higgsfield: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"the connection to Higgsfield broke while submitting ({exc}). The job may have "
                "been accepted; check your Higgsfield account before running again"
            ) from exc
        except self._sdk.HiggsfieldClientError as exc:
            raise ProviderError(f"Higgsfield rejected the job for '{self._model}': {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise ProviderError(f"unexpected answer from Higgsfield on submit: {exc}") from exc

    def _wait(self, controller: Any, key: str) -> None:
        """Poll until the job ends. Network hiccups are ignored until the deadline."""
        deadline = self._clock() + self._options.job_timeout_s
        while True:
            try:
                if isinstance(controller.status(), self._sdk.DONE_STATUSES):
                    return
            except self._httpx.HTTPError as exc:
                log.debug("status of %s unavailable, asking again: %s", controller.request_id, exc)
            except (self._sdk.HiggsfieldClientError, ValueError) as exc:
                problem = f"its status cannot be read: {exc}"
                raise ProviderError(self._stuck(controller, key, problem)) from exc
            if self._clock() >= deadline:
                waited = f"it did not finish within {self._options.job_timeout_s:g} s"
                raise ProviderError(self._stuck(controller, key, waited))
            self._sleep(self._options.poll_interval_s)

    def _result(self, controller: Any, key: str) -> dict[str, Any]:
        try:
            result = controller.get()
        except (self._httpx.HTTPError, self._sdk.HiggsfieldClientError, ValueError) as exc:
            problem = f"its result cannot be read: {exc}"
            raise ProviderError(self._stuck(controller, key, problem)) from exc
        if not isinstance(result, dict):
            raise ProviderError(f"unexpected result from Higgsfield: {type(result).__name__}")
        return result

    def _stuck(self, controller: Any, key: str, problem: str) -> str:
        return (
            f"Higgsfield job {controller.request_id}: {problem}. Run the same command again to "
            f"keep waiting for this job without paying for a new one, or delete "
            f"{self._ledger.path(key)} to submit a new one"
        )

    def _fetch(self, url: str, partial: Path) -> None:
        limit = self._options.max_download_mb * _BYTES_PER_MB
        received = 0
        with (
            self._opener(url, timeout=self._options.request_timeout_s) as response,
            partial.open("wb") as handle,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                received += len(chunk)
                if received > limit:
                    raise ProviderError(
                        f"the output is larger than max_download_mb "
                        f"({self._options.max_download_mb} MB); refusing to save it"
                    )
                handle.write(chunk)
        if not received:
            raise ProviderError("Higgsfield returned an empty file")
