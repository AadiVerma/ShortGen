"""LLM planner backed by a local Ollama server (free, runs on your machine).

API used (Ollama REST API): ``POST /api/chat`` with ``model``, ``messages``,
``stream: false``, ``format`` (a JSON schema), ``options`` and ``keep_alive``;
``GET /api/tags`` to list pulled models; ``POST /api/generate`` with
``keep_alive: 0`` to unload a model. The request and response handling is
tested against a local stub server, not against a real Ollama install.

The LLM never rewrites narration. Scenes are cut deterministically by
``shortgen.domain.scene_planner``; the model only adds an image prompt, a
motion prompt and a camera move per scene.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from pydantic import Field, ValidationError

from shortgen.domain.models import CameraMove, FrozenModel, Scene
from shortgen.domain.scene_planner import (
    SceneEnrichment,
    apply_enrichment,
    draft_scenes,
    keep_author_directions,
)
from shortgen.errors import PlanningError, ProviderUnavailableError, TransientError
from shortgen.providers.health import HealthCheck, HealthStatus
from shortgen.providers.interfaces import PlanRequest, ScriptPlanner

log = logging.getLogger(__name__)

_HTTP_NOT_FOUND = 404
_HEALTH_TIMEOUT_S = 3.0
_STORYBOARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_prompt": {"type": "string"},
                    "motion_prompt": {"type": "string"},
                    "camera": {"type": "string", "enum": [move.value for move in CameraMove]},
                },
                "required": ["image_prompt", "motion_prompt", "camera"],
            },
        }
    },
    "required": ["scenes"],
}
_STORYBOARD_SYSTEM = (
    "You are a storyboard artist for short narrated videos. For each scene you are given, "
    "describe one still illustration and how it should move. Reply with JSON only.\n"
    "- image_prompt: one sentence, at most 40 words, describing only what is visible "
    "(subject, setting, lighting, framing). No text or captions in the image. "
    "Refer to recurring characters by name.\n"
    "- motion_prompt: one short sentence describing subtle motion within the scene.\n"
    "- camera: one of the allowed camera moves.\n"
    "Return exactly one entry per scene, in the same order."
)
_SCRIPT_SYSTEM = (
    "You write narration scripts for short vertical videos. Write plain prose only: no "
    "headings, no stage directions, no emojis, no hashtags. Use short sentences and put a "
    "blank line between paragraphs; each paragraph is one visual scene."
)


class OllamaOptions(FrozenModel):
    """Where the Ollama server is and how to sample from it."""

    host: str = Field(pattern=r"^https?://[^\s/]+$")
    temperature: float = Field(ge=0, le=2)
    timeout_s: float = Field(gt=0)
    keep_alive: str


class OllamaPlanner(ScriptPlanner):
    """Enriches deterministic scenes with visual direction from a local LLM."""

    def write_script(self, topic: str, target_words: int, language: str) -> str:
        user = (
            f"Write a script of about {target_words} words, in language '{language}', "
            f"about: {topic}"
        )
        script = self._chat(_SCRIPT_SYSTEM, user, schema=None, seed=0).strip()
        if not script:
            raise TransientError("the model returned an empty script")
        return script

    def plan(self, request: PlanRequest) -> list[Scene]:
        scenes = draft_scenes(request.script, request.rules, request.characters)
        brief = {
            "characters": [
                c.model_dump(include={"name", "description"}) for c in request.characters
            ],
            "scenes": [{"id": scene.id, "narration": scene.narration} for scene in scenes],
        }
        reply = self._chat(
            _STORYBOARD_SYSTEM, json.dumps(brief), schema=_STORYBOARD_SCHEMA, seed=request.seed
        )
        try:
            enriched = apply_enrichment(scenes, _parse_storyboard(reply))
            return keep_author_directions(scenes, enriched)
        except PlanningError as exc:
            # LLM output varies between attempts, so a malformed plan is worth a retry.
            raise TransientError(f"the model returned an unusable storyboard: {exc}") from exc

    def unload(self) -> None:
        """Ask Ollama to drop the model so its memory is free for the next stage."""
        try:
            self._request("/api/generate", {"model": self.spec.model_id, "keep_alive": 0})
        except (TransientError, ProviderUnavailableError) as exc:
            log.warning("could not unload Ollama model %s: %s", self.spec.model_id, exc)

    def health(self) -> list[HealthCheck]:
        options = self.parse_options(OllamaOptions)
        name = f"{self.name}: server"
        try:
            tags = self._request("/api/tags", None, timeout=_HEALTH_TIMEOUT_S)
        except (TransientError, ProviderUnavailableError) as exc:
            return [
                HealthCheck(
                    name=name,
                    status=HealthStatus.FAIL,
                    detail=str(exc),
                    hint="Install Ollama from https://ollama.com and run `ollama serve`.",
                )
            ]
        pulled = {str(model.get("name", "")) for model in tags.get("models", [])}
        present = self.spec.model_id in pulled
        return [
            HealthCheck(name=name, status=HealthStatus.OK, detail=f"reachable at {options.host}"),
            HealthCheck(
                name=f"{self.name}: model",
                status=HealthStatus.OK if present else HealthStatus.FAIL,
                detail=f"{self.spec.model_id} {'is pulled' if present else 'is not pulled'}",
                hint=None if present else f"ollama pull {self.spec.model_id}",
            ),
        ]

    def _chat(self, system: str, user: str, *, schema: dict[str, Any] | None, seed: int) -> str:
        options = self.parse_options(OllamaOptions)
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": options.keep_alive,
            "options": {"temperature": options.temperature, "seed": seed},
        }
        if schema is not None:
            payload["format"] = schema
        reply = self._request("/api/chat", payload)
        return str(reply.get("message", {}).get("content", ""))

    def _request(
        self, path: str, payload: dict[str, Any] | None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """One HTTP round trip, with failures mapped onto the shortgen error hierarchy."""
        options = self.parse_options(OllamaOptions)
        request = urllib.request.Request(  # noqa: S310 - host is validated as http(s) above
            f"{options.host}{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or options.timeout_s) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == _HTTP_NOT_FOUND:
                raise ProviderUnavailableError(
                    f"Ollama does not have model '{self.spec.model_id}'; "
                    f"run `ollama pull {self.spec.model_id}`"
                ) from exc
            raise TransientError(f"Ollama returned HTTP {exc.code} for {path}") from exc
        except TimeoutError as exc:
            raise TransientError(f"Ollama did not answer {path} in time") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TransientError(f"Ollama did not answer {path} in time") from exc
            raise ProviderUnavailableError(
                f"cannot reach Ollama at {options.host} ({exc.reason}); is `ollama serve` running?"
            ) from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TransientError(f"Ollama sent a non-JSON response for {path}") from exc
        return parsed if isinstance(parsed, dict) else {}


def _parse_storyboard(reply: str) -> list[SceneEnrichment]:
    """Validate the model's JSON; an unknown camera move falls back to the draft's."""
    try:
        entries = json.loads(reply)["scenes"]
        return [
            SceneEnrichment(
                image_prompt=str(entry.get("image_prompt", "")),
                motion_prompt=str(entry.get("motion_prompt", "")),
                camera=_camera(entry.get("camera")),
            )
            for entry in entries
        ]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValidationError) as exc:
        raise PlanningError(f"storyboard is not the expected JSON ({exc})") from exc


def _camera(value: object) -> CameraMove | None:
    try:
        return CameraMove(str(value))
    except ValueError:
        return None
