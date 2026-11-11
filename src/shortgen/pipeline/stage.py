"""The stage abstraction: units of work, their outputs and provider access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from shortgen.domain.models import Asset, AssetKind
from shortgen.errors import StageError
from shortgen.pipeline.context import PipelineContext
from shortgen.pipeline.hashing import file_sha256, hash_model
from shortgen.pipeline.manifest import Manifest, job_key
from shortgen.pipeline.workspace import Workspace, partial_path
from shortgen.providers.interfaces import Provider
from shortgen.providers.spec import ProviderKind

InputsT = TypeVar("InputsT", bound=BaseModel)
ProviderT = TypeVar("ProviderT", bound=Provider)


class StageName(StrEnum):
    """Pipeline stages, declared in execution order."""

    PLAN = "plan"
    SPEECH = "speech"
    ALIGN = "align"
    IMAGE = "image"
    VIDEO = "video"
    SUBTITLES = "subtitles"
    COMPOSE = "compose"
    RENDER = "render"


STAGE_ORDER: tuple[StageName, ...] = tuple(StageName)


@dataclass(frozen=True)
class OutputSpec:
    """A file a unit promises to write, relative to the project directory."""

    kind: AssetKind
    path: str


@dataclass(frozen=True)
class WorkUnit(Generic[InputsT]):
    """One resumable piece of a stage: typed inputs in, declared files out."""

    stage: StageName
    scene_id: str | None
    inputs: InputsT
    outputs: tuple[OutputSpec, ...] = ()

    @property
    def key(self) -> str:
        """The manifest key of this unit's job."""
        return job_key(self.stage, self.scene_id)

    @property
    def label(self) -> str:
        """Short name for progress output."""
        return self.scene_id or "project"

    @property
    def inputs_hash(self) -> str:
        """Changes exactly when the unit needs to run again."""
        return hash_model(self.inputs)


class Targets:
    """Where a running unit writes.

    Stages write to temp files next to the final paths. Only after the whole
    unit succeeds does :meth:`commit` rename them into place, so an interrupted
    or failed unit can never leave a truncated file that looks finished.
    """

    def __init__(self, workspace: Workspace, outputs: tuple[OutputSpec, ...]) -> None:
        self._workspace = workspace
        self._outputs = outputs
        self._temp = {spec.kind: partial_path(workspace.resolve(spec.path)) for spec in outputs}

    def __getitem__(self, kind: AssetKind) -> Path:
        """Temp path for the output of ``kind``; its directory is created on demand."""
        path = self._temp[kind]
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commit(self, scene_id: str | None) -> list[Asset]:
        """Check every promised file exists, move all into place, return their records."""
        missing = [spec.path for spec in self._outputs if not self._temp[spec.kind].is_file()]
        if missing:
            raise StageError(f"stage finished without writing: {', '.join(missing)}")
        assets: list[Asset] = []
        for spec in self._outputs:
            final = self._workspace.resolve(spec.path)
            self._temp[spec.kind].replace(final)
            assets.append(
                Asset(kind=spec.kind, path=spec.path, sha256=file_sha256(final), scene_id=scene_id)
            )
        return assets

    def discard(self) -> None:
        """Remove any temp files left by a failed or interrupted attempt."""
        for path in self._temp.values():
            path.unlink(missing_ok=True)


class ProviderHandle:
    """Loads a stage's provider on first use and guarantees it is unloaded after.

    Laziness matters: a stage whose units are all up to date, or that turns out
    not to need its model (alignment with native timings), never pays for loading.
    """

    def __init__(self, factory: Callable[[], Provider] | None) -> None:
        self._factory = factory
        self._provider: Provider | None = None

    def get(self, interface: type[ProviderT]) -> ProviderT:
        """The loaded provider, checked against the interface the stage expects."""
        if self._provider is None:
            if self._factory is None:
                raise StageError("this stage has no provider")
            provider = self._factory()
            provider.load()
            self._provider = provider
        if not isinstance(self._provider, interface):
            raise StageError(f"{self._provider.name} is not a {interface.__name__}")
        return self._provider

    def close(self) -> None:
        """Unload the provider if it was ever loaded."""
        if self._provider is not None:
            self._provider.unload()
            self._provider = None


@dataclass(frozen=True)
class UnitRun:
    """What a stage may touch while executing one unit."""

    ctx: PipelineContext
    manifest: Manifest
    provider: ProviderHandle
    targets: Targets


class Stage(ABC, Generic[InputsT]):
    """A pipeline step. Subclasses declare their units and how to execute one."""

    name: ClassVar[StageName]
    provider_kind: ClassVar[ProviderKind | None] = None

    @abstractmethod
    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[InputsT]]:
        """All units this stage consists of right now, fresh or not.

        Raises :class:`StageError` if an upstream stage has not produced what
        this one needs.
        """

    @abstractmethod
    def execute(self, unit: WorkUnit[InputsT], run: UnitRun) -> None:
        """Do the work for one unit, writing every declared output to ``run.targets``."""
