"""Everything a pipeline run needs, passed explicitly instead of living in globals."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from shortgen.config import AppConfig
from shortgen.device import DeviceInfo
from shortgen.pipeline.manifest import ManifestStore
from shortgen.pipeline.progress import LogReporter, Reporter
from shortgen.pipeline.retry import RetryPolicy
from shortgen.pipeline.workspace import Workspace
from shortgen.process import CommandRunner
from shortgen.providers.interfaces import Provider, ProviderContext
from shortgen.providers.spec import ProviderFingerprint, ProviderKind
from shortgen.registry import ModelRegistry


@dataclass(frozen=True)
class PipelineContext:
    """The dependencies of one run. Tests build this with fakes; the CLI with real ones."""

    config: AppConfig
    workspace: Workspace
    store: ManifestStore
    registry: ModelRegistry
    device: DeviceInfo
    commands: CommandRunner
    reporter: Reporter = field(default_factory=LogReporter)
    sleep: Callable[[float], None] = time.sleep
    env: Mapping[str, str] = field(default_factory=dict)

    @property
    def retry_policy(self) -> RetryPolicy:
        """Retry settings from configuration."""
        runtime = self.config.runtime
        return RetryPolicy(runtime.max_attempts, runtime.backoff_base_s, runtime.backoff_max_s)

    def provider_name(self, kind: ProviderKind) -> str:
        """The registry name configured for a provider role."""
        return str(getattr(self.config.providers, kind.value))

    def fingerprint(self, kind: ProviderKind) -> ProviderFingerprint:
        """Identity of the configured provider for ``kind``, for input hashing."""
        name = self.provider_name(kind)
        return self.registry.fingerprint(name, kind, self.config.provider_options.get(name, {}))

    def create_provider(self, kind: ProviderKind) -> Provider:
        """Build (but do not load) the configured provider for ``kind``."""
        name = self.provider_name(kind)
        context = ProviderContext(
            device=self.device,
            commands=self.commands,
            ffmpeg_bin=self.config.render.ffmpeg_bin,
            project_dir=self.workspace.root,
            env=self.env,
        )
        return self.registry.create(name, kind, context, self.config.provider_options.get(name, {}))
