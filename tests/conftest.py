"""Shared fixtures. Nothing here needs a GPU, a model or the network."""

from __future__ import annotations

import shutil

import pytest

from shortgen.config import AppConfig, load_config


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """The shipped defaults, exactly as a user gets them."""
    return load_config()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``ffmpeg``-marked tests when the binaries are not installed."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not installed")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)
