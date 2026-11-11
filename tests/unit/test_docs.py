"""Documentation that must not drift from the registry and the code."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from shortgen import __version__
from shortgen.config import default_config_dir
from shortgen.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ModelRegistry.from_files(default_config_dir() / "models.yaml")


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_every_registered_model_is_in_the_license_table() -> None:
    licenses = read("docs/licenses.md")
    for spec in REGISTRY.specs():
        assert f"`{spec.name}`" in licenses, f"{spec.name} is missing from docs/licenses.md"
        if spec.model_id:
            assert spec.model_id in licenses, f"{spec.model_id} is missing from docs/licenses.md"
            assert spec.license in licenses, f"license of {spec.name} differs from docs/licenses.md"
        for extra in spec.extra_weights:
            assert extra.repo in licenses, f"{extra.repo} is missing from docs/licenses.md"


def test_no_registered_model_is_non_commercial_without_saying_so() -> None:
    licenses = read("docs/licenses.md")
    for spec in REGISTRY.specs():
        row = next(line for line in licenses.splitlines() if f"`{spec.name}`" in line)
        if spec.commercial_use is False:
            assert "No" in row, f"{spec.name} is non-commercial but its row does not say so"
        if spec.commercial_use is None:
            assert "terms" in row, f"{spec.name}: the row must point at the provider's terms"
        if spec.hosted:
            assert "hosted" in row.lower() and "paid" in row.lower()


def test_hosted_adapters_are_documented_as_optional_and_unverified() -> None:
    hosted = read("docs/hosted.md")
    for phrase in ("off by default", "not been verified", "HF_KEY", "configs/higgsfield.yaml"):
        assert phrase in hosted, f"docs/hosted.md must mention: {phrase}"
    assert "docs/hosted.md" in read("README.md")


def test_readme_documents_every_cli_command_and_real_files() -> None:
    readme = read("README.md")
    for command in ("init", "plan", "generate", "render", "run", "status", "doctor"):
        assert f"shortgen {command}" in readme
    visible = re.sub(r"<!--.*?-->", "", readme, flags=re.DOTALL)
    for link in re.findall(r"\]\((?!https?://|#)([^)]+)\)", visible):
        assert (ROOT / link).exists(), f"README links to a missing file: {link}"
    pictures = re.findall(r'<img[^>]*\ssrc="(?!https?://)([^"]+)"', visible)
    assert pictures, "the README shows real output"
    for picture in pictures:
        assert (ROOT / picture).is_file(), f"README shows a missing picture: {picture}"
        assert (ROOT / picture).stat().st_size < 4_000_000, f"{picture} is too large to commit"


def test_docs_only_link_to_files_that_exist() -> None:
    for document in sorted((ROOT / "docs").glob("*.md")):
        for link in re.findall(
            r"\]\((?!https?://|#)([^)#]+)", document.read_text(encoding="utf-8")
        ):
            assert (document.parent / link).exists(), f"{document.name} links to missing {link}"


def test_default_config_only_names_registered_providers() -> None:
    defaults = yaml.safe_load(read("configs/default.yaml"))
    smoke = yaml.safe_load(read("configs/smoke.yaml"))
    names = {spec.name for spec in REGISTRY.specs()}
    for config in (defaults, smoke):
        assert set(config["providers"].values()) <= names


def test_version_is_consistent() -> None:
    assert f'version = "{__version__}"' in read("pyproject.toml")
    assert f"## [{__version__}]" in read("CHANGELOG.md")
