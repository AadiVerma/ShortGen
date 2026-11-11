"""Hashing, the workspace and the manifest: the foundations of resumability."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shortgen.domain.models import Asset, AssetKind, Job, JobStatus, Project
from shortgen.errors import ManifestError, PathValidationError, StageError
from shortgen.pipeline import workspace as workspace_module
from shortgen.pipeline.hashing import file_sha256, hash_model, stable_hash
from shortgen.pipeline.manifest import Manifest, ManifestStore, job_key
from shortgen.pipeline.workspace import (
    Workspace,
    atomic_write_text,
    partial_path,
    resolve_inside,
    validate_input_file,
)


def manifest() -> Manifest:
    return Manifest(project=Project(name="demo", script="Hello.", created_at=datetime.now(UTC)))


def test_stable_hash_ignores_key_order_but_not_values() -> None:
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_model_hash_tracks_content() -> None:
    first, second = manifest().project, manifest().project
    second.created_at = first.created_at
    assert hash_model(first) == hash_model(second)
    second.script = "Changed."
    assert hash_model(first) != hash_model(second)


def test_file_hash_is_sha256_of_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_paths_cannot_escape_the_project(tmp_path: Path) -> None:
    assert resolve_inside(tmp_path, "audio/s001.wav") == tmp_path.resolve() / "audio/s001.wav"
    for escape in ("../outside.txt", "/etc/passwd", "audio/../../x"):
        with pytest.raises(PathValidationError, match="escapes"):
            resolve_inside(tmp_path, escape)


def test_input_files_are_vetted(tmp_path: Path) -> None:
    image = tmp_path / "ref.PNG"
    image.write_bytes(b"x")
    assert validate_input_file(image, {".png"}) == image.resolve()
    with pytest.raises(PathValidationError, match="not a file"):
        validate_input_file(tmp_path / "missing.png", {".png"})
    with pytest.raises(PathValidationError, match="unsupported file type"):
        validate_input_file(image, {".jpg"})


def test_partial_path_keeps_the_extension() -> None:
    assert partial_path(Path("clips/s001.mp4")) == Path("clips/s001.partial.mp4")


def test_atomic_write_replaces_whole_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.txt"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text() == "two"
    assert [p.name for p in target.parent.iterdir()] == ["file.txt"]  # no temp files left


def test_failed_atomic_write_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.txt"
    atomic_write_text(target, "original")

    def boom(_fd: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(workspace_module.os, "fsync", boom)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_text(target, "half-written")
    assert target.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]


def test_manifest_round_trips_through_the_store(tmp_path: Path) -> None:
    store = ManifestStore(Workspace(tmp_path))
    assert not store.exists()
    original = manifest()
    original.jobs[job_key("image", "s001")] = Job(stage="image", scene_id="s001", inputs_hash="h")
    store.save(original)
    assert store.load() == original


def test_missing_manifest_points_at_init(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="shortgen init"):
        ManifestStore(Workspace(tmp_path)).load()


def test_corrupt_or_foreign_manifest_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    store = ManifestStore(workspace)
    workspace.manifest_path.write_text("{not json")
    with pytest.raises(ManifestError, match="cannot read manifest"):
        store.load()
    future = manifest().model_copy(update={"schema_version": 99})
    workspace.manifest_path.write_text(future.model_dump_json())
    with pytest.raises(ManifestError, match="schema 99 is not supported"):
        store.load()


def test_asset_lookup_requires_a_finished_job() -> None:
    m = manifest()
    with pytest.raises(StageError, match="'image' has not completed for s001"):
        m.asset("image", "s001", AssetKind.IMAGE)
    asset = Asset(kind=AssetKind.IMAGE, path="images/s001.png", sha256="abc", scene_id="s001")
    m.jobs[job_key("image", "s001")] = Job(
        stage="image", scene_id="s001", inputs_hash="h", status=JobStatus.FAILED, outputs=[asset]
    )
    with pytest.raises(StageError):
        m.asset("image", "s001", AssetKind.IMAGE)
    m.jobs[job_key("image", "s001")].status = JobStatus.DONE
    assert m.asset("image", "s001", AssetKind.IMAGE) == asset
    with pytest.raises(ManifestError, match="recorded no clip output"):
        m.asset("image", "s001", AssetKind.CLIP)
