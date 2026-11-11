"""Content hashing: the basis of idempotent, resumable stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_CHUNK_BYTES = 1 << 20


def stable_hash(value: Any) -> str:
    """SHA-256 of a JSON-serialisable value, independent of key order."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_model(model: BaseModel) -> str:
    """Stable hash of a Pydantic model's JSON form."""
    return stable_hash(model.model_dump(mode="json"))


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large clips stay off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
