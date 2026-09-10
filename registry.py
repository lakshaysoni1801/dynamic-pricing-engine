"""A minimal but real model registry backed by the local filesystem.

Each registered model gets a monotonically increasing integer version, its serialized
artifact, and a metadata record (metrics, params, created_at, content hash). A named
alias ("production") can be pointed at a version, so serving can always resolve "the
current production model" without hardcoding a version. This is the same shape as a
hosted registry (MLflow, SageMaker Model Registry), kept dependency-light so it runs
anywhere.

Concurrency and integrity notes:
  * `register` and `promote` both mutate `index.json` via read-modify-write. A cross-
    process file lock (`filelock`) guards that critical section so two concurrent
    callers (e.g. two training jobs) can't race and silently clobber each other's
    version entry.
  * Index writes are atomic: we write to a temp file in the same directory and
    `os.replace` it over the real index, so a crash mid-write can't leave a half-written
    (corrupt) index behind.
  * `load` recomputes the artifact's SHA-256 and compares it against the hash recorded
    at registration time, raising if they don't match, rather than storing a hash that
    is never actually checked.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from filelock import FileLock

from .config import settings
from .demand_model import DemandModel


@dataclass
class ModelVersion:
    version: int
    created_at: str
    metrics: dict
    params: dict
    artifact_path: str
    content_hash: str


class ModelRegistry:
    def __init__(self, root: str | None = None):
        self.root = Path(root or settings.registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        # A single lock file guards all read-modify-write access to index.json across
        # processes. Using a lock rather than relying on GIL/thread safety matters here
        # because the registry is explicitly designed to be shared across separate
        # training/eval processes (that's the point of a registry).
        self._lock = FileLock(str(self.root / ".registry.lock"))
        if not self._index_path.exists():
            with self._lock:
                if not self._index_path.exists():
                    self._write_index({"versions": [], "aliases": {}})

    def _read_index(self) -> dict:
        return json.loads(self._index_path.read_text())

    def _write_index(self, data: dict) -> None:
        # Atomic write: write to a temp file then rename over the target. os.replace is
        # atomic on POSIX and Windows, so readers never observe a partially-written file.
        tmp_path = self._index_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        os.replace(tmp_path, self._index_path)

    def register(self, model: DemandModel, metrics: dict, params: dict) -> ModelVersion:
        with self._lock:
            index = self._read_index()
            version = len(index["versions"]) + 1
            artifact_path = self.root / f"model_v{version}.joblib"
            joblib.dump(model, artifact_path)
            content_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:16]
            mv = ModelVersion(
                version=version,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metrics=metrics,
                params=params,
                artifact_path=str(artifact_path),
                content_hash=content_hash,
            )
            index["versions"].append(asdict(mv))
            self._write_index(index)
            return mv

    def promote(self, version: int, alias: str = "production") -> None:
        with self._lock:
            index = self._read_index()
            known = {v["version"] for v in index["versions"]}
            if version not in known:
                raise ValueError(f"version {version} is not registered")
            index["aliases"][alias] = version
            self._write_index(index)

    def resolve(self, alias: str = "production") -> int | None:
        return self._read_index()["aliases"].get(alias)

    def load(self, version: int | None = None, alias: str = "production") -> DemandModel:
        index = self._read_index()
        if version is None:
            version = index["aliases"].get(alias)
            if version is None:
                raise ValueError(f"no model promoted to alias '{alias}'")
        record = next((v for v in index["versions"] if v["version"] == version), None)
        if record is None:
            raise ValueError(f"version {version} not found")

        artifact_bytes = Path(record["artifact_path"]).read_bytes()
        actual_hash = hashlib.sha256(artifact_bytes).hexdigest()[:16]
        expected_hash = record["content_hash"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"content hash mismatch for version {version}: expected {expected_hash}, "
                f"got {actual_hash}. The artifact may be corrupted or was modified "
                "outside the registry."
            )
        return joblib.load(io.BytesIO(artifact_bytes))

    def list_versions(self) -> list[dict]:
        return self._read_index()["versions"]
