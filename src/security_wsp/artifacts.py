"""Canonical identities, atomic artifacts, and explicit run lifecycle."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def edit_manifest_path(checkpoint: Path) -> Path:
    return checkpoint / "aawden_edit.json"


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".aawden-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
                   if p.is_file() and p.suffix in {".py", ".html", ".css", ".js"}})


def environment() -> dict:
    packages = {}
    for package in ("torch", "transformers", "transformer-lens", "circuitkit", "auditkit", "safetensors"):
        try:
            dist = importlib.metadata.distribution(package)
            direct = dist.read_text("direct_url.json")
            packages[package] = {"version": dist.version, "direct_url": json.loads(direct) if direct else None}
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=root).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    locks = {name: file_hash(root / name) for name in
             ("requirements-core.lock", "requirements-research.lock", "dependency-sources.lock.json") if (root / name).is_file()}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "code_commit": commit,
            "code_sha256": source_fingerprint(), "dependency_lock_hashes": locks}


class RunStore:
    """Resume only an identical experiment; validate every completed prediction."""
    def __init__(self, path: str | Path, identity: dict, *, stage: str, evidence_kind: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity_hash = digest(identity)
        self.manifest_path = self.path / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            if self.manifest["identity_sha256"] != self.identity_hash:
                raise ValueError("Resume rejected: configuration, source, model, prompts, or dataset changed. Use a new output directory.")
        else:
            self.manifest = {"run_id": self.identity_hash[:16], "parent_run": identity.get("parent_run"), "identity_sha256": self.identity_hash,
                             "identity": identity, "stage": stage, "status": "planned", "evidence_kind": evidence_kind,
                             "created_at": utc_now(), "environment": environment(), "failure_reason": None}
            self._save()

    def _save(self):
        atomic_json(self.manifest_path, self.manifest)

    def start(self, max_attempts=None):
        attempts = self.manifest.setdefault("attempts", [])
        if max_attempts is not None and len(attempts) >= max_attempts:
            raise ValueError("Run retry limit reached; inspect the previous failures")
        now = utc_now()
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1].update(status="interrupted", ended_at=now, reason="Previous process stopped without recording an outcome")
        attempts.append({"status": "running", "started_at": now})
        self.manifest.update(status="running", started_at=now, failure_reason=None)
        self._save()

    def bind_model(self, arm, metadata):
        models = self.manifest.setdefault("model_sha256_by_arm", {})
        identity = digest(metadata)
        if arm in models and models[arm] != identity:
            raise ValueError(f"Loaded checkpoint or runtime changed for {arm}; resume rejected")
        models[arm] = identity
        self._save()

    def finish(self, expected: list[str], artifacts: list[str]):
        for request_id in expected:
            if self.load_prediction(request_id) is None:
                raise ValueError(f"Missing prediction: {request_id}")
        for name in artifacts:
            if not (self.path / name).is_file():
                raise ValueError(f"Missing artifact: {name}")
        hashes = {name: file_hash(self.path / name) for name in artifacts}
        hashes.update({f"predictions/{r}.json": file_hash(self.path / "predictions" / f"{r}.json") for r in expected})
        now = utc_now()
        if self.manifest.get("attempts") and self.manifest["attempts"][-1]["status"] == "running":
            self.manifest["attempts"][-1].update(status="completed", ended_at=now)
        self.manifest.update(status="completed", completed_at=now, artifact_hashes=hashes, expected_predictions=len(expected))
        self._save()

    def fail(self, error: Exception | str, interrupted=False):
        # Exceptions must contain diagnostic text only, never authenticated URLs.
        status, now = ("interrupted" if interrupted else "failed"), utc_now()
        if self.manifest.get("attempts") and self.manifest["attempts"][-1]["status"] == "running":
            self.manifest["attempts"][-1].update(status=status, ended_at=now, reason=str(error))
        self.manifest.update(status=status, failure_reason=str(error), ended_at=now)
        self._save()

    def save_prediction(self, request_id: str, record: dict):
        # Hash the bytes that will actually be written. JSON turns integer keys into strings, so
        # digesting the in-memory object would record a fingerprint of a different structure and
        # load_prediction would then report the run as corrupted (measured on calls_per_layer,
        # runs/wsp-v3/localize-aya). Round-trip first so the digest covers the stored form.
        stored = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
        atomic_json(self.path / "predictions" / f"{request_id}.json", {"record": stored, "sha256": digest(stored)})

    def load_prediction(self, request_id: str) -> dict | None:
        path = self.path / "predictions" / f"{request_id}.json"
        if not path.exists():
            return None
        item = read_json(path)
        if digest(item["record"]) != item["sha256"]:
            raise ValueError(f"Corrupted prediction artifact: {request_id}")
        return item["record"]


def verify_run(path: str | Path) -> dict:
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("Run is not complete")
    if digest(manifest["identity"]) != manifest["identity_sha256"]:
        raise ValueError("Run identity integrity failure")
    root = path.resolve()
    for name, expected in manifest["artifact_hashes"].items():
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or file_hash(artifact) != expected:
            raise ValueError(f"Artifact integrity failure: {name}")
    return {"valid": True, "run_id": manifest["run_id"], "verified_artifacts": len(manifest["artifact_hashes"])}

