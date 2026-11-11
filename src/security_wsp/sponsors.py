"""Pinned sponsor imports and genuine AuditKIT scoring of recorded outputs."""
from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path
from .artifacts import atomic_json, read_json, digest
from .parsing import parse_response

PINS = {"CircuitKIT": "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78",
        "AuditKIT": "71316da345784ff220d5ed59337441a59ab31526"}


def enable_vendor():
    root = Path(__file__).resolve().parents[2]
    for name, revision in PINS.items():
        source = root / ".vendor" / f"{name}-{revision}" / "src"
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))


def audit_records(rows: list[dict], output: Path) -> dict:
    """AuditKIT evaluates precomputed raw JSON through our independent parser."""
    enable_vendor()
    import auditkit as ak
    samples = [ak.Sample(input=row["request_id"], target=row["oracle_verdict"],
                         id=f"{row['method']}:{row['request_id']}", actual_output=row["raw_output"],
                         metadata={"category": row["category"], "line_count": row["source_line_count"]})
               for row in rows]
    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="aawden_strict_verdict")
    def strict_verdict(sample, output):
        parsed, _ = parse_response(output, category=sample.metadata["category"], line_count=sample.metadata["line_count"])
        return float(parsed is not None and parsed["verdict"] == sample.target)
    # AuditKIT creates its cache even for recorded outputs. Keep all writes
    # inside the selected run rather than using a global user cache.
    old_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str((output.parent / ".audit-cache").resolve())
    try:
        result = ak.evaluate(samples, model="precomputed", scorers=strict_verdict,
                             config=ak.RunConfig(max_retries=0, track_performance=False,
                                                 extra={"aawden_records_sha256": digest(rows), "parser": "strict-v1"}))
    finally:
        if old_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = old_cache
    serialized = result.to_dict()
    atomic_json(output, serialized)
    return {"status": "executed", "toolkit": "AuditKIT", "version": ak.__version__,
            "pinned_source_revision": PINS["AuditKIT"], "mode": "strict semantic scoring of precomputed raw generations",
            "samples": len(samples), "fingerprint": result.fingerprint, "artifact": output.name,
            "scope": "Benchmark labels and statistical inference remain owned by Aawden."}


def toolkit_status() -> dict:
    enable_vendor()
    result = {}
    for name, module in (("CircuitKIT", "circuitkit"), ("AuditKIT", "auditkit")):
        try:
            package = importlib.import_module(module)
            result[name] = {"available": True, "version": package.__version__, "required_revision": PINS[name],
                            "source": str(Path(package.__file__).resolve())}
        except Exception as exc:
            result[name] = {"available": False, "required_revision": PINS[name], "error_type": type(exc).__name__}
    return result

