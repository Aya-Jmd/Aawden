"""Deterministic builder for the sealed WSP misleading-opinion datasets.

Usage: python scripts/build_wsp_data.py [OUTPUT_ROOT]

Writes datasets/wsp-{discovery,validation,final}.json, configs/wsp-data.json and
dist/wsp-final-label-review/. Output bytes depend only on the authored catalogs in
src/security_wsp/wsp_cases.py, so two runs produce identical files. The script fails if
any family, source, exact code, or AST-normalized code overlaps an existing dataset or
any other WSP split.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from security_wsp.artifacts import atomic_json, file_hash  # noqa: E402
from security_wsp.data import CLEAN, VULNERABLE, Case, oracle_cases, validate_cases  # noqa: E402
from security_wsp.wsp_cases import SPLITS, run_wsp_oracle, wsp_cases  # noqa: E402

FILES = {
    "wsp_discovery": "datasets/wsp-discovery.json",
    "wsp_validation": "datasets/wsp-validation.json",
    "wsp_final": "datasets/wsp-final.json",
}
CONFIG = "configs/wsp-data.json"
PACKET_DIR = "dist/wsp-final-label-review"
EXPECTED = {"wsp_discovery": (60, 30), "wsp_validation": (40, 20), "wsp_final": (120, 60)}
SEALED = ["wsp_final"]
SEAL_NOTE = ("wsp_final is sealed. It must not be opened for model selection, prompt tuning, or family "
             "edits after any model has run on it. Its labels are for the single final evaluation only.")
CATEGORIES = ("sql_injection", "access_control")
LABELS = (VULNERABLE, CLEAN)


def syntax(code: str) -> str:
    return ast.dump(ast.parse(code), include_attributes=False)


def existing_fingerprints(root: Path) -> dict:
    """Family, source, exact-code and AST keys of every existing dataset case and canonical fixture."""
    seen = {"family": {}, "source": {}, "sha": {}, "ast": {}}
    entries = []
    for path in sorted((root / "datasets").glob("*.json")):
        if f"datasets/{path.name}" in FILES.values():
            continue  # this builder's own outputs are regenerated, so they are not inputs
        for item in json.loads(path.read_text(encoding="utf-8")).get("cases", []):
            entries.append((f"datasets/{path.name}", item))
    for case in oracle_cases().values():
        entries.append(("canonical:" + case.split, asdict(case)))
    for origin, item in entries:
        code = item.get("code")
        if not isinstance(code, str):
            continue
        for key, value in (("family", item.get("family_id")), ("source", item.get("source_id")),
                           ("sha", hashlib.sha256(code.encode("utf-8")).hexdigest()), ("ast", syntax(code))):
            if value:
                seen[key].setdefault(value, origin)
    return seen


def check_disjoint(cases: list[Case], existing: dict) -> dict:
    problems = []
    fresh = {"family": {}, "source": {}, "sha": {}, "ast": {}}
    for case in cases:
        keys = {"family": case.family_id, "source": case.source_id, "sha": case.source_sha256, "ast": syntax(case.code)}
        for kind, value in keys.items():
            if value in existing[kind]:
                problems.append(f"{kind} overlaps existing {existing[kind][value]}: {case.case_id}")
            if value in fresh[kind] and fresh[kind][value] != case.split:
                problems.append(f"{kind} crosses splits: {case.case_id}")
            fresh[kind].setdefault(value, case.split)
    if problems:
        raise SystemExit("WSP disjointness check failed:\n" + "\n".join(problems[:50]))
    return {"existing_fingerprints": {k: len(v) for k, v in existing.items()}, "result": "pass"}


def dataset_payload(split: str, cases: list[Case]) -> dict:
    info = validate_cases(cases)
    info["label_review"] = ("executable_oracle_single_author; sealed; blinded label-review packet prepared"
                            if split in SEALED else "executable_oracle_single_author; development use only")
    info["final_test_protected"] = split in SEALED
    return {"schema_version": 1, "metadata": info, "cases": [asdict(c) for c in cases]}


def counts(cases: list[Case]) -> dict:
    table = {}
    for category in CATEGORIES:
        table[category] = {label: sum(1 for c in cases if c.target_category == category and c.oracle_verdict == label)
                           for label in LABELS}
    return table


def review_packet(root: Path, final_cases: list[Case], final_sha: str) -> dict:
    ordered = sorted(final_cases, key=lambda c: c.case_id)
    instructions = ("Independent label review. For each case, read the code and threat model only, run your own "
                    "trace, then record verdict (vulnerable or not_vulnerable_for_target), the test ids you "
                    "checked, and a note. Do not use any other source of labels.")
    packet = {"schema_version": 1, "dataset_sha256": final_sha, "instructions": instructions,
              "cases": [{"case_id": c.case_id, "source_sha256": c.source_sha256,
                         "required_test_ids": list(c.oracle_test_ids), "language": c.language,
                         "target_category": c.target_category, "threat_model": c.threat_model, "code": c.code}
                        for c in ordered]}
    form = {"schema_version": 1, "dataset_sha256": final_sha, "reviewer_id": None,
            "independent_of_author_and_edit_selection": False, "blinded_to_oracle_labels": False,
            "blinded_to_model_outputs": False, "allowed_verdicts": list(LABELS),
            "cases": [{"case_id": c.case_id, "source_sha256": c.source_sha256, "verdict": None,
                       "passed_test_ids": [], "note": ""} for c in ordered]}
    out = root / PACKET_DIR
    atomic_json(out / "packet.json", packet)
    atomic_json(out / "review-form.json", form)
    return {"packet": f"{PACKET_DIR}/packet.json", "packet_sha256": file_hash(out / "packet.json"),
            "form": f"{PACKET_DIR}/review-form.json", "form_sha256": file_hash(out / "review-form.json"),
            "cases": len(ordered)}


def build(root: Path) -> dict:
    cases = list(wsp_cases())
    existing = existing_fingerprints(REPO)
    disjoint = check_disjoint(cases, existing)
    results = [run_wsp_oracle(c) for c in cases]
    failed = [r["case_id"] for r in results if not r["passed"]]
    if failed:
        raise SystemExit(f"Oracle failures: {failed[:20]}")
    config = {"schema_version": 1, "builder": "scripts/build_wsp_data.py", "generator": "wsp-data-v1",
              "seeds": {"random_seed": None, "note": "No randomness. Cases follow catalog declaration order."},
              "sealed": SEALED, "sealing_note": SEAL_NOTE, "files": {}, "disjointness": disjoint,
              "oracle": {"runner": "security_wsp.wsp_cases.run_wsp_oracle", "checked": len(cases),
                         "passed": sum(1 for r in results if r["passed"])}}
    final_sha = None
    for split in SPLITS:
        split_cases = [c for c in cases if c.split == split]
        expected_count, expected_families = EXPECTED[split]
        if len(split_cases) != expected_count or len({c.family_id for c in split_cases}) != expected_families:
            raise SystemExit(f"{split} has the wrong size")
        payload = dataset_payload(split, split_cases)
        path = root / FILES[split]
        atomic_json(path, payload)
        sha = file_hash(path)
        if split == "wsp_final":
            final_sha = sha
        config["files"][split] = {"path": FILES[split], "sha256": sha, "cases": len(split_cases),
                                  "families": expected_families, "counts": counts(split_cases),
                                  "sealed": split in SEALED}
    final_cases = [c for c in cases if c.split == "wsp_final"]
    config["label_review_packet"] = review_packet(root, final_cases, final_sha)
    atomic_json(root / CONFIG, config)
    return config


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO
    result = build(target)
    print(json.dumps({s: result["files"][s]["sha256"] for s in SPLITS}, indent=2))
