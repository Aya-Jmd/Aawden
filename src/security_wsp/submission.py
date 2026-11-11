"""Explicit allowlist packaging, without any network publication."""
import zipfile
import json
from pathlib import Path
from .artifacts import file_hash, read_json, verify_run


def package_submission(run: Path, output: Path):
    verify_run(run)
    manifest = read_json(run / "manifest.json")
    root = Path(__file__).resolve().parents[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    for folder in ("src", "configs", "tests", "scripts", "datasets", "docs", "fixtures"):
        directory = root / folder
        if directory.is_dir():
            sources.extend(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".tmp"})
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
                 "requirements-core.lock", "requirements-research.in", "requirements-research.lock",
                 "dependency-sources.lock.json", "aawden.py", ".gitignore"):
        if (root / name).is_file():
            sources.append(root / name)
    summary = read_json(run / "summary.json")
    kind = summary.get("evidence_kind", manifest["evidence_kind"])
    adapter_run = manifest["stage"] in {"adapter_competence", "adapter_prompt_review"}
    if adapter_run:
        from .adapter_evidence import audit_adapter
        audit_adapter(run, root / "datasets/neutral-confirmation-v5.json", root / "configs/selected-review.json")
    note = ("Aawden local research bundle. Evidence kind: " + kind + ".\n" +
            ("Constructed toy measurements do not establish a Cohere improvement.\n" if kind == "constructed_toy" else
             "Saved Aya measurements are a single-author competence feasibility study.\n") +
            "Independent final data, advice-robustness and coding-regression measurements, explanation review, "
            "owner-approved public artifacts and submission remain pending. This bundle has not been published.\n")
    secrets = []
    env = root / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET")):
                value = value.strip().strip("'\"")
                if len(value) >= 12:
                    secrets.append(value.encode())
    inventory = {}
    staging = output.with_name("." + output.name + ".tmp")
    with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SUBMISSION_STATUS.txt", note)
        def include(path, name):
            data = path.read_bytes()
            if any(secret in data for secret in secrets):
                raise ValueError("A private credential was found in a proposed bundle member")
            archive.writestr(name, data)
            inventory[name] = file_hash(path)
        for path in sorted(set(sources)):
            include(path, path.relative_to(root).as_posix())
        for path in run.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".md") and "__pycache__" not in path.parts and ".audit-cache" not in path.parts:
                include(path, "evidence/" + path.relative_to(run).as_posix())
        if adapter_run:
            for name in ("protocol.json", "frozen.json", "training.json", "outcome.json"):
                path = run.parent / name
                if path.is_file():
                    include(path, "supporting/" + name)
            for path in (run.parent / "development").rglob("*.json"):
                include(path, "supporting/development/" + path.relative_to(run.parent / "development").as_posix())
            for study in ("adapter-v1", "adapter-v2", "adapter-v3", "adapter-v5", "adapter-v6", "adapter-v7",
                          "adapter-review-v1", "mental-review-v1", "mental-review-v2"):
                for name in ("protocol.json", "outcome.json", "selection.json"):
                    path = run.parents[2] / study / "study" / name
                    if path.is_file():
                        include(path, "supporting/attempts/" + study + "/" + name)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps({"files": inventory, "evidence_kind": kind,
                           "uploaded": False, "submission_ready": False, "model_weights_included": False}, indent=2))
    with zipfile.ZipFile(staging) as archive:
        if archive.testzip() is not None:
            raise ValueError("The research bundle failed its ZIP integrity check")
    staging.replace(output)
    return {"bundle": str(output.resolve()), "sha256": file_hash(output), "evidence_kind": kind,
            "uploaded": False, "submission_ready": False}

