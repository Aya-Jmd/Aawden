#!/usr/bin/env python3
"""Phase 6 Hugging Face publishing: plan, stage, build Space examples, and an owner-gated publish.

Commands:
  plan                       list what would be uploaded, sizes, availability and secret scan (offline)
  stage --out dist/hf-staging
                             copy files and render cards into per-repo folders (offline)
  build-space-examples       build hf/space/examples.json from wsp_measure results (offline)
  publish --confirm-owner-approved
                             create public, ungated repos and collections with HF_PUBLISH_TOKEN (network)

Run it as a script (python scripts/hf_publish.py ...). It imports only the standard library until
publish, which then imports huggingface_hub. build-space-examples reads run results.json files
directly and needs no project imports.

Nothing in this file publishes without --confirm-owner-approved. Sealed splits (wsp_final) are
excluded unless --include-final is passed. Base model weights and checkpoint files outside the
explicit allowlist are refused.

The secret scanner mirrors src/security_wsp/submission.py: it matches the values of .env keys
containing TOKEN, KEY or SECRET (values of at least 12 characters) byte-for-byte. submission.py keeps
that logic inline rather than as an importable function, so it is reproduced here. Two pattern
checks are added: an hf_ token shape and a heuristic for Beam-style tokens. Matched values are
never printed; only the file and the rule name are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hf-publish.json"
DEFAULT_STAGE = ROOT / "dist" / "hf-staging"
SEALED_FILE_MARK = "sealed"
PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")

HF_TOKEN_SHAPE = re.compile(r"hf_[A-Za-z0-9]{30,}")
ASSIGNED_SECRET = re.compile(r"(?i)(token|secret|api[_-]?key|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-=/+.]{20,})")
# Beam tokens are base64url-ish runs of 72+ characters mixing upper, lower and digits. Shorter runs and
# '='-padded shares are common in logs, so they are left to the exact .env value match.
BEAM_TOKEN_SHAPE = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]{72,}(?![A-Za-z0-9_\-])")


class PublishRefused(Exception):
    """Raised when a safety gate stops the run. The message says which gate and why."""


# ---------------------------------------------------------------- config and env

def load_config(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. Values are returned to the caller only; never printed."""
    if not Path(path).is_file():
        return {}
    values = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and not key.strip().startswith("#"):
            values[key.strip()] = value.strip().strip("'\"")
    return values


def env_secret_values(env: dict[str, str]) -> list[tuple[str, bytes]]:
    """Same rule as submission.py: key contains TOKEN/KEY/SECRET and the value has 12+ characters."""
    out = []
    for key, value in env.items():
        if any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET")) and len(value) >= 12:
            out.append((key, value.encode()))
    return out


# ---------------------------------------------------------------- secret scanning

def scan_bytes(data: bytes, secrets: list[tuple[str, bytes]]) -> list[str]:
    """Return rule names for every match. Matched text is never returned."""
    findings = []
    for name, secret in secrets:
        if secret and secret in data:
            findings.append(f"env value of {name}")
    text = data.decode("utf-8", errors="ignore")
    if HF_TOKEN_SHAPE.search(text):
        findings.append("hf_ token pattern")
    if ASSIGNED_SECRET.search(text):
        findings.append("assigned token/secret/key value")
    for match in BEAM_TOKEN_SHAPE.finditer(text):
        token = match.group(0)
        if re.search(r"[0-9]", token) and re.search(r"[A-Z]", token) and re.search(r"[a-z]", token):
            findings.append("Beam-style token pattern (heuristic)")
            break
    return findings


# ---------------------------------------------------------------- resolution

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _glob_match(relative: str, pattern: str) -> bool:
    """Glob match against a posix relative path, with ** spanning directories (Python 3.11)."""
    if "/" not in pattern:
        return Path(relative).match(pattern)
    translated = ""
    for part in pattern.split("/"):
        if part == "**":
            translated += "(?:.*/)?"
        else:
            translated += re.escape(part).replace(r"\*", "[^/]*").replace(r"\?", "[^/]") + "/"
    return re.fullmatch(translated.rstrip("/") + "", relative) is not None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_entry(entry: dict, root: Path, include_final: bool, excludes: list[str] | None = None) -> dict:
    """Expand an entry's sources into concrete files. Returns files plus missing and excluded lists."""
    files, missing, excluded = [], [], []
    excludes = excludes or []
    for spec in entry.get("sources", []):
        if spec.get("sealed") and not include_final:
            excluded.append(f"sealed: {spec.get('path') or spec.get('glob')}")
            continue
        if "path" in spec:
            matches = [root / spec["path"]]
        else:
            matches = sorted(p for p in root.glob(spec["glob"]) if p.is_file())
        required = spec.get("required", True)
        existing = [m for m in matches if m.is_file()]
        if not existing:
            if required:
                missing.append(spec.get("path") or spec.get("glob"))
            continue
        for match in existing:
            relative = _rel(match, root)
            if any(_glob_match(relative, pattern) for pattern in excludes):
                excluded.append(f"pattern-excluded: {relative}")
                continue
            dest = spec["as"] if "as" in spec and "path" in spec else _rel(match, root)
            files.append({"src": match, "dest": dest, "size": match.stat().st_size, "required": required,
                          "explicit": "path" in spec})
    return {"files": files, "missing": missing, "excluded": excluded}


def _checkpoint_violations(entry: dict, root: Path) -> list[str]:
    """checkpoints/ may only be uploaded through explicit `path` entries, never globs."""
    problems = []
    for spec in entry.get("sources", []):
        glob = spec.get("glob", "")
        if glob.startswith("checkpoints") or "checkpoints/" in glob:
            problems.append(f"glob under checkpoints/ is not allowed: {glob}")
    return problems


def _safetensors_keys(path: Path) -> list[str]:
    """Key names from the safetensors header JSON. Only the header is read."""
    with Path(path).open("rb") as handle:
        head = handle.read(8)
        if len(head) != 8:
            raise PublishRefused(f"{path.name}: not a safetensors file (header shorter than 8 bytes)")
        (length,) = struct.unpack("<Q", head)
        if length <= 0 or length > 1 << 26:
            raise PublishRefused(f"{path.name}: implausible safetensors header length")
        header = json.loads(handle.read(length).decode("utf-8"))
    return [key for key in header if key != "__metadata__"]


def patch_violations(entry: dict, path: Path) -> list[str]:
    """A patch may not carry base-weight slices; the revert copies of base tensors are base weights."""
    if not entry.get("forbid_base_slices") or path.suffix.lower() != ".safetensors":
        return []
    keys = _safetensors_keys(path)
    if any(key.endswith("::original") for key in keys):
        return [f"{path.name}: contains ::original base tensor slices, which are base weights; refused "
                f"(strip them from the patch before publishing)"]
    return []


def _weight_violations(path: Path, limits: dict) -> list[str]:
    name = path.name
    suffix = path.suffix.lower()
    problems = []
    if suffix in limits["weight_suffixes"] and path.stat().st_size > limits["max_weight_bytes"]:
        problems.append(f"{name}: weight file over {limits['max_weight_bytes']} bytes refused")
    for pattern in limits.get("base_shard_name_patterns", []):
        if re.search(pattern, name):
            problems.append(f"{name}: base model shard name refused")
    if suffix in limits["weight_suffixes"] and suffix not in limits["allowed_suffixes"]:
        problems.append(f"{name}: weight suffix {suffix} is not allowed")
    if suffix not in limits["allowed_suffixes"]:
        problems.append(f"{name}: suffix {suffix or '(none)'} is not on the allowlist")
    return problems


def build_plan(config: dict, root: Path = ROOT, include_final: bool = False, env_path: Path | None = None) -> dict:
    """Offline plan: availability, sizes, weight/checkpoint rules, expected hashes and the secret scan."""
    env_path = env_path if env_path is not None else root / ".env"
    env = read_env(env_path)
    secrets = env_secret_values(env)
    aya = env.get("AAWDEN_AYA_HF_TOKEN", "")
    if aya:
        # The Aya read token is never used for writes and must never appear in a published log.
        secrets = secrets + [("AAWDEN_AYA_HF_TOKEN (read token)", aya.encode())]
    limits = config["limits"]
    report = {"errors": [], "repos": [], "include_final": include_final, "secret_values_loaded": len(secrets)}
    for entry in config["repos"]:
        item = {"id": entry["id"], "type": entry["type"], "collection": entry.get("collection"),
                "declared_available": entry.get("available"), "files": [], "errors": [], "warnings": [],
                "excluded": [], "missing": [], "total_bytes": 0}
        resolved = resolve_entry(entry, root, include_final, config.get("exclude_patterns", []))
        item["missing"], item["excluded"] = resolved["missing"], resolved["excluded"]
        item["errors"].extend(_checkpoint_violations(entry, root))
        item["available"] = not resolved["missing"] and bool(resolved["files"])
        if entry.get("available") is not None and entry["available"] != item["available"]:
            item["warnings"].append(f"config says available={entry['available']}, filesystem says {item['available']}")
        for name, expected in entry.get("expected_sha256", {}).items():
            path = root / name
            if path.is_file() and _sha256(path) != expected:
                item["errors"].append(f"{name}: sha256 does not match configs/hf-publish.json")
        for file in resolved["files"]:
            src = file["src"]
            problems = _weight_violations(src, limits)
            if src.suffix.lower() in limits["weight_suffixes"] and not file["explicit"]:
                problems.append(f"{file['dest']}: weight files must be listed by explicit path")
            try:
                problems.extend(patch_violations(entry, src))
            except PublishRefused as problem:
                problems.append(str(problem))
            if problems:
                item["errors"].extend(problems)
                item["files"].append({"dest": file["dest"], "size": file["size"], "status": "refused"})
                continue
            findings = scan_bytes(src.read_bytes(), secrets)
            if findings:
                item["errors"].append(f"{file['dest']}: secret scan matched ({', '.join(sorted(set(findings)))})")
                item["files"].append({"dest": file["dest"], "size": file["size"], "status": "secret-match"})
                continue
            item["files"].append({"dest": file["dest"], "size": file["size"], "status": "ok"})
            item["total_bytes"] += file["size"]
        card = entry.get("card")
        if card and (root / card).is_file():
            card_text = (root / card).read_text(encoding="utf-8")
            if scan_bytes(card_text.encode(), secrets):
                item["errors"].append(f"{card}: secret scan matched")
        if item["available"] is False:
            item["warnings"].append("not available: skipped by stage and publish until its sources exist")
        report["errors"].extend(f"{entry['id']}: {e}" for e in item["errors"])
        report["repos"].append(item)
    return report


def format_plan(report: dict) -> str:
    lines = [f"include_final={report['include_final']}  secret values loaded={report['secret_values_loaded']}"]
    for repo in report["repos"]:
        state = "AVAILABLE" if repo["available"] else "UNAVAILABLE"
        lines.append(f"\n[{repo['type']}] {repo['id']}  {state}  collection={repo['collection']}  "
                     f"upload={repo['total_bytes'] / 1e6:.2f} MB")
        for file in repo["files"]:
            lines.append(f"    {file['status']:<12} {file['size']:>12,d}  {file['dest']}")
        for missing in repo["missing"]:
            lines.append(f"    missing      {'':>12}  {missing}")
        for excluded in repo["excluded"]:
            lines.append(f"    excluded     {'':>12}  {excluded}")
        for warning in repo["warnings"]:
            lines.append(f"    warning: {warning}")
        for error in repo["errors"]:
            lines.append(f"    ERROR: {error}")
    lines.append("\nRESULT: " + ("refused" if report["errors"] else "clean"))
    return "\n".join(lines)


# ---------------------------------------------------------------- cards and results

def _fmt(value) -> str:
    return "-" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def results_table(summary_paths: list[Path], root: Path) -> str:
    """Tabulate saved summary.json files. Unknown schemas are listed, not guessed at."""
    single, arms, unknown = [], [], []
    for path in summary_paths:
        rel = _rel(path, root)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("metrics"), dict):
            m = data["metrics"]
            splits = ", ".join((data.get("dataset") or {}).get("splits", [])) or "-"
            single.append(f"| `{rel}` | {splits} | {_fmt(m.get('n'))} | {_fmt(m.get('balanced_accuracy'))} | "
                          f"{_fmt(m.get('false_positive_rate'))} | {_fmt(m.get('false_negative_rate'))} | "
                          f"{_fmt(m.get('invalid_count'))} |")
        elif isinstance(data.get("summary"), dict) and isinstance(data["summary"].get("by_arm"), dict):
            for arm, conditions in data["summary"]["by_arm"].items():
                for condition, m in conditions.items():
                    arms.append(f"| `{rel}` | {arm} | {condition} | {_fmt(m.get('n'))} | "
                                f"{_fmt(m.get('balanced_accuracy'))} |")
        else:
            unknown.append(f"`{rel}`")
    if not (single or arms or unknown):
        return "Results pending: no saved summary.json matched this artifact at staging time. No numbers are shown."
    parts = []
    if single:
        parts.append("**Single-run metrics** (balanced accuracy on executable oracle labels)\n\n"
                     "| Summary | Splits | n | Balanced acc. | FPR | FNR | Invalid |\n"
                     "|---|---|---|---|---|---|---|\n" + "\n".join(single))
    if arms:
        parts.append("**Arm by condition**\n\n| Summary | Arm | Condition | n | Balanced acc. |\n"
                     "|---|---|---|---|---|\n" + "\n".join(arms))
    if unknown:
        parts.append("Summaries with an unrecognized schema (not tabulated): " + ", ".join(unknown))
    return "\n\n".join(parts)


def _render_card(entry: dict, root: Path, staged: list[dict], namespace: str, results: str) -> str:
    template = (root / entry["card"]).read_text(encoding="utf-8")
    summaries = []
    for pattern in entry.get("summary_globs", []):
        summaries.extend(sorted(p for p in root.glob(pattern) if p.is_file()))
    summary_list = "\n".join(f"- `{_rel(p, root)}` (sha256 `{_sha256(p)[:16]}...`)" for p in summaries) or \
        "- none found at staging time"
    files_rows = "\n".join(f"| `{f['dest']}` | {f['size']:,d} | `{f['sha256'][:16]}...` |" for f in staged)
    values = {
        "REPO_ID": f"{namespace}/{entry['id']}",
        "RESULTS_TABLE": results,
        "SUMMARY_FILES": summary_list,
        "FILES_TABLE": "| File | Bytes | SHA-256 (first 16) |\n|---|---|---|\n" + files_rows,
    }
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    leftover = PLACEHOLDER.findall(text)
    if leftover:
        raise PublishRefused(f"{entry['id']}: unfilled placeholders {sorted(set(leftover))}")
    return text


# ---------------------------------------------------------------- stage

def stage(config: dict, out: Path, root: Path = ROOT, include_final: bool = False,
          env_path: Path | None = None) -> dict:
    report = build_plan(config, root=root, include_final=include_final, env_path=env_path)
    if report["errors"]:
        raise PublishRefused("plan has errors; nothing staged:\n  " + "\n  ".join(report["errors"]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    secrets = env_secret_values(read_env(env_path if env_path is not None else root / ".env"))
    manifest = {"namespace": config["namespace"], "include_final": include_final, "repos": []}
    for entry in config["repos"]:
        item = next(r for r in report["repos"] if r["id"] == entry["id"])
        if not item["available"]:
            manifest["repos"].append({"id": entry["id"], "staged": False, "reason": "unavailable"})
            continue
        folder = out / entry["id"]
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        resolved = resolve_entry(entry, root, include_final, config.get("exclude_patterns", []))
        staged = []
        for file in resolved["files"]:
            target = folder / file["dest"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file["src"], target)
            staged.append({"dest": file["dest"], "size": file["size"], "sha256": _sha256(target)})
        if entry.get("card"):
            results = results_table(sorted(p for g in entry.get("summary_globs", []) for p in root.glob(g)
                                           if p.is_file()), root)
            card = _render_card(entry, root, staged, config["namespace"], results)
            (folder / "README.md").write_text(card, encoding="utf-8")
        verify_staged(folder, config["limits"], secrets)
        manifest["repos"].append({"id": entry["id"], "type": entry["type"], "staged": True,
                                  "folder": str(folder), "files": staged})
    (out / "stage-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_staged(folder: Path, limits: dict, secrets: list[tuple[str, bytes]]) -> None:
    """Defense in depth on the exact bytes that will be uploaded."""
    problems = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        problems.extend(_weight_violations(path, limits))
        if scan_bytes(path.read_bytes(), secrets):
            problems.append(f"{path.relative_to(folder).as_posix()}: secret scan matched")
    if problems:
        raise PublishRefused(f"staged folder {folder.name} refused:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- Space examples

DEFAULT_CASE_FILES = ["datasets/wsp-discovery.json", "datasets/wsp-validation.json", "datasets/wsp-coding.json"]


def _load_cases(root: Path, include_final: bool, names: list[str] | None = None) -> dict[str, dict]:
    index = {}
    names = list(names or DEFAULT_CASE_FILES)
    if include_final:
        names.append("datasets/wsp-final.json")
    for name in names:
        path = root / name
        if path.is_file():
            for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
                index[case["case_id"]] = case
    return index


def build_space_examples(run_dirs, patch_manifest, out, *, include_final: bool = False,
                         cases: dict | None = None, root: Path = ROOT,
                         cases_paths: list[str] | None = None) -> dict:
    """Build examples.json from wsp_measure results.json files. Sealed splits are excluded by default.

    Each row needs case_id, category, code, threat_model, condition, method, raw_output,
    predicted_verdict and oracle_verdict. Code and threat_model are looked up by case_id when a
    row does not carry them. The label comes from the run's summary.json when present.
    """
    sealed = {"wsp_final"}
    excluded = set() if include_final else sealed
    index = cases if cases is not None else _load_cases(root, include_final, cases_paths)
    examples, sources = [], []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        results_path = run_dir / "results.json"
        rows = json.loads(results_path.read_text(encoding="utf-8"))
        label = run_dir.name
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            label = json.loads(summary_path.read_text(encoding="utf-8")).get("label", label)
        sources.append({"run": run_dir.name, "label": label, "results_sha256": _sha256(results_path)})
        for row in rows:
            if row.get("split") in excluded:
                continue
            case = index.get(row["case_id"], {})
            examples.append({
                "case_id": row["case_id"],
                "split": row.get("split"),
                "category": row.get("category") or case.get("target_category"),
                "code": row.get("code", case.get("code")),
                "threat_model": row.get("threat_model", case.get("threat_model")),
                "condition": row["condition"],
                "method": row["method"],
                "label": label,
                "raw_output": row.get("raw_output"),
                "predicted_verdict": row.get("predicted_verdict"),
                "oracle_verdict": row.get("oracle_verdict"),
            })
    examples.sort(key=lambda e: (e["case_id"], e["condition"], e["method"], e["label"]))
    patch_summary = None
    if patch_manifest:
        manifest = json.loads(Path(patch_manifest).read_text(encoding="utf-8"))
        patch_summary = {
            "format": manifest.get("format"),
            "changed_scalars": manifest.get("changed_scalars"),
            "total_parameters": manifest.get("total_parameters"),
            "percent_changed": manifest.get("percent_changed"),
            "tensor_count": len(manifest.get("tensors", [])),
            "changed_tensors": [t["name"] for t in manifest.get("tensors", [])][:200],
        }
    payload = {"schema_version": 1, "include_final": include_final, "excluded_splits": sorted(excluded),
               "sources": sources, "patch_summary": patch_summary, "examples": examples}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


# ---------------------------------------------------------------- publish

def publish(config: dict, out: Path, confirm_owner_approved: bool, *, root: Path = ROOT,
            include_final: bool = False, api=None, env_path: Path | None = None) -> dict:
    if not confirm_owner_approved:
        raise PublishRefused("publish refused: pass --confirm-owner-approved after the owner approves the plan")
    env = read_env(env_path if env_path is not None else root / ".env")
    token = env.get("HF_PUBLISH_TOKEN", "")
    if not token:
        raise PublishRefused("HF_PUBLISH_TOKEN is not set in .env")
    if env.get("AAWDEN_AYA_HF_TOKEN") and re.search(
            re.escape(env["AAWDEN_AYA_HF_TOKEN"]), token):
        raise PublishRefused("HF_PUBLISH_TOKEN contains the Aya read token; use a publish-only token")
    manifest = stage(config, out, root=root, include_final=include_final, env_path=env_path)
    ns = config["namespace"]
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    who = api.whoami(token=token).get("name")
    if who != ns:
        raise PublishRefused(f"token resolves to '{who}', expected namespace '{ns}'")
    published = []
    available = {r["id"]: r for r in manifest["repos"] if r.get("staged")}
    for entry in config["repos"]:
        if entry["id"] not in available:
            continue
        repo_id = f"{ns}/{entry['id']}"
        kwargs = {"space_sdk": "gradio"} if entry["type"] == "space" else {}
        api.create_repo(repo_id=repo_id, repo_type=entry["type"], private=False, exist_ok=True, token=token, **kwargs)
        if entry["type"] != "space":
            api.update_repo_settings(repo_id=repo_id, private=False, gated=False, repo_type=entry["type"], token=token)
        api.upload_folder(repo_id=repo_id, repo_type=entry["type"], folder_path=str(out / entry["id"]),
                          commit_message=f"Aawden Phase 6: {entry['id']}", token=token)
        published.append(repo_id)
    collections = {}
    for key, spec in config["collections"].items():
        collection_items = [e for e in config["repos"] if e.get("collection") == key and e["id"] in available]
        if not collection_items:
            continue
        slug = None
        for existing in api.list_collections(owner=ns, token=token):
            if existing.title == spec["title"]:
                slug = existing.slug
                break
        if slug is None:
            slug = api.create_collection(title=spec["title"], namespace=ns, description=spec["description"],
                                         private=False, token=token).slug
        for entry in collection_items:
            api.add_collection_item(collection_slug=slug, item_id=f"{ns}/{entry['id']}", item_type=entry["type"],
                                    exists_ok=True, token=token)
        collections[spec["title"]] = slug
    return {"published": published, "collections": collections}


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=CONFIG)
    common.add_argument("--include-final", action="store_true", help="include sealed wsp_final (after freeze only)")
    sub.add_parser("plan", parents=[common], help="offline plan and secret scan")
    p_stage = sub.add_parser("stage", parents=[common], help="copy files and render cards, offline")
    p_stage.add_argument("--out", type=Path, default=DEFAULT_STAGE)
    p_pub = sub.add_parser("publish", parents=[common], help="owner-gated upload to Hugging Face")
    p_pub.add_argument("--out", type=Path, default=DEFAULT_STAGE)
    p_pub.add_argument("--confirm-owner-approved", action="store_true")
    p_space = sub.add_parser("build-space-examples", help="build hf/space/examples.json, offline")
    p_space.add_argument("--run", type=Path, action="append", required=True, help="wsp_measure output dir")
    p_space.add_argument("--patch-manifest", type=Path)
    p_space.add_argument("--out", type=Path, default=ROOT / "hf" / "space" / "examples.json")
    p_space.add_argument("--include-final", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "build-space-examples":
        payload = build_space_examples(args.run, args.patch_manifest, args.out, include_final=args.include_final)
        print(f"wrote {args.out} with {len(payload['examples'])} examples; excluded splits: {payload['excluded_splits']}")
        return 0

    config = load_config(args.config)
    try:
        if args.command == "plan":
            report = build_plan(config, include_final=args.include_final)
            print(format_plan(report))
            return 1 if report["errors"] else 0
        if args.command == "stage":
            manifest = stage(config, args.out, include_final=args.include_final)
            staged = [r["id"] for r in manifest["repos"] if r.get("staged")]
            print(f"staged {len(staged)} repos into {args.out}: {', '.join(staged)}")
            return 0
        if args.command == "publish":
            result = publish(config, args.out, args.confirm_owner_approved, include_final=args.include_final)
            print(json.dumps(result, indent=2))
            return 0
    except PublishRefused as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
