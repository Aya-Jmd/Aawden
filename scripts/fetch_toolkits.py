"""Fetch pinned sponsor source snapshots into the ignored workspace vendor folder."""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "CircuitKIT": "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78",
    "AuditKIT": "71316da345784ff220d5ed59337441a59ab31526",
}
records = {}
for name, revision in PINS.items():
    url = f"https://codeload.github.com/Lexsi-Labs/{name}/zip/{revision}"
    destination = ROOT / ".vendor" / f"{name}-{revision}"
    if destination.is_dir():
        print(f"{name}: pinned source already present", flush=True)
        continue
    request = urllib.request.Request(url, headers={"User-Agent": "Aawden-hackathon/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read()
    expected = json.loads((ROOT / "dependency-sources.lock.json").read_text(encoding="utf-8"))["sources"][name]["archive_sha256"]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Pinned source archive SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        vendor = (ROOT / ".vendor").resolve()
        for member in archive.infolist():
            if not (vendor / member.filename).resolve().is_relative_to(vendor):
                raise ValueError("Unsafe archive member")
        archive.extractall(vendor)
    records[name] = {"revision": revision, "archive_sha256": hashlib.sha256(payload).hexdigest(), "url": url}
    print(f"{name}: downloaded and verified archive paths", flush=True)
if records:
    lock = ROOT / ".vendor" / "source-lock.json"
    previous = json.loads(lock.read_text()) if lock.exists() else {}
    previous.update(records)
    lock.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
