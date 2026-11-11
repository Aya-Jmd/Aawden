"""Write a redacted copy of runs/wsp-v3/deference-summary.json.

The two adapters trained by other teams are replaced by neutral labels so the
file can be committed to a public repo without naming their authors. Every
other field is copied unchanged; nothing is recomputed.

    python3 scripts/redact_deference.py

Input:  runs/wsp-v3/deference-summary.json
Output: runs/wsp-v3/deference-summary-redacted.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs/wsp-v3/deference-summary.json"
DST = ROOT / "runs/wsp-v3/deference-summary-redacted.json"

# adapter_id -> (replacement id, replacement key suffix in `sets`)
ADAPTERS = {
    "kenil1710/injectno-aya-expanse-8b-v3": "third-party-adapter-A",
    "Mizore66/tiny-aya-global-evidencetune-lora": "third-party-adapter-B",
}

# `sets` keys that carry the adapter's short name
KEY_NAMES = {
    "plain-aya8b-injectno": "plain-aya8b-thirdparty-a",
    "plain-tinyaya-evidencetune": "plain-tinyaya-thirdparty-b",
}


def redact(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "adapter_id" and v in ADAPTERS:
                out[k] = ADAPTERS[v]
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        for real, fake in ADAPTERS.items():
            if real in obj:
                obj = obj.replace(real, fake)
        return obj
    return obj


def main() -> None:
    data = json.loads(SRC.read_text())
    out = redact(data)

    out["sets"] = {
        group: {KEY_NAMES.get(key, key): block for key, block in blocks.items()}
        for group, blocks in out["sets"].items()
    }

    out["redaction"] = {
        "what": "the two adapters trained by other teams are renamed; all numbers unchanged",
        "labels": sorted(ADAPTERS.values()),
    }
    DST.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
