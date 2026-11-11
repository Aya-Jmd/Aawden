"""Real-code evaluation set from a public dataset (Arm 3 of docs/FIX_PROTOCOL.md).

Why this exists
    Every fixture in the main study is templated by one author. This draws Python code written by
    someone else: `CyberNative/Code_Vulnerability_Security_DPO`, where each row pairs a `rejected`
    (vulnerable) and a `chosen` (fixed) implementation of the same task.

Label honesty
    The labels are ASSERTED BY THE DATASET, not checked by an executable oracle. Cases are marked
    `label_status="dataset_asserted_sink_checked"` and the arm is reported as secondary.

Hand-check findings (8 random pairs read before any model run) that shaped the filters below
    - Comments and names CARRY THE LABEL: "# Vulnerable code", "# Secure code", `VulnerableClass` vs
      `SecureClass`. Left in, a model could score by reading the comment. Every comment and docstring
      is stripped and label-bearing words in identifiers and strings are neutralised.
    - Some labels are WRONG: in row 243 the "chosen" side still calls
      `os.system('echo "{}" | bash -s'.format(user_input))`. Others are not vulnerabilities at all
      ("dynamic typing", "buffer overflow" in pure Python).
    So only four categories with a crisp dangerous sink are kept, and each pair must pass a mechanical
    SINK CHECK: the sink is present on the vulnerable side and absent on the fixed side. The label is
    then "dataset-asserted AND sink-checked" -- still not an executable oracle.

Filters, fixed before any model sees the set
    - Python rows only (the `Case` schema requires the code to parse).
    - Both sides parse, are 3-60 lines, and are not byte-identical.
    - Length ratio between the pair within [0.5, 2.0]. In this dataset the fixed side is often much
      longer, so a model could separate the pair by length alone. The script also reports the accuracy
      of a "longer = clean" rule on the kept set, so that leak is measured rather than assumed away.
    - Stratified sample by coarse category, seed 17.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.data import CLEAN, VULNERABLE, Case, write_dataset  # noqa: E402,F401

REPO = "CyberNative/Code_Vulnerability_Security_DPO"
REVISION = "403bbc40790391a595acd591326c5b7c5077b291"

CATEGORIES = (  # first match wins; keyword in the dataset's description -> category shown to the model
    (("pickle", "deserializ", "yaml.load", "marshal"), "insecure_deserialization"),
    (("sql",), "sql_injection"),
    (("os.system", "subprocess", "shell", "command"), "command_injection"),
    (("eval", "exec("), "code_injection"),
)

# Dangerous sink per category: must appear on the vulnerable side and NOT on the fixed side.
SINKS = {
    "insecure_deserialization": re.compile(r"pickle\.loads?\(|cPickle\.loads?\(|marshal\.loads?\("
                                           r"|yaml\.load\((?![^)]*SafeLoader)|yaml\.unsafe_load\("),
    "code_injection": re.compile(r"(?<![\w.])(eval|exec)\("),
    "command_injection": re.compile(r"os\.system\(|os\.popen\(|shell\s*=\s*True|commands\.getoutput\("),
    "sql_injection": None,   # line-based check in sql_built_from_strings()
}

_SQL_WORD = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", re.I)
_SQL_BUILD = re.compile(r"""(\bf["'])|(\.format\()|(["']\s*%\s*[\w(])|(["']\s*\+\s*\w)|(\w\s*\+\s*["'])""")


def sql_built_from_strings(code: str) -> bool:
    """True if any line holding a SQL keyword builds the query by f-string, %, + or .format."""
    return any(_SQL_WORD.search(line) and _SQL_BUILD.search(line) for line in code.splitlines())

LABEL_WORDS = re.compile(r"(?i)(in)?secure|(un)?safe|vulnerab\w*|sanitiz\w*|exploit\w*|malicious|attack\w*|hack\w*")


def scrub(code: str) -> str | None:
    """Drop comments and docstrings, neutralise label-bearing words; None if it no longer parses."""
    import io
    import tokenize
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())
        code = ast.unparse(tree)          # unparse also discards every comment
    except (SyntaxError, ValueError):
        return None
    toks = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in (tokenize.NAME, tokenize.STRING):
                tok = tok._replace(string=LABEL_WORDS.sub("Sample", tok.string))
            toks.append(tok)
        out = tokenize.untokenize(toks)
        ast.parse(out)
    except (SyntaxError, tokenize.TokenError, ValueError):
        return None
    return out if out.endswith("\n") else out + "\n"


def sink_ok(cat: str, bad: str, good: str) -> bool:
    if cat == "sql_injection":
        return sql_built_from_strings(bad) and not sql_built_from_strings(good)
    sink = SINKS[cat]
    return bool(sink.search(bad)) and not sink.search(good)


def strip_fence(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip("\n") + "\n"


def category(vuln_text: str) -> str | None:
    low = vuln_text.lower()
    for keys, name in CATEGORIES:
        if any(k in low for k in keys):
            return name
    return None


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def make_case(code, verdict, family, idx, vuln_text, cat):
    side = "rejected" if verdict == VULNERABLE else "chosen"
    return Case(
        case_id=f"rc_{hashlib.sha256(f'{family}:{side}'.encode()).hexdigest()[:12]}",
        family_id=family, source_id=f"{family}_{side}", split="realcode",
        language="python", target_category=cat, code=code,
        threat_model=(f"Python. Any value that reaches this code from a user, request, file or network "
                      f"is untrusted. Assess this weakness only: {vuln_text.strip()}"),
        oracle_verdict=verdict, source_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        oracle_test_ids=(), label_status="dataset_asserted_sink_checked", reviewer_id=None,
        license="apache-2.0 (CyberNative/Code_Vulnerability_Security_DPO)",
        provenance_url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}#row={idx}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=100, help="upper bound; fewer survive the filters")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "realcode-sinkchecked.json")
    args = parser.parse_args(argv)

    from datasets import load_dataset
    rows = load_dataset(REPO, split="train", revision=REVISION)
    kept, reasons = [], Counter()
    for idx, r in enumerate(rows):
        if r["lang"].strip().lower() != "python":
            continue
        cat = category(r["vulnerability"])
        if cat is None:
            reasons["no_crisp_category"] += 1
            continue
        bad, good = strip_fence(r["rejected"]), strip_fence(r["chosen"])
        if not (parses(bad) and parses(good)):
            reasons["no_parse"] += 1
            continue
        bad, good = scrub(bad), scrub(good)
        if bad is None or good is None:
            reasons["scrub_failed"] += 1
            continue
        if not sink_ok(cat, bad, good):
            reasons["sink_check_failed"] += 1
            continue
        if bad == good:
            reasons["identical"] += 1
            continue
        lb, lg = len(bad.splitlines()), len(good.splitlines())
        if not (3 <= lb <= 60 and 3 <= lg <= 60):
            reasons["length_bounds"] += 1
            continue
        if not 0.5 <= len(good) / len(bad) <= 2.0:
            reasons["length_ratio"] += 1
            continue
        if LABEL_WORDS.search(bad) or LABEL_WORDS.search(good):
            reasons["label_word_survived"] += 1
            continue
        kept.append((idx, r["vulnerability"], bad, good))
    print(f"[realcode] python pairs passing filters: {len(kept)}; dropped {dict(reasons)}", flush=True)

    by_cat = defaultdict(list)
    for item in kept:
        by_cat[category(item[1])].append(item)
    rng = random.Random(args.seed)
    for items in by_cat.values():
        rng.shuffle(items)
    chosen, cats = [], sorted(by_cat)
    while len(chosen) < args.pairs and any(by_cat.values()):   # round-robin = stratified
        for c in cats:
            if by_cat[c] and len(chosen) < args.pairs:
                chosen.append(by_cat[c].pop())

    cases = []
    for n, (idx, vuln_text, bad, good) in enumerate(sorted(chosen)):
        family, cat = f"realcode_{idx}", category(vuln_text)
        cases.append(make_case(bad, VULNERABLE, family, idx, vuln_text, cat))
        cases.append(make_case(good, CLEAN, family, idx, vuln_text, cat))

    info = write_dataset(args.output, cases)
    # the length leak, measured: predict "clean" for the longer side of each pair
    longer_is_clean = sum(len(g) > len(b) for _, _, b, g in chosen)
    extra = {"source": REPO, "revision": REVISION, "pairs": len(chosen),
             "categories": dict(Counter(category(v) for _, v, _, _ in chosen)),
             "longer_is_clean_pairs": longer_is_clean,
             "longer_is_clean_rate": round(longer_is_clean / max(1, len(chosen)), 3),
             "label_status": "dataset_asserted_sink_checked (no executable oracle)",
             "filters_dropped": dict(reasons)}
    data = json.loads(args.output.read_text())
    data["metadata"].update(extra)
    args.output.write_text(json.dumps(data, indent=1) + "\n")
    print(f"[realcode] wrote {info['cases']} cases / {len(chosen)} families -> {args.output}", flush=True)
    print(f"[realcode] {json.dumps(extra)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
