"""Larger real-code set in six languages (round 2, R7 of docs/FIX_PROTOCOL.md).

Why this exists
    The round-1 real-code set is 32 Python pairs, and a "longer side is clean" rule scores 0.844 on
    it. This draws from the same public source at the same commit
    (`CyberNative/Code_Vulnerability_Security_DPO`), in PHP, C#, Ruby, Java, JavaScript and Kotlin,
    for the two categories whose dangerous operation can be checked mechanically in any of them.

Label honesty
    Labels are ASSERTED BY THE DATASET and then mechanically sink-checked. There is no executable
    oracle. `label_status="dataset_asserted_sink_checked"`. Secondary evidence.

Filters, fixed before any model sees the set
    - Comments are stripped with a string-aware scanner (`//`, `/* */`, `#` where the language has it).
    - A pair is DROPPED, not rewritten, if either side still holds a label-bearing word
      (secure, safe, vulnerable, sanitize, exploit, malicious, attack, hack). Rewriting identifiers in
      six languages risks changing what the code does; dropping cannot.
    - ASCII only (the source has stray characters in some rows), 3-60 lines, length ratio in [0.5, 2].
    - Sink check.
        sql_injection   vulnerable side builds a SQL string from a variable and binds no parameter;
                        fixed side binds parameters and builds no SQL string from a variable.
        code_injection  vulnerable side calls eval (or a named equivalent); fixed side has none.
    - Untrusted source: the vulnerable side must read a request parameter, console line or program
      argument. The hand check found pairs whose "user input" is a hard-coded constant.
    - Stray text: the hand check (6 random pairs read before any model run) found junk words after a
      statement on the FIXED side of 5 pairs (`echo "Login failed.";ellow`, `println(...)Bs`). That is
      both broken code and a give-away of the label, so a pair is dropped if either side has text
      after a top-level `;` or a bare word glued to a closing bracket or quote at the end of a line.
    - Near-duplicates: this source repeats templates. A pair is dropped if the 5-token shingles of its
      vulnerable side overlap an already-kept pair's by more than 0.6 (Jaccard).
    - At most `--per-cell` pairs per (category, language), so one template family cannot dominate.

Length
    Binding a parameter takes more code than concatenating a string, so the fixed side is the longer
    one in almost every pair that passes the sink check (348 of 367 before the other filters). The
    set therefore cannot be balanced on length without rewriting code. The score of a "longer = clean"
    rule is reported, the families where the fixed side is longer are listed in the metadata, and the
    report gives the flip rate on the remaining families separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.data import CLEAN, VULNERABLE, Case  # noqa: E402

REPO = "CyberNative/Code_Vulnerability_Security_DPO"
REVISION = "403bbc40790391a595acd591326c5b7c5077b291"
LANGS = {"php": "PHP", "c#": "C#", "ruby": "Ruby", "java": "Java", "javascript": "JavaScript", "kotlin": "Kotlin"}
HASH_COMMENT = {"php", "ruby"}
LABEL_WORDS = re.compile(r"(?i)secur|safe|vulnerab|sanitiz|exploit|malicious|attack|hack|inject|\bfix(ed)?\b|\bbad\b|\bgood\b")


def strip_fence(text: str) -> str:
    m = re.search(r"```[\w#+]*[ \t]*\n(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip("\n") + "\n"


def strip_comments(code: str, lang: str) -> str:
    """Remove comments outside string literals. Blank lines left behind are dropped."""
    out, i, n, quote = [], 0, len(code), None
    while i < n:
        ch = code[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if lang != "ruby" and code.startswith("//", i):
            while i < n and code[i] != "\n":
                i += 1
            continue
        if lang != "ruby" and code.startswith("/*", i):
            end = code.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if lang in HASH_COMMENT and ch == "#" and not code.startswith("#{", i):
            while i < n and code[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    lines = [line.rstrip() for line in "".join(out).splitlines()]
    return "\n".join(line for line in lines if line.strip()) + "\n"


_SQL_WORD = re.compile(r"\b(SELECT\b.*\bFROM|INSERT\s+INTO|UPDATE\b.*\bSET|DELETE\s+FROM)\b", re.I)
_SQL_BUILD = re.compile(
    r"""["']\s*[+.]\s*[$@\w(]"""          # "..." + name    "..." . $name
    r"""|[\w)\]]\s*[+.]\s*["']"""         # name + "..."    $name . "..."
    r"""|#\{|\$\{"""                      # Ruby / JS / Kotlin interpolation
    r"""|\$"[^"]*\{"""                    # C# $"...{name}"
    r"""|"[^"]*\$[A-Za-z_][^"]*\""""      # PHP / Kotlin "$name" inside a string
    r"""|String\.format\(|string\.Format\(|sprintf\(|\.format\(""")
_SQL_BIND = re.compile(
    r"""prepare\(|bind_param|bindParam|bindValue|PreparedStatement|prepareStatement|\.setString\(|\.setInt\("""
    r"""|Parameters\.Add|AddWithValue|SqlParameter|exec_params|execute\(\s*(\[|array\()"""
    r"""|["'][^"'\n]*(\?|:[A-Za-z_]\w*|@[A-Za-z_]\w*)[^"'\n]*["']\s*,""")
_EVAL = {
    "php": re.compile(r"(?<![\w>$])(eval|assert|create_function|extract)\s*\("),
    "ruby": re.compile(r"(?<![\w.:])(eval|instance_eval|class_eval|module_eval)\b|\.(instance_eval|class_eval|send|public_send)\b"),
    "javascript": re.compile(r"(?<![\w.$])eval\s*\(|new\s+Function\s*\(|(?<![\w.$])Function\s*\(|vm\.run"),
    "java": re.compile(r"ScriptEngine|\.eval\s*\("),
    "kotlin": re.compile(r"ScriptEngine|\.eval\s*\("),
    "c#": re.compile(r"CSharpScript|CodeDom|\.Evaluate\w*\("),
}
_EVAL_CALL = {
    "php": re.compile(r"(?<![\w>$])eval\s*\("),
    "ruby": re.compile(r"(?<![\w.:])(eval|instance_eval|class_eval|module_eval)\b"),
    "javascript": re.compile(r"(?<![\w.$])eval\s*\("),
    "java": re.compile(r"\.eval\s*\("),
    "kotlin": re.compile(r"\.eval\s*\("),
    "c#": re.compile(r"\.Evaluate\w*\(|CSharpScript"),
}


_SOURCE = re.compile(r"""\$_(GET|POST|REQUEST|COOKIE)|params\[|request\.|req\.(query|body|params)|Request\.|Request\["""
                     r"""|Console\.ReadLine|readLine\(|readline\(|Scanner\(|\bgets\b|ARGV|args\[|getParameter\(|process\.argv|prompt\(""")
_TRAILING_WORD_OK = {"do", "then", "else", "end", "begin", "or", "and", "in", "as", "is", "const", "override"}


def has_stray_text(code: str) -> bool:
    for line in code.splitlines():
        depth, quote, i = 0, None, 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == ";" and depth <= 0:
                rest = line[i + 1:].strip()
                if rest and not re.fullmatch(r"[}\s]*(\?>)?|\?>|end|else\b.*|break;?|\}\s*else\b.*", rest):
                    return True
            i += 1
        m = re.search(r"""[)\]"']\s*([A-Za-z_]+)\s*$""", line)
        if m and m.group(1) not in _TRAILING_WORD_OK:
            return True
    return False


def sql_built(code: str) -> bool:
    return any(_SQL_WORD.search(line) and _SQL_BUILD.search(line) for line in code.splitlines())


def category(text: str) -> str | None:
    low = text.lower()
    if "sql" in low:
        return "sql_injection"
    if "eval" in low or "code injection" in low or "code execution" in low:
        return "code_injection"
    return None


def sink_ok(cat: str, lang: str, bad: str, good: str) -> bool:
    if cat == "sql_injection":
        return (sql_built(bad) and not _SQL_BIND.search(bad)
                and not sql_built(good) and bool(_SQL_BIND.search(good)) and bool(_SQL_WORD.search(good)))
    return bool(_EVAL_CALL[lang].search(bad)) and not _EVAL[lang].search(good)


def shingles(code: str) -> frozenset:
    toks = re.findall(r"\w+|[^\w\s]", code)
    return frozenset(tuple(toks[i:i + 5]) for i in range(max(1, len(toks) - 4)))


def make_case(code, verdict, family, idx, vuln_text, cat, lang):
    side = "rejected" if verdict == VULNERABLE else "chosen"
    return Case(
        case_id=f"rm_{hashlib.sha256(f'{family}:{side}'.encode()).hexdigest()[:12]}",
        family_id=family, source_id=f"{family}_{side}", split="realcode_multilang",
        language=lang, target_category=cat, code=code,
        threat_model=(f"{LANGS[lang]}. Any value that reaches this code from a user, request, file or network "
                      f"is untrusted. Assess this weakness only: {vuln_text.strip()}"),
        oracle_verdict=verdict, source_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        oracle_test_ids=(), label_status="dataset_asserted_sink_checked", reviewer_id=None,
        license="apache-2.0 (CyberNative/Code_Vulnerability_Security_DPO)",
        provenance_url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}#row={idx}")


def load_rows(jsonl: Path | None):
    if jsonl:
        return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    from datasets import load_dataset
    rows = load_dataset(REPO, split="train", revision=REVISION)
    return [{"idx": i, **r} for i, r in enumerate(rows)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", type=Path, default=None, help="pre-exported rows (offline development)")
    parser.add_argument("--per-cell", type=int, default=20, help="max pairs per (category, language)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "realcode-multilang.json")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    rows = load_rows(args.rows_jsonl)
    order = list(range(len(rows)))
    rng.shuffle(order)                      # near-duplicate removal keeps the first seen, so shuffle first
    kept, reasons, seen = [], Counter(), []
    for pos in order:
        r = rows[pos]
        lang = r["lang"].strip().lower()
        if lang not in LANGS:
            continue
        cat = category(r["vulnerability"])
        if cat is None:
            reasons["no_crisp_category"] += 1
            continue
        bad = strip_comments(strip_fence(r["rejected"]), lang)
        good = strip_comments(strip_fence(r["chosen"]), lang)
        if not (bad.isascii() and good.isascii()):
            reasons["non_ascii"] += 1
            continue
        if bad == good:
            reasons["identical"] += 1
            continue
        if LABEL_WORDS.search(bad) or LABEL_WORDS.search(good):
            reasons["label_word_present"] += 1
            continue
        if has_stray_text(bad) or has_stray_text(good):
            reasons["stray_text"] += 1
            continue
        if not _SOURCE.search(bad):
            reasons["no_untrusted_source"] += 1
            continue
        if not sink_ok(cat, lang, bad, good):
            reasons["sink_check_failed"] += 1
            continue
        lb, lg = len(bad.splitlines()), len(good.splitlines())
        if not (3 <= lb <= 60 and 3 <= lg <= 60):
            reasons["length_bounds"] += 1
            continue
        if not 0.5 <= len(good) / len(bad) <= 2.0:
            reasons["length_ratio"] += 1
            continue
        sh = shingles(bad)
        if any(len(sh & other) / len(sh | other) > 0.6 for other in seen):
            reasons["near_duplicate"] += 1
            continue
        seen.append(sh)
        kept.append((r["idx"], r["vulnerability"], bad, good, cat, lang))
    print(f"[multilang] pairs passing filters: {len(kept)}; dropped {dict(reasons)}", flush=True)
    print(f"[multilang] passing by (category, language): "
          f"{dict(Counter((k[4], k[5]) for k in kept))}", flush=True)

    cells = defaultdict(list)                # (category, language) -> pairs, in shuffled order
    for item in kept:
        cells[(item[4], item[5])].append(item)
    chosen = [item for key in sorted(cells) for item in cells[key][:args.per_cell]]
    chosen.sort()

    cases = []
    for idx, vuln_text, bad, good, cat, lang in chosen:
        family = f"realml_{idx}"
        cases.append(make_case(bad, VULNERABLE, family, idx, vuln_text, cat, lang))
        cases.append(make_case(good, CLEAN, family, idx, vuln_text, cat, lang))
    assert len({c.case_id for c in cases}) == len(cases)
    longer_is_clean = sum(len(g) > len(b) for _, _, b, g, _, _ in chosen)
    meta = {"cases": len(cases), "families": len(chosen), "splits": ["realcode_multilang"],
            "source": REPO, "revision": REVISION, "pairs": len(chosen),
            "categories": dict(Counter(c for *_, c, _ in chosen)),
            "languages": dict(Counter(l for *_, l in chosen)),
            "longer_is_clean_pairs": longer_is_clean,
            "longer_is_clean_rate": round(longer_is_clean / max(1, len(chosen)), 3),
            "clean_is_longer_family_ids": sorted(f"realml_{i}" for i, _, b, g, _, _ in chosen if len(g) > len(b)),
            "passing_before_cap": len(kept),
            "label_status": "dataset_asserted_sink_checked (no executable oracle)",
            "label_review": "single_author; no independent review", "final_test_protected": False,
            "python_validation": "not applicable: non-Python sources",
            "filters_dropped": dict(reasons), "seed": args.seed}
    args.output.write_text(json.dumps({"schema_version": 1, "metadata": meta,
                                       "cases": [asdict(c) for c in cases]}, indent=1) + "\n")
    print(f"[multilang] wrote {len(cases)} cases / {len(chosen)} pairs -> {args.output}", flush=True)
    print(f"[multilang] {json.dumps({k: v for k, v in meta.items() if k != 'clean_is_longer_family_ids'})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
