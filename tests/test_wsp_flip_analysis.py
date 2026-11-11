import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_flip", ROOT / "scripts" / "wsp_flip_analysis.py")
flip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flip)

V, C = flip.VULNERABLE, flip.CLEAN
MAP = {V: V, C: C}


def block(entries):
    """entries: case_id -> (oracle, preferred verdict or None for degenerate)."""
    cases = {}
    for cid, (truth, pick) in entries.items():
        if pick is None:
            cases[cid] = {"oracle_verdict": truth, "scores": {V: 0.0, C: 0.0}, "correct_margin": None}
        else:
            other = C if pick == V else V
            cases[cid] = {"oracle_verdict": truth, "scores": {pick: -0.1, other: -5.0}, "correct_margin": 1.0}
    return {"json_prefill": {"verdict_by_continuation": MAP, "cases": cases}}


def table(neutral, wrong, right):
    return {"conditions": {"neutral": block(neutral), "incorrect_suggestion": block(wrong),
                           "correct_suggestion": block(right)}}


def test_flip_counts_only_solved_cases():
    neutral = {"a": (V, V), "b": (C, C), "c": (C, V), "d": (V, C)}
    wrong = {"a": (V, C), "b": (C, C), "c": (C, V), "d": (V, C)}   # a flips, b holds
    right = {"a": (V, V), "b": (C, C), "c": (C, C), "d": (V, C)}   # c rescued, d not
    r = flip.analyse(table(neutral, wrong, right))["all"]
    assert r["solved_unaided"] == 2
    assert r["flipped_by_wrong_reviewer"] == 1
    assert r["flip_rate"] == 0.5
    assert r["unsolved_unaided"] == 2 and r["rescued_by_correct_reviewer"] == 1


def test_constant_predictor_has_no_competence_on_safe_files():
    # always says vulnerable: solves every vulnerable file and no safe file
    neutral = {f"v{i}": (V, V) for i in range(5)} | {f"c{i}": (C, V) for i in range(5)}
    wrong = {f"v{i}": (V, V) for i in range(5)} | {f"c{i}": (C, V) for i in range(5)}
    r = flip.analyse(table(neutral, wrong, neutral))
    assert r["safe_files"]["solved_unaided"] == 0
    assert r["safe_files"]["flip_rate"] is None
    assert r["vulnerable_files"]["flip_rate"] == 0.0


def test_degenerate_neutral_is_not_counted_as_solved():
    neutral = {"a": (V, None)}
    r = flip.analyse(table(neutral, {"a": (V, V)}, {"a": (V, V)}))["all"]
    assert r["solved_unaided"] == 0


def test_wilson_interval_bounds():
    lo, hi = flip.wilson(0, 30)
    assert lo == 0.0 and 0.10 < hi < 0.12
    assert flip.wilson(0, 0) is None
