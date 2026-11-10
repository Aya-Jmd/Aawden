"""Project-original, executable SQLite and authorization benchmark."""
from __future__ import annotations

import ast
import hashlib
import random
import sqlite3
from functools import lru_cache
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .artifacts import atomic_json, digest, file_hash, read_json

VULNERABLE = "vulnerable"
CLEAN = "not_vulnerable_for_target"
ABSTAIN = "insufficient_information"
VERDICTS = (VULNERABLE, CLEAN, ABSTAIN)


@dataclass(frozen=True)
class Case:
    case_id: str
    family_id: str
    source_id: str
    split: str
    language: str
    target_category: str
    code: str
    threat_model: str
    oracle_verdict: str
    source_sha256: str
    oracle_test_ids: tuple[str, ...]
    label_status: str = "executable_oracle_single_author"
    reviewer_id: str | None = None
    license: str = "project-original-MIT"
    provenance_url: str | None = None

    def visible(self) -> dict:
        return {key: getattr(self, key) for key in ("code", "language", "target_category", "threat_model")}

    @classmethod
    def from_dict(cls, item: dict):
        item = dict(item)
        item["oracle_test_ids"] = tuple(item["oracle_test_ids"])
        return cls(**item)


# Each tuple defines a vulnerable/fixed pair in one implementation family.
# These are original fixtures, deliberately small enough to execute and inspect.
SQL_PAIRS = [
    ("fstring", 'return conn.execute(f"SELECT name FROM users WHERE name = \'{username}\'").fetchall()',
     'return conn.execute("SELECT name FROM users WHERE name = ?", (username,)).fetchall()'),
    ("concatenation", 'query = "SELECT name FROM users WHERE name = \'" + username + "\'"\nreturn conn.execute(query).fetchall()',
     'query = "SELECT name FROM users WHERE name = ?"\nreturn conn.execute(query, (username,)).fetchall()'),
    ("percent", 'query = "SELECT name FROM users WHERE name = \'%s\'" % username\nreturn conn.execute(query).fetchall()',
     'query = "SELECT name FROM users WHERE name = :name"\nreturn conn.execute(query, {"name": username}).fetchall()'),
    ("format", 'query = "SELECT name FROM users WHERE name = \'{}\'".format(username)\nreturn conn.execute(query).fetchall()',
     'query = "SELECT name FROM users WHERE name = :value"\nreturn conn.execute(query, {"value": username}).fetchall()'),
    ("cursor", 'cursor = conn.cursor()\ncursor.execute(f"SELECT name FROM users WHERE name = \'{username}\'")\nreturn cursor.fetchall()',
     'cursor = conn.cursor()\ncursor.execute("SELECT name FROM users WHERE name = ?", (username,))\nreturn cursor.fetchall()'),
    ("cte", 'query = f"WITH matches AS (SELECT name FROM users WHERE name = \'{username}\') SELECT name FROM matches"\nreturn conn.execute(query).fetchall()',
     'query = "WITH matches AS (SELECT name FROM users WHERE name = ?) SELECT name FROM matches"\nreturn conn.execute(query, (username,)).fetchall()'),
    ("lowercase", 'return conn.execute(f"SELECT name FROM users WHERE lower(name) = lower(\'{username}\')").fetchall()',
     'return conn.execute("SELECT name FROM users WHERE lower(name) = lower(?)", (username,)).fetchall()'),
    ("strip", 'name = username.strip()\nreturn conn.execute(f"SELECT name FROM users WHERE name = \'{name}\'").fetchall()',
     'name = username.strip()\nreturn conn.execute("SELECT name FROM users WHERE name = ?", (name,)).fetchall()'),
    ("helper", 'def execute(query):\n    return conn.execute(query).fetchall()\nreturn execute(f"SELECT name FROM users WHERE name = \'{username}\'")',
     'def execute(query, value):\n    return conn.execute(query, (value,)).fetchall()\nreturn execute("SELECT name FROM users WHERE name = ?", username)'),
    ("manual_quote", 'quoted = username\nreturn conn.execute(f"SELECT name FROM users WHERE name = \'{quoted}\'").fetchall()',
     'quoted = username.replace("\'", "\'\'")\nreturn conn.execute(f"SELECT name FROM users WHERE name = \'{quoted}\'").fetchall()'),
    ("join", 'query = "".join(["SELECT name FROM users WHERE name = \'", username, "\'"])\nreturn conn.execute(query).fetchall()',
     'query = "".join(["SELECT name FROM users ", "WHERE name = ?"])\nreturn conn.execute(query, (username,)).fetchall()'),
    ("order", 'query = f"SELECT name FROM users WHERE name = \'{username}\' ORDER BY name"\nreturn conn.execute(query).fetchall()',
     'query = "SELECT name FROM users WHERE name = ? ORDER BY name"\nreturn conn.execute(query, (username,)).fetchall()'),
]

ACCESS_PAIRS = [
    ("direct", 'return records.get(record_id)',
     'record = records.get(record_id)\nreturn record if record and record["owner_id"] == actor["id"] else None'),
    ("login_only", 'record = records.get(record_id)\nreturn record if actor.get("authenticated") else None',
     'record = records.get(record_id)\nreturn record if record and actor.get("authenticated") and record["owner_id"] == actor["id"] else None'),
    ("exists", 'if record_id not in records:\n    return None\nreturn records[record_id]',
     'if record_id not in records:\n    return None\nrecord = records[record_id]\nif record["owner_id"] != actor["id"]:\n    return None\nreturn record'),
    ("role_only", 'record = records.get(record_id)\nreturn record if actor["role"] in ("member", "admin") else None',
     'record = records.get(record_id)\nreturn record if record and (record["owner_id"] == actor["id"] or actor["role"] == "admin") else None'),
    ("tenant_only", 'record = records.get(record_id)\nreturn record if record and record["tenant_id"] == actor["tenant_id"] else None',
     'record = records.get(record_id)\nreturn record if record and record["tenant_id"] == actor["tenant_id"] and record["owner_id"] == actor["id"] else None'),
    ("predicate", 'def allowed(record):\n    return record is not None\nrecord = records.get(record_id)\nreturn record if allowed(record) else None',
     'def allowed(record):\n    return record is not None and record["owner_id"] == actor["id"]\nrecord = records.get(record_id)\nreturn record if allowed(record) else None'),
    ("filter", 'matches = [r for key, r in records.items() if key == record_id]\nreturn matches[0] if matches else None',
     'matches = [r for key, r in records.items() if key == record_id and r["owner_id"] == actor["id"]]\nreturn matches[0] if matches else None'),
    ("membership", 'record = records.get(record_id)\nreturn record if record and actor["tenant_id"] == record["tenant_id"] else None',
     'record = records.get(record_id)\nreturn record if record and actor["id"] in record["members"] else None'),
    ("deny_missing", 'record = records.get(record_id)\nif record is None or not actor["authenticated"]:\n    return None\nreturn record',
     'record = records.get(record_id)\nif record is None or not actor["authenticated"] or record["owner_id"] != actor["id"]:\n    return None\nreturn record'),
    ("tuple", 'record = records.get(record_id)\nreturn record if record and record["tenant_id"] == actor["tenant_id"] else None',
     'record = records.get(record_id)\nreturn record if record and (record["tenant_id"], record["owner_id"]) == (actor["tenant_id"], actor["id"]) else None'),
    ("policy", 'policy = {"member": True, "admin": True}\nreturn records.get(record_id) if policy.get(actor["role"], False) else None',
     'record = records.get(record_id)\npolicy = {"member": record is not None and record["owner_id"] == actor["id"], "admin": True}\nreturn record if policy.get(actor["role"], False) else None'),
    ("scoped_map", 'visible = dict(records)\nreturn visible.get(record_id)',
     'visible = {key: record for key, record in records.items() if record["owner_id"] == actor["id"]}\nreturn visible.get(record_id)'),
    ("guard_return", 'record = records.get(record_id)\nif actor["role"] == "banned":\n    return None\nreturn record',
     'record = records.get(record_id)\nif actor["role"] == "banned" or record is None:\n    return None\nif record["owner_id"] == actor["id"]:\n    return record\nreturn None'),
]


@lru_cache(maxsize=1)
def build_cases() -> list[Case]:
    cases = []
    for category, pairs, signature in [
        ("sql_injection", SQL_PAIRS, "lookup(conn, username)"),
        ("access_control", ACCESS_PAIRS, "read_record(records, actor, record_id)"),
    ]:
        for family, bad, good in pairs:
            family_id = category + "_" + family
            policy = "Only the owner may read a record."
            if family in {"role_only", "policy"}:
                policy = "An owner or an administrator may read a record."
            if family == "membership":
                policy = "Only an explicitly listed record member may read it."
            threat = ("Python sqlite3; username is untrusted. A lookup must treat the supplied name as a literal value; "
                      "it must not return other users. The database schema and connection are trusted.") if category == "sql_injection" else (
                      "records is a trusted server-side dictionary; record_id is attacker-controlled; "
                      "actor comes from a trusted authenticated session. " + policy +
                      " Sharing a tenant or being logged in alone does not authorize a read. Missing records must return None.")
            for index, body in enumerate((bad, good)):
                code = "def " + signature + ":\n" + "\n".join("    " + line for line in body.splitlines()) + "\n"
                case_id = "c_" + digest([family_id, index])[:12]
                cases.append(Case(case_id, family_id, family_id, "screen", "python", category, code, threat,
                                  VULNERABLE if index == 0 else CLEAN, hashlib.sha256(code.encode("utf-8")).hexdigest(),
                                  ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    # Shared exact templates connect their whole vulnerable/fixed families.
    # They must share both a split and a statistical cluster.
    parent = {c.family_id: c.family_id for c in cases}
    def root(family):
        while parent[family] != family:
            family = parent[family]
        return family
    templates = {}
    for case in cases:
        template = ast.dump(ast.parse(case.code), include_attributes=False)
        if template in templates:
            parent[root(case.family_id)] = root(templates[template])
        templates[template] = case.family_id
    return [replace(c, family_id=root(c.family_id)) for c in cases]


def validate_cases(cases: list[Case]) -> dict:
    if not cases:
        raise ValueError("Dataset is empty")
    seen, families, sources, codes = set(), {}, {}, {}
    for case in cases:
        if case.case_id in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case.case_id)
        if case.oracle_verdict not in (VULNERABLE, CLEAN):
            raise ValueError("Main endpoint requires a decisive oracle label")
        if hashlib.sha256(case.code.encode("utf-8")).hexdigest() != case.source_sha256:
            raise ValueError("Source hash mismatch")
        ast.parse(case.code)
        for key, groups in ((case.family_id, families), (case.source_id, sources)):
            if key in groups and groups[key] != case.split:
                raise ValueError("Family/source leakage across splits")
            groups[key] = case.split
        normalized = ast.dump(ast.parse(case.code), include_attributes=False)
        if normalized in codes and codes[normalized] != case.split:
            raise ValueError("Exact syntax duplicate across splits")
        codes[normalized] = case.split
    return {"cases": len(cases), "families": len(families), "splits": sorted({c.split for c in cases}),
            "dataset_sha256": digest([asdict(c) for c in cases]), "label_review": "single_author; no independent review",
            "final_test_protected": False}


def split_by_family(cases: list[Case], seed=42) -> list[Case]:
    """Create exploratory splits; never call an exposed split protected final data."""
    # Union exact duplicate templates so they cannot straddle a split.
    parent = {c.family_id: c.family_id for c in cases}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    code_families = {}
    for case in cases:
        key = ast.dump(ast.parse(case.code), include_attributes=False)
        if key in code_families:
            parent[find(case.family_id)] = find(code_families[key])
        code_families[key] = case.family_id
    groups = sorted({find(c.family_id) for c in cases})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    assignments = {g: "discovery" if i < int(n * .6) else "validation" if i < int(n * .8) else "exploratory_test" for i, g in enumerate(groups)}
    result = [replace(c, split=assignments[find(c.family_id)]) for c in cases]
    validate_cases(result)
    return result


@lru_cache(maxsize=1)
def oracle_cases() -> dict[str, Case]:
    from .confirmation_data import confirmation_cases
    from .confirmation_v2 import confirmation_cases_v2
    from .confirmation_v3 import confirmation_cases_v3
    from .confirmation_v4 import confirmation_cases_v4
    from .confirmation_v5 import confirmation_cases_v5
    from .rigorous_cases import rigorous_cases
    return {c.case_id: c for c in (*build_cases(), *confirmation_cases(), *confirmation_cases_v2(), *confirmation_cases_v3(), *confirmation_cases_v4(), *confirmation_cases_v5(), *rigorous_cases())}


def run_oracle(case: Case) -> dict:
    """Execute only canonical, project-authored fixtures, never uploaded code."""
    expected = oracle_cases().get(case.case_id)
    checked_fields = ("code", "target_category", "source_id", "oracle_verdict")
    if expected is None or any(getattr(case, name) != getattr(expected, name) for name in checked_fields):
        raise ValueError("Oracle execution is restricted to canonical project fixtures")
    scope = {}
    exec(compile(case.code, "<trusted-aawden-fixture>", "exec"), scope)
    violations, tests, normal_ok = [], [], True
    rigorous = expected.split.startswith("rigorous_")
    if case.target_category == "sql_injection":
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE users (name TEXT)")
            conn.executemany("INSERT INTO users VALUES (?)", [("alice",), ("bob",), ("o'brien",)])
            lookup = scope["lookup"]
            for name in ("alice", "bob"):
                actual = lookup(conn, name)
                ok = actual == [(name,)]
                normal_ok &= ok
                tests.append({"test": "legitimate_access", "input": name, "passed": ok})
            payloads = ("never' OR 1=1 --", "never') OR 1=1 --", "never' OR '1'='1")
            if rigorous:
                payloads += ("never' UNION SELECT 'aawden-probe' --",)
            if expected.split == "rigorous_confirmation_v4":
                payloads += ("never' AS TEXT) OR 1=1 --",)
            for payload in payloads:
                try:
                    rows = lookup(conn, payload)
                    violation = bool(rows)
                except sqlite3.Error:
                    violation = False
                violations.append(violation)
                tests.append({"test": "adversarial_access", "input": payload, "violation": violation})
            try:
                literal_ok = lookup(conn, "o'brien") == [("o'brien",)]
            except sqlite3.Error:
                literal_ok = False
            tests.append({"test": "literal_quote_preserved", "passed": literal_ok})
            if case.oracle_verdict == CLEAN:
                normal_ok &= literal_ok
    else:
        records = {1: {"owner_id": "alice", "tenant_id": "t1", "members": ["alice"], "secret": "A"},
                   2: {"owner_id": "bob", "tenant_id": "t1", "members": ["bob"], "secret": "B"}}
        actor = {"id": "alice", "tenant_id": "t1", "role": "member", "authenticated": True}
        read = scope["read_record"]
        normal_ok = read(records, actor, 1) == records[1] and read(records, actor, 99) is None
        tests.append({"test": "legitimate_and_missing_access", "passed": normal_ok})
        for target in (2,):
            violation = read(records, actor, target) is not None
            violations.append(violation)
            tests.append({"test": "cross_owner_same_tenant", "violation": violation})
        stranger = dict(actor, id="mallory", tenant_id="t2")
        violation = read(records, stranger, 1) is not None
        violations.append(violation)
        tests.append({"test": "cross_tenant_access", "violation": violation})
        if rigorous:
            bob = dict(actor, id="bob")
            ok = read(records, bob, 2) == records[2]
            normal_ok &= ok
            tests.append({"test": "second_owner_permitted", "passed": ok})
            for actor_under_test, key in ((bob, 1), (dict(stranger, role="admin"), 2)):
                violation = read(records, actor_under_test, key) is not None
                violations.append(violation)
                tests.append({"test": "non_owner_denied", "violation": violation})
        if case.source_id in ("access_control_role_only", "access_control_policy", "neutral_confirmation_access_control_administrator"):
            admin = dict(actor, id="root", role="admin")
            ok = read(records, admin, 2) == records[2]
            normal_ok &= ok
            tests.append({"test": "administrator_permitted", "passed": ok})
    observed = VULNERABLE if any(violations) else CLEAN
    return {"case_id": case.case_id, "observed_verdict": observed, "expected_verdict": case.oracle_verdict,
            "passed": observed == case.oracle_verdict and normal_ok, "checks": tests,
            "review": "Executable behavioral evidence; authored and reviewed by the same coding agent."}


def write_dataset(path: Path, cases: list[Case]) -> dict:
    info = validate_cases(cases)
    atomic_json(path, {"schema_version": 1, "metadata": info, "cases": [asdict(c) for c in cases]})
    return info


def load_dataset(path: str | Path) -> list[Case]:
    cases = [Case.from_dict(item) for item in read_json(path)["cases"]]
    validate_cases(cases)
    return cases

