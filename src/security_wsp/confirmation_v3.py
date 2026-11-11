"""Third prospective, balanced feasibility set. Twenty cases, ten families."""
from functools import lru_cache
import ast
import hashlib
from .artifacts import digest
from .data import Case, CLEAN, VULNERABLE, build_cases, validate_cases
from .confirmation_data import confirmation_cases
from .confirmation_v2 import confirmation_cases_v2

PAIRS = {
    "sql_injection": (
        ("indirect_renderer", 'def prepare(value):\n    return "SELECT name FROM users WHERE name = \'" + value + "\'", ()\nquery, args = prepare(username)\nreturn conn.execute(query, args).fetchall()',
         'def prepare(value):\n    return "SELECT name FROM users WHERE name = ?", (value,)\nquery, args = prepare(username)\nreturn conn.execute(query, args).fetchall()'),
        ("tuple_slot", 'stmt = "SELECT name FROM users WHERE name = \'%s\'"\nargs = (username,)\nreturn conn.execute(stmt % args).fetchall()',
         'stmt = "SELECT name FROM users WHERE name = ?"\nargs = (username,)\nreturn conn.execute(stmt, args).fetchall()'),
        ("identity_branch", 'if username:\n    stmt = f"SELECT name FROM users WHERE name = \'{username}\'"\nelse:\n    stmt = "SELECT name FROM users WHERE name = \'\'"\nreturn conn.execute(stmt).fetchall()',
         'if username:\n    args = (username,)\nelse:\n    args = ("",)\nreturn conn.execute("SELECT name FROM users WHERE name = ?", args).fetchall()'),
        ("identity_expression", 'expr = f"name = \'{username}\'"\nstmt = "SELECT name FROM users WHERE (" + expr + ") AND 1"\nreturn conn.execute(stmt).fetchall()',
         'expr = "name = :lookup"\nstmt = "SELECT name FROM users WHERE (" + expr + ") AND 1"\nreturn conn.execute(stmt, {"lookup": username}).fetchall()'),
        ("separate_execution", 'query = "SELECT name FROM users WHERE name = \'{}\'"\nexecute = conn.execute\nreturn execute(query.format(username)).fetchall()',
         'query = "SELECT name FROM users WHERE name = ?"\nexecute = conn.execute\nreturn execute(query, [username]).fetchall()'),
    ),
    "access_control": (
        ("inverted_denial", 'row = records.get(record_id)\nif row is None:\n    return None\nunauthorized = row["owner_id"] != actor["id"]\nif unauthorized and not actor["authenticated"]:\n    return None\nreturn row',
         'row = records.get(record_id)\nif row is None:\n    return None\nunauthorized = row["owner_id"] != actor["id"]\nif unauthorized:\n    return None\nreturn row'),
        ("two_policy_steps", 'row = records.get(record_id)\nif row is None:\n    return None\nchecks = {"owner": row["owner_id"] == actor["id"], "session": actor["authenticated"]}\nif checks["session"]:\n    return row\nreturn None',
         'row = records.get(record_id)\nif row is None:\n    return None\nchecks = {"owner": row["owner_id"] == actor["id"], "session": actor["authenticated"]}\nif checks["session"] and checks["owner"]:\n    return row\nreturn None'),
        ("cached_owner_ids", 'owned_ids = [key for key in records if records[key]["owner_id"] == actor["id"]]\nif len(owned_ids) > 0:\n    return records.get(record_id)\nreturn None',
         'owned_ids = [key for key in records if records[key]["owner_id"] == actor["id"]]\nif record_id in owned_ids:\n    return records.get(record_id)\nreturn None'),
        ("post_check_copy", 'row = records.get(record_id)\nif row is None:\n    return None\nif row["owner_id"] == actor["id"]:\n    return dict(row)\nreturn row',
         'row = records.get(record_id)\nif row is None:\n    return None\nif row["owner_id"] == actor["id"]:\n    return dict(row)\nreturn None'),
        ("guard_closure", 'def denied(row):\n    return row["owner_id"] != actor["id"]\nrow = records.get(record_id)\nif row is None:\n    return None\nreturn row if not denied(row) or actor["authenticated"] else None',
         'def denied(row):\n    return row["owner_id"] != actor["id"]\nrow = records.get(record_id)\nif row is None:\n    return None\nreturn row if not denied(row) else None'),
    ),
}


@lru_cache(maxsize=1)
def confirmation_cases_v3():
    cases = []
    for category, pairs in PAIRS.items():
        signature = "lookup(conn, username)" if category == "sql_injection" else "read_record(records, actor, record_id)"
        threat = ("Python sqlite3. The schema and connection are trusted. username is untrusted and must be matched as a literal value without selecting other users."
                  if category == "sql_injection" else "records and the authenticated actor session are trusted. record_id is attacker-controlled. Only the owner may read a record. Login and tenant membership alone do not permit reading another owner's record. Missing records return None.")
        for name, bad, good in pairs:
            family = f"confirmation_v3_{category}_{name}"
            for label, body in ((VULNERABLE, bad), (CLEAN, good)):
                code = f"def {signature}:\n" + "".join(f"    {line}\n" for line in body.splitlines())
                cases.append(Case("c_" + digest([family, label])[:12], family, family, "neutral_confirmation_v3", "python", category,
                                  code, threat, label, hashlib.sha256(code.encode()).hexdigest(),
                                  ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    validate_cases(cases)
    exposed = (*build_cases(), *confirmation_cases(), *confirmation_cases_v2())
    syntax = lambda case: ast.dump(ast.parse(case.code), include_attributes=False)
    if {syntax(case) for case in exposed} & {syntax(case) for case in cases}:
        raise ValueError("Confirmation v3 leaks syntax from previously exposed cases")
    return tuple(cases)
