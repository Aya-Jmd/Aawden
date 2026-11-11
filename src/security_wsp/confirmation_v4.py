"""Prospective cases for the frozen adapter plus selected review instructions."""
import ast
import hashlib
from functools import lru_cache
from .artifacts import digest
from .data import Case, CLEAN, VULNERABLE, build_cases, validate_cases
from .confirmation_data import confirmation_cases
from .confirmation_v2 import confirmation_cases_v2
from .confirmation_v3 import confirmation_cases_v3

PAIRS = {
    "sql_injection": (
        ("partial_executor", 'from functools import partial\nrun = partial(conn.execute, f"SELECT name FROM users WHERE name = \'{username}\'")\nreturn run().fetchall()', 'from functools import partial\nrun = partial(conn.execute, "SELECT name FROM users WHERE name = ?")\nreturn run((username,)).fetchall()'),
        ("duplicate_argument", 'sql = f"SELECT name FROM users WHERE name = \'{username}\' OR name = \'{username}\'"\nreturn list(conn.execute(sql))', 'sql = "SELECT name FROM users WHERE name = ? OR name = ?"\nreturn list(conn.execute(sql, (username, username)))'),
        ("trusted_table", 'table = "users"\nstmt = "SELECT name FROM {table} WHERE name = \'{value}\'".format(table=table, value=username)\nreturn conn.execute(stmt).fetchall()', 'table = "users"\nstmt = "SELECT name FROM {table} WHERE name = :value".format(table=table)\nreturn conn.execute(stmt, {"value": username}).fetchall()'),
        ("command_object", 'command = {"sql": "SELECT name FROM users WHERE name = \'" + username + "\'", "values": ()}\nreturn conn.execute(command["sql"], command["values"]).fetchall()', 'command = {"sql": "SELECT name FROM users WHERE name = :value", "values": {"value": username}}\nreturn conn.execute(command["sql"], command["values"]).fetchall()'),
        ("token_builder", 'tokens = ["SELECT name", "FROM users", "WHERE name =", "\'" + username + "\'"]\nreturn conn.execute(" ".join(tokens)).fetchall()', 'tokens = ["SELECT name", "FROM users", "WHERE name =", "?"]\nreturn conn.execute(" ".join(tokens), [username]).fetchall()'),
    ),
    "access_control": (
        ("lambda_policy", 'row = records.get(record_id)\nallowed = lambda owner: actor["authenticated"]\nreturn row if row is not None and allowed(row["owner_id"]) else None', 'row = records.get(record_id)\nallowed = lambda owner: owner == actor["id"]\nreturn row if row is not None and allowed(row["owner_id"]) else None'),
        ("mapping_match", 'row = records.get(record_id)\nmatch row:\n    case {"owner_id": owner} if actor["authenticated"]:\n        return row\n    case _:\n        return None', 'row = records.get(record_id)\nmatch row:\n    case {"owner_id": owner} if owner == actor["id"]:\n        return row\n    case _:\n        return None'),
        ("key_owner_pairs", 'ownership = {(key, row["owner_id"]) for key, row in records.items()}\nif any(owner == actor["id"] for key, owner in ownership):\n    return records.get(record_id)\nreturn None', 'ownership = {(key, row["owner_id"]) for key, row in records.items()}\nif (record_id, actor["id"]) in ownership:\n    return records.get(record_id)\nreturn None'),
        ("identity_pair", 'row = records.get(record_id)\nif row is None:\n    return None\nidentity = (row["owner_id"], actor["id"])\nreturn row if identity[0] == identity[1] or actor["authenticated"] else None', 'row = records.get(record_id)\nif row is None:\n    return None\nidentity = (row["owner_id"], actor["id"])\nreturn row if identity[0] == identity[1] else None'),
        ("filtered_result", 'def restrict(row):\n    return row if row["owner_id"] == actor["id"] else None\nrow = records.get(record_id)\nif row is not None:\n    restrict(row)\nreturn row', 'def restrict(row):\n    return row if row["owner_id"] == actor["id"] else None\nrow = records.get(record_id)\nif row is not None:\n    return restrict(row)\nreturn None'),
    ),
}


@lru_cache(maxsize=1)
def confirmation_cases_v4():
    cases = []
    for category, pairs in PAIRS.items():
        signature = "lookup(conn, username)" if category == "sql_injection" else "read_record(records, actor, record_id)"
        threat = ("Python sqlite3. The schema and connection are trusted. username is untrusted and must be matched as a literal value without selecting other users."
                  if category == "sql_injection" else "records and the authenticated actor session are trusted. record_id is attacker-controlled. Only the owner may read a record. Login and tenant membership alone do not permit reading another owner's record. Missing records return None.")
        for name, bad, good in pairs:
            family = f"confirmation_v4_{category}_{name}"
            for label, body in ((VULNERABLE, bad), (CLEAN, good)):
                code = f"def {signature}:\n" + "".join(f"    {line}\n" for line in body.splitlines())
                cases.append(Case("c_" + digest([family, label])[:12], family, family, "neutral_confirmation_v4", "python", category,
                                  code, threat, label, hashlib.sha256(code.encode()).hexdigest(),
                                  ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    validate_cases(cases)
    exposed = (*build_cases(), *confirmation_cases(), *confirmation_cases_v2(), *confirmation_cases_v3())
    syntax = lambda case: ast.dump(ast.parse(case.code), include_attributes=False)
    if {syntax(case) for case in exposed} & {syntax(case) for case in cases}:
        raise ValueError("Confirmation v4 leaks syntax from previously exposed cases")
    return tuple(cases)
