"""Second prospective set, authored after the 72.5% first confirmation."""
from functools import lru_cache
from .confirmation_data import confirmation_cases
from .data import Case, CLEAN, VULNERABLE, validate_cases
from .artifacts import digest
import ast
import hashlib

SQL = (
    ("executemany_read", 'stmt = "SELECT name FROM users WHERE name = \'%s\'" % username\nreturn [row for row in conn.execute(stmt)]', 'stmt = "SELECT name FROM users WHERE name = $value"\nreturn [row for row in conn.execute(stmt, {"value": username})]'),
    ("template_path", 'parts = ["SELECT name FROM users WHERE name = \'", username, "\'"]\nstmt = "".join(parts)\nreturn list(conn.execute(stmt))', 'parts = ["SELECT name FROM users WHERE name = ", ":value"]\nstmt = "".join(parts)\nreturn list(conn.execute(stmt, {"value": username}))'),
    ("nested_cursor", 'def rows():\n    cur = conn.cursor()\n    cur.execute("SELECT name FROM users WHERE name = \'" + username + "\'")\n    return cur.fetchall()\nreturn rows()', 'def rows():\n    cur = conn.cursor()\n    cur.execute("SELECT name FROM users WHERE name = ?", [username])\n    return cur.fetchall()\nreturn rows()'),
    ("trim_query", 'sql = "SELECT name FROM users WHERE name = \'VALUE\'".replace("VALUE", username)\nreturn conn.execute(sql.strip()).fetchall()', 'sql = " SELECT name FROM users WHERE name = ? "\nreturn conn.execute(sql.strip(), (username,)).fetchall()'),
    ("bracket_escape", 'cleaned = username.replace(";", "")\nstmt = "SELECT name FROM users WHERE name = \'{}\'".format(cleaned)\nreturn conn.execute(stmt).fetchall()', 'cleaned = username.replace("\'", "\'\'")\nstmt = "SELECT name FROM users WHERE name = \'{}\'".format(cleaned)\nreturn conn.execute(stmt).fetchall()'),
    ("union_constant", 'stmt = f"SELECT name FROM users WHERE name = \'{username}\' UNION SELECT name FROM users WHERE 0"\nreturn conn.execute(stmt).fetchall()', 'stmt = "SELECT name FROM users WHERE name = ? UNION SELECT name FROM users WHERE 0"\nreturn conn.execute(stmt, (username,)).fetchall()'),
    ("dict_query", 'request = {"name": username}\nstmt = "SELECT name FROM users WHERE name = \'%(name)s\'" % request\nreturn conn.execute(stmt).fetchall()', 'request = {"name": username}\nstmt = "SELECT name FROM users WHERE name = @name"\nreturn conn.execute(stmt, request).fetchall()'),
    ("escape_helper", 'def quote(value):\n    return value.replace("\\\\", "\\\\\\\\")\nreturn conn.execute("SELECT name FROM users WHERE name = \'" + quote(username) + "\'").fetchall()', 'def quote(value):\n    return value.replace("\'", "\'\'")\nreturn conn.execute("SELECT name FROM users WHERE name = \'" + quote(username) + "\'").fetchall()'),
    ("selector", 'queries = {"lookup": f"SELECT name FROM users WHERE name = \'{username}\'"}\nreturn conn.execute(queries["lookup"]).fetchall()', 'queries = {"lookup": "SELECT name FROM users WHERE name = ?"}\nreturn conn.execute(queries["lookup"], [username]).fetchall()'),
    ("bound_other", 'sql = f"SELECT name FROM users WHERE name = \'{username}\' AND length(name) > :minimum"\nreturn conn.execute(sql, {"minimum": 0}).fetchall()', 'sql = "SELECT name FROM users WHERE name = :name AND length(name) > :minimum"\nreturn conn.execute(sql, {"name": username, "minimum": 0}).fetchall()'),
)
ACCESS = (
    ("all_guards", 'row = records.get(record_id)\nif row is None:\n    return None\nchecks = [actor["authenticated"], row["owner_id"] == actor["id"]]\nreturn row if any(checks) else None', 'row = records.get(record_id)\nif row is None:\n    return None\nchecks = [actor["authenticated"], row["owner_id"] == actor["id"]]\nreturn row if all(checks) else None'),
    ("negative_list", 'row = records.get(record_id)\nif row and actor["role"] != "banned":\n    return row\nreturn None', 'row = records.get(record_id)\nif row and actor["role"] != "banned" and row["owner_id"] == actor["id"]:\n    return row\nreturn None'),
    ("branch_default", 'row = records.get(record_id)\nif row is None:\n    return None\nif row["owner_id"] != actor["id"]:\n    allowed = actor["authenticated"]\nelse:\n    allowed = True\nreturn row if allowed else None', 'row = records.get(record_id)\nif row is None:\n    return None\nif row["owner_id"] != actor["id"]:\n    allowed = False\nelse:\n    allowed = True\nreturn row if allowed else None'),
    ("owner_set", 'keys = {key for key, row in records.items() if row["owner_id"] == actor["id"]}\nif keys:\n    return records.get(record_id)\nreturn None', 'keys = {key for key, row in records.items() if row["owner_id"] == actor["id"]}\nif record_id in keys:\n    return records[record_id]\nreturn None'),
    ("boolean_policy", 'row = records.get(record_id)\nreturn row and (row if row["tenant_id"] == actor["tenant_id"] else None)', 'row = records.get(record_id)\nreturn row and (row if row["owner_id"] == actor["id"] else None)'),
    ("loop_break", 'for key, row in records.items():\n    if key == record_id:\n        return row\nreturn None', 'for key, row in records.items():\n    if key == record_id and row["owner_id"] == actor["id"]:\n        return row\nreturn None'),
    ("guard_result", 'def check(row):\n    return row["owner_id"] == actor["id"]\nrow = records.get(record_id)\nif row is None:\n    return None\ncheck(row)\nreturn row', 'def check(row):\n    return row["owner_id"] == actor["id"]\nrow = records.get(record_id)\nif row is None:\n    return None\nreturn row if check(row) else None'),
    ("deny_exception", 'row = records.get(record_id)\nif row is None:\n    return None\ntry:\n    if row["owner_id"] != actor["id"]:\n        raise ValueError("denied")\nexcept ValueError:\n    pass\nreturn row', 'row = records.get(record_id)\nif row is None:\n    return None\ntry:\n    if row["owner_id"] != actor["id"]:\n        raise ValueError("denied")\nexcept ValueError:\n    return None\nreturn row'),
    ("table_policy", 'row = records.get(record_id)\nif row is None:\n    return None\nallow = {True: row, False: row}\nreturn allow[row["owner_id"] == actor["id"]]', 'row = records.get(record_id)\nif row is None:\n    return None\nallow = {True: row, False: None}\nreturn allow[row["owner_id"] == actor["id"]]'),
    ("compound_read", 'row = records.get(record_id)\nif row and (row["owner_id"] == actor["id"] or row["tenant_id"] == actor["tenant_id"]):\n    return row\nreturn None', 'row = records.get(record_id)\nif row and (row["owner_id"] == actor["id"] and row["tenant_id"] == actor["tenant_id"]):\n    return row\nreturn None'),
)


@lru_cache(maxsize=1)
def confirmation_cases_v2():
    from .data import build_cases
    result = []
    for category, pairs, signature in (("sql_injection", SQL, "lookup(conn, username)"),
                                       ("access_control", ACCESS, "read_record(records, actor, record_id)")):
        threat = ("Python sqlite3. The schema and connection are trusted. username is untrusted and must be matched as a literal value without selecting other users."
                  if category == "sql_injection" else "records and the authenticated actor session are trusted. record_id is attacker-controlled. Only the owner may read a record. Login and tenant membership alone do not permit reading another owner's record. Missing records return None.")
        for family, bad, good in pairs:
            family = f"confirmation_v2_{category}_{family}"
            for label, body in ((VULNERABLE, bad), (CLEAN, good)):
                code = f"def {signature}:\n" + "".join(f"    {line}\n" for line in body.splitlines())
                result.append(Case("c_" + digest([family, label])[:12], family, family, "neutral_confirmation_v2", "python", category,
                                   code, threat, label, hashlib.sha256(code.encode()).hexdigest(),
                                   ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    validate_cases(result)
    exposed = (*build_cases(), *confirmation_cases())
    syntax = lambda case: ast.dump(ast.parse(case.code), include_attributes=False)
    if {syntax(c) for c in exposed} & {syntax(c) for c in result}:
        raise ValueError("Confirmation v2 leaks syntax from previously exposed cases")
    return tuple(result)
