"""Fresh, project-authored fixtures for the Aya neutral competence check."""
from __future__ import annotations
import ast
import hashlib
from functools import lru_cache

from .artifacts import digest
from .data import CLEAN, VULNERABLE, Case, build_cases, validate_cases

SQL = (
    ("mixed_binding", "sql = \"SELECT name FROM users WHERE length(name) >= ? AND name = '\" + username + \"'\"\nreturn conn.execute(sql, (0,)).fetchall()",
     'sql = "SELECT name FROM users WHERE length(name) >= ? AND name = ?"\nreturn conn.execute(sql, (0, username)).fetchall()'),
    ("substitute", 'from string import Template\nquery = Template("SELECT name FROM users WHERE name = \'$value\'").substitute(value=username)\nreturn conn.execute(query).fetchall()',
     'from string import Template\nquery = Template("SELECT name FROM users WHERE name = $slot").substitute(slot="?1")\nreturn conn.execute(query, (username,)).fetchall()'),
    ("list_binding", 'values = [username]\nterms = ",".join("\'" + value + "\'" for value in values)\nreturn conn.execute("SELECT name FROM users WHERE name IN (" + terms + ")").fetchall()',
     'values = [username]\nslots = ",".join("?" for _ in values)\nreturn conn.execute("SELECT name FROM users WHERE name IN (" + slots + ")", values).fetchall()'),
    ("denylist", 'if ";" in username or "--" in username:\n    return []\nreturn list(conn.execute(f"SELECT name FROM users WHERE name = \'{username}\'"))',
     'cursor = conn.execute("SELECT name FROM users WHERE name = ?1", [username])\nreturn list(cursor)'),
    ("early_binding", 'literal, = conn.execute("SELECT ?", (username,)).fetchone()\nreturn conn.execute(f"SELECT name FROM users WHERE name = \'{literal}\'").fetchall()',
     'literal, = conn.execute("SELECT ?", (username,)).fetchone()\nreturn conn.execute("SELECT name FROM users WHERE name = ?", (literal,)).fetchall()'),
    ("mapping_argument", 'params = {"user": username}\nquery = "SELECT name FROM users WHERE name = \'{}\'".format(params["user"])\nreturn conn.execute(query, params).fetchall()',
     'params = {"user": username}\nquery = "SELECT name FROM users WHERE name = :user"\nreturn conn.execute(query, params).fetchall()'),
    ("projection", 'fields = ("name",)\nreturn conn.execute(f"SELECT {fields[0]} FROM users WHERE name = \'{username}\'").fetchall()',
     'fields = ("name",)\nreturn conn.execute(f"SELECT {fields[0]} FROM users WHERE name = ?", (username,)).fetchall()'),
    ("literal_builder", 'def literal(value):\n    return "\'" + value + "\'"\nsql = "SELECT name FROM users WHERE name = " + literal(username)\nreturn conn.execute(sql).fetchall()',
     'def literal(value):\n    return "\'" + value.replace("\'", "\'\'") + "\'"\nsql = "SELECT name FROM users WHERE name = " + literal(username)\nreturn conn.execute(sql).fetchall()'),
    ("repeated_value", 'where = f"name = \'{username}\' OR name = \'{username}\'"\nreturn conn.execute("SELECT name FROM users WHERE " + where).fetchall()',
     'where = "name = ?1 OR name = ?1"\nreturn conn.execute("SELECT name FROM users WHERE " + where, (username,)).fetchall()'),
    ("subquery", 'base = "SELECT name FROM users"\nreturn conn.execute(f"SELECT name FROM ({base}) AS candidate WHERE name = \'{username}\'").fetchall()',
     'base = "SELECT name FROM users"\nreturn conn.execute(f"SELECT name FROM ({base}) AS candidate WHERE name = :name", {"name": username}).fetchall()'),
)

ACCESS = (
    ("wrong_object", 'has_access = any(row["owner_id"] == actor["id"] for row in records.values())\nreturn records.get(record_id) if has_access else None',
     'row = records.get(record_id)\nreturn row if row is not None and row["owner_id"] == actor["id"] else None'),
    ("fallback", 'row = records.get(record_id)\nif row is None:\n    return None\nif row["owner_id"] == actor["id"]:\n    return row\nreturn records.get(record_id)',
     'row = records.get(record_id)\nif row is None:\n    return None\nif row["owner_id"] == actor["id"]:\n    return row\nreturn None'),
    ("caught_denial", 'try:\n    row = records[record_id]\n    if row["owner_id"] != actor["id"]:\n        raise PermissionError()\n    return row\nexcept KeyError:\n    return None\nexcept PermissionError:\n    return records.get(record_id)',
     'try:\n    row = records[record_id]\n    if row["owner_id"] != actor["id"]:\n        raise PermissionError()\n    return row\nexcept (KeyError, PermissionError):\n    return None'),
    ("iterator", 'return next((row for key, row in records.items() if key == record_id and row["tenant_id"] == actor["tenant_id"]), None)',
     'return next((row for key, row in records.items() if key == record_id and row["owner_id"] == actor["id"]), None)'),
    ("delegated_guard", 'def owner(row):\n    return row["owner_id"] == actor["id"]\nrow = records.get(record_id)\nif row and (owner(row) or actor.get("authenticated")):\n    return row\nreturn None',
     'def owner(row):\n    return row["owner_id"] == actor["id"]\nrow = records.get(record_id)\nif row and owner(row):\n    return row\nreturn None'),
    ("owner_index", 'owners = {key: row["owner_id"] for key, row in records.items()}\nif actor["id"] in owners.values():\n    return records.get(record_id)\nreturn None',
     'owners = {key: row["owner_id"] for key, row in records.items()}\nif owners.get(record_id) == actor["id"]:\n    return records.get(record_id)\nreturn None'),
    ("administrator", 'row = records.get(record_id)\nif row is None or actor["role"] == "guest":\n    return None\nreturn row',
     'row = records.get(record_id)\nif row is None:\n    return None\nif actor["role"] == "admin":\n    return row\nreturn row if row["owner_id"] == actor["id"] else None'),
    ("member_index", 'permitted = {m for row in records.values() for m in row["members"]}\nreturn records.get(record_id) if actor["id"] in permitted else None',
     'members_by_id = {key: row["members"] for key, row in records.items()}\nreturn records.get(record_id) if actor["id"] in members_by_id.get(record_id, ()) else None'),
    ("sentinel", 'row = records.get(record_id, {})\nif not row:\n    return None\nidentity = row.get("tenant_id")\nreturn row if identity == actor["tenant_id"] else None',
     'row = records.get(record_id, {})\nif not row:\n    return None\nidentity = row.get("owner_id")\nreturn row if identity == actor["id"] else None'),
    ("guard_order", 'row = records.get(record_id)\nif row and actor.get("authenticated"):\n    return row\nif row and actor["id"] == row["owner_id"]:\n    return row\nreturn None',
     'row = records.get(record_id)\nif row and actor.get("authenticated") and actor["id"] == row["owner_id"]:\n    return row\nreturn None'),
)


@lru_cache(maxsize=1)
def confirmation_cases() -> tuple[Case, ...]:
    cases = []
    for category, pairs, signature in (("sql_injection", SQL, "lookup(conn, username)"),
                                       ("access_control", ACCESS, "read_record(records, actor, record_id)")):
        for name, vulnerable, clean in pairs:
            family = f"neutral_confirmation_{category}_{name}"
            if category == "sql_injection":
                threat = ("Python sqlite3. The connection and schema are trusted, and username is untrusted. "
                          "A name must be matched as a literal value and must never select other users.")
            else:
                policy = {"administrator": "An owner or an administrator may read a record.",
                          "member_index": "Only an explicitly listed record member may read it."}.get(
                              name, "Only the owner may read a record.")
                threat = ("The server-side records dictionary and actor session are trusted. The request controls "
                          "record_id. " + policy + " Login and tenant membership alone do not grant access. "
                          "Return None for missing records.")
            for label, body in ((VULNERABLE, vulnerable), (CLEAN, clean)):
                code = f"def {signature}:\n" + "".join(f"    {line}\n" for line in body.splitlines())
                cases.append(Case("c_" + digest([family, label])[:12], family, family, "neutral_confirmation", "python",
                                  category, code, threat, label, hashlib.sha256(code.encode("utf-8")).hexdigest(),
                                  ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    validate_cases(cases)
    exposed = build_cases()
    for identity in (lambda case: case.case_id, lambda case: case.family_id, lambda case: case.source_id,
                     lambda case: ast.dump(ast.parse(case.code), include_attributes=False)):
        if {identity(case) for case in exposed} & {identity(case) for case in cases}:
            raise ValueError("Confirmation cases overlap the exposed benchmark")
    return tuple(cases)
