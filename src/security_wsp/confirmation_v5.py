"""New cases for the preregistered mental-execution review pipeline."""
import ast
import hashlib
from functools import lru_cache
from .artifacts import digest
from .data import Case, CLEAN, VULNERABLE, build_cases, validate_cases
from .confirmation_data import confirmation_cases
from .confirmation_v2 import confirmation_cases_v2
from .confirmation_v3 import confirmation_cases_v3
from .confirmation_v4 import confirmation_cases_v4

PAIRS = {
    "sql_injection": (
        ("transaction_scope", 'with conn:\n    return conn.execute(f"SELECT name FROM users WHERE name = \'{username}\'").fetchall()', 'with conn:\n    return conn.execute("SELECT name FROM users WHERE name = ?", (username,)).fetchall()'),
        ("query_prefix", 'prefix = "SELECT name FROM users WHERE name = "\nstatement = prefix + "\'" + username + "\'"\nreturn conn.execute(statement).fetchall()', 'prefix = "SELECT name FROM users WHERE name = "\nstatement = prefix + "?"\nreturn conn.execute(statement, tuple([username])).fetchall()'),
        ("lambda_sink", 'sink = lambda query, values: conn.execute(query, values).fetchall()\nreturn sink(f"SELECT name FROM users WHERE name = \'{username}\'", ())', 'sink = lambda query, values: conn.execute(query, values).fetchall()\nreturn sink("SELECT name FROM users WHERE name = ?", (username,))'),
        ("typed_command", 'from collections import namedtuple\nCommand = namedtuple("Command", "text params")\ncommand = Command(f"SELECT name FROM users WHERE name = \'{username}\'", ())\nreturn conn.execute(command.text, command.params).fetchall()', 'from collections import namedtuple\nCommand = namedtuple("Command", "text params")\ncommand = Command("SELECT name FROM users WHERE name = ?", (username,))\nreturn conn.execute(command.text, command.params).fetchall()'),
        ("command_stack", 'commands = [(f"SELECT name FROM users WHERE name = \'{username}\'", ())]\nquery, values = commands.pop()\nreturn conn.execute(query, values).fetchall()', 'commands = [("SELECT name FROM users WHERE name = :value", {"value": username})]\nquery, values = commands.pop()\nreturn conn.execute(query, values).fetchall()'),
    ),
    "access_control": (
        ("predicate_sequence", 'owner = lambda item: item["owner_id"] == actor["id"]\nreturn next((row for key, row in records.items() if key == record_id and (owner(row) or actor["authenticated"])), None)', 'owner = lambda item: item["owner_id"] == actor["id"]\nreturn next((row for key, row in records.items() if key == record_id and owner(row)), None)'),
        ("request_policy", 'row = records.get(record_id)\nif row is None:\n    return None\nallowed = {record_id: actor["authenticated"]}\nreturn row if allowed[record_id] else None', 'row = records.get(record_id)\nif row is None:\n    return None\nallowed = {record_id: row["owner_id"] == actor["id"]}\nreturn row if allowed[record_id] else None'),
        ("pair_match", 'row = records.get(record_id)\nmatch (row, actor["id"]):\n    case ({"owner_id": owner}, who) if actor["authenticated"]:\n        return row\n    case _:\n        return None', 'row = records.get(record_id)\nmatch (row, actor["id"]):\n    case ({"owner_id": owner}, who) if owner == who:\n        return row\n    case _:\n        return None'),
        ("deferred_return", 'row = records.get(record_id)\nif row is None:\n    return None\nresponse = (lambda: row) if actor["authenticated"] else (lambda: None)\nreturn response()', 'row = records.get(record_id)\nif row is None:\n    return None\nresponse = (lambda: row) if row["owner_id"] == actor["id"] else (lambda: None)\nreturn response()'),
        ("indexed_result", 'row = records.get(record_id)\nif row is None:\n    return None\nallowed = actor["authenticated"]\nreturn (None, row)[int(allowed)]', 'row = records.get(record_id)\nif row is None:\n    return None\nallowed = row["owner_id"] == actor["id"]\nreturn (None, row)[int(allowed)]'),
    ),
}


@lru_cache(maxsize=1)
def confirmation_cases_v5():
    cases = []
    for category, pairs in PAIRS.items():
        signature = "lookup(conn, username)" if category == "sql_injection" else "read_record(records, actor, record_id)"
        threat = ("Python sqlite3. The schema and connection are trusted. username is untrusted and must be matched as a literal value without selecting other users."
                  if category == "sql_injection" else "records and the authenticated actor session are trusted. record_id is attacker-controlled. Only the owner may read a record. Login and tenant membership alone do not permit reading another owner's record. Missing records return None.")
        for name, bad, good in pairs:
            family = f"confirmation_v5_{category}_{name}"
            for label, body in ((VULNERABLE, bad), (CLEAN, good)):
                code = f"def {signature}:\n" + "".join(f"    {line}\n" for line in body.splitlines())
                cases.append(Case("c_" + digest([family, label])[:12], family, family, "neutral_confirmation_v5", "python", category,
                                  code, threat, label, hashlib.sha256(code.encode()).hexdigest(),
                                  ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    validate_cases(cases)
    exposed = (*build_cases(), *confirmation_cases(), *confirmation_cases_v2(), *confirmation_cases_v3(), *confirmation_cases_v4())
    syntax = lambda case: ast.dump(ast.parse(case.code), include_attributes=False)
    if {syntax(case) for case in exposed} & {syntax(case) for case in cases}:
        raise ValueError("Confirmation v5 leaks syntax from previously exposed cases")
    return tuple(cases)
