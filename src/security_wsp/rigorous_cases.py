"""Paired security fixtures for development and preregistered confirmation."""
from functools import lru_cache
import hashlib
import random
import re
from textwrap import indent

from .artifacts import digest
from .data import CLEAN, VULNERABLE, Case, validate_cases


SQL_THREAT = (
    "Python sqlite3. The connection and schema are trusted. The caller controls username. "
    "A lookup must treat it as a literal and return only matching names."
)
ACCESS_THREAT = (
    "The session actor and records are trusted. The caller chooses record_id. Only the owner "
    "of the requested record may receive it. Login, role and tenant membership grant no exception. "
    "A missing or unauthorized record returns None."
)

# Each pair shares a body. The delimited lines are the sole difference.
DEVELOPMENT = {
    "sql_injection": {
        "query_finalization": """
head = "SELECT name FROM users"
predicate, params = ("name = ?", (username,))
<PAIR>predicate, params = f"name = '{username}'", ()||predicate, params = predicate, params</PAIR>
return conn.execute(head + " WHERE " + predicate, params).fetchall()
""",
        "binding_misdirection": """
statement = "SELECT name FROM users WHERE name = :value"
values = {"value": username}
<PAIR>statement = statement.replace(":value", "'" + values["value"] + "'")||statement = statement.replace("SELECT", "SELECT")</PAIR>
return conn.execute(statement, values).fetchall()
""",
        "quoted_intermediate": """
quoted = "'" + username + "'"
<PAIR>predicate, values = quoted, ()||predicate, values = "?", (quoted[1:-1],)</PAIR>
return conn.execute("SELECT name FROM users WHERE name = " + predicate, values).fetchall()
""",
        "fallback_on_quote": """
if "'" in username:
    <PAIR>return conn.execute("SELECT name FROM users WHERE name = '" + username + "'").fetchall()||return conn.execute("SELECT name FROM users WHERE name = ?", (username,)).fetchall()</PAIR>
return conn.execute("SELECT name FROM users WHERE name = ?", (username,)).fetchall()
""",
    },
    "access_control": {
        "boolean_any": """
row = records.get(record_id)
if row is None:
    return None
tests = (row["owner_id"] == actor["id"], actor["authenticated"])
return row if <PAIR>any(tests)||next(iter(tests))</PAIR> else None
""",
        "disallowed_state": """
row = records.get(record_id)
if row is None:
    return None
denials = {"owner": row["owner_id"] != actor["id"], "login": not actor["authenticated"]}
if denials["owner"] <PAIR>and||or</PAIR> denials["login"]:
    return None
return row
""",
        "wrong_comparison_field": """
row = records.get(record_id)
if row is None:
    return None
key = <PAIR>"tenant_id"||"owner_id"</PAIR>
field = <PAIR>"tenant_id"||"id"</PAIR>
expected = actor[field]
return row if row[key] == expected else None
""",
        "cached_policy": """
row = records.get(record_id)
if row is None:
    return None
permit = {record_id: row["owner_id"] == actor["id"]}
permit.setdefault(record_id, True)
<PAIR>permit.update({record_id: actor["authenticated"]})||permit.update({record_id: permit[record_id]})</PAIR>
return row if permit[record_id] else None
""",
    },
}

CONFIRMATION = {
    "sql_injection": {
        "trusted_alias": """
alias = "u"
fragment = f"{alias}.name = ?"
bindings = (username,)
<PAIR>fragment, bindings = f"{alias}.name = '{bindings[0]}'", ()||fragment, bindings = fragment, bindings</PAIR>
return conn.execute(f"SELECT {alias}.name FROM users AS {alias} WHERE " + fragment, bindings).fetchall()
""",
        "conditional_normalization": """
if username.isalpha():
    operand = "'" + username + "'"
    params = ()
else:
    operand, params = <PAIR>f"'{username}'", ()||"?", (username,)</PAIR>
return conn.execute("SELECT name FROM users WHERE name = " + operand, params).fetchall()
""",
        "dictionary_override": """
options = {"query": "SELECT name FROM users WHERE name = ?", "values": (username,)}
if not username.isascii() or "'" in username:
    options = options | <PAIR>{"query": "SELECT name FROM users WHERE name = '" + username + "'", "values": ()}||{"values": (username,)}</PAIR>
return conn.execute(options["query"], options["values"]).fetchall()
""",
        "sort_after_interpolation": """
column = "name"
raw = "SELECT name FROM users WHERE name = '{}'"
bound = "SELECT name FROM users WHERE name = ?"
statement, arguments = <PAIR>raw.format(username), ()||bound, (username,)</PAIR>
return conn.execute(f"{statement} ORDER BY {column}", arguments).fetchall()
""",
        "commented_decoy": """
# Prefer parameter binding for user values.
bound = ("SELECT name FROM users WHERE name = ?", (username,))
fallback = (f"SELECT name FROM users WHERE name = '{username}'", ())
index = <PAIR>1 if "'" in username else 0||0</PAIR>
selected = (bound, fallback)[index]
return conn.execute(*selected).fetchall()
""",
        "escape_then_recover": """
escaped = username.replace("'", "''")
value = <PAIR>escaped.replace("''", "'")||escaped</PAIR>
return conn.execute("SELECT name FROM users WHERE name = '" + value + "'").fetchall()
""",
        "named_parameter_and_clause": """
params = {"name": username}
where = <PAIR>"name = :name OR name = '" + username + "'"||"name = :name AND name IS NOT NULL"</PAIR>
return conn.execute("SELECT name FROM users WHERE " + where, params).fetchall()
""",
        "keyed_callable": """
def by_value(value):
    return conn.execute("SELECT name FROM users WHERE name = ?", (value,)).fetchall()
def by_expression(value):
    return conn.execute(f"SELECT name FROM users WHERE name = '{value}'").fetchall()
modes = {True: by_value, False: <PAIR>by_expression||by_value</PAIR>}
return modes[username.isidentifier()](username)
""",
    },
    "access_control": {
        "tuple_match_guard": """
item = records.get(record_id)
if item is None:
    return None
match (item["owner_id"] == actor["id"], actor["tenant_id"] == item["tenant_id"]):
    case (True, _) <PAIR>| (_, True)||</PAIR>:
        return item
    case _:
        return None
""",
        "approval_reset": """
record = records.get(record_id)
if record is None:
    return None
allowed = record["owner_id"] == actor["id"]
if actor["authenticated"] and actor["role"] == "member":
    allowed = allowed <PAIR>or||and</PAIR> actor["tenant_id"] == record["tenant_id"]
return record if allowed else None
""",
        "early_cache_hit": """
record = records.get(record_id)
if record is None:
    return None
owned = record["owner_id"] == actor["id"]
cached = {record_id: record} if <PAIR>actor["authenticated"]||owned</PAIR> else {}
if record_id in cached:
    return cached[record_id]
return record if owned else None
""",
        "negated_denials": """
record = records.get(record_id)
if record is None:
    return None
reasons = [record["owner_id"] != actor["id"], not actor["authenticated"]]
blocked = <PAIR>all||any</PAIR>(reasons)
return None if blocked else record
""",
        "filter_projection": """
rows = [(key, value) for key, value in records.items() if key == record_id]
allowed = [value for _, value in rows if value["owner_id"] == actor["id"]]
return next(iter(<PAIR>allowed or [value for _, value in rows]||allowed</PAIR>), None)
""",
        "delegated_decision": """
def allowed(item, session):
    return item["owner_id"] == session["id"]
row = records.get(record_id)
if row is None:
    return None
decide = lambda item: <PAIR>actor["role"] == "member" or allowed(item, actor)||allowed(item, actor)</PAIR>
return row if decide(row) else None
""",
        "fallback_principal": """
row = records.get(record_id)
if row is None:
    return None
principal = {"id": actor["id"]}
if actor["authenticated"]:
    principal.setdefault("id", row["owner_id"])
    <PAIR>principal["id"] = row["owner_id"]||principal["id"] = principal["id"]</PAIR>
return row if row["owner_id"] == principal["id"] else None
""",
        "request_key_guard": """
def permitted(key):
    candidate = records.get(key)
    return candidate is not None and candidate["owner_id"] == actor["id"]
own_key = next((k for k, item in records.items() if item["owner_id"] == actor["id"]), None)
check_key = <PAIR>own_key||record_id</PAIR>
return records.get(record_id) if permitted(check_key) else None
""",
    },
}


FOLLOWUP = {
    "sql_injection": {
        "projection_template": """
columns = ("name",)
target = "users"
fragment = "<PAIR>'{}'||?</PAIR>"
query = "SELECT {projection} FROM {table} WHERE name = {operand}".format(
    projection=", ".join(columns), table=target, operand=fragment)
<PAIR>return conn.cursor().execute(query.format(username)).fetchall()||return conn.cursor().execute(query, (username,)).fetchall()</PAIR>
""",
        "json_request": """
import json
query = "SELECT name FROM users WHERE name = ?"
message = json.loads(json.dumps({"text": query, "args": [username]}))
<PAIR>return conn.execute(message["text"].replace("?", "'" + message["args"][0] + "'")).fetchall()||return conn.execute(message["text"], message["args"]).fetchall()</PAIR>
""",
        "late_query_selection": """
cursor = conn.cursor()
queries = ("SELECT name FROM users WHERE name = ?", f"SELECT name FROM users WHERE name = '{username}'")
chosen = <PAIR>(1 if not username.isalnum() else 0)||0</PAIR>
args = (username,) if chosen == 0 else ()
return cursor.execute(queries[chosen], args).fetchall()
""",
        "nested_quote_escaping": """
def quoted(value):
    return "'" + value.replace("'", <PAIR>"'"||"''"</PAIR>) + "'"
value = quoted(username)
return conn.execute("SELECT name FROM users WHERE (name = " + value + ")").fetchall()
""",
        "replacement_map": """
import string
selector = string.Template("SELECT name FROM users WHERE name = $name")
filters = {"name": <PAIR>"'" + username + "'"||":match"</PAIR>}
query = selector.substitute(filters)
return conn.execute(query, {"match": username}).fetchall()
""",
        "normalization_then_restore": """
original = username
literal = original.replace("'", "''")
pieces = ("SELECT name FROM users WHERE name = '", <PAIR>original||literal</PAIR>, "'")
return conn.execute("".join(pieces)).fetchall()
""",
        "switch_after_binding": """
clause = "name = ?"
values = [username]
match username:
    case str() as text if "'" in text:
        <PAIR>clause, values = f"name = '{text}'", []||clause, values = "name = ?", [text]</PAIR>
return conn.execute("SELECT name FROM users WHERE " + clause, values).fetchall()
""",
        "identifier_only_branch": """
value = username
if value.isidentifier():
    query = f"SELECT name FROM users WHERE name = '{value}'"
    return conn.execute(query).fetchall()
lookup = <PAIR>(f"SELECT name FROM users WHERE name = '{value}'", ())||("SELECT name FROM users WHERE name = ?", (value,))</PAIR>
return conn.execute(*lookup).fetchall()
""",
    },
    "access_control": {
        "dictionary_decision_merge": """
entry = records.get(record_id)
if entry is None:
    return None
rules = {"owner": entry["owner_id"] == actor["id"]}
tenant_rules = {"tenant": entry["tenant_id"] == actor["tenant_id"]}
decision = rules | tenant_rules
return entry if <PAIR>any(decision.values())||decision["owner"]</PAIR> else None
""",
        "identity_set": """
entry = records.get(record_id)
if entry is None:
    return None
users = {entry["owner_id"]}
if actor["authenticated"]:
    users = users | <PAIR>{actor["id"]}||{entry["owner_id"]}</PAIR>
if actor["id"] in users:
    return entry
return None
""",
        "conditioned_early_return": """
row = records.get(record_id)
if row is None:
    return None
if row["tenant_id"] == actor["tenant_id"] and <PAIR>actor["authenticated"]||row["owner_id"] == actor["id"]</PAIR>:
    return row
if row["owner_id"] == actor["id"]:
    return row
return None
""",
        "auth_filter_cache": """
candidate = records.get(record_id)
if candidate is None:
    return None
def same_owner(pair):
    return pair[1]["owner_id"] == actor["id"]
permitted = dict(filter(same_owner, records.items()))
response = <PAIR>permitted.get(record_id, candidate)||permitted.get(record_id)</PAIR>
return response
""",
        "membership_inversion": """
row = records.get(record_id)
if row is None:
    return None
not_owner = row["owner_id"] != actor["id"]
reject = not_owner <PAIR>and||or</PAIR> actor["role"] == "suspended"
if reject:
    return None
return row
""",
        "short_circuit_return": """
row = records.get(record_id)
if row is None:
    return None
owner = row["owner_id"] == actor["id"]
allowed = owner <PAIR>or||and</PAIR> actor["authenticated"]
denied = not allowed
return (denied and None) or (allowed and row) or None
""",
        "attribute_projection": """
from types import SimpleNamespace
entry = records.get(record_id)
if entry is None:
    return None
record = SimpleNamespace(**entry)
session = SimpleNamespace(**actor)
member = <PAIR>record.tenant_id == session.tenant_id||record.owner_id == session.id</PAIR>
return entry if member else None
""",
        "different_record_selector": """
record = records.get(record_id)
if record is None:
    return None
keys = [key for key, item in records.items() if item["owner_id"] == actor["id"]]
matched = <PAIR>bool(keys)||record_id in keys</PAIR>
return record if matched else None
""",
    },
}


CHALLENGE = {
    "sql_injection": {
        "filter_list": """
conditions = ["name = ?"]
arguments = [username]
if "'" in username:
    <PAIR>conditions[0] = f"name = '{username}'"||conditions[0] = "name = ?"</PAIR>
    <PAIR>arguments.clear()||arguments = [username]</PAIR>
conditions.append("name IS NOT NULL")
query = "SELECT name FROM users WHERE " + " AND ".join(conditions)
return conn.execute(query, arguments).fetchall()
""",
        "separator_builder": """
def operand():
    normalized = username.encode("utf-8").decode("utf-8")
    return <PAIR>"'" + normalized + "'"||"'" + normalized.replace("'", "''") + "'"</PAIR>
query = " ".join(("SELECT name FROM users", "WHERE name =", operand()))
return conn.execute(query).fetchall()
""",
        "prepared_callable": """
def create_statement(text):
    statement = "SELECT name FROM users WHERE name = :requested"
    <PAIR>return statement.replace(":requested", f"'{text}'"), {}||return statement, {"requested": text}</PAIR>
request = create_statement(username)
return conn.cursor().execute(*request).fetchall()
""",
        "quote_count_dispatch": """
templates = {
    "literal": (f"SELECT name FROM users WHERE name = '{username}'", ()),
    "placeholder": ("SELECT name FROM users WHERE name = ?", (username,)),
}
mode = "literal" if username.count("'") <PAIR>>||==</PAIR> 0 else "placeholder"
statement, values = templates[mode]
return conn.execute(statement, values).fetchall()
""",
        "conditional_query_object": """
from types import SimpleNamespace
options = SimpleNamespace(sql="SELECT name FROM users WHERE name = ?", values=[username])
if "'" in username:
    <PAIR>options.sql, options.values = f"SELECT name FROM users WHERE name = '{username}'", []||options.values = [username]</PAIR>
cursor = conn.cursor()
return cursor.execute(options.sql, options.values).fetchall()
""",
        "aliased_projection": """
columns = ("u.name",)
relation = "users AS u"
prefix = f"SELECT {', '.join(columns)} FROM {relation}"
operand, parameters = <PAIR>(f"'{username}'", ())||(":lookup", {"lookup": username})</PAIR>
return conn.execute(prefix + f" WHERE u.name = {operand}", parameters).fetchall()
""",
        "doubled_quote_branch": """
protected = username.replace("'", "''")
choices = {"source": username, "protected": protected}
picked = choices[<PAIR>"source"||"protected"</PAIR>]
sql = f"SELECT name FROM users WHERE name = '{picked}'"
return conn.execute(sql).fetchall()
""",
        "where_transform": """
base = "SELECT name FROM users WHERE name = :username"
def specialize(sql, value):
    return <PAIR>sql.replace(":username", "'" + value + "'")||sql.replace("SELECT", "SELECT DISTINCT", 1)</PAIR>
params = {"username": username}
return conn.execute(specialize(base, username), params).fetchall()
""",
    },
    "access_control": {
        "different_query_keys": """
requested = records.get(record_id)
if requested is None:
    return None
owned = [key for key, entry in records.items() if entry["owner_id"] == actor["id"]]
verified_key = <PAIR>next(iter(owned), None)||record_id</PAIR>
return requested if verified_key in owned else None
""",
        "compound_exception": """
row = records.get(record_id)
if row is None:
    return None
owner = row["owner_id"] == actor["id"]
internal = actor["role"] == "member" and row["tenant_id"] == actor["tenant_id"]
if owner or (internal and <PAIR>actor["authenticated"]||owner</PAIR>):
    return row
return None
""",
        "set_gate": """
row = records.get(record_id)
if row is None:
    return None
issues = {"ownership"} if row["owner_id"] != actor["id"] else set()
if actor["authenticated"]:
    issues.discard(<PAIR>"ownership"||"authentication"</PAIR>)
return None if issues else row
""",
        "indexed_criteria": """
entry = records.get(record_id)
if entry is None:
    return None
criteria = [entry["tenant_id"] == actor["tenant_id"], entry["owner_id"] == actor["id"]]
selection = <PAIR>int(not actor["authenticated"])||1</PAIR>
return entry if criteria[selection] else None
""",
        "policy_payload": """
entry = records.get(record_id)
if entry is None:
    return None
session = {"principal": actor["id"], "tenant": actor["tenant_id"]}
fields = ("owner_id", "tenant_id")
identity = entry[fields[<PAIR>1||0</PAIR>]]
return entry if identity == session[<PAIR>"tenant"||"principal"</PAIR>] else None
""",
        "permission_tuple_truth": """
record = records.get(record_id)
if record is None:
    return None
checks = (record["owner_id"] == actor["id"],)
if <PAIR>bool(checks)||all(checks)</PAIR>:
    return record
return None
""",
        "multiple_authorized_paths": """
record = records.get(record_id)
if record is None:
    return None
if actor["role"] == "admin" and <PAIR>actor["authenticated"]||actor["id"] == record["owner_id"]</PAIR>:
    return record
if actor["id"] != record["owner_id"]:
    return None
return record
""",
        "record_permission_projection": """
def subject(row):
    return row[<PAIR>"tenant_id"||"owner_id"</PAIR>]
row = records.get(record_id)
if row is None:
    return None
who = actor[<PAIR>"tenant_id"||"id"</PAIR>]
return row if subject(row) == who else None
""",
    },
}


CALIBRATION_CHECK = {
    "sql_injection": {
        "literal_membership": """
fragment = <PAIR>f"'{username}'"||"?"</PAIR>
values = <PAIR>()||(username,)</PAIR>
sql = "SELECT name FROM users WHERE name IN (" + fragment + ", 'unused')"
return conn.execute(sql, values).fetchall()
""",
        "cast_argument": """
token = <PAIR>f"'{username}'"||"?"</PAIR>
arguments = <PAIR>[]||[username]</PAIR>
statement = "SELECT name FROM users WHERE name = CAST(" + token + " AS TEXT)"
return conn.execute(statement, arguments).fetchall()
""",
        "trusted_format_columns": """
table = "users"
column = "name"
left = "SELECT {0} FROM {1} WHERE {0} = ".format(column, table)
right = <PAIR>"'" + username + "'"||"?"</PAIR>
parameters = <PAIR>[]||[username]</PAIR>
return conn.execute(left + right, parameters).fetchall()
""",
        "clause_substitution": """
pattern = "SELECT name FROM users WHERE name = {operand}"
args = {"lookup": username}
operand = <PAIR>"'" + args["lookup"] + "'"||":lookup"</PAIR>
statement = pattern.format(operand=operand)
return conn.execute(statement, args).fetchall()
""",
        "translate_escape": """
escapes = {ord("'"): <PAIR>"'"||"''"</PAIR>}
literal = username.translate(escapes)
return conn.execute(f"SELECT name FROM users WHERE name = '{literal}'").fetchall()
""",
        "two_quote_branches": """
query = "SELECT name FROM users WHERE name = ?"
args = (username,)
if "'" in username:
    if username != "o'brien":
        query, args = <PAIR>f"SELECT name FROM users WHERE name = '{username}'", ()||query, args</PAIR>
return conn.execute(query, args).fetchall()
""",
        "indexed_command_options": """
options = [
    ("SELECT name FROM users WHERE name = ?", (username,)),
    (f"SELECT name FROM users WHERE name = '{username}'", ()),
]
options_for_input = <PAIR>[False, True]||[False, False]</PAIR>
index = int("'" in username)
sql, values = options[int(options_for_input[index])]
return conn.execute(sql, values).fetchall()
""",
        "query_factory_lambda": """
make_command = lambda value: <PAIR>(f"SELECT name FROM users WHERE name = '{value}'", ())||("SELECT name FROM users WHERE name = ?", (value,))</PAIR>
statement, bindings = make_command(username)
cursor = conn.cursor()
return cursor.execute(statement, bindings).fetchall()
""",
    },
    "access_control": {
        "truth_of_mapping": """
item = records.get(record_id)
if item is None:
    return None
decision = {"read": item["owner_id"] == actor["id"]}
return item if <PAIR>decision||decision["read"]</PAIR> else None
""",
        "conditional_grant_selection": """
row = records.get(record_id)
if row is None:
    return None
private = row["owner_id"] == actor["id"]
office = row["tenant_id"] == actor["tenant_id"]
allowed = (<PAIR>office||private</PAIR>) if actor["role"] == "member" else private
return row if allowed else None
""",
        "parallel_gate": """
row = records.get(record_id)
if row is None:
    return None
if actor["authenticated"] and actor["tenant_id"] == row["tenant_id"]:
    if <PAIR>actor["role"] == "member"||row["owner_id"] == actor["id"]</PAIR>:
        return row
if row["owner_id"] != actor["id"]:
    return None
return row
""",
        "permission_cache_key": """
row = records.get(record_id)
if row is None:
    return None
permissions = {key: value["owner_id"] == actor["id"] for key, value in records.items()}
key_to_check = <PAIR>next((key for key, allowed in permissions.items() if allowed), record_id)||record_id</PAIR>
if permissions.get(key_to_check):
    return row
return None
""",
        "reject_flag_cleanup": """
item = records.get(record_id)
if item is None:
    return None
flags = {"foreign": item["owner_id"] != actor["id"]}
if actor["authenticated"]:
    flags.pop(<PAIR>"foreign"||"other"</PAIR>, None)
return None if flags.get("foreign", False) else item
""",
        "numeric_gate": """
record = records.get(record_id)
if record is None:
    return None
total = int(record["owner_id"] == actor["id"]) + int(actor["authenticated"])
if total >= <PAIR>1||2</PAIR>:
    return record
return None
""",
        "branch_on_owner_mismatch": """
record = records.get(record_id)
if record is None:
    return None
if actor["id"] != record["owner_id"]:
    return <PAIR>record if actor["authenticated"] else None||None</PAIR>
return record
""",
        "closure_policy_combination": """
entry = records.get(record_id)
if entry is None:
    return None
owned = lambda value: value["owner_id"] == actor["id"]
grant = lambda value: owned(value) <PAIR>or||and</PAIR> actor["authenticated"]
return entry if grant(entry) else None
""",
    },
}


PROJECTED_CHECK = {
    "sql_injection": {
        "executed_alias_filter": """
alias = "person"
source = f"SELECT {alias}.name FROM users {alias}"
rules = {"direct": f"{alias}.name = '{username}'", "bound": f"{alias}.name = :user"}
branch = <PAIR>"direct"||"bound"</PAIR>
return conn.execute(source + " WHERE " + rules[branch], {"user": username}).fetchall()
""",
        "disjunctive_search": """
first = "name = :name"
second = <PAIR>"name = '" + username + "'"||"name = :name"</PAIR>
text = f"SELECT name FROM users WHERE {first} OR {second}"
return conn.execute(text, {"name": username}).fetchall()
""",
        "guarded_reassignment": """
command = ("SELECT name FROM users WHERE name = ?", [username])
if username.count("'"):
    statement = "SELECT name FROM users WHERE name = "
    command = <PAIR>(statement + "'" + username + "'", [])||(statement + "?", [username])</PAIR>
return conn.execute(command[0], command[1]).fetchall()
""",
        "selected_subexpression": """
candidate = (f"'{username}'", "?", ":name")
chosen = <PAIR>0||2</PAIR>
argument = candidate[chosen]
sql = "SELECT name FROM users WHERE name = " + argument
return conn.cursor().execute(sql, {"name": username}).fetchall()
""",
        "template_operand_rewrite": """
expression = "name = :name"
if "'" in username:
    quoted = "'" + username + "'"
    expression = <PAIR>expression.replace(":name", quoted)||expression</PAIR>
return conn.execute("SELECT name FROM users WHERE " + expression, {"name": username}).fetchall()
""",
        "literal_recovery_mapping": """
quoted = username.replace("'", "''")
options = {"canonical": username, "literal": quoted}
value = options[<PAIR>"canonical"||"literal"</PAIR>]
clause = "name = " + "'" + value + "'"
return conn.execute("SELECT name FROM users WHERE " + clause).fetchall()
""",
        "alpha_only_early_return": """
fixed = "SELECT name FROM users WHERE name = "
if username.isalpha():
    return conn.execute(fixed + f"'{username}'").fetchall()
query = <PAIR>(fixed + f"'{username}'", ())||(fixed + "?", (username,))</PAIR>
return conn.execute(query[0], query[1]).fetchall()
""",
        "select_distinct_command": """
raw = f"SELECT DISTINCT name FROM users WHERE name = '{username}'"
fixed = "SELECT DISTINCT name FROM users WHERE name = :name"
selected = {"text": <PAIR>raw||fixed</PAIR>, "parameters": {"name": username}}
return conn.execute(selected["text"], selected["parameters"]).fetchall()
""",
    },
    "access_control": {
        "snapshot_decision": """
row = records.get(record_id)
if row is None:
    return None
snapshot = (row["owner_id"] == actor["id"], actor["authenticated"])
selected = <PAIR>snapshot[1]||snapshot[0]</PAIR>
if selected:
    return row
return None
""",
        "privileged_early_branch": """
row = records.get(record_id)
if row is None:
    return None
owner = row["owner_id"] == actor["id"]
if actor["role"] == "admin":
    permit = <PAIR>actor["authenticated"]||owner</PAIR>
    return row if permit else None
return row if owner else None
""",
        "policy_key_indirection": """
item = records.get(record_id)
if item is None:
    return None
checks = {"scope": item["owner_id"] == actor["id"], "live": actor["authenticated"]}
criterion = <PAIR>"live"||"scope"</PAIR>
return item if checks[criterion] else None
""",
        "negative_gate_pair": """
row = records.get(record_id)
if row is None:
    return None
wrong = row["owner_id"] != actor["id"]
closed = not actor["authenticated"]
denied = <PAIR>wrong and closed||wrong or closed</PAIR>
return None if denied else row
""",
        "structured_owner_identity": """
record = records.get(record_id)
if record is None:
    return None
required = (record["owner_id"], record["tenant_id"])
supplied = (<PAIR>record["owner_id"]||actor["id"]</PAIR>, actor["tenant_id"])
if required == supplied:
    return record
return None
""",
        "response_lookup_boolean": """
entry = records.get(record_id)
if entry is None:
    return None
responses = (None, entry)
owner = entry["owner_id"] == actor["id"]
selected = int(<PAIR>actor["authenticated"]||owner</PAIR>)
return responses[selected]
""",
        "tenant_branch_update": """
entry = records.get(record_id)
if entry is None:
    return None
permit = entry["owner_id"] == actor["id"]
if entry["tenant_id"] == actor["tenant_id"]:
    if actor["role"] == "member":
        permit = permit <PAIR>or||and</PAIR> actor["authenticated"]
return entry if permit else None
""",
        "aggregate_permission_flags": """
row = records.get(record_id)
if row is None:
    return None
ownership = row["owner_id"] == actor["id"]
local = row["tenant_id"] == actor["tenant_id"]
checks = (<PAIR>ownership or local||ownership</PAIR>, actor["authenticated"])
return row if all(checks) else None
""",
    },
}


SECOND_OPINION_CHECK = {
    "sql_injection": {
        "character_class_switch": """
prefix = "SELECT name FROM users WHERE name = "
if username.<PAIR>isascii||isalnum</PAIR>():
    return conn.execute(prefix + "'" + username + "'").fetchall()
return conn.execute(prefix + "?", (username,)).fetchall()
""",
        "literal_pipeline": """
first = username.replace("'", "''")
chosen = <PAIR>first.replace("''", "'")||first</PAIR>
expression = "SELECT name FROM users WHERE name = '{}'"
return conn.execute(expression.format(chosen)).fetchall()
""",
        "comparison_substitution": """
template = "SELECT name FROM users WHERE name = {value} AND name IS NOT NULL"
value = <PAIR>f"'{username}'"||":requested"</PAIR>
sql = template.format(value=value)
return conn.execute(sql, {"requested": username}).fetchall()
""",
        "positional_choice": """
queries = (f"SELECT name FROM users WHERE name = '{username}'", "SELECT name FROM users WHERE name = ?")
bound = (username,)
selected = <PAIR>0||1</PAIR>
parameters = () if selected == 0 else bound
return conn.execute(queries[selected], parameters).fetchall()
""",
        "quoted_branch_binding": """
lookup = "SELECT name FROM users WHERE name = ?"
arguments = (username,)
if "'" in username:
    lookup = <PAIR>f"SELECT name FROM users WHERE name = '{username}'"||lookup</PAIR>
    arguments = <PAIR>()||(username,)</PAIR>
return conn.cursor().execute(lookup, arguments).fetchall()
""",
        "selected_named_plan": """
plan = ("SELECT name FROM users WHERE name = :request", {"request": username})
fallback = (f"SELECT name FROM users WHERE name = '{username}'", {})
text, values = <PAIR>fallback||plan</PAIR>
return conn.execute(text, values).fetchall()
""",
        "format_explicit_alias": """
name_col = "u.name"
trusted_head = "SELECT {} FROM users u WHERE {} = ".format(name_col, name_col)
clause = <PAIR>"'" + username + "'"||"?"</PAIR>
bindings = <PAIR>()||(username,)</PAIR>
statement = trusted_head + clause
return conn.execute(statement, bindings).fetchall()
""",
        "guarded_named_binding": """
plan = "SELECT name FROM users WHERE name = :value"
mapping = {"value": username}
if username.isidentifier():
    return conn.execute(plan, mapping).fetchall()
statement = <PAIR>plan.replace(":value", f"'{username}'")||plan</PAIR>
return conn.execute(statement, mapping).fetchall()
""",
    },
    "access_control": {
        "scoped_membership_tuple": """
row = records.get(record_id)
if row is None:
    return None
key = (row["owner_id"], row["tenant_id"])
values = (<PAIR>row["owner_id"]||actor["id"]</PAIR>, actor["tenant_id"])
return row if values == key else None
""",
        "necessary_login_flag": """
row = records.get(record_id)
if row is None:
    return None
owner = actor["id"] == row["owner_id"]
peer = actor["tenant_id"] == row["tenant_id"]
requirements = [<PAIR>peer||owner</PAIR>, actor["authenticated"]]
return row if all(requirements) else None
""",
        "later_flag_assignment": """
row = records.get(record_id)
if row is None:
    return None
permitted = row["owner_id"] == actor["id"]
if actor["role"] == "member":
    permitted = permitted <PAIR>or||and</PAIR> (row["tenant_id"] == actor["tenant_id"])
if permitted:
    return row
return None
""",
        "bool_guard_lookup": """
row = records.get(record_id)
if row is None:
    return None
choice = {"identity": actor["id"] == row["owner_id"], "session": actor["authenticated"]}
return (None, row)[int(choice[<PAIR>"session"||"identity"</PAIR>])]
""",
        "same_tenant_return_gate": """
record = records.get(record_id)
if record is None:
    return None
if record["tenant_id"] == actor["tenant_id"]:
    if <PAIR>actor["authenticated"]||record["owner_id"] == actor["id"]</PAIR>:
        return record
return record if record["owner_id"] == actor["id"] else None
""",
        "inverted_deny_flags": """
item = records.get(record_id)
if item is None:
    return None
wrong_user = item["owner_id"] != actor["id"]
session_open = actor["authenticated"]
deny = <PAIR>wrong_user and not session_open||wrong_user or not session_open</PAIR>
return None if deny else item
""",
        "privilege_or_identity": """
item = records.get(record_id)
if item is None:
    return None
own = item["owner_id"] == actor["id"]
privilege = actor["role"] == "admin" and actor["authenticated"]
if own or (privilege <PAIR>or||and</PAIR> own):
    return item
return None
""",
        "select_permitted_record": """
record = records.get(record_id)
if record is None:
    return None
owner_check = actor["id"] == record["owner_id"]
other_check = actor["tenant_id"] == record["tenant_id"]
states = (owner_check, other_check)
allowed = states[<PAIR>1||0</PAIR>]
return record if allowed else None
""",
    },
}


def expand(source, side):
    """Replace each paired edit with the vulnerable or clean alternative."""
    return re.sub(r"<PAIR>(.*?)\|\|(.*?)</PAIR>", lambda match: match.group(side), source.strip())


@lru_cache(maxsize=1)
def rigorous_cases():
    cases = []
    splits = (("rigorous_development", DEVELOPMENT), ("rigorous_confirmation", CONFIRMATION),
              ("rigorous_confirmation_v2", FOLLOWUP), ("rigorous_confirmation_v3", CHALLENGE),
              ("rigorous_confirmation_v4", CALIBRATION_CHECK), ("rigorous_confirmation_v5", PROJECTED_CHECK),
              ("rigorous_confirmation_v6", SECOND_OPINION_CHECK))
    for split, catalog in splits:
        for category, families in catalog.items():
            signature, threat = ("lookup(conn, username)", SQL_THREAT) if category == "sql_injection" else (
                "read_record(records, actor, record_id)", ACCESS_THREAT)
            for name, body in families.items():
                family = f"{split}_{category}_{name}"
                for side, label in ((1, VULNERABLE), (2, CLEAN)):
                    code = f"def {signature}:\n" + indent(expand(body, side), "    ") + "\n"
                    cases.append(Case("c_" + digest([family, label])[:12], family, family, split, "python",
                                      category, code, threat, label, hashlib.sha256(code.encode()).hexdigest(),
                                      ("legitimate_access", "adversarial_access", "missing_or_literal_input")))
    original = [case for case in cases if case.split in ("rigorous_development", "rigorous_confirmation")]
    followup = [case for case in cases if case.split == "rigorous_confirmation_v2"]
    challenge = [case for case in cases if case.split == "rigorous_confirmation_v3"]
    calibration = [case for case in cases if case.split == "rigorous_confirmation_v4"]
    projected = [case for case in cases if case.split == "rigorous_confirmation_v5"]
    second_opinion = [case for case in cases if case.split == "rigorous_confirmation_v6"]
    random.Random(429).shuffle(original)
    random.Random(430).shuffle(followup)
    random.Random(431).shuffle(challenge)
    random.Random(432).shuffle(calibration)
    random.Random(433).shuffle(projected)
    random.Random(434).shuffle(second_opinion)
    combined = original + followup + challenge + calibration + projected + second_opinion
    validate_cases(combined)
    return tuple(combined)
