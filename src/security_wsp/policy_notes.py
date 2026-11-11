"""Project symbolic paths under explicit security-review assumptions."""
import ast
from string import Formatter

from .symbolic_review import Expand, flow_notes


def expression_key(node):
    return ast.dump(node, include_attributes=False)


OWNER_FIELDS = {expression_key(ast.parse(expr, mode="eval").body) for expr in (
    "records.get(record_id)['owner_id']", "records[record_id]['owner_id']",
)}
ACTOR_FIELDS = {expression_key(ast.parse(expr, mode="eval").body) for expr in (
    "actor['id']", "actor.get('id')",
)}
REQUESTED_TENANTS = {expression_key(ast.parse(expr, mode="eval").body) for expr in (
    "records.get(record_id)['tenant_id']", "records[record_id]['tenant_id']",
)}
ACTOR_TENANT = expression_key(ast.parse("actor['tenant_id']", mode="eval").body)


class BooleanProjection(Expand):
    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node


class NonOwner(BooleanProjection):
    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            operands = {expression_key(node.left), expression_key(node.comparators[0])}
            if operands & ACTOR_FIELDS and operands & OWNER_FIELDS:
                return ast.Constant(isinstance(node.ops[0], ast.NotEq))
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        require_all = isinstance(node.op, ast.And)
        while len(node.values) > 1 and isinstance(node.values[0], ast.Constant):
            if bool(node.values[0].value) != require_all:
                return node.values[0]
            node.values.pop(0)
        return node.values[0] if len(node.values) == 1 else node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if node.keywords or len(node.args) != 1 or not isinstance(node.func, ast.Name):
            return node
        argument = node.args[0]
        if node.func.id == "int" and isinstance(argument, ast.Constant) and type(argument.value) is bool:
            return ast.Constant(int(argument.value))
        if node.func.id in ("all", "any") and isinstance(argument, (ast.Tuple, ast.List)):
            values = argument.elts
            if values and all(isinstance(value, ast.Constant) for value in values):
                aggregate = all if node.func.id == "all" else any
                return ast.Constant(aggregate(value.value for value in values))
        return node


class SqlText(Expand):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        receiver, method = node.func.value, node.func.attr
        if method == "replace" and not node.keywords and len(node.args) == 2:
            old, new = node.args
            if isinstance(old, ast.Constant) and isinstance(new, ast.Constant) and old.value == new.value:
                return receiver
            if (isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute)
                    and receiver.func.attr == "replace" and not receiver.keywords
                    and [ast.unparse(arg) for arg in receiver.args] == [repr("'"), repr("''")]
                    and [ast.unparse(arg) for arg in node.args] == [repr("''"), repr("'")]):
                return receiver.func.value
        if method == "format" and isinstance(receiver, ast.Constant) and isinstance(receiver.value, str):
            return self.format_string(receiver.value, node)
        return node

    @staticmethod
    def format_string(template, original):
        if any(item.arg is None for item in original.keywords):
            return original
        fields = {item.arg: item.value for item in original.keywords}
        position = 0
        numbered = False
        parts = []
        try:
            for text, field, spec, conversion in Formatter().parse(template):
                if text:
                    parts.append(ast.Constant(text))
                if field is None:
                    continue
                if spec or conversion:
                    return original
                if not field:
                    if numbered:
                        return original
                    value = original.args[position]
                    position += 1
                elif field.isdecimal():
                    if position:
                        return original
                    numbered = True
                    value = original.args[int(field)]
                else:
                    value = fields[field]
                if isinstance(value, ast.Starred):
                    return original
                parts.append(ast.FormattedValue(value=value, conversion=-1, format_spec=None))
        except (ValueError, KeyError, IndexError):
            return original
        return ast.JoinedStr(values=parts)


def policy_notes(case):
    source = case.visible()["code"]
    if case.target_category == "sql_injection":
        return "Symbolic SQL text and bind arguments\n" + flow_notes(source, SqlText({}).visit)
    if "Only the owner" in case.threat_model:
        return ("Symbolic paths assuming the caller is not the requested record owner\n"
                + flow_notes(source, NonOwner({}).visit))
    return "Symbolic paths\n" + flow_notes(source)


class SessionProjection(NonOwner):
    """Simplify expressions under explicit session properties, leaving records symbolic."""

    def __init__(self, role, same_tenant):
        super().__init__({})
        self.role = role
        self.same_tenant = same_tenant

    def visit_Subscript(self, node):
        node = super().visit_Subscript(node)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Dict) and len(node.value.keys) == 1:
            key = node.value.keys[0]
            if key is not None and expression_key(key) == expression_key(node.slice):
                return node.value.values[0]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "actor":
            if isinstance(node.slice, ast.Constant):
                if node.slice.value == "authenticated":
                    return ast.Constant(True)
                if node.slice.value == "role":
                    return ast.Constant(self.role)
        return node

    def visit_Compare(self, node):
        node = super().visit_Compare(node)
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return node
        operator, left, right = node.ops[0], node.left, node.comparators[0]
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            keys = {expression_key(left), expression_key(right)}
            if ACTOR_TENANT in keys and keys & REQUESTED_TENANTS:
                return ast.Constant(self.same_tenant == isinstance(operator, ast.Eq))
            if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                return ast.Constant((left.value == right.value) == isinstance(operator, ast.Eq))
        return node

    def visit_Call(self, node):
        node = super().visit_Call(node)
        if (isinstance(node, ast.Call) and not node.keywords and len(node.args) in (1, 2)
                and isinstance(node.func, ast.Name) and node.func.id == "next"):
            iterator = node.args[0]
            if (isinstance(iterator, ast.Call) and not iterator.keywords and len(iterator.args) == 1
                    and isinstance(iterator.func, ast.Name) and iterator.func.id == "iter"):
                value = iterator.args[0]
                if isinstance(value, (ast.Tuple, ast.List)) and value.elts and not isinstance(value.elts[0], ast.Starred):
                    return value.elts[0]
        if (isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1
                and isinstance(node.func, ast.Name) and node.func.id == "bool"):
            value = node.args[0]
            if isinstance(value, ast.Constant):
                return ast.Constant(bool(value.value))
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)) and not any(
                isinstance(item, ast.Starred) for item in value.elts
            ):
                return ast.Constant(bool(value.elts))
        return node


class QuotedName(SqlText, BooleanProjection):
    """Specialize branches for an arbitrary username containing a single quote."""

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        operator, left, right = node.ops[0], node.left, node.comparators[0]
        if (isinstance(left, ast.Constant) and left.value == "'" and isinstance(right, ast.Name)
                and right.id == "username" and isinstance(operator, (ast.In, ast.NotIn))):
            return ast.Constant(isinstance(operator, ast.In))
        if (isinstance(right, ast.Constant) and isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute) and left.func.attr == "count"
                and isinstance(left.func.value, ast.Name) and left.func.value.id == "username"
                and [ast.unparse(arg) for arg in left.args] == [repr("'")] and not left.keywords):
            if right.value == 0 and isinstance(operator, (ast.Eq, ast.NotEq, ast.Gt, ast.LtE)):
                return ast.Constant(isinstance(operator, (ast.NotEq, ast.Gt)))
            if right.value == 1 and isinstance(operator, (ast.GtE, ast.Lt)):
                return ast.Constant(isinstance(operator, ast.GtE))
        return node

    def visit_Call(self, node):
        node = super().visit_Call(node)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "username"
                and node.func.attr in ("isalpha", "isalnum", "isidentifier")
                and not node.args and not node.keywords):
            return ast.Constant(False)
        return node

def counterfactual_notes(case):
    source = case.visible()["code"]
    if case.target_category == "sql_injection":
        return ("Assume the name contains an apostrophe\n"
                + flow_notes(source, QuotedName({}).visit, specialize_assignments=True))
    if "Only the owner" not in case.threat_model:
        return policy_notes(case)
    projections = (
        ("Logged-in non-owner, same tenant", "member", True),
        ("Logged-in non-owner, different tenant", "member", False),
        ("Logged-in non-owner with admin role", "admin", True),
    )
    return "\n\n".join(heading + "\n" + flow_notes(source, SessionProjection(role, tenant).visit,
                                                      specialize_assignments=True)
                       for heading, role, tenant in projections)


def compact_projection(case):
    """Limit a code-only projection to snippets without unmodelled statements or mutations."""
    source = case.visible()["code"]
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    statement_types = (ast.FunctionDef, ast.If, ast.Return, ast.Assign, ast.Pass)
    unsupported = (ast.Lambda, ast.comprehension, ast.NamedExpr, ast.Starred)
    allowed_methods = {"get", "execute", "fetchall", "cursor", "replace", "format", "join", "translate",
                       "isalpha", "isalnum", "isascii", "isidentifier", "count", "encode", "decode"}
    allowed_functions = {"int", "bool", "any", "all", "ord", "set"}
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].decorator_list:
        return ""
    for node in nodes:
        if isinstance(node, unsupported) or (isinstance(node, ast.stmt) and not isinstance(node, statement_types)):
            return ""
        if isinstance(node, ast.Assign) and not all(isinstance(target, (ast.Name, ast.Tuple, ast.List))
                                                    for target in node.targets):
            return ""
        if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Load):
            return ""
        if isinstance(node, ast.Call):
            known = ((isinstance(node.func, ast.Name) and node.func.id in allowed_functions)
                     or (isinstance(node.func, ast.Attribute) and node.func.attr in allowed_methods))
            if not known:
                return ""
    return policy_notes(case) + "\n\n" + counterfactual_notes(case)
