"""Surface control flow and call arguments from Python syntax without executing code."""
import ast


def source_notes(source):
    """Describe visible syntax only. The reviewer still decides reachability and policy."""
    lines, operations = [], []
    nodes = list(ast.walk(ast.parse(source)))
    for function in (node for node in nodes if isinstance(node, ast.FunctionDef)):
        lines.append(f"Function {function.name} starts at line {function.lineno}.")
        _describe_returns(function.body, (), lines)
    for node in nodes:
        if isinstance(node, ast.BoolOp):
            operations.append(f"Line {node.lineno} needs {_condition(node)}.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("any", "all"):
            behavior = "at least one truthy element" if node.func.id == "any" else "every element truthy"
            operations.append(f"Line {node.lineno} tests {behavior} in {ast.unparse(node.args[0])}."
                              if node.args else f"Line {node.lineno} calls {node.func.id}.")
        if isinstance(node, ast.JoinedStr):
            values = [ast.unparse(part.value) for part in node.values if isinstance(part, ast.FormattedValue)]
            operations.append(f"Line {node.lineno} interpolates {', '.join(values)} into string text.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            arguments = [ast.unparse(argument) for argument in node.args]
            if node.args and isinstance(node.args[0], ast.Starred):
                operations.append(f"Line {node.lineno} expands {arguments[0]} into separate execute arguments.")
            elif arguments:
                bindings = arguments[1] if len(arguments) > 1 else "none supplied positionally"
                operations.append(f"Line {node.lineno} calls execute with SQL text {arguments[0]} and bindings {bindings}.")
    return "\n".join(lines + operations)


def _condition(node):
    if isinstance(node, ast.BoolOp):
        relation = "ALL true" if isinstance(node.op, ast.And) else "AT LEAST ONE true"
        return relation + " of [" + "; ".join(_condition(item) for item in node.values) + "]"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "falsehood of (" + _condition(node.operand) + ")"
    return ast.unparse(node)


def _describe_returns(statements, conditions, lines):
    for statement in statements:
        if isinstance(statement, ast.If):
            condition = _condition(statement.test)
            yes = f"line {statement.lineno} ({condition}) is true"
            no = f"line {statement.lineno} ({condition}) is false"
            then_returns = _describe_returns(statement.body, (*conditions, yes), lines)
            else_returns = _describe_returns(statement.orelse, (*conditions, no), lines)
            if then_returns and else_returns:
                return True
            if then_returns:
                conditions = (*conditions, no)
            if else_returns:
                conditions = (*conditions, yes)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                pattern = f"{ast.unparse(statement.subject)} matches {ast.unparse(case.pattern)}"
                guard = f" and {_condition(case.guard)} is true" if case.guard else ""
                _describe_returns(case.body, (*conditions, f"line {statement.lineno} ({pattern}{guard})"), lines)
        elif isinstance(statement, ast.Return):
            value = statement.value
            prefix = f"Line {statement.lineno}"
            if conditions:
                prefix += " under " + ", ".join(conditions)
            if isinstance(value, ast.IfExp):
                condition = _condition(value.test)
                lines.append(f"{prefix} returns {ast.unparse(value.body)} if {condition}; "
                             f"otherwise it returns {ast.unparse(value.orelse)}.")
            else:
                lines.append(f"{prefix} returns {ast.unparse(value) if value else 'None'}.")
            return True
    return False
