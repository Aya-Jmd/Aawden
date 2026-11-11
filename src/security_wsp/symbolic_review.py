"""Bounded source-only data-flow summaries for simple Python review snippets."""
import ast
from copy import deepcopy


class Expand(ast.NodeTransformer):
    """Substitute earlier local assignments and simplify literal containers."""

    def __init__(self, definitions):
        self.definitions = definitions

    def visit_Name(self, node):
        return deepcopy(self.definitions.get(node.id, node))

    def visit_Lambda(self, node):
        return node

    def visit_ListComp(self, node):
        return node

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.slice, ast.Constant):
            return node
        index = node.slice.value
        if type(index) is int and isinstance(node.value, (ast.Tuple, ast.List)):
            if not any(isinstance(item, ast.Starred) for item in node.value.elts):
                if -len(node.value.elts) <= index < len(node.value.elts):
                    return node.value.elts[index]
        if isinstance(node.value, ast.Dict) and all(isinstance(key, ast.Constant) for key in node.value.keys):
            for key, value in reversed(list(zip(node.value.keys, node.value.values, strict=True))):
                if key.value == index:
                    return value
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.BitOr) or not all(isinstance(side, ast.Dict) for side in (node.left, node.right)):
            return node
        entries = list(zip(node.left.keys + node.right.keys, node.left.values + node.right.values, strict=True))
        if not all(isinstance(key, ast.Constant) for key, _ in entries):
            return node
        unique = {key.value: value for key, value in entries}
        return ast.Dict(keys=[ast.Constant(key) for key in unique], values=list(unique.values()))


def expand(node, definitions):
    return Expand(definitions).visit(deepcopy(node))


def bind(target, value, definitions):
    if isinstance(target, ast.Name):
        if value is not None and len(ast.unparse(value)) <= 320:
            definitions[target.id] = value
        else:
            definitions.pop(target.id, None)
    elif isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
            for item, operand in zip(target.elts, value.elts, strict=True):
                bind(item, operand, definitions)
        else:
            for item in target.elts:
                bind(item, None, definitions)
    elif isinstance(target, ast.Starred):
        bind(target.value, None, definitions)


def summarize_paths(statements, definitions, conditions, notes, pending, transform=None, *, specialize_assignments=False):
    if not statements or pending[0] == 0:
        return
    pending[0] -= 1
    statement, *rest = statements
    if isinstance(statement, ast.Assign) and all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) for t in statement.targets):
        value = expand(statement.value, definitions)
        if specialize_assignments:
            value = transform(value)
        for target in statement.targets:
            bind(target, value, definitions)
    elif isinstance(statement, ast.If):
        test = expand(statement.test, definitions)
        if transform is not None:
            test = transform(test)
        if transform is not None and isinstance(test, ast.Constant):
            branch = statement.body if test.value else statement.orelse
            summarize_paths(branch + rest, definitions, conditions, notes, pending, transform,
                            specialize_assignments=specialize_assignments)
            return
        expression = ast.unparse(test)
        summarize_paths(statement.body + rest, definitions.copy(), conditions + (f"truthy({expression})",),
                        notes, pending, transform, specialize_assignments=specialize_assignments)
        summarize_paths(statement.orelse + rest, definitions.copy(), conditions + (f"falsy({expression})",),
                        notes, pending, transform, specialize_assignments=specialize_assignments)
        return
    elif isinstance(statement, ast.Return):
        value = expand(statement.value, definitions) if statement.value is not None else ast.Constant(None)
        if transform is not None:
            value = transform(value)
        if isinstance(value, ast.IfExp):
            test = ast.unparse(value.test)
            notes.append((statement.lineno, conditions + (f"truthy({test})",), ast.unparse(value.body)))
            notes.append((statement.lineno, conditions + (f"falsy({test})",), ast.unparse(value.orelse)))
        else:
            notes.append((statement.lineno, conditions, ast.unparse(value)))
        return
    elif not isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Pass)):
        # Mutations and other unsupported constructs can invalidate local substitutions.
        definitions.clear()
    summarize_paths(rest, definitions, conditions, notes, pending, transform,
                    specialize_assignments=specialize_assignments)


def flow_notes(source, transform=None, *, specialize_assignments=False):
    """Show syntactic substitutions, leaving external values and unknown calls symbolic."""
    if len(source) > 12000:
        return ""
    lines = []
    for function in (node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)):
        notes = []
        summarize_paths(function.body, {}, (), notes, [96], transform, specialize_assignments=specialize_assignments)
        for line, conditions, expression in notes:
            if len(expression) > 600 or sum(map(len, conditions)) > 600:
                continue
            context = " when " + " and ".join(f"({condition})" for condition in conditions) if conditions else ""
            lines.append(f"{function.name}, line {line}{context}, returns {expression}")
    return "\n".join(dict.fromkeys(lines))
