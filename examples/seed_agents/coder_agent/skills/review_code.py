"""Code review skill — analyzes Python code quality and suggests improvements."""

import ast
from typing import Any


def review(code: str) -> dict[str, Any]:
    """Review Python code for quality issues.

    Args:
        code: Python source code to review.

    Returns:
        Dict with 'issues', 'suggestions', and 'score'.
    """
    issues: list[str] = []
    suggestions: list[str] = []

    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "issues": [f"Syntax error: {e}"],
            "suggestions": ["Fix syntax errors before review"],
            "score": 0.0,
        }

    # Analyze complexity
    complexity = _measure_complexity(tree)
    if complexity > 10:
        issues.append(f"High cyclomatic complexity: {complexity}")
        suggestions.append("Break complex functions into smaller ones")

    # Check for docstrings
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    undocumented = []
    for func in functions:
        if not (func.body and isinstance(func.body[0], ast.Expr)
                and isinstance(func.body[0].value, ast.Constant)
                and isinstance(func.body[0].value.value, str)):
            undocumented.append(func.name)

    if undocumented:
        issues.append(f"Missing docstrings: {', '.join(undocumented)}")
        suggestions.append("Add docstrings to all public functions")

    # Check function length
    for func in functions:
        length = func.end_lineno - func.lineno if func.end_lineno else 0
        if length > 50:
            issues.append(f"Function '{func.name}' is {length} lines (max recommended: 50)")
            suggestions.append(f"Refactor '{func.name}' into smaller functions")

    # Check for bare except
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append("Bare 'except:' clause found — catches all exceptions including SystemExit")
            suggestions.append("Specify the exception type (e.g., 'except ValueError:')")

    # Score
    max_issues = 10
    score = max(0.0, 1.0 - len(issues) / max_issues)

    return {
        "issues": issues,
        "suggestions": suggestions,
        "score": score,
    }


def _measure_complexity(tree: ast.AST) -> int:
    """Measure cyclomatic complexity of the code."""
    complexity = 1  # Base complexity
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return complexity
