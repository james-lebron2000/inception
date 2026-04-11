"""Code debugging skill — analyzes and fixes Python code errors."""

import ast
import traceback
from typing import Any


def analyze_error(code: str, error_message: str) -> dict[str, Any]:
    """Analyze a code error and suggest fixes.

    Args:
        code: The Python source code with the error.
        error_message: The error message or traceback.

    Returns:
        Dict with 'diagnosis', 'suggestion', and 'fixed_code'.
    """
    # Check for syntax errors first
    syntax_result = check_syntax(code)
    if syntax_result["has_error"]:
        return {
            "diagnosis": f"Syntax error at line {syntax_result['line']}: {syntax_result['message']}",
            "suggestion": "Fix the syntax error before running the code",
            "fixed_code": code,
        }

    # Pattern-based error analysis
    patterns = {
        "NameError": _fix_name_error,
        "TypeError": _fix_type_error,
        "IndexError": _fix_index_error,
        "KeyError": _fix_key_error,
        "AttributeError": _fix_attribute_error,
    }

    for error_type, fixer in patterns.items():
        if error_type in error_message:
            return fixer(code, error_message)

    return {
        "diagnosis": f"Error detected: {error_message}",
        "suggestion": "Review the code logic and add error handling",
        "fixed_code": code,
    }


def check_syntax(code: str) -> dict[str, Any]:
    """Check Python code for syntax errors."""
    try:
        ast.parse(code)
        return {"has_error": False}
    except SyntaxError as e:
        return {
            "has_error": True,
            "line": e.lineno,
            "message": str(e.msg),
            "offset": e.offset,
        }


def _fix_name_error(code: str, error: str) -> dict[str, Any]:
    return {
        "diagnosis": "A variable or function is used before being defined",
        "suggestion": "Check variable spelling and ensure all names are defined before use",
        "fixed_code": code,
    }


def _fix_type_error(code: str, error: str) -> dict[str, Any]:
    return {
        "diagnosis": "An operation received an argument of the wrong type",
        "suggestion": "Add type checking or convert types before the operation",
        "fixed_code": code,
    }


def _fix_index_error(code: str, error: str) -> dict[str, Any]:
    return {
        "diagnosis": "A list/tuple index is out of range",
        "suggestion": "Add bounds checking before accessing by index",
        "fixed_code": code,
    }


def _fix_key_error(code: str, error: str) -> dict[str, Any]:
    return {
        "diagnosis": "A dictionary key does not exist",
        "suggestion": "Use .get() method or check key existence with 'in' operator",
        "fixed_code": code,
    }


def _fix_attribute_error(code: str, error: str) -> dict[str, Any]:
    return {
        "diagnosis": "An object does not have the expected attribute",
        "suggestion": "Check the object type and use hasattr() for safety",
        "fixed_code": code,
    }
