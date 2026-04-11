"""Sandboxed code execution — safe evaluation of LLM-generated code.

Executes agent skill code in a restricted subprocess with resource limits.
This is critical for fitness evaluation: we run evolved code but need to
prevent runaway processes, infinite loops, or dangerous operations.

Biological analog: immune system — protecting the organism from harmful mutations.
"""

from __future__ import annotations

import ast
import asyncio
import multiprocessing
import sys
import traceback
from typing import Any


# Dangerous modules/builtins that should not be used in evolved code
BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "sys", "importlib",
    "ctypes", "socket", "http", "urllib", "requests",
    "pathlib",  # Can be allowed selectively
}

BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__",
    "open", "input", "breakpoint",
}


class SandboxViolation(Exception):
    """Raised when code attempts a disallowed operation."""
    pass


def check_code_safety(source: str) -> list[str]:
    """Static analysis to detect potentially dangerous code patterns.

    Returns a list of safety violations found.
    """
    violations: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import from: {node.module}")

        # Check for dangerous builtins
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_BUILTINS:
                    violations.append(f"Blocked builtin call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_BUILTINS:
                    violations.append(f"Blocked attribute call: {node.func.attr}")

    return violations


def _run_in_process(code: str, function_name: str, args: list, result_queue: multiprocessing.Queue):
    """Worker function that runs in a separate process."""
    try:
        # Create a restricted globals dict
        safe_globals: dict[str, Any] = {"__builtins__": {}}

        # Allow safe builtins
        import builtins
        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if name not in BLOCKED_BUILTINS and not name.startswith("_")
        }
        safe_globals["__builtins__"] = safe_builtins

        # Execute the code to define functions
        exec(code, safe_globals)  # noqa: S102

        # Call the target function
        if function_name not in safe_globals:
            result_queue.put(("error", f"Function '{function_name}' not found"))
            return

        func = safe_globals[function_name]
        result = func(*args)
        result_queue.put(("ok", result))

    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


async def execute_code_safely(
    code: str,
    function_name: str = "solve",
    args: list | None = None,
    timeout: int = 10,
) -> Any:
    """Execute code in a sandboxed subprocess.

    Args:
        code: Python source code to execute.
        function_name: Name of the function to call.
        args: Arguments to pass to the function.
        timeout: Maximum execution time in seconds.

    Returns:
        The function's return value.

    Raises:
        SandboxViolation: If code fails safety checks.
        TimeoutError: If execution exceeds timeout.
        RuntimeError: If execution fails.
    """
    if args is None:
        args = []

    # Static safety check
    violations = check_code_safety(code)
    if violations:
        raise SandboxViolation(f"Code safety violations: {'; '.join(violations)}")

    # Run in a separate process with timeout
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_in_process,
        args=(code, function_name, args, result_queue),
    )

    process.start()

    # Wait with timeout
    try:
        process.join(timeout=timeout)
    except Exception:
        process.terminate()
        raise TimeoutError(f"Code execution exceeded {timeout}s timeout")

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        raise TimeoutError(f"Code execution exceeded {timeout}s timeout")

    if process.exitcode != 0 and result_queue.empty():
        raise RuntimeError(f"Process exited with code {process.exitcode}")

    if result_queue.empty():
        raise RuntimeError("No result returned from sandboxed execution")

    status, result = result_queue.get_nowait()
    if status == "error":
        raise RuntimeError(result)

    return result
