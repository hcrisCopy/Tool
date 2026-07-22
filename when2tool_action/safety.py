"""Explicit execution policy for model-selected benchmark tools.

The upstream CodeExecutor executes arbitrary model text in-process.  This module
never does so: only exact code snippets present in the benchmark instructions
are eligible, and those run in a resource-limited child process.  Other tools
remain the pinned implementations, with additional benchmark-range guards and
a wall-clock timeout.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n([\s\S]*?)```", re.IGNORECASE)


def normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def trusted_code_from_tasks(tasks: Iterable[dict[str, Any]]) -> frozenset[str]:
    snippets: set[str] = set()
    code_task_count = 0
    for task in tasks:
        if task.get("gold_env_name") != "CodeExecutorEnv":
            continue
        code_task_count += 1
        instruction = task.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(f"Code task {task.get('id')} has invalid instruction")
        blocks = [normalize_code(match) for match in CODE_BLOCK.findall(instruction)]
        blocks = [block for block in blocks if block]
        if not blocks:
            raise ValueError(f"Code task {task.get('id')} has no fenced code block")
        snippets.update(blocks)
    if code_task_count and not snippets:
        raise AssertionError("No trusted snippets extracted from code tasks")
    return frozenset(snippets)


def _json_shape(value: Any, *, depth: int = 0) -> int:
    if depth > 8:
        raise ValueError("Argument nesting exceeds depth 8")
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, int) and abs(value) > 10**12:
            raise ValueError("Integer argument exceeds 1e12")
        return 1
    if isinstance(value, str):
        if len(value) > 5000:
            raise ValueError("String argument exceeds 5000 characters")
        return 1
    if isinstance(value, list):
        if len(value) > 1000:
            raise ValueError("Array argument exceeds 1000 elements")
        total = 1 + sum(_json_shape(item, depth=depth + 1) for item in value)
        if total > 5000:
            raise ValueError("Nested argument exceeds 5000 cells")
        return total
    if isinstance(value, dict):
        if len(value) > 100:
            raise ValueError("Object argument exceeds 100 keys")
        total = 1
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 200:
                raise ValueError("Object keys must be short strings")
            total += _json_shape(item, depth=depth + 1)
        if total > 5000:
            raise ValueError("Nested argument exceeds 5000 cells")
        return total
    raise ValueError(f"Argument is not JSON-compatible: {type(value).__name__}")


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    if not isinstance(matrix, list) or not matrix:
        return (0, 0)
    if len(matrix) > 10 or not all(isinstance(row, list) for row in matrix):
        raise ValueError("Matrix must be a list of at most 10 rows")
    widths = {len(row) for row in matrix}
    if len(widths) != 1 or next(iter(widths)) > 10:
        raise ValueError("Matrix must be rectangular with at most 10 columns")
    for row in matrix:
        for cell in row:
            if not isinstance(cell, (int, float)) or isinstance(cell, bool):
                raise ValueError("Matrix cells must be numeric")
            if abs(cell) > 10**12:
                raise ValueError("Matrix cell magnitude exceeds 1e12")
    return len(matrix), next(iter(widths))


def validate_tool_arguments(tool_name: str, arguments: Any) -> tuple[bool, str | None]:
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"
    try:
        _json_shape(arguments)
        if tool_name in {"factorial"} and int(arguments.get("n", 0)) > 1000:
            raise ValueError("factorial n exceeds safety cap 1000")
        if tool_name in {"combination", "permutation"}:
            if int(arguments.get("n", 0)) > 10000:
                raise ValueError(f"{tool_name} n exceeds safety cap 10000")
        if tool_name == "nth_prime" and int(arguments.get("n", 0)) > 10000:
            raise ValueError("nth_prime n exceeds safety cap 10000")
        if tool_name in {"is_prime", "factorize"}:
            if abs(int(arguments.get("n", 0))) > 10**9:
                raise ValueError(f"{tool_name} n exceeds safety cap 1e9")
        if tool_name.startswith("matrix_"):
            for key in ("matrix", "matrix_a", "matrix_b"):
                if key in arguments:
                    _matrix_shape(arguments[key])
        if tool_name == "regex_match":
            if len(str(arguments.get("pattern", ""))) > 512:
                raise ValueError("regex pattern exceeds 512 characters")
            if len(str(arguments.get("text", ""))) > 5000:
                raise ValueError("regex text exceeds 5000 characters")
        if tool_name == "evaluate_expression":
            expression = arguments.get("expression", "")
            if not isinstance(expression, str) or len(expression) > 512:
                raise ValueError("calculator expression must be <=512 characters")
            tree = ast.parse(expression, mode="eval")
            nodes = list(ast.walk(tree))
            if len(nodes) > 128:
                raise ValueError("calculator AST exceeds 128 nodes")
            for node in nodes:
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                    if not isinstance(node.right, ast.Constant):
                        raise ValueError("calculator exponent must be a literal")
                    if abs(float(node.right.value)) > 10000:
                        raise ValueError("calculator exponent exceeds 10000")
    except (ValueError, TypeError, SyntaxError, OverflowError) as error:
        return False, str(error)
    return True, None


@contextmanager
def wall_clock_timeout(seconds: int):
    if os.name != "posix":
        raise RuntimeError("Tool timeout policy requires the Linux experiment server")

    def handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Tool execution exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _limit_child() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


def run_trusted_code(code: Any, trusted: frozenset[str]) -> dict[str, Any]:
    if not isinstance(code, str):
        return {"success": False, "message": "Code must be a string."}
    normalized = normalize_code(code)
    if normalized not in trusted:
        return {
            "success": False,
            "message": "[SAFETY_REJECTED] Code is not an exact benchmark snippet.",
        }
    if os.name != "posix":
        raise RuntimeError("Trusted code execution requires the Linux experiment server")
    with tempfile.TemporaryDirectory(prefix="calltool-code-") as directory:
        env = {
            "PATH": str(Path(sys.executable).parent),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
        }
        try:
            process = subprocess.run(
                [sys.executable, "-I", "-c", normalized],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                timeout=7,
                preexec_fn=_limit_child,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            partial = (error.stdout or "") if isinstance(error.stdout, str) else ""
            return {
                "success": False,
                "stdout": partial[:500],
                "stderr": "Code execution timed out.",
                "exit_code": -1,
                "message": "Code execution timed out.",
            }
    stdout = process.stdout.rstrip("\n")
    stderr = process.stderr.rstrip("\n")
    if len(stdout) > 5000:
        stdout = stdout[:5000] + "... [truncated]"
    if len(stderr) > 1000:
        stderr = stderr[:1000] + "... [truncated]"
    success = process.returncode == 0
    result: dict[str, Any] = {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": process.returncode,
    }
    if success:
        result["result"] = stdout
    else:
        result["message"] = stderr or "Code process failed."
    return result


def compact_json(value: Any, max_chars: int = 8000) -> Any:
    """Bound logged tool results without hiding that truncation occurred."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(payload) <= max_chars:
        return value
    return {
        "success": bool(value.get("success")) if isinstance(value, dict) else False,
        "truncated": True,
        "original_chars": len(payload),
        "preview": payload[:max_chars],
    }
