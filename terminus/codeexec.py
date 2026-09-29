"""Isolated, resource-limited execution for code-task scoring (HumanEval/MBPP).

Runs candidate code plus assertions in a subprocess with a wall-clock timeout,
CPU-time and address-space rlimits, an empty environment, and a temporary
working directory. This is a containment measure for scoring, not a security
boundary: run code tasks only on trusted evaluation records, and only when the
caller explicitly enables it.

Note: RLIMIT_AS is not enforced by macOS; the memory limit applies on Linux.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile


def _limits(cpu_seconds, memory_bytes):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, OSError):
            pass  # not enforced on every platform
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        os.chdir(tempfile.gettempdir())
    return apply


def run_program(source, *, timeout_seconds=5, cpu_seconds=5, memory_mb=512, python=None):
    """Run a Python program; return {"status", "stdout", "stderr"}.

    status: "passed" | "failed" | "timeout" | "error".
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        completed = subprocess.run(
            [python or sys.executable, path],
            capture_output=True, text=True, timeout=timeout_seconds,
            env={"PATH": os.environ.get("PATH", "")}, cwd=tempfile.gettempdir(),
            preexec_fn=_limits(cpu_seconds, memory_mb * 1024 * 1024))
        return {"status": "passed" if completed.returncode == 0 else "failed",
                "stdout": completed.stdout[-4096:], "stderr": completed.stderr[-4096:],
                "returncode": completed.returncode}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": "", "returncode": None}
    finally:
        os.unlink(path)


def score_code_item(candidate_code, test_code, **kwargs):
    """HumanEval/MBPP style: candidate definitions plus assertion tests."""
    result = run_program(candidate_code + "\n\n" + test_code + "\n", **kwargs)
    return {"passed": result["status"] == "passed", **result}
