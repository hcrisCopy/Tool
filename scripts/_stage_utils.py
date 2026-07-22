"""Shared helpers for the Stage 5--8 Python launchers."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("when2tool_action/configs/qwen3_4b_instruct_2507.yaml")
DEFAULT_RUN_ROOT = Path(
    "../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507"
)


def repo_path(path: Path | str) -> Path:
    """Resolve a CLI path relative to the repository root."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def positive_int(value: str) -> int:
    """Argparse type for strictly positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def require_files(paths: Iterable[Path], *, context: str) -> None:
    """Fail once with every missing required file listed."""

    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing {context} file(s):\n{rendered}")


def require_directories(paths: Iterable[Path], *, context: str) -> None:
    """Fail once with every missing required directory listed."""

    missing = [path for path in paths if not path.is_dir()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing {context} directory/directories:\n{rendered}")


def require_unique(values: Sequence[object], *, name: str) -> None:
    """Reject duplicate panel entries before any experiment starts."""

    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate values: {list(values)}")


class StageRunner:
    """Run package modules with this interpreter and tee output to a stage log."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def message(self, value: str) -> None:
        print(value, flush=True)
        with self.log_path.open("a", encoding="utf-8", newline="") as log:
            log.write(value + "\n")

    def run_module(self, module: str, *arguments: object) -> None:
        command = [
            sys.executable,
            "-m",
            module,
            *(str(argument) for argument in arguments),
        ]
        self.message(f"$ {shlex.join(command)}")

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with self.log_path.open("a", encoding="utf-8", newline="") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("Subprocess stdout pipe was not created")
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def audit_provenance(runner: StageRunner, *, config: Path, output: Path) -> None:
    """Create or validate the stage-specific runtime provenance file."""

    runner.run_module(
        "when2tool_action.scripts.audit_provenance",
        "--config",
        config,
        "--output",
        output,
    )
