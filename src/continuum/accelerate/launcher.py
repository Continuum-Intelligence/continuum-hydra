from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from continuum.accelerate.reporting import write_json

_CHECKPOINT_PATTERNS = ("*.ckpt", "*.pt", "*.pth", "*.safetensors")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_run_id() -> str:
    return f"launch-run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def _scan_checkpoints(cwd: Path) -> Path | None:
    # Restrict checkpoint discovery to training artifact directories.
    # This avoids false matches from environment files like *.pth in .venv.
    roots = [cwd / "checkpoints", cwd / "outputs", cwd / "runs"]
    candidates: list[Path] = []

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in _CHECKPOINT_PATTERNS:
            candidates.extend(path for path in root.rglob(pattern) if path.is_file())

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _infer_resume_args(script_args: list[str], checkpoint: Path | None) -> tuple[list[str], str]:
    if checkpoint is None:
        return list(script_args), "no checkpoint discovered"

    known_resume_flags = {
        "--resume",
        "--resume-from",
        "--checkpoint",
        "--checkpoint-path",
        "--ckpt",
        "--ckpt_path",
    }
    if any(arg in known_resume_flags for arg in script_args):
        return list(script_args), "resume flag already supplied"

    return [*script_args, "--resume", str(checkpoint)], "appended --resume <checkpoint>"


def _stderr_print(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr)


def _terminate_process(process: subprocess.Popen[str], quiet: bool) -> None:
    if process.poll() is not None:
        return
    _stderr_print("[launch] interrupt received; sending SIGINT to child", quiet)
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        return
    except Exception:  # noqa: BLE001
        pass
    if process.poll() is not None:
        return

    _stderr_print("[launch] child still running; sending SIGTERM", quiet)
    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except Exception:  # noqa: BLE001
        pass
    if process.poll() is not None:
        return

    _stderr_print("[launch] child still running; sending SIGKILL", quiet)
    process.kill()
    process.wait(timeout=5)


def _run_once(
    script: Path,
    script_args: list[str],
    cwd: Path,
    log_path: Path,
    quiet: bool,
    verbose: bool,
    known_checkpoint: Path | None,
) -> tuple[int, dict[str, Any], Path | None]:
    command = [sys.executable, "-u", str(script), *script_args]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if known_checkpoint is not None:
        env["CONTINUUM_LAUNCH_RESUME_CHECKPOINT"] = str(known_checkpoint)

    started = _utc_now()
    started_mono = monotonic()
    checkpoint_seen = known_checkpoint
    recent_lines: deque[str] = deque(maxlen=200)
    checkpoint_poll_mono = monotonic()

    _stderr_print(f"[launch] starting: {' '.join(command)}", quiet)

    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
        bufsize=1,
        env=env,
    )

    try:
        with log_path.open("a", encoding="utf-8") as handle:
            if process.stdout is not None:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    recent_lines.append(line.rstrip("\n"))
                    if not quiet:
                        print(line, end="", file=sys.stderr)

                    now = monotonic()
                    if now - checkpoint_poll_mono >= 5.0:
                        checkpoint_poll_mono = now
                        discovered = _scan_checkpoints(cwd)
                        if discovered is not None and (checkpoint_seen is None or str(discovered) != str(checkpoint_seen)):
                            checkpoint_seen = discovered
                            if verbose:
                                _stderr_print(f"[launch] checkpoint discovered: {discovered}", quiet=False)

            return_code = process.wait()
    except KeyboardInterrupt:
        _terminate_process(process, quiet=quiet)
        raise

    ended = _utc_now()
    duration = round(monotonic() - started_mono, 3)

    attempt_report = {
        "started_at": started,
        "ended_at": ended,
        "duration_seconds": duration,
        "command_argv": command,
        "return_code": return_code,
        "checkpoint_seen": str(checkpoint_seen) if checkpoint_seen else None,
        "stdout_tail": list(recent_lines)[-40:],
    }

    return return_code, attempt_report, checkpoint_seen


def launch_training_script(
    script: Path,
    script_args: list[str],
    cwd: Path,
    max_restarts: int,
    auto_resume: bool,
    quiet: bool,
    verbose: bool,
    json_output: bool,
    out: Path | None,
    no_state_write: bool,
    dry_run: bool,
    debug: bool = False,
) -> tuple[int, dict[str, Any]]:
    run_id = _build_run_id()
    run_dir = cwd / ".hydra" / "launch" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "launch.log"

    base_command_argv = [sys.executable, "-u", str(script), *script_args]
    if debug:
        _stderr_print(f"[launch][debug] command_argv={base_command_argv!r}", quiet=False)
        _stderr_print(f"[launch][debug] script_args={script_args!r}", quiet=False)

    if dry_run:
        report = {
            "schema_version": "launch.runtime.v1",
            "run_id": run_id,
            "mode": "dry-run",
            "script": str(script),
            "script_args": list(script_args),
            "command_argv": list(base_command_argv),
            "status": "dry-run",
            "attempts": [],
            "restarts_used": 0,
            "max_restarts": max_restarts,
            "latest_checkpoint": str(_scan_checkpoints(cwd)) if _scan_checkpoints(cwd) else None,
            "log_path": str(log_path),
            "error": None,
            "exit_code": 0,
        }
        if not no_state_write:
            write_json(cwd / ".hydra" / "state" / "launch_latest.json", report)
        if out is not None:
            write_json(out, report)
        if json_output:
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0, report

    attempts: list[dict[str, Any]] = []
    restarts_used = 0
    current_args = list(script_args)
    latest_checkpoint = _scan_checkpoints(cwd)

    status = "failed"
    error: str | None = None

    interrupted = False
    try:
        while True:
            return_code, attempt_report, latest_checkpoint = _run_once(
                script=script,
                script_args=current_args,
                cwd=cwd,
                log_path=log_path,
                quiet=quiet,
                verbose=verbose,
                known_checkpoint=latest_checkpoint,
            )
            attempts.append(attempt_report)
            latest_checkpoint = _scan_checkpoints(cwd) or latest_checkpoint

            if return_code == 0:
                status = "completed"
                break

            if not auto_resume or restarts_used >= max_restarts:
                status = "failed"
                error = f"training process exited with code {return_code}"
                break

            resume_args, resume_note = _infer_resume_args(script_args, latest_checkpoint)
            if resume_args == script_args:
                status = "failed"
                error = f"training process exited with code {return_code}; resume unavailable ({resume_note})"
                break

            restarts_used += 1
            current_args = resume_args
            _stderr_print(
                f"[launch] auto-resume attempt {restarts_used}/{max_restarts}: {resume_note}",
                quiet,
            )
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        error = "Interrupted by user"

    exit_code = 130 if interrupted else (0 if status == "completed" else 1)
    report = {
        "schema_version": "launch.runtime.v1",
        "run_id": run_id,
        "mode": "apply",
        "script": str(script),
        "script_args": list(script_args),
        "command_argv": list(base_command_argv),
        "status": status,
        "attempts": attempts,
        "restarts_used": restarts_used,
        "max_restarts": max_restarts,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "log_path": str(log_path),
        "error": error,
        "exit_code": exit_code,
    }

    write_json(run_dir / "report.json", report)
    if not no_state_write:
        write_json(cwd / ".hydra" / "state" / "launch_latest.json", report)
    if out is not None:
        write_json(out, report)

    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    elif error is not None:
        _stderr_print(f"[launch] error: {error}", quiet)

    return exit_code, report


__all__ = ["launch_training_script"]
