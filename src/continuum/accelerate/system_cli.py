from __future__ import annotations

import json
from pathlib import Path

import typer

from continuum.accelerate.system_formatter import render_status
from continuum.accelerate.system_state import load_state, save_state, utc_now
from continuum.accelerate.system_tuner import apply_acceleration, capture_previous_state, detect_context, restore_acceleration


def accelerate_command(
    on: bool = typer.Option(False, "--on", help="Enable acceleration mode."),
    off: bool = typer.Option(False, "--off", help="Restore previous system settings."),
    status: bool = typer.Option(False, "--status", help="Show current acceleration state."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be changed."),
    verbose: bool = typer.Option(False, "--verbose", help="Print detailed state/command information."),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Apply CPU optimizations only."),
    gpu_only: bool = typer.Option(False, "--gpu-only", help="Apply GPU optimizations only."),
) -> None:
    try:
        if cpu_only and gpu_only:
            typer.echo("Usage error: cannot combine --cpu-only and --gpu-only", err=True)
            raise typer.Exit(code=2)

        selected = sum(1 for flag in (on, off, status) if flag)
        if selected > 1:
            typer.echo("Usage error: use only one of --on, --off, --status", err=True)
            raise typer.Exit(code=2)

        if selected == 0:
            status = True

        ctx = detect_context()

        if status:
            current = load_state(Path.cwd())
            if current is None:
                payload = {
                    "active": False,
                    "platform": ctx["platform"],
                    "timestamp": utc_now(),
                    "changes_applied": [],
                    "previous_state": {},
                    "failures": [],
                }
                render_status(payload, verbose=verbose)
            else:
                render_status(current, verbose=verbose)
            raise typer.Exit(code=0)

        if on:
            previous_state = capture_previous_state(ctx, cpu_only=cpu_only, gpu_only=gpu_only)
            changes, failures = apply_acceleration(
                ctx,
                previous_state=previous_state,
                dry_run=dry_run,
                cpu_only=cpu_only,
                gpu_only=gpu_only,
            )

            payload = {
                "active": not dry_run,
                "platform": ctx["platform"],
                "timestamp": utc_now(),
                "changes_applied": changes,
                "previous_state": previous_state,
                "failures": failures,
                "mode": "dry-run" if dry_run else "on",
            }

            if not dry_run:
                save_state(payload, Path.cwd())

            if verbose:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), err=True)
            render_status(payload, verbose=verbose)
            raise typer.Exit(code=0)

        # off
        existing = load_state(Path.cwd())
        if existing is None:
            payload = {
                "active": False,
                "platform": ctx["platform"],
                "timestamp": utc_now(),
                "changes_applied": [],
                "previous_state": {},
                "failures": [],
                "mode": "off",
                "message": "No active acceleration state found.",
            }
            render_status(payload, verbose=verbose)
            raise typer.Exit(code=0)

        previous_state = existing.get("previous_state", {})
        changes, failures = restore_acceleration(ctx, previous_state=previous_state, dry_run=dry_run)

        payload = {
            "active": False,
            "platform": ctx["platform"],
            "timestamp": utc_now(),
            "changes_applied": changes,
            "previous_state": previous_state,
            "failures": failures,
            "mode": "dry-run" if dry_run else "off",
        }

        if not dry_run:
            save_state(payload, Path.cwd())

        if verbose:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), err=True)
        render_status(payload, verbose=verbose)
        raise typer.Exit(code=0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Accelerate critical failure: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=4)


__all__ = ["accelerate_command"]
