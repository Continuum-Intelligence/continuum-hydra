from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import typer

from continuum.profiler.benchmarks import run_benchmarks
from continuum.profiler.formatters import build_profile_report, render_profile_human, write_profile_json
from continuum.profiler.static_profile import collect_static_profile


def profile_command(
    static_only: bool = typer.Option(
        False,
        "--static-only",
        help="Run only static machine characterization and skip benchmarks.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Also print JSON report to stdout."),
    export: Path | None = typer.Option(None, "--export", help="Directory to write JSON report."),
    no_write: bool = typer.Option(False, "--no-write", help="Do not write JSON report to disk."),
    verbose: bool = typer.Option(False, "--verbose", help="Print traceback on unexpected profiler errors."),
) -> None:
    try:
        context: dict[str, Any] = {"facts": {}, "static_only": static_only}
        static_profile = collect_static_profile(context)
        benchmark_results = run_benchmarks(static_only=static_only)

        report = build_profile_report(static_profile, benchmark_results)

        render_profile_human(report)

        if json_output:
            typer.echo(json.dumps(report, indent=2, ensure_ascii=False))

        if not no_write:
            output_dir = export if export is not None else Path(".hydra/reports")
            output = write_profile_json(report, output_dir)
            typer.echo(f"Profile JSON written: {output}")

        raise typer.Exit(code=0)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Profiler failed: {type(exc).__name__}: {exc}", err=True)
        if verbose:
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=4)


__all__ = ["profile_command"]
