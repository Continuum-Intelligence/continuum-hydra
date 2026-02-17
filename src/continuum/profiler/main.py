from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

try:
    import typer
except Exception:  # pragma: no cover
    class _TyperShim:
        class Exit(Exception):
            def __init__(self, code: int = 0) -> None:
                self.code = code

        @staticmethod
        def Option(default, *args, **kwargs):  # noqa: ANN001, ANN003
            return default

        @staticmethod
        def echo(message: str, err: bool = False) -> None:
            print(message)

    typer = _TyperShim()  # type: ignore[assignment]

from continuum.profiler.formatters import build_profile_report, render_profile_human, write_profile_json
from continuum.profiler.cpu_benchmark import run_cpu_benchmark
from continuum.profiler.memory_bandwidth import run_memory_bandwidth
from continuum.profiler.static_profile import collect_static_profile

AVAILABLE_BENCHMARKS = {
    "static": collect_static_profile,
    "cpu": run_cpu_benchmark,
    "memory": run_memory_bandwidth,
}
_BENCHMARK_ORDER = ("static", "cpu", "memory")
_OUTPUT_FORMATS = {"human", "json", "both"}


def profile_command(
    benchmarks: str | None = typer.Option(
        None,
        "--benchmarks",
        help="Comma-separated benchmark keys to run: static,cpu,memory",
    ),
    static_only: bool = typer.Option(
        False,
        "--static-only",
        help="Run only static machine characterization and skip benchmarks.",
    ),
    no_static: bool = typer.Option(False, "--no-static", help="Exclude static profile section from output."),
    no_benchmarks: bool = typer.Option(False, "--no-benchmarks", help="Exclude benchmarks section from output."),
    output_format: str = typer.Option("human", "--output-format", help="Output format: human, json, or both."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress human output (equivalent to --output-format json)."),
    json_output: bool = typer.Option(False, "--json", help="Also print JSON report to stdout."),
    export: Path | None = typer.Option(None, "--export", help="Directory to write JSON report."),
    no_write: bool = typer.Option(False, "--no-write", help="Do not write JSON report to disk."),
    cpu_duration: float = typer.Option(8.0, "--cpu-duration", help="CPU benchmark measurement duration in seconds."),
    cpu_warmup: float = typer.Option(2.0, "--cpu-warmup", help="CPU benchmark warmup duration in seconds."),
    mem_duration: float = typer.Option(8.0, "--mem-duration", help="Memory benchmark measurement duration in seconds."),
    mem_warmup: float = typer.Option(2.0, "--mem-warmup", help="Memory benchmark warmup duration in seconds."),
    mem_mb: int | None = typer.Option(None, "--mem-mb", help="Memory benchmark buffer size in MB."),
    verbose: bool = typer.Option(False, "--verbose", help="Print traceback on unexpected profiler errors."),
) -> None:
    try:
        selected = _parse_selected_benchmarks(benchmarks)
        if static_only:
            selected = {"static"}

        context: dict[str, Any] = {
            "facts": {},
            "notes": [],
            "static_only": static_only,
            "cpu_duration": cpu_duration,
            "cpu_warmup": cpu_warmup,
            "mem_duration": mem_duration,
            "mem_warmup": mem_warmup,
            "mem_mb": mem_mb,
        }
        static_profile: dict[str, Any] = {}
        benchmarks_payload: dict[str, Any] = {}

        for name in _BENCHMARK_ORDER:
            if name not in selected:
                continue
            fn = AVAILABLE_BENCHMARKS[name]
            value = fn(context)
            if name == "static":
                if isinstance(value, dict):
                    static_profile = value
            elif isinstance(value, dict):
                benchmarks_payload.update(value)

        static_notes = static_profile.get("notes") if isinstance(static_profile, dict) else None
        if isinstance(static_notes, list):
            for note in context.get("notes", []):
                if isinstance(note, str) and note not in static_notes:
                    static_notes.append(note)

        if no_static:
            static_profile = {}
        if no_benchmarks:
            benchmarks_payload = {}

        report = build_profile_report(static_profile, benchmarks=benchmarks_payload)

        effective_output_format = _resolve_output_format(
            output_format=output_format,
            quiet=quiet,
            json_output=json_output,
        )

        if effective_output_format in {"human", "both"}:
            render_profile_human(report)
        if effective_output_format in {"json", "both"}:
            typer.echo(json.dumps(report, indent=2, ensure_ascii=False))

        if not no_write:
            output_dir = export if export is not None else Path(".hydra/reports")
            output = write_profile_json(report, output_dir)
            typer.echo(f"Profile JSON written: {output}")

        raise typer.Exit(code=0)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Profiler failed: {type(exc).__name__}: {exc}", err=True)
        if verbose:
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=4)


__all__ = ["profile_command"]


def _parse_selected_benchmarks(raw: str | None) -> set[str]:
    if raw is None or not raw.strip():
        return set(_BENCHMARK_ORDER)

    selected = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = selected - set(AVAILABLE_BENCHMARKS.keys())
    if unknown:
        bad = ", ".join(sorted(unknown))
        valid = ", ".join(_BENCHMARK_ORDER)
        raise ValueError(f"Unknown benchmark(s): {bad}. Valid values: {valid}")
    return selected


def _resolve_output_format(output_format: str, quiet: bool, json_output: bool) -> str:
    if quiet:
        return "json"

    normalized = (output_format or "human").strip().lower()
    if normalized not in _OUTPUT_FORMATS:
        raise ValueError("Unknown output format. Valid values: human, json, both")

    if json_output and normalized == "human":
        return "both"
    return normalized
