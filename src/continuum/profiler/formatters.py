from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def build_profile_report(
    static_profile: dict[str, Any],
    benchmark_results: list[dict[str, Any]] | None = None,
    benchmarks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "static_profile": static_profile,
        "benchmark_results": list(benchmark_results or []),
        "benchmarks": dict(benchmarks or {}),
    }


def write_profile_json(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"profile_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def render_profile_human(report: dict[str, Any], console: Console | None = None) -> None:
    if Console is None or Table is None:
        _render_profile_compact(report)
        return

    active_console = console or Console()
    rows = _build_status_rows(report)
    if rows:
        table = Table(title="Continuum Profile Report")
        table.add_column("Status", no_wrap=True)
        table.add_column("Item", overflow="fold")
        table.add_column("Result", overflow="fold")
        table.add_column("Benchmark", overflow="fold")
        for row in rows:
            table.add_row(_style_status(row["status"]), row["item"], row["result"], row["benchmark"])
        active_console.print(table)
    _render_cpu_sustained_rich(report, active_console)

    static = report.get("static_profile", {}) if isinstance(report, dict) else {}
    notes = static.get("notes") if isinstance(static, dict) else None
    if isinstance(notes, list) and notes:
        active_console.print("Notes:")
        for note in notes:
            active_console.print(f"- {note}")


def _render_profile_compact(report: dict[str, Any]) -> None:
    rows = _build_status_rows(report)
    if rows:
        print("Continuum Profile Report")
        print("STATUS ITEM RESULT BENCHMARK")
        for row in rows:
            print(f"[{row['status']}] {row['item']} | {row['result']} | {row['benchmark']}")
    _render_cpu_sustained_compact(report)

    static = report.get("static_profile", {}) if isinstance(report, dict) else {}
    notes = static.get("notes") if isinstance(static, dict) else None
    if isinstance(notes, list) and notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


def _style_status(status: str) -> str:
    if status == "PASS":
        return "[green][PASS][/green]"
    if status == "WARN":
        return "[yellow][WARN][/yellow]"
    if status == "FAIL":
        return "[red][FAIL][/red]"
    return f"[{status}]"


def _status_for_value(value: Any) -> str:
    return "PASS" if value is not None else "WARN"


def _build_status_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    static = report.get("static_profile", {}) if isinstance(report, dict) else {}

    def _section(key: str) -> dict[str, Any]:
        value = static.get(key) if isinstance(static, dict) else None
        return value if isinstance(value, dict) else {}

    rows: list[dict[str, str]] = []
    if isinstance(static, dict) and static:
        cpu = _section("cpu")
        memory = _section("memory")
        storage = _section("storage")
        os_info = _section("os")
        runtime = _section("runtime")

        fields = [
            ("cpu.model", cpu.get("model")),
            ("cpu.cores_physical", cpu.get("cores_physical")),
            ("cpu.cores_logical", cpu.get("cores_logical")),
            ("cpu.arch", cpu.get("arch")),
            ("memory.total_bytes", memory.get("total_bytes")),
            ("storage.root_mount", storage.get("root_mount")),
            ("storage.root_device", storage.get("root_device")),
            ("storage.filesystem_type", storage.get("filesystem_type")),
            ("storage.is_nvme", storage.get("is_nvme")),
            ("storage.is_ssd", storage.get("is_ssd")),
            ("os.name", os_info.get("name")),
            ("os.version", os_info.get("version")),
            ("os.kernel", os_info.get("kernel")),
            ("runtime.python_version", runtime.get("python_version")),
            ("runtime.torch_version", runtime.get("torch_version")),
            ("runtime.torch_cuda_available", runtime.get("torch_cuda_available")),
            ("runtime.torch_cuda_version", runtime.get("torch_cuda_version")),
            ("runtime.platform", runtime.get("platform")),
        ]

        for item, value in fields:
            rows.append(
                {
                    "status": _status_for_value(value),
                    "item": item,
                    "result": "null" if value is None else str(value),
                    "benchmark": "-",
                }
            )

    benchmark_results = report.get("benchmark_results") if isinstance(report, dict) else None
    if isinstance(benchmark_results, list):
        for benchmark in benchmark_results:
            if not isinstance(benchmark, dict):
                continue
            status = str(benchmark.get("status", "WARN")).upper()
            result_value = benchmark.get("result")
            unit = benchmark.get("unit")
            message = benchmark.get("message")
            result_text = "null" if result_value is None else str(result_value)
            if unit:
                result_text = f"{result_text} {unit}"
            if message:
                result_text = f"{result_text} ({message})"
            rows.append(
                {
                    "status": status if status in {"PASS", "WARN", "FAIL"} else "WARN",
                    "item": str(benchmark.get("name", "benchmark.unknown")),
                    "result": result_text,
                    "benchmark": str(benchmark.get("name", "benchmark.unknown")),
                }
            )

    benchmarks = report.get("benchmarks") if isinstance(report, dict) else None
    cpu_sustained = benchmarks.get("cpu_sustained") if isinstance(benchmarks, dict) else None
    if isinstance(cpu_sustained, dict):
        cpu_fields = [
            ("benchmarks.cpu_sustained.mean_iter_per_sec", cpu_sustained.get("mean_iter_per_sec")),
            ("benchmarks.cpu_sustained.p95_iter_per_sec", cpu_sustained.get("p95_iter_per_sec")),
            ("benchmarks.cpu_sustained.std_iter_per_sec", cpu_sustained.get("std_iter_per_sec")),
            ("benchmarks.cpu_sustained.iterations", cpu_sustained.get("iterations")),
            ("benchmarks.cpu_sustained.duration_sec", cpu_sustained.get("duration_sec")),
        ]
        for item, value in cpu_fields:
            rows.append(
                {
                    "status": _status_for_value(value),
                    "item": item,
                    "result": "null" if value is None else str(value),
                    "benchmark": "cpu_sustained",
                }
            )

    return rows


def _render_cpu_sustained_rich(report: dict[str, Any], console: Console) -> None:
    benchmarks = report.get("benchmarks") if isinstance(report, dict) else None
    cpu = benchmarks.get("cpu_sustained") if isinstance(benchmarks, dict) else None
    if not isinstance(cpu, dict):
        return

    section = Table(title="CPU Sustained")
    section.add_column("Metric", overflow="fold")
    section.add_column("Value", overflow="fold")
    section.add_row("mean_iter_per_sec", "null" if cpu.get("mean_iter_per_sec") is None else str(cpu.get("mean_iter_per_sec")))
    section.add_row("p95_iter_per_sec", "null" if cpu.get("p95_iter_per_sec") is None else str(cpu.get("p95_iter_per_sec")))
    section.add_row("std_iter_per_sec", "null" if cpu.get("std_iter_per_sec") is None else str(cpu.get("std_iter_per_sec")))
    console.print(section)


def _render_cpu_sustained_compact(report: dict[str, Any]) -> None:
    benchmarks = report.get("benchmarks") if isinstance(report, dict) else None
    cpu = benchmarks.get("cpu_sustained") if isinstance(benchmarks, dict) else None
    if not isinstance(cpu, dict):
        return

    print("CPU Sustained:")
    print(f"mean_iter_per_sec: {'null' if cpu.get('mean_iter_per_sec') is None else cpu.get('mean_iter_per_sec')}")
    print(f"p95_iter_per_sec: {'null' if cpu.get('p95_iter_per_sec') is None else cpu.get('p95_iter_per_sec')}")
    print(f"std_iter_per_sec: {'null' if cpu.get('std_iter_per_sec') is None else cpu.get('std_iter_per_sec')}")


__all__ = [
    "build_profile_report",
    "write_profile_json",
    "render_profile_human",
]
