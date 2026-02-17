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
        table = Table(title="Continuum Profile Report (Static + Benchmarks + Analysis + Remediation)")
        table.add_column("Status", no_wrap=True)
        table.add_column("Section", overflow="fold")
        table.add_column("Item", overflow="fold")
        table.add_column("Result", overflow="fold")
        for row in rows:
            table.add_row(
                _style_status(row["status"]),
                row["section"],
                row["item"],
                row["result"],
            )
        active_console.print(table)
    _render_summary_details_rich(report, active_console)

    static = report.get("static_profile", {}) if isinstance(report, dict) else {}
    notes = static.get("notes") if isinstance(static, dict) else None
    if isinstance(notes, list) and notes:
        active_console.print("Notes:")
        for note in notes:
            active_console.print(f"- {note}")


def _render_profile_compact(report: dict[str, Any]) -> None:
    rows = _build_status_rows(report)
    if rows:
        print("Continuum Profile Report (Static + Benchmarks + Analysis + Remediation)")
        print("STATUS ITEM SECTION RESULT")
        for row in rows:
            print(f"[{row['status']}] {row['item']} | {row['section']} | {row['result']}")
    _render_summary_details_compact(report)

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
                    "section": "Static Profile",
                    "item": item,
                    "result": "null" if value is None else str(value),
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
                    "section": "Legacy Benchmarks",
                    "item": str(benchmark.get("name", "benchmark.unknown")),
                    "result": result_text,
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
                    "section": "CPU Sustained",
                    "item": item,
                    "result": "null" if value is None else str(value),
                }
            )

    memory_bandwidth = benchmarks.get("memory_bandwidth") if isinstance(benchmarks, dict) else None
    if isinstance(memory_bandwidth, dict):
        mem_fields = [
            ("benchmarks.memory_bandwidth.mean_gbps", memory_bandwidth.get("mean_gbps")),
            ("benchmarks.memory_bandwidth.p95_gbps", memory_bandwidth.get("p95_gbps")),
            ("benchmarks.memory_bandwidth.std_gbps", memory_bandwidth.get("std_gbps")),
            ("benchmarks.memory_bandwidth.iterations", memory_bandwidth.get("iterations")),
            ("benchmarks.memory_bandwidth.duration_sec", memory_bandwidth.get("duration_sec")),
        ]
        for item, value in mem_fields:
            rows.append(
                {
                    "status": _status_for_value(value),
                    "section": "Memory Bandwidth",
                    "item": item,
                    "result": "null" if value is None else str(value),
                }
            )

    gpu_sustained = benchmarks.get("gpu_sustained") if isinstance(benchmarks, dict) else None
    if isinstance(gpu_sustained, dict):
        gpu_fields = [
            ("benchmarks.gpu_sustained.mean_iter_per_sec", gpu_sustained.get("mean_iter_per_sec")),
            ("benchmarks.gpu_sustained.p95_iter_per_sec", gpu_sustained.get("p95_iter_per_sec")),
            ("benchmarks.gpu_sustained.std_iter_per_sec", gpu_sustained.get("std_iter_per_sec")),
            ("benchmarks.gpu_sustained.backend", gpu_sustained.get("backend")),
            ("benchmarks.gpu_sustained.dtype", gpu_sustained.get("dtype")),
        ]
        for item, value in gpu_fields:
            rows.append(
                {
                    "status": _status_for_value(value),
                    "section": "GPU Sustained",
                    "item": item,
                    "result": "null" if value is None else str(value),
                }
            )

    disk_random_io = benchmarks.get("disk_random_io") if isinstance(benchmarks, dict) else None
    if isinstance(disk_random_io, dict):
        disk_fields = [
            ("benchmarks.disk_random_io.mean_read_mb_s", disk_random_io.get("mean_read_mb_s")),
            ("benchmarks.disk_random_io.p95_read_mb_s", disk_random_io.get("p95_read_mb_s")),
            ("benchmarks.disk_random_io.std_read_mb_s", disk_random_io.get("std_read_mb_s")),
            ("benchmarks.disk_random_io.mean_iops", disk_random_io.get("mean_iops")),
            ("benchmarks.disk_random_io.iterations", disk_random_io.get("iterations")),
            ("benchmarks.disk_random_io.duration_sec", disk_random_io.get("duration_sec")),
        ]
        for item, value in disk_fields:
            rows.append(
                {
                    "status": _status_for_value(value),
                    "section": "Disk Random I/O",
                    "item": item,
                    "result": "null" if value is None else str(value),
                }
            )

    analysis = report.get("analysis") if isinstance(report, dict) else None
    if isinstance(analysis, dict):
        primary = analysis.get("primary_bottleneck")
        confidence = analysis.get("confidence")
        rows.append(
            {
                "status": _status_for_value(primary),
                "section": "Analysis",
                "item": "analysis.primary_bottleneck",
                "result": "null" if primary is None else str(primary),
            }
        )
        rows.append(
            {
                "status": _status_for_value(confidence),
                "section": "Analysis",
                "item": "analysis.confidence",
                "result": "null" if confidence is None else str(confidence),
            }
        )

    remediation = report.get("remediation") if isinstance(report, dict) else None
    if isinstance(remediation, dict):
        priority = remediation.get("priority")
        rows.append(
            {
                "status": _status_for_value(priority),
                "section": "Remediation",
                "item": "remediation.priority",
                "result": "null" if priority is None else str(priority),
            }
        )

    return rows


def _render_summary_details_rich(report: dict[str, Any], console: Console) -> None:
    analysis = report.get("analysis") if isinstance(report, dict) else None
    remediation = report.get("remediation") if isinstance(report, dict) else None
    details = Table(title="Summary Details")
    details.add_column("Section", overflow="fold")
    details.add_column("Detail", overflow="fold")
    details.add_column("Value", overflow="fold")
    if isinstance(analysis, dict):
        details.add_row("Analysis", "primary_bottleneck", "null" if analysis.get("primary_bottleneck") is None else str(analysis.get("primary_bottleneck")))
        details.add_row("Analysis", "secondary_bottleneck", "null" if analysis.get("secondary_bottleneck") is None else str(analysis.get("secondary_bottleneck")))
        details.add_row("Analysis", "confidence", "null" if analysis.get("confidence") is None else str(analysis.get("confidence")))
    if isinstance(remediation, dict):
        actions = remediation.get("actions")
        details.add_row("Remediation", "priority", "null" if remediation.get("priority") is None else str(remediation.get("priority")))
        details.add_row("Remediation", "actions", str(len(actions) if isinstance(actions, list) else 0))
    if details.row_count > 0:
        console.print(details)

    if isinstance(analysis, dict):
        reasons = analysis.get("reasons")
        if isinstance(reasons, list) and reasons:
            console.print("Top Reasons:")
            for reason in reasons[:3]:
                console.print(f"- {reason}")

        recommendations = analysis.get("recommendations")
        if isinstance(recommendations, list) and recommendations:
            console.print("Recommendations:")
            for rec in recommendations[:3]:
                console.print(f"- {rec}")

    if isinstance(remediation, dict):
        actions = remediation.get("actions")
        if isinstance(actions, list) and actions:
            console.print("Top Actions:")
            for action in actions[:3]:
                if not isinstance(action, dict):
                    continue
                title = action.get("title")
                impact = action.get("impact")
                difficulty = action.get("difficulty")
                reason = action.get("reason")
                console.print(
                    f"- {title} "
                    f"(impact={impact}, difficulty={difficulty})"
                    f"{': ' + str(reason) if reason else ''}"
                )


def _render_summary_details_compact(report: dict[str, Any]) -> None:
    analysis = report.get("analysis") if isinstance(report, dict) else None
    remediation = report.get("remediation") if isinstance(report, dict) else None
    if not isinstance(analysis, dict) and not isinstance(remediation, dict):
        return

    print("Summary Details:")
    if isinstance(analysis, dict):
        print(f"analysis.primary_bottleneck: {'null' if analysis.get('primary_bottleneck') is None else analysis.get('primary_bottleneck')}")
        print(f"analysis.secondary_bottleneck: {'null' if analysis.get('secondary_bottleneck') is None else analysis.get('secondary_bottleneck')}")
        print(f"analysis.confidence: {'null' if analysis.get('confidence') is None else analysis.get('confidence')}")
        reasons = analysis.get("reasons")
        if isinstance(reasons, list) and reasons:
            print("Top Reasons:")
            for reason in reasons[:3]:
                print(f"- {reason}")
        recommendations = analysis.get("recommendations")
        if isinstance(recommendations, list) and recommendations:
            print("Recommendations:")
            for rec in recommendations[:3]:
                print(f"- {rec}")

    if isinstance(remediation, dict):
        print(f"remediation.priority: {'null' if remediation.get('priority') is None else remediation.get('priority')}")
        actions = remediation.get("actions")
        if isinstance(actions, list) and actions:
            print("Top Actions:")
            for action in actions[:3]:
                if not isinstance(action, dict):
                    continue
                title = action.get("title")
                impact = action.get("impact")
                difficulty = action.get("difficulty")
                reason = action.get("reason")
                reason_text = f": {reason}" if reason else ""
                print(f"- {title} (impact={impact}, difficulty={difficulty}){reason_text}")


__all__ = [
    "build_profile_report",
    "write_profile_json",
    "render_profile_human",
]
