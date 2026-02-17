from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from continuum.accelerate.models import ACCELERATE_SCHEMA_VERSION, AccelerationActionResult, AccelerationPlan, ExecutionContext
from continuum.accelerate.plugins.loader import PluginLoadResult


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_state_report(report: dict[str, Any], out: Path | None = None, cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    state_dir = base / ".hydra" / "state"
    latest_path = state_dir / "launch_latest.json"
    write_json(latest_path, report)
    if out is not None:
        write_json(out, report)
    return latest_path


def build_report(
    plan: AccelerationPlan,
    action_results: list[AccelerationActionResult],
    ctx: ExecutionContext,
    selected_action_ids: set[str],
    dry_run: bool,
    plugin_result: PluginLoadResult,
    hook_warnings: list[str] | None = None,
) -> dict[str, Any]:
    sorted_results = sorted(action_results, key=lambda result: result.action_id)
    applied_count = sum(1 for result in sorted_results if result.applied)
    unsupported_count = sum(1 for result in sorted_results if not result.supported)
    skipped_count = sum(1 for result in sorted_results if not result.applied and result.skipped_reason is not None)

    return {
        "schema_version": ACCELERATE_SCHEMA_VERSION,
        "mode": "dry-run" if dry_run else "apply",
        "plan": plan.to_dict(),
        "context": ctx.to_dict(),
        "selected_action_ids": sorted(selected_action_ids),
        "summary": {
            "applied": applied_count,
            "skipped": skipped_count,
            "unsupported": unsupported_count,
            "total": len(action_results),
        },
        "results": [result.to_dict() for result in sorted_results],
        "plugin_summary": {
            "actions_loaded": plugin_result.actions_loaded,
            "loaded_files": list(plugin_result.loaded_files),
            "failures": list(plugin_result.failures),
            "pre_apply_shell": [str(path) for path in plugin_result.hooks.pre_apply_shell],
            "post_apply_shell": [str(path) for path in plugin_result.hooks.post_apply_shell],
            "pre_apply_py_count": len(plugin_result.hooks.pre_apply_py),
            "post_apply_py_count": len(plugin_result.hooks.post_apply_py),
        },
        "warnings": list(plan.warnings) + list(hook_warnings or []),
    }


def render_summary(report: dict[str, Any], console: Console | None = None) -> None:
    active_console = console or Console()
    summary = report.get("summary", {})
    active_console.print(
        "Launch Summary: "
        f"Applied={summary.get('applied', 0)} "
        f"Skipped={summary.get('skipped', 0)} "
        f"Unsupported={summary.get('unsupported', 0)}"
    )

    for result in report.get("results", []):
        status = "APPLIED" if result.get("applied") else "SKIPPED"
        reason = result.get("skipped_reason")
        suffix = f" ({reason})" if reason else ""
        active_console.print(f"- {result.get('action_id')}: {status}{suffix}")


__all__ = [
    "write_json",
    "write_state_report",
    "build_report",
    "render_summary",
]
