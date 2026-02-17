from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from continuum.accelerate.actions import register_builtin_actions
from continuum.accelerate.models import ActionDescriptor, AccelerationAction, AccelerationPlan, ExecutionContext, normalize_profile
from continuum.accelerate.plugins.loader import PluginLoadResult, load_plugins
from continuum.accelerate.registry import clear_registry, filter_actions, get_actions, register_action


def _load_doctor_facts(cwd: Path) -> dict[str, Any] | None:
    state_candidate = cwd / ".hydra" / "state" / "doctor_latest.json"
    if state_candidate.exists():
        try:
            return json.loads(state_candidate.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    reports_dir = cwd / ".hydra" / "reports"
    if reports_dir.exists() and reports_dir.is_dir():
        candidates = sorted(reports_dir.glob("doctor_*.json"), reverse=True)
        if candidates:
            try:
                return json.loads(candidates[0].read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None

    return None


def build_context(cwd: Path | None = None) -> ExecutionContext:
    base = cwd if cwd is not None else Path.cwd()
    os_name = platform.system().lower()
    return ExecutionContext(
        os_name=os_name,
        is_linux=os_name == "linux",
        is_windows=os_name == "windows",
        is_macos=os_name == "darwin",
        user_is_root=(hasattr(os, "geteuid") and os.geteuid() == 0),
        has_nvidia_smi=shutil.which("nvidia-smi") is not None,
        doctor_facts=_load_doctor_facts(base),
        env=dict(os.environ),
        cwd=str(base),
        repo_root=str(base),
    )


def build_plan(
    profile: str,
    only: set[str] | None,
    exclude: set[str] | None,
    expert_mode: bool = False,
    include_timestamp: bool = True,
    cwd: Path | None = None,
) -> tuple[AccelerationPlan, list[dict[str, Any]], ExecutionContext, PluginLoadResult]:
    base = cwd if cwd is not None else Path.cwd()
    ctx = build_context(base)
    normalized_profile = normalize_profile(profile)

    clear_registry()
    register_builtin_actions()

    plugin_result = load_plugins(register_action, cwd=base)

    all_actions = get_actions()
    filtered_actions = filter_actions(
        actions=all_actions,
        only=only,
        exclude=exclude,
        profile=normalized_profile,
        categories=None,
    )

    descriptors: list[ActionDescriptor] = []
    internal_data: list[dict[str, Any]] = []

    runtime_ctx = ExecutionContext(
        os_name=ctx.os_name,
        is_linux=ctx.is_linux,
        is_windows=ctx.is_windows,
        is_macos=ctx.is_macos,
        user_is_root=ctx.user_is_root,
        has_nvidia_smi=ctx.has_nvidia_smi,
        doctor_facts=ctx.doctor_facts,
        env={**ctx.env, "ACCELERATE_PROFILE": normalized_profile},
        cwd=ctx.cwd,
        repo_root=ctx.repo_root,
    )

    for action in filtered_actions:
        supported = False
        before: dict[str, Any] = {}
        check_notes: list[str] = []
        plan_notes: list[str] = []
        commands: list[str] = []
        after_preview: dict[str, Any] = {}
        recommended = False

        try:
            supported, before, check_notes = action.check(runtime_ctx)
            if supported:
                recommended, commands, after_preview, plan_notes = action.plan(runtime_ctx)
        except Exception as exc:  # noqa: BLE001
            supported = False
            check_notes = [f"{type(exc).__name__}: {exc}"]

        if action.risk.lower() == "high" and not expert_mode:
            recommended = False
            plan_notes.append("High risk action is disabled unless expert profile is used")

        descriptors.append(
            ActionDescriptor(
                action_id=action.id,
                title=action.title,
                category=action.category,
                recommended=recommended,
                risk=action.risk,
                requires_root=action.requires_root,
                supported=supported,
                why=action.why,
                commands=commands,
            )
        )

        internal_data.append(
            {
                "action": action,
                "supported": supported,
                "recommended": recommended,
                "before": before,
                "after_preview": after_preview,
                "check_notes": check_notes,
                "plan_notes": plan_notes,
                "commands": commands,
            }
        )

    plan = AccelerationPlan.create(
        profile=normalized_profile,
        recommendations=descriptors,
        warnings=plugin_result.warnings,
        include_timestamp=include_timestamp,
    )

    return plan, internal_data, runtime_ctx, plugin_result


__all__ = ["build_context", "build_plan"]
