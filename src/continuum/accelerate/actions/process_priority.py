from __future__ import annotations

import shutil
from typing import Any

from continuum.accelerate.models import AccelerationAction, AccelerationActionResult, ExecutionContext, profile_gte


class ProcessPriorityAction(AccelerationAction):
    id = "process.priority"
    title = "Process Priority Suggestions"
    category = "process"
    why = "Process niceness and IO priority can reduce scheduling jitter during training runs."
    risk = "low"
    requires_root = False
    platforms = ["linux", "windows", "macos"]
    profile_min = "minimal"

    def _commands(self, ctx: ExecutionContext) -> list[str]:
        commands = ["nice -n -5 <your_command>"]
        if ctx.is_linux and shutil.which("ionice"):
            commands.append("ionice -c2 -n0 <your_command>")
        return commands

    def check(self, ctx: ExecutionContext) -> tuple[bool, dict[str, Any], list[str]]:
        return True, {
            "ionice_available": bool(shutil.which("ionice")) if ctx.is_linux else False,
            "os_name": ctx.os_name,
        }, []

    def plan(self, ctx: ExecutionContext) -> tuple[bool, list[str], dict[str, Any], list[str]]:
        supported, before, notes = self.check(ctx)
        recommend = supported and profile_gte(ctx.env.get("ACCELERATE_PROFILE", "balanced"), "balanced")
        commands = self._commands(ctx)
        if not recommend:
            notes.append("Lower profile requested; suggestions remain optional")
        return recommend, commands, {"suggestions": commands}, notes

    def apply(self, ctx: ExecutionContext) -> AccelerationActionResult:
        supported, before, notes = self.check(ctx)
        commands = self._commands(ctx)
        return AccelerationActionResult(
            action_id=self.id,
            title=self.title,
            supported=supported,
            applied=False,
            skipped_reason="No-op action. Use suggested command wrappers for training runs.",
            requires_root=self.requires_root,
            risk=self.risk,
            before=before,
            after={"suggestions": commands, "notes": notes},
            commands=commands,
            errors=[],
        )


__all__ = ["ProcessPriorityAction"]
