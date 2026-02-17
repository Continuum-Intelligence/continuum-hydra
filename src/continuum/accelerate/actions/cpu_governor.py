from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from continuum.accelerate.models import AccelerationAction, AccelerationActionResult, ExecutionContext, profile_gte

_SCALING_GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


class CpuGovernorAction(AccelerationAction):
    id = "cpu.governor"
    title = "CPU Governor"
    category = "cpu"
    why = "Keep CPU frequency policy aligned for consistent training throughput."
    risk = "medium"
    requires_root = True
    platforms = ["linux"]
    profile_min = "minimal"

    def _read_governor(self) -> str | None:
        if not _SCALING_GOVERNOR.exists():
            return None
        try:
            return _SCALING_GOVERNOR.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return None

    def check(self, ctx: ExecutionContext) -> tuple[bool, dict[str, Any], list[str]]:
        if not self.is_platform_supported(ctx):
            return False, {"reason": "Unsupported OS"}, ["Linux only action"]

        cpupower_path = shutil.which("cpupower")
        current_governor = self._read_governor()
        supported = cpupower_path is not None and current_governor is not None

        notes: list[str] = []
        if cpupower_path is None:
            notes.append("cpupower not found")
        if current_governor is None:
            notes.append("scaling governor path missing")

        return supported, {
            "cpupower_path": cpupower_path,
            "current_governor": current_governor,
        }, notes

    def plan(self, ctx: ExecutionContext) -> tuple[bool, list[str], dict[str, Any], list[str]]:
        supported, before, notes = self.check(ctx)
        if not supported:
            return False, [], before, notes

        current = before.get("current_governor")
        recommend = profile_gte(ctx.env.get("ACCELERATE_PROFILE", "balanced"), "balanced") and current != "performance"
        commands = ["cpupower frequency-set -g performance"]

        if not recommend:
            notes.append("No change needed for current profile/governor")

        return recommend, commands, {"target_governor": "performance"}, notes

    def apply(self, ctx: ExecutionContext) -> AccelerationActionResult:
        supported, before, notes = self.check(ctx)
        if not supported:
            return AccelerationActionResult(
                action_id=self.id,
                title=self.title,
                supported=False,
                applied=False,
                skipped_reason="; ".join(notes) or "Unsupported",
                requires_root=self.requires_root,
                risk=self.risk,
                before=before,
                after=before,
                commands=[],
                errors=[],
            )

        if self.requires_root and not ctx.user_is_root:
            return AccelerationActionResult(
                action_id=self.id,
                title=self.title,
                supported=True,
                applied=False,
                skipped_reason="Root privileges required",
                requires_root=self.requires_root,
                risk=self.risk,
                before=before,
                after=before,
                commands=["cpupower frequency-set -g performance"],
                errors=[],
            )

        command = ["cpupower", "frequency-set", "-g", "performance"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        except Exception as exc:  # noqa: BLE001
            return AccelerationActionResult(
                action_id=self.id,
                title=self.title,
                supported=True,
                applied=False,
                skipped_reason="Command execution failed",
                requires_root=self.requires_root,
                risk=self.risk,
                before=before,
                after=before,
                commands=[" ".join(command)],
                errors=[f"{type(exc).__name__}: {exc}"],
            )

        after = {
            "current_governor": self._read_governor(),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "returncode": completed.returncode,
        }

        return AccelerationActionResult(
            action_id=self.id,
            title=self.title,
            supported=True,
            applied=completed.returncode == 0,
            skipped_reason=None if completed.returncode == 0 else "cpupower returned non-zero exit code",
            requires_root=self.requires_root,
            risk=self.risk,
            before=before,
            after=after,
            commands=[" ".join(command)],
            errors=[] if completed.returncode == 0 else [completed.stderr.strip() or "Unknown cpupower error"],
            returncodes={"cpupower": completed.returncode},
            stdout_tail=[line for line in completed.stdout.strip().splitlines()[-5:] if line],
            stderr_tail=[line for line in completed.stderr.strip().splitlines()[-5:] if line],
        )


__all__ = ["CpuGovernorAction"]
