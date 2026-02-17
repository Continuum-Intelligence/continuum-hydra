from __future__ import annotations

import re
import subprocess
from typing import Any

from continuum.accelerate.models import AccelerationAction, AccelerationActionResult, ExecutionContext, profile_gte

_PERSISTENCE_PATTERN = re.compile(r"Persistence Mode\s*:\s*(Enabled|Disabled)", re.IGNORECASE)


class NvidiaPersistenceAction(AccelerationAction):
    id = "gpu.nvidia_persistence"
    title = "NVIDIA Persistence Mode"
    category = "gpu"
    why = "Persistence mode reduces startup latency and stabilizes GPU initialization."
    risk = "medium"
    requires_root = True
    platforms = ["linux"]
    profile_min = "minimal"

    def _read_persistence(self) -> tuple[bool, dict[str, Any], list[str]]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            return False, {}, [f"nvidia-smi failed: {type(exc).__name__}: {exc}"]

        if completed.returncode != 0:
            return False, {
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "returncode": completed.returncode,
            }, ["nvidia-smi -q returned non-zero exit code"]

        text = completed.stdout
        match = _PERSISTENCE_PATTERN.search(text)
        state = None if match is None else match.group(1).lower()
        return True, {
            "persistence_mode": state,
            "raw_excerpt": text[:600],
        }, [] if state is not None else ["Could not parse persistence mode"]

    def check(self, ctx: ExecutionContext) -> tuple[bool, dict[str, Any], list[str]]:
        if not self.is_platform_supported(ctx):
            return False, {"reason": "Unsupported OS"}, ["Linux only action"]
        if not ctx.has_nvidia_smi:
            return False, {"reason": "nvidia-smi missing"}, ["nvidia-smi not available"]

        ok, before, notes = self._read_persistence()
        return ok, before, notes

    def plan(self, ctx: ExecutionContext) -> tuple[bool, list[str], dict[str, Any], list[str]]:
        supported, before, notes = self.check(ctx)
        if not supported:
            return False, [], before, notes

        state = before.get("persistence_mode")
        recommend = profile_gte(ctx.env.get("ACCELERATE_PROFILE", "balanced"), "balanced") and state != "enabled"
        commands = ["nvidia-smi -pm 1"]

        if not recommend:
            notes.append("No change needed for current profile/state")

        return recommend, commands, {"target_persistence_mode": "enabled"}, notes

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
                commands=["nvidia-smi -pm 1"],
                errors=[],
            )

        command = ["nvidia-smi", "-pm", "1"]
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

        recheck_supported, after_state, recheck_notes = self._read_persistence()
        after = {
            **after_state,
            "recheck_supported": recheck_supported,
            "recheck_notes": recheck_notes,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "returncode": completed.returncode,
        }

        return AccelerationActionResult(
            action_id=self.id,
            title=self.title,
            supported=True,
            applied=completed.returncode == 0,
            skipped_reason=None if completed.returncode == 0 else "nvidia-smi returned non-zero exit code",
            requires_root=self.requires_root,
            risk=self.risk,
            before=before,
            after=after,
            commands=[" ".join(command)],
            errors=[] if completed.returncode == 0 else [completed.stderr.strip() or "Unknown nvidia-smi error"],
            returncodes={"nvidia-smi -pm 1": completed.returncode},
            stdout_tail=[line for line in completed.stdout.strip().splitlines()[-5:] if line],
            stderr_tail=[line for line in completed.stderr.strip().splitlines()[-5:] if line],
        )


__all__ = ["NvidiaPersistenceAction"]
