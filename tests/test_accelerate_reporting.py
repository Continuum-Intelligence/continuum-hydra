from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuum.accelerate.models import ActionDescriptor, AccelerationActionResult, AccelerationPlan, ExecutionContext
from continuum.accelerate.plugins.loader import HookBundle, PluginLoadResult
from continuum.accelerate.reporting import build_report, write_state_report


class TestAccelerateReporting(unittest.TestCase):
    def test_write_state_report_and_build_report(self) -> None:
        plan = AccelerationPlan.create(
            profile="balanced",
            recommendations=[
                ActionDescriptor(
                    action_id="process.priority",
                    title="Process Priority",
                    category="process",
                    recommended=True,
                    risk="low",
                    requires_root=False,
                    supported=True,
                    why="test",
                    commands=["nice -n -5 <your_command>"],
                )
            ],
        )
        ctx = ExecutionContext(
            os_name="linux",
            is_linux=True,
            is_windows=False,
            is_macos=False,
            user_is_root=False,
            has_nvidia_smi=False,
            doctor_facts=None,
            env={},
            cwd="/tmp",
            repo_root="/tmp",
        )
        results = [
            AccelerationActionResult(
                action_id="process.priority",
                title="Process Priority",
                supported=True,
                applied=False,
                skipped_reason="Dry run - not applied",
                requires_root=False,
                risk="low",
            )
        ]
        plugin_result = PluginLoadResult(
            actions_loaded=0,
            hooks=HookBundle(),
            warnings=[],
            loaded_files=[],
            failures=[],
        )

        report = build_report(plan, results, ctx, {"process.priority"}, dry_run=True, plugin_result=plugin_result)

        with tempfile.TemporaryDirectory() as tmp:
            latest = write_state_report(report, cwd=Path(tmp))
            self.assertTrue(latest.exists())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total"], 1)
            self.assertEqual(payload["mode"], "dry-run")
            self.assertIn("plugin_summary", payload)


if __name__ == "__main__":
    unittest.main()
