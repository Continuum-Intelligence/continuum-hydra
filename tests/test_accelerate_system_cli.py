from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

if find_spec("typer") is not None:
    from typer.testing import CliRunner

    from continuum.cli import app
else:
    CliRunner = None
    app = None


@unittest.skipIf(find_spec("typer") is None, "typer is not installed in this interpreter")
class TestAccelerateSystemCli(unittest.TestCase):
    def test_status_without_state(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["accelerate", "--status"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                self.assertIn("Active", result.stdout)
            finally:
                os.chdir(previous)

    def test_on_dry_run_does_not_write_state(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["accelerate", "--on", "--dry-run"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                self.assertFalse((Path(tmp) / ".continuum" / "state" / "accelerate_state.json").exists())
            finally:
                os.chdir(previous)

    def test_off_dry_run_with_state(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                state_path = Path(tmp) / ".continuum" / "state"
                state_path.mkdir(parents=True, exist_ok=True)
                payload = {
                    "active": True,
                    "platform": "linux",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "changes_applied": [],
                    "previous_state": {"cpu_governor": "powersave"},
                    "failures": [],
                }
                (state_path / "accelerate_state.json").write_text(json.dumps(payload), encoding="utf-8")

                result = runner.invoke(app, ["accelerate", "--off", "--dry-run"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                # dry-run restore should not rewrite/clear the original payload on disk
                restored = json.loads((state_path / "accelerate_state.json").read_text(encoding="utf-8"))
                self.assertTrue(restored["active"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
