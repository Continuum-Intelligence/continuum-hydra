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
class TestLaunchRuntime(unittest.TestCase):
    def test_launch_script_auto_resume(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                script = Path("train.py")
                script.write_text(
                    """
from pathlib import Path
import sys

Path('checkpoints').mkdir(parents=True, exist_ok=True)
ckpt = Path('checkpoints/last.ckpt')
if '--resume' in sys.argv:
    print('resumed from checkpoint')
    sys.exit(0)
ckpt.write_text('checkpoint')
print('first attempt failed')
sys.exit(1)
""".strip()
                    + "\n",
                    encoding="utf-8",
                )

                result = runner.invoke(app, ["launch", "train.py", "--max-restarts", "1"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)

                state_path = Path(tmp) / ".hydra" / "state" / "launch_latest.json"
                self.assertTrue(state_path.exists())
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "completed")
                self.assertEqual(payload["restarts_used"], 1)
                self.assertEqual(len(payload["attempts"]), 2)
            finally:
                os.chdir(previous)

    def test_launch_script_dry_run(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                script = Path("train.py")
                script.write_text("print('hello')\n", encoding="utf-8")

                result = runner.invoke(app, ["launch", "train.py", "--dry-run", "--json"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["mode"], "dry-run")
                self.assertEqual(payload["attempts"], [])
            finally:
                os.chdir(previous)

    def test_launch_preserves_exact_output_dir_arg(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                script = Path("train.py")
                script.write_text("print('ok')\n", encoding="utf-8")

                result = runner.invoke(
                    app,
                    [
                        "launch",
                        "train.py",
                        "--dry-run",
                        "--json",
                        "--",
                        "--output-dir",
                        "./outputs/mmfine_100m",
                    ],
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 0)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["script_args"], ["--output-dir", "./outputs/mmfine_100m"])
                self.assertIn("./outputs/mmfine_100m", payload["command_argv"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
