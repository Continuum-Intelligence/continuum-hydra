from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

if find_spec("typer") is not None:
    from typer.testing import CliRunner

    from continuum.cli import app
else:
    CliRunner = None
    app = None


@unittest.skipIf(find_spec("typer") is None, "typer is not installed in this interpreter")
class TestLaunchRuntime(unittest.TestCase):
    def test_launch_missing_script_returns_usage_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["launch", "missing_train.py"], catch_exceptions=False)
        self.assertEqual(result.exit_code, 2)
        err_text = getattr(result, "stderr", result.output)
        self.assertIn("Training script not found: missing_train.py", err_text)
        self.assertIn("Tip: use an absolute or correct relative path.", err_text)

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

    def test_launch_passthrough_args_exact_with_fixture(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                fixture = Path(__file__).parent / "fixtures" / "print_args.py"
                result = runner.invoke(
                    app,
                    [
                        "launch",
                        str(fixture),
                        "--json",
                        "--",
                        "--output-dir",
                        "./outputs/mmfine_100m",
                        "--tag",
                        "alpha beta",
                    ],
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 0)
                payload = json.loads(result.stdout)
                self.assertEqual(
                    payload["script_args"],
                    ["--output-dir", "./outputs/mmfine_100m", "--tag", "alpha beta"],
                )
                self.assertEqual(
                    payload["command_argv"][-4:],
                    ["--output-dir", "./outputs/mmfine_100m", "--tag", "alpha beta"],
                )

                stdout_line = payload["attempts"][0]["stdout_tail"][-1]
                self.assertEqual(
                    json.loads(stdout_line),
                    ["--output-dir", "./outputs/mmfine_100m", "--tag", "alpha beta"],
                )
            finally:
                os.chdir(previous)

    def test_launch_accelerate_restores_on_keyboard_interrupt(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                script = Path("train.py")
                script.write_text("print('ok')\n", encoding="utf-8")

                with patch(
                    "continuum.accelerate.cli.execute_acceleration_action",
                    side_effect=[
                        {"active_status": "Partial", "effective_active": False},
                        {"active_status": "False", "effective_active": False},
                    ],
                ) as accelerate_mock:
                    with patch(
                        "continuum.accelerate.cli.launch_training_script",
                        side_effect=KeyboardInterrupt,
                    ):
                        result = runner.invoke(
                            app,
                            ["launch", "train.py", "--accelerate"],
                            catch_exceptions=False,
                        )

                self.assertEqual(result.exit_code, 130)
                self.assertEqual(accelerate_mock.call_count, 2)
                self.assertEqual(accelerate_mock.call_args_list[0].kwargs["action"], "on")
                self.assertEqual(accelerate_mock.call_args_list[1].kwargs["action"], "off")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
