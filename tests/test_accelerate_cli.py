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
class TestAccelerateCli(unittest.TestCase):
    def test_json_prints_only_json_to_stdout(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["launch", "--dry-run", "--json"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["mode"], "dry-run")
                self.assertNotIn("Hydra Launch Plan", result.stdout)
                self.assertNotIn("Hydra Launch Plan", result.stderr)
            finally:
                os.chdir(previous)

    def test_invalid_profile_returns_2(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["launch", "--profile", "ultra"], catch_exceptions=False)
        self.assertEqual(result.exit_code, 2)

    def test_unknown_only_category_returns_2(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["launch", "--dry-run", "--only", "unknown"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 2)
            finally:
                os.chdir(previous)

    def test_deterministic_action_ordering_in_json(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(
                    app,
                    ["launch", "--dry-run", "--json", "--no-timestamp"],
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 0)
                payload = json.loads(result.stdout)
                ids = [entry["action_id"] for entry in payload["plan"]["recommendations"]]
                self.assertEqual(ids, sorted(ids))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
