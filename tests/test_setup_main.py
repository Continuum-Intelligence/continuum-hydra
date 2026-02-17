from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from continuum.setup import main as setup_main


class TestSetupCommand(unittest.TestCase):
    def _assert_exit_code(self, exc: BaseException, expected: int) -> None:
        code = getattr(exc, "code", None)
        if code is None:
            code = getattr(exc, "exit_code", None)
        if code is None and isinstance(exc, SystemExit):
            code = exc.code
        self.assertEqual(code, expected)

    def test_pip_commands_constructed_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "env_manifest.json"
            req_path = Path(tmpdir) / "req.txt"
            req_path.write_text("requests\n", encoding="utf-8")

            run_calls: list[list[str]] = []

            def _fake_run(cmd, capture_output, text, check):  # noqa: ANN001, ANN202
                run_calls.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            fake_torch = SimpleNamespace(
                __version__="2.5.0",
                cuda=SimpleNamespace(is_available=lambda: True),
                version=SimpleNamespace(cuda="12.1"),
            )

            with patch("continuum.setup.main._resolve_manifest_path", return_value=manifest_path):
                with patch("continuum.setup.main.subprocess.run", side_effect=_fake_run):
                    with patch("continuum.setup.main.import_module", return_value=fake_torch):
                        with patch("continuum.setup.main.platform.system", return_value="Linux"):
                            with patch("continuum.setup.main._safe_dist_version", side_effect=lambda name: "1.0.0" if name == "numpy" else None):
                                with self.assertRaises(BaseException) as cm:
                                    setup_main.setup_command(
                                        with_torch=True,
                                        torch_spec="torch==2.5.*",
                                        torch_index="https://download.pytorch.org/whl/cu121",
                                        numpy_spec="numpy==1.26.4",
                                        upgrade=True,
                                        requirements=req_path,
                                        dry_run=False,
                                        verbose=False,
                                    )

            self._assert_exit_code(cm.exception, 0)
            self.assertGreaterEqual(len(run_calls), 5)
            self.assertEqual(run_calls[0], [setup_main.sys.executable, "-m", "pip", "--version"])
            self.assertEqual(
                run_calls[1],
                [setup_main.sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            )
            self.assertEqual(
                run_calls[2],
                [setup_main.sys.executable, "-m", "pip", "install", "--upgrade", "numpy==1.26.4"],
            )
            self.assertEqual(
                run_calls[3],
                [
                    setup_main.sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu121",
                    "--extra-index-url",
                    "https://pypi.org/simple",
                    "torch==2.5.*",
                ],
            )
            self.assertEqual(
                run_calls[4],
                [setup_main.sys.executable, "-m", "pip", "install", "--upgrade", "-r", str(req_path)],
            )

    def test_dry_run_does_not_call_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "env_manifest.json"
            with patch("continuum.setup.main._resolve_manifest_path", return_value=manifest_path):
                with patch("continuum.setup.main.subprocess.run") as mock_run:
                    with self.assertRaises(BaseException) as cm:
                        setup_main.setup_command(
                            with_torch=False,
                            dry_run=True,
                            verbose=False,
                        )

            self._assert_exit_code(cm.exception, 0)
            for call in mock_run.call_args_list:
                cmd = call.args[0] if call.args else []
                if isinstance(cmd, list) and cmd:
                    self.assertFalse(
                        cmd[:3] == [setup_main.sys.executable, "-m", "pip"],
                        "Dry-run should not execute pip subprocess commands.",
                    )
            self.assertTrue(manifest_path.exists())
            self.assertTrue((manifest_path.parent / "requirements.txt").exists())
            self.assertTrue((manifest_path.parent / "README.md").exists())

    def test_manifest_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "env_manifest.json"

            def _fake_run(cmd, capture_output, text, check):  # noqa: ANN001, ANN202
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("continuum.setup.main._resolve_manifest_path", return_value=manifest_path):
                with patch("continuum.setup.main.subprocess.run", side_effect=_fake_run):
                    with patch("continuum.setup.main._safe_dist_version", side_effect=lambda name: "1.26.4" if name == "numpy" else "0.1.0"):
                        with self.assertRaises(BaseException) as cm:
                            setup_main.setup_command(dry_run=False, with_torch=False)

            self._assert_exit_code(cm.exception, 0)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("python_version", payload)
            self.assertIn("platform", payload)
            self.assertIn("architecture", payload)
            self.assertIn("venv", payload)
            self.assertIn("installed", payload)
            self.assertIn("timestamp", payload)
            self.assertIn("continuum_version", payload)
            installed = payload["installed"]
            self.assertIn("numpy", installed)
            self.assertIn("torch", installed)
            self.assertIn("torch_cuda_available", installed)
            self.assertIn("torch_cuda_version", installed)
            self.assertTrue((manifest_path.parent / "requirements.txt").exists())
            self.assertTrue((manifest_path.parent / "README.md").exists())
            req_lines = (manifest_path.parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(line.startswith("numpy") for line in req_lines))

    def test_torch_import_failure_is_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "env_manifest.json"

            def _fake_run(cmd, capture_output, text, check):  # noqa: ANN001, ANN202
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("continuum.setup.main._resolve_manifest_path", return_value=manifest_path):
                with patch("continuum.setup.main.subprocess.run", side_effect=_fake_run):
                    with patch("continuum.setup.main.import_module", side_effect=ImportError("no torch")):
                        with patch("continuum.setup.main.platform.system", return_value="Darwin"):
                            with self.assertRaises(BaseException) as cm:
                                setup_main.setup_command(with_torch=True)

            self._assert_exit_code(cm.exception, 0)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            installed = payload["installed"]
            self.assertIsNone(installed["torch"])
            self.assertIsNone(installed["torch_cuda_available"])
            self.assertIsNone(installed["torch_cuda_version"])
            notes = payload.get("notes", [])
            self.assertTrue(any("torch validation failed" in str(note).lower() for note in notes))

    def test_non_cuda_torch_fails_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "env_manifest.json"

            def _fake_run(cmd, capture_output, text, check):  # noqa: ANN001, ANN202
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            fake_torch = SimpleNamespace(
                __version__="2.5.0",
                cuda=SimpleNamespace(is_available=lambda: False),
                version=SimpleNamespace(cuda=None),
            )

            with patch("continuum.setup.main._resolve_manifest_path", return_value=manifest_path):
                with patch("continuum.setup.main.subprocess.run", side_effect=_fake_run):
                    with patch("continuum.setup.main.import_module", return_value=fake_torch):
                        with patch("continuum.setup.main.platform.system", return_value="Linux"):
                            with self.assertRaises(BaseException) as cm:
                                setup_main.setup_command(with_torch=True)

            self._assert_exit_code(cm.exception, 4)

    def test_state_readme_contains_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "env_manifest.json"

            def _fake_run(cmd, capture_output, text, check):  # noqa: ANN001, ANN202
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("continuum.setup.main._resolve_manifest_path", return_value=manifest_path):
                with patch("continuum.setup.main.subprocess.run", side_effect=_fake_run):
                    with self.assertRaises(BaseException) as cm:
                        setup_main.setup_command(with_torch=False)

            self._assert_exit_code(cm.exception, 0)
            readme_path = manifest_path.parent / "README.md"
            content = readme_path.read_text(encoding="utf-8")
            self.assertIn("env_manifest.json", content)
            self.assertIn("requirements.txt", content)


if __name__ == "__main__":
    unittest.main()
