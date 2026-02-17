from __future__ import annotations

import importlib.machinery
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from continuum.profiler.formatters import build_profile_report, render_profile_human
from continuum.profiler.static_profile import collect_static_profile


class TestStaticProfile(unittest.TestCase):
    def test_report_contains_static_profile_shape(self) -> None:
        with patch("continuum.profiler.static_profile.platform.system", return_value="Linux"):
            with patch("continuum.profiler.static_profile.platform.machine", return_value="x86_64"):
                with patch("continuum.profiler.static_profile.platform.release", return_value="6.8.0"):
                    with patch("continuum.profiler.static_profile.platform.version", return_value="#1 SMP"):
                        with patch("continuum.profiler.static_profile.platform.python_version", return_value="3.12.3"):
                            with patch("continuum.profiler.static_profile.platform.platform", return_value="Linux-6.8.0"):
                                with patch("continuum.profiler.static_profile.os.cpu_count", return_value=16):
                                    fake_psutil = SimpleNamespace(
                                        cpu_count=lambda logical=False: 8 if not logical else 16,
                                        virtual_memory=lambda: SimpleNamespace(total=64 * 1024**3),
                                        disk_partitions=lambda all=True: [
                                            SimpleNamespace(mountpoint="/", device="/dev/sda1", fstype="ext4")
                                        ],
                                    )
                                    with patch("continuum.profiler.static_profile._get_psutil", return_value=fake_psutil):
                                        with patch(
                                            "continuum.profiler.static_profile.Path.read_text",
                                            return_value="model name\t: Test CPU\n/dev/root / ext4 rw 0 0\nNAME=test\nVERSION=1\n",
                                        ):
                                            with patch(
                                                "continuum.profiler.static_profile.importlib.util.find_spec",
                                                return_value=None,
                                            ):
                                                profile = collect_static_profile({"facts": {}})

        report = build_profile_report(profile)
        self.assertEqual(report["schema_version"], "1.0.0")
        self.assertIn("static_profile", report)
        self.assertIn("benchmark_results", report)
        self.assertIsInstance(report["benchmark_results"], list)

        static = report["static_profile"]
        self.assertIn("cpu", static)
        self.assertIn("memory", static)
        self.assertIn("storage", static)
        self.assertIn("os", static)
        self.assertIn("runtime", static)

        self.assertIn("model", static["cpu"])
        self.assertIn("cores_physical", static["cpu"])
        self.assertIn("cores_logical", static["cpu"])
        self.assertIn("arch", static["cpu"])
        self.assertIn("total_bytes", static["memory"])
        self.assertIn("root_mount", static["storage"])
        self.assertIn("root_device", static["storage"])
        self.assertIn("is_nvme", static["storage"])
        self.assertIn("is_ssd", static["storage"])
        self.assertIn("filesystem_type", static["storage"])
        self.assertIn("notes", static["storage"])

    def test_torch_import_failure_is_non_fatal(self) -> None:
        fake_spec = importlib.machinery.ModuleSpec("torch", loader=None)
        with patch("continuum.profiler.static_profile.importlib.util.find_spec", return_value=fake_spec):
            with patch(
                "continuum.profiler.static_profile.importlib.import_module",
                side_effect=ImportError("broken torch install"),
            ):
                profile = collect_static_profile({"facts": {}})

        runtime = profile["runtime"]
        self.assertIsNone(runtime["torch_version"])
        self.assertIsNone(runtime["torch_cuda_available"])
        self.assertIsNone(runtime["torch_cuda_version"])
        self.assertTrue(any("could not be imported" in note.lower() for note in profile["notes"]))

    def test_storage_probe_graceful_when_proc_and_sysfs_missing(self) -> None:
        def _read_text_side_effect(self, *args, **kwargs):  # noqa: ANN001
            raise OSError("missing")

        with patch("continuum.profiler.static_profile.platform.system", return_value="Linux"):
            with patch("continuum.profiler.static_profile.Path.read_text", _read_text_side_effect):
                fake_psutil = SimpleNamespace(disk_partitions=lambda all=True: [])
                with patch("continuum.profiler.static_profile._get_psutil", return_value=fake_psutil):
                    profile = collect_static_profile({"facts": {}})

        storage = profile["storage"]
        self.assertIsNone(storage["root_device"])
        self.assertIsNone(storage["filesystem_type"])
        self.assertIsNone(storage["is_ssd"])
        self.assertGreater(len(storage["notes"]), 0)
        # Top-level notes should not duplicate storage notes.
        overlap = set(storage["notes"]).intersection(set(profile["notes"]))
        self.assertEqual(overlap, set())

    def test_psutil_missing_is_non_fatal(self) -> None:
        with patch("continuum.profiler.static_profile._get_psutil", return_value=None):
            with patch("continuum.profiler.static_profile.importlib.util.find_spec", return_value=None):
                with patch("continuum.profiler.static_profile.platform.system", return_value="Linux"):
                    with patch(
                        "continuum.profiler.static_profile.Path.read_text",
                        return_value="model name\t: Test CPU\nMemTotal: 1234 kB\n/dev/root / ext4 rw 0 0\nNAME=test\nVERSION=1\n",
                    ):
                        profile = collect_static_profile({"facts": {}})

        self.assertIn("cpu", profile)
        self.assertIn("memory", profile)
        self.assertIn("storage", profile)
        self.assertIn("os", profile)
        self.assertIn("runtime", profile)
        self.assertIsInstance(profile.get("notes"), list)
        self.assertTrue(any("psutil" in note.lower() for note in profile["notes"]))

    def test_rich_missing_fallback_is_compact_text(self) -> None:
        report = build_profile_report(
            {
                "cpu": {"model": "CPU", "cores_physical": 8, "cores_logical": 16, "arch": "x86_64"},
                "memory": {"total_bytes": 1024},
                "storage": {
                    "root_mount": "/",
                    "root_device": "/dev/sda1",
                    "is_nvme": False,
                    "is_ssd": True,
                    "filesystem_type": "ext4",
                    "notes": [],
                },
                "os": {"name": "Linux", "version": "test", "kernel": "6.8.0"},
                "runtime": {
                    "python_version": "3.12.3",
                    "torch_version": None,
                    "torch_cuda_available": None,
                    "torch_cuda_version": None,
                    "platform": "Linux",
                },
                "notes": [],
            },
            benchmark_results=[
                {
                    "name": "benchmark.cpu_loop_ops",
                    "status": "PASS",
                    "result": 12345,
                    "unit": "ops/s",
                    "message": "CPU integer-loop throughput measured.",
                }
            ],
        )
        stream = io.StringIO()
        with patch("continuum.profiler.formatters.Console", None):
            with patch("continuum.profiler.formatters.Table", None):
                with redirect_stdout(stream):
                    render_profile_human(report)

        output = stream.getvalue()
        self.assertIn("Continuum Profile Report", output)
        self.assertIn("[PASS] cpu.model", output)
        self.assertIn("benchmark.cpu_loop_ops", output)
        self.assertNotIn('"static_profile"', output)


if __name__ == "__main__":
    unittest.main()
