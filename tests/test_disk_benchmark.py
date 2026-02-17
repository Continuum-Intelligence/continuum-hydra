from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

from continuum.profiler.disk_benchmark import run_disk_benchmark


class TestDiskBenchmark(unittest.TestCase):
    def test_returns_expected_keys_and_types(self) -> None:
        ticks = itertools.count(step=0.01)
        with patch("continuum.profiler.disk_benchmark.perf_counter", side_effect=lambda: next(ticks)):
            result = run_disk_benchmark(
                {
                    "notes": [],
                    "disk_warmup": 0.0,
                    "disk_duration": 0.05,
                    "disk_size_mb": 1,
                }
            )

        self.assertIn("disk_random_io", result)
        payload = result["disk_random_io"]
        expected = {
            "mean_read_mb_s",
            "std_read_mb_s",
            "p50_read_mb_s",
            "p95_read_mb_s",
            "mean_iops",
            "iterations",
            "duration_sec",
        }
        self.assertEqual(set(payload.keys()), expected)
        self.assertIn(type(payload["mean_read_mb_s"]), {float, type(None)})
        self.assertIn(type(payload["mean_iops"]), {float, type(None)})
        self.assertIn(type(payload["iterations"]), {int, type(None)})
        self.assertIn(type(payload["duration_sec"]), {float, type(None)})

    def test_static_only_skips_benchmark(self) -> None:
        ctx = {"static_only": True, "notes": []}
        result = run_disk_benchmark(ctx)
        payload = result["disk_random_io"]
        self.assertIsNone(payload["mean_read_mb_s"])
        self.assertIsNone(payload["iterations"])
        self.assertTrue(any("static-only" in note.lower() for note in ctx["notes"]))

    def test_no_disk_skips_benchmark(self) -> None:
        ctx = {"no_disk": True, "notes": []}
        result = run_disk_benchmark(ctx)
        payload = result["disk_random_io"]
        self.assertIsNone(payload["mean_read_mb_s"])
        self.assertIsNone(payload["iterations"])
        self.assertTrue(any("--no-disk" in note.lower() for note in ctx["notes"]))

    def test_io_failure_returns_null_payload(self) -> None:
        ctx = {"notes": [], "disk_warmup": 0.0, "disk_duration": 0.05, "disk_size_mb": 1}
        with patch("continuum.profiler.disk_benchmark.tempfile.mkstemp", side_effect=OSError("boom")):
            result = run_disk_benchmark(ctx)

        payload = result["disk_random_io"]
        self.assertIsNone(payload["mean_read_mb_s"])
        self.assertIsNone(payload["mean_iops"])
        self.assertTrue(any("failed" in note.lower() for note in ctx["notes"]))


if __name__ == "__main__":
    unittest.main()
