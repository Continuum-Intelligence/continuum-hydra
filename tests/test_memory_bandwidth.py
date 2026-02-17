from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

from continuum.profiler.memory_bandwidth import run_memory_bandwidth


class TestMemoryBandwidth(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        ticks = itertools.count(step=0.01)
        with patch("continuum.profiler.memory_bandwidth.importlib.util.find_spec", return_value=None):
            with patch("continuum.profiler.memory_bandwidth.perf_counter", side_effect=lambda: next(ticks)):
                result = run_memory_bandwidth(
                    {
                        "mem_mb": 1,
                        "mem_warmup": 0.0,
                        "mem_duration": 0.2,
                        "notes": [],
                    }
                )

        self.assertIn("memory_bandwidth", result)
        payload = result["memory_bandwidth"]
        expected = {
            "mean_gbps",
            "std_gbps",
            "p50_gbps",
            "p95_gbps",
            "iterations",
            "duration_sec",
            "bytes_per_iter",
        }
        self.assertEqual(set(payload.keys()), expected)

    def test_numpy_missing_uses_fallback(self) -> None:
        ticks = itertools.count(step=0.02)
        ctx = {"mem_mb": 1, "mem_warmup": 0.0, "mem_duration": 0.1, "notes": []}
        with patch("continuum.profiler.memory_bandwidth.importlib.util.find_spec", return_value=None):
            with patch("continuum.profiler.memory_bandwidth.perf_counter", side_effect=lambda: next(ticks)):
                result = run_memory_bandwidth(ctx)

        payload = result["memory_bandwidth"]
        self.assertIn(type(payload["mean_gbps"]), {float, type(None)})
        self.assertTrue(any("stdlib bytearray copy fallback" in note.lower() for note in ctx["notes"]))

    def test_static_only_skips_benchmark(self) -> None:
        ctx = {"static_only": True, "notes": []}
        result = run_memory_bandwidth(ctx)
        payload = result["memory_bandwidth"]

        self.assertIsNone(payload["mean_gbps"])
        self.assertIsNone(payload["iterations"])
        self.assertTrue(any("static-only" in note.lower() for note in ctx["notes"]))

    def test_deterministic_structure_validation(self) -> None:
        ticks = itertools.count(step=0.01)
        with patch("continuum.profiler.memory_bandwidth.importlib.util.find_spec", return_value=None):
            with patch("continuum.profiler.memory_bandwidth.perf_counter", side_effect=lambda: next(ticks)):
                result = run_memory_bandwidth(
                    {
                        "mem_mb": 1,
                        "mem_warmup": 0.0,
                        "mem_duration": 0.1,
                        "notes": [],
                    }
                )

        payload = result["memory_bandwidth"]
        self.assertIn(type(payload["mean_gbps"]), {float, type(None)})
        self.assertIn(type(payload["std_gbps"]), {float, type(None)})
        self.assertIn(type(payload["p50_gbps"]), {float, type(None)})
        self.assertIn(type(payload["p95_gbps"]), {float, type(None)})
        self.assertIn(type(payload["iterations"]), {int, type(None)})
        self.assertIn(type(payload["duration_sec"]), {float, type(None)})
        self.assertIn(type(payload["bytes_per_iter"]), {int, type(None)})


if __name__ == "__main__":
    unittest.main()
