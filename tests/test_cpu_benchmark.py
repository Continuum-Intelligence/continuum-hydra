from __future__ import annotations

import importlib
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from continuum.profiler.cpu_benchmark import run_cpu_benchmark


class _FakeMatrix:
    def astype(self, _dtype):  # noqa: ANN001
        return self

    def __matmul__(self, _other):  # noqa: ANN001
        return self


class _FakeNumpy:
    float32 = "float32"

    def __init__(self) -> None:
        self.random = SimpleNamespace(rand=lambda _r, _c: _FakeMatrix())


class TestCpuBenchmark(unittest.TestCase):
    def test_cpu_benchmark_returns_expected_keys(self) -> None:
        fake_np = _FakeNumpy()
        ticks = itertools.count(step=0.01)
        real_import_module = importlib.import_module

        def _import_module(name: str):  # noqa: ANN202
            if name == "numpy":
                return fake_np
            return real_import_module(name)

        with patch("continuum.profiler.cpu_benchmark.importlib.util.find_spec", return_value=object()):
            with patch("continuum.profiler.cpu_benchmark.importlib.import_module", side_effect=_import_module):
                with patch("continuum.profiler.cpu_benchmark.perf_counter", side_effect=lambda: next(ticks)):
                    result = run_cpu_benchmark({"cpu_warmup": 0.0, "cpu_duration": 0.2, "notes": []})

        self.assertIn("cpu_sustained", result)
        payload = result["cpu_sustained"]
        expected = {
            "mean_iter_per_sec",
            "std_iter_per_sec",
            "p50_iter_per_sec",
            "p95_iter_per_sec",
            "iterations",
            "duration_sec",
        }
        self.assertEqual(set(payload.keys()), expected)

    def test_numpy_missing_does_not_crash(self) -> None:
        ctx = {"notes": []}
        with patch("continuum.profiler.cpu_benchmark.importlib.util.find_spec", return_value=None):
            result = run_cpu_benchmark(ctx)

        payload = result["cpu_sustained"]
        self.assertIsNone(payload["mean_iter_per_sec"])
        self.assertIsNone(payload["std_iter_per_sec"])
        self.assertTrue(any("numpy" in note.lower() for note in ctx["notes"]))

    def test_static_only_skips_cpu_benchmark(self) -> None:
        ctx = {"static_only": True, "notes": []}
        result = run_cpu_benchmark(ctx)
        payload = result["cpu_sustained"]

        self.assertIsNone(payload["mean_iter_per_sec"])
        self.assertIsNone(payload["iterations"])
        self.assertTrue(any("static-only" in note.lower() for note in ctx["notes"]))

    def test_deterministic_structure_validation(self) -> None:
        fake_np = _FakeNumpy()
        ticks = itertools.count(step=0.02)
        real_import_module = importlib.import_module

        def _import_module(name: str):  # noqa: ANN202
            if name == "numpy":
                return fake_np
            return real_import_module(name)

        with patch("continuum.profiler.cpu_benchmark.importlib.util.find_spec", return_value=object()):
            with patch("continuum.profiler.cpu_benchmark.importlib.import_module", side_effect=_import_module):
                with patch("continuum.profiler.cpu_benchmark.perf_counter", side_effect=lambda: next(ticks)):
                    result = run_cpu_benchmark({"cpu_warmup": 0.0, "cpu_duration": 0.1, "notes": []})

        payload = result["cpu_sustained"]
        self.assertIn(type(payload["mean_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["std_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["p50_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["p95_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["iterations"]), {int, type(None)})
        self.assertIn(type(payload["duration_sec"]), {float, type(None)})


if __name__ == "__main__":
    unittest.main()
