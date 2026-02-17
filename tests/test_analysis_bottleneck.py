from __future__ import annotations

import unittest

from continuum.profiler.analysis import classify_bottleneck


class TestBottleneckAnalysis(unittest.TestCase):
    def test_missing_benchmarks_returns_low_confidence(self) -> None:
        report = {
            "schema_version": "1.0.0",
            "static_profile": {},
            "benchmarks": {},
        }
        analysis = classify_bottleneck(report)

        self.assertIsNone(analysis["primary_bottleneck"])
        self.assertIsNone(analysis["secondary_bottleneck"])
        self.assertLessEqual(float(analysis["confidence"]), 0.4)
        self.assertTrue(any("missing" in reason.lower() for reason in analysis["reasons"]))

    def test_memory_bottleneck_scenario(self) -> None:
        report = {
            "schema_version": "1.0.0",
            "static_profile": {
                "os": {"name": "Darwin"},
                "cpu": {"arch": "arm64"},
            },
            "benchmarks": {
                "cpu_sustained": {
                    "mean_iter_per_sec": 1.2,
                    "std_iter_per_sec": 0.05,
                    "p95_iter_per_sec": 1.15,
                },
                "memory_bandwidth": {
                    "mean_gbps": 40.0,
                    "std_gbps": 2.0,
                    "p95_gbps": 39.0,
                },
                "gpu_sustained": {
                    "mean_iter_per_sec": 0.4,
                    "std_iter_per_sec": 0.03,
                    "p95_iter_per_sec": 0.38,
                },
            },
        }
        analysis = classify_bottleneck(report)
        self.assertEqual(analysis["primary_bottleneck"], "memory_bandwidth")

    def test_gpu_instability_scenario(self) -> None:
        report = {
            "schema_version": "1.0.0",
            "static_profile": {"os": {"name": "Linux"}, "cpu": {"arch": "x86_64"}},
            "benchmarks": {
                "gpu_sustained": {
                    "backend": "cuda",
                    "mean_iter_per_sec": 1.0,
                    "std_iter_per_sec": 0.35,
                    "p95_iter_per_sec": 0.70,
                }
            },
        }
        analysis = classify_bottleneck(report)
        self.assertEqual(analysis["primary_bottleneck"], "gpu_instability")

    def test_cpu_instability_scenario(self) -> None:
        report = {
            "schema_version": "1.0.0",
            "static_profile": {"os": {"name": "Linux"}, "cpu": {"arch": "x86_64"}},
            "benchmarks": {
                "cpu_sustained": {
                    "mean_iter_per_sec": 1.0,
                    "std_iter_per_sec": 0.30,
                    "p95_iter_per_sec": 0.90,
                }
            },
        }
        analysis = classify_bottleneck(report)
        self.assertEqual(analysis["primary_bottleneck"], "cpu_instability")

    def test_confidence_clamped_and_keys_exist(self) -> None:
        report = {
            "schema_version": "1.0.0",
            "static_profile": {"os": {"name": "Linux"}, "cpu": {"arch": "x86_64"}},
            "benchmarks": {
                "cpu_sustained": {"mean_iter_per_sec": 1.0, "std_iter_per_sec": 0.01, "p95_iter_per_sec": 0.99},
                "memory_bandwidth": {"mean_gbps": 10.0, "std_gbps": 0.1, "p95_gbps": 9.9},
                "gpu_sustained": {"mean_iter_per_sec": 0.04, "std_iter_per_sec": 0.001, "p95_iter_per_sec": 0.039},
            },
        }
        analysis = classify_bottleneck(report)
        self.assertIn("primary_bottleneck", analysis)
        self.assertIn("secondary_bottleneck", analysis)
        self.assertIn("confidence", analysis)
        self.assertIn("signals", analysis)
        self.assertIn("reasons", analysis)
        self.assertIn("recommendations", analysis)
        self.assertGreaterEqual(float(analysis["confidence"]), 0.0)
        self.assertLessEqual(float(analysis["confidence"]), 1.0)

    def test_disk_io_bottleneck_scenario(self) -> None:
        report = {
            "schema_version": "1.0.0",
            "static_profile": {
                "storage": {"is_nvme": True, "is_ssd": True},
                "os": {"name": "Linux"},
                "cpu": {"arch": "x86_64"},
            },
            "benchmarks": {
                "disk_random_io": {
                    "mean_read_mb_s": 50.0,
                    "std_read_mb_s": 2.0,
                    "p95_read_mb_s": 48.0,
                    "mean_iops": 12000.0,
                },
                "cpu_sustained": {
                    "mean_iter_per_sec": 1.1,
                    "std_iter_per_sec": 0.03,
                    "p95_iter_per_sec": 1.05,
                },
            },
        }
        analysis = classify_bottleneck(report)
        self.assertEqual(analysis["primary_bottleneck"], "disk_io")


if __name__ == "__main__":
    unittest.main()
