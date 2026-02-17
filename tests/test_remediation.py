from __future__ import annotations

import unittest

from continuum.profiler.remediation import generate_remediation


class TestRemediation(unittest.TestCase):
    def test_memory_bottleneck_contains_fp16_action(self) -> None:
        report = {"analysis": {"primary_bottleneck": "memory_bandwidth", "confidence": 0.75}}
        remediation = generate_remediation(report)
        titles = [action.get("title", "") for action in remediation["actions"]]
        self.assertTrue(any("fp16" in title.lower() or "bf16" in title.lower() for title in titles))

    def test_gpu_instability_contains_thermal_action(self) -> None:
        report = {"analysis": {"primary_bottleneck": "gpu_instability", "confidence": 0.8}}
        remediation = generate_remediation(report)
        reasons = [action.get("reason", "") for action in remediation["actions"]]
        self.assertTrue(any("thermal" in reason.lower() or "power" in reason.lower() for reason in reasons))

    def test_unknown_bottleneck_has_empty_actions(self) -> None:
        report = {"analysis": {"primary_bottleneck": None, "confidence": 0.2}}
        remediation = generate_remediation(report)
        self.assertEqual(remediation["actions"], [])

    def test_priority_levels(self) -> None:
        low = generate_remediation({"analysis": {"primary_bottleneck": "cpu_compute", "confidence": 0.2}})
        medium = generate_remediation({"analysis": {"primary_bottleneck": "cpu_compute", "confidence": 0.4}})
        high = generate_remediation({"analysis": {"primary_bottleneck": "cpu_compute", "confidence": 0.7}})
        self.assertEqual(low["priority"], "low")
        self.assertEqual(medium["priority"], "medium")
        self.assertEqual(high["priority"], "high")

    def test_structure_is_deterministic(self) -> None:
        remediation = generate_remediation({"analysis": {"primary_bottleneck": "cpu_compute", "confidence": 0.5}})
        self.assertIn("priority", remediation)
        self.assertIn("actions", remediation)
        self.assertIn(remediation["priority"], {"low", "medium", "high"})
        self.assertIsInstance(remediation["actions"], list)
        for action in remediation["actions"]:
            self.assertIn("title", action)
            self.assertIn("impact", action)
            self.assertIn("difficulty", action)
            self.assertIn("reason", action)


if __name__ == "__main__":
    unittest.main()
