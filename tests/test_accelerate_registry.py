from __future__ import annotations

import unittest

from continuum.accelerate.models import AccelerationAction, AccelerationActionResult, ExecutionContext
from continuum.accelerate.registry import clear_registry, filter_actions, get_actions, register_action


class _DummyAction(AccelerationAction):
    def __init__(
        self,
        action_id: str,
        category: str,
        profile_min: str = "minimal",
    ) -> None:
        self.id = action_id
        self.title = action_id
        self.category = category
        self.why = "test"
        self.risk = "low"
        self.requires_root = False
        self.platforms = ["linux", "windows", "macos"]
        self.profile_min = profile_min

    def check(self, ctx: ExecutionContext):
        return True, {}, []

    def plan(self, ctx: ExecutionContext):
        return True, [], {}, []

    def apply(self, ctx: ExecutionContext):
        return AccelerationActionResult(
            action_id=self.id,
            title=self.title,
            supported=True,
            applied=False,
            skipped_reason="noop",
            requires_root=False,
            risk="low",
        )


class TestAccelerateRegistry(unittest.TestCase):
    def setUp(self) -> None:
        clear_registry()

    def tearDown(self) -> None:
        clear_registry()

    def test_register_and_get_actions_sorted(self) -> None:
        register_action(_DummyAction("z.last", "misc"))
        register_action(_DummyAction("a.first", "gpu"))
        ids = [action.id for action in get_actions()]
        self.assertEqual(ids, ["a.first", "z.last"])

    def test_filter_actions_profile_and_categories(self) -> None:
        actions = [
            _DummyAction("gpu.one", "gpu", profile_min="minimal"),
            _DummyAction("cpu.two", "cpu", profile_min="max"),
            _DummyAction("process.three", "process", profile_min="balanced"),
        ]

        filtered = filter_actions(actions, only={"gpu", "process"}, exclude={"cpu"}, profile="balanced")
        self.assertEqual([action.id for action in filtered], ["gpu.one", "process.three"])

    def test_filter_actions_honors_only_by_action_id(self) -> None:
        actions = [
            _DummyAction("gpu.one", "gpu"),
            _DummyAction("process.two", "process"),
        ]
        filtered = filter_actions(actions, only={"process.two"}, exclude=None, profile="balanced")
        self.assertEqual([action.id for action in filtered], ["process.two"])


if __name__ == "__main__":
    unittest.main()
