from __future__ import annotations

from continuum.launch.actions.cpu_governor import CpuGovernorAction
from continuum.launch.actions.nvidia_persistence import NvidiaPersistenceAction
from continuum.launch.actions.process_priority import ProcessPriorityAction
from continuum.launch.registry import register_action


def register_builtin_actions() -> None:
    register_action(CpuGovernorAction())
    register_action(NvidiaPersistenceAction())
    register_action(ProcessPriorityAction())


__all__ = [
    "register_builtin_actions",
    "CpuGovernorAction",
    "NvidiaPersistenceAction",
    "ProcessPriorityAction",
]
