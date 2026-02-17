from __future__ import annotations

from typing import Iterable

from continuum.accelerate.models import PROFILE_ORDER, AccelerationAction

_REGISTRY: dict[str, AccelerationAction] = {}


def register_action(action: AccelerationAction | type[AccelerationAction]) -> None:
    instance = action() if isinstance(action, type) else action
    _REGISTRY[instance.id] = instance


def get_actions() -> list[AccelerationAction]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY.keys())]


def clear_registry() -> None:
    _REGISTRY.clear()


def filter_actions(
    actions: Iterable[AccelerationAction],
    only: set[str] | None,
    exclude: set[str] | None,
    profile: str,
    categories: set[str] | None = None,
) -> list[AccelerationAction]:
    required_level = PROFILE_ORDER.get(profile, PROFILE_ORDER["balanced"])
    only_norm = {value.lower() for value in only} if only else None
    exclude_norm = {value.lower() for value in exclude} if exclude else None
    category_norm = {value.lower() for value in categories} if categories else None

    filtered: list[AccelerationAction] = []

    for action in actions:
        action_category = action.category.lower()
        action_profile_min = PROFILE_ORDER.get(action.profile_min, PROFILE_ORDER["minimal"])

        if action_profile_min > required_level:
            continue

        if exclude_norm and action_category in exclude_norm:
            continue

        if category_norm and action_category not in category_norm:
            continue

        if only_norm and action_category not in only_norm and action.id.lower() not in only_norm:
            continue

        filtered.append(action)

    return sorted(filtered, key=lambda item: item.id)


__all__ = [
    "register_action",
    "get_actions",
    "clear_registry",
    "filter_actions",
]
