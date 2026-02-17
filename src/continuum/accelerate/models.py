from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any


ACCELERATE_SCHEMA_VERSION = "launch.v1"
PROFILE_ORDER = {
    "minimal": 0,
    "balanced": 1,
    "max": 2,
    "expert": 3,
}


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    os_name: str
    is_linux: bool
    is_windows: bool
    is_macos: bool
    user_is_root: bool
    has_nvidia_smi: bool
    doctor_facts: dict[str, Any] | None
    env: dict[str, str]
    cwd: str
    repo_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "is_linux": self.is_linux,
            "is_windows": self.is_windows,
            "is_macos": self.is_macos,
            "user_is_root": self.user_is_root,
            "has_nvidia_smi": self.has_nvidia_smi,
            "doctor_facts": self.doctor_facts,
            "env": dict(self.env),
            "cwd": self.cwd,
            "repo_root": self.repo_root,
        }


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    action_id: str
    title: str
    category: str
    recommended: bool
    risk: str
    requires_root: bool
    supported: bool
    why: str
    commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "category": self.category,
            "recommended": self.recommended,
            "risk": self.risk,
            "requires_root": self.requires_root,
            "supported": self.supported,
            "why": self.why,
            "commands": list(self.commands),
        }


@dataclass(frozen=True, slots=True)
class AccelerationPlan:
    schema_version: str
    plan_id: str
    created_at: str
    profile: str
    recommendations: list[ActionDescriptor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        profile: str,
        recommendations: list[ActionDescriptor],
        warnings: list[str] | None = None,
        include_timestamp: bool = True,
    ) -> "AccelerationPlan":
        now = datetime.now(timezone.utc)
        if include_timestamp:
            created_at = now.isoformat()
            plan_id = f"launch-{now.strftime('%Y%m%d%H%M%S')}"
        else:
            created_at = ""
            signature = "|".join([profile, *sorted(rec.action_id for rec in recommendations)])
            plan_id = f"launch-{sha1(signature.encode('utf-8')).hexdigest()[:12]}"
        return cls(
            schema_version=ACCELERATE_SCHEMA_VERSION,
            plan_id=plan_id,
            created_at=created_at,
            profile=profile,
            recommendations=recommendations,
            warnings=list(warnings or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "profile": self.profile,
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class AccelerationActionResult:
    action_id: str
    title: str
    supported: bool
    applied: bool
    skipped_reason: str | None
    requires_root: bool
    risk: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    returncodes: dict[str, int] = field(default_factory=dict)
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.applied and not self.skipped_reason:
            raise ValueError("skipped_reason is required when applied is False")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "supported": self.supported,
            "applied": self.applied,
            "skipped_reason": self.skipped_reason,
            "requires_root": self.requires_root,
            "risk": self.risk,
            "before": dict(self.before),
            "after": dict(self.after),
            "commands": list(self.commands),
            "errors": list(self.errors),
            "returncodes": dict(self.returncodes),
            "stdout_tail": list(self.stdout_tail),
            "stderr_tail": list(self.stderr_tail),
        }


class AccelerationAction(ABC):
    id: str
    title: str
    category: str
    why: str
    risk: str = "low"
    requires_root: bool = False
    platforms: list[str] = ["linux", "windows", "macos"]
    profile_min: str = "minimal"

    def is_platform_supported(self, ctx: ExecutionContext) -> bool:
        if ctx.is_linux and "linux" in self.platforms:
            return True
        if ctx.is_windows and "windows" in self.platforms:
            return True
        if ctx.is_macos and "macos" in self.platforms:
            return True
        return False

    @abstractmethod
    def check(self, ctx: ExecutionContext) -> tuple[bool, dict[str, Any], list[str]]:
        raise NotImplementedError

    @abstractmethod
    def plan(self, ctx: ExecutionContext) -> tuple[bool, list[str], dict[str, Any], list[str]]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, ctx: ExecutionContext) -> AccelerationActionResult:
        raise NotImplementedError

    def rollback(self, ctx: ExecutionContext) -> AccelerationActionResult | None:  # pragma: no cover
        return None


def profile_gte(profile: str, minimum: str) -> bool:
    return PROFILE_ORDER.get(profile, -1) >= PROFILE_ORDER.get(minimum, -1)


def normalize_profile(profile: str) -> str:
    candidate = profile.strip().lower()
    if candidate not in PROFILE_ORDER:
        return "balanced"
    return candidate


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    tokens = {part.strip().lower() for part in value.split(",") if part.strip()}
    return tokens or None


def state_root(cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    return base / ".hydra" / "state"


__all__ = [
    "ACCELERATE_SCHEMA_VERSION",
    "PROFILE_ORDER",
    "ExecutionContext",
    "ActionDescriptor",
    "AccelerationPlan",
    "AccelerationActionResult",
    "AccelerationAction",
    "profile_gte",
    "normalize_profile",
    "parse_csv_set",
    "state_root",
]
