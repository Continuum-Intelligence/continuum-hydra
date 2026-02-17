from __future__ import annotations

from typing import Any


def generate_remediation(report: dict[str, Any]) -> dict[str, Any]:
    analysis = report.get("analysis")
    if not isinstance(analysis, dict):
        return _empty_remediation(priority="low")

    primary = analysis.get("primary_bottleneck")
    confidence = _to_float(analysis.get("confidence"))
    priority = _priority_from_confidence(confidence)

    actions = _actions_for_primary(str(primary) if primary is not None else None)
    return {
        "priority": priority,
        "actions": actions,
    }


def _actions_for_primary(primary: str | None) -> list[dict[str, str]]:
    if primary == "memory_bandwidth":
        return [
            _action("Switch to fp16/bf16 precision", "high", "medium", "Reduced bytes per operation lowers memory pressure."),
            _action("Use fused attention/kernels", "high", "high", "Kernel fusion reduces memory traffic and launch overhead."),
            _action("Increase arithmetic intensity", "medium", "medium", "Larger batch/matmul can improve compute-to-memory ratio."),
            _action("Use dataset sharding", "medium", "medium", "Sharding improves data locality and reduces input stalls."),
        ]
    if primary == "gpu_instability":
        return [
            _action("Check power/thermal limits", "high", "medium", "Instability patterns often align with thermal or power throttling."),
            _action("Close background GPU processes", "medium", "low", "Background load can create throughput variance."),
            _action("Test reduced matrix size", "medium", "low", "Smaller workloads help verify throttling sensitivity."),
        ]
    if primary == "cpu_instability":
        return [
            _action("Close background CPU processes", "high", "low", "CPU contention increases benchmark variance."),
            _action("Check power governor/mode", "medium", "medium", "Aggressive power saving can destabilize sustained throughput."),
            _action("Limit BLAS thread count", "medium", "medium", "Oversubscribed threads can cause oscillating performance."),
        ]
    if primary == "cpu_compute":
        return [
            _action("Install optimized BLAS libraries", "high", "medium", "Vectorized kernels can improve CPU throughput significantly."),
            _action("Avoid Python loops in hot paths", "high", "medium", "Interpreter overhead limits sustained CPU compute."),
            _action("Vectorize workloads", "high", "medium", "Batch/vector operations improve CPU utilization."),
        ]
    if primary == "gpu_compute":
        return [
            _action("Verify correct backend (cuda/mps)", "high", "low", "Incorrect backend selection can cap GPU performance."),
            _action("Enable mixed precision where safe", "high", "medium", "Reduced precision often increases GPU throughput."),
            _action("Confirm model/device placement", "high", "low", "Host-side execution prevents expected GPU utilization."),
        ]
    if primary == "disk_io":
        return [
            _action("Use NVMe-class storage", "high", "high", "Low random I/O throughput indicates storage bottlenecks."),
            _action("Use sharded dataset formats", "high", "medium", "Sharded formats reduce random small-file access overhead."),
            _action("Increase DataLoader workers and prefetch", "medium", "medium", "Parallel prefetch can hide random I/O latency."),
        ]
    return []


def _action(title: str, impact: str, difficulty: str, reason: str) -> dict[str, str]:
    return {
        "title": title,
        "impact": impact,
        "difficulty": difficulty,
        "reason": reason,
    }


def _priority_from_confidence(confidence: float | None) -> str:
    value = confidence if confidence is not None else 0.0
    if value >= 0.7:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _empty_remediation(priority: str = "low") -> dict[str, Any]:
    return {
        "priority": priority,
        "actions": [],
    }


__all__ = ["generate_remediation"]
