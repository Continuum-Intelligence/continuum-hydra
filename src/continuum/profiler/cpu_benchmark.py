from __future__ import annotations

import importlib
import importlib.util
from time import perf_counter
from typing import Any


def run_cpu_benchmark(context: dict[str, Any]) -> dict[str, Any]:
    notes = context.setdefault("notes", [])
    if bool(context.get("static_only")):
        notes.append("CPU sustained benchmark skipped due to --static-only.")
        return _empty_payload()

    warmup_sec = _as_positive_float(context.get("cpu_warmup"), default=2.0)
    duration_sec = _as_positive_float(context.get("cpu_duration"), default=8.0)
    empty = _empty_payload()

    if importlib.util.find_spec("numpy") is None:
        notes.append("NumPy is not installed; CPU sustained benchmark skipped.")
        return empty

    try:
        np = importlib.import_module("numpy")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"NumPy import failed; CPU sustained benchmark skipped: {type(exc).__name__}: {exc}")
        return empty

    try:
        size = 2048
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Failed to allocate benchmark matrices: {type(exc).__name__}: {exc}")
        return empty

    try:
        warmup_end = perf_counter() + warmup_sec
        while perf_counter() < warmup_end:
            _ = a @ b

        iter_rates: list[float] = []
        started = perf_counter()
        end_at = started + duration_sec
        iterations = 0

        while perf_counter() < end_at:
            lap_start = perf_counter()
            _ = a @ b
            lap_elapsed = perf_counter() - lap_start
            if lap_elapsed > 0:
                iter_rates.append(1.0 / lap_elapsed)
            iterations += 1

        measured_duration = perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        notes.append(f"CPU sustained benchmark failed: {type(exc).__name__}: {exc}")
        return empty

    if not iter_rates:
        notes.append("CPU sustained benchmark collected zero valid iterations.")
        return empty

    if iterations < 5:
        notes.append("CPU sustained benchmark collected fewer than 5 iterations; variance may be noisy.")

    payload = {
        "mean_iter_per_sec": _round(_mean(iter_rates)),
        "std_iter_per_sec": _round(_std(iter_rates)),
        "p50_iter_per_sec": _round(_percentile(iter_rates, 50.0)),
        "p95_iter_per_sec": _round(_percentile(iter_rates, 95.0)),
        "iterations": int(iterations),
        "duration_sec": _round(measured_duration),
    }
    return {"cpu_sustained": payload}


def _as_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if number > 0 else default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _round(value: float) -> float:
    return round(float(value), 6)


def _empty_payload() -> dict[str, Any]:
    return {
        "cpu_sustained": {
            "mean_iter_per_sec": None,
            "std_iter_per_sec": None,
            "p50_iter_per_sec": None,
            "p95_iter_per_sec": None,
            "iterations": None,
            "duration_sec": None,
        }
    }


__all__ = ["run_cpu_benchmark"]
