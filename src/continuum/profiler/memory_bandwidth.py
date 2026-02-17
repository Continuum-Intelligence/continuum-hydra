from __future__ import annotations

import importlib
import importlib.util
from time import perf_counter
from typing import Any


_MB = 1024 * 1024
_GB_DIVISOR = 1_000_000_000.0


def run_memory_bandwidth(context: dict[str, Any]) -> dict[str, Any]:
    notes = context.setdefault("notes", [])
    if bool(context.get("static_only")):
        notes.append("Memory bandwidth benchmark skipped due to --static-only.")
        return _empty_payload()

    warmup_sec = _as_positive_float(context.get("mem_warmup"), default=2.0)
    duration_sec = _as_positive_float(context.get("mem_duration"), default=8.0)
    mem_mb = _as_positive_int(context.get("mem_mb"), default=None)
    target_bytes = _resolve_target_bytes(context=context, mem_mb=mem_mb, numpy_available=_numpy_available())

    if _numpy_available():
        result = _run_numpy_path(target_bytes=target_bytes, warmup_sec=warmup_sec, duration_sec=duration_sec, notes=notes)
        if result is not None:
            return result

    notes.append("NumPy not installed; using stdlib bytearray copy fallback (lower fidelity).")
    return _run_stdlib_path(target_bytes=target_bytes, warmup_sec=warmup_sec, duration_sec=duration_sec, notes=notes)


def _run_numpy_path(
    target_bytes: int,
    warmup_sec: float,
    duration_sec: float,
    notes: list[str],
) -> dict[str, Any] | None:
    if importlib.util.find_spec("numpy") is None:
        return None

    try:
        np = importlib.import_module("numpy")
    except Exception:
        return None

    try:
        dtype = np.uint8
        size = max(1, int(target_bytes))
        src = np.empty(size, dtype=dtype)
        dst = np.empty(size, dtype=dtype)
        src.fill(7)
        bytes_per_iter = int(src.nbytes)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Memory bandwidth numpy allocation failed: {type(exc).__name__}: {exc}")
        return _empty_payload(bytes_per_iter=None)

    try:
        warmup_end = perf_counter() + warmup_sec
        while perf_counter() < warmup_end:
            np.copyto(dst, src)

        started = perf_counter()
        end_at = started + duration_sec
        rates: list[float] = []
        iterations = 0
        while perf_counter() < end_at:
            lap_start = perf_counter()
            np.copyto(dst, src)
            elapsed = perf_counter() - lap_start
            if elapsed > 0:
                rates.append((bytes_per_iter / elapsed) / _GB_DIVISOR)
            iterations += 1

        measured = perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Memory bandwidth numpy benchmark failed: {type(exc).__name__}: {exc}")
        return _empty_payload(bytes_per_iter=bytes_per_iter)

    return _finalize_payload(rates=rates, iterations=iterations, duration_sec=measured, bytes_per_iter=bytes_per_iter, notes=notes)


def _run_stdlib_path(
    target_bytes: int,
    warmup_sec: float,
    duration_sec: float,
    notes: list[str],
) -> dict[str, Any]:
    try:
        src = bytearray(target_bytes)
        dst = bytearray(target_bytes)
        mv_src = memoryview(src)
        mv_dst = memoryview(dst)
        bytes_per_iter = len(src)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Memory bandwidth stdlib allocation failed: {type(exc).__name__}: {exc}")
        return _empty_payload()

    try:
        warmup_end = perf_counter() + warmup_sec
        while perf_counter() < warmup_end:
            mv_dst[:] = mv_src

        started = perf_counter()
        end_at = started + duration_sec
        rates: list[float] = []
        iterations = 0
        while perf_counter() < end_at:
            lap_start = perf_counter()
            mv_dst[:] = mv_src
            elapsed = perf_counter() - lap_start
            if elapsed > 0:
                rates.append((bytes_per_iter / elapsed) / _GB_DIVISOR)
            iterations += 1

        measured = perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Memory bandwidth stdlib benchmark failed: {type(exc).__name__}: {exc}")
        return _empty_payload(bytes_per_iter=bytes_per_iter)

    return _finalize_payload(rates=rates, iterations=iterations, duration_sec=measured, bytes_per_iter=bytes_per_iter, notes=notes)


def _finalize_payload(
    rates: list[float],
    iterations: int,
    duration_sec: float,
    bytes_per_iter: int | None,
    notes: list[str],
) -> dict[str, Any]:
    if not rates:
        notes.append("Memory bandwidth benchmark collected zero valid iterations.")
        return _empty_payload(bytes_per_iter=bytes_per_iter)

    if iterations < 5:
        notes.append("Memory bandwidth benchmark collected fewer than 5 iterations; variance may be noisy.")

    return {
        "memory_bandwidth": {
            "mean_gbps": _round(_mean(rates)),
            "std_gbps": _round(_std(rates)),
            "p50_gbps": _round(_percentile(rates, 50.0)),
            "p95_gbps": _round(_percentile(rates, 95.0)),
            "iterations": int(iterations),
            "duration_sec": _round(duration_sec),
            "bytes_per_iter": int(bytes_per_iter) if isinstance(bytes_per_iter, int) else None,
        }
    }


def _resolve_target_bytes(context: dict[str, Any], mem_mb: int | None, numpy_available: bool) -> int:
    if mem_mb is not None and mem_mb > 0:
        return mem_mb * _MB

    total_ram = _extract_total_ram(context)
    if isinstance(total_ram, int) and total_ram > 0:
        target = int(total_ram * 0.05)
        capped = min(256 * _MB, max(64 * _MB, target))
        return capped

    # No RAM hint available.
    if numpy_available:
        return 128 * _MB
    return 64 * _MB


def _extract_total_ram(context: dict[str, Any]) -> int | None:
    facts = context.get("facts")
    if isinstance(facts, dict):
        static_profile = facts.get("static_profile")
        if isinstance(static_profile, dict):
            memory = static_profile.get("memory")
            if isinstance(memory, dict):
                total = memory.get("total_bytes")
                if isinstance(total, int) and total > 0:
                    return total
    return None


def _numpy_available() -> bool:
    return importlib.util.find_spec("numpy") is not None


def _as_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if number > 0 else default


def _as_positive_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        number = int(value)
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


def _empty_payload(bytes_per_iter: int | None = None) -> dict[str, Any]:
    return {
        "memory_bandwidth": {
            "mean_gbps": None,
            "std_gbps": None,
            "p50_gbps": None,
            "p95_gbps": None,
            "iterations": None,
            "duration_sec": None,
            "bytes_per_iter": bytes_per_iter,
        }
    }


__all__ = ["run_memory_bandwidth"]
