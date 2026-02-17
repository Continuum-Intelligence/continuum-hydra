from __future__ import annotations

import os
import random
import tempfile
from time import perf_counter
from typing import Any


_MB = 1024 * 1024


def run_disk_benchmark(context: dict[str, Any]) -> dict[str, Any]:
    notes = context.setdefault("notes", [])
    if bool(context.get("static_only")):
        notes.append("Disk random I/O benchmark skipped due to --static-only.")
        return _empty_payload()
    if bool(context.get("no_disk")):
        notes.append("Disk random I/O benchmark skipped due to --no-disk.")
        return _empty_payload()

    warmup_sec = _as_positive_float(context.get("disk_warmup"), default=2.0)
    duration_sec = _as_positive_float(context.get("disk_duration"), default=8.0)
    size_mb = _as_positive_int(context.get("disk_size_mb"), default=256)
    file_size = max(1 * _MB, size_mb * _MB)
    block_size = 4 * 1024

    file_path: str | None = None
    try:
        fd, file_path = tempfile.mkstemp(prefix="continuum_disk_", suffix=".bin")
        os.close(fd)

        with open(file_path, "wb") as f:
            chunk = os.urandom(1024 * 1024)
            remaining = file_size
            while remaining > 0:
                write_n = min(len(chunk), remaining)
                f.write(chunk[:write_n])
                remaining -= write_n
            f.flush()
            os.fsync(f.fileno())

        max_offset = max(0, file_size - block_size)

        rates: list[float] = []
        iops_samples: list[float] = []
        iterations = 0

        with open(file_path, "rb", buffering=0) as f:
            warmup_end = perf_counter() + warmup_sec
            while perf_counter() < warmup_end:
                offset = random.randint(0, max_offset) if max_offset > 0 else 0
                f.seek(offset)
                _ = f.read(block_size)

            started = perf_counter()
            end_at = started + duration_sec
            while perf_counter() < end_at:
                offset = random.randint(0, max_offset) if max_offset > 0 else 0
                lap_start = perf_counter()
                f.seek(offset)
                data = f.read(block_size)
                elapsed = perf_counter() - lap_start
                if elapsed > 0 and data:
                    mb_s = (len(data) / _MB) / elapsed
                    rates.append(mb_s)
                    iops_samples.append(1.0 / elapsed)
                iterations += 1
            measured_duration = perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Disk random I/O benchmark failed: {type(exc).__name__}: {exc}")
        return _empty_payload()
    finally:
        if file_path:
            try:
                os.remove(file_path)
            except OSError:
                pass

    if not rates:
        notes.append("Disk random I/O benchmark collected zero valid iterations.")
        return _empty_payload()
    if iterations < 5:
        notes.append("Disk random I/O benchmark collected fewer than 5 iterations; variance may be noisy.")

    return {
        "disk_random_io": {
            "mean_read_mb_s": _round(_mean(rates)),
            "std_read_mb_s": _round(_std(rates)),
            "p50_read_mb_s": _round(_percentile(rates, 50.0)),
            "p95_read_mb_s": _round(_percentile(rates, 95.0)),
            "mean_iops": _round(_mean(iops_samples)),
            "iterations": int(iterations),
            "duration_sec": _round(measured_duration),
        }
    }


def _as_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if number > 0 else default


def _as_positive_int(value: Any, default: int) -> int:
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


def _empty_payload() -> dict[str, Any]:
    return {
        "disk_random_io": {
            "mean_read_mb_s": None,
            "std_read_mb_s": None,
            "p50_read_mb_s": None,
            "p95_read_mb_s": None,
            "mean_iops": None,
            "iterations": None,
            "duration_sec": None,
        }
    }


__all__ = ["run_disk_benchmark"]
