from __future__ import annotations

from time import perf_counter
from typing import Any


def run_benchmarks(static_only: bool) -> list[dict[str, Any]]:
    if static_only:
        return [
            {
                "name": "benchmark.suite",
                "status": "WARN",
                "message": "Benchmarks skipped due to --static-only.",
                "result": None,
                "unit": None,
                "duration_ms": 0.0,
            }
        ]

    results: list[dict[str, Any]] = []

    try:
        started = perf_counter()
        iterations = 200_000
        acc = 0
        for i in range(iterations):
            acc += i ^ 3
        elapsed = (perf_counter() - started) * 1000.0
        ops_per_sec = int(iterations / max(elapsed / 1000.0, 1e-9))
        results.append(
            {
                "name": "benchmark.cpu_loop_ops",
                "status": "PASS",
                "message": "CPU integer-loop throughput measured.",
                "result": ops_per_sec,
                "unit": "ops/s",
                "duration_ms": round(elapsed, 3),
                "details": {"iterations": iterations, "checksum": acc & 0xFFFF},
            }
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            {
                "name": "benchmark.cpu_loop_ops",
                "status": "FAIL",
                "message": f"CPU loop benchmark failed: {type(exc).__name__}: {exc}",
                "result": None,
                "unit": None,
                "duration_ms": 0.0,
            }
        )

    try:
        started = perf_counter()
        size = 4 * 1024 * 1024
        source = bytearray(size)
        dest = bytearray(size)
        rounds = 16
        for _ in range(rounds):
            dest[:] = source
        elapsed = (perf_counter() - started) * 1000.0
        total_mb = (size * rounds) / (1024 * 1024)
        mbps = round(total_mb / max(elapsed / 1000.0, 1e-9), 3)
        results.append(
            {
                "name": "benchmark.memory_copy",
                "status": "PASS",
                "message": "In-memory copy throughput measured.",
                "result": mbps,
                "unit": "MB/s",
                "duration_ms": round(elapsed, 3),
                "details": {"bytes_per_round": size, "rounds": rounds},
            }
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            {
                "name": "benchmark.memory_copy",
                "status": "FAIL",
                "message": f"Memory benchmark failed: {type(exc).__name__}: {exc}",
                "result": None,
                "unit": None,
                "duration_ms": 0.0,
            }
        )

    return results


__all__ = ["run_benchmarks"]
