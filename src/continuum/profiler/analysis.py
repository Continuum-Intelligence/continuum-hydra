from __future__ import annotations

from typing import Any


def classify_bottleneck(report: dict[str, Any]) -> dict[str, Any]:
    static_profile = _as_dict(report.get("static_profile"))
    benchmarks = _as_dict(report.get("benchmarks"))

    cpu = _as_dict(benchmarks.get("cpu_sustained"))
    mem = _as_dict(benchmarks.get("memory_bandwidth"))
    gpu = _as_dict(benchmarks.get("gpu_sustained"))
    disk = _as_dict(benchmarks.get("disk_random_io"))

    cpu_mean = _to_float(cpu.get("mean_iter_per_sec"))
    cpu_std = _to_float(cpu.get("std_iter_per_sec"))
    cpu_p95 = _to_float(cpu.get("p95_iter_per_sec"))

    mem_mean = _to_float(mem.get("mean_gbps"))
    mem_std = _to_float(mem.get("std_gbps"))
    mem_p95 = _to_float(mem.get("p95_gbps"))

    gpu_mean = _to_float(gpu.get("mean_iter_per_sec"))
    gpu_std = _to_float(gpu.get("std_iter_per_sec"))
    gpu_p95 = _to_float(gpu.get("p95_iter_per_sec"))
    disk_mean = _to_float(disk.get("mean_read_mb_s"))

    cpu_cv = _safe_div(cpu_std, cpu_mean)
    mem_cv = _safe_div(mem_std, mem_mean)
    gpu_cv = _safe_div(gpu_std, gpu_mean)

    cpu_stability_ratio = _safe_div(cpu_p95, cpu_mean)
    mem_stability_ratio = _safe_div(mem_p95, mem_mean)
    gpu_stability_ratio = _safe_div(gpu_p95, gpu_mean)

    os_name = str(_as_dict(static_profile.get("os")).get("name") or "")
    cpu_info = _as_dict(static_profile.get("cpu"))
    arch = str(cpu_info.get("arch") or "")
    storage = _as_dict(static_profile.get("storage"))

    mem_floor = 60.0 if ("darwin" in os_name.lower() and "arm" in arch.lower()) else 25.0
    disk_floor = _disk_floor(storage)

    signals: dict[str, Any] = {
        "cpu_cv": _rounded(cpu_cv),
        "mem_cv": _rounded(mem_cv),
        "gpu_cv": _rounded(gpu_cv),
        "cpu_stability_ratio": _rounded(cpu_stability_ratio),
        "mem_stability_ratio": _rounded(mem_stability_ratio),
        "gpu_stability_ratio": _rounded(gpu_stability_ratio),
        "mem_expected_floor_gbps": mem_floor,
        "disk_expected_floor_mb_s": disk_floor,
        "disk_mean_read_mb_s": _rounded(disk_mean),
    }

    scores = {
        "gpu_compute": 0.0,
        "memory_bandwidth": 0.0,
        "disk_io": 0.0,
        "cpu_compute": 0.0,
        "gpu_instability": 0.0,
        "cpu_instability": 0.0,
        "unknown": 0.2,
    }
    reasons: list[str] = []

    if gpu_mean is not None:
        if gpu_stability_ratio is not None and gpu_stability_ratio < 0.85:
            scores["gpu_instability"] += 0.55
            reasons.append(f"GPU stability ratio p95/mean={gpu_stability_ratio:.3f} suggests throttling/instability.")
        if gpu_cv is not None and gpu_cv > 0.20:
            scores["gpu_instability"] += 0.45
            reasons.append(f"GPU coefficient of variation {gpu_cv:.3f} indicates unstable sustained throughput.")

    if cpu_mean is not None:
        if cpu_stability_ratio is not None and cpu_stability_ratio < 0.85:
            scores["cpu_instability"] += 0.55
            reasons.append(f"CPU stability ratio p95/mean={cpu_stability_ratio:.3f} suggests scheduler/power instability.")
        if cpu_cv is not None and cpu_cv > 0.20:
            scores["cpu_instability"] += 0.45
            reasons.append(f"CPU coefficient of variation {cpu_cv:.3f} suggests scheduler contention.")

    if mem_mean is not None and mem_cv is not None and mem_cv <= 0.15 and (gpu_mean is not None or cpu_mean is not None):
        if mem_mean < mem_floor:
            scores["memory_bandwidth"] += 0.8
            reasons.append(f"Memory bandwidth mean {mem_mean:.3f} GB/s below heuristic floor {mem_floor:.1f} GB/s.")
        else:
            scores["memory_bandwidth"] += 0.1

    if cpu_mean is not None:
        cpu_iters = _to_float(cpu.get("iterations"))
        low_cpu = cpu_mean < 0.12
        mem_ok = mem_mean is None or mem_mean >= mem_floor
        gpu_ok_or_missing = gpu_mean is None or gpu_mean >= 0.05
        if low_cpu and mem_ok and gpu_ok_or_missing:
            scores["cpu_compute"] += 0.4
            reasons.append(f"CPU sustained mean {cpu_mean:.3f} iter/s is low with no strong memory pressure signal.")
            if cpu_iters is not None and cpu_iters < 5:
                reasons.append("CPU iteration count is very low; confidence in compute classification is limited.")

    if gpu_mean is not None:
        gpu_instability_score = scores["gpu_instability"]
        mem_ok = mem_mean is None or mem_mean >= mem_floor
        if gpu_instability_score < 0.35 and mem_ok and gpu_mean < 0.05:
            scores["gpu_compute"] += 0.45
            reasons.append(f"GPU sustained mean {gpu_mean:.3f} iter/s is consistently low without instability flags.")

    if disk_mean is not None:
        if disk_mean < disk_floor:
            scores["disk_io"] += 0.85
            reasons.append(f"Disk random read mean {disk_mean:.3f} MB/s below heuristic floor {disk_floor:.1f} MB/s.")
        else:
            scores["disk_io"] += 0.05

    families_present = sum(
        1
        for value in (cpu_mean, mem_mean, gpu_mean, disk_mean)
        if value is not None
    )
    missing_families = 4 - families_present
    if families_present == 0:
        reasons.append("CPU, memory, GPU, and disk sustained benchmark signals are missing.")
        scores["unknown"] += 0.4

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    second_name, second_score = ordered[1]

    primary: str | None = None
    secondary: str | None = None
    if best_name != "unknown" and best_score >= 0.35:
        primary = best_name
        if second_name != "unknown" and second_score >= 0.25:
            secondary = second_name

    confidence = 0.25
    margin = max(0.0, best_score - second_score)
    confidence += min(0.45, margin * 0.45)
    if families_present >= 2:
        confidence += 0.15
    if missing_families >= 2:
        confidence -= 0.20
    if primary is None:
        confidence = min(confidence, 0.4)
    confidence = max(0.0, min(1.0, confidence))

    recommendations = _recommendations_for(primary)

    return {
        "primary_bottleneck": primary,
        "secondary_bottleneck": secondary,
        "confidence": round(confidence, 6),
        "signals": signals,
        "reasons": reasons[:8],
        "recommendations": recommendations[:6],
    }


def _recommendations_for(primary: str | None) -> list[str]:
    if primary == "memory_bandwidth":
        return [
            "Prefer bf16/fp16 where possible; fp32 increases bandwidth pressure.",
            "Increase batch size only if it improves compute utilization without OOM.",
            "Use fused attention/kernels when available.",
        ]
    if primary == "gpu_instability":
        return [
            "Check power/thermal limits and background GPU processes.",
            "Reduce GPU matrix size or batch temporarily to confirm throttling.",
            "Keep GPU clocks/power mode stable during profiling.",
        ]
    if primary == "cpu_instability":
        return [
            "Close background processes and verify power mode/governor.",
            "Pin threads or limit BLAS threads if oversubscribed.",
            "Run profile on an idle machine for cleaner baselines.",
        ]
    if primary == "cpu_compute":
        return [
            "Install NumPy/BLAS optimizations and avoid Python-level loops.",
            "Increase compute intensity (larger fused kernels) where possible.",
            "Validate thread affinity and BLAS thread settings.",
        ]
    if primary == "gpu_compute":
        return [
            "Verify torch backend and dtype; ensure work is on cuda/mps device.",
            "Increase arithmetic intensity before increasing memory traffic.",
            "Use mixed precision and kernel fusion where safe.",
        ]
    if primary == "disk_io":
        return [
            "Use NVMe-class storage when possible for dataset reads.",
            "Use sharded dataset formats to reduce random small-file overhead.",
            "Increase DataLoader workers and enable prefetch buffering.",
        ]
    return [
        "Collect additional benchmark runs to improve confidence.",
        "Ensure CPU, memory, and GPU benchmarks are all available.",
    ]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _disk_floor(storage: dict[str, Any]) -> float:
    is_nvme = storage.get("is_nvme")
    is_ssd = storage.get("is_ssd")
    if is_nvme is True or is_ssd is True:
        return 150.0
    if is_ssd is False:
        return 40.0
    return 150.0


__all__ = ["classify_bottleneck"]
