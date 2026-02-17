# continuum-hydra

Hydra is a performance-first ML systems toolkit under the Continuum infrastructure initiative.

It currently ships two production CLI flows:
- `continuum doctor`: static system/runtime diagnostics before training
- `continuum profile`: static + sustained benchmark profiling with analysis and remediation hints

## Design Principles

- Structured diagnostics with explicit `PASS`/`WARN`/`FAIL` states
- No silent system modification
- Reproducible JSON artifacts
- Graceful degradation when optional dependencies are missing
- Infrastructure-grade visibility for ML environments

## Prerequisites

- Python **3.10+**
- `pip` (latest recommended)
- For full GPU visibility: NVIDIA drivers (`nvidia-smi`) and/or PyTorch GPU backend

## Install

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .\.venv\Scripts\activate           # Windows PowerShell

python -m pip install -U pip
python -m pip install -e .
```

Optional profiling extras:

```bash
python -m pip install -e .[profile]
```

## CLI Overview

```bash
continuum doctor --help
continuum profile --help
```

Also available as a direct script entrypoint:

```bash
continuum-profile --help
```

## `continuum doctor`

Hydra Doctor is safe and side-effect free: it reads environment/system facts and reports health.

Common usage:

```bash
continuum doctor
continuum doctor --json
continuum doctor --export /tmp/reports
continuum doctor --no-write
continuum doctor --only gpu,cuda
continuum doctor --exclude nccl.env_config
continuum doctor --list-checks
continuum doctor --deterministic
```

Doctor coverage:
- Environment/runtime (Python, isolation, container/WSL hints)
- Driver/GPU discovery (`nvidia-smi`, NVML, visibility)
- CUDA toolkit/runtime compatibility hints
- PyTorch install/runtime checks
- GPU properties/health hints
- NCCL soft checks
- Linux `/dev/shm` system checks

Doctor output:
- Human table in terminal
- JSON artifact: `.hydra/reports/doctor_YYYYMMDD_HHMMSS.json`

Doctor JSON top-level shape:

```json
{
  "schema_version": "1.0.0",
  "environment": { "...": "..." },
  "checks": [],
  "summary": { "PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0 },
  "overall_status": "healthy",
  "total_duration_ms": 0.0
}
```

Doctor exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | Healthy (no warnings/failures/errors) |
| `1` | Warnings present |
| `2` | Failed checks or check errors |
| `4` | Tool-level crash/unhandled failure |

## `continuum profile`

Profiler produces a combined machine + sustained throughput profile.

### Implemented profiler feature set

- Feature #1: Static Profile
  - CPU model/cores, RAM, storage/root fs hints, OS/kernel, Python/Torch runtime facts
- Feature #2: Sustained CPU benchmark
  - Timed warmup + measurement loop (iter/sec stats)
- Feature #3: Sustained Memory Bandwidth benchmark
  - Timed copy throughput with NumPy path + stdlib fallback
- Feature #4: Sustained GPU compute benchmark
  - CUDA/MPS timed matmul loop (iter/sec stats), dependency-safe
- Feature #5: Bottleneck classification
  - Deterministic heuristic analysis with confidence, reasons, recommendations
- Feature #6: Remediation engine
  - Priority + actionable suggestions derived from analysis
- Feature #7: Realistic disk random I/O benchmark
  - Timed random reads with MB/s + IOPS distribution stats

### Profiler command examples

```bash
continuum profile
continuum profile --static-only
continuum profile --benchmarks static,cpu,memory,gpu,disk

continuum profile --cpu-warmup 2 --cpu-duration 8
continuum profile --mem-warmup 2 --mem-duration 8 --mem-mb 128
continuum profile --gpu-warmup 2 --gpu-duration 8 --gpu-size 4096 --gpu-dtype float16
continuum profile --disk-warmup 2 --disk-duration 8 --disk-size-mb 256

continuum profile --no-gpu --no-disk
continuum profile --output-format human
continuum profile --output-format json
continuum profile --output-format both
continuum profile --quiet
continuum profile --json
continuum profile --export /tmp/reports
continuum profile --no-write
continuum profile --verbose
```

### Profiler output behavior

- Terminal output:
  - One combined table with proper headers covering static profile, all benchmarks, analysis, and remediation.
  - Rows carry `PASS`/`WARN` status for readability.
- JSON artifact:
  - Default path: `.hydra/reports/profile_YYYYMMDD_HHMMSS.json`
  - Stable `schema_version: "1.0.0"`
  - Backward-compatible top-level shape:

```json
{
  "schema_version": "1.0.0",
  "static_profile": {},
  "benchmark_results": [],
  "benchmarks": {
    "cpu_sustained": {},
    "memory_bandwidth": {},
    "gpu_sustained": {},
    "disk_random_io": {}
  },
  "analysis": {},
  "remediation": {}
}
```

Notes:
- Optional dependencies (`torch`, `numpy`, `psutil`) never hard-crash profiling.
- Missing backends/dependencies produce partial/null payloads and explanatory notes.

## Testing

Run full suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Profiler-focused tests include:
- `tests/test_profile_main.py` (CLI routing/selection/output behavior)
- `tests/test_profile_static.py` (static profile probes and formatter fallback)
- `tests/test_cpu_benchmark.py`
- `tests/test_memory_bandwidth.py`
- `tests/test_gpu_benchmark.py`
- `tests/test_disk_benchmark.py`
- `tests/test_analysis_bottleneck.py`
- `tests/test_remediation.py`

Doctor-focused tests include:
- `tests/test_runner.py`
- `tests/test_doctor_v02_integration.py`
- `tests/test_system_checks.py`
- `tests/test_gpu_checks.py`
- `tests/test_cuda_checks.py`
- `tests/test_pytorch_checks.py`
- `tests/test_nccl_checks.py`

Smoke flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[profile]
continuum doctor --deterministic --json --no-write
continuum profile --static-only --json --no-write
continuum profile --benchmarks static,cpu,memory,gpu,disk --output-format both
```

## Status

![Status](https://img.shields.io/badge/status-doctor%2Bprofiler-active-green)
![Profiler](https://img.shields.io/badge/profiler-features%201--7-implemented-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
