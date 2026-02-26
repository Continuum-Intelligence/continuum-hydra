# continuum-hydra

Hydra is a performance-first ML systems toolkit under the Continuum infrastructure initiative.

It ships five production CLI flows:
- `continuum doctor`: static system/runtime diagnostics before training
- `continuum profile`: static + sustained benchmark profiling with bottleneck analysis and remediation
- `continuum accelerate`: system tuning / optimization planning and optional apply
- `continuum launch`: training script runtime with checkpoint-aware auto-restart / auto-resume
- `continuum setup`: environment bootstrap for NumPy/PyTorch with reproducibility state artifacts

## Design Principles

- Structured diagnostics with explicit `PASS`/`WARN`/`FAIL`/`SKIP` states
- No silent system modification in `doctor`/`profile` flows
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
continuum --help
continuum accelerate --help
continuum launch --help
continuum doctor --help
continuum profile --help
continuum setup --help
```

Also available as a direct profiler entrypoint:

```bash
continuum-profile --help
```

## Feature Inventory

### `continuum doctor` checks (all implemented)

- `environment.python_version`
- `environment.venv`
- `environment.runtime`
- `driver.nvidia_smi`
- `gpu.nvml_available`
- `gpu.nvml_devices`
- `runtime.gpu_passthrough`
- `gpu.persistence_mode`
- `gpu.clock_throttle_reasons`
- `cuda.driver_version`
- `cuda.toolkit_nvcc`
- `cuda.torch_cuda_version`
- `cuda.driver_cuda_compat`
- `cuda.runtime_hint`
- `pytorch.installed`
- `pytorch.cuda_available`
- `pytorch.cuda_version`
- `gpu.device_properties`
- `nccl.env_config`
- `nccl.torch_backend`
- `system.dev_shm`

### `continuum profile` modules (all implemented)

- Static profile collection (`cpu`, `memory`, `storage`, `os`, `runtime`, `notes`)
- Sustained CPU benchmark
- Sustained memory bandwidth benchmark (NumPy path + stdlib fallback)
- Sustained GPU benchmark (CUDA/MPS when available)
- Disk random I/O benchmark
- Bottleneck classifier (`primary_bottleneck`, `secondary_bottleneck`, confidence, reasons, recommendations)
- Remediation action generator (`priority`, `actions`)
- Unified human + JSON formatting layer

### `continuum accelerate` modules (all implemented)

- Acceleration plan builder (environment-aware recommendations)
- Built-in tuning action: CPU governor tuning (Linux)
- Built-in tuning action: NVIDIA persistence mode tuning (Linux)
- Built-in tuning action: process priority / `nice` / `ionice` suggestions
- Plugin loading from `.hydra/launch.d` (`.py` + shell hooks)
- Interactive action selection UI
- Plan/apply reporting with JSON state artifacts

### `continuum launch` runtime features (all implemented)

- Python training script execution with pass-through script args
- Runtime log capture (`.hydra/launch/runs/.../launch.log`)
- Runtime report JSON (`report.json` + `.hydra/state/launch_latest.json`)
- Checkpoint discovery in `checkpoints/`, `outputs/`, `runs/`
- Auto-restart with checkpoint-based `--resume` injection
- Dry-run mode for command/report preview

### `continuum setup` capabilities (all implemented)

- Optional install/validation of PyTorch (`--with-torch/--no-torch`)
- Custom torch package spec and index (`--torch-spec`, `--torch-index`)
- Custom NumPy package spec (`--numpy-spec`)
- Optional upgrade behavior (`--upgrade`)
- Optional requirements file install (`--requirements`)
- Dry-run planning mode (`--dry-run`)
- Manifest and reproducibility artifact generation:
  - `.continuum/state/env_manifest.json`
  - `.continuum/state/requirements.txt`
  - `.continuum/state/README.md`
- Linux CUDA torch enforcement (fails setup when torch is CPU-only on Linux)

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
continuum doctor --verbose
```

Doctor coverage:
- Environment/runtime (Python, isolation, container/WSL hints)
- Driver/GPU discovery (`nvidia-smi`, NVML, visibility, passthrough hints)
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

Profiler command examples:

```bash
continuum profile
continuum profile --static-only
continuum profile --benchmarks static,cpu,memory,gpu,disk
continuum profile --no-static
continuum profile --no-benchmarks

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

Profiler output behavior:

- Terminal output:
  - One combined table covering static profile, benchmarks, analysis, and remediation.
  - Rows carry `PASS`/`WARN`/`FAIL` status tags for readability.
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
- Missing backends/dependencies produce partial/null payloads plus explanatory notes.

Profile exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | Successful run |
| `2` | Invalid CLI value (for example unknown benchmark or output format) |
| `4` | Tool-level crash/unhandled failure |

## `continuum accelerate`

Accelerate focuses on system tuning and optimization recommendations before launching training.

Common usage:

```bash
continuum accelerate
continuum accelerate --dry-run
continuum accelerate --apply
continuum accelerate --interactive
continuum accelerate --profile balanced
continuum accelerate --only gpu,cpu
continuum accelerate --exclude process
continuum accelerate --json
continuum accelerate --out /tmp/accelerate.json
continuum accelerate --no-state-write
continuum accelerate --no-timestamp
continuum accelerate --verbose
```

Accelerate output:
- Human-readable plan/apply summary
- JSON report (stdout with `--json`, and/or `.hydra/state/launch_latest.json`)

Accelerate exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | Successful plan/apply run |
| `2` | Usage error / invalid option value |
| `1` | Runtime failure |

## `continuum launch`

Launch runs a training script and manages checkpoint-aware restart/resume behavior.

Common usage:

```bash
continuum launch train.py
continuum launch train.py --max-restarts 3
continuum launch train.py --no-auto-resume
continuum launch train.py --dry-run --json
continuum launch train.py --out /tmp/launch_report.json
continuum launch train.py -- --output-dir ./outputs/run1
```

Launch runtime artifacts:
- `.hydra/launch/runs/<run_id>/launch.log`
- `.hydra/launch/runs/<run_id>/report.json`
- `.hydra/state/launch_latest.json` (unless `--no-state-write`)

Launch exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | Training completed successfully (or dry-run success) |
| `1` | Training/runtime failure |
| `2` | Usage error (for example missing script path) |
| `130` | Interrupted by user |

## `continuum setup`

Setup installs baseline ML dependencies into the active Python environment and records reproducibility metadata.

Common usage:

```bash
continuum setup
continuum setup --dry-run
continuum setup --no-torch
continuum setup --torch-spec "torch==2.5.*" --torch-index https://download.pytorch.org/whl/cu121
continuum setup --numpy-spec "numpy==1.26.4"
continuum setup --upgrade
continuum setup --requirements requirements.txt
continuum setup --verbose
```

Setup artifacts:
- `.continuum/state/env_manifest.json`: environment snapshot + installed package metadata
- `.continuum/state/requirements.txt`: pinned requirements snapshot generated from installed package versions
- `.continuum/state/README.md`: local replay instructions and snapshot summary

Setup exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | Successful run |
| `2` | Invalid user input (for example missing requirements file path) |
| `4` | Install/validation/runtime failure |

## Testing

Run full suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Profiler-focused tests include:
- `tests/test_profile_main.py`
- `tests/test_profile_static.py`
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
- `tests/test_gpu_props_checks.py`

Setup-focused tests include:
- `tests/test_setup_main.py`

Smoke flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[profile]
continuum doctor --deterministic --json --no-write
continuum accelerate --dry-run --json
continuum launch test_training/train_100m_mmfine_reason.py --dry-run --json
continuum profile --static-only --json --no-write
continuum profile --benchmarks static,cpu,memory,gpu,disk --output-format both
continuum setup --dry-run
```

## Status

![Status](https://img.shields.io/badge/status-doctor%2Bprofile%2Baccelerate%2Blaunch%2Bsetup-active-green)
![Doctor](https://img.shields.io/badge/doctor-checks%2021-implemented-blue)
![Profiler](https://img.shields.io/badge/profiler-modules%208-implemented-blue)
![Accelerate](https://img.shields.io/badge/accelerate-tuning%20planner-implemented-blue)
![Launch](https://img.shields.io/badge/launch-runtime%20resume-implemented-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
