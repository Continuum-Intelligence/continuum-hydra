# continuum-hydra

> Stop guessing why your training is slow. Hydra diagnoses, optimizes, and runs your ML environment, from Python version to kernel-level settings.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platform](https://img.shields.io/badge/platform-linux&windows%20%7C%20macOS-lightgrey)
![Continuum Hydra demo](continuum-hydra-default.gif)

**Hydra is a performance-first ML systems toolkit** that gives engineers infrastructure-grade visibility into their training environment. It ships five CLI flows: diagnose your system, benchmark and profile it, apply kernel-level optimizations, launch training with automatic checkpoint-aware restart, and bootstrap reproducible environments.

Part of the [Continuum Intelligence](https://github.com/Continuum-Intelligence) infrastructure initiative.

---

## Why Hydra?

ML training failures are rarely about your model. They are about your environment: mismatched CUDA versions, CPU throttling, missing checkpoint recovery, degraded memory bandwidth, or a PyTorch install that is not using the hardware you think it is.

Hydra gives you a single CLI to catch and fix those issues before they waste GPU hours or derail long-running jobs.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .\.venv\Scripts\activate      # Windows PowerShell

pip install -U pip
pip install -e .

# Optional: profiling extras
pip install -e .[profile]
```

---

## The five commands

### `continuum doctor` - diagnose your environment

Safe and read-only. Checks Python version, venv isolation, CUDA compatibility, PyTorch install, GPU health, NCCL config, and related runtime signals. Produces a structured `PASS` / `WARN` / `FAIL` report.

```bash
continuum doctor
continuum doctor --only gpu,cuda
continuum doctor --json --export /tmp/reports
```

### `continuum profile` - benchmark your system

Runs static profiling plus sustained CPU, memory, GPU, and disk benchmarks. Identifies the primary bottleneck and generates remediation recommendations.

```bash
continuum profile
continuum profile --benchmarks cpu,memory,gpu,disk
continuum profile --output-format both
```

### `continuum accelerate` - apply optimizations

Builds an environment-aware tuning plan: CPU governor, NVIDIA persistence mode, process priority, and related runtime actions. Preview with `--dry-run` or apply directly.

```bash
continuum accelerate --dry-run
continuum accelerate --apply
continuum accelerate --interactive
```

### `continuum launch` - training with auto-resume

Runs your training script with checkpoint-aware restart and resume behavior. If training crashes, Hydra can discover the latest checkpoint and continue from it.

```bash
continuum launch train.py
continuum launch train.py --max-restarts 3
continuum launch train.py -- --output-dir ./outputs/run1
```

### `continuum setup` - reproducible environment bootstrap

Installs PyTorch and NumPy into the active environment and generates reproducibility artifacts so the environment can be recreated later.

```bash
continuum setup
continuum setup --torch-spec "torch==2.5.*" --torch-index https://download.pytorch.org/whl/cu121
continuum setup --dry-run
```

---

## macOS support

Hydra runs on macOS. CUDA and NVIDIA-specific checks commonly report `WARN` or `SKIP` on Apple Silicon or systems without NVIDIA drivers, which is expected. CPU, memory, and disk benchmarks still run fully. Some `continuum accelerate` actions are Linux-specific and will show as unsupported in dry-run plans.

---

## Output artifacts

Hydra writes structured JSON artifacts for CI integration, reproducibility, and debugging:

```text
.hydra/reports/doctor_YYYYMMDD_HHMMSS.json
.hydra/reports/profile_YYYYMMDD_HHMMSS.json
.hydra/state/launch_latest.json
.continuum/state/env_manifest.json
.continuum/state/requirements.txt
```

---

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Quick smoke test:

```bash
continuum doctor --deterministic --json --no-write
continuum profile --static-only --json --no-write
continuum accelerate --dry-run --json
continuum launch test_training/train_100m_mmfine_reason.py --dry-run --json
continuum setup --dry-run
```

---

## License

Apache 2.0 - see [LICENSE](LICENSE).

Part of the Continuum Intelligence infrastructure initiative.
