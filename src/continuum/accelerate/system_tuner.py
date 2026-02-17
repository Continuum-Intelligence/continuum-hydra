from __future__ import annotations

import os
import platform
import re
import resource
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run_cmd(command: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"{type(exc).__name__}: {exc}"


def detect_context() -> dict[str, Any]:
    system = platform.system().lower()
    is_linux = system == "linux"
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    nvidia_smi = shutil.which("nvidia-smi")

    return {
        "platform": system,
        "is_linux": is_linux,
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_root": bool(is_root),
        "nvidia_smi": nvidia_smi,
        "nvidia_present": nvidia_smi is not None,
    }


def _read_cpu_governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return None


def _read_swappiness() -> int | None:
    path = Path("/proc/sys/vm/swappiness")
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:  # noqa: BLE001
        return None


def _read_nvidia_persistence() -> str | None:
    code, out, _err = _run_cmd(["nvidia-smi", "-q", "-d", "PERFORMANCE"])
    if code != 0:
        return None
    match = re.search(r"Persistence Mode\s*:\s*(Enabled|Disabled)", out, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).lower()


def _read_windows_power_plan() -> str | None:
    code, out, _err = _run_cmd(["powercfg", "/getactivescheme"])
    if code != 0:
        return None
    match = re.search(r"([0-9a-fA-F\-]{36})", out)
    return match.group(1) if match else None


def capture_previous_state(ctx: dict[str, Any], cpu_only: bool, gpu_only: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "nice": os.nice(0) if hasattr(os, "nice") else None,
        "rlimit_nofile": None,
    }

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        state["rlimit_nofile"] = {"soft": int(soft), "hard": int(hard)}
    except Exception:  # noqa: BLE001
        state["rlimit_nofile"] = None

    if ctx["is_linux"] and not gpu_only:
        state["cpu_governor"] = _read_cpu_governor()
        state["swappiness"] = _read_swappiness()

    if ctx["nvidia_present"] and not cpu_only:
        state["nvidia_persistence_mode"] = _read_nvidia_persistence()

    if ctx["is_windows"] and not gpu_only:
        try:
            import psutil  # type: ignore

            state["process_priority"] = int(psutil.Process().nice())
        except Exception:  # noqa: BLE001
            state["process_priority"] = None
        state["power_plan_guid"] = _read_windows_power_plan()

    return state


def apply_acceleration(
    ctx: dict[str, Any],
    previous_state: dict[str, Any],
    dry_run: bool,
    cpu_only: bool,
    gpu_only: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    changes: list[dict[str, Any]] = []
    failures: list[str] = []

    def add_change(name: str, result: str, message: str, command: str | None = None) -> None:
        payload = {"name": name, "result": result, "message": message}
        if command is not None:
            payload["command"] = command
        changes.append(payload)

    if ctx["is_linux"] and not gpu_only:
        governor = previous_state.get("cpu_governor")
        if governor is None:
            add_change("cpu_governor", "skipped", "governor path unavailable")
        elif dry_run:
            add_change("cpu_governor", "planned", "would set governor to performance", "cpupower frequency-set -g performance")
        elif not ctx["is_root"]:
            add_change("cpu_governor", "skipped", "root required")
        elif shutil.which("cpupower") is None:
            add_change("cpu_governor", "skipped", "cpupower not installed")
        else:
            code, _out, err = _run_cmd(["cpupower", "frequency-set", "-g", "performance"])
            if code == 0:
                add_change("cpu_governor", "applied", "set to performance", "cpupower frequency-set -g performance")
            else:
                failures.append(f"cpu_governor: {err or 'unknown error'}")
                add_change("cpu_governor", "failed", err or "unknown error")

        # process nice for current process only (reversible in-session)
        if hasattr(os, "nice"):
            if dry_run:
                add_change("process_nice", "planned", "would raise process priority (nice -5)")
            else:
                try:
                    os.nice(-5)
                    add_change("process_nice", "applied", "raised process priority")
                except Exception as exc:  # noqa: BLE001
                    add_change("process_nice", "skipped", f"insufficient permission: {exc}")

        swappiness = previous_state.get("swappiness")
        if swappiness is None:
            add_change("swappiness", "skipped", "swappiness not available")
        elif dry_run:
            add_change("swappiness", "planned", "would set vm.swappiness=10", "sysctl -w vm.swappiness=10")
        elif not ctx["is_root"]:
            add_change("swappiness", "skipped", "root required")
        else:
            code, _out, err = _run_cmd(["sysctl", "-w", "vm.swappiness=10"])
            if code == 0:
                add_change("swappiness", "applied", "vm.swappiness set to 10", "sysctl -w vm.swappiness=10")
            else:
                failures.append(f"swappiness: {err or 'unknown error'}")
                add_change("swappiness", "failed", err or "unknown error")

        # ulimit soft value for current process only.
        limits = previous_state.get("rlimit_nofile") or {}
        soft = limits.get("soft")
        hard = limits.get("hard")
        if soft is None or hard is None:
            add_change("ulimit_nofile", "skipped", "rlimit unavailable")
        elif dry_run:
            add_change("ulimit_nofile", "planned", "would raise soft open-file limit")
        else:
            target = min(int(hard), max(int(soft), 65535))
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, int(hard)))
                add_change("ulimit_nofile", "applied", f"soft limit set to {target}")
            except Exception as exc:  # noqa: BLE001
                add_change("ulimit_nofile", "skipped", f"unable to set rlimit: {exc}")

    if ctx["is_windows"] and not gpu_only:
        if dry_run:
            changes.append({"name": "windows_process_priority", "result": "planned", "message": "would set HIGH priority"})
        else:
            try:
                import psutil  # type: ignore

                process = psutil.Process()
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                changes.append({"name": "windows_process_priority", "result": "applied", "message": "set HIGH priority"})
            except Exception as exc:  # noqa: BLE001
                changes.append({"name": "windows_process_priority", "result": "skipped", "message": f"{exc}"})

        if dry_run:
            changes.append({"name": "windows_power_plan", "result": "planned", "message": "would set high performance power plan"})
        else:
            code, _out, err = _run_cmd(["powercfg", "/setactive", "SCHEME_MIN"])
            if code == 0:
                changes.append({"name": "windows_power_plan", "result": "applied", "message": "set high performance power plan"})
            else:
                changes.append({"name": "windows_power_plan", "result": "skipped", "message": err or "unable to change power plan"})

    if ctx["nvidia_present"] and not cpu_only:
        if dry_run:
            changes.append({"name": "nvidia_persistence", "result": "planned", "message": "would enable persistence mode", "command": "nvidia-smi -pm 1"})
        elif not ctx["is_root"]:
            changes.append({"name": "nvidia_persistence", "result": "skipped", "message": "root/admin may be required"})
        else:
            code, _out, err = _run_cmd(["nvidia-smi", "-pm", "1"])
            if code == 0:
                changes.append({"name": "nvidia_persistence", "result": "applied", "message": "enabled persistence mode", "command": "nvidia-smi -pm 1"})
            else:
                changes.append({"name": "nvidia_persistence", "result": "skipped", "message": err or "unable to enable persistence mode"})
    elif not gpu_only:
        changes.append({"name": "nvidia_persistence", "result": "skipped", "message": "nvidia-smi not found"})

    return changes, failures


def restore_acceleration(
    ctx: dict[str, Any],
    previous_state: dict[str, Any],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    changes: list[dict[str, Any]] = []
    failures: list[str] = []

    def add_change(name: str, result: str, message: str, command: str | None = None) -> None:
        payload = {"name": name, "result": result, "message": message}
        if command is not None:
            payload["command"] = command
        changes.append(payload)

    if ctx["is_linux"]:
        governor = previous_state.get("cpu_governor")
        if governor:
            if dry_run:
                add_change("cpu_governor", "planned", f"would restore governor={governor}")
            elif ctx["is_root"] and shutil.which("cpupower"):
                code, _out, err = _run_cmd(["cpupower", "frequency-set", "-g", str(governor)])
                if code == 0:
                    add_change("cpu_governor", "restored", f"restored governor={governor}")
                else:
                    failures.append(f"cpu_governor restore: {err or 'unknown error'}")
                    add_change("cpu_governor", "failed", err or "unknown error")
            else:
                add_change("cpu_governor", "skipped", "root/cpupower unavailable for restore")

        swappiness = previous_state.get("swappiness")
        if swappiness is not None:
            if dry_run:
                add_change("swappiness", "planned", f"would restore vm.swappiness={swappiness}")
            elif ctx["is_root"]:
                code, _out, err = _run_cmd(["sysctl", "-w", f"vm.swappiness={int(swappiness)}"])
                if code == 0:
                    add_change("swappiness", "restored", f"restored vm.swappiness={swappiness}")
                else:
                    add_change("swappiness", "failed", err or "unknown error")
            else:
                add_change("swappiness", "skipped", "root required for restore")

        limits = previous_state.get("rlimit_nofile") or {}
        soft = limits.get("soft")
        hard = limits.get("hard")
        if soft is not None and hard is not None:
            if dry_run:
                add_change("ulimit_nofile", "planned", f"would restore soft={soft}, hard={hard}")
            else:
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (int(soft), int(hard)))
                    add_change("ulimit_nofile", "restored", f"restored soft={soft}, hard={hard}")
                except Exception as exc:  # noqa: BLE001
                    add_change("ulimit_nofile", "skipped", f"unable to restore rlimit: {exc}")

    if ctx["nvidia_present"]:
        previous = previous_state.get("nvidia_persistence_mode")
        if previous in {"enabled", "disabled"}:
            target = "1" if previous == "enabled" else "0"
            if dry_run:
                add_change("nvidia_persistence", "planned", f"would restore persistence={previous}")
            elif ctx["is_root"]:
                code, _out, err = _run_cmd(["nvidia-smi", "-pm", target])
                if code == 0:
                    add_change("nvidia_persistence", "restored", f"restored persistence={previous}")
                else:
                    add_change("nvidia_persistence", "failed", err or "unknown error")
            else:
                add_change("nvidia_persistence", "skipped", "root/admin may be required for restore")

    if ctx["is_windows"]:
        priority = previous_state.get("process_priority")
        if priority is not None:
            if dry_run:
                add_change("windows_process_priority", "planned", f"would restore priority={priority}")
            else:
                try:
                    import psutil  # type: ignore

                    psutil.Process().nice(int(priority))
                    add_change("windows_process_priority", "restored", f"restored priority={priority}")
                except Exception as exc:  # noqa: BLE001
                    add_change("windows_process_priority", "skipped", f"unable to restore priority: {exc}")

        power_plan = previous_state.get("power_plan_guid")
        if power_plan:
            if dry_run:
                add_change("windows_power_plan", "planned", f"would restore power plan={power_plan}")
            else:
                code, _out, err = _run_cmd(["powercfg", "/setactive", str(power_plan)])
                if code == 0:
                    add_change("windows_power_plan", "restored", f"restored power plan={power_plan}")
                else:
                    add_change("windows_power_plan", "skipped", err or "unable to restore power plan")

    return changes, failures


__all__ = [
    "detect_context",
    "capture_previous_state",
    "apply_acceleration",
    "restore_acceleration",
]
