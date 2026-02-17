from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
from pathlib import Path
from typing import Any


_PSUTIL_UNSET = object()
_PSUTIL_CACHE: Any = _PSUTIL_UNSET


def collect_static_profile(context: dict[str, Any]) -> dict[str, Any]:
    facts = context.setdefault("facts", {})
    cached = facts.get("static_profile")
    if isinstance(cached, dict):
        return cached

    notes: list[str] = []

    profile = {
        "cpu": _probe_cpu(notes),
        "memory": _probe_memory(notes),
        "storage": _probe_storage(notes),
        "os": _probe_os(notes),
        "runtime": _probe_runtime(notes),
        "notes": notes,
    }

    facts["static_profile"] = profile
    return profile


def _probe_cpu(notes: list[str]) -> dict[str, Any]:
    model: str | None = None

    system = platform.system()
    if system == "Linux":
        model = _cpu_model_from_proc_cpuinfo()
    elif system == "Darwin":
        model = _sysctl_value("machdep.cpu.brand_string")
    elif system == "Windows":
        model = platform.processor() or None

    if not model:
        notes.append("CPU model could not be determined.")

    physical = None
    psutil_mod = _get_psutil()
    if psutil_mod is None:
        notes.append("psutil is not installed; physical core count may be unavailable.")
    else:
        try:
            physical = psutil_mod.cpu_count(logical=False)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Physical core count probe failed: {type(exc).__name__}: {exc}")

    logical = None
    try:
        logical = os.cpu_count()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Logical CPU count probe failed: {type(exc).__name__}: {exc}")

    return {
        "model": model,
        "cores_physical": int(physical) if isinstance(physical, int) else None,
        "cores_logical": int(logical) if isinstance(logical, int) else None,
        "arch": platform.machine() or None,
    }


def _cpu_model_from_proc_cpuinfo() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            model = value.strip()
            if model:
                return model
    return None


def _probe_memory(notes: list[str]) -> dict[str, Any]:
    total: int | None = None

    psutil_mod = _get_psutil()
    if psutil_mod is not None:
        try:
            total = int(psutil_mod.virtual_memory().total)
        except Exception:
            total = None
    else:
        notes.append("psutil is not installed; using fallback memory detection.")

    if total is None:
        total = _memory_total_fallback()

    if total is None:
        notes.append("Total RAM could not be determined.")

    return {"total_bytes": total}


def _memory_total_fallback() -> int | None:
    system = platform.system()

    if system == "Linux":
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"^MemTotal:\s*(\d+)\s+kB", text, flags=re.MULTILINE)
            if match:
                return int(match.group(1)) * 1024
        except OSError:
            return None

    if system == "Darwin":
        value = _sysctl_value("hw.memsize")
        if value and value.isdigit():
            return int(value)

    if system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            return None

    return None


def _probe_storage(notes: list[str]) -> dict[str, Any]:
    root_mount = os.path.abspath(os.sep)
    root_device: str | None = None
    filesystem_type: str | None = None
    is_nvme: bool | None = None
    is_ssd: bool | None = None
    storage_notes: list[str] = []

    system = platform.system()
    if system == "Linux":
        root_device, filesystem_type = _linux_root_device_and_fs(root_mount)
    else:
        root_device, filesystem_type = _root_device_and_fs_from_psutil(root_mount)

    if root_device:
        dev_lower = root_device.lower()
        is_nvme = "nvme" in dev_lower

        if system == "Linux" and root_device.startswith("/dev/"):
            base = _linux_base_block_device(root_device)
            if base is None:
                storage_notes.append(f"Unable to map partition to base block device: {root_device}")
            else:
                rotational_path = Path("/sys/block") / base / "queue" / "rotational"
                try:
                    rotational = rotational_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if rotational == "0":
                        is_ssd = True
                    elif rotational == "1":
                        is_ssd = False
                    else:
                        storage_notes.append(f"Unexpected rotational value for {base}: {rotational}")
                except OSError:
                    storage_notes.append(f"Could not read rotational flag for {base}.")
        elif system == "Linux" and not root_device.startswith("/dev/"):
            if _is_network_filesystem(filesystem_type, root_device):
                storage_notes.append("Root mount appears to be network-backed; SSD/HDD heuristic is not applicable.")
            else:
                storage_notes.append("Root device is not a local /dev block device; SSD/HDD heuristic unavailable.")
    else:
        storage_notes.append("Root filesystem device could not be determined.")

    if filesystem_type is None:
        storage_notes.append("Root filesystem type could not be determined.")

    return {
        "root_mount": root_mount,
        "root_device": root_device,
        "is_nvme": is_nvme,
        "is_ssd": is_ssd,
        "filesystem_type": filesystem_type,
        "notes": storage_notes,
    }


def _linux_root_device_and_fs(root_mount: str) -> tuple[str | None, str | None]:
    try:
        text = Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return _root_device_and_fs_from_psutil(root_mount)

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        device, mountpoint, fstype = parts[0], parts[1], parts[2]
        if mountpoint == root_mount:
            return device, fstype

    return _root_device_and_fs_from_psutil(root_mount)


def _root_device_and_fs_from_psutil(root_mount: str) -> tuple[str | None, str | None]:
    psutil_mod = _get_psutil()
    if psutil_mod is None:
        return None, None

    try:
        parts = psutil_mod.disk_partitions(all=True)
    except Exception:
        return None, None

    selected = None
    for item in parts:
        if item.mountpoint == root_mount:
            selected = item
            break

    if selected is None:
        # Best-effort fallback for unusual mount layouts.
        for item in parts:
            if root_mount.startswith(item.mountpoint):
                selected = item
                break

    if selected is None:
        return None, None

    return getattr(selected, "device", None) or None, getattr(selected, "fstype", None) or None


def _is_network_filesystem(filesystem_type: str | None, device: str) -> bool:
    fs = (filesystem_type or "").lower()
    dev = device.lower()
    network_fs = {"nfs", "cifs", "smbfs", "sshfs", "fuse.sshfs", "glusterfs", "ceph"}
    return fs in network_fs or dev.startswith("//") or ":/" in dev


def _linux_base_block_device(device_path: str) -> str | None:
    devname = Path(device_path).name

    # nvme0n1p2 -> nvme0n1
    nvme_match = re.match(r"^(nvme\d+n\d+)p\d+$", devname)
    if nvme_match:
        return nvme_match.group(1)

    # mmcblk0p1 -> mmcblk0
    mmc_match = re.match(r"^(mmcblk\d+)p\d+$", devname)
    if mmc_match:
        return mmc_match.group(1)

    # sda2 -> sda, vda1 -> vda, xvda1 -> xvda
    sd_match = re.match(r"^([a-zA-Z]+)\d+$", devname)
    if sd_match:
        return sd_match.group(1)

    if devname:
        return devname

    return None


def _probe_os(notes: list[str]) -> dict[str, Any]:
    name = platform.system() or None
    version = platform.version() or None
    kernel = platform.release() or None

    if platform.system() == "Linux":
        os_release = _linux_os_release()
        if os_release.get("name"):
            name = os_release["name"]
        if os_release.get("version"):
            version = os_release["version"]

    if name is None:
        notes.append("OS name could not be determined.")
    if version is None:
        notes.append("OS version could not be determined.")
    if kernel is None:
        notes.append("Kernel version could not be determined.")

    return {
        "name": name,
        "version": version,
        "kernel": kernel,
    }


def _linux_os_release() -> dict[str, str | None]:
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"name": None, "version": None}

    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')

    return {
        "name": values.get("PRETTY_NAME") or values.get("NAME"),
        "version": values.get("VERSION") or values.get("VERSION_ID"),
    }


def _probe_runtime(notes: list[str]) -> dict[str, Any]:
    torch_version: str | None = None
    torch_cuda_available: bool | None = None
    torch_cuda_version: str | None = None

    spec = importlib.util.find_spec("torch")
    if spec is None:
        notes.append("Torch is not installed in the active environment.")
    else:
        try:
            torch = importlib.import_module("torch")
            torch_version = str(getattr(torch, "__version__", None)) if getattr(torch, "__version__", None) is not None else None
            torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            try:
                torch_cuda_available = bool(torch.cuda.is_available())
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Torch CUDA availability probe failed: {type(exc).__name__}: {exc}")
                torch_cuda_available = None
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Torch appears installed but could not be imported: {type(exc).__name__}: {exc}")

    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_version": torch_cuda_version,
        "platform": platform.platform(),
    }


def _sysctl_value(key: str) -> str | None:
    try:
        import subprocess

        proc = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=0.1,
            check=False,
        )
        if proc.returncode == 0:
            value = proc.stdout.strip()
            return value or None
    except Exception:
        return None
    return None


def _get_psutil() -> Any | None:
    global _PSUTIL_CACHE
    if _PSUTIL_CACHE is not _PSUTIL_UNSET:
        return _PSUTIL_CACHE

    if importlib.util.find_spec("psutil") is None:
        _PSUTIL_CACHE = None
        return None

    try:
        _PSUTIL_CACHE = importlib.import_module("psutil")
    except Exception:
        _PSUTIL_CACHE = None
    return _PSUTIL_CACHE


__all__ = ["collect_static_profile"]
