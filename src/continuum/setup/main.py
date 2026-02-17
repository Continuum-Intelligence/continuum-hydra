from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    import typer
except Exception:  # pragma: no cover
    class _TyperShim:
        class Exit(Exception):
            def __init__(self, code: int = 0) -> None:
                self.code = code

        @staticmethod
        def Option(default, *args, **kwargs):  # noqa: ANN001, ANN003
            return default

        @staticmethod
        def echo(message: str, err: bool = False) -> None:
            print(message)

    typer = _TyperShim()  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

_DEFAULT_CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
_PYPI_INDEX = "https://pypi.org/simple"


def setup_command(
    with_torch: bool = typer.Option(True, "--with-torch/--no-torch", help="Install and validate PyTorch."),
    torch_spec: str | None = typer.Option(None, "--torch-spec", help='Torch package spec (e.g. "torch==2.5.*").'),
    torch_index: str | None = typer.Option(None, "--torch-index", help="Package index URL for torch wheels."),
    numpy_spec: str = typer.Option("numpy", "--numpy-spec", help='NumPy package spec (default: "numpy").'),
    upgrade: bool = typer.Option(False, "--upgrade", help="Use pip --upgrade during installs."),
    requirements: Path | None = typer.Option(None, "--requirements", help="Optional requirements file to install."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned pip commands without executing."),
    verbose: bool = typer.Option(False, "--verbose", help="Print executed commands and pip outputs."),
) -> None:
    try:
        effective_requirements = requirements
        if effective_requirements is None:
            default_requirements = Path("requirements.txt")
            if default_requirements.exists():
                effective_requirements = default_requirements

        if effective_requirements is not None and not effective_requirements.exists():
            raise ValueError(f"Requirements file not found: {effective_requirements}")

        commands = _build_install_commands(
            numpy_spec=numpy_spec,
            with_torch=with_torch,
            torch_spec=torch_spec,
            torch_index=torch_index,
            upgrade=upgrade,
            requirements=effective_requirements,
        )

        if dry_run:
            typer.echo("Dry-run mode: planned installs for active environment.")
        else:
            typer.echo("Installing packages into active environment...")
        for cmd in commands:
            _run_command(cmd, dry_run=dry_run, verbose=verbose)

        manifest = _build_manifest(with_torch=with_torch, dry_run=dry_run)
        if with_torch and not dry_run:
            _enforce_cuda_torch(manifest)
        manifest_path = _resolve_manifest_path()
        _write_manifest(manifest, manifest_path)
        requirements_path = manifest_path.parent / "requirements.txt"
        readme_path = manifest_path.parent / "README.md"
        _write_state_requirements(manifest, requirements_path)
        _write_state_readme(manifest, manifest_path, requirements_path, readme_path, dry_run=dry_run)
        _render_summary(manifest, manifest_path)

        raise typer.Exit(code=0)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Setup failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=4)


def _build_install_commands(
    *,
    numpy_spec: str,
    with_torch: bool,
    torch_spec: str | None,
    torch_index: str | None,
    upgrade: bool,
    requirements: Path | None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    commands.append([sys.executable, "-m", "pip", "--version"])

    if upgrade:
        commands.append([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    numpy_cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        numpy_cmd.append("--upgrade")
    numpy_cmd.append(numpy_spec)
    commands.append(numpy_cmd)

    if with_torch:
        effective_torch_index = torch_index or _default_torch_index()
        torch_cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            torch_cmd.append("--upgrade")
        if effective_torch_index:
            # Force torch wheel resolution from the selected index (CUDA wheels on Linux/Windows).
            torch_cmd.extend(["--index-url", effective_torch_index, "--extra-index-url", _PYPI_INDEX])
        torch_cmd.append(torch_spec or "torch")
        commands.append(torch_cmd)

    if requirements is not None:
        req_cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            req_cmd.append("--upgrade")
        req_cmd.extend(["-r", str(requirements)])
        commands.append(req_cmd)

    return commands


def _run_command(cmd: list[str], *, dry_run: bool, verbose: bool) -> None:
    if dry_run:
        typer.echo(f"DRY-RUN: {' '.join(cmd)}")
        return

    if verbose:
        typer.echo(f"$ {' '.join(cmd)}")

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if verbose:
        if completed.stdout:
            typer.echo(completed.stdout.rstrip())
        if completed.stderr:
            typer.echo(completed.stderr.rstrip(), err=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _build_manifest(*, with_torch: bool, dry_run: bool) -> dict[str, Any]:
    notes: list[str] = []
    installed: dict[str, Any] = {
        "numpy": _safe_dist_version("numpy"),
        "torch": None,
        "torch_cuda_available": None,
        "torch_cuda_version": None,
    }

    if with_torch:
        try:
            torch = import_module("torch")
            installed["torch"] = getattr(torch, "__version__", None)
            cuda = getattr(torch, "cuda", None)
            installed["torch_cuda_available"] = bool(getattr(cuda, "is_available", lambda: False)())
            version_obj = getattr(torch, "version", None)
            installed["torch_cuda_version"] = getattr(version_obj, "cuda", None)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Torch validation failed: {type(exc).__name__}: {exc}")

    if dry_run:
        notes.append("Dry-run enabled; install commands were not executed.")

    manifest: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "venv": bool(sys.prefix != getattr(sys, "base_prefix", sys.prefix)),
        "installed": installed,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "continuum_version": _safe_dist_version("continuum-intelligence"),
    }
    if notes:
        manifest["notes"] = notes
    return manifest


def _safe_dist_version(dist_name: str) -> str | None:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _default_torch_index() -> str | None:
    if platform.system().lower() == "darwin":
        return None
    return _DEFAULT_CUDA_TORCH_INDEX


def _enforce_cuda_torch(manifest: dict[str, Any]) -> None:
    if platform.system().lower() == "darwin":
        return

    installed = manifest.get("installed")
    if not isinstance(installed, dict):
        raise RuntimeError("Torch installation validation failed: missing install metadata.")

    torch_version = installed.get("torch")
    torch_cuda_version = installed.get("torch_cuda_version")
    notes = manifest.setdefault("notes", [])
    if not isinstance(notes, list):
        notes = []
        manifest["notes"] = notes

    if torch_version is None:
        raise RuntimeError("Torch installation validation failed: torch is not importable after install.")
    if torch_cuda_version is None:
        notes.append("Torch appears to be CPU-only (torch.version.cuda is null).")
        raise RuntimeError("Torch installation validation failed: expected CUDA-enabled torch build.")


def _resolve_manifest_path() -> Path:
    return Path(".continuum/state/env_manifest.json")


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_state_requirements(manifest: dict[str, Any], path: Path) -> None:
    installed = manifest.get("installed", {})
    lines: list[str] = []
    if isinstance(installed, dict):
        numpy_version = installed.get("numpy")
        torch_version = installed.get("torch")
        if isinstance(numpy_version, str) and numpy_version.strip():
            lines.append(f"numpy=={numpy_version}")
        else:
            lines.append("numpy")
        if isinstance(torch_version, str) and torch_version.strip():
            lines.append(f"torch=={torch_version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_state_readme(
    manifest: dict[str, Any],
    manifest_path: Path,
    requirements_path: Path,
    readme_path: Path,
    *,
    dry_run: bool,
) -> None:
    installed = manifest.get("installed", {})
    numpy_version = installed.get("numpy") if isinstance(installed, dict) else None
    torch_version = installed.get("torch") if isinstance(installed, dict) else None
    cuda_available = installed.get("torch_cuda_available") if isinstance(installed, dict) else None
    lines = [
        "# Continuum Setup State",
        "",
        "This directory contains reproducibility artifacts from `continuum setup`.",
        "",
        "## Files",
        f"- `env_manifest.json`: captured environment and install metadata",
        f"- `requirements.txt`: pinned package snapshot for setup-installed ML deps",
        "",
        "## Snapshot",
        f"- python_version: {manifest.get('python_version')}",
        f"- platform: {manifest.get('platform')}",
        f"- architecture: {manifest.get('architecture')}",
        f"- numpy: {'null' if numpy_version is None else numpy_version}",
        f"- torch: {'null' if torch_version is None else torch_version}",
        f"- torch_cuda_available: {'null' if cuda_available is None else cuda_available}",
        "",
        "## Re-apply",
        f"`{sys.executable} -m pip install -r {requirements_path}`",
        "",
        "## Manifest Path",
        str(manifest_path),
    ]
    if dry_run:
        lines.extend(["", "Note: setup was run with `--dry-run`; no pip installs were executed."])
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_summary(manifest: dict[str, Any], manifest_path: Path) -> None:
    installed = manifest.get("installed", {})
    numpy_version = installed.get("numpy") if isinstance(installed, dict) else None
    torch_version = installed.get("torch") if isinstance(installed, dict) else None
    torch_cuda_available = installed.get("torch_cuda_available") if isinstance(installed, dict) else None

    if Console is not None and Table is not None:
        console = Console()
        table = Table(title="Continuum Setup Summary")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("python_version", str(manifest.get("python_version")))
        table.add_row("numpy_version", "null" if numpy_version is None else str(numpy_version))
        table.add_row("torch_version", "null" if torch_version is None else str(torch_version))
        table.add_row(
            "torch_cuda_available",
            "null" if torch_cuda_available is None else str(torch_cuda_available),
        )
        table.add_row("manifest_path", str(manifest_path))
        table.add_row("requirements_path", str(manifest_path.parent / "requirements.txt"))
        table.add_row("readme_path", str(manifest_path.parent / "README.md"))
        console.print(table)
        return

    typer.echo(f"python_version: {manifest.get('python_version')}")
    typer.echo(f"numpy_version: {'null' if numpy_version is None else numpy_version}")
    typer.echo(f"torch_version: {'null' if torch_version is None else torch_version}")
    typer.echo(f"torch_cuda_available: {'null' if torch_cuda_available is None else torch_cuda_available}")
    typer.echo(f"manifest_path: {manifest_path}")
    typer.echo(f"requirements_path: {manifest_path.parent / 'requirements.txt'}")
    typer.echo(f"readme_path: {manifest_path.parent / 'README.md'}")


__all__ = ["setup_command"]
