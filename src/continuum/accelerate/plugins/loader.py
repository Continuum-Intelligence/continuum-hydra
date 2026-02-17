from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PrePostHook = Callable[[dict[str, Any], dict[str, Any], set[str]], None]


@dataclass(slots=True)
class HookBundle:
    pre_apply_shell: list[Path] = field(default_factory=list)
    post_apply_shell: list[Path] = field(default_factory=list)
    pre_apply_py: list[PrePostHook] = field(default_factory=list)
    post_apply_py: list[PrePostHook] = field(default_factory=list)


@dataclass(slots=True)
class PluginLoadResult:
    actions_loaded: int
    hooks: HookBundle
    warnings: list[str]
    loaded_files: list[str]
    failures: list[str]


def _load_module(file_path: Path) -> Any:
    module_name = f"continuum_launch_plugin_{file_path.stem}_{abs(hash(str(file_path))) & 0xFFFF}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plugins(register_action: Callable[[Any], None], cwd: Path | None = None) -> PluginLoadResult:
    base = cwd if cwd is not None else Path.cwd()
    plugin_dir = base / ".hydra" / "launch.d"
    hooks = HookBundle()
    warnings: list[str] = []
    loaded_files: list[str] = []
    failures: list[str] = []
    actions_loaded = 0

    if not plugin_dir.exists() or not plugin_dir.is_dir():
        return PluginLoadResult(actions_loaded=0, hooks=hooks, warnings=[], loaded_files=[], failures=[])

    files = sorted(path for path in plugin_dir.iterdir() if path.is_file())

    for file_path in files:
        try:
            if file_path.suffix == ".sh":
                loaded_files.append(file_path.name)
                if "post" in file_path.stem:
                    hooks.post_apply_shell.append(file_path)
                else:
                    hooks.pre_apply_shell.append(file_path)
                continue

            if file_path.suffix != ".py":
                continue

            loaded_files.append(file_path.name)
            module = _load_module(file_path)

            if hasattr(module, "register") and callable(module.register):
                before = actions_loaded

                def _counting_register(action: Any) -> None:
                    nonlocal actions_loaded
                    register_action(action)
                    actions_loaded += 1

                module.register(_counting_register)
                if actions_loaded == before:
                    warnings.append(f"Plugin {file_path.name} register() did not add actions")

            if file_path.name.endswith("_hook.py"):
                if hasattr(module, "pre_apply") and callable(module.pre_apply):
                    hooks.pre_apply_py.append(module.pre_apply)
                if hasattr(module, "post_apply") and callable(module.post_apply):
                    hooks.post_apply_py.append(module.post_apply)
        except Exception as exc:  # noqa: BLE001
            message = f"Plugin load failed for {file_path.name}: {type(exc).__name__}: {exc}"
            warnings.append(message)
            failures.append(message)

    return PluginLoadResult(
        actions_loaded=actions_loaded,
        hooks=hooks,
        warnings=warnings,
        loaded_files=sorted(loaded_files),
        failures=sorted(failures),
    )


def run_shell_hooks(paths: list[Path], ctx: dict[str, Any], plan: dict[str, Any], selected_ids: set[str]) -> list[str]:
    warnings: list[str] = []
    for path in paths:
        try:
            completed = subprocess.run(
                ["sh", str(path)],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env={**ctx.get("env", {}), "ACCELERATE_SELECTED_IDS": ",".join(sorted(selected_ids))},
            )
            if completed.returncode != 0:
                warnings.append(f"Hook {path.name} failed: {completed.stderr.strip() or completed.stdout.strip()}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Hook {path.name} failed: {type(exc).__name__}: {exc}")
    return warnings


__all__ = [
    "HookBundle",
    "PluginLoadResult",
    "load_plugins",
    "run_shell_hooks",
]
