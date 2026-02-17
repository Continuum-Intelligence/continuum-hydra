from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from continuum.accelerate.launcher import launch_training_script
from continuum.accelerate.models import AccelerationActionResult, ExecutionContext, parse_csv_set
from continuum.accelerate.plan_builder import build_plan
from continuum.accelerate.plugins.loader import PluginLoadResult, run_shell_hooks
from continuum.accelerate.reporting import build_report, render_summary, write_state_report
from continuum.accelerate.ui.interactive import select_actions_interactively

Profile = Literal["minimal", "balanced", "max", "expert"]


class UsageError(Exception):
    pass


def _eprint(message: str) -> None:
    typer.echo(message, err=True)


def _render_plan(plan_dict: dict, console: Console) -> None:
    table = Table(title=f"Hydra Launch Plan ({plan_dict['profile']})")
    table.add_column("Recommended", no_wrap=True)
    table.add_column("Supported", no_wrap=True)
    table.add_column("ID")
    table.add_column("Category", no_wrap=True)
    table.add_column("Risk", no_wrap=True)
    table.add_column("Root", no_wrap=True)

    for item in sorted(plan_dict.get("recommendations", []), key=lambda rec: rec.get("action_id", "")):
        table.add_row(
            "yes" if item.get("recommended") else "no",
            "yes" if item.get("supported") else "no",
            item.get("action_id", ""),
            item.get("category", ""),
            item.get("risk", ""),
            "yes" if item.get("requires_root") else "no",
        )

    console.print(table)


def _build_dry_run_results(plan_dict: dict) -> list[AccelerationActionResult]:
    results: list[AccelerationActionResult] = []
    for item in sorted(plan_dict.get("recommendations", []), key=lambda rec: rec.get("action_id", "")):
        results.append(
            AccelerationActionResult(
                action_id=item["action_id"],
                title=item["title"],
                supported=bool(item["supported"]),
                applied=False,
                skipped_reason="Dry run - not applied",
                requires_root=bool(item["requires_root"]),
                risk=item["risk"],
                before={},
                after={},
                commands=list(item.get("commands", [])),
                errors=[],
            )
        )
    return results


def _auto_selection(plan_dict: dict, expert_mode: bool) -> set[str]:
    selected: set[str] = set()
    for item in plan_dict.get("recommendations", []):
        if not item.get("recommended") or not item.get("supported"):
            continue
        if item.get("risk", "").lower() == "high" and not expert_mode:
            continue
        selected.add(item["action_id"])
    return selected


def _is_supported_os(ctx: ExecutionContext) -> bool:
    return ctx.is_linux or ctx.is_windows or ctx.is_macos


def _parse_mode_flags(dry_run: bool, apply: bool) -> bool:
    if dry_run and apply:
        raise UsageError("Cannot pass both --dry-run and --apply")
    if not dry_run and not apply:
        return True
    return dry_run


def _validate_filter_option(name: str, value: str | None, known_categories: set[str]) -> set[str] | None:
    if value is None:
        return None
    parsed = parse_csv_set(value)
    if parsed is None:
        raise UsageError(f"Malformed {name}; expected comma-separated categories")

    unknown = sorted(token for token in parsed if token not in known_categories)
    if unknown:
        raise UsageError(f"Unknown categories in {name}: {', '.join(unknown)}")
    return parsed


def _write_report_if_enabled(report: dict, out: Path | None, no_state_write: bool) -> None:
    if no_state_write:
        if out is not None:
            write_state_report(report, out=out, cwd=Path.cwd())
        return
    write_state_report(report, out=out, cwd=Path.cwd())


def _print_json_stdout(report: dict) -> None:
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


def _run_plan_mode(
    *,
    dry_run: bool,
    apply: bool,
    interactive: bool,
    profile: Profile,
    only: str | None,
    exclude: str | None,
    json_output: bool,
    out: Path | None,
    verbose: bool,
    quiet_human: bool,
    no_state_write: bool,
    no_timestamp: bool,
    console: Console,
) -> int:
    effective_dry_run = _parse_mode_flags(dry_run=dry_run, apply=apply)
    expert_mode = profile == "expert"

    probe_plan, _, _, _ = build_plan(
        profile=profile,
        only=None,
        exclude=None,
        expert_mode=expert_mode,
        include_timestamp=not no_timestamp,
        cwd=Path.cwd(),
    )
    known_categories = {rec.category.lower() for rec in probe_plan.recommendations}

    only_set = _validate_filter_option("--only", only, known_categories)
    exclude_set = _validate_filter_option("--exclude", exclude, known_categories)

    plan, internal_data, ctx, plugin_result = build_plan(
        profile=profile,
        only=only_set,
        exclude=exclude_set,
        expert_mode=expert_mode,
        include_timestamp=not no_timestamp,
        cwd=Path.cwd(),
    )

    if verbose:
        _eprint(f"Detected categories: {', '.join(sorted(known_categories))}")
        _eprint(f"Plugin files loaded: {', '.join(plugin_result.loaded_files) if plugin_result.loaded_files else '<none>'}")
        if plugin_result.failures:
            _eprint(f"Plugin load failures: {len(plugin_result.failures)}")

    if not _is_supported_os(ctx):
        report = build_report(
            plan=plan,
            action_results=[],
            ctx=ctx,
            selected_action_ids=set(),
            dry_run=effective_dry_run,
            plugin_result=plugin_result,
            hook_warnings=["Skipped: not supported on this OS."],
        )
        _write_report_if_enabled(report, out, no_state_write)
        if json_output:
            _print_json_stdout(report)
        elif not quiet_human:
            _eprint("Skipped: not supported on this OS.")
        return 0

    plan_dict = plan.to_dict()
    if not quiet_human:
        _render_plan(plan_dict, console)

    selected_ids: set[str]
    hook_warnings: list[str] = []

    if effective_dry_run:
        selected_ids = _auto_selection(plan_dict, expert_mode)
        results = _build_dry_run_results(plan_dict)
        report = build_report(
            plan=plan,
            action_results=results,
            ctx=ctx,
            selected_action_ids=selected_ids,
            dry_run=True,
            plugin_result=plugin_result,
            hook_warnings=[],
        )
        _write_report_if_enabled(report, out, no_state_write)
        if not quiet_human:
            render_summary(report, console)
        if json_output:
            _print_json_stdout(report)
        return 0

    if json_output and interactive:
        raise UsageError("--interactive cannot be used with --json")

    if interactive:
        selected_ids = select_actions_interactively(plan.recommendations, console=console)
        if not typer.confirm("Apply selected actions?", default=False):
            if not quiet_human:
                _eprint("Apply cancelled by user.")
            return 0
    else:
        selected_ids = _auto_selection(plan_dict, expert_mode)

    plan_payload = plan.to_dict()
    ctx_payload = ctx.to_dict()

    hook_warnings.extend(run_shell_hooks(plugin_result.hooks.pre_apply_shell, ctx_payload, plan_payload, selected_ids))
    for callback in plugin_result.hooks.pre_apply_py:
        try:
            callback(ctx_payload, plan_payload, selected_ids)
        except Exception as exc:  # noqa: BLE001
            hook_warnings.append(f"Python pre hook failed: {type(exc).__name__}: {exc}")

    results: list[AccelerationActionResult] = []
    for item in internal_data:
        action = item["action"]
        supported = bool(item["supported"])

        if action.id not in selected_ids:
            results.append(
                AccelerationActionResult(
                    action_id=action.id,
                    title=action.title,
                    supported=supported,
                    applied=False,
                    skipped_reason="Not selected",
                    requires_root=action.requires_root,
                    risk=action.risk,
                    before=item.get("before", {}),
                    after=item.get("after_preview", {}),
                    commands=list(item.get("commands", [])),
                    errors=[],
                )
            )
            continue

        if not supported:
            results.append(
                AccelerationActionResult(
                    action_id=action.id,
                    title=action.title,
                    supported=False,
                    applied=False,
                    skipped_reason="Unsupported on this environment",
                    requires_root=action.requires_root,
                    risk=action.risk,
                    before=item.get("before", {}),
                    after=item.get("before", {}),
                    commands=list(item.get("commands", [])),
                    errors=[],
                )
            )
            continue

        try:
            results.append(action.apply(ctx))
        except Exception as exc:  # noqa: BLE001
            results.append(
                AccelerationActionResult(
                    action_id=action.id,
                    title=action.title,
                    supported=True,
                    applied=False,
                    skipped_reason="Action apply raised an exception",
                    requires_root=action.requires_root,
                    risk=action.risk,
                    before=item.get("before", {}),
                    after=item.get("before", {}),
                    commands=list(item.get("commands", [])),
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            )

    hook_warnings.extend(run_shell_hooks(plugin_result.hooks.post_apply_shell, ctx_payload, plan_payload, selected_ids))
    for callback in plugin_result.hooks.post_apply_py:
        try:
            callback(ctx_payload, plan_payload, selected_ids)
        except Exception as exc:  # noqa: BLE001
            hook_warnings.append(f"Python post hook failed: {type(exc).__name__}: {exc}")

    report = build_report(
        plan=plan,
        action_results=results,
        ctx=ctx,
        selected_action_ids=selected_ids,
        dry_run=False,
        plugin_result=plugin_result,
        hook_warnings=hook_warnings,
    )
    _write_report_if_enabled(report, out, no_state_write)

    if not quiet_human:
        render_summary(report, console)

    if report.get("summary", {}).get("applied", 0) == 0:
        _eprint("Warning: --apply completed but no actions were applied.")

    if json_output:
        _print_json_stdout(report)

    return 0


def launch_command(
    ctx: typer.Context,
    script: Path | None = typer.Argument(None, help="Training script to run (example: train.py)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only and do not apply actions / execute training."),
    apply: bool = typer.Option(False, "--apply", help="Apply selected/recommended actions in planner mode."),
    interactive: bool = typer.Option(False, "--interactive", help="Interactively choose actions in planner mode."),
    profile: Profile = typer.Option("balanced", "--profile", help="minimal|balanced|max|expert"),
    only: str | None = typer.Option(None, "--only", help="Comma-separated categories to include (planner mode)."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated categories to exclude (planner mode)."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON report to stdout only."),
    out: Path | None = typer.Option(None, "--out", help="Write report JSON to this path."),
    verbose: bool = typer.Option(False, "--verbose", help="Print detection and plugin details to stderr."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress human-readable output."),
    no_state_write: bool = typer.Option(False, "--no-state-write", help="Do not write .hydra/state/launch_latest.json."),
    no_timestamp: bool = typer.Option(False, "--no-timestamp", help="Disable timestamps for deterministic planner JSON outputs."),
    max_restarts: int = typer.Option(1, "--max-restarts", min=0, help="Maximum auto-resume restarts in script mode."),
    auto_resume: bool = typer.Option(True, "--auto-resume/--no-auto-resume", help="Attempt automatic resume from latest checkpoint in script mode."),
    debug: bool = typer.Option(False, "--debug", help="Print debug argv tracing for script mode."),
) -> None:
    console = Console(stderr=True)
    quiet_human = quiet or json_output

    try:
        if script is not None:
            if not script.exists() or not script.is_file():
                raise UsageError(f"Training script not found: {script}")
            script_args = list(ctx.args)
            if dry_run and apply:
                raise UsageError("Cannot pass both --dry-run and --apply")
            runtime_dry_run = dry_run
            exit_code, _report = launch_training_script(
                script=script,
                script_args=script_args,
                cwd=Path.cwd(),
                max_restarts=max_restarts,
                auto_resume=auto_resume,
                quiet=quiet_human,
                verbose=verbose,
                json_output=json_output,
                out=out,
                no_state_write=no_state_write,
                dry_run=runtime_dry_run,
                debug=debug,
            )
            raise typer.Exit(code=exit_code)

        exit_code = _run_plan_mode(
            dry_run=dry_run,
            apply=apply,
            interactive=interactive,
            profile=profile,
            only=only,
            exclude=exclude,
            json_output=json_output,
            out=out,
            verbose=verbose,
            quiet_human=quiet_human,
            no_state_write=no_state_write,
            no_timestamp=no_timestamp,
            console=console,
        )
        raise typer.Exit(code=exit_code)
    except KeyboardInterrupt:
        _eprint("Interrupted")
        raise typer.Exit(code=130)
    except UsageError as exc:
        _eprint(f"Usage error: {exc}")
        raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _eprint(f"Launch failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)


__all__ = ["launch_command"]
