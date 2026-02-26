from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from continuum.evals.reporting.markdown import build_compare_markdown, write_text
from continuum.evals.runner import compare_runs, run_suite
from continuum.evals.suites.loader import SuiteError, init_core_suite, load_suite

evals_app = typer.Typer(help="Local, file-based eval suites for RAG and prompt regression testing.", no_args_is_help=True)


def _console() -> Any:
    try:
        from rich.console import Console

        return Console(stderr=True)
    except Exception:  # pragma: no cover
        return None


def _eprint(message: str) -> None:
    typer.echo(message, err=True)


@evals_app.command("init")
def evals_init() -> None:
    path = init_core_suite(Path.cwd())
    _eprint(f"Initialized eval suite scaffold at `{path}`")
    _eprint("Next steps:")
    _eprint("1. Edit hydra-evals/suites/core.yaml")
    _eprint("2. Run: continuum evals run --suite core --model dummy")
    raise typer.Exit(code=0)


@evals_app.command("run")
def evals_run(
    suite: str = typer.Option(..., "--suite", help="Suite name from hydra-evals/suites/<suite>.yaml"),
    model: str = typer.Option(..., "--model", help="Model spec: dummy or http(s) endpoint base URL"),
    fail_under: float | None = typer.Option(None, "--fail-under", min=0.0, max=1.0, help="Exit 3 if pass_rate < threshold."),
    json_output: bool = typer.Option(False, "--json", help="Print summary JSON to stdout."),
    verbose: bool = typer.Option(False, "--verbose", help="Print per-case progress to stderr."),
) -> None:
    try:
        suite_obj = load_suite(suite, Path.cwd())
        console = _console()

        def progress_cb(idx: int, total: int, row: dict[str, Any]) -> None:
            if console is not None:
                status = "PASS" if row.get("pass") else "FAIL"
                console.print(f"[{idx}/{total}] {row.get('case_id')} {status}")
            elif verbose:
                _eprint(f"[{idx}/{total}] {row.get('case_id')} {'PASS' if row.get('pass') else 'FAIL'}")

        summary, _results, run_dir = run_suite(
            suite_name=suite,
            suite=suite_obj,
            model_spec=model,
            cwd=Path.cwd(),
            verbose=verbose,
            progress_cb=progress_cb if verbose else None,
        )

        if json_output:
            print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _eprint(f"Evals run complete: {summary['run_id']}")
            _eprint(f"Artifacts: {run_dir}")
            _eprint(f"Pass rate: {summary['pass_rate']:.3f} ({summary['pass_count']}/{summary['total_cases']})")

        if fail_under is not None and float(summary.get("pass_rate", 0.0)) < fail_under:
            raise typer.Exit(code=3)
        raise typer.Exit(code=0)
    except SuiteError as exc:
        _eprint(f"Evals suite error: {exc}")
        raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _eprint(f"Evals run failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)


@evals_app.command("compare")
def evals_compare(
    left: str = typer.Option(..., "--left", help="Left run ID or path"),
    right: str = typer.Option(..., "--right", help="Right run ID or path"),
    format: str = typer.Option("markdown", "--format", help="Output format (markdown)"),
) -> None:
    try:
        data = compare_runs(left, right, Path.cwd())
        left_summary = data["left_summary"]
        right_summary = data["right_summary"]
        flipped = {"pass_to_fail": data["pass_to_fail"], "fail_to_pass": data["fail_to_pass"]}

        if format != "markdown":
            raise ValueError(f"Unsupported format: {format}")

        text = build_compare_markdown(left_summary, right_summary, flipped)
        out_name = f"evals-compare-{Path(str(left_summary.get('run_id', 'left'))).name}-vs-{Path(str(right_summary.get('run_id', 'right'))).name}.md"
        out_path = Path.cwd() / out_name
        write_text(out_path, text)
        _eprint(text.rstrip())
        _eprint(f"Wrote compare report: {out_path}")
        raise typer.Exit(code=0)
    except FileNotFoundError as exc:
        _eprint(str(exc))
        raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _eprint(f"Evals compare failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)


__all__ = ["evals_app"]
