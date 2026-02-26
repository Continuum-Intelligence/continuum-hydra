from __future__ import annotations

from pathlib import Path
from typing import Any


def build_run_markdown_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Continuum Evals Report",
        "",
        f"- Run ID: `{summary.get('run_id', '')}`",
        f"- Suite: `{summary.get('suite', '')}`",
        f"- Model: `{summary.get('model_spec', '')}`",
        f"- Pass Rate: `{summary.get('pass_rate', 0):.3f}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "total_cases",
        "pass_count",
        "pass_rate",
        "citation_present_rate",
        "grounded_rate",
        "idk_rate",
        "forbidden_hits",
    ]:
        lines.append(f"| {key} | {summary.get(key)} |")

    failures = [r for r in results if not r.get("pass")]
    if failures:
        lines.extend(["", "## Failed Cases", ""])
        for result in failures:
            lines.append(f"- `{result.get('case_id')}`: {result.get('error') or 'grading failure'}")
    return "\n".join(lines) + "\n"


def build_compare_markdown(left_summary: dict[str, Any], right_summary: dict[str, Any], flipped: dict[str, list[str]]) -> str:
    def _metric(summary: dict[str, Any], key: str) -> object:
        return summary.get(key)

    keys = ["pass_rate", "citation_present_rate", "grounded_rate", "idk_rate", "forbidden_hits"]
    lines = [
        "# Continuum Evals Compare",
        "",
        f"- Left: `{left_summary.get('run_id', left_summary.get('run_path', ''))}`",
        f"- Right: `{right_summary.get('run_id', right_summary.get('run_path', ''))}`",
        "",
        "| Metric | Left | Right | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in keys:
        left = _metric(left_summary, key)
        right = _metric(right_summary, key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta = right - left
            lines.append(f"| {key} | {left} | {right} | {delta:+.3f} |")
        else:
            lines.append(f"| {key} | {left} | {right} | n/a |")

    lines.extend(["", "## Case Flips", ""])
    lines.append(f"- pass -> fail: {', '.join(flipped['pass_to_fail']) if flipped['pass_to_fail'] else 'none'}")
    lines.append(f"- fail -> pass: {', '.join(flipped['fail_to_pass']) if flipped['fail_to_pass'] else 'none'}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


__all__ = ["build_run_markdown_report", "build_compare_markdown", "write_text"]
