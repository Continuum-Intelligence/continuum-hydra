from __future__ import annotations

from typing import Any


def format_human_lines(status: dict[str, Any]) -> list[str]:
    lines = [
        f"active={status.get('active', False)}",
        f"platform={status.get('platform', 'unknown')}",
        f"timestamp={status.get('timestamp', '')}",
    ]

    changes = status.get("changes_applied", [])
    lines.append(f"changes_applied={len(changes)}")
    for change in changes:
        name = change.get("name", "unknown")
        result = change.get("result", "unknown")
        message = change.get("message", "")
        lines.append(f"- {name}: {result}{(' - ' + message) if message else ''}")

    failures = status.get("failures", [])
    if failures:
        lines.append(f"failures={len(failures)}")
        for failure in failures:
            lines.append(f"- failure: {failure}")

    return lines


def render_status(status: dict[str, Any], verbose: bool = False) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Continuum Accelerate Status")
        table.add_column("Key", no_wrap=True)
        table.add_column("Value")

        table.add_row("Active", str(status.get("active", False)))
        table.add_row("Platform", str(status.get("platform", "unknown")))
        table.add_row("Timestamp", str(status.get("timestamp", "")))
        table.add_row("Changes", str(len(status.get("changes_applied", []))))
        table.add_row("Failures", str(len(status.get("failures", []))))

        console.print(table)

        if verbose:
            for line in format_human_lines(status):
                console.print(line)
        return
    except Exception:  # noqa: BLE001
        pass

    for line in format_human_lines(status):
        print(line)


__all__ = ["render_status", "format_human_lines"]
