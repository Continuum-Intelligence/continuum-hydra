from __future__ import annotations

try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table
except Exception:  # pragma: no cover
    class Console:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def print(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            print(*args)

    class Table:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.rows: list[tuple[str, ...]] = []

        def add_column(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def add_row(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.rows.append(tuple(str(arg) for arg in args))

    class Prompt:  # type: ignore[no-redef]
        @staticmethod
        def ask(prompt: str, default: str = "") -> str:
            rendered = f"{prompt} [{default}]: " if default else f"{prompt}: "
            response = input(rendered)
            return response if response else default

from continuum.launch.models import ActionDescriptor


def select_actions_interactively(
    recommendations: list[ActionDescriptor],
    console: Console | None = None,
) -> set[str]:
    active_console = console or Console()
    default_selected = {
        rec.action_id for rec in recommendations if rec.recommended and rec.supported and rec.risk.lower() != "high"
    }

    table = Table(title="Accelerate Actions")
    table.add_column("#", no_wrap=True)
    table.add_column("Selected", no_wrap=True)
    table.add_column("ID")
    table.add_column("Category", no_wrap=True)
    table.add_column("Risk", no_wrap=True)
    table.add_column("Root", no_wrap=True)
    table.add_column("Title")

    for idx, rec in enumerate(recommendations, start=1):
        selected = "[x]" if rec.action_id in default_selected else "[ ]"
        table.add_row(str(idx), selected, rec.action_id, rec.category, rec.risk, "yes" if rec.requires_root else "no", rec.title)

    active_console.print(table)
    raw = Prompt.ask(
        "Select actions (all | none | comma-separated indexes/ids)",
        default="all" if default_selected else "none",
    ).strip()

    if raw.lower() == "all":
        return {rec.action_id for rec in recommendations}
    if raw.lower() == "none":
        return set()

    selected_ids: set[str] = set()
    by_index = {str(i): rec.action_id for i, rec in enumerate(recommendations, start=1)}
    known_ids = {rec.action_id for rec in recommendations}

    for part in [token.strip() for token in raw.split(",") if token.strip()]:
        if part in by_index:
            selected_ids.add(by_index[part])
        elif part in known_ids:
            selected_ids.add(part)

    return selected_ids


__all__ = ["select_actions_interactively"]
