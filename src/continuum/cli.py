from __future__ import annotations

import typer

from continuum.accelerate.cli import launch_command
from continuum.doctor.main import doctor_command
from continuum.profiler.main import profile_command
from continuum.setup.main import setup_command

app = typer.Typer(
    help="Continuum CLI — Performance-first ML infrastructure toolkit.",
    no_args_is_help=True,
)

@app.callback()
def main() -> None:
    """Root CLI group for Continuum subcommands."""


app.command(name="doctor")(doctor_command)
app.command(
    "launch",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(launch_command)
app.command(name="profile")(profile_command)
app.command(name="setup")(setup_command)

__all__ = ["app"]
