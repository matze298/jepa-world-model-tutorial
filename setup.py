#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "typer>=0.12.5",
# ]
# ///
"""Bootstrap local development for this repository."""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def ensure_venv() -> None:
    """Create the project virtual environment if it does not exist."""
    if not VENV_DIR.exists():
        _run(["uv", "venv"])


def build_app() -> typer.Typer:
    """Build the Typer CLI used after the venv is available.

    Returns:
        A Typer application with the bootstrap subcommands.
    """
    app = typer.Typer(
        add_completion=False,
        help="Bootstrap and maintain the local development environment.",
        no_args_is_help=False,
    )

    @app.command()
    def sync() -> None:
        """Sync the environment from uv.lock."""
        _run(["uv", "sync", "--all-groups", "--frozen"])

    @app.command()
    def hooks() -> None:
        """Install prek Git hooks and prepare hook environments."""
        _run(["uv", "run", "prek", "install", "--prepare-hooks"])

    @app.command(name="all")
    def setup_all() -> None:
        """Run the full local setup."""
        ensure_venv()
        sync()
        hooks()

    @app.callback(invoke_without_command=True)
    def main(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            setup_all()

    return app


def main() -> None:
    """Entry point for the bootstrap script."""
    app = build_app()
    app()


if __name__ == "__main__":
    main()
