"""
Berkelium CLI — Textual TUI entry point.

Shows graph stats on startup, provides a button to build or update the code
graph, displays real-time progress during extraction, and shows repo path + git
branch in a status bar at the bottom.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Static
from textual_pyfiglet import FigletWidget


def _get_version() -> str:
    try:
        return importlib.metadata.version("berkelium")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _get_db_path() -> Path:
    return Path.cwd() / ".berkelium" / "graph.db"


def _detect_git_branch(cwd: Path) -> str:
    """Return the current git branch name, or 'not a git repo' on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "not a git repo"


class BerkeliumCLI(App):
    """Berkelium CLI — Textual TUI for code graph management."""

    TITLE = "Berkelium CLI"
    SUB_TITLE = "Context Engineering Tool"
    CSS_PATH = "styles.tcss"
    BINDINGS = [Binding("q", "quit", "Quit")]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        version = _get_version()

        yield FigletWidget(
            "Berkelium CLI",
            font="ansi_shadow",
            justify="center",
            colors=["#e6a08f", "#e05d38"],
            animate=True,
            classes="bkc-title",
        )
        yield Static(
            f"V{version} (Beta)",
            classes="bkc-subtitle",
        )

        with Container(id="content-area"):
            yield Static(
                "No code graph found. Click 'Build code graph' to index the current directory.",
                id="no-graph-msg",
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        cwd = Path.cwd()
        branch = _detect_git_branch(cwd)
        self.query_one("#status-bar", Static).update(
            f" {cwd}  |  branch: {branch}  |  q: Quit"
        )
        self._refresh_graph_state()


def main() -> None:
    """Run the Berkelium CLI app."""
    app = BerkeliumCLI()
    app.run()


if __name__ == "__main__":
    main()
