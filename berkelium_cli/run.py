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

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Container, VerticalScroll
from textual.widgets import Button, DataTable, ProgressBar, Static
from textual_pyfiglet import FigletWidget

from berkelium_cli.extractor import CodebaseExtractor
from berkelium_cli.store import GraphQLiteStore


def _get_version() -> str:
    try:
        return importlib.metadata.version("berkelium-cli")
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

        with VerticalScroll(id="content-area"):
            yield Static(
                "No code graph found. Click 'Build code graph' to index the current directory.",
                id="no-graph-msg",
            )
            yield DataTable(id="stats-table", show_cursor=False)

        with Container(id="action-area"):
            yield Button("Build code graph", id="action-btn", variant="primary")

        with Container(id="progress-area"):
            yield Static("", id="progress-label")
            with Center(id="progress-bar-row"):
                yield ProgressBar(id="progress-bar", total=100, show_eta=True)

        yield Static("", id="status-bar", classes="bkc-status")

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

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "action-btn":
            return
        is_update = str(event.button.label) == "Update code graph"
        event.button.disabled = True
        self.query_one("#progress-area").display = True
        self.query_one("#progress-label", Static).update("Preparing…")
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=100, progress=0)
        self._run_graph_worker(is_update)

    # ------------------------------------------------------------------
    # Background worker (runs in a thread — extraction is synchronous)
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def _run_graph_worker(self, is_update: bool) -> None:
        root = Path.cwd()
        db_path = _get_db_path()
        store = GraphQLiteStore(str(db_path))
        try:

            def on_progress(rel_path: str, current: int, total: int) -> None:
                self.call_from_thread(self._update_progress, rel_path, current, total)

            if is_update:
                self._worker_update(root, store, on_progress)
            else:
                self._worker_build(root, store, on_progress)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._on_build_error, str(exc))
        finally:
            store.close()

    def _worker_build(
        self,
        root: Path,
        store: GraphQLiteStore,
        on_progress: object,
    ) -> None:
        extractor = CodebaseExtractor(root, store=store, progress_callback=on_progress)
        nodes, edges = extractor.extract()
        self.call_from_thread(
            self._on_build_success,
            f"Built: {len(nodes)} nodes, {len(edges)} edges.",
        )

    def _worker_update(
        self,
        root: Path,
        store: GraphQLiteStore,
        on_progress: object,
    ) -> None:
        from berkelium_cli.sync import IncrementalSync, NotAGitRepoError

        try:
            sync = IncrementalSync(root, store)
            result = sync.sync("HEAD", progress_callback=on_progress)
            msg = (
                f"Updated: {result.files_parsed} file(s) parsed, "
                f"{result.nodes_added} node(s) added."
            )
            if result.errors:
                msg += f" ({len(result.errors)} warning(s))"
            self.call_from_thread(self._on_build_success, msg)
        except NotAGitRepoError:
            self.call_from_thread(
                self._update_progress_label, "Not a git repo — running full build…"
            )
            self._worker_build(root, store, on_progress)

    # ------------------------------------------------------------------
    # UI update helpers (called from main thread via call_from_thread)
    # ------------------------------------------------------------------

    def _update_progress(self, rel_path: str, current: int, total: int) -> None:
        bar = self.query_one("#progress-bar", ProgressBar)
        label = self.query_one("#progress-label", Static)
        bar.update(total=total, progress=current)
        label.update(f"[{current}/{total}]  {rel_path}")

    def _update_progress_label(self, text: str) -> None:
        self.query_one("#progress-label", Static).update(text)

    def _on_build_success(self, summary: str) -> None:
        self.query_one("#progress-area").display = False
        btn = self.query_one("#action-btn", Button)
        btn.disabled = False
        self._refresh_graph_state()
        self.notify(summary, title="Done", severity="information")

    def _on_build_error(self, message: str) -> None:
        self.query_one("#progress-area").display = False
        btn = self.query_one("#action-btn", Button)
        btn.disabled = False
        self.notify(f"Error: {message}", title="Build failed", severity="error")

    # ------------------------------------------------------------------
    # Graph state helpers
    # ------------------------------------------------------------------

    def _refresh_graph_state(self) -> None:
        db_path = _get_db_path()
        try:
            store = GraphQLiteStore(str(db_path))
            try:
                stats = store.stats()
                has_graph = stats.get("node_count", 0) > 0
                self.query_one("#no-graph-msg").display = not has_graph
                self.query_one("#stats-table").display = has_graph
                btn = self.query_one("#action-btn", Button)
                btn.label = "Update code graph" if has_graph else "Build code graph"
                if has_graph:
                    self._populate_stats_table(store, stats)
            finally:
                store.close()
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), title="Store error", severity="warning")

    def _populate_stats_table(self, store: GraphQLiteStore, stats: dict) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Metric", "Count")

        node_count = stats.get("node_count", 0)
        edge_count = stats.get("edge_count", 0)

        def _count(cypher: str) -> str:
            rows = store.query(cypher)
            if rows:
                val = next(iter(rows[0].values()), 0)
                return str(val) if val is not None else "0"
            return "0"

        file_count = _count("MATCH (n) WHERE n.kind = 'File' RETURN count(n) AS c")
        class_count = _count("MATCH (n) WHERE n.kind = 'Class' RETURN count(n) AS c")
        fn_count = _count(
            "MATCH (n) WHERE n.kind = 'Function' OR n.kind = 'Method' RETURN count(n) AS c"
        )

        table.add_rows(
            [
                ("Total Nodes", str(node_count)),
                ("Total Edges", str(edge_count)),
                ("Files", file_count),
                ("Classes", class_count),
                ("Functions / Methods", fn_count),
            ]
        )


def main() -> None:
    """Run the Berkelium CLI app."""
    app = BerkeliumCLI()
    app.run()


if __name__ == "__main__":
    main()
