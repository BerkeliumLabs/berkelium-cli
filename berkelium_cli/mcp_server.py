"""
MCP server for berkelium-cli — exposes the codebase graph to AI coding assistants.

Provides two tools and one prompt workflow:

  build_or_update_graph  — Full extraction on first run; incremental git-diff sync thereafter.
  get_impact_radius      — Blast-radius analysis (upstream callers + downstream deps).
  review_my_pr           — Prompt workflow: sync → git diff → impact → test suggestions.

Usage (stdio transport for Claude Code / Cursor)::

    berkelium-mcp          # installed entry point
    uv run berkelium-mcp   # via uv without installing

MCP config entry (claude_desktop_config.json / .claude/settings.json)::

    {
      "mcpServers": {
        "berkelium": {
          "command": "berkelium-mcp"
        }
      }
    }
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from fastmcp import FastMCP

from berkelium_cli.extractor import CodebaseExtractor
from berkelium_cli.retriever import SurgicalRetriever
from berkelium_cli.store import GraphQLiteStore
from berkelium_cli.sync import IncrementalSync, NotAGitRepoError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "berkelium-cli-server",
    instructions=(
        "Use this server to understand structural relationships in a codebase. "
        "Always call build_or_update_graph first to ensure the graph reflects "
        "the current state of the code on disk. "
        "Then call get_impact_radius to discover which files and functions are "
        "affected by a given set of changes before writing reviews or tests."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: build_or_update_graph
# ---------------------------------------------------------------------------

@mcp.tool()
def build_or_update_graph(repo_root: str = ".") -> str:
    """
    Build or incrementally update the code knowledge graph for a repository.

    On first run (empty store) performs a full extraction of all supported source
    files (Python, JS/TS, Go, Java, Rust, C/C++).  On subsequent runs uses
    ``git diff HEAD`` to process only changed files — typically milliseconds vs
    minutes for a full rebuild.  Falls back automatically to full extraction
    when the directory is not a git repository.

    Args:
        repo_root: Absolute or relative path to the repository root.
                   Defaults to the current working directory (``"."``).

    Returns:
        A human-readable summary string describing what was done, including
        counts of files parsed, nodes added, and CALLS edges resolved.
        Returns a descriptive error string (never raises) if anything goes wrong.
    """
    # --- Resolve and validate path ------------------------------------------
    root = Path(repo_root).resolve()
    if not root.exists():
        return f"Error: repo_root '{repo_root}' does not exist."
    if not root.is_dir():
        return f"Error: repo_root '{repo_root}' is a file, not a directory."

    db_path = root / ".berkelium" / "graph.db"

    # --- Open store and decide strategy -------------------------------------
    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return f"Error: could not open graph store at '{db_path}': {exc}"

    try:
        stats = store.stats()
        is_empty = stats.get("node_count", 0) == 0

        if is_empty:
            return _full_extract(root, store)
        else:
            return _incremental_sync(root, store)
    finally:
        store.close()


def _full_extract(root: Path, store: GraphQLiteStore) -> str:
    """Run CodebaseExtractor.extract() and return a summary string."""
    try:
        extractor = CodebaseExtractor(root_path=root, store=store)
        nodes, edges = extractor.extract()
        return (
            f"Full extraction complete for '{root}': "
            f"{len(nodes)} node(s), {len(edges)} edge(s) stored."
        )
    except Exception as exc:
        logger.exception("Full extraction failed for '%s'", root)
        return f"Error during full extraction of '{root}': {exc}"


def _incremental_sync(root: Path, store: GraphQLiteStore) -> str:
    """
    Try IncrementalSync; fall back to full extraction on git errors.
    Returns a summary string.
    """
    try:
        syncer = IncrementalSync(root=root, store=store)
        result = syncer.sync(base_ref="HEAD")
        errors_note = (
            f" ({len(result.errors)} non-fatal error(s))" if result.errors else ""
        )
        return (
            f"Incremental sync complete for '{root}': "
            f"{result.files_parsed} file(s) parsed, "
            f"{result.files_skipped_hash} skipped (unchanged), "
            f"{result.nodes_added} node(s) added, "
            f"{result.call_edges_resolved} CALLS edge(s) resolved"
            f"{errors_note}."
        )
    except (NotAGitRepoError, subprocess.CalledProcessError) as exc:
        logger.info(
            "Incremental sync unavailable (%s), falling back to full extraction", exc
        )
        return _full_extract(root, store) + " (git sync unavailable; ran full extraction)"
    except Exception as exc:
        logger.exception("Incremental sync failed unexpectedly for '%s'", root)
        return f"Error during incremental sync of '{root}': {exc}"


# ---------------------------------------------------------------------------
# Tool 2: get_impact_radius
# ---------------------------------------------------------------------------

@mcp.tool()
def get_impact_radius(
    changed_files: list[str],
    repo_root: str = ".",
    depth: int = 2,
) -> dict:
    """
    Analyze the functional blast radius of a set of changed files.

    Traverses CALLS edges in the stored code graph to discover:

    - **Upstream callers**: components that call into the changed files
      (i.e. "who will break if I change this?").
    - **Downstream dependencies**: components called by the changed files
      (i.e. "what does my change depend on?").

    The graph must be populated first via ``build_or_update_graph``.

    Args:
        changed_files: List of relative file paths using forward slashes,
                       e.g. ``["src/auth.py", "src/models/user.py"]``.
        repo_root:     Path to the repository root. Defaults to ``"."``.
        depth:         Maximum number of CALLS hops to traverse in each
                       direction.  Higher values find more transitive
                       dependents but increase query time.  Default: 2.

    Returns:
        Dict with keys:

        - ``"upstream"``:   list of ``{name, qualified_name, file, kind, distance}``
          dicts for callers, sorted by hop distance then kind.
        - ``"downstream"``: same structure for callees.
        - ``"summary"``:    human-readable one-line summary.
        - ``"error"``:      present only when a non-fatal problem occurred
          (results may be partial).
    """
    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "upstream": [], "downstream": [],
            "summary": "",
            "error": f"repo_root '{repo_root}' does not exist.",
        }

    if not changed_files:
        return {
            "upstream": [], "downstream": [],
            "summary": "No changed files provided — nothing to analyse.",
        }

    db_path = root / ".berkelium" / "graph.db"

    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return {
            "upstream": [], "downstream": [],
            "summary": "",
            "error": f"Could not open graph store at '{db_path}': {exc}",
        }

    try:
        stats = store.stats()
        if stats.get("node_count", 0) == 0:
            return {
                "upstream": [], "downstream": [],
                "summary": (
                    "Graph is empty — run build_or_update_graph first."
                ),
            }

        retriever = SurgicalRetriever(store=store, max_depth=depth)
        result = retriever.get_full_impact(changed_files, max_depth=depth)

        upstream = [
            {
                "name": s.name,
                "qualified_name": s.qualified_name,
                "file": s.file_rel_path,
                "kind": s.kind,
                "distance": s.distance,
            }
            for s in result.upstream
        ]
        downstream = [
            {
                "name": s.name,
                "qualified_name": s.qualified_name,
                "file": s.file_rel_path,
                "kind": s.kind,
                "distance": s.distance,
            }
            for s in result.downstream
        ]

        summary = (
            f"Found {len(upstream)} upstream caller(s) and "
            f"{len(downstream)} downstream dependency/dependencies "
            f"across {depth} hop(s) for {len(changed_files)} file(s)."
        )

        return {"upstream": upstream, "downstream": downstream, "summary": summary}

    except Exception as exc:
        logger.exception("get_impact_radius failed for %s", changed_files)
        return {
            "upstream": [], "downstream": [],
            "summary": "",
            "error": f"Impact analysis failed: {exc}",
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Prompt: review_my_pr
# ---------------------------------------------------------------------------

@mcp.prompt()
def review_my_pr() -> str:
    """
    A guided workflow to review uncommitted changes using the code graph.

    Instructs the AI to sync the graph, identify changed files, run impact
    analysis, and suggest targeted test cases — all in one coherent sequence.
    """
    return (
        "Please help me review my uncommitted changes using the code graph:\n\n"
        "1. Call build_or_update_graph (use '.' for the current directory) to ensure "
        "the graph reflects the latest code on disk.\n"
        "2. Run the shell command: git diff --name-only HEAD\n"
        "   to get the list of files I have changed.\n"
        "3. Call get_impact_radius with those changed files to discover upstream "
        "callers and downstream dependencies.\n"
        "4. Based on the impacted symbols found, suggest specific test cases that "
        "should be run or written to validate correctness after my changes."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the MCP server over stdio (standard transport for Claude Code / Cursor)."""
    logging.basicConfig(level=logging.WARNING)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
