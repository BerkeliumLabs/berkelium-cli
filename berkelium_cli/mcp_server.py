"""
MCP server for berkelium-cli — exposes the codebase graph to AI coding assistants.

Provides three tools and one prompt workflow:

  build_or_update_graph  — Full extraction on first run; incremental git-diff sync thereafter.
  analyze_impact_radius  — Pre-change static impact analysis (upstream callers + downstream deps).
  query_code_call_graph  — Direct read-only Cypher queries against the code graph.
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
        "Workflow: "
        "1. Call build_or_update_graph first to sync the graph with the current code on disk. "
        "2. Call analyze_impact_radius BEFORE proposing or applying changes — it identifies "
        "all callers, dependencies, and affected files so you can assess risk. "
        "3. Use query_code_call_graph for advanced Cypher queries when the above tools "
        "are insufficient."
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

    Always call this tool first before analyze_impact_radius or
    query_code_call_graph to ensure the graph reflects the current state of
    code on disk.

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
        return (
            f"Error: repo_root '{repo_root}' does not exist. "
            "Provide an absolute path or a path relative to the current working directory."
        )
    if not root.is_dir():
        return (
            f"Error: repo_root '{repo_root}' is a file, not a directory. "
            "Provide the path to the repository root directory."
        )

    db_path = root / ".berkelium" / "graph.db"

    # --- Open store and decide strategy -------------------------------------
    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return (
            f"Error: could not open graph store at '{db_path}': {exc}. "
            "Ensure the process has write permissions to the repository directory."
        )

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
        return (
            f"Error during full extraction of '{root}': {exc}. "
            "Check that the directory contains supported source files "
            "(Python, JS/TS, Go, Java, Rust, C/C++)."
        )


def _incremental_sync(root: Path, store: GraphQLiteStore) -> str:
    """
    Try IncrementalSync; fall back to full extraction on git errors.
    Returns a summary string.
    """
    try:
        syncer = IncrementalSync(root=root, store=store)
        result = syncer.sync(base_ref="HEAD")
        errors_note = (
            f" ({len(result.errors)} non-fatal error(s): "
            + "; ".join(result.errors[:3])
            + ("..." if len(result.errors) > 3 else "")
            + ")"
            if result.errors
            else ""
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
        return (
            f"Error during incremental sync of '{root}': {exc}. "
            "Try calling build_or_update_graph again; if the error persists, "
            "delete '.berkelium/graph.db' to force a full rebuild."
        )


# ---------------------------------------------------------------------------
# Tool 2: analyze_impact_radius
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_impact_radius(
    target_files: list[str],
    repo_root: str = ".",
    depth: int = 2,
    include_pagerank: bool = False,
) -> dict:
    """
    Use this tool before proposing or applying any code changes.

    Performs a static analysis of the call graph to identify all upstream callers,
    downstream dependencies, and affected files for the given target files.
    Use this to assess risk, identify what tests need to run, and understand the
    full impact of a planned modification before writing any code.

    Upstream callers  = components that call INTO the target files
                        ("who will break if I change this?")
    Downstream deps   = components called BY the target files
                        ("what does my change depend on?")

    The graph must be populated first via build_or_update_graph.

    Args:
        target_files:     List of relative file paths (forward slashes) you plan
                          to modify, e.g. ["src/auth.py", "src/models/user.py"].
        repo_root:        Path to the repository root. Defaults to ".".
        depth:            Max CALLS hops to traverse in each direction. Default 2.
                          Increase to 3-4 for deeper transitive impact; decrease to
                          1 for a quick first-hop check on large codebases.
        include_pagerank: If True, enriches results with PageRank centrality scores
                          to rank symbols by architectural importance. Default False.

    Returns:
        Dict with keys:
          - "context":        Markdown-formatted analysis ready for LLM injection.
                              Includes hop distances, symbol kinds, and line ranges.
          - "upstream":       List of {name, qualified_name, file, kind, distance}
                              for callers — sorted by hop distance then kind.
          - "downstream":     Same structure for callees / dependencies.
          - "affected_files": Sorted list of unique file paths that will be touched.
          - "risk_summary":   One-line risk assessment,
                              e.g. "3 upstream caller(s), 7 downstream dep(s)".
          - "summary":        Human-readable count summary.
          - "error":          Present only when a non-fatal problem occurred;
                              results may be partial.
    """
    # --- Validate repo_root -------------------------------------------------
    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "context": "", "upstream": [], "downstream": [],
            "affected_files": [], "risk_summary": "", "summary": "",
            "error": (
                f"repo_root '{repo_root}' does not exist. "
                "Provide an absolute path or a path relative to the current "
                "working directory."
            ),
        }
    if not root.is_dir():
        return {
            "context": "", "upstream": [], "downstream": [],
            "affected_files": [], "risk_summary": "", "summary": "",
            "error": (
                f"repo_root '{repo_root}' is a file, not a directory. "
                "Provide the path to the repository root directory."
            ),
        }

    # --- Validate target_files ----------------------------------------------
    if not target_files:
        return {
            "context": "", "upstream": [], "downstream": [],
            "affected_files": [], "risk_summary": "No files provided.",
            "summary": (
                "No target_files provided — nothing to analyse. "
                "Pass a list of relative file paths you plan to modify."
            ),
        }

    # Normalise paths: strip leading slashes / backslashes, convert to forward slashes
    normalised = [f.lstrip("/\\").replace("\\", "/") for f in target_files]

    db_path = root / ".berkelium" / "graph.db"

    # --- Open graph store ---------------------------------------------------
    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return {
            "context": "", "upstream": [], "downstream": [],
            "affected_files": [], "risk_summary": "", "summary": "",
            "error": (
                f"Could not open graph store at '{db_path}': {exc}. "
                "Run build_or_update_graph first to initialise the graph."
            ),
        }

    try:
        # --- Guard: empty graph ---------------------------------------------
        if store.stats().get("node_count", 0) == 0:
            return {
                "context": "", "upstream": [], "downstream": [],
                "affected_files": [], "risk_summary": "Graph is empty.",
                "summary": (
                    "Graph is empty — run build_or_update_graph first, then retry."
                ),
            }

        # --- Traversal ------------------------------------------------------
        retriever = SurgicalRetriever(store=store, max_depth=depth)
        result = retriever.get_full_impact(
            normalised,
            max_depth=depth,
            include_pagerank=include_pagerank,
        )

        # --- Serialise upstream / downstream --------------------------------
        upstream = [
            {
                "name": s.name,
                "qualified_name": s.qualified_name,
                "file": s.file_rel_path,
                "kind": s.kind,
                "distance": s.distance,
                **({"pagerank_score": round(s.pagerank_score, 4)} if s.pagerank_score else {}),
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
                **({"pagerank_score": round(s.pagerank_score, 4)} if s.pagerank_score else {}),
            }
            for s in result.downstream
        ]

        # --- Affected files (unique, sorted) --------------------------------
        affected_files = sorted(
            {s["file"] for s in upstream + downstream if s["file"]}
        )

        # --- Risk summary ---------------------------------------------------
        risk_summary = (
            f"{len(upstream)} upstream caller(s), "
            f"{len(downstream)} downstream dep(s) "
            f"across {depth} hop(s) for {len(normalised)} file(s)."
        )

        summary = (
            f"Impact analysis complete: {risk_summary} "
            f"{len(affected_files)} unique file(s) affected."
        )

        # --- Markdown context (for direct LLM injection) --------------------
        context = retriever.assemble_context(result)

        return {
            "context": context,
            "upstream": upstream,
            "downstream": downstream,
            "affected_files": affected_files,
            "risk_summary": risk_summary,
            "summary": summary,
        }

    except Exception as exc:
        logger.exception("analyze_impact_radius failed for %s", target_files)
        return {
            "context": "", "upstream": [], "downstream": [],
            "affected_files": [], "risk_summary": "", "summary": "",
            "error": (
                f"Impact analysis failed: {exc}. "
                "Ensure the graph is populated (run build_or_update_graph) "
                "and that the file paths use forward slashes relative to repo_root."
            ),
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Write-query keywords blocked by query_code_call_graph
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS = frozenset({"create", "set", "delete", "merge", "remove", "drop"})


# ---------------------------------------------------------------------------
# Tool 3: query_code_call_graph
# ---------------------------------------------------------------------------

@mcp.tool()
def query_code_call_graph(
    cypher: str,
    repo_root: str = ".",
) -> dict:
    """
    Use this tool to find relationships between functions, callers, and callees
    in the codebase. Input must be a valid Cypher query.

    Use this when analyze_impact_radius cannot answer your question — for example,
    to find all classes in a module, trace a specific call chain, or count symbols
    by kind. Write operations (CREATE, SET, DELETE, MERGE, REMOVE, DROP) are blocked.

    Node schema — properties on every non-External node:
      name            str   Short symbol name, e.g. "authenticate"
      qualified_name  str   Unique ID, e.g. "src/auth.py::AuthService.authenticate"
      kind            str   File | Function | Class | Method | Interface | Test
      file_rel_path   str   Relative path, e.g. "src/auth.py"
      file_path       str   Absolute path
      line_start      int   First line of the symbol definition
      line_end        int   Last line of the symbol definition
      language        str   e.g. "python", "typescript", "go"

    Edge schema:
      CALLS      (caller)-[:CALLS]->(callee)   function call relationships
      CONTAINS   (file)-[:CONTAINS]->(symbol)  symbol belongs to a file
      INHERITS   (child)-[:INHERITS]->(parent) class inheritance

    Example queries:
      MATCH (n) WHERE n.kind = 'Class' RETURN n.name, n.file_rel_path
      MATCH (n)-[:CALLS]->(m) WHERE n.name = 'login' RETURN m.name, m.file_rel_path
      MATCH (n) WHERE n.file_rel_path = 'src/auth.py' RETURN n.name, n.kind, n.line_start

    The graph must be populated first via build_or_update_graph.

    Args:
        cypher:    A read-only Cypher query string. Must contain a RETURN clause.
                   Write keywords (CREATE, SET, DELETE, MERGE, REMOVE, DROP) are rejected.
        repo_root: Path to the repository root. Defaults to ".".

    Returns:
        Dict with keys:
          - "rows":    list of result row dicts (keys match your RETURN aliases)
          - "count":   number of rows returned
          - "summary": human-readable summary
          - "error":   present only when something went wrong
    """
    # --- Validate query -----------------------------------------------------
    if not cypher or not cypher.strip():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                "cypher query cannot be empty. "
                "Provide a valid MATCH ... RETURN ... Cypher query."
            ),
        }

    stripped = cypher.strip()
    first_word = stripped.split()[0].lower()

    if first_word in _WRITE_KEYWORDS:
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                f"Write queries are not allowed (detected '{first_word.upper()}' "
                "as the first keyword). Use read-only Cypher — MATCH ... RETURN ..."
            ),
        }

    if "return" not in cypher.lower():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                "Query must include a RETURN clause. "
                "Example: MATCH (n) WHERE n.kind = 'Function' RETURN n.name, n.file_rel_path"
            ),
        }

    # --- Validate repo_root -------------------------------------------------
    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                f"repo_root '{repo_root}' does not exist. "
                "Provide an absolute path or a path relative to the current "
                "working directory."
            ),
        }
    if not root.is_dir():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                f"repo_root '{repo_root}' is a file, not a directory. "
                "Provide the path to the repository root directory."
            ),
        }

    db_path = root / ".berkelium" / "graph.db"

    # --- Open graph store ---------------------------------------------------
    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                f"Could not open graph store at '{db_path}': {exc}. "
                "Run build_or_update_graph first to initialise the graph."
            ),
        }

    try:
        # --- Guard: empty graph ---------------------------------------------
        if store.stats().get("node_count", 0) == 0:
            return {
                "rows": [], "count": 0,
                "summary": "",
                "error": (
                    "Graph is empty — run build_or_update_graph first, then retry."
                ),
            }

        rows = store.query(cypher)
        count = len(rows)
        return {
            "rows": rows,
            "count": count,
            "summary": f"Query returned {count} row(s).",
        }

    except Exception as exc:
        logger.exception("query_code_call_graph failed for cypher: %s", cypher[:200])
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                f"Query failed: {exc}. "
                "Check your Cypher syntax — properties must be accessed as n.property_name, "
                "and edge types must be uppercase (e.g. [:CALLS], [:CONTAINS], [:INHERITS])."
            ),
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
        "3. Call analyze_impact_radius with those changed files to discover all upstream "
        "callers and downstream dependencies before reviewing the changes.\n"
        "4. Based on the impacted symbols found, suggest specific test cases that "
        "should be run or written to validate correctness after the changes."
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
