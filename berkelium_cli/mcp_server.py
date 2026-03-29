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
# Tool 3: get_structural_context
# ---------------------------------------------------------------------------

@mcp.tool()
def get_structural_context(
    changed_files: list[str],
    repo_root: str = ".",
    depth: int = 2,
    include_pagerank: bool = False,
) -> str:
    """
    Return a Markdown-formatted structural context for a set of changed files.

    Analyses the code graph to identify upstream callers ("blast radius") and
    downstream dependencies ("implementation context"), then formats the result
    as structured Markdown ready for direct LLM injection.

    Prefer this tool over get_impact_radius for code-review and impact-analysis
    workflows — the output includes hop distances, symbol kinds, line ranges, and
    optional PageRank centrality scores in a format the AI can reason about without
    further parsing.

    The graph must be populated first via build_or_update_graph.

    Args:
        changed_files:    List of relative file paths (forward slashes),
                          e.g. ["src/auth.py", "src/models/user.py"].
        repo_root:        Path to the repository root. Defaults to ".".
        depth:            Max CALLS hops to traverse in each direction. Default 2.
        include_pagerank: If True, enriches results with PageRank centrality
                          scores (adds an extra graph traversal). Default False.

    Returns:
        Markdown string with a seed-files header, upstream callers section,
        downstream dependencies section, and optional PageRank scores.
        Returns a plain error message string if the operation fails.
    """
    root = Path(repo_root).resolve()
    if not root.exists():
        return f"Error: repo_root '{repo_root}' does not exist."
    if not root.is_dir():
        return f"Error: repo_root '{repo_root}' is a file, not a directory."

    if not changed_files:
        return "No changed files provided — nothing to analyse."

    db_path = root / ".berkelium" / "graph.db"

    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return f"Error: could not open graph store at '{db_path}': {exc}"

    try:
        if store.stats().get("node_count", 0) == 0:
            return (
                "Graph is empty — run build_or_update_graph first, "
                "then call get_structural_context."
            )

        retriever = SurgicalRetriever(store=store, max_depth=depth)
        result = retriever.get_full_impact(
            changed_files,
            max_depth=depth,
            include_pagerank=include_pagerank,
        )
        return retriever.assemble_context(result)

    except Exception as exc:
        logger.exception("get_structural_context failed for %s", changed_files)
        return f"Error during structural context retrieval: {exc}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 4: get_file_symbols
# ---------------------------------------------------------------------------

@mcp.tool()
def get_file_symbols(
    file_path: str,
    repo_root: str = ".",
) -> dict:
    """
    List all symbols (functions, classes, etc.) defined in a source file.

    Queries the code graph for all nodes belonging to the given file, returning
    their names, kinds, and line ranges.  Useful for understanding a file's
    public interface without reading raw source — especially for large or
    unfamiliar files.

    The graph must be populated first via build_or_update_graph.

    Args:
        file_path: Relative path to the source file (forward slashes),
                   e.g. "src/auth.py" or "berkelium_cli/store.py".
        repo_root: Path to the repository root. Defaults to ".".

    Returns:
        Dict with keys:
          - "file":    the queried file path
          - "symbols": list of {name, qualified_name, kind, line_start, line_end}
                       sorted by line_start (ascending)
          - "summary": human-readable count string
          - "error":   present only when something went wrong (results may be empty)
    """
    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "file": file_path, "symbols": [], "summary": "",
            "error": f"repo_root '{repo_root}' does not exist.",
        }

    db_path = root / ".berkelium" / "graph.db"

    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return {
            "file": file_path, "symbols": [], "summary": "",
            "error": f"Could not open graph store at '{db_path}': {exc}",
        }

    try:
        nodes, _edges = store.get_file_data(file_path)

        # Filter out the File-level container node — only return actual symbols
        symbols = [
            {
                "name": n.name,
                "qualified_name": n.qualified_name,
                "kind": n.kind,
                "line_start": n.line_start,
                "line_end": n.line_end,
            }
            for n in nodes
            if n.kind != "File"
        ]
        symbols.sort(key=lambda s: s["line_start"])

        if not symbols and not any(n.kind == "File" for n in nodes):
            summary = (
                f"'{file_path}' was not found in the graph. "
                "Run build_or_update_graph first, or check the file path."
            )
        else:
            summary = f"Found {len(symbols)} symbol(s) in '{file_path}'."

        return {"file": file_path, "symbols": symbols, "summary": summary}

    except Exception as exc:
        logger.exception("get_file_symbols failed for '%s'", file_path)
        return {
            "file": file_path, "symbols": [], "summary": "",
            "error": f"Symbol lookup failed: {exc}",
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Kind priority for search_symbols sorting (matches retriever._KIND_PRIORITY)
# ---------------------------------------------------------------------------

_KIND_PRIORITY: dict[str, int] = {
    "Class":     0,
    "Interface": 1,
    "Function":  2,
    "Method":    2,
    "Test":      3,
    "File":      4,
}

# Write-query keywords blocked by query_graph
_WRITE_KEYWORDS = {"create", "set", "delete", "merge", "remove", "drop"}


# ---------------------------------------------------------------------------
# Tool 5: search_symbols
# ---------------------------------------------------------------------------

@mcp.tool()
def search_symbols(
    name: str,
    repo_root: str = ".",
    kind: str = "",
    limit: int = 30,
) -> dict:
    """
    Search for symbols (functions, classes, methods) by name across the codebase.

    Performs a case-insensitive substring match against all symbol names in the
    code graph.  Use this to locate where a function or class is defined before
    calling get_file_symbols or get_structural_context on the containing file.

    Typical workflow:
      1. search_symbols("authenticate") → find which file defines it
      2. get_file_symbols("src/auth.py") → see all symbols in that file
      3. get_structural_context(["src/auth.py"]) → understand blast radius

    The graph must be populated first via build_or_update_graph.

    Args:
        name:      Name fragment to search for (case-insensitive substring match),
                   e.g. "authenticate", "User", "parse_token".
        repo_root: Path to the repository root. Defaults to ".".
        kind:      Optional filter — one of "Function", "Class", "Method",
                   "Interface", "Test". Empty string returns all kinds. Default "".
        limit:     Maximum number of results to return. Default 30.

    Returns:
        Dict with keys:
          - "matches": list of {name, qualified_name, kind, line_start, line_end}
                       sorted by kind priority then name (alphabetical)
          - "total":   total number of matches before the limit was applied
          - "summary": human-readable count string
          - "error":   present only when something went wrong
    """
    if not name or not name.strip():
        return {
            "matches": [], "total": 0,
            "summary": "Provide a non-empty name fragment to search for.",
        }

    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "matches": [], "total": 0, "summary": "",
            "error": f"repo_root '{repo_root}' does not exist.",
        }

    db_path = root / ".berkelium" / "graph.db"

    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return {
            "matches": [], "total": 0, "summary": "",
            "error": f"Could not open graph store at '{db_path}': {exc}",
        }

    try:
        if store.stats().get("node_count", 0) == 0:
            return {
                "matches": [], "total": 0,
                "summary": "Graph is empty — run build_or_update_graph first.",
            }

        nodes = store.get_all_nodes(exclude_external=True)

        needle = name.strip().lower()
        matched = [n for n in nodes if needle in n.name.lower() and n.kind != "File"]

        if kind:
            matched = [n for n in matched if n.kind == kind]

        matched.sort(key=lambda n: (_KIND_PRIORITY.get(n.kind, 99), n.name.lower()))

        total = len(matched)
        matched = matched[:limit]

        results = [
            {
                "name": n.name,
                "qualified_name": n.qualified_name,
                "kind": n.kind,
                "line_start": n.line_start,
                "line_end": n.line_end,
            }
            for n in matched
        ]

        summary = f"Found {total} match(es) for '{name.strip()}'"
        if kind:
            summary += f" (kind={kind})"
        if total > limit:
            summary += f" — showing first {limit}"
        summary += "."

        return {"matches": results, "total": total, "summary": summary}

    except Exception as exc:
        logger.exception("search_symbols failed for name='%s'", name)
        return {
            "matches": [], "total": 0, "summary": "",
            "error": f"Symbol search failed: {exc}",
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 6: query_graph
# ---------------------------------------------------------------------------

@mcp.tool()
def query_graph(
    cypher: str,
    repo_root: str = ".",
) -> dict:
    """
    Run a read-only Cypher query directly against the code knowledge graph.

    Use this when the higher-level tools (search_symbols, get_file_symbols,
    get_structural_context) cannot answer your question.  Results depend
    entirely on what your RETURN clause specifies.

    Write operations (CREATE, SET, DELETE, MERGE, REMOVE, DROP) are blocked.

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
      CONTAINS   (file)-[:CONTAINS]->(symbol)  symbol belongs to file
      INHERITS   (child)-[:INHERITS]->(parent) class inheritance

    Example queries:
      MATCH (n) WHERE n.kind = 'Class' RETURN n.name, n.file_rel_path
      MATCH (n)-[:CALLS]->(m) WHERE n.name = 'login' RETURN m.name, m.file_rel_path
      MATCH (n) WHERE n.file_rel_path = 'src/auth.py' RETURN n.name, n.kind, n.line_start

    The graph must be populated first via build_or_update_graph.

    Args:
        cypher:    A read-only Cypher query string (must contain RETURN).
        repo_root: Path to the repository root. Defaults to ".".

    Returns:
        Dict with keys:
          - "rows":    list of result row dicts (keys match your RETURN aliases)
          - "count":   number of rows returned
          - "summary": human-readable summary
          - "error":   present only when something went wrong
    """
    if not cypher or not cypher.strip():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": "cypher query cannot be empty.",
        }

    first_word = cypher.strip().split()[0].lower()
    if first_word in _WRITE_KEYWORDS:
        return {
            "rows": [], "count": 0, "summary": "",
            "error": (
                f"Write queries are not allowed (detected '{first_word.upper()}')."
                " Use read-only Cypher — MATCH ... RETURN ..."
            ),
        }

    if "return" not in cypher.lower():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": "Query must include a RETURN clause.",
        }

    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "rows": [], "count": 0, "summary": "",
            "error": f"repo_root '{repo_root}' does not exist.",
        }

    db_path = root / ".berkelium" / "graph.db"

    try:
        store = GraphQLiteStore(str(db_path))
    except RuntimeError as exc:
        return {
            "rows": [], "count": 0, "summary": "",
            "error": f"Could not open graph store at '{db_path}': {exc}",
        }

    try:
        rows = store.query(cypher)
        count = len(rows)
        return {
            "rows": rows,
            "count": count,
            "summary": f"Query returned {count} row(s).",
        }
    except Exception as exc:
        logger.exception("query_graph failed for cypher: %s", cypher[:120])
        return {
            "rows": [], "count": 0, "summary": "",
            "error": f"Query failed: {exc}",
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
