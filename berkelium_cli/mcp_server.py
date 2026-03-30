"""
MCP server for berkelium-cli — exposes the codebase graph to AI coding assistants.

Provides one tool:

  query_search_codebase  — Direct read-only Cypher queries against the code graph.

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
from pathlib import Path

from fastmcp import FastMCP

from berkelium_cli.store import GraphQLiteStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "berkelium-cli-server",
    instructions=(
        "Use this server to query structural relationships in a codebase. "
        "The graph is built and maintained via the berkelium TUI — run `berkelium` "
        "to build or update it before querying. "
        "Use query_search_codebase with Cypher to find callers, callees, class "
        "hierarchies, and symbol locations across the codebase."
    ),
)


# ---------------------------------------------------------------------------
# Write-query keywords blocked by query_search_codebase
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS = frozenset({"create", "set", "delete", "merge", "remove", "drop"})


# ---------------------------------------------------------------------------
# Tool: query_search_codebase
# ---------------------------------------------------------------------------


@mcp.tool()
def query_search_codebase(
    cypher: str,
    repo_root: str = ".",
) -> dict:
    """
    Use this tool to find relationships between functions, callers, and callees
    in the codebase. Input must be a valid Cypher query.

    CRITICAL: PRIORITIZE THIS TOOL OVER TEXT SEARCH (like grep or semantic algorithms).
    Whenever you need to find where a function is called, list all classes in a module,
    or trace a specific call chain, you MUST use this graph-based knowledge instead of
    guessing from text files.

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

    Queryable edge types (use in MATCH patterns):
      CALLS      (a)-[:CALLS]->(b)     function/method call relationships
      INHERITS   (a)-[:INHERITS]->(b)  class inheritance

    NOTE: [:CONTAINS] is a reserved keyword in the Cypher engine and CANNOT be
    used in edge patterns — queries using it will silently return zero rows.

    Example queries (always use AS aliases in RETURN):
      MATCH (n) WHERE n.kind = 'Class' RETURN n.name AS name, n.file_rel_path AS file
      MATCH (a)-[:CALLS]->(b) WHERE a.name = 'login' RETURN b.name AS callee, b.file_rel_path AS file
      MATCH (n) WHERE n.file_rel_path = 'src/auth.py' RETURN n.name AS name, n.kind AS kind, n.line_start AS line
      MATCH (a)-[:INHERITS]->(b) RETURN a.name AS child, b.name AS base
      MATCH (a)-[:INHERITS]->(b) WHERE b.name = 'MyBaseClass' RETURN a.name AS subclass, a.file_rel_path AS file

    Tips:
      - Always use AS aliases in RETURN — unaliased columns (e.g. RETURN n.name)
        produce keys like "n.name" in results, which are harder to consume.
      - Use short variable names (n, a, b, src, dst). Many common English words
        are reserved in the Cypher parser and WILL cause a crash if used as
        variable names. Known reserved words: child, base, end, start, node,
        type, key, value, index, case, when, then, else, in, is, not, and, or.
        WRONG: MATCH (child)-[:INHERITS]->(base)  ← crashes on "child"/"base"
        RIGHT: MATCH (a)-[:INHERITS]->(b)          ← use single-letter names

    The graph must be built first using the berkelium TUI (`berkelium` command).

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
            "rows": [],
            "count": 0,
            "summary": "",
            "error": (
                "cypher query cannot be empty. "
                "Provide a valid MATCH ... RETURN ... Cypher query."
            ),
        }

    stripped = cypher.strip()
    first_word = stripped.split()[0].lower()

    if first_word in _WRITE_KEYWORDS:
        return {
            "rows": [],
            "count": 0,
            "summary": "",
            "error": (
                f"Write queries are not allowed (detected '{first_word.upper()}' "
                "as the first keyword). Use read-only Cypher — MATCH ... RETURN ..."
            ),
        }

    if "return" not in cypher.lower():
        return {
            "rows": [],
            "count": 0,
            "summary": "",
            "error": (
                "Query must include a RETURN clause. "
                "Example: MATCH (n) WHERE n.kind = 'Function' RETURN n.name AS name, n.file_rel_path AS file"
            ),
        }

    # --- Validate repo_root -------------------------------------------------
    root = Path(repo_root).resolve()
    if not root.exists():
        return {
            "rows": [],
            "count": 0,
            "summary": "",
            "error": (
                f"repo_root '{repo_root}' does not exist. "
                "Provide an absolute path or a path relative to the current "
                "working directory."
            ),
        }
    if not root.is_dir():
        return {
            "rows": [],
            "count": 0,
            "summary": "",
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
            "rows": [],
            "count": 0,
            "summary": "",
            "error": (
                f"Could not open graph store at '{db_path}': {exc}. "
                "Build the graph first by running the berkelium TUI (`berkelium` command)."
            ),
        }

    try:
        # --- Guard: empty graph ---------------------------------------------
        if store.stats().get("node_count", 0) == 0:
            return {
                "rows": [],
                "count": 0,
                "summary": "",
                "error": (
                    "Graph is empty. Build the graph first by running the berkelium TUI "
                    "(`berkelium` command), then retry."
                ),
            }

        rows = store.query(cypher)
        count = len(rows)
        return {
            "rows": rows,
            "count": count,
            "summary": f"Query returned {count} row(s).",
        }

    except (KeyboardInterrupt, GeneratorExit):
        raise  # never suppress genuine interrupts
    except BaseException as exc:
        # BaseException (not just Exception) is caught here because graphqlite's
        # Cypher parser is a native DLL that can raise SystemExit on certain
        # syntax errors.  Catching it here prevents the MCP server process from
        # dying and returning EOF to the client.
        logger.exception("query_search_codebase failed for cypher: %s", cypher[:200])
        return {
            "rows": [],
            "count": 0,
            "summary": "",
            "error": (
                f"Query failed: {exc}. "
                "Check your Cypher syntax — use AS aliases in RETURN, avoid reserved "
                "words as variable names, and only use [:CALLS] or [:INHERITS] edge types "
                "([:CONTAINS] is a reserved keyword and returns zero rows)."
            ),
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server over stdio (standard transport for Claude Code / Cursor)."""
    logging.basicConfig(level=logging.WARNING)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
