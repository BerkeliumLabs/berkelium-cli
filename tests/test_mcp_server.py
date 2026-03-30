"""
Tests for the query_search_codebase MCP tool in berkelium_cli.mcp_server.

Follows the same integration-first, minimal-mocking style as the rest of the
test suite.  The graph store is pre-populated directly via GraphQLiteStore
helpers so tests do not need a full codebase on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from berkelium_cli.extractor import EdgeInfo, NodeInfo
from berkelium_cli.mcp_server import query_search_codebase
from berkelium_cli.store import GraphQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_dir(tmp_path: Path) -> Path:
    """Return the .berkelium directory, creating it if needed."""
    d = tmp_path / ".berkelium"
    d.mkdir(exist_ok=True)
    return d


def _make_store(tmp_path: Path) -> GraphQLiteStore:
    """Open (or create) the graph store in the location the MCP tool expects."""
    return GraphQLiteStore(str(_db_dir(tmp_path) / "graph.db"))


def _file_node(rel_path: str, file_hash: str = "h") -> NodeInfo:
    return NodeInfo(
        kind="File",
        name=rel_path.split("/")[-1],
        qualified_name=rel_path,
        file_path=f"/proj/{rel_path}",
        line_start=1,
        line_end=50,
        language="python",
        file_hash=file_hash,
    )


def _fn_node(
    rel_path: str,
    fn_name: str,
    line_start: int = 2,
    line_end: int = 8,
    file_hash: str = "h",
) -> NodeInfo:
    return NodeInfo(
        kind="Function",
        name=fn_name,
        qualified_name=f"{rel_path}::{fn_name}",
        file_path=f"/proj/{rel_path}",
        line_start=line_start,
        line_end=line_end,
        language="python",
        file_hash=file_hash,
    )


def _contains(parent: str, child: str) -> EdgeInfo:
    return EdgeInfo(kind="CONTAINS", source=parent, target=child)


def _calls(caller: str, callee: str) -> EdgeInfo:
    return EdgeInfo(kind="CALLS", source=caller, target=callee)


def _make_call_graph(tmp_path: Path) -> GraphQLiteStore:
    """
    Build a 3-file call graph:

        auth.py  →  models.py  →  db.py
        login()  →  get_user()  →  connect()
    """
    store = _make_store(tmp_path)
    store.store_file_data(
        [_file_node("auth.py"), _fn_node("auth.py", "login", 5, 15)],
        [_contains("auth.py", "auth.py::login")],
    )
    store.store_file_data(
        [_file_node("models.py"), _fn_node("models.py", "get_user", 5, 20)],
        [_contains("models.py", "models.py::get_user")],
    )
    store.store_file_data(
        [_file_node("db.py"), _fn_node("db.py", "connect", 3, 10)],
        [_contains("db.py", "db.py::connect")],
    )
    store.store_call_edges(
        [
            _calls("auth.py::login", "models.py::get_user"),
            _calls("models.py::get_user", "db.py::connect"),
        ]
    )
    return store


# ---------------------------------------------------------------------------
# TestQueryCodeCallGraph
# ---------------------------------------------------------------------------


class TestQueryCodeCallGraph:
    def test_empty_cypher_returns_error(self, tmp_path):
        result = query_search_codebase("", str(tmp_path))
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_whitespace_cypher_returns_error(self, tmp_path):
        result = query_search_codebase("   ", str(tmp_path))
        assert "error" in result
        assert "empty" in result["error"].lower()

    @pytest.mark.parametrize(
        "keyword", ["CREATE", "SET", "DELETE", "MERGE", "REMOVE", "DROP"]
    )
    def test_write_keyword_blocked(self, tmp_path, keyword):
        result = query_search_codebase(f"{keyword} (n) RETURN n", str(tmp_path))
        assert "error" in result
        assert keyword in result["error"]

    def test_write_keyword_lowercase_also_blocked(self, tmp_path):
        result = query_search_codebase("create (n) RETURN n", str(tmp_path))
        assert "error" in result
        assert "CREATE" in result["error"]

    def test_missing_return_clause_returns_error(self, tmp_path):
        result = query_search_codebase(
            "MATCH (n) WHERE n.kind = 'Class'", str(tmp_path)
        )
        assert "error" in result
        assert "RETURN" in result["error"]

    def test_nonexistent_repo_root_returns_error(self, tmp_path):
        result = query_search_codebase("MATCH (n) RETURN n", str(tmp_path / "no_such"))
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_file_as_repo_root_returns_error(self, tmp_path):
        f = tmp_path / "file.py"
        f.write_text("x = 1")
        result = query_search_codebase("MATCH (n) RETURN n", str(f))
        assert "error" in result
        assert "not a directory" in result["error"]

    def test_empty_graph_returns_error_with_hint(self, tmp_path):
        _make_store(tmp_path).close()
        result = query_search_codebase("MATCH (n) RETURN n.name", str(tmp_path))
        assert "error" in result
        assert "berkelium" in result["error"].lower()

    def test_valid_query_returns_rows_and_count(self, tmp_path):
        store = _make_call_graph(tmp_path)
        store.close()
        result = query_search_codebase(
            "MATCH (n) WHERE n.kind = 'Function' "
            "RETURN n.name AS name, n.file_rel_path AS file",
            str(tmp_path),
        )
        assert "error" not in result
        assert result["count"] > 0
        assert len(result["rows"]) == result["count"]

    def test_valid_query_returns_required_keys(self, tmp_path):
        store = _make_call_graph(tmp_path)
        store.close()
        result = query_search_codebase(
            "MATCH (n) WHERE n.kind = 'Function' RETURN n.name AS name",
            str(tmp_path),
        )
        for key in ("rows", "count", "summary"):
            assert key in result, f"missing key: '{key}'"
        assert "error" not in result

    def test_calls_edges_are_queryable(self, tmp_path):
        """CALLS edges inserted via store_call_edges appear in Cypher results."""
        store = _make_call_graph(tmp_path)
        store.close()
        result = query_search_codebase(
            "MATCH (a)-[:CALLS]->(b) RETURN a.name AS caller, b.name AS callee",
            str(tmp_path),
        )
        assert "error" not in result
        assert result["count"] >= 1
        callers = {row["caller"] for row in result["rows"]}
        assert "login" in callers  # auth.py::login → models.py::get_user

    def test_zero_row_result_is_not_an_error(self, tmp_path):
        store = _make_call_graph(tmp_path)
        store.close()
        result = query_search_codebase(
            "MATCH (n) WHERE n.name = 'this_function_xyz_does_not_exist' RETURN n.name",
            str(tmp_path),
        )
        assert "error" not in result
        assert result["count"] == 0
        assert result["rows"] == []

    def test_summary_reports_row_count(self, tmp_path):
        store = _make_call_graph(tmp_path)
        store.close()
        result = query_search_codebase(
            "MATCH (n) WHERE n.kind = 'Function' RETURN n.name",
            str(tmp_path),
        )
        assert str(result["count"]) in result["summary"]

    def test_filter_by_kind_returns_only_matching_nodes(self, tmp_path):
        """Only File nodes are returned when filtering by kind = 'File'."""
        store = _make_call_graph(tmp_path)
        store.close()
        result = query_search_codebase(
            "MATCH (n) WHERE n.kind = 'File' RETURN n.name AS name, n.kind AS kind",
            str(tmp_path),
        )
        assert "error" not in result
        assert result["count"] >= 1
        for row in result["rows"]:
            assert row["kind"] == "File"
