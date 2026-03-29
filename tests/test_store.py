"""Tests for berkelium_cli.store.GraphQLiteStore."""
from __future__ import annotations

import pytest
from pathlib import Path

from berkelium_cli.extractor import CodebaseExtractor, NodeInfo, EdgeInfo
from berkelium_cli.store import GraphQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> GraphQLiteStore:
    return GraphQLiteStore(str(tmp_path / "graph.db"))


def _file_node(rel_path: str = "src/auth.py", file_hash: str = "abc123") -> NodeInfo:
    return NodeInfo(
        kind="File",
        name=rel_path.split("/")[-1],
        qualified_name=rel_path,
        file_path=f"/project/{rel_path}",
        line_start=1,
        line_end=50,
        language="python",
        file_hash=file_hash,
    )


def _class_node(
    rel_path: str = "src/auth.py",
    class_name: str = "AuthService",
    file_hash: str = "abc123",
) -> NodeInfo:
    return NodeInfo(
        kind="Class",
        name=class_name,
        qualified_name=f"{rel_path}::{class_name}",
        file_path=f"/project/{rel_path}",
        line_start=3,
        line_end=20,
        language="python",
        file_hash=file_hash,
    )


def _fn_node(
    rel_path: str = "src/auth.py",
    fn_name: str = "login",
    parent: str = "AuthService",
    file_hash: str = "abc123",
) -> NodeInfo:
    return NodeInfo(
        kind="Function",
        name=fn_name,
        qualified_name=f"{rel_path}::{parent}.{fn_name}",
        file_path=f"/project/{rel_path}",
        line_start=5,
        line_end=10,
        language="python",
        file_hash=file_hash,
    )


def _contains_edge(parent_qname: str, child_qname: str) -> EdgeInfo:
    return EdgeInfo(kind="CONTAINS", source=parent_qname, target=child_qname)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "sub" / "graph.db"
        store = GraphQLiteStore(str(db))
        store.close()
        assert db.exists()

    def test_default_path_used_when_omitted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = GraphQLiteStore()
        store.close()
        assert (tmp_path / ".berkelium" / "graph.db").exists()

    def test_in_memory_store(self):
        store = GraphQLiteStore(":memory:")
        store.close()  # should not raise

    def test_context_manager(self, tmp_path):
        with _make_store(tmp_path) as store:
            assert store.stats() == {"node_count": 0, "edge_count": 0}
        # close() was called; further use would raise — just verifying no exception above

    def test_bad_path_raises_runtime_error(self, tmp_path):
        # Use an existing regular file as the parent directory — mkdir will fail
        blocker = tmp_path / "notadir"
        blocker.write_text("I am a file, not a directory")
        with pytest.raises(RuntimeError):
            GraphQLiteStore(str(blocker / "child.db"))


# ---------------------------------------------------------------------------
# Cache check: has_file_cached
# ---------------------------------------------------------------------------

class TestHasFileCached:
    def test_empty_store_returns_false(self, tmp_path):
        with _make_store(tmp_path) as store:
            assert store.has_file_cached("src/auth.py", "deadbeef") is False

    def test_returns_true_after_storing_with_matching_hash(self, tmp_path):
        file_hash = "cafecafe"
        nodes = [_file_node(file_hash=file_hash), _class_node(file_hash=file_hash)]
        edges = [_contains_edge("src/auth.py", "src/auth.py::AuthService")]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, edges)
            assert store.has_file_cached("src/auth.py", file_hash) is True

    def test_returns_false_on_hash_mismatch(self, tmp_path):
        nodes = [_file_node(file_hash="hash_v1")]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, [])
            assert store.has_file_cached("src/auth.py", "hash_v2") is False

    def test_returns_false_for_empty_hash(self, tmp_path):
        nodes = [_file_node(file_hash="")]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, [])
            # Empty hash must never count as a cache hit
            assert store.has_file_cached("src/auth.py", "") is False


# ---------------------------------------------------------------------------
# Store and retrieve
# ---------------------------------------------------------------------------

class TestStoreAndRetrieve:
    def test_store_file_data_and_get_nodes_back(self, tmp_path):
        file_hash = "h1"
        nodes = [
            _file_node(file_hash=file_hash),
            _class_node(file_hash=file_hash),
            _fn_node(file_hash=file_hash),
        ]
        edges = [
            _contains_edge("src/auth.py", "src/auth.py::AuthService"),
            _contains_edge("src/auth.py::AuthService", "src/auth.py::AuthService.login"),
        ]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, edges)
            back_nodes, back_edges = store.get_file_data("src/auth.py")

        qnames = {n.qualified_name for n in back_nodes}
        assert "src/auth.py" in qnames
        assert "src/auth.py::AuthService" in qnames
        assert "src/auth.py::AuthService.login" in qnames

    def test_node_fields_round_trip(self, tmp_path):
        original = _fn_node(file_hash="myhash")
        with _make_store(tmp_path) as store:
            store.store_file_data([_file_node(file_hash="myhash"), original], [])
            back_nodes, _ = store.get_file_data("src/auth.py")

        fn = next(n for n in back_nodes if n.qualified_name == original.qualified_name)
        assert fn.kind == "Function"
        assert fn.name == "login"
        assert fn.language == "python"
        assert fn.line_start == 5
        assert fn.line_end == 10
        assert fn.file_hash == "myhash"

    def test_empty_file_returns_empty(self, tmp_path):
        with _make_store(tmp_path) as store:
            nodes, edges = store.get_file_data("nonexistent.py")
        assert nodes == []
        assert edges == []

    def test_edges_round_trip(self, tmp_path):
        file_hash = "edgehash"
        nodes = [
            _file_node(file_hash=file_hash),
            _class_node(file_hash=file_hash),
        ]
        edge = _contains_edge("src/auth.py", "src/auth.py::AuthService")
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, [edge])
            _, back_edges = store.get_file_data("src/auth.py")

        contains = [e for e in back_edges if e.kind == "CONTAINS"]
        assert len(contains) >= 1
        assert any(e.target == "src/auth.py::AuthService" for e in contains)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDeleteFileData:
    def test_delete_removes_nodes(self, tmp_path):
        nodes = [_file_node(), _class_node()]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, [])
            store.delete_file_data("src/auth.py")
            back_nodes, _ = store.get_file_data("src/auth.py")
        assert back_nodes == []

    def test_delete_nonexistent_file_is_harmless(self, tmp_path):
        with _make_store(tmp_path) as store:
            store.delete_file_data("does_not_exist.py")  # must not raise

    def test_subsequent_store_after_delete_succeeds(self, tmp_path):
        nodes_v1 = [_file_node(file_hash="v1"), _class_node(file_hash="v1")]
        nodes_v2 = [_file_node(file_hash="v2")]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes_v1, [])
            store.delete_file_data("src/auth.py")
            store.store_file_data(nodes_v2, [])
            back, _ = store.get_file_data("src/auth.py")
        # Only the v2 File node should exist; Class was deleted
        qnames = {n.qualified_name for n in back}
        assert "src/auth.py" in qnames
        assert "src/auth.py::AuthService" not in qnames


# ---------------------------------------------------------------------------
# Stats and clear
# ---------------------------------------------------------------------------

class TestStatsAndClear:
    def test_stats_node_count_increases(self, tmp_path):
        nodes = [_file_node(), _class_node(), _fn_node()]
        with _make_store(tmp_path) as store:
            assert store.stats()["node_count"] == 0
            store.store_file_data(nodes, [])
            assert store.stats()["node_count"] >= len(nodes)

    def test_clear_empties_store(self, tmp_path):
        nodes = [_file_node(), _class_node()]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, [])
            store.clear()
            s = store.stats()
        assert s["node_count"] == 0
        assert s["edge_count"] == 0


# ---------------------------------------------------------------------------
# External symbol placeholders
# ---------------------------------------------------------------------------

class TestExternalNodes:
    def test_inherits_edge_with_external_target_stored(self, tmp_path):
        """INHERITS edge whose target ('Base') is not in codebase must not crash."""
        nodes = [_file_node(), _class_node()]
        edges = [
            _contains_edge("src/auth.py", "src/auth.py::AuthService"),
            EdgeInfo(kind="INHERITS", source="src/auth.py::AuthService", target="Base"),
        ]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, edges)  # must not raise
            s = store.stats()
        assert s["node_count"] >= 2  # at least our 2 + External placeholder

    def test_imports_edge_with_external_module(self, tmp_path):
        nodes = [_file_node()]
        edges = [
            EdgeInfo(kind="IMPORTS", source="src/auth.py", target="pathlib"),
        ]
        with _make_store(tmp_path) as store:
            store.store_file_data(nodes, edges)  # must not raise


# ---------------------------------------------------------------------------
# Incremental extraction (integration tests)
# ---------------------------------------------------------------------------

class TestIncrementalExtraction:
    SIMPLE_SRC = "class Foo:\n    def bar(self):\n        pass\n"

    def _make_extractor(self, root: Path, store: GraphQLiteStore) -> CodebaseExtractor:
        return CodebaseExtractor(root, store=store)

    def test_second_run_uses_cache(self, tmp_path):
        """Extracting the same codebase twice should hit cache on the second run."""
        src = tmp_path / "app.py"
        src.write_text(self.SIMPLE_SRC, encoding="utf-8")

        with GraphQLiteStore(str(tmp_path / "graph.db")) as store:
            e = self._make_extractor(tmp_path, store)
            nodes1, _ = e.extract()

            # Second run — file unchanged; nodes should come from store
            nodes2, _ = e.extract()

        qnames1 = {n.qualified_name for n in nodes1}
        qnames2 = {n.qualified_name for n in nodes2}
        assert qnames1 == qnames2

    def test_changed_file_reparsed(self, tmp_path):
        """Changing a file's content must cause it to be re-parsed."""
        src = tmp_path / "model.py"
        src.write_text("class Dog:\n    pass\n", encoding="utf-8")

        with GraphQLiteStore(str(tmp_path / "graph.db")) as store:
            e = self._make_extractor(tmp_path, store)
            nodes1, _ = e.extract()
            qnames1 = {n.qualified_name for n in nodes1}

            # Modify the file
            src.write_text("class Cat:\n    pass\n", encoding="utf-8")
            nodes2, _ = e.extract()
            qnames2 = {n.qualified_name for n in nodes2}

        assert "model.py::Dog" in qnames1
        assert "model.py::Dog" not in qnames2
        assert "model.py::Cat" in qnames2

    def test_file_hash_stored_on_nodes(self, tmp_path):
        """All nodes produced for a file must carry the file's MD5 hash."""
        src = tmp_path / "util.py"
        src.write_text("def helper():\n    pass\n", encoding="utf-8")

        with GraphQLiteStore(str(tmp_path / "graph.db")) as store:
            e = self._make_extractor(tmp_path, store)
            nodes, _ = e.extract()

        import hashlib
        expected_hash = hashlib.md5(src.read_bytes()).hexdigest()
        for n in nodes:
            assert n.file_hash == expected_hash, (
                f"Node {n.qualified_name!r} has hash {n.file_hash!r}, "
                f"expected {expected_hash!r}"
            )

    def test_no_store_extraction_unaffected(self, tmp_path):
        """Extraction without a store must behave identically to the original."""
        src = tmp_path / "app.py"
        src.write_text(self.SIMPLE_SRC, encoding="utf-8")

        nodes, edges = CodebaseExtractor(tmp_path).extract()
        assert any(n.qualified_name == "app.py" for n in nodes)
        assert any(n.name == "Foo" for n in nodes)
