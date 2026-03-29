"""Tests for berkelium_cli.retriever.SurgicalRetriever."""
from __future__ import annotations

import pytest
from pathlib import Path

from berkelium_cli.extractor import NodeInfo, EdgeInfo
from berkelium_cli.store import GraphQLiteStore
from berkelium_cli.retriever import ImpactedSymbol, RetrievalResult, SurgicalRetriever


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> GraphQLiteStore:
    return GraphQLiteStore(str(tmp_path / "graph.db"))


def _file_node(rel_path: str, file_hash: str = "h") -> NodeInfo:
    return NodeInfo(
        kind="File", name=rel_path.split("/")[-1],
        qualified_name=rel_path, file_path=f"/proj/{rel_path}",
        line_start=1, line_end=50, language="python", file_hash=file_hash,
    )


def _fn_node(
    rel_path: str,
    fn_name: str,
    line_start: int = 2,
    line_end: int = 8,
    file_hash: str = "h",
) -> NodeInfo:
    return NodeInfo(
        kind="Function", name=fn_name,
        qualified_name=f"{rel_path}::{fn_name}",
        file_path=f"/proj/{rel_path}",
        line_start=line_start, line_end=line_end,
        language="python", file_hash=file_hash,
    )


def _class_node(rel_path: str, class_name: str, file_hash: str = "h") -> NodeInfo:
    return NodeInfo(
        kind="Class", name=class_name,
        qualified_name=f"{rel_path}::{class_name}",
        file_path=f"/proj/{rel_path}",
        line_start=1, line_end=30, language="python", file_hash=file_hash,
    )


def _contains(parent: str, child: str) -> EdgeInfo:
    return EdgeInfo(kind="CONTAINS", source=parent, target=child)


def _calls(caller: str, callee: str) -> EdgeInfo:
    return EdgeInfo(kind="CALLS", source=caller, target=callee)


def _make_call_graph(tmp_path: Path) -> GraphQLiteStore:
    """
    Build a 3-file call graph:

        utils.py:   helper()                          ← leaf
        service.py: Service (class), Service.run()    ← calls utils.py::helper
        main.py:    main()                            ← calls service.py::Service.run
    """
    store = GraphQLiteStore(str(tmp_path / "graph.db"))

    # utils.py
    store.store_file_data(
        [_file_node("utils.py"), _fn_node("utils.py", "helper", 2, 6)],
        [_contains("utils.py", "utils.py::helper")],
    )
    # service.py
    store.store_file_data(
        [
            _file_node("service.py"),
            _class_node("service.py", "Service"),
            _fn_node("service.py", "Service.run", 5, 12),
        ],
        [
            _contains("service.py", "service.py::Service"),
            _contains("service.py::Service", "service.py::Service.run"),
        ],
    )
    # main.py
    store.store_file_data(
        [_file_node("main.py"), _fn_node("main.py", "main", 1, 5)],
        [_contains("main.py", "main.py::main")],
    )

    # CALLS edges: main → Service.run → helper
    store.store_call_edges([
        _calls("service.py::Service.run", "utils.py::helper"),
        _calls("main.py::main", "service.py::Service.run"),
    ])

    return store


# ---------------------------------------------------------------------------
# TestSurgicalRetrieverInit
# ---------------------------------------------------------------------------

class TestSurgicalRetrieverInit:
    def test_default_max_depth(self, tmp_path):
        with _make_store(tmp_path) as store:
            r = SurgicalRetriever(store)
        assert r.max_depth == SurgicalRetriever.DEFAULT_MAX_DEPTH

    def test_custom_max_depth(self, tmp_path):
        with _make_store(tmp_path) as store:
            r = SurgicalRetriever(store, max_depth=5)
        assert r.max_depth == 5

    def test_store_reference_preserved(self, tmp_path):
        with _make_store(tmp_path) as store:
            r = SurgicalRetriever(store)
            assert r.store is store


# ---------------------------------------------------------------------------
# TestUpstreamImpact
# ---------------------------------------------------------------------------

class TestUpstreamImpact:
    def test_direct_caller_found(self, tmp_path):
        """service.py::Service.run is a direct (d=1) caller of utils.py."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py")

        qnames = {s.qualified_name for s in syms}
        assert "service.py::Service.run" in qnames

    def test_transitive_caller_found(self, tmp_path):
        """main.py::main is a transitive (d=2) caller of utils.py."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py", depth=3)

        qnames = {s.qualified_name for s in syms}
        assert "main.py::main" in qnames

    def test_distance_assigned_correctly(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py", depth=3)

        by_qname = {s.qualified_name: s for s in syms}
        assert by_qname["service.py::Service.run"].distance == 1
        assert by_qname["main.py::main"].distance == 2

    def test_direction_is_upstream(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py")
        assert all(s.direction == "upstream" for s in syms)

    def test_same_file_excluded(self, tmp_path):
        """Symbols from the seed file itself must not appear in results."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py")
        assert not any(s.file_rel_path == "utils.py" for s in syms)

    def test_nonexistent_file_returns_empty(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("does_not_exist.py")
        assert syms == []

    def test_depth_one_limits_results(self, tmp_path):
        """With depth=1, only direct callers should be returned."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py", depth=1)

        qnames = {s.qualified_name for s in syms}
        # Service.run is at distance 1 — should be present
        assert "service.py::Service.run" in qnames
        # main is at distance 2 — must NOT appear with depth=1
        assert "main.py::main" not in qnames

    def test_leaf_has_no_upstream(self, tmp_path):
        """main.py is at the top of the call chain — nothing calls it."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("main.py")
        assert syms == []


# ---------------------------------------------------------------------------
# TestDownstreamDeps
# ---------------------------------------------------------------------------

class TestDownstreamDeps:
    def test_direct_callee_found(self, tmp_path):
        """service.py directly calls utils.py::helper (d=1)."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("service.py")

        qnames = {s.qualified_name for s in syms}
        assert "utils.py::helper" in qnames

    def test_transitive_callee_found(self, tmp_path):
        """main.py transitively calls utils.py::helper (d=2 via Service.run)."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("main.py", depth=3)

        qnames = {s.qualified_name for s in syms}
        assert "utils.py::helper" in qnames

    def test_distance_assigned_correctly(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("main.py", depth=3)

        by_qname = {s.qualified_name: s for s in syms}
        assert by_qname["service.py::Service.run"].distance == 1
        assert by_qname["utils.py::helper"].distance == 2

    def test_direction_is_downstream(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("main.py")
        assert all(s.direction == "downstream" for s in syms)

    def test_same_file_excluded(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("main.py")
        assert not any(s.file_rel_path == "main.py" for s in syms)

    def test_leaf_has_no_downstream(self, tmp_path):
        """utils.py::helper is a leaf — it calls nothing."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("utils.py")
        assert syms == []

    def test_nonexistent_file_returns_empty(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_downstream_deps("ghost.py")
        assert syms == []


# ---------------------------------------------------------------------------
# TestFullImpact
# ---------------------------------------------------------------------------

class TestFullImpact:
    def test_returns_retrieval_result(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"])
        assert isinstance(result, RetrievalResult)

    def test_seed_files_preserved(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py", "main.py"])
        assert result.seed_files == ["utils.py", "main.py"]

    def test_upstream_and_downstream_populated(self, tmp_path):
        """service.py has both callers (main) and callees (helper)."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["service.py"], max_depth=3)

        up_qnames = {s.qualified_name for s in result.upstream}
        down_qnames = {s.qualified_name for s in result.downstream}
        assert "main.py::main" in up_qnames
        assert "utils.py::helper" in down_qnames

    def test_max_depth_override(self, tmp_path):
        """With max_depth=1, transitive symbols should not appear."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store, max_depth=5)  # instance default is 5
            result = r.get_full_impact(["utils.py"], max_depth=1)

        up_qnames = {s.qualified_name for s in result.upstream}
        assert "service.py::Service.run" in up_qnames
        assert "main.py::main" not in up_qnames
        assert result.max_depth == 1

    def test_dedup_keeps_shortest_path(self, tmp_path):
        """
        When two seed files both reach the same target symbol, the result
        should contain the symbol only once (at the shorter distance).
        """
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            # utils.py::helper is downstream of BOTH service.py (d=1) and main.py (d=2)
            result = r.get_full_impact(["service.py", "main.py"], max_depth=3)

        helper_entries = [s for s in result.downstream if s.qualified_name == "utils.py::helper"]
        assert len(helper_entries) == 1
        assert helper_entries[0].distance == 1  # shorter path from service.py

    def test_empty_store_returns_empty_result(self, tmp_path):
        with _make_store(tmp_path) as store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["any.py"])
        assert result.upstream == []
        assert result.downstream == []

    def test_sorted_by_distance_then_kind(self, tmp_path):
        """Upstream list should be sorted: distance ASC, then kind priority."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"], max_depth=3)
        distances = [s.distance for s in result.upstream]
        assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# TestPageRankEnrichment
# ---------------------------------------------------------------------------

class TestPageRankEnrichment:
    def test_include_pagerank_sets_scores(self, tmp_path):
        """include_pagerank=True should populate non-negative pagerank_score values."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"], max_depth=3, include_pagerank=True)

        all_syms = result.upstream + result.downstream
        # All scores should be non-negative floats (0.0 if node not in PR map)
        assert all(isinstance(s.pagerank_score, float) for s in all_syms)
        assert all(s.pagerank_score >= 0.0 for s in all_syms)

    def test_without_pagerank_scores_are_zero(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"], max_depth=3, include_pagerank=False)

        all_syms = result.upstream + result.downstream
        assert all(s.pagerank_score == 0.0 for s in all_syms)


# ---------------------------------------------------------------------------
# TestContextAssembly
# ---------------------------------------------------------------------------

class TestContextAssembly:
    def test_header_contains_seed_files(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"])
            text = r.assemble_context(result)
        assert "utils.py" in text
        assert "### Structural Context" in text

    def test_hop_label_present(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"])
            text = r.assemble_context(result)
        assert "[Hop 1]" in text

    def test_upstream_section_present(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"])
            text = r.assemble_context(result)
        assert "Upstream Impact" in text

    def test_downstream_section_present(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["service.py"])
            text = r.assemble_context(result)
        assert "Downstream Dependencies" in text

    def test_line_numbers_shown_when_available(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"])
            text = r.assemble_context(result)
        # service.py::Service.run has line_start=5 stored; should appear
        assert "Lines" in text

    def test_external_nodes_excluded_from_context(self, tmp_path):
        """IMPORTS/INHERITS edges create External placeholder nodes; exclude them."""
        store = _make_store(tmp_path)
        file_node = NodeInfo(
            kind="File", name="auth.py", qualified_name="auth.py",
            file_path="/proj/auth.py", line_start=1, line_end=20,
            language="python", file_hash="h",
        )
        cls_node = NodeInfo(
            kind="Class", name="Child", qualified_name="auth.py::Child",
            file_path="/proj/auth.py", line_start=3, line_end=15,
            language="python", file_hash="h",
        )
        with store:
            store.store_file_data(
                [file_node, cls_node],
                [
                    EdgeInfo(kind="CONTAINS", source="auth.py", target="auth.py::Child"),
                    EdgeInfo(kind="INHERITS", source="auth.py::Child", target="ExternalBase"),
                ],
            )
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["auth.py"])
            text = r.assemble_context(result)

        assert "ExternalBase" not in text

    def test_no_callers_message(self, tmp_path):
        """When upstream is empty the context should say 'No callers found'."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["main.py"])
            text = r.assemble_context(result)
        assert "No callers found" in text

    def test_no_deps_message(self, tmp_path):
        """When downstream is empty the context should say 'No dependencies found'."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"])
            text = r.assemble_context(result)
        assert "No dependencies found" in text

    def test_pagerank_score_shown_when_nonzero(self, tmp_path):
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["utils.py"], include_pagerank=True)
            # Manually set a non-zero score to test rendering
            if result.upstream:
                result.upstream[0].pagerank_score = 0.123
            text = r.assemble_context(result)
        if result.upstream:
            assert "PageRank=0.123" in text


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_store_upstream(self, tmp_path):
        with _make_store(tmp_path) as store:
            r = SurgicalRetriever(store)
            assert r.get_upstream_impact("any.py") == []

    def test_empty_store_downstream(self, tmp_path):
        with _make_store(tmp_path) as store:
            r = SurgicalRetriever(store)
            assert r.get_downstream_deps("any.py") == []

    def test_store_with_only_contains_no_calls(self, tmp_path):
        """A graph with only CONTAINS edges should produce no upstream/downstream."""
        with _make_store(tmp_path) as store:
            store.store_file_data(
                [_file_node("a.py"), _fn_node("a.py", "foo")],
                [_contains("a.py", "a.py::foo")],
            )
            r = SurgicalRetriever(store)
            result = r.get_full_impact(["a.py"])

        assert result.upstream == []
        assert result.downstream == []

    def test_max_depth_zero_returns_empty(self, tmp_path):
        """depth=0 means no hops — result must be empty."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            assert r.get_upstream_impact("utils.py", depth=0) == []
            assert r.get_downstream_deps("main.py", depth=0) == []

    def test_impacted_symbol_fields(self, tmp_path):
        """All ImpactedSymbol fields are populated from the graph query."""
        store = _make_call_graph(tmp_path)
        with store:
            r = SurgicalRetriever(store)
            syms = r.get_upstream_impact("utils.py", depth=1)

        assert syms
        sym = next(s for s in syms if s.qualified_name == "service.py::Service.run")
        assert sym.name == "Service.run"
        assert sym.file_rel_path == "service.py"
        assert sym.kind == "Function"
        assert sym.distance == 1
        assert sym.line_start > 0
        assert sym.line_end >= sym.line_start
