"""
Surgical Retriever — graph-powered context retrieval for code changes.

Given a set of changed files, ``SurgicalRetriever`` traverses the CALLS edges in
the ``GraphQLiteStore`` to discover:

- **Upstream callers** ("blast radius") — every component that calls into the
  changed files, up to *max_depth* hops away.
- **Downstream dependencies** ("implementation context") — every component
  called by the changed files, up to *max_depth* hops away.

Results are ranked by graph distance and optionally enriched with PageRank
centrality scores.  The ``assemble_context`` method formats findings as
structured Markdown suitable for injection into an LLM context window.

Usage::

    from berkelium_cli.store import GraphQLiteStore
    from berkelium_cli.retriever import SurgicalRetriever

    with GraphQLiteStore(".berkelium/graph.db") as store:
        retriever = SurgicalRetriever(store, max_depth=3)
        result = retriever.get_full_impact(["src/auth.py"], include_pagerank=True)
        print(retriever.assemble_context(result))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from berkelium_cli.store import GraphQLiteStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class ImpactedSymbol:
    """
    A code symbol discovered during upstream or downstream graph traversal.

    Attributes:
        name:            Short symbol name (e.g. ``"login"``).
        qualified_name:  Globally unique ID (e.g. ``"src/auth.py::AuthService.login"``).
        file_rel_path:   Relative path of the containing file.
        file_path:       Absolute path of the containing file.
        kind:            Symbol type: ``"File"``, ``"Class"``, ``"Function"``, etc.
        direction:       ``"upstream"`` (caller) or ``"downstream"`` (callee).
        distance:        Number of CALLS hops from the seed file.
        line_start:      First line of the symbol definition (1-indexed).
        line_end:        Last line of the symbol definition (1-indexed).
        pagerank_score:  PageRank centrality score (0.0 if not computed).
    """
    name: str
    qualified_name: str
    file_rel_path: str
    file_path: str
    kind: str
    direction: str
    distance: int
    line_start: int = 0
    line_end: int = 0
    pagerank_score: float = 0.0


@dataclass
class RetrievalResult:
    """
    Complete output of a surgical retrieval pass.

    Attributes:
        seed_files: The changed file paths that were used as traversal seeds.
        upstream:   Callers, sorted by distance then kind priority then name.
        downstream: Callees, sorted by distance then kind priority then name.
        max_depth:  The traversal depth used to produce these results.
    """
    seed_files: list[str]
    upstream: list[ImpactedSymbol] = field(default_factory=list)
    downstream: list[ImpactedSymbol] = field(default_factory=list)
    max_depth: int = 3


# ---------------------------------------------------------------------------
# Kind priority for sorting (lower = higher priority in output)
# ---------------------------------------------------------------------------

_KIND_PRIORITY: dict[str, int] = {
    "Class":     0,
    "Interface": 1,
    "Function":  2,
    "Test":      3,
    "File":      4,
}


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class SurgicalRetriever:
    """
    Traverses the code graph to find components functionally linked to a change.

    The retriever is intentionally read-only with respect to the store — it
    never writes nodes or edges.  All traversal is done via Cypher queries
    executed through ``GraphQLiteStore.query()``.

    Thread safety: Not thread-safe.  Create one instance per thread if needed.
    """

    DEFAULT_MAX_DEPTH: int = 3

    def __init__(
        self,
        store: "GraphQLiteStore",
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        """
        Args:
            store:     A ``GraphQLiteStore`` instance (must be open).
            max_depth: Maximum number of CALLS hops to follow in either
                       direction.  Higher values find more transitive
                       dependents but increase query time.  Default: 3.
        """
        self.store = store
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_upstream_impact(
        self,
        file_rel_path: str,
        depth: int | None = None,
    ) -> list[ImpactedSymbol]:
        """
        Find all callers of symbols defined in *file_rel_path*.

        Follows CALLS edges in the upstream direction (caller → callee).
        Results are deduped; each symbol appears at its *shortest* distance
        from the seed file.

        Args:
            file_rel_path: Relative path of the changed file (forward slashes).
            depth:         Override for max hops.  Defaults to ``self.max_depth``.

        Returns:
            Sorted list of :class:`ImpactedSymbol` objects (distance ASC).
        """
        return self._traverse(file_rel_path, "upstream", depth if depth is not None else self.max_depth)

    def get_downstream_deps(
        self,
        file_rel_path: str,
        depth: int | None = None,
    ) -> list[ImpactedSymbol]:
        """
        Find all symbols called by code in *file_rel_path*.

        Follows CALLS edges in the downstream direction (caller → callee).
        Results are deduped; each symbol appears at its *shortest* distance
        from the seed file.

        Args:
            file_rel_path: Relative path of the changed file (forward slashes).
            depth:         Override for max hops.  Defaults to ``self.max_depth``.

        Returns:
            Sorted list of :class:`ImpactedSymbol` objects (distance ASC).
        """
        return self._traverse(file_rel_path, "downstream", depth if depth is not None else self.max_depth)

    def get_full_impact(
        self,
        changed_files: list[str],
        max_depth: int | None = None,
        include_pagerank: bool = False,
    ) -> RetrievalResult:
        """
        Run a full surgical retrieval for a set of changed files.

        Combines upstream and downstream traversals across all seed files,
        deduplicates (keeping shortest path when a symbol appears from multiple
        seeds), and optionally enriches results with PageRank centrality.

        Args:
            changed_files:    List of relative file paths (forward slashes).
            max_depth:        Override for traversal depth.
            include_pagerank: If True, enriches all results with PageRank scores.
                              Requires an extra ``load_graph()`` + ``pagerank()``
                              call; omit for large codebases where speed matters.

        Returns:
            :class:`RetrievalResult` with upstream and downstream lists ranked
            by distance, then kind priority, then qualified name.
        """
        depth = max_depth if max_depth is not None else self.max_depth
        result = RetrievalResult(seed_files=list(changed_files), max_depth=depth)

        upstream_seen: dict[str, ImpactedSymbol] = {}
        downstream_seen: dict[str, ImpactedSymbol] = {}

        for fp in changed_files:
            for sym in self.get_upstream_impact(fp, depth=depth):
                if sym.qualified_name not in upstream_seen or \
                        sym.distance < upstream_seen[sym.qualified_name].distance:
                    upstream_seen[sym.qualified_name] = sym

            for sym in self.get_downstream_deps(fp, depth=depth):
                if sym.qualified_name not in downstream_seen or \
                        sym.distance < downstream_seen[sym.qualified_name].distance:
                    downstream_seen[sym.qualified_name] = sym

        sort_key = lambda s: (s.distance, _KIND_PRIORITY.get(s.kind, 99), s.qualified_name)
        result.upstream = sorted(upstream_seen.values(), key=sort_key)
        result.downstream = sorted(downstream_seen.values(), key=sort_key)

        if include_pagerank:
            self._enrich_with_pagerank(result.upstream + result.downstream)

        return result

    def assemble_context(self, result: RetrievalResult) -> str:
        """
        Format a :class:`RetrievalResult` as structured Markdown for LLM injection.

        The output includes:
        - A summary header with seed files and traversal depth.
        - Upstream callers with hop distance, kind, and line range.
        - Downstream dependencies with hop distance, kind, and line range.
        - PageRank scores when available (non-zero).

        ``External`` placeholder nodes (third-party symbols) are excluded from
        the output because they don't correspond to editable source files.

        Args:
            result: The :class:`RetrievalResult` to format.

        Returns:
            A Markdown string ready for inclusion in an LLM prompt.
        """
        lines: list[str] = []

        seed_label = ", ".join(f"`{f}`" for f in result.seed_files)
        lines.append(f"### Structural Context for {seed_label}")
        lines.append("")
        lines.append(f"**Changed Files:** {seed_label}")
        lines.append(f"**Traversal depth:** {result.max_depth} hops")
        lines.append("")

        # ---- Upstream section -----------------------------------------------
        visible_upstream = [s for s in result.upstream if s.kind != "External"]
        if visible_upstream:
            lines.append(
                f"**Upstream Impact (Callers)** — "
                f"{len(visible_upstream)} component(s) call into the changed files:"
            )
            lines.append("")
            for sym in visible_upstream:
                score_str = f" · PageRank={sym.pagerank_score:.3f}" if sym.pagerank_score else ""
                lines.append(
                    f"- [Hop {sym.distance}] `{sym.file_rel_path}` → "
                    f"`{sym.qualified_name}` ({sym.kind}){score_str}"
                )
                if sym.line_start:
                    end = f"–{sym.line_end}" if sym.line_end and sym.line_end != sym.line_start else ""
                    lines.append(f"  Lines {sym.line_start}{end}")
        else:
            lines.append("**Upstream Impact:** No callers found within traversal depth.")
        lines.append("")

        # ---- Downstream section ---------------------------------------------
        visible_downstream = [s for s in result.downstream if s.kind != "External"]
        if visible_downstream:
            lines.append(
                f"**Downstream Dependencies** — "
                f"{len(visible_downstream)} component(s) used by the changed files:"
            )
            lines.append("")
            for sym in visible_downstream:
                score_str = f" · PageRank={sym.pagerank_score:.3f}" if sym.pagerank_score else ""
                lines.append(
                    f"- [Hop {sym.distance}] `{sym.file_rel_path}` → "
                    f"`{sym.qualified_name}` ({sym.kind}){score_str}"
                )
                if sym.line_start:
                    end = f"–{sym.line_end}" if sym.line_end and sym.line_end != sym.line_start else ""
                    lines.append(f"  Lines {sym.line_start}{end}")
        else:
            lines.append("**Downstream Dependencies:** No dependencies found within traversal depth.")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _traverse(
        self,
        file_rel_path: str,
        direction: str,
        max_d: int,
    ) -> list[ImpactedSymbol]:
        """Dispatch to the correct traversal strategy by direction."""
        if direction == "upstream":
            return self._traverse_upstream(file_rel_path, max_d)
        return self._traverse_downstream(file_rel_path, max_d)

    def _traverse_upstream(
        self,
        file_rel_path: str,
        max_d: int,
    ) -> list[ImpactedSymbol]:
        """
        File-level BFS to find all callers up to *max_d* hops away.

        GraphQLite cannot use a WHERE-bound variable as the endpoint of a
        variable-length path (``MATCH (caller)-[:CALLS*1..n]->(pre_bound)``
        fails with "ambiguous column name").  The workaround is to run a
        single-hop query per BFS level:

            MATCH (caller)-[:CALLS]->(target) WHERE target.file_rel_path = $path

        The BFS frontier tracks files rather than individual qnames, so the
        number of Cypher calls per depth level equals the number of unique
        source files at that depth — manageable even for large codebases.
        """
        found: dict[str, ImpactedSymbol] = {}
        frontier_files: set[str] = {file_rel_path}
        searched_files: set[str] = set()

        for d in range(1, max_d + 1):
            if not frontier_files:
                break

            new_caller_qnames: dict[str, dict] = {}

            for fp in frontier_files:
                searched_files.add(fp)
                try:
                    rows = self.store.query(
                        """
                        MATCH (caller)-[:CALLS]->(target)
                        WHERE target.file_rel_path = $path
                        RETURN DISTINCT
                               caller.qualified_name AS qualified_name,
                               caller.name           AS name,
                               caller.kind           AS kind,
                               caller.file_path      AS file_path,
                               caller.file_rel_path  AS file_rel_path,
                               caller.line_start     AS line_start,
                               caller.line_end       AS line_end
                        """,
                        params={"path": fp},
                    )
                except Exception as exc:
                    logger.warning(
                        "Upstream BFS query failed (depth=%d, file=%s): %s", d, fp, exc
                    )
                    continue

                for row in rows:
                    qname = row.get("qualified_name", "")
                    caller_file = row.get("file_rel_path", "")
                    kind = row.get("kind", "")
                    # Exclude seed-file self-callers, already-found nodes, and Externals
                    if (qname and qname not in found
                            and caller_file != file_rel_path
                            and kind != "External"):
                        new_caller_qnames[qname] = row

            if not new_caller_qnames:
                break

            new_frontier_files: set[str] = set()
            for qname, row in new_caller_qnames.items():
                caller_file = row.get("file_rel_path", "")
                found[qname] = ImpactedSymbol(
                    name=row.get("name", ""),
                    qualified_name=qname,
                    file_rel_path=caller_file,
                    file_path=row.get("file_path", ""),
                    kind=row.get("kind", ""),
                    direction="upstream",
                    distance=d,
                    line_start=int(row.get("line_start") or 0),
                    line_end=int(row.get("line_end") or 0),
                )
                if caller_file and caller_file not in searched_files:
                    new_frontier_files.add(caller_file)

            frontier_files = new_frontier_files

        return sorted(found.values(), key=lambda s: (s.distance, s.qualified_name))

    def _traverse_downstream(
        self,
        file_rel_path: str,
        max_d: int,
    ) -> list[ImpactedSymbol]:
        """
        Iterative Cypher query to find all callees up to *max_d* hops away.

        Downstream traversal (source-bound variable at head of path) works with
        GraphQLite's variable-length path support.  Running separate queries for
        depth 1, 2, … *max_d* lets us assign each node its exact first-encounter
        distance without needing ``length(path)`` support in the Cypher engine.
        """
        found: dict[str, ImpactedSymbol] = {}

        for d in range(1, max_d + 1):
            try:
                rows = self.store.query(
                    f"""
                    MATCH (source) WHERE source.file_rel_path = $path
                    MATCH (source)-[:CALLS*1..{d}]->(dep)
                    RETURN DISTINCT
                           dep.qualified_name  AS qualified_name,
                           dep.name            AS name,
                           dep.kind            AS kind,
                           dep.file_path       AS file_path,
                           dep.file_rel_path   AS file_rel_path,
                           dep.line_start      AS line_start,
                           dep.line_end        AS line_end
                    """,
                    params={"path": file_rel_path},
                )
            except Exception as exc:
                logger.warning(
                    "Downstream query failed (depth=%d, file=%s): %s", d, file_rel_path, exc
                )
                continue

            for row in rows:
                qname = row.get("qualified_name", "")
                kind = row.get("kind", "")
                if (qname and qname not in found
                        and row.get("file_rel_path") != file_rel_path
                        and kind != "External"):
                    found[qname] = ImpactedSymbol(
                        name=row.get("name", ""),
                        qualified_name=qname,
                        file_rel_path=row.get("file_rel_path", ""),
                        file_path=row.get("file_path", ""),
                        kind=row.get("kind", ""),
                        direction="downstream",
                        distance=d,
                        line_start=int(row.get("line_start") or 0),
                        line_end=int(row.get("line_end") or 0),
                    )

        return sorted(found.values(), key=lambda s: (s.distance, s.qualified_name))

    def _enrich_with_pagerank(self, symbols: list[ImpactedSymbol]) -> None:
        """
        Enrich *symbols* in-place with PageRank centrality scores.

        PageRank results have the form ``[{"node_id": str, "score": float, ...}]``.
        Scores are matched to symbols via ``node_id == sym.qualified_name``.
        Any error (e.g. empty graph, algorithm failure) is silently logged so
        that the caller's result is still returned without scores.
        """
        try:
            pr_results = self.store.pagerank()
            pr_map: dict[str, float] = {}
            for row in pr_results:
                node_id = str(row.get("node_id") or row.get("id") or "")
                score = float(row.get("score") or row.get("pagerank") or 0.0)
                if node_id:
                    pr_map[node_id] = score

            for sym in symbols:
                sym.pagerank_score = pr_map.get(sym.qualified_name, 0.0)

            logger.debug(
                "PageRank enriched %d symbol(s) from %d scored nodes",
                len(symbols), len(pr_map),
            )
        except Exception as exc:
            logger.warning("PageRank enrichment failed: %s", exc)
