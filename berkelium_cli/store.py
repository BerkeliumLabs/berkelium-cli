"""
GraphQLite-backed persistence layer for the codebase knowledge graph.

GraphQLiteStore saves NodeInfo and EdgeInfo objects extracted by CodebaseExtractor
into a SQLite database (via GraphQLite), enabling:
  - Incremental extraction (skip re-parsing unchanged files)
  - Impact analysis (blast-radius Cypher queries)
  - Graph algorithms (PageRank, Louvain community detection)
  - Persistent cross-session graph state

Usage::

    from berkelium_cli.store import GraphQLiteStore
    from berkelium_cli.extractor import CodebaseExtractor

    with GraphQLiteStore(".berkelium/graph.db") as store:
        extractor = CodebaseExtractor("/path/to/project", store=store)
        nodes, edges = extractor.extract()
        print(store.stats())
        print(store.get_impacted_symbols("src/auth.py"))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from berkelium_cli.extractor import EdgeInfo, NodeInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (free functions used by the class)
# ---------------------------------------------------------------------------

def _file_rel_path_from_qname(qualified_name: str) -> str:
    """
    Extract the file's relative path from a node qualified name.

    Examples:
        "src/auth.py"                  → "src/auth.py"   (File node)
        "src/auth.py::AuthService"     → "src/auth.py"
        "src/auth.py::AuthService.login" → "src/auth.py"
    """
    return qualified_name.split("::")[0]


def _node_properties(node: "NodeInfo") -> dict:
    """Build the properties dict stored for a NodeInfo in GraphQLite."""
    return {
        # Store qualified_name explicitly so Cypher can RETURN it directly.
        # GraphQLite's Cypher RETURN n.id yields the internal integer rowid,
        # not the external string ID we passed to insert_nodes_bulk/upsert_node.
        "qualified_name": node.qualified_name,
        "name": node.name,
        "file_path": node.file_path,
        "file_rel_path": _file_rel_path_from_qname(node.qualified_name),
        "line_start": node.line_start,
        "line_end": node.line_end,
        "language": node.language,
        "file_hash": node.file_hash,
        # Store kind as property too (label cannot be retrieved easily in Cypher)
        "kind": node.kind,
    }


def _build_node_info_from_row(row: dict) -> "NodeInfo":
    """Reconstruct a NodeInfo from a Cypher result row (flat dict of properties)."""
    from berkelium_cli.extractor import NodeInfo

    return NodeInfo(
        kind=row.get("kind") or "Entity",
        name=row.get("name", ""),
        qualified_name=row.get("qualified_name", ""),
        file_path=row.get("file_path", ""),
        line_start=int(row.get("line_start") or 0),
        line_end=int(row.get("line_end") or 0),
        language=row.get("language", ""),
        file_hash=row.get("file_hash", ""),
    )


def _build_edge_info(source_qname: str, edge: dict) -> "EdgeInfo":
    """
    Reconstruct an EdgeInfo from a GraphQLite ``get_edges_from`` result dict.

    GraphQLite returns edges in the shape::

        {
            "source": "<external-id>",
            "target": "<external-id>",
            "r": {
                "id": <int>,
                "type": "REL_CONTAINS",   # sanitised rel type
                "startNode": <int>,
                "endNode": <int>,
                "properties": {"kind": "CONTAINS", "target_qname": "..."},
            },
        }

    We stored the original edge ``kind`` string as a ``kind`` property so we
    can recover it exactly (the ``r.type`` field is sanitised by GraphQLite).
    """
    from berkelium_cli.extractor import EdgeInfo

    # Properties are nested inside edge["r"]["properties"]
    r = edge.get("r") or {}
    props = r.get("properties") or {}
    kind = props.get("kind") or r.get("type", "")
    # edge["target"] is the external string ID passed to upsert/insert
    target = edge.get("target", "") or props.get("target_qname", "")
    return EdgeInfo(kind=kind, source=source_qname, target=str(target) if target else "")


# ---------------------------------------------------------------------------
# Main store class
# ---------------------------------------------------------------------------

class GraphQLiteStore:
    """
    Persists the codebase knowledge graph in a GraphQLite (SQLite-backed) database.

    The store is the source of truth for the extracted graph.  It is designed
    to work together with ``CodebaseExtractor`` for incremental extraction::

        with GraphQLiteStore() as store:
            extractor = CodebaseExtractor(root, store=store)
            nodes, edges = extractor.extract()   # skips unchanged files

    Thread safety: this class is NOT thread-safe.  All public methods should
    be called from a single thread.  ``CodebaseExtractor`` handles concurrency
    internally before writing to the store.
    """

    DEFAULT_DB_PATH: str = ".berkelium/graph.db"

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """
        Open (or create) the GraphQLite store at *db_path*.

        Parent directories are created automatically.  Use ``":memory:"`` for
        an in-process store that is discarded on close.

        Raises:
            RuntimeError: if the database cannot be opened.
        """
        from graphqlite import Graph  # deferred so tests can mock if needed

        resolved = Path(db_path)
        if str(db_path) != ":memory:":
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot create store directory '{resolved.parent}': {exc}"
                ) from exc

        try:
            self._graph: Graph = Graph(str(resolved))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open GraphQLite store at '{resolved}': {exc}"
            ) from exc

        logger.debug("GraphQLiteStore opened at '%s'", resolved)

    # ------------------------------------------------------------------
    # Incremental cache helpers
    # ------------------------------------------------------------------

    def has_file_cached(self, rel_path: str, file_hash: str) -> bool:
        """
        Return True iff the store contains a File node for *rel_path* whose
        ``file_hash`` property matches *file_hash*.

        A False return means the file is new or has changed and needs to be
        (re-)parsed.  Errors are caught and logged; False is returned on any
        failure so that re-parsing is the safe fallback.
        """
        try:
            node = self._graph.get_node(rel_path)
            if node is None:
                return False
            cached_hash = node.get("properties", {}).get("file_hash", "")
            return cached_hash == file_hash and bool(file_hash)
        except Exception as exc:
            logger.warning("Cache check failed for '%s': %s", rel_path, exc)
            return False

    def delete_file_data(self, rel_path: str) -> None:
        """
        Remove all nodes (and their edges) that belong to *rel_path* from the
        store.

        This is called before re-parsing a file to avoid duplicate data.
        Errors are logged but not re-raised so that extraction can continue.
        """
        try:
            # Use n.qualified_name (a stored property) — NOT n.id, which yields
            # GraphQLite's internal integer rowid rather than our external string ID.
            results = self._graph.query(
                "MATCH (n) WHERE n.file_rel_path = $p RETURN n.qualified_name AS qname",
                params={"p": rel_path},
            )
            qnames = [row["qname"] for row in results if row.get("qname")]
            for qname in qnames:
                try:
                    self._graph.delete_node(qname)
                except Exception as del_exc:
                    logger.debug("Could not delete node '%s': %s", qname, del_exc)
            if qnames:
                logger.debug(
                    "Deleted %d node(s) for file '%s'", len(qnames), rel_path
                )
        except Exception as exc:
            logger.warning("Failed to delete file data for '%s': %s", rel_path, exc)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store_file_data(
        self,
        nodes: list["NodeInfo"],
        edges: list["EdgeInfo"],
    ) -> None:
        """
        Persist *nodes* and *edges* for a single file into the store.

        The method is idempotent when called after ``delete_file_data`` for the
        same file.  It uses bulk insert for nodes (fast path) and upsert for
        edges, creating lightweight ``External`` placeholder nodes for any edge
        target that does not exist in the current batch (e.g. third-party
        imports, base classes from other packages).

        Errors are logged but not re-raised so that a single bad file does not
        abort the entire extraction.
        """
        if not nodes:
            return

        try:
            # ---- Phase A: bulk-insert nodes --------------------------------
            node_tuples = [
                (n.qualified_name, _node_properties(n), n.kind)
                for n in nodes
            ]
            try:
                self._graph.insert_nodes_bulk(node_tuples)
            except Exception as bulk_exc:
                # Fallback to upsert in case some nodes already exist
                logger.debug(
                    "Bulk node insert failed (%s), falling back to upsert", bulk_exc
                )
                self._graph.upsert_nodes_batch(node_tuples)

            # ---- Phase B: ensure external targets exist --------------------
            known_ids = {t[0] for t in node_tuples}
            external_targets: set[str] = set()
            for e in edges:
                if e.target and e.target not in known_ids:
                    external_targets.add(e.target)
            if external_targets:
                self._ensure_external_nodes(external_targets)

            # ---- Phase C: upsert edges -------------------------------------
            if edges:
                edge_tuples = [
                    # Store target_qname explicitly: GraphQLite's get_edges_from
                    # may return the internal integer rowid as "target" rather
                    # than the external string ID, so we persist it ourselves.
                    (e.source, e.target, {"kind": e.kind, "target_qname": e.target}, e.kind)
                    for e in edges
                    if e.source in known_ids and e.target  # only edges from our nodes
                ]
                if edge_tuples:
                    try:
                        self._graph.upsert_edges_batch(edge_tuples)
                    except Exception as edge_exc:
                        logger.warning(
                            "Edge batch upsert failed for file data: %s", edge_exc
                        )

        except Exception as exc:
            logger.error(
                "Failed to store file data (%d nodes, %d edges): %s",
                len(nodes), len(edges), exc,
                exc_info=True,
            )

    def store_call_edges(self, call_edges: list["EdgeInfo"]) -> None:
        """
        Persist CALLS edges produced by pass-2 call resolution.

        Both endpoints of CALLS edges are guaranteed to be in the definition
        index (the extractor only emits resolved targets), so no placeholder
        nodes are needed here.  Errors are logged but not re-raised.
        """
        if not call_edges:
            return
        try:
            edge_tuples = [
                (e.source, e.target, {"kind": "CALLS", "target_qname": e.target}, "CALLS")
                for e in call_edges
                if e.source and e.target
            ]
            if edge_tuples:
                self._graph.upsert_edges_batch(edge_tuples)
                logger.debug("Stored %d CALLS edge(s)", len(edge_tuples))
        except Exception as exc:
            logger.warning("Failed to store CALLS edges: %s", exc)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_file_data(
        self, rel_path: str
    ) -> tuple[list["NodeInfo"], list["EdgeInfo"]]:
        """
        Retrieve all nodes and their outgoing edges for *rel_path* from the
        store, reconstructing ``NodeInfo`` / ``EdgeInfo`` objects.

        Returns ``([], [])`` on any error so that callers can fall back to
        re-parsing gracefully.
        """
        try:
            # Retrieve all node properties in one Cypher query.
            # We use n.qualified_name (a stored property) instead of n.id because
            # GraphQLite's Cypher RETURN n.id yields the internal integer rowid.
            results = self._graph.query(
                """
                MATCH (n) WHERE n.file_rel_path = $p
                RETURN n.qualified_name AS qualified_name,
                       n.name          AS name,
                       n.kind          AS kind,
                       n.language      AS language,
                       n.file_path     AS file_path,
                       n.line_start    AS line_start,
                       n.line_end      AS line_end,
                       n.file_hash     AS file_hash
                """,
                params={"p": rel_path},
            )
            nodes: list["NodeInfo"] = []
            edges: list["EdgeInfo"] = []

            for row in results:
                qname = row.get("qualified_name", "")
                if not qname:
                    continue
                try:
                    nodes.append(_build_node_info_from_row(row))
                except Exception as nb_exc:
                    logger.debug(
                        "Could not rebuild NodeInfo for '%s': %s", qname, nb_exc
                    )
                    continue

                # Retrieve outgoing edges using the external string ID
                try:
                    for edge in self._graph.get_edges_from(qname):
                        try:
                            edges.append(_build_edge_info(qname, edge))
                        except Exception as eb_exc:
                            logger.debug(
                                "Could not rebuild EdgeInfo from '%s': %s", qname, eb_exc
                            )
                except Exception as eg_exc:
                    logger.debug(
                        "get_edges_from failed for '%s': %s", qname, eg_exc
                    )

            return nodes, edges

        except Exception as exc:
            logger.warning(
                "Failed to load cached data for '%s': %s", rel_path, exc
            )
            return [], []

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def get_impacted_symbols(self, target_file: str) -> list[dict]:
        """
        Return symbols that could be affected if *target_file* changes.

        Uses a Cypher traversal of up to 5 CALLS hops from any function
        defined in *target_file* to find callers across the codebase.

        Args:
            target_file: Relative path of the file (e.g. ``"src/auth.py"``).

        Returns:
            List of dicts with keys ``name``, ``file_rel_path``, ``kind``.
            Returns an empty list on any query error.
        """
        cypher = """
        MATCH (f {file_rel_path: $path})-[:CONTAINS]->(func)
        MATCH (impacted)-[:CALLS*1..5]->(func)
        RETURN DISTINCT impacted.name AS name,
                        impacted.file_rel_path AS file_rel_path,
                        impacted.kind AS kind
        """
        try:
            return list(self._graph.query(cypher, params={"path": target_file}))
        except Exception as exc:
            logger.error(
                "Impact analysis query failed for '%s': %s", target_file, exc
            )
            return []

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """
        Execute a raw Cypher query against the store.

        This is the public escape hatch for advanced consumers (e.g. the
        ``SurgicalRetriever``) that need to run custom traversals.  Errors
        are caught, logged, and an empty list is returned so callers can
        degrade gracefully.

        Args:
            cypher: A Cypher query string.  Use ``$param`` placeholders.
            params: Optional dict of parameter bindings.

        Returns:
            List of result row dicts, or ``[]`` on any error.
        """
        try:
            return list(self._graph.query(cypher, params=params or {}))
        except Exception as exc:
            logger.warning("Cypher query failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def pagerank(self) -> list[dict]:
        """
        Run PageRank over the full graph to identify the most critical nodes.

        The graph is loaded into the in-memory CSR cache before running the
        algorithm.  Returns an empty list on error.
        """
        try:
            self._graph.load_graph()
            return list(self._graph.pagerank())
        except Exception as exc:
            logger.error("PageRank failed: %s", exc)
            return []

    def louvain(self) -> list[dict]:
        """
        Run the Louvain community-detection algorithm to find code modules.

        Returns an empty list on error.
        """
        try:
            self._graph.load_graph()
            return list(self._graph.louvain())
        except Exception as exc:
            logger.error("Louvain community detection failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return ``{"node_count": int, "edge_count": int}``."""
        try:
            return self._graph.stats()
        except Exception as exc:
            logger.warning("stats() failed: %s", exc)
            return {"node_count": 0, "edge_count": 0}

    def clear(self) -> None:
        """
        Delete ALL nodes and edges from the store.

        Use this to force a full re-extraction on the next ``extract()`` call.
        """
        try:
            self._graph.query("MATCH (n) DETACH DELETE n")
            logger.info("Store cleared")
        except Exception as exc:
            logger.warning("clear() failed: %s", exc)

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self._graph.close()
        except Exception as exc:
            logger.debug("close() raised: %s", exc)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GraphQLiteStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_external_nodes(self, targets: set[str]) -> None:
        """
        Create lightweight ``External`` placeholder nodes for *targets* that
        do not yet exist in the store.

        These placeholders allow edges to reference symbols from third-party
        libraries, base classes outside the codebase, etc.
        """
        for target in targets:
            try:
                if not self._graph.has_node(target):
                    self._graph.upsert_node(
                        target,
                        {
                            "name": target.split("::")[-1].split(".")[-1],
                            "file_path": "",
                            "file_rel_path": "",
                            "line_start": 0,
                            "line_end": 0,
                            "language": "",
                            "file_hash": "",
                            "kind": "External",
                            "external": True,
                        },
                        "External",
                    )
            except Exception as exc:
                logger.debug(
                    "Could not create external placeholder for '%s': %s", target, exc
                )
