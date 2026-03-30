"""
IncrementalSync — git-diff-based incremental update of the codebase graph.

A full graph rebuild on a large repo (10,000+ files) can take minutes because
``CodebaseExtractor.extract()`` walks the entire directory tree and computes MD5
hashes for every file.  ``IncrementalSync`` reduces this to milliseconds by
asking git which files actually changed.

Pipeline for :meth:`IncrementalSync.sync`::

    1. git diff --name-status <base_ref>  →  SyncDelta (added/modified/deleted/renamed)
    2. Purge store data for deleted/modified/renamed-old paths.
    3. For each added/modified/renamed-new path:
         a. Compute MD5 — skip if hash unchanged (double-check against git).
         b. Parse with CodebaseExtractor._process_file().
         c. Store nodes and non-CALLS edges.
    4. Load ALL nodes from store  →  full definition_index (includes unchanged files).
    5. Resolve CALLS for newly-parsed call_sites.
    6. Store CALLS edges.

Usage::

    from berkelium_cli.sync import IncrementalSync
    from berkelium_cli.store import GraphQLiteStore

    with GraphQLiteStore(".berkelium/graph.db") as store:
        syncer = IncrementalSync(root="/path/to/repo", store=store)
        result = syncer.sync(base_ref="HEAD~1")
        print(f"Parsed {result.files_parsed} file(s), resolved {result.call_edges_resolved} CALLS edges")
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from berkelium_cli.store import GraphQLiteStore

from berkelium_cli.extractor import (
    EXTENSION_MAP,
    MAX_FILE_SIZE_BYTES,
    SKIP_DIRS,
    CodebaseExtractor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotAGitRepoError(Exception):
    """Raised when the root path is not inside a git repository."""


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class SyncDelta:
    """
    The set of file changes detected by ``git diff --name-status``.

    All paths are POSIX-style relative paths from the repo root
    (e.g. ``"src/auth.py"``), matching the format used as store keys.

    Attributes:
        added:    Paths of newly-created files (git status ``A``).
        modified: Paths of changed files (git status ``M``).
        deleted:  Paths of removed files (git status ``D``).
        renamed:  ``(old_path, new_path)`` pairs (git status ``R*``).
    """
    added:    list[str]             = field(default_factory=list)
    modified: list[str]             = field(default_factory=list)
    deleted:  list[str]             = field(default_factory=list)
    renamed:  list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_to_process(self) -> list[str]:
        """Paths to (re-)parse: added + modified + rename targets (new paths)."""
        return list(self.added) + list(self.modified) + [n for _, n in self.renamed]

    @property
    def all_to_purge(self) -> list[str]:
        """Paths whose store data must be deleted before re-parsing: modified + deleted + rename sources."""
        return list(self.modified) + list(self.deleted) + [o for o, _ in self.renamed]


@dataclass
class SyncResult:
    """
    Summary of what an :class:`IncrementalSync` run did.

    Attributes:
        delta:               The file-change delta that was processed.
        files_parsed:        Number of files actually re-parsed.
        files_skipped_hash:  Files git flagged as changed but whose MD5 matched the store.
        nodes_added:         Total graph nodes stored this run.
        edges_added:         Total graph edges stored this run (CONTAINS + CALLS + …).
        call_edges_resolved: Number of CALLS edges resolved and stored.
        errors:              Non-fatal error messages (file not found, parse errors, …).
        base_ref:            The git ref used for diffing.
    """
    delta:               SyncDelta
    files_parsed:        int       = 0
    files_skipped_hash:  int       = 0
    nodes_added:         int       = 0
    edges_added:         int       = 0
    call_edges_resolved: int       = 0
    errors:              list[str] = field(default_factory=list)
    base_ref:            str       = ""


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _compute_md5(file_path: Path) -> str:
    """Return the MD5 hex digest of *file_path*, or ``''`` on any OSError."""
    try:
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IncrementalSync:
    """
    Orchestrates incremental graph updates using ``git diff`` to find changed files.

    The class delegates all per-file parsing to ``CodebaseExtractor._process_file``
    and CALLS resolution to ``CodebaseExtractor._resolve_calls``.  It is
    intentionally self-contained — no changes to the existing extractor or store
    pipeline are required.

    Thread safety: Not thread-safe.  Create one instance per thread if needed.
    """

    def __init__(
        self,
        root: str | Path,
        store: "GraphQLiteStore",
        max_workers: int = 4,
        skip_dirs: frozenset[str] | None = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        """
        Args:
            root:          Absolute path to the git repository root.
            store:         An open :class:`~berkelium_cli.store.GraphQLiteStore`.
            max_workers:   Passed through to :class:`CodebaseExtractor` (unused in
                           the current serial implementation; reserved for future
                           parallel parsing).
            skip_dirs:     Override for the set of directory names to skip.
                           Defaults to ``SKIP_DIRS`` from the extractor.
            max_file_size: Files larger than this (bytes) are silently ignored.
        """
        self.root = Path(root).resolve()
        self.store = store
        self._extractor = CodebaseExtractor(
            root_path=self.root,
            store=store,
            max_workers=max_workers,
            skip_dirs=skip_dirs if skip_dirs is not None else SKIP_DIRS,
            max_file_size=max_file_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(
        self,
        base_ref: str = "HEAD",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> SyncResult:
        """
        Run an incremental sync against *base_ref*.

        Args:
            base_ref: Any git ref accepted by ``git diff --name-status``:

                      - ``"HEAD"`` (default) — staged + unstaged changes vs HEAD.
                      - ``"HEAD~1"`` — changes introduced by the last commit.
                      - A branch name like ``"main"`` — divergence from main.
                      - ``"--cached"`` — staged-only changes.

        Returns:
            :class:`SyncResult` with counts and any non-fatal error messages.

        Raises:
            NotAGitRepoError: if :attr:`root` is not inside a git repository,
                              or if ``git`` is not installed.
            subprocess.CalledProcessError: for other non-zero git exit codes
                (e.g. *base_ref* does not exist).
        """
        delta = self._compute_delta(base_ref)
        result = SyncResult(delta=delta, base_ref=base_ref)

        # Step 2: unconditionally purge deleted files and rename-old paths —
        # these are gone from disk, so no hash check is possible or needed.
        always_purge = list(delta.deleted) + [old for old, _ in delta.renamed]
        for rel_path in always_purge:
            try:
                self.store.delete_file_data(rel_path)
                logger.debug("Purged store data for '%s'", rel_path)
            except Exception as exc:
                msg = f"Purge failed for '{rel_path}': {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        # Step 3: parse changed / new files.
        # The MD5 double-check must happen BEFORE purging modified files so that
        # has_file_cached() can still find the node with its stored hash.
        # If the hash is unchanged, we skip both the purge and the re-parse.
        all_call_sites: list = []
        all_import_maps: dict[str, dict[str, str]] = {}

        _total_to_process = len(delta.all_to_process)
        for _idx, rel_path in enumerate(delta.all_to_process, start=1):
            file_path = self.root / rel_path

            # Resolve (validate existence, size, hash, language) before touching store
            pre = self._pre_check(file_path, rel_path, result)
            if pre is None:
                continue  # skipped or error recorded
            file_hash, language = pre

            # Now it's safe to purge the old version (hash confirmed changed)
            self.store.delete_file_data(rel_path)

            try:
                parse_out = self._extractor._process_file(file_path, language, file_hash)
            except Exception as exc:
                msg = f"Parse error for '{rel_path}': {exc}"
                logger.warning(msg)
                result.errors.append(msg)
                continue

            nodes, edges, call_sites, import_map = parse_out
            non_call_edges = [e for e in edges if e.kind != "CALLS"]
            self.store.store_file_data(nodes, non_call_edges)

            result.nodes_added  += len(nodes)
            result.edges_added  += len(non_call_edges)
            result.files_parsed += 1
            all_call_sites.extend(call_sites)
            all_import_maps[rel_path] = import_map
            if progress_callback:
                progress_callback(rel_path, _idx, _total_to_process, "extracting")

        # Step 4: load full definition_index from store (includes unchanged files)
        all_nodes = self.store.get_all_nodes()
        definition_index = {n.qualified_name: n for n in all_nodes}

        # Steps 5 & 6: resolve and store CALLS edges
        if all_call_sites:
            call_edges = self._extractor._resolve_calls(
                all_call_sites, definition_index, all_import_maps,
                progress_callback=progress_callback,
            )
            if call_edges:
                self.store.store_call_edges(call_edges)
                result.call_edges_resolved  = len(call_edges)
                result.edges_added         += len(call_edges)

        logger.info(
            "IncrementalSync complete (ref=%s): %d parsed, %d skipped (hash), "
            "%d nodes, %d edges (%d CALLS), %d error(s)",
            base_ref,
            result.files_parsed,
            result.files_skipped_hash,
            result.nodes_added,
            result.edges_added,
            result.call_edges_resolved,
            len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_delta(self, base_ref: str) -> SyncDelta:
        """
        Run ``git diff --name-status <base_ref>`` and return a :class:`SyncDelta`.

        Raises:
            NotAGitRepoError: git not found, or "not a git repository" in stderr.
            subprocess.CalledProcessError: any other non-zero exit code.
        """
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-status", base_ref],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise NotAGitRepoError(
                "git executable not found — is git installed?"
            ) from exc

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if "not a git repository" in stderr.lower():
                raise NotAGitRepoError(
                    f"'{self.root}' is not inside a git repository: {stderr}"
                )
            raise subprocess.CalledProcessError(
                proc.returncode, "git diff", output=proc.stdout, stderr=proc.stderr
            )

        return self._parse_name_status(proc.stdout)

    def _parse_name_status(self, output: str) -> SyncDelta:
        """
        Parse ``git diff --name-status`` text output into a :class:`SyncDelta`.

        Git output format (tab-separated)::

            A   src/new_file.py
            M   src/changed.py
            D   src/deleted.py
            R100    old/path.py   new/path.py
            R075    old/name.py   new/name.py

        The status field for renames includes a similarity score (``R100``, ``R075``,
        etc.) — any status starting with ``"R"`` is treated as a rename.

        Unsupported file extensions and paths in :attr:`skip_dirs` are filtered out.
        Git always outputs forward slashes so no platform-specific conversion is
        necessary; :meth:`_is_supported_file` uses :class:`pathlib.Path` which
        handles forward slashes on Windows correctly.
        """
        delta = SyncDelta()
        for line in output.splitlines():
            line = line.rstrip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0].strip()

            if status == "A":
                rel_path = parts[1].strip()
                if self._is_supported_file(rel_path):
                    delta.added.append(rel_path)

            elif status == "M":
                rel_path = parts[1].strip()
                if self._is_supported_file(rel_path):
                    delta.modified.append(rel_path)

            elif status == "D":
                rel_path = parts[1].strip()
                if self._is_supported_file(rel_path):
                    delta.deleted.append(rel_path)

            elif status.startswith("R"):
                # Rename: parts[1] = old path, parts[2] = new path
                if len(parts) < 3:
                    continue
                old_path = parts[1].strip()
                new_path = parts[2].strip()
                old_supported = self._is_supported_file(old_path)
                new_supported = self._is_supported_file(new_path)
                if old_supported and new_supported:
                    delta.renamed.append((old_path, new_path))
                elif old_supported:
                    # Renamed to an unsupported extension — treat as deletion
                    delta.deleted.append(old_path)
                # If only new is supported (old was unsupported), treat as addition
                elif new_supported:
                    delta.added.append(new_path)

            # C (copy), U (unmerged), X (unknown) — ignored intentionally

        return delta

    def _is_supported_file(self, rel_path: str) -> bool:
        """
        Return True if *rel_path* has a supported extension and is not inside a
        skip directory.

        Uses :data:`~berkelium_cli.extractor.EXTENSION_MAP` and the extractor's
        configured ``skip_dirs``.
        """
        p = Path(rel_path)
        if p.suffix.lower() not in EXTENSION_MAP:
            return False
        # Check every component of the path against skip_dirs
        if set(p.parts) & self._extractor.skip_dirs:
            return False
        return True

    def _pre_check(
        self,
        file_path: Path,
        rel_path: str,
        result: SyncResult,
    ) -> tuple[str, str] | None:
        """
        Validate a file and perform the MD5 double-check **before** any store
        mutation.

        Returns ``(file_hash, language)`` if the file should be (re-)parsed, or
        ``None`` if it should be skipped (missing, too large, hash unchanged,
        or unsupported extension).

        Non-fatal errors (file missing) are appended to *result.errors*.
        """
        if not file_path.exists():
            msg = f"File not found (deleted before parse?): '{rel_path}'"
            logger.warning(msg)
            result.errors.append(msg)
            return None

        try:
            if file_path.stat().st_size > self._extractor.max_file_size:
                logger.debug("Skipping large file: %s", rel_path)
                return None
        except OSError:
            return None

        # MD5 double-check: skip if the store already has this exact content.
        # This check must happen BEFORE store.delete_file_data() so the cached
        # node (with its file_hash property) is still present.
        file_hash = _compute_md5(file_path)
        if self.store.has_file_cached(rel_path, file_hash):
            logger.debug("Hash unchanged, skipping '%s'", rel_path)
            result.files_skipped_hash += 1
            return None

        # Determine language (reuse extractor's .h dialect sniffing)
        suffix = file_path.suffix.lower()
        language = EXTENSION_MAP.get(suffix)
        if language is None:
            return None
        if suffix == ".h":
            language = self._extractor._detect_h_dialect(file_path)

        return file_hash, language
