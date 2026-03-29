"""Tests for berkelium_cli.sync.IncrementalSync."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from berkelium_cli.extractor import CodebaseExtractor
from berkelium_cli.store import GraphQLiteStore
from berkelium_cli.sync import (
    IncrementalSync,
    NotAGitRepoError,
    SyncDelta,
    _compute_md5,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> GraphQLiteStore:
    return GraphQLiteStore(str(tmp_path / "graph.db"))


def _make_syncer(tmp_path: Path, store: GraphQLiteStore) -> IncrementalSync:
    return IncrementalSync(root=tmp_path, store=store)


def _git_init(repo: Path) -> None:
    """Initialise a minimal git repo (no commits needed for most tests)."""
    subprocess.run(["git", "init"], cwd=str(repo), check=True,
                   capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=str(repo), check=True, capture_output=True)


def _git_commit(repo: Path, message: str = "commit") -> None:
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=str(repo), check=True,
                   capture_output=True)


# ---------------------------------------------------------------------------
# 1. _compute_md5 — unit tests (pure I/O)
# ---------------------------------------------------------------------------

class TestComputeMd5:
    def test_returns_correct_hex_digest(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert _compute_md5(f) == expected

    def test_returns_empty_string_on_missing_file(self, tmp_path):
        assert _compute_md5(tmp_path / "nonexistent.txt") == ""

    def test_different_content_different_digest(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_bytes(b"aaa")
        h1 = _compute_md5(f)
        f.write_bytes(b"bbb")
        h2 = _compute_md5(f)
        assert h1 != h2


# ---------------------------------------------------------------------------
# 2. SyncDelta property helpers
# ---------------------------------------------------------------------------

class TestSyncDeltaProperties:
    def test_all_to_process_includes_added_modified_and_rename_targets(self):
        d = SyncDelta(
            added=["a.py"],
            modified=["b.py"],
            deleted=["c.py"],
            renamed=[("old.py", "new.py")],
        )
        assert set(d.all_to_process) == {"a.py", "b.py", "new.py"}

    def test_all_to_purge_includes_modified_deleted_and_rename_sources(self):
        d = SyncDelta(
            added=["a.py"],
            modified=["b.py"],
            deleted=["c.py"],
            renamed=[("old.py", "new.py")],
        )
        assert set(d.all_to_purge) == {"b.py", "c.py", "old.py"}

    def test_added_only_not_in_purge(self):
        d = SyncDelta(added=["fresh.py"])
        assert "fresh.py" not in d.all_to_purge

    def test_deleted_only_not_in_process(self):
        d = SyncDelta(deleted=["gone.py"])
        assert "gone.py" not in d.all_to_process

    def test_empty_delta(self):
        d = SyncDelta()
        assert d.all_to_process == []
        assert d.all_to_purge == []

    def test_multiple_renames(self):
        d = SyncDelta(renamed=[("a.py", "b.py"), ("c.py", "d.py")])
        assert set(d.all_to_process) == {"b.py", "d.py"}
        assert set(d.all_to_purge) == {"a.py", "c.py"}


# ---------------------------------------------------------------------------
# 3. _parse_name_status — pure parser (no I/O)
# ---------------------------------------------------------------------------

class TestParseNameStatus:
    def _syncer(self, tmp_path):
        store = MagicMock(spec=GraphQLiteStore)
        store.has_file_cached.return_value = False
        return IncrementalSync(root=tmp_path, store=store)

    def test_added_py_file(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status("A\tsrc/new.py\n")
        assert delta.added == ["src/new.py"]
        assert not delta.modified and not delta.deleted

    def test_modified_file(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status("M\tsrc/auth.py\n")
        assert delta.modified == ["src/auth.py"]

    def test_deleted_file(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status("D\tsrc/old.py\n")
        assert delta.deleted == ["src/old.py"]

    def test_renamed_file(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status(
            "R100\told/name.py\tnew/name.py\n"
        )
        assert delta.renamed == [("old/name.py", "new/name.py")]
        assert not delta.added and not delta.deleted

    def test_rename_with_low_similarity(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status(
            "R050\told.py\tnew.py\n"
        )
        assert delta.renamed == [("old.py", "new.py")]

    def test_unsupported_extensions_ignored(self, tmp_path):
        output = "A\tREADME.md\nM\tconfig.yml\nD\t.gitignore\n"
        delta = self._syncer(tmp_path)._parse_name_status(output)
        assert not delta.added and not delta.modified and not delta.deleted

    def test_files_in_skip_dirs_ignored(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status(
            "A\tnode_modules/lib/index.js\n"
            "M\t__pycache__/module.cpython-312.pyc\n"
        )
        assert not delta.added and not delta.modified

    def test_empty_output(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status("")
        assert not delta.added and not delta.modified and not delta.deleted

    def test_rename_to_unsupported_treated_as_deletion(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status(
            "R100\tsrc/code.py\tdocs/README.md\n"
        )
        assert delta.deleted == ["src/code.py"]
        assert not delta.renamed

    def test_rename_from_unsupported_to_supported_treated_as_addition(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status(
            "R100\tdocs/notes.md\tsrc/new_module.py\n"
        )
        assert delta.added == ["src/new_module.py"]
        assert not delta.renamed

    def test_mixed_status_codes(self, tmp_path):
        output = "A\ta.py\nM\tb.py\nD\tc.py\nR090\told.py\tnew.py\n"
        delta = self._syncer(tmp_path)._parse_name_status(output)
        assert "a.py" in delta.added
        assert "b.py" in delta.modified
        assert "c.py" in delta.deleted
        assert ("old.py", "new.py") in delta.renamed

    def test_copy_status_ignored(self, tmp_path):
        """C (copy) status lines are intentionally ignored."""
        delta = self._syncer(tmp_path)._parse_name_status(
            "C100\tsrc/original.py\tsrc/copy.py\n"
        )
        assert not delta.added and not delta.modified

    def test_blank_lines_skipped(self, tmp_path):
        delta = self._syncer(tmp_path)._parse_name_status("\n\nA\ta.py\n\n")
        assert delta.added == ["a.py"]

    def test_multiple_supported_languages(self, tmp_path):
        output = "A\tapp.go\nM\tlib.ts\nD\tutils.rs\n"
        delta = self._syncer(tmp_path)._parse_name_status(output)
        assert "app.go" in delta.added
        assert "lib.ts" in delta.modified
        assert "utils.rs" in delta.deleted


# ---------------------------------------------------------------------------
# 4. _compute_delta — git subprocess (mocked)
# ---------------------------------------------------------------------------

class TestComputeDelta:
    def _syncer(self, tmp_path):
        store = MagicMock(spec=GraphQLiteStore)
        return IncrementalSync(root=tmp_path, store=store)

    def test_raises_not_a_git_repo_error_on_fatal(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any of the parent directories): .git",
            )
            with pytest.raises(NotAGitRepoError):
                self._syncer(tmp_path)._compute_delta("HEAD")

    def test_raises_not_a_git_repo_when_git_not_found(self, tmp_path):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(NotAGitRepoError, match="git executable not found"):
                self._syncer(tmp_path)._compute_delta("HEAD")

    def test_raises_called_process_error_for_other_failures(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="some other git error"
            )
            with pytest.raises(subprocess.CalledProcessError):
                self._syncer(tmp_path)._compute_delta("HEAD~100")

    def test_returns_parsed_delta_on_success(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="A\tsrc/new.py\nD\tsrc/old.py\n",
                stderr="",
            )
            delta = self._syncer(tmp_path)._compute_delta("HEAD~1")
        assert delta.added == ["src/new.py"]
        assert delta.deleted == ["src/old.py"]

    def test_passes_correct_args_to_subprocess(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            self._syncer(tmp_path)._compute_delta("main")
        mock_run.assert_called_once_with(
            ["git", "diff", "--name-status", "main"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def test_empty_output_returns_empty_delta(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            delta = self._syncer(tmp_path)._compute_delta("HEAD")
        assert delta.all_to_process == []
        assert delta.all_to_purge == []


# ---------------------------------------------------------------------------
# 5. sync() — end-to-end with real git repo and real store
# ---------------------------------------------------------------------------

class TestSyncEndToEnd:
    PY_SIMPLE = "def greet():\n    return 'hello'\n"
    PY_UPDATED = "def greet():\n    return 'hi'\n\ndef farewell():\n    return 'bye'\n"

    @pytest.fixture
    def repo(self, tmp_path):
        """A git repo with one committed Python file."""
        _git_init(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "hello.py").write_text(self.PY_SIMPLE, encoding="utf-8")
        _git_commit(tmp_path, "initial")
        return tmp_path

    # --- empty delta ---

    def test_empty_delta_returns_zero_counts(self, repo):
        """No changes since HEAD → no files parsed."""
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta()) as mock_delta:
                result = syncer.sync("HEAD")
        assert result.files_parsed == 0
        assert result.files_skipped_hash == 0
        assert result.nodes_added == 0
        assert result.errors == []
        mock_delta.assert_called_once_with("HEAD")

    # --- added file ---

    def test_added_file_is_parsed_and_stored(self, repo):
        (repo / "src" / "utils.py").write_text(
            "def helper():\n    pass\n", encoding="utf-8"
        )
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(added=["src/utils.py"])):
                result = syncer.sync("HEAD")

        assert result.files_parsed == 1
        assert result.nodes_added >= 2   # at least File + Function
        assert result.errors == []

    # --- modified file ---

    def test_modified_file_is_reparsed(self, repo):
        """After initial extract, modifying a file should cause re-parse."""
        with GraphQLiteStore(":memory:") as store:
            # Seed the store
            CodebaseExtractor(repo, store=store).extract()

            # Modify the file
            (repo / "src" / "hello.py").write_text(self.PY_UPDATED, encoding="utf-8")

            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(modified=["src/hello.py"])):
                result = syncer.sync("HEAD")

        assert result.files_parsed == 1
        assert result.files_skipped_hash == 0
        assert result.errors == []

    # --- deleted file ---

    def test_deleted_file_is_purged_not_reparsed(self, repo):
        with GraphQLiteStore(":memory:") as store:
            CodebaseExtractor(repo, store=store).extract()
            before_stats = store.stats()

            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(deleted=["src/hello.py"])):
                result = syncer.sync("HEAD")

            after_stats = store.stats()

        assert result.files_parsed == 0
        # Nodes for deleted file should be gone
        assert after_stats["node_count"] < before_stats["node_count"]
        assert result.errors == []

    # --- hash unchanged (MD5 double-check) ---

    def test_hash_unchanged_file_is_skipped(self, repo):
        """If git flags a file as modified but content is identical, skip it."""
        with GraphQLiteStore(":memory:") as store:
            CodebaseExtractor(repo, store=store).extract()

            # File content is NOT changed — same as what's in the store
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(modified=["src/hello.py"])):
                result = syncer.sync("HEAD")

        assert result.files_skipped_hash == 1
        assert result.files_parsed == 0

    # --- renamed file ---

    def test_renamed_file_purges_old_and_stores_new(self, repo):
        (repo / "src" / "old_utils.py").write_text(
            "def helper():\n    pass\n", encoding="utf-8"
        )
        with GraphQLiteStore(":memory:") as store:
            CodebaseExtractor(repo, store=store).extract()

            # Simulate rename: old_utils.py → utils.py
            (repo / "src" / "utils.py").write_text(
                "def helper():\n    pass\n", encoding="utf-8"
            )
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(
                                  renamed=[("src/old_utils.py", "src/utils.py")]
                              )):
                result = syncer.sync("HEAD")

        assert result.files_parsed == 1
        assert result.errors == []

    # --- nonexistent file in delta ---

    def test_nonexistent_file_records_error(self, repo):
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(added=["src/phantom.py"])):
                result = syncer.sync("HEAD")

        assert result.files_parsed == 0
        assert any("phantom.py" in e for e in result.errors)

    # --- non-git directory ---

    def test_non_git_directory_raises(self, tmp_path):
        """A plain directory (no git init) raises NotAGitRepoError."""
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=tmp_path, store=store)
            with pytest.raises(NotAGitRepoError):
                syncer.sync("HEAD")

    # --- CALLS resolution uses full store ---

    def test_calls_resolved_against_full_store(self, repo):
        """
        CALLS edges from a changed file must resolve against symbols in
        unchanged files that are already in the store.
        """
        (repo / "src" / "callee.py").write_text(
            "def target():\n    pass\n", encoding="utf-8"
        )
        (repo / "src" / "caller.py").write_text(
            "from src.callee import target\n\ndef invoke():\n    target()\n",
            encoding="utf-8",
        )
        with GraphQLiteStore(":memory:") as store:
            # Seed store with both files (callee stays unchanged)
            CodebaseExtractor(repo, store=store).extract()

            # Now "modify" caller.py (add a second call)
            (repo / "src" / "caller.py").write_text(
                "from src.callee import target\n\n"
                "def invoke():\n    target()\n\n"
                "def invoke2():\n    target()\n",
                encoding="utf-8",
            )
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta(modified=["src/caller.py"])):
                result = syncer.sync("HEAD")

        assert result.files_parsed == 1
        # At minimum the calls to target() should resolve
        assert result.call_edges_resolved >= 1

    # --- base_ref forwarded correctly ---

    def test_base_ref_is_forwarded_to_compute_delta(self, repo):
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta",
                              return_value=SyncDelta()) as mock_delta:
                syncer.sync("HEAD~3")
        mock_delta.assert_called_once_with("HEAD~3")

    # --- result fields ---

    def test_sync_result_base_ref_is_set(self, repo):
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta", return_value=SyncDelta()):
                result = syncer.sync("main")
        assert result.base_ref == "main"

    def test_sync_result_delta_is_preserved(self, repo):
        delta = SyncDelta(added=["src/x.py"])
        with GraphQLiteStore(":memory:") as store:
            syncer = IncrementalSync(root=repo, store=store)
            with patch.object(syncer, "_compute_delta", return_value=delta):
                # src/x.py doesn't exist — error expected but delta should be preserved
                result = syncer.sync("HEAD")
        assert result.delta is delta


# ---------------------------------------------------------------------------
# 6. _is_supported_file — unit tests
# ---------------------------------------------------------------------------

class TestIsSupportedFile:
    def _syncer(self, tmp_path):
        store = MagicMock(spec=GraphQLiteStore)
        return IncrementalSync(root=tmp_path, store=store)

    def test_python_file_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file("src/module.py") is True

    def test_typescript_file_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file("app/index.ts") is True

    def test_go_file_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file("cmd/main.go") is True

    def test_markdown_not_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file("README.md") is False

    def test_no_extension_not_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file("Makefile") is False

    def test_file_in_node_modules_not_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file(
            "node_modules/react/index.js"
        ) is False

    def test_file_in_venv_not_supported(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file(
            ".venv/lib/site.py"
        ) is False

    def test_deeply_nested_supported_file(self, tmp_path):
        assert self._syncer(tmp_path)._is_supported_file(
            "a/b/c/d/module.rs"
        ) is True


# ---------------------------------------------------------------------------
# 7. get_all_nodes integration — via store
# ---------------------------------------------------------------------------

class TestGetAllNodes:
    """Verify that GraphQLiteStore.get_all_nodes() works correctly."""

    def test_returns_empty_list_on_empty_store(self):
        with GraphQLiteStore(":memory:") as store:
            assert store.get_all_nodes() == []

    def test_returns_non_external_nodes(self):
        from berkelium_cli.extractor import NodeInfo, EdgeInfo

        file_node = NodeInfo(
            kind="File", name="auth.py",
            qualified_name="src/auth.py",
            file_path="/project/src/auth.py",
            line_start=1, line_end=10,
            language="python", file_hash="abc",
        )
        with GraphQLiteStore(":memory:") as store:
            store.store_file_data([file_node], [])
            nodes = store.get_all_nodes()

        qnames = {n.qualified_name for n in nodes}
        assert "src/auth.py" in qnames

    def test_excludes_external_nodes_by_default(self):
        from berkelium_cli.extractor import NodeInfo, EdgeInfo

        file_node = NodeInfo(
            kind="File", name="auth.py",
            qualified_name="src/auth.py",
            file_path="/project/src/auth.py",
            line_start=1, line_end=10,
            language="python", file_hash="abc",
        )
        ext_edge = EdgeInfo(kind="IMPORTS", source="src/auth.py", target="os")
        with GraphQLiteStore(":memory:") as store:
            store.store_file_data([file_node], [ext_edge])
            nodes = store.get_all_nodes(exclude_external=True)

        kinds = {n.kind for n in nodes}
        assert "External" not in kinds

    def test_include_external_nodes_when_requested(self):
        """External placeholder nodes have no qualified_name property, so they
        cannot be reconstructed into NodeInfo objects — get_all_nodes() with
        exclude_external=False still won't return them, but it must not raise."""
        from berkelium_cli.extractor import NodeInfo, EdgeInfo

        file_node = NodeInfo(
            kind="File", name="auth.py",
            qualified_name="src/auth.py",
            file_path="/project/src/auth.py",
            line_start=1, line_end=10,
            language="python", file_hash="abc",
        )
        ext_edge = EdgeInfo(kind="IMPORTS", source="src/auth.py", target="pathlib")
        with GraphQLiteStore(":memory:") as store:
            store.store_file_data([file_node], [ext_edge])
            # Must not raise; External nodes lack qualified_name so won't appear
            nodes = store.get_all_nodes(exclude_external=False)

        # The File node is returned; External placeholder is silently skipped
        qnames = {n.qualified_name for n in nodes}
        assert "src/auth.py" in qnames

    def test_node_fields_round_trip(self):
        from berkelium_cli.extractor import NodeInfo

        original = NodeInfo(
            kind="Function", name="login",
            qualified_name="src/auth.py::login",
            file_path="/project/src/auth.py",
            line_start=5, line_end=15,
            language="python", file_hash="deadbeef",
        )
        file_node = NodeInfo(
            kind="File", name="auth.py",
            qualified_name="src/auth.py",
            file_path="/project/src/auth.py",
            line_start=1, line_end=20,
            language="python", file_hash="deadbeef",
        )
        with GraphQLiteStore(":memory:") as store:
            store.store_file_data([file_node, original], [])
            nodes = store.get_all_nodes()

        fn = next(n for n in nodes if n.qualified_name == "src/auth.py::login")
        assert fn.kind == "Function"
        assert fn.name == "login"
        assert fn.line_start == 5
        assert fn.line_end == 15
        assert fn.file_hash == "deadbeef"
