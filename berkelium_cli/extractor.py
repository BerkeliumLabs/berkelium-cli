"""
Codebase extractor: walks a directory tree, parses source files with tree-sitter,
and produces a normalized graph of NodeInfo (definitions) and EdgeInfo (relationships).
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",   # may be reclassified to "cpp" by content sniffing
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
}

SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".env", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", "vendor",
})

MAX_FILE_SIZE_BYTES: int = 1_000_000  # 1 MB

# C++ content signals used for .h dialect sniffing
_CPP_SIGNALS: tuple[bytes, ...] = (
    b"class ", b"namespace ", b"template<", b"template <",
    b"public:", b"private:", b"protected:", b"::",
)

# ---------------------------------------------------------------------------
# Language → Tree-sitter node type mappings
# ---------------------------------------------------------------------------

# Maps language → category → list of tree-sitter node type strings
LANGUAGE_NODE_MAPS: dict[str, dict[str, list[str]]] = {
    "python": {
        "class":    ["class_definition"],
        "function": ["function_definition"],
        "decorated": ["decorated_definition"],
        "import":   ["import_statement", "import_from_statement"],
        "call":     ["call"],
        "variable": ["assignment", "augmented_assignment"],
    },
    "javascript": {
        "class":    ["class_declaration", "class_expression"],
        "function": ["function_declaration", "function_expression",
                     "arrow_function", "method_definition"],
        "import":   ["import_statement"],
        "call":     ["call_expression"],
        "variable": ["variable_declarator"],
    },
    "typescript": {
        "class":     ["class_declaration", "class_expression"],
        "interface": ["interface_declaration"],
        "function":  ["function_declaration", "function_expression",
                      "arrow_function", "method_definition"],
        "import":    ["import_statement"],
        "call":      ["call_expression"],
        "variable":  ["variable_declarator"],
    },
    "tsx": {
        "class":     ["class_declaration", "class_expression"],
        "interface": ["interface_declaration"],
        "function":  ["function_declaration", "function_expression",
                      "arrow_function", "method_definition"],
        "import":    ["import_statement"],
        "call":      ["call_expression"],
        "variable":  ["variable_declarator"],
    },
    "go": {
        "class":    ["type_spec"],
        "function": ["function_declaration", "method_declaration"],
        "import":   ["import_declaration", "import_spec"],
        "call":     ["call_expression"],
        "variable": ["var_declaration", "short_var_declaration", "const_declaration"],
    },
    "java": {
        "class":    ["class_declaration", "interface_declaration",
                     "enum_declaration", "annotation_type_declaration"],
        "function": ["method_declaration", "constructor_declaration"],
        "import":   ["import_declaration"],
        "call":     ["method_invocation", "object_creation_expression"],
        "variable": ["field_declaration", "local_variable_declaration"],
    },
    "rust": {
        "class":    ["struct_item", "enum_item", "trait_item"],
        "impl":     ["impl_item"],
        "function": ["function_item", "function_signature_item"],
        "import":   ["use_declaration"],
        "call":     ["call_expression", "method_call_expression"],
        "variable": ["let_declaration", "static_item", "const_item"],
    },
    "c": {
        "class":    ["struct_specifier", "union_specifier", "type_definition"],
        "function": ["function_definition"],
        "import":   ["preproc_include"],
        "call":     ["call_expression"],
        "variable": ["declaration"],
    },
    "cpp": {
        "class":    ["class_specifier", "struct_specifier", "union_specifier"],
        "function": ["function_definition"],
        "import":   ["preproc_include"],
        "call":     ["call_expression"],
        "variable": ["declaration", "field_declaration"],
    },
}

# ---------------------------------------------------------------------------
# Exported schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    """A named symbol (definition) found in the codebase."""
    kind: str           # "File", "Class", "Interface", "Function", "Variable", "Test"
    name: str           # short name, e.g. "login"
    qualified_name: str # globally unique ID, e.g. "src/auth.py::AuthService.login"
    file_path: str      # absolute path as string
    line_start: int     # 1-indexed
    line_end: int       # 1-indexed
    language: str


@dataclass
class EdgeInfo:
    """A directional relationship between two symbols."""
    kind: str    # "CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "CONTAINS"
    source: str  # qualified_name of source node
    target: str  # qualified_name of target node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _CallSite:
    """Unresolved call reference collected during pass 1."""
    caller_qname: str
    raw_callee: str      # as it appears in source, e.g. "obj.method", "foo"
    line: int            # 1-indexed
    file_rel_path: str   # caller's file rel path, for import_map lookup


@dataclass
class _FileContext:
    """Working state for a single file being processed."""
    file_path: Path
    rel_path: str        # relative to root_path, forward-slash separated
    language: str
    source: bytes
    all_nodes: list[NodeInfo] = field(default_factory=list)
    all_edges: list[EdgeInfo] = field(default_factory=list)
    call_sites: list[_CallSite] = field(default_factory=list)
    import_map: dict[str, str] = field(default_factory=dict)  # local_name -> module fragment


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class CodebaseExtractor:
    """
    Walks a codebase root, parses each source file with tree-sitter, and
    returns a unified graph of NodeInfo and EdgeInfo objects.

    Usage::

        extractor = CodebaseExtractor("/path/to/project")
        nodes, edges = extractor.extract()
    """

    def __init__(
        self,
        root_path: str | Path,
        *,
        max_workers: int = 4,
        skip_dirs: frozenset[str] | None = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.max_workers = max_workers
        self.skip_dirs = skip_dirs if skip_dirs is not None else SKIP_DIRS
        self.max_file_size = max_file_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> tuple[list[NodeInfo], list[EdgeInfo]]:
        """
        Two-pass extraction.

        Pass 1 (parallel): parse files, emit File/Class/Function/Variable nodes
                           and CONTAINS/INHERITS/IMPLEMENTS/IMPORTS edges.
        Pass 2 (serial):  resolve call-site records into CALLS edges.

        Returns (nodes, edges).
        """
        file_language_map = self._detect_languages()
        if not file_language_map:
            logger.warning("No source files found under %s", self.root_path)
            return [], []

        all_nodes: list[NodeInfo] = []
        all_edges: list[EdgeInfo] = []
        all_call_sites: list[_CallSite] = []
        all_import_maps: dict[str, dict[str, str]] = {}

        # Pass 1: parallel file processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_file, fp, lang): fp
                for fp, lang in file_language_map.items()
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    nodes, edges, call_sites, import_map = future.result()
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    all_call_sites.extend(call_sites)
                    rel = str(fp.relative_to(self.root_path)).replace(os.sep, "/")
                    all_import_maps[rel] = import_map
                except Exception:
                    logger.exception("Unexpected error processing file: %s", fp)

        # Build definition index for pass 2
        definition_index: dict[str, NodeInfo] = {n.qualified_name: n for n in all_nodes}

        # Pass 2: call resolution (serial)
        call_edges = self._resolve_calls(all_call_sites, definition_index, all_import_maps)
        all_edges.extend(call_edges)

        logger.info(
            "Extraction complete: %d nodes, %d edges from %d files",
            len(all_nodes), len(all_edges), len(file_language_map),
        )
        return all_nodes, all_edges

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _detect_languages(self) -> dict[Path, str]:
        """Walk root_path and return a map of {abs_path -> language}."""
        result: dict[Path, str] = {}
        for file_path in self.root_path.rglob("*"):
            if not file_path.is_file():
                continue
            # Skip directories on the skip list (check every part of the path)
            if any(part in self.skip_dirs for part in file_path.parts):
                continue
            # Skip oversized files
            try:
                if file_path.stat().st_size > self.max_file_size:
                    logger.debug("Skipping large file: %s", file_path)
                    continue
            except OSError:
                continue
            suffix = file_path.suffix.lower()
            if suffix not in EXTENSION_MAP:
                continue
            language = EXTENSION_MAP[suffix]
            if suffix == ".h":
                language = self._detect_h_dialect(file_path)
            result[file_path] = language
        return result

    def _detect_h_dialect(self, file_path: Path) -> str:
        """Return 'cpp' if the .h file looks like C++, otherwise 'c'."""
        try:
            with open(file_path, "rb") as fh:
                header = fh.read(4096)
            if any(sig in header for sig in _CPP_SIGNALS):
                return "cpp"
        except OSError:
            pass
        return "c"

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def _process_file(
        self, file_path: Path, language: str
    ) -> tuple[list[NodeInfo], list[EdgeInfo], list[_CallSite], dict[str, str]]:
        """Parse one file and return (nodes, edges, call_sites, import_map)."""
        try:
            from tree_sitter_language_pack import get_parser, LanguageNotFoundError
        except ImportError:
            logger.error("tree_sitter_language_pack is not installed")
            return [], [], [], {}

        try:
            parser = get_parser(language)
        except Exception as exc:
            logger.warning("Cannot get parser for language '%s': %s", language, exc)
            return [], [], [], {}

        try:
            source = file_path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return [], [], [], {}

        try:
            tree = parser.parse(source)
        except Exception as exc:
            logger.warning("Parse failed for %s (%s): %s", file_path, language, exc)
            return [], [], [], {}

        if tree.root_node.has_error:
            logger.debug("Parse errors in %s; continuing with partial AST", file_path)

        rel_path = str(file_path.relative_to(self.root_path)).replace(os.sep, "/")
        line_count = source.count(b"\n") + 1

        file_node = NodeInfo(
            kind="File",
            name=file_path.name,
            qualified_name=rel_path,
            file_path=str(file_path),
            line_start=1,
            line_end=line_count,
            language=language,
        )

        ctx = _FileContext(
            file_path=file_path,
            rel_path=rel_path,
            language=language,
            source=source,
        )
        ctx.all_nodes.append(file_node)

        # Walk the AST
        try:
            for _ in self._walk_ast(tree.root_node, ctx, [], rel_path):
                pass
        except Exception as exc:
            logger.warning("AST walk failed for %s: %s", file_path, exc)

        return ctx.all_nodes, ctx.all_edges, ctx.call_sites, ctx.import_map

    # ------------------------------------------------------------------
    # AST walker
    # ------------------------------------------------------------------

    def _walk_ast(
        self,
        node,
        ctx: _FileContext,
        scope_stack: list[str],
        parent_qname: str,
    ) -> Generator[None, None, None]:
        """
        Recursive AST walker. Yields nothing (uses ctx for side-effects).
        Maintains scope_stack for qualified name building.
        """
        lang = ctx.language
        lang_map = LANGUAGE_NODE_MAPS.get(lang, {})

        # Flatten all known structural categories for fast lookup
        class_types = set(lang_map.get("class", []))
        interface_types = set(lang_map.get("interface", []))
        function_types = set(lang_map.get("function", []))
        import_types = set(lang_map.get("import", []))
        impl_types = set(lang_map.get("impl", []))
        decorated_types = set(lang_map.get("decorated", []))

        ntype = node.type

        # --- Python decorated definitions ---
        if ntype in decorated_types:
            # Find the actual inner definition (skip decorator nodes)
            inner = None
            for child in node.children:
                if child.type in class_types or child.type in function_types:
                    inner = child
                    break
            if inner is not None:
                # Process the inner node but use the outer node's line range
                name = self._extract_name(inner, lang)
                if name:
                    kind = self._classify_node_kind(inner, lang, name, scope_stack, ctx.file_path)
                    qname = self._build_qualified_name(ctx.rel_path, scope_stack + [name])
                    node_info = NodeInfo(
                        kind=kind,
                        name=name,
                        qualified_name=qname,
                        file_path=str(ctx.file_path),
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=lang,
                    )
                    ctx.all_nodes.append(node_info)
                    ctx.all_edges.append(EdgeInfo(kind="CONTAINS", source=parent_qname, target=qname))

                    # Collect calls from this definition
                    if inner.type in function_types:
                        ctx.call_sites.extend(self._collect_call_sites(inner, lang, qname, ctx.rel_path))

                    # Recurse into inner's children with updated scope
                    for child in inner.children:
                        yield from self._walk_ast(child, ctx, scope_stack + [name], qname)
                    return  # Don't double-process children below
            # Fall through to children if no inner found
            for child in node.children:
                yield from self._walk_ast(child, ctx, scope_stack, parent_qname)
            return

        # --- Rust impl_item ---
        if ntype in impl_types:
            impl_name, impl_edges = self._rust_impl_scope_and_edges(node, ctx.rel_path)
            ctx.all_edges.extend(impl_edges)
            for child in node.children:
                yield from self._walk_ast(child, ctx, scope_stack + [impl_name], parent_qname)
            return

        # --- Class / struct / interface definitions ---
        if ntype in class_types or ntype in interface_types:
            name = self._extract_name(node, lang)
            if name:
                kind = self._classify_node_kind(node, lang, name, scope_stack, ctx.file_path)
                qname = self._build_qualified_name(ctx.rel_path, scope_stack + [name])
                node_info = NodeInfo(
                    kind=kind,
                    name=name,
                    qualified_name=qname,
                    file_path=str(ctx.file_path),
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=lang,
                )
                ctx.all_nodes.append(node_info)
                ctx.all_edges.append(EdgeInfo(kind="CONTAINS", source=parent_qname, target=qname))
                # Inheritance / implements edges
                ctx.all_edges.extend(self._extract_inheritance(node, lang, qname))
                # Recurse into children
                for child in node.children:
                    yield from self._walk_ast(child, ctx, scope_stack + [name], qname)
                return

        # --- Function / method definitions ---
        if ntype in function_types:
            name = self._extract_name(node, lang)
            if name:
                # For Go methods, push receiver type as outer scope
                extra_scope: list[str] = []
                if lang == "go" and ntype == "method_declaration":
                    receiver = self._go_method_receiver_type(node)
                    if receiver and (not scope_stack or scope_stack[-1] != receiver):
                        extra_scope = [receiver]

                full_scope = scope_stack + extra_scope
                kind = self._classify_node_kind(node, lang, name, full_scope, ctx.file_path)
                qname = self._build_qualified_name(ctx.rel_path, full_scope + [name])
                parent = self._build_qualified_name(ctx.rel_path, full_scope) if extra_scope else parent_qname

                node_info = NodeInfo(
                    kind=kind,
                    name=name,
                    qualified_name=qname,
                    file_path=str(ctx.file_path),
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=lang,
                )
                ctx.all_nodes.append(node_info)
                ctx.all_edges.append(EdgeInfo(kind="CONTAINS", source=parent, target=qname))

                # Collect call sites from the function body
                ctx.call_sites.extend(self._collect_call_sites(node, lang, qname, ctx.rel_path))

                # Recurse for nested definitions
                for child in node.children:
                    yield from self._walk_ast(child, ctx, full_scope + [name], qname)
                return

        # --- Import statements ---
        if ntype in import_types:
            edges, imp_map = self._extract_imports(node, ctx)
            ctx.all_edges.extend(edges)
            ctx.import_map.update(imp_map)
            return  # imports have no interesting children to walk

        # --- Default: recurse into children ---
        for child in node.children:
            yield from self._walk_ast(child, ctx, scope_stack, parent_qname)

    # ------------------------------------------------------------------
    # Name extraction
    # ------------------------------------------------------------------

    def _extract_name(self, node, language: str) -> str | None:
        """Extract the short identifier name from a definition node."""
        ntype = node.type

        # C / C++ function: function_definition → function_declarator → identifier
        if language in ("c", "cpp") and ntype == "function_definition":
            return self._c_function_name(node)

        # Go type_spec: name is the first identifier child
        if language == "go" and ntype == "type_spec":
            name_node = node.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8")
            return None

        # Standard field_name "name" covers most languages
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")

        # Fallback: first identifier or type_identifier child
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier",
                               "field_identifier"):
                return child.text.decode("utf-8")

        return None

    def _c_function_name(self, node) -> str | None:
        """Extract function name from a C/C++ function_definition node."""
        for child in node.children:
            if child.type == "function_declarator":
                for gc in child.children:
                    if gc.type == "identifier":
                        return gc.text.decode("utf-8")
                    # Pointer declarator: (*func_name)(args)
                    if gc.type == "pointer_declarator":
                        inner = gc.child_by_field_name("declarator")
                        if inner and inner.type == "identifier":
                            return inner.text.decode("utf-8")
        return None

    # ------------------------------------------------------------------
    # Qualified name builder
    # ------------------------------------------------------------------

    def _build_qualified_name(self, rel_path: str, scope_stack: list[str]) -> str:
        """Build a qualified name. Format: 'rel/path/file.ext::A.B.name'"""
        if not scope_stack:
            return rel_path
        return f"{rel_path}::{'.' .join(scope_stack)}"

    # ------------------------------------------------------------------
    # Kind classifier
    # ------------------------------------------------------------------

    def _classify_node_kind(
        self, node, language: str, name: str, scope_stack: list[str], file_path: Path
    ) -> str:
        ntype = node.type
        lang_map = LANGUAGE_NODE_MAPS.get(language, {})

        # Interface check
        if ntype in lang_map.get("interface", []):
            return "Interface"

        # Class / struct / enum check
        if ntype in lang_map.get("class", []):
            return "Class"

        # Test detection (before generic Function check)
        if ntype in lang_map.get("function", []):
            if self._is_test(name, file_path, language):
                return "Test"
            return "Function"

        # Variable
        if ntype in lang_map.get("variable", []):
            return "Variable"

        return "Function"  # safe fallback

    def _is_test(self, name: str, file_path: Path, language: str) -> bool:
        """Return True if this definition looks like a test."""
        parts = {p.lower() for p in file_path.parts}
        test_in_path = bool(parts & {"test", "tests", "spec", "specs", "__tests__"})

        if language == "python":
            return name.startswith("test_") or name.startswith("Test") or test_in_path
        if language in ("javascript", "typescript", "tsx"):
            return name.startswith("test") or name.startswith("spec") or test_in_path
        if language == "java":
            return name.startswith("test") or test_in_path
        return test_in_path

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_imports(
        self, node, ctx: _FileContext
    ) -> tuple[list[EdgeInfo], dict[str, str]]:
        """Extract IMPORTS edges and build import_map from an import node."""
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}
        lang = ctx.language
        ntype = node.type

        try:
            if lang == "python":
                edges, imp_map = self._py_imports(node, ctx.rel_path)
            elif lang in ("javascript", "typescript", "tsx"):
                edges, imp_map = self._js_imports(node, ctx.rel_path)
            elif lang == "go":
                edges, imp_map = self._go_imports(node, ctx.rel_path)
            elif lang == "java":
                edges, imp_map = self._java_imports(node, ctx.rel_path)
            elif lang == "rust":
                edges, imp_map = self._rust_imports(node, ctx.rel_path)
            elif lang in ("c", "cpp"):
                edges, imp_map = self._c_imports(node, ctx.rel_path)
        except Exception as exc:
            logger.debug("Import extraction failed in %s (%s): %s", ctx.rel_path, ntype, exc)

        return edges, imp_map

    def _py_imports(self, node, rel_path: str) -> tuple[list[EdgeInfo], dict[str, str]]:
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}
        ntype = node.type

        if ntype == "import_statement":
            # import os, import os.path, import os as operating_system
            for child in node.children:
                if child.type == "dotted_name":
                    mod = child.text.decode("utf-8")
                    alias = mod.split(".")[0]
                    imp_map[alias] = mod
                    imp_map[mod] = mod
                    edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=mod))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    alias_node = child.child_by_field_name("alias")
                    if name_node and alias_node:
                        mod = name_node.text.decode("utf-8")
                        alias = alias_node.text.decode("utf-8")
                        imp_map[alias] = mod
                        edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=mod))

        elif ntype == "import_from_statement":
            # from pathlib import Path, PurePath
            mod_node = node.child_by_field_name("module_name")
            if mod_node is None:
                # try dotted_name child
                for c in node.children:
                    if c.type == "dotted_name":
                        mod_node = c
                        break
            mod = mod_node.text.decode("utf-8") if mod_node else ""
            edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=mod))
            for child in node.children:
                if child.type == "dotted_name" and child is not mod_node:
                    sym = child.text.decode("utf-8")
                    imp_map[sym] = f"{mod}.{sym}" if mod else sym
                elif child.type == "aliased_import":
                    name_n = child.child_by_field_name("name")
                    alias_n = child.child_by_field_name("alias")
                    if name_n and alias_n:
                        sym = name_n.text.decode("utf-8")
                        alias = alias_n.text.decode("utf-8")
                        imp_map[alias] = f"{mod}.{sym}" if mod else sym
                elif child.type == "wildcard_import":
                    imp_map[f"{mod}.*"] = mod

        return edges, imp_map

    def _js_imports(self, node, rel_path: str) -> tuple[list[EdgeInfo], dict[str, str]]:
        """Handle ES6 import_statement nodes."""
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}

        # Find the source string
        source_node = node.child_by_field_name("source")
        if source_node is None:
            for child in node.children:
                if child.type == "string":
                    source_node = child
                    break
        if source_node is None:
            return edges, imp_map

        src = source_node.text.decode("utf-8").strip("'\"")
        edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=src))

        # Parse import clause
        for child in node.children:
            if child.type == "import_clause":
                for gc in child.children:
                    if gc.type == "identifier":
                        # default import
                        imp_map[gc.text.decode("utf-8")] = f"{src}::default"
                    elif gc.type == "named_imports":
                        for spec in gc.children:
                            if spec.type == "import_specifier":
                                name_n = spec.child_by_field_name("name")
                                alias_n = spec.child_by_field_name("alias")
                                if name_n:
                                    orig = name_n.text.decode("utf-8")
                                    alias = alias_n.text.decode("utf-8") if alias_n else orig
                                    imp_map[alias] = f"{src}::{orig}"
                    elif gc.type == "namespace_import":
                        for ggc in gc.children:
                            if ggc.type == "identifier":
                                imp_map[ggc.text.decode("utf-8")] = src

        return edges, imp_map

    def _go_imports(self, node, rel_path: str) -> tuple[list[EdgeInfo], dict[str, str]]:
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}

        def process_spec(spec):
            path_node = spec.child_by_field_name("path")
            if path_node is None:
                for c in spec.children:
                    if c.type in ("interpreted_string_literal", "raw_string_literal"):
                        path_node = c
                        break
            if path_node is None:
                return
            pkg_path = path_node.text.decode("utf-8").strip('"').strip("`")
            alias_node = spec.child_by_field_name("name")
            if alias_node is None:
                for c in spec.children:
                    if c.type == "identifier":
                        alias_node = c
                        break
            alias = alias_node.text.decode("utf-8") if alias_node else pkg_path.split("/")[-1]
            if alias == ".":
                alias = pkg_path.split("/")[-1]
            imp_map[alias] = pkg_path
            edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=pkg_path))

        if node.type == "import_spec":
            process_spec(node)
        else:
            for child in node.children:
                if child.type == "import_spec":
                    process_spec(child)
                elif child.type == "import_spec_list":
                    for gc in child.children:
                        if gc.type == "import_spec":
                            process_spec(gc)

        return edges, imp_map

    def _java_imports(self, node, rel_path: str) -> tuple[list[EdgeInfo], dict[str, str]]:
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}

        # import_declaration: import fully.qualified.ClassName;
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                full = child.text.decode("utf-8")
                simple = full.split(".")[-1]
                if simple == "*":
                    imp_map[f"{full}"] = full
                else:
                    imp_map[simple] = full
                edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=full))

        return edges, imp_map

    def _rust_imports(self, node, rel_path: str) -> tuple[list[EdgeInfo], dict[str, str]]:
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}

        def process_use_tree(tree_node):
            text = tree_node.text.decode("utf-8")
            # e.g. "std::collections::HashMap" or "crate::utils::{foo, bar}"
            if "::" in text:
                parts = text.split("::")
                last = parts[-1].strip("{}")
                for sym in last.split(","):
                    sym = sym.strip()
                    if sym and sym != "*":
                        full = "::".join(parts[:-1]) + "::" + sym
                        imp_map[sym] = full
                        edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=full))
            else:
                imp_map[text] = text
                edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=text))

        for child in node.children:
            if child.type in ("scoped_identifier", "identifier", "use_wildcard",
                               "use_list", "use_as_clause", "scoped_use_list"):
                try:
                    process_use_tree(child)
                except Exception:
                    pass

        return edges, imp_map

    def _c_imports(self, node, rel_path: str) -> tuple[list[EdgeInfo], dict[str, str]]:
        edges: list[EdgeInfo] = []
        imp_map: dict[str, str] = {}

        for child in node.children:
            if child.type in ("string_literal", "system_lib_string"):
                inc = child.text.decode("utf-8").strip('"<>')
                base = inc.split(".")[0]
                imp_map[base] = inc
                edges.append(EdgeInfo(kind="IMPORTS", source=rel_path, target=inc))

        return edges, imp_map

    # ------------------------------------------------------------------
    # Inheritance extraction
    # ------------------------------------------------------------------

    def _extract_inheritance(
        self, node, language: str, class_qname: str
    ) -> list[EdgeInfo]:
        """Extract INHERITS and IMPLEMENTS edges from a class definition node."""
        edges: list[EdgeInfo] = []
        ntype = node.type

        try:
            if language == "python":
                # class_definition: superclasses field is an argument_list
                super_node = node.child_by_field_name("superclasses")
                if super_node:
                    for child in super_node.children:
                        if child.type in ("identifier", "attribute"):
                            parent = child.text.decode("utf-8")
                            edges.append(EdgeInfo(kind="INHERITS", source=class_qname, target=parent))

            elif language in ("javascript", "typescript", "tsx"):
                for child in node.children:
                    if child.type == "class_heritage":
                        # extends Foo
                        for gc in child.children:
                            if gc.type in ("identifier", "member_expression"):
                                edges.append(EdgeInfo(kind="INHERITS", source=class_qname, target=gc.text.decode("utf-8")))
                    elif child.type == "implements_clause":
                        for gc in child.children:
                            if gc.type in ("type_identifier", "generic_type", "identifier"):
                                edges.append(EdgeInfo(kind="IMPLEMENTS", source=class_qname, target=gc.text.decode("utf-8")))

            elif language == "java":
                super_node = node.child_by_field_name("superclass")
                if super_node:
                    edges.append(EdgeInfo(kind="INHERITS", source=class_qname, target=super_node.text.decode("utf-8")))
                interfaces_node = node.child_by_field_name("super_interfaces")
                if interfaces_node:
                    for child in interfaces_node.children:
                        if child.type in ("type_identifier", "generic_type"):
                            edges.append(EdgeInfo(kind="IMPLEMENTS", source=class_qname, target=child.text.decode("utf-8")))
                # interface extends
                extends_node = node.child_by_field_name("extends_interfaces")
                if extends_node:
                    for child in extends_node.children:
                        if child.type in ("type_identifier", "generic_type"):
                            edges.append(EdgeInfo(kind="INHERITS", source=class_qname, target=child.text.decode("utf-8")))

            elif language == "cpp":
                for child in node.children:
                    if child.type == "base_class_clause":
                        for gc in child.children:
                            if gc.type in ("type_identifier", "qualified_identifier"):
                                edges.append(EdgeInfo(kind="INHERITS", source=class_qname, target=gc.text.decode("utf-8")))

        except Exception as exc:
            logger.debug("Inheritance extraction failed for %s: %s", class_qname, exc)

        return edges

    # ------------------------------------------------------------------
    # Call site collection
    # ------------------------------------------------------------------

    def _collect_call_sites(
        self, func_node, language: str, caller_qname: str, rel_path: str
    ) -> list[_CallSite]:
        """Walk a function body and collect raw call references."""
        call_sites: list[_CallSite] = []
        lang_map = LANGUAGE_NODE_MAPS.get(language, {})
        call_types = set(lang_map.get("call", []))
        # Don't descend into nested function/class definitions
        struct_types = (
            set(lang_map.get("function", [])) |
            set(lang_map.get("class", [])) |
            set(lang_map.get("decorated", []))
        )

        def walk(node, is_root: bool = False):
            if not is_root and node.type in struct_types:
                return  # skip nested defs
            if node.type in call_types:
                raw = self._extract_callee_text(node, language)
                if raw:
                    call_sites.append(_CallSite(
                        caller_qname=caller_qname,
                        raw_callee=raw,
                        line=node.start_point[0] + 1,
                        file_rel_path=rel_path,
                    ))
            for child in node.children:
                walk(child)

        walk(func_node, is_root=True)
        return call_sites

    def _extract_callee_text(self, call_node, language: str) -> str | None:
        """Extract the callee name from a call node."""
        try:
            # Python: call -> function field
            fn_node = call_node.child_by_field_name("function")
            if fn_node is not None:
                return fn_node.text.decode("utf-8")

            # JS/TS/Go/Rust: call_expression -> function field
            fn_node = call_node.child_by_field_name("function")
            if fn_node is not None:
                return fn_node.text.decode("utf-8")

            # Rust method_call_expression: receiver.method(args)
            if language == "rust" and call_node.type == "method_call_expression":
                method = call_node.child_by_field_name("method")
                if method:
                    return method.text.decode("utf-8")

            # Java method_invocation
            if language == "java" and call_node.type == "method_invocation":
                name_node = call_node.child_by_field_name("name")
                if name_node:
                    obj_node = call_node.child_by_field_name("object")
                    if obj_node:
                        return f"{obj_node.text.decode('utf-8')}.{name_node.text.decode('utf-8')}"
                    return name_node.text.decode("utf-8")

            # Fallback: first identifier child
            for child in call_node.children:
                if child.type in ("identifier", "member_expression", "attribute",
                                   "field_expression", "scoped_identifier"):
                    return child.text.decode("utf-8")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Go and Rust helpers
    # ------------------------------------------------------------------

    def _go_method_receiver_type(self, node) -> str | None:
        """Extract receiver struct name from a Go method_declaration node."""
        for child in node.children:
            if child.type == "parameter_list":
                for param in child.children:
                    if param.type == "parameter_declaration":
                        for p in param.children:
                            if p.type == "type_identifier":
                                return p.text.decode("utf-8")
                            if p.type == "pointer_type":
                                # Try named field "type" first, then any type_identifier child
                                inner = p.child_by_field_name("type")
                                if inner:
                                    return inner.text.decode("utf-8")
                                for gc in p.children:
                                    if gc.type == "type_identifier":
                                        return gc.text.decode("utf-8")
                break
        return None

    def _rust_impl_scope_and_edges(
        self, node, rel_path: str
    ) -> tuple[str, list[EdgeInfo]]:
        """
        Extract the struct/type name from a Rust impl_item and any INHERITS edge
        for `impl Trait for Struct` patterns.
        """
        type_ids = [c for c in node.children if c.type == "type_identifier"]
        has_for = any(c.type == "for" for c in node.children)

        if has_for and len(type_ids) >= 2:
            trait_name = type_ids[0].text.decode("utf-8")
            struct_name = type_ids[1].text.decode("utf-8")
            struct_qname = f"{rel_path}::{struct_name}"
            edge = EdgeInfo(kind="INHERITS", source=struct_qname, target=trait_name)
            return struct_name, [edge]
        elif type_ids:
            return type_ids[0].text.decode("utf-8"), []
        return "<unknown_impl>", []

    # ------------------------------------------------------------------
    # Pass 2: call resolution
    # ------------------------------------------------------------------

    def _resolve_calls(
        self,
        call_sites: list[_CallSite],
        definition_index: dict[str, NodeInfo],
        import_maps: dict[str, dict[str, str]],
    ) -> list[EdgeInfo]:
        """
        Resolve raw call sites to CALLS edges using the definition index
        and per-file import maps.
        """
        edges: list[EdgeInfo] = []

        # Build a name → list[qname] lookup for fuzzy matching
        name_to_qnames: dict[str, list[str]] = {}
        for qname in definition_index:
            # The short name is the last segment after '.' in the qualified name
            short = qname.split("::")[-1].split(".")[-1] if "::" in qname else qname.split("/")[-1]
            name_to_qnames.setdefault(short, []).append(qname)

        for site in call_sites:
            raw = site.raw_callee.strip()
            if not raw:
                continue

            resolved = self._resolve_single_call(
                raw, site.caller_qname, site.file_rel_path,
                definition_index, import_maps, name_to_qnames
            )
            for target in resolved:
                edges.append(EdgeInfo(kind="CALLS", source=site.caller_qname, target=target))

        return edges

    def _resolve_single_call(
        self,
        raw: str,
        caller_qname: str,
        file_rel_path: str,
        definition_index: dict[str, NodeInfo],
        import_maps: dict[str, dict[str, str]],
        name_to_qnames: dict[str, list[str]],
    ) -> list[str]:
        """Resolve a single raw callee string to one or more qualified names."""
        imp_map = import_maps.get(file_rel_path, {})

        # Strip self/this prefix
        if raw.startswith("self.") or raw.startswith("this."):
            raw = raw.split(".", 1)[1]

        # 1. Exact match in definition index
        if raw in definition_index:
            return [raw]

        # 2. Handle dotted/scoped calls: "obj.method" or "pkg::Func"
        separator = None
        if "." in raw:
            separator = "."
        elif "::" in raw:
            separator = "::"

        if separator:
            parts = raw.split(separator, 1)
            prefix, method = parts[0], parts[1]

            # Resolve prefix via import_map
            resolved_prefix = imp_map.get(prefix)
            if resolved_prefix:
                # Try "resolved_prefix::method" or "resolved_prefix.method"
                candidates = [
                    f"{resolved_prefix}::{method}",
                    f"{resolved_prefix}.{method}",
                ]
                for cand in candidates:
                    if cand in definition_index:
                        return [cand]

            # Try fuzzy: find any qname ending with ".{method}" or "::{method}"
            method_short = method.split(".")[-1].split("::")[-1]
            matches = [
                qn for qn in name_to_qnames.get(method_short, [])
                if qn != caller_qname
            ]
            if matches:
                logger.debug("Fuzzy resolved '%s' -> %s", raw, matches)
                return matches[:3]  # cap at 3 candidates to avoid noise

            return []

        # 3. Simple name lookup
        # Check import_map first
        resolved_mod = imp_map.get(raw)
        if resolved_mod:
            # The call may be to a module-level function exported from the module
            candidates = [qn for qn in name_to_qnames.get(raw, [])
                          if qn.startswith(resolved_mod.replace(".", "/").replace("::", "/"))]
            if candidates:
                return candidates

        # 4. Search definition_index by short name
        matches = name_to_qnames.get(raw, [])
        if matches:
            # Prefer definitions in the same file
            same_file = [m for m in matches if m.startswith(file_rel_path)]
            if same_file:
                return same_file
            if len(matches) == 1:
                return matches
            # Multiple candidates: return all (ambiguous)
            logger.debug("Ambiguous call '%s' in %s -> %s", raw, file_rel_path, matches)
            return matches[:5]

        logger.debug("Unresolved call '%s' in %s", raw, file_rel_path)
        return []
