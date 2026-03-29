"""Tests for berkelium_cli.extractor.CodebaseExtractor."""
from __future__ import annotations

import pytest
from pathlib import Path

from berkelium_cli.extractor import (
    CodebaseExtractor,
    EdgeInfo,
    NodeInfo,
    EXTENSION_MAP,
    SKIP_DIRS,
    MAX_FILE_SIZE_BYTES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(tmp_path: Path, files: dict[str, str]) -> tuple[list[NodeInfo], list[EdgeInfo]]:
    """Write files into tmp_path and run extraction."""
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return CodebaseExtractor(tmp_path).extract()


def _qnames(nodes: list[NodeInfo]) -> set[str]:
    return {n.qualified_name for n in nodes}


def _kinds(nodes: list[NodeInfo]) -> dict[str, str]:
    return {n.qualified_name: n.kind for n in nodes}


def _edges_of(edges: list[EdgeInfo], kind: str) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in edges if e.kind == kind}


# ---------------------------------------------------------------------------
# Extension / language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_known_extensions_in_map(self):
        assert EXTENSION_MAP[".py"] == "python"
        assert EXTENSION_MAP[".ts"] == "typescript"
        assert EXTENSION_MAP[".tsx"] == "tsx"
        assert EXTENSION_MAP[".go"] == "go"
        assert EXTENSION_MAP[".rs"] == "rust"
        assert EXTENSION_MAP[".java"] == "java"
        assert EXTENSION_MAP[".cpp"] == "cpp"
        assert EXTENSION_MAP[".h"] == "c"   # default; sniffed at runtime

    def test_skip_dirs_contains_common_dirs(self):
        assert "node_modules" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert ".venv" in SKIP_DIRS
        assert ".git" in SKIP_DIRS

    def test_skips_node_modules(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("function foo() {}")
        nodes, _ = _extract(tmp_path, {})
        assert not any("node_modules" in n.qualified_name for n in nodes)

    def test_skips_oversized_file(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_bytes(b"x = 1\n" * ((MAX_FILE_SIZE_BYTES // 6) + 1))
        nodes, _ = CodebaseExtractor(tmp_path).extract()
        assert not any(n.qualified_name == "big.py" for n in nodes)

    def test_h_file_classified_as_cpp_when_cpp_content(self, tmp_path):
        extractor = CodebaseExtractor(tmp_path)
        header = tmp_path / "mylib.h"
        header.write_text("class Foo { public: void bar(); };")
        assert extractor._detect_h_dialect(header) == "cpp"

    def test_h_file_classified_as_c_when_plain_c(self, tmp_path):
        extractor = CodebaseExtractor(tmp_path)
        header = tmp_path / "mylib.h"
        header.write_text("typedef struct { int x; } Point;")
        assert extractor._detect_h_dialect(header) == "c"


# ---------------------------------------------------------------------------
# Qualified name builder
# ---------------------------------------------------------------------------

class TestQualifiedName:
    def test_file_level(self):
        e = CodebaseExtractor("/tmp")
        assert e._build_qualified_name("src/auth.py", []) == "src/auth.py"

    def test_top_level_class(self):
        e = CodebaseExtractor("/tmp")
        assert e._build_qualified_name("src/auth.py", ["AuthService"]) == "src/auth.py::AuthService"

    def test_nested_method(self):
        e = CodebaseExtractor("/tmp")
        assert e._build_qualified_name("src/auth.py", ["AuthService", "login"]) == "src/auth.py::AuthService.login"


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------

class TestPythonExtraction:
    SIMPLE_PY = """\
class AuthService:
    def login(self, user: str) -> bool:
        return True

def standalone():
    pass
"""

    def test_file_node_present(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"auth.py": self.SIMPLE_PY})
        assert "auth.py" in _qnames(nodes)
        file_node = next(n for n in nodes if n.qualified_name == "auth.py")
        assert file_node.kind == "File"
        assert file_node.language == "python"

    def test_class_extracted(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"auth.py": self.SIMPLE_PY})
        assert "auth.py::AuthService" in _qnames(nodes)
        assert _kinds(nodes)["auth.py::AuthService"] == "Class"

    def test_method_extracted(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"auth.py": self.SIMPLE_PY})
        assert "auth.py::AuthService.login" in _qnames(nodes)
        assert _kinds(nodes)["auth.py::AuthService.login"] == "Function"

    def test_standalone_function(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"auth.py": self.SIMPLE_PY})
        assert "auth.py::standalone" in _qnames(nodes)

    def test_contains_edges(self, tmp_path):
        _, edges = _extract(tmp_path, {"auth.py": self.SIMPLE_PY})
        contains = _edges_of(edges, "CONTAINS")
        assert ("auth.py::AuthService", "auth.py::AuthService.login") in contains
        assert ("auth.py", "auth.py::AuthService") in contains

    def test_line_numbers(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"auth.py": self.SIMPLE_PY})
        cls = next(n for n in nodes if n.qualified_name == "auth.py::AuthService")
        assert cls.line_start == 1

    def test_decorated_function(self, tmp_path):
        src = """\
import functools

def my_decorator(fn):
    return fn

@my_decorator
def decorated_fn():
    pass
"""
        nodes, _ = _extract(tmp_path, {"dec.py": src})
        assert "dec.py::decorated_fn" in _qnames(nodes)

    def test_decorated_class(self, tmp_path):
        src = """\
def decorator(cls):
    return cls

@decorator
class MyClass:
    pass
"""
        nodes, _ = _extract(tmp_path, {"dec.py": src})
        assert "dec.py::MyClass" in _qnames(nodes)
        assert _kinds(nodes)["dec.py::MyClass"] == "Class"

    def test_inheritance_edge(self, tmp_path):
        src = "class Child(Base):\n    pass\n"
        _, edges = _extract(tmp_path, {"model.py": src})
        inherits = _edges_of(edges, "INHERITS")
        assert ("model.py::Child", "Base") in inherits

    def test_import_edge(self, tmp_path):
        src = "from pathlib import Path\nimport os\n\ndef fn(): pass\n"
        _, edges = _extract(tmp_path, {"util.py": src})
        import_targets = {t for _, t in _edges_of(edges, "IMPORTS")}
        assert "pathlib" in import_targets
        assert "os" in import_targets


# ---------------------------------------------------------------------------
# Test detection
# ---------------------------------------------------------------------------

class TestTestDetection:
    def test_python_test_function(self, tmp_path):
        src = "def test_login():\n    assert True\n"
        nodes, _ = _extract(tmp_path, {"test_auth.py": src})
        fn = next((n for n in nodes if n.name == "test_login"), None)
        assert fn is not None
        assert fn.kind == "Test"

    def test_python_non_test_function(self, tmp_path):
        src = "def login():\n    pass\n"
        nodes, _ = _extract(tmp_path, {"auth.py": src})
        fn = next((n for n in nodes if n.name == "login"), None)
        assert fn is not None
        assert fn.kind == "Function"

    def test_function_in_test_directory(self, tmp_path):
        src = "def my_helper():\n    pass\n"
        nodes, _ = _extract(tmp_path, {"tests/helpers.py": src})
        fn = next((n for n in nodes if n.name == "my_helper"), None)
        assert fn is not None
        assert fn.kind == "Test"


# ---------------------------------------------------------------------------
# JavaScript / TypeScript extraction
# ---------------------------------------------------------------------------

class TestJavaScriptExtraction:
    def test_class_and_method(self, tmp_path):
        src = """\
class Animal {
    constructor(name) {
        this.name = name;
    }
    speak() {
        console.log(this.name);
    }
}
"""
        nodes, _ = _extract(tmp_path, {"animal.js": src})
        qn = _qnames(nodes)
        assert "animal.js::Animal" in qn
        assert _kinds(nodes)["animal.js::Animal"] == "Class"

    def test_function_declaration(self, tmp_path):
        src = "function greet(name) { return 'hello ' + name; }\n"
        nodes, _ = _extract(tmp_path, {"greet.js": src})
        assert "greet.js::greet" in _qnames(nodes)

    def test_class_inheritance(self, tmp_path):
        src = "class Dog extends Animal { bark() {} }\n"
        _, edges = _extract(tmp_path, {"dog.js": src})
        inherits = _edges_of(edges, "INHERITS")
        assert ("dog.js::Dog", "Animal") in inherits

    def test_import_edge(self, tmp_path):
        src = "import { foo } from './utils';\nfunction main() { foo(); }\n"
        _, edges = _extract(tmp_path, {"main.js": src})
        import_targets = {t for _, t in _edges_of(edges, "IMPORTS")}
        assert "./utils" in import_targets

    def test_typescript_interface(self, tmp_path):
        src = "interface IAnimal { name: string; speak(): void; }\n"
        nodes, _ = _extract(tmp_path, {"animal.ts": src})
        iface = next((n for n in nodes if n.name == "IAnimal"), None)
        assert iface is not None
        assert iface.kind == "Interface"


# ---------------------------------------------------------------------------
# Call resolution (cross-file)
# ---------------------------------------------------------------------------

class TestCallResolution:
    def test_cross_file_call(self, tmp_path):
        """Function in main.py calls helper() defined in utils.py."""
        utils_src = "def helper():\n    return 42\n"
        main_src = "from utils import helper\n\ndef main():\n    helper()\n"
        nodes, edges = _extract(tmp_path, {
            "utils.py": utils_src,
            "main.py": main_src,
        })
        calls = _edges_of(edges, "CALLS")
        # main.py::main should call utils.py::helper
        assert any(
            src == "main.py::main" and "helper" in tgt
            for src, tgt in calls
        )

    def test_same_file_call(self, tmp_path):
        src = """\
def helper():
    return 1

def main():
    helper()
"""
        _, edges = _extract(tmp_path, {"app.py": src})
        calls = _edges_of(edges, "CALLS")
        assert ("app.py::main", "app.py::helper") in calls

    def test_method_self_call(self, tmp_path):
        src = """\
class Service:
    def _internal(self):
        pass

    def run(self):
        self._internal()
"""
        _, edges = _extract(tmp_path, {"svc.py": src})
        calls = _edges_of(edges, "CALLS")
        assert any(
            src == "svc.py::Service.run" and "_internal" in tgt
            for src, tgt in calls
        )


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_malformed_python_file(self, tmp_path):
        """Partially invalid Python should not raise; extractor continues."""
        (tmp_path / "bad.py").write_bytes(b"def foo():\n    \xff\xfe invalid bytes here\n")
        # Should not raise
        nodes, edges = CodebaseExtractor(tmp_path).extract()
        # At minimum a File node should exist (or extraction silently skipped)
        # The key assertion: no exception was raised
        assert isinstance(nodes, list)
        assert isinstance(edges, list)

    def test_binary_file_skipped(self, tmp_path):
        """A .py file with all binary content should not crash the extractor."""
        (tmp_path / "binary.py").write_bytes(bytes(range(256)) * 100)
        nodes, edges = CodebaseExtractor(tmp_path).extract()
        assert isinstance(nodes, list)

    def test_empty_directory(self, tmp_path):
        nodes, edges = CodebaseExtractor(tmp_path).extract()
        assert nodes == []
        assert edges == []

    def test_empty_source_file(self, tmp_path):
        nodes, edges = _extract(tmp_path, {"empty.py": ""})
        file_node = next((n for n in nodes if n.qualified_name == "empty.py"), None)
        assert file_node is not None
        assert file_node.kind == "File"


# ---------------------------------------------------------------------------
# Go extraction (smoke test)
# ---------------------------------------------------------------------------

class TestGoExtraction:
    GO_SRC = """\
package main

import "fmt"

type Dog struct {
    Name string
}

func (d *Dog) Sound() string {
    return "Woof"
}

func main() {
    d := Dog{Name: "Rex"}
    fmt.Println(d.Sound())
}
"""

    def test_struct_extracted(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"main.go": self.GO_SRC})
        assert any(n.name == "Dog" for n in nodes)
        dog = next(n for n in nodes if n.name == "Dog")
        assert dog.kind == "Class"

    def test_method_under_receiver(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"main.go": self.GO_SRC})
        qnames = _qnames(nodes)
        assert "main.go::Dog.Sound" in qnames

    def test_import_edge(self, tmp_path):
        _, edges = _extract(tmp_path, {"main.go": self.GO_SRC})
        import_targets = {t for _, t in _edges_of(edges, "IMPORTS")}
        assert "fmt" in import_targets


# ---------------------------------------------------------------------------
# Rust extraction (smoke test)
# ---------------------------------------------------------------------------

class TestRustExtraction:
    RUST_SRC = """\
use std::fmt;

struct Dog {
    name: String,
}

impl Dog {
    fn new(name: &str) -> Self {
        Dog { name: name.to_string() }
    }

    fn sound(&self) -> &str {
        "Woof"
    }
}
"""

    def test_struct_extracted(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"lib.rs": self.RUST_SRC})
        assert any(n.name == "Dog" and n.kind == "Class" for n in nodes)

    def test_impl_methods_scoped_to_struct(self, tmp_path):
        nodes, _ = _extract(tmp_path, {"lib.rs": self.RUST_SRC})
        qnames = _qnames(nodes)
        assert "lib.rs::Dog.new" in qnames or any("Dog" in qn and "new" in qn for qn in qnames)

    def test_use_import_edge(self, tmp_path):
        _, edges = _extract(tmp_path, {"lib.rs": self.RUST_SRC})
        import_targets = {t for _, t in _edges_of(edges, "IMPORTS")}
        assert any("fmt" in t for t in import_targets)
