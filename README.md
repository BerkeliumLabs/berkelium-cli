# Berkelium CLI
<p align="center">
<a href="https://pypi.org/project/berkelium/">
    <img src="https://img.shields.io/pypi/v/berkelium?style=flat-square&logo=pypi" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/berkelium/">
    <img src="https://img.shields.io/pypi/dm/berkelium?style=flat-square" alt="PyPI Downloads">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/BerkeliumLabs/berkelium-cli?style=flat-square" alt="License">
  </a>
  <img src="https://img.shields.io/badge/MCP-Compatible-green?style=flat-square" alt="MCP Compatible">
</p>

Berkelium CLI is a **Code Graph Management** tool and **Model Context Protocol (MCP)** server. It uses `tree-sitter` to parse your codebase into a structured graph stored in SQLite, enabling high-fidelity impact analysis and surgical context retrieval for AI assistants.

## Quick Start

### Installation

```bash
# Install
pip install berkelium

# Run the TUI
berkelium-cli
```

### Development Setup
```bash
git clone https://github.com/BerkeliumLabs/berkelium-cli
cd berkelium-cli
uv sync
```

## TUI Features
The Berkelium TUI (`berkelium-cli`) provides a terminal interface for:
- **Build/Update Graph**: Index supported languages (Python, JS/TS, Go, Java, Rust, C/C++).
- **Incremental Sync**: Uses git-diff to update the graph in milliseconds.
- **Exploration**: Visualize symbols and relationships directly in your terminal.

## MCP Server (for AI Assistants)
Connect Berkelium to Claude, Cursor, or any MCP-compatible client to give your AI "graph-vision" over your code.

### Configuration (Claude)
Add this to your `settings.json`:
```json
{
  "mcpServers": {
    "berkelium": {
      "command": "berkelium-mcp",
    }
  }
}
```

### Available Tools
- `query_search_codebase`: Executes read-only Cypher queries directly against the code graph for advanced structural search.

### Available Prompts
- `review_my_pr`: A guided workflow that syncs the graph, identifies changes via git, and suggests targeted tests based on impact analysis.

## Usage Examples (AI-Assisted Development)
Berkelium is most powerful when used through an AI assistant (like Claude or Cursor). Here are some real-world scenarios where it excels:

### 1. Architectural Exploration
**Prompt:** *"I want to add a new database provider. Show me all classes that inherit from `BaseProvider` to see the existing pattern."*
* **AI Action:** Uses `query_search_codebase` with a Cypher query like:
  ```cypher
  MATCH (child)-[:INHERITS]->(parent) WHERE parent.name = 'BaseProvider' RETURN child.name AS name, child.file_rel_path AS file
  ```

### 2. Tracing Complex Call Chains
**Prompt:** *"Trace the execution flow starting from `IncrementalSync.sync()`. What internal methods does it reach?"*
* **AI Action:** Uses `query_search_codebase` to traverse the `[:CALLS]` edges.

### 3. Identifying Dead Code
**Prompt:** *"Are there any functions in the `utils/` directory that are never called by any other part of the project?"*
* **AI Action:** Uses `query_search_codebase` to find nodes with zero incoming `[:CALLS]` edges.

## Architecture
- **Extractor (`extractor.py`)**: Language-agnostic extraction using `tree-sitter`.
- **Store (`store.py`)**: Persistence layer using `GraphQLite` on SQLite.
- **Sync (`sync.py`)**: Git-based incremental synchronization logic.
- **Retriever (`retriever.py`)**: Graph traversal algorithms for impact and context.

## License
MIT License - see [LICENSE](LICENSE).
