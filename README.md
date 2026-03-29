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
- `build_or_update_graph`: Performs a full extraction or an incremental git-diff sync.
- `get_impact_radius`: Analyzes functional blast radius (upstream callers and downstream dependencies).
- `get_structural_context`: Returns Markdown-formatted context optimized for LLM injection.
- `get_file_symbols`: Lists all symbols (functions, classes, etc.) defined in a specific file.
- `search_symbols`: Locates symbols by name fragment across the entire codebase.
- `query_graph`: Executes read-only Cypher queries directly against the code graph.

### Available Prompts
- `review_my_pr`: A guided workflow that syncs the graph, identifies changes via git, and suggests targeted tests based on impact analysis.

## Architecture
- **Extractor (`extractor.py`)**: Language-agnostic extraction using `tree-sitter`.
- **Store (`store.py`)**: Persistence layer using `GraphQLite` on SQLite.
- **Sync (`sync.py`)**: Git-based incremental synchronization logic.
- **Retriever (`retriever.py`)**: Graph traversal algorithms for impact and context.

## License
MIT License - see [LICENSE](LICENSE).
