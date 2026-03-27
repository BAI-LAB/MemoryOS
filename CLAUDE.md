# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemoryOS is a memory operating system for personalized AI agents, implementing a hierarchical memory architecture inspired by OS memory management. It provides short-term, mid-term, and long-term memory layers with automated updating, retrieval, and generation. Published at EMNLP 2025 (paper: arxiv.org/abs/2506.06326).

## Repository Structure

The repo contains **four parallel implementations** of the same core architecture, each in its own directory:

- **memoryos-pypi/** — The canonical PyPI-installable package. Uses FAISS for vector search. This is the primary implementation.
- **memoryos-chromadb/** — Alternative backend using ChromaDB instead of FAISS for vector storage. Adds `storage_provider.py` for the ChromaDB abstraction.
- **memoryos-mcp/** — MCP (Model Context Protocol) server wrapping MemoryOS. Entry point is `server_new.py` which exposes `add_memory`, `retrieve_memory`, and `get_user_profile` as MCP tools via FastMCP with stdio transport. Config is loaded from a JSON file.
- **memoryos-playground/** — Flask web app demo with `memdemo/app.py` as the entry point.

Each implementation shares the same internal module structure:
- `memoryos.py` — Main `Memoryos` class orchestrating all components
- `short_term.py` — FIFO queue (deque) for recent QA pairs
- `mid_term.py` — Session/segment storage with heat-based promotion; uses FAISS indexes
- `long_term.py` — User profiles and knowledge bases with FAISS similarity search
- `updater.py` — Promotes short-term to mid-term, triggers profile/knowledge extraction when heat threshold exceeded
- `retriever.py` — Parallel retrieval across all memory layers
- `prompts.py` — LLM prompt templates for summarization, profile analysis, knowledge extraction
- `utils.py` — OpenAIClient wrapper (handles both LLM chat and embeddings via OpenAI-compatible API), parallel execution utilities

**eval/** — Evaluation scripts for the LoCoMo benchmark (F1, BLEU-1 scores).

## Core Architecture

**Memory flow:** User/agent QA pairs enter short-term memory. When short-term is full, the Updater consolidates interactions into mid-term memory segments. Mid-term segments accumulate "heat" (based on visit frequency, interaction length, time recency). When heat exceeds a threshold, the Updater extracts user profile updates and knowledge entries into long-term memory.

**Key parameters** (all configurable via `Memoryos.__init__`):
- `short_term_capacity` (default 10) — triggers flush to mid-term when full
- `mid_term_heat_threshold` (default 5.0) — triggers long-term promotion
- `mid_term_similarity_threshold` (default 0.6) — for session merging
- `embedding_model_name` (default "text-embedding-3-small") — any OpenAI-compatible embedding API

**LLM & Embedding integration:** All LLM and embedding calls go through `OpenAIClient` which wraps the OpenAI SDK. Any OpenAI-compatible API works (Deepseek, Qwen, vLLM, etc.) by setting `openai_base_url`. LLM and embedding can use **separate providers** via `embedding_api_key` and `embedding_base_url` parameters. Configuration is stored in `.env` (see `.env.example`). No local embedding models are used — all embeddings are API-based.

**Data persistence:** Each implementation uses JSON files for storage, organized under `data_storage_path/users/{user_id}/` and `data_storage_path/assistants/{assistant_id}/`.

## Commands

All commands use `uv` — never use system `python` or `python3` directly.

### Install dependencies
```bash
cd memoryos-pypi && uv sync        # PyPI version
cd memoryos-chromadb && uv sync    # ChromaDB version
cd memoryos-mcp && uv sync         # MCP server
cd memoryos-playground && uv sync  # Playground
cd eval && uv sync                 # Evaluation scripts
```
For GPU support (faiss-gpu): `uv sync --extra gpu`

### Run tests
```bash
cd memoryos-pypi && uv run python test.py          # Basic demo/integration test (requires API key)
cd memoryos-mcp && uv run python test_simple.py    # MCP test
```
There is no pytest suite; tests are script-based demos that require a valid OpenAI-compatible API key and base URL configured in the test file.

### Run MCP server
```bash
cd memoryos-mcp && uv run python server_new.py --config config.json
```

### Run playground
```bash
cd memoryos-playground/memdemo && uv run python app.py
```

### Run evaluation (LoCoMo benchmark)
```bash
cd eval && uv run python evalution_loco.py <results_json_path>
```

### Docker
```bash
docker build -t memoryos .
```

## Development Notes

- Code comments and some error messages are in Chinese (the team's primary language).
- Import patterns use try/except to support both relative imports (package mode) and absolute imports (standalone mode).
- The `utils.py` contains `clean_reasoning_model_output()` to strip `<think>` tags from reasoning models (Deepseek-R1, Qwen3).
- Mid-term memory heat formula: `H = alpha * N_visit + beta * L_interaction + gamma * R_recency` (configurable constants in `mid_term.py`).
- Thread safety: all memory modules use `threading.Lock()` for concurrent access.
- `faiss-cpu` is the default vector index dependency; use `uv sync --extra gpu` for `faiss-gpu`.
- Package management uses `uv` (pyproject.toml per subproject). Never invoke bare `python`/`python3` — always use `uv run python`.
