# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Internal Update Loop

Whenever a change is made to any agent, workflow node, prompt, or
architectural pattern in this repo, this file should be updated to reflect
the new behavior. After any significant update, re-read this file in full
to ensure the documented architecture still matches the actual code. If a
discrepancy is found, correct this file first before continuing. This
creates a self-reinforcing loop: code changes trigger doc updates, and doc
updates guide future code changes accurately.

## Commands

```bash
# Run the app
streamlit run streamlit_app.py

# CLI mode (hardcoded test conversation, useful for backend debugging)
python main.py

# Tests
python tests/test_pantry_agent_comprehensive.py   # 26-test pantry agent suite
python tests/test_hybrid_search.py               # Hybrid search tests

# Pantry utilities
python scripts/validate_pantry.py   # Inspect current DB contents
python scripts/clear_pantry.py      # Reset pantry database

# RAG evaluation
python scripts/evaluate_rag.py                      # 20 samples, top-10 (default)
python scripts/evaluate_rag.py --sample 50 --k 5   # Custom params

# Recipe ingestion (one-time setup, ~10-15 min for ~13k recipes)
python scripts/ingest_recipes_milvus.py --input assets/full_dataset.csv --outdir data --build-milvus
python scripts/ingest_recipes_qdrant.py   # Local Qdrant alternative

# Dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file (see `.env.example`):

```env
OPENAI_API_KEY=...                    # Required
ZILLIZ_CLUSTER_ENDPOINT=...          # Required for recipe search (Milvus/Zilliz Cloud)
ZILLIZ_TOKEN=...
LANGCHAIN_TRACING_V2=true            # Optional LangSmith tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=leftovr-app
```

On Streamlit Cloud, set these as app secrets (TOML format). `streamlit_app.py` bridges `st.secrets` → `os.environ` before importing `main.py`.

## Architecture

**Strict frontend/backend separation:**
- `streamlit_app.py` — pure UI; imports `create_workflow` from `main.py` and caches it via `@st.cache_resource`
- `main.py` — all LangGraph workflow logic; agents are instantiated here

**LangGraph workflow (`main.py`):**
- State schema: `RecipeWorkflowState(MessagesState)` with fields for user context, pantry data, recipe results, and response
- 3 LLM instances: `llm` (general, temp=0.7), `llm_classifier` (JSON mode, structured extraction), `llm_creative` (temp=0.8, recipe recommendations)
- Routing via `query_type`: `"pantry"` → PantryAgent, `"recipe"` → RecipeKnowledgeAgent + SousChefAgent, `"general"` / `"preference"` / `"off_topic"` → ExecutiveChefAgent

**Four agents (`agents/`):**
- `executive_chef_agent.py` — classifies queries, orchestrates routing, handles general conversation and preference collection
- `pantry_agent.py` — MCP **client**; spawns `mcp/server.py` as a subprocess and communicates via stdio JSON-RPC; no direct DB access
- `recipe_knowledge_agent.py` — hybrid vector+keyword search using fastembed (`sentence-transformers/all-MiniLM-L6-v2`, dim=384) against Milvus/Zilliz Cloud; falls back to local Qdrant
- `sous_chef_agent.py` — ranks top-3 recipes from search results and adapts selected recipe to user's pantry

**MCP architecture (`mcp/server.py`):**
- Standalone subprocess, communicates via stdio JSON-RPC
- Owns the SQLite database; `PantryAgent` has zero direct DB access
- The `LeftovrWorkflow.__init__` connects `PantryAgent` to the MCP server synchronously, with event-loop-aware fallback using `ThreadPoolExecutor`

**Data persistence:**
- Pantry: SQLite at `~/.leftovr/pantry.db`, table `food_items(id, name, quantity, expire_date)`
- Food item IDs are normalized: singularized → lowercase → hyphenated (e.g., `"chicken-breast"`)
- Adding an existing item increments its quantity rather than overwriting

**Recipe vector database:**
- Primary: Milvus/Zilliz Cloud (`pymilvus`), collection `recipes`
- Fallback: local Qdrant (`./qdrant_data/`)
- `RecipeKnowledgeAgent.setup_milvus()` is **lazy** — called on first recipe query, not at startup, to avoid loading the ~22 MB ONNX model during cold start
