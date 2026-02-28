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
- `streamlit_app.py` — pure UI; imports `create_workflow` from `main.py` and caches it via `@st.cache_resource`. Builds `context_messages` from `st.session_state.chat_history[:-1]` (excluding the current user message) and passes it as `messages` to the workflow. This provides conversation context to both the orchestrator's classifier and the pantry agent's LLM. The `messages` state is **not** persisted in session state — it is derived fresh from `chat_history` each invocation.
- `main.py` — all LangGraph workflow logic; agents are instantiated here

**LangGraph workflow (`main.py`):**
- State schema: `RecipeWorkflowState(MessagesState)` with fields for user context, pantry data, recipe results, and response
- 3 LLM instances: `llm` (general, temp=0.7), `llm_classifier` (JSON mode, structured extraction), `llm_creative` (temp=0.8, recipe recommendations)
- State schema includes `preference_action` field (from classifier) used by `_format_preferences_response`
- `classify_and_extract` is called **before** the bypass check in `_orchestrator_node` so `wants_to_exit_flow` and `is_new_recipe_search` are available early
- Routing via `query_type`: `"pantry"` → PantryAgent, `"recipe"` → RecipeKnowledgeAgent + SousChefAgent, `"general"` / `"preference"` / `"off_topic"` → ExecutiveChefAgent

**Four agents (`agents/`):**
- `executive_chef_agent.py` — classifies queries, orchestrates routing, handles general conversation and preference collection. `classify_and_extract` returns `wants_to_exit_flow`, `is_new_recipe_search`, and `preference_action` alongside existing fields. Prompt includes implicit-replacement examples ("only X", "just X", "switch to X"). Prompt also includes context-dependent pantry correction rules: when a user refers to a quantity without naming a food item right after a pantry message (e.g., "nvm i have 20 in hand"), the classifier routes to `"pantry"` instead of `"general"` and sets `wants_to_exit_flow` to false.
- `pantry_agent.py` — MCP **client**; spawns `mcp/server.py` as a subprocess and communicates via stdio JSON-RPC; no direct DB access. `handle_query` accepts optional `conversation_history` (last 6 turns) and injects both the current pantry inventory and pending-items context into the LLM system prompt. This lets the LLM distinguish "I have 11 eggs" (set, when eggs exist) from adding new items, and resolve ambiguous references like "i have 11" from conversation context. The system prompt includes context-dependent reference resolution: when the user says "nvm i have 20 in hand" without naming an item, the LLM infers the item from the most recently discussed food in conversation history and calls the appropriate tool. The pending-items prompt explicitly handles qualifier phrases like "from leftovers", "from yesterday", "approximately" so the LLM doesn't mistake them for new items. `handle_query`'s conversation history processing handles both plain dicts (`{"role": ..., "content": ...}`) and LangChain message objects (`HumanMessage`/`AIMessage` with `.type` and `.content`), mapping `"human"→"user"` and `"ai"→"assistant"`. All keyword-based methods (`_is_food_item`, `_simple_quantity_check`, `_detect_items_without_quantity`, `_is_quantity_response`, `_handle_quantity_response`, `_extract_quantities`) have been removed — the LLM handles all intent classification via function calling. `handle_query` returns `{"result": PantryItemsResponse, "operations": [...]}` with structured operation metadata.
- `recipe_knowledge_agent.py` — hybrid vector+keyword search using fastembed (`sentence-transformers/all-MiniLM-L6-v2`, dim=384) against Milvus/Zilliz Cloud; falls back to local Qdrant
- `sous_chef_agent.py` — ranks top-3 recipes from search results and adapts selected recipe to user's pantry. `converse_about_recommendations` uses a `[SELECTION: N]` tag from the LLM as a fallback when keyword matching doesn't detect a selection.

**MCP architecture (`mcp/server.py`):**
- Standalone subprocess, communicates via stdio JSON-RPC
- Owns the SQLite database; `PantryAgent` has zero direct DB access
- The `LeftovrWorkflow.__init__` connects `PantryAgent` to the MCP server synchronously, with event-loop-aware fallback using `ThreadPoolExecutor`

**Data persistence:**
- Pantry: SQLite at `~/.leftovr/pantry.db`, table `food_items(id, name, quantity, expire_date)`
- Food item IDs are normalized: singularized → lowercase → hyphenated (e.g., `"chicken-breast"`)
- `add_food_item` increments quantity if item exists; `handle_query` now uses `set_food_quantity` when the user states a current total for an existing item (e.g., "I have 11 eggs" with eggs already in pantry)
- `_format_pantry_response_smart` in `main.py` formats responses based on operation metadata returned by `handle_query`, not keyword matching on the user message

**Recipe vector database:**
- Primary: Milvus/Zilliz Cloud (`pymilvus`), collection `recipes`
- Fallback: local Qdrant (`./qdrant_data/`)
- `RecipeKnowledgeAgent.setup_milvus()` is **lazy** — called on first recipe query, not at startup, to avoid loading the ~22 MB ONNX model during cold start
