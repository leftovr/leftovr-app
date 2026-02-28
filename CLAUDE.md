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
- `streamlit_app.py` — pure UI; imports `create_workflow` from `main.py` and caches it via `@st.cache_resource`. Builds `context_messages` from `st.session_state.chat_history[:-1]` (excluding the current user message) and passes it as `messages` to the workflow. This provides conversation context to both the orchestrator's classifier and the pantry agent's LLM. The `messages` state is **not** persisted in session state — it is derived fresh from `chat_history` each invocation. `recipe_results`, `recommendation_history`, and `current_recommendation_page` **are** persisted in session state and passed back to the workflow on every invocation so that "show me more" and back-reference flows work across turns.
- `main.py` — all LangGraph workflow logic; agents are instantiated here

**LangGraph workflow (`main.py`):**
- State schema: `RecipeWorkflowState(MessagesState)` with fields for user context, pantry data, recipe results, and response. `user_preferences` dict holds `allergies`, `restrictions`, `cuisines`, `excluded_food_types`, `preferred_food_types`, `diet`, and `skill`.
- State schema includes `recommendation_history` (list of batches) and `current_recommendation_page` (0-indexed) to support recommendation pagination and back-references.
- 3 LLM instances: `llm` (general, temp=0.7), `llm_classifier` (JSON mode, structured extraction), `llm_creative` (temp=0.8, recipe recommendations)
- State schema includes `preference_action` field (from classifier) used by `_format_preferences_response`
- `classify_and_extract` is called **before** the bypass check in `_orchestrator_node` so `wants_to_exit_flow`, `is_new_recipe_search`, `wants_more_recommendations`, and `wants_previous_recommendations` are available early
- Routing via `query_type`: `"pantry"` → PantryAgent, `"recipe"` → RecipeKnowledgeAgent + SousChefAgent, `"general"` / `"preference"` / `"off_topic"` → ExecutiveChefAgent. A `"more_recipes"` route goes directly to the recommendation node, bypassing search.

**Four agents (`agents/`):**
- `executive_chef_agent.py` — classifies queries, orchestrates routing, handles general conversation and preference collection. `classify_and_extract` returns `wants_to_exit_flow`, `is_new_recipe_search`, `wants_more_recommendations`, `wants_previous_recommendations`, `previous_batch_selection`, and `preference_action` alongside existing fields. `wants_more_recommendations` triggers the next batch from existing search results (distinct from `is_new_recipe_search` which re-runs the search). `wants_previous_recommendations` restores an earlier batch, optionally with a `previous_batch_selection` (1-3). Affirmative replies ("yes", "sure") after a recipe suggestion prompt classify as `query_type: "recipe"`. Supports `excluded_food_types` / `removed_excluded_food_types` / `clear_excluded_food_types` for negative food-type preferences (e.g., "no desserts", "skip soups and salads") and `preferred_food_types` / `removed_preferred_food_types` / `clear_preferred_food_types` for positive food-type preferences (e.g., "I only want pasta related", "just curry dishes"). The prompt includes a guard instruction: positive requests ("I only want X") must use `preferred_food_types`, not infer `excluded_food_types`. Prompt includes implicit-replacement examples ("only X", "just X", "switch to X") for all preference categories including `preferred_food_types`. Prompt includes two pantry classification priority rules: (1) ingredient declarations ("I have chicken, garlic and pasta") are ALWAYS classified as `"pantry"` regardless of conversation context, and (2) context-dependent quantity corrections (e.g., "nvm i have 20 in hand") right after a pantry message also route to `"pantry"` with `wants_to_exit_flow` false.
- `pantry_agent.py` — MCP **client**; spawns `mcp/server.py` as a subprocess and communicates via stdio JSON-RPC; no direct DB access. `handle_query` accepts optional `conversation_history` (last 6 turns) and injects both the current pantry inventory and pending-items context into the LLM system prompt. This lets the LLM distinguish "I have 11 eggs" (set, when eggs exist) from adding new items, and resolve ambiguous references like "i have 11" from conversation context. Quantity clarification rules explicitly handle word-form numbers ("two broccolis" = quantity 2) as explicit quantities — `ask_for_quantity` is only used when truly no number is present. When the LLM calls `add_food_item` without a quantity, the item is tracked in `items_missing_qty` and automatically merged into `pending_items` so the clarification question covers ALL quantity-less items (code-level safety net, not LLM-dependent). The pending-items prompt uses strict ordered rules: multiple numbers map positionally to pending items (first number → first item, etc.), a single number with one pending item applies to that item, and a single number with multiple pending items applies to all. The prompt forbids modifying any items not in the pending list, even if mentioned in earlier messages. `_generate_quantity_question` in `main.py` uses a clear per-item list format for multiple items. `handle_query`'s conversation history processing handles both plain dicts and LangChain message objects. `handle_query` returns `{"result": PantryItemsResponse, "operations": [...]}` with structured operation metadata.
- `recipe_knowledge_agent.py` — hybrid vector+keyword search using fastembed (`sentence-transformers/all-MiniLM-L6-v2`, dim=384) against Milvus/Zilliz Cloud; falls back to local Qdrant. `hybrid_query` accepts `allergies`, `preferred_cuisines`, `excluded_food_types`, and `preferred_food_types` parameters. Post-filters results to exclude allergen-containing recipes and recipes matching excluded food type categories (dessert, soup, appetizer, etc.) via title-keyword matching using the `FOOD_TYPE_KEYWORDS` class dict, and to include only preferred food types when specified via the `PREFERRED_FOOD_TYPE_KEYWORDS` dict (pasta, curry, stir fry, casserole, sandwich, rice dish, main dish, seafood, pizza, taco). Uses adaptive `allow_missing` (starts at 0, escalates to the requested limit) and requires `num_pantry_used >= 1` to prevent irrelevant recipes from appearing.
- `sous_chef_agent.py` — ranks top-3 recipes from search results and adapts selected recipe to user's pantry. Guarantees exactly 3 recommendations by filling gaps with `build_fallback_recommendations` when the LLM returns fewer. `match_percentage` is carried over from the search pipeline's real ingredient-overlap data, not fabricated by the LLM. `converse_about_recommendations` uses a `[SELECTION: N]` tag from the LLM as a fallback when keyword matching doesn't detect a selection.

**MCP architecture (`mcp/server.py`):**
- Standalone subprocess, communicates via stdio JSON-RPC
- Owns the SQLite database; `PantryAgent` has zero direct DB access
- The `LeftovrWorkflow.__init__` connects `PantryAgent` to the MCP server synchronously, with event-loop-aware fallback using `ThreadPoolExecutor`

**Data persistence:**
- Pantry: SQLite at `~/.leftovr/pantry.db`, table `food_items(id, name, quantity, expire_date)`
- Food item IDs are normalized: singularized → lowercase → hyphenated (e.g., `"chicken-breast"`)
- `add_food_item` increments quantity if item exists; `handle_query` now uses `set_food_quantity` when the user states a current total for an existing item (e.g., "I have 11 eggs" with eggs already in pantry)
- `_format_pantry_response_smart` in `main.py` formats responses based on operation metadata returned by `handle_query`, not keyword matching on the user message
- `_pantry_node` always sets `current_stage="pantry_complete"` (no longer overrides to `"presenting_options"` based on stale recommendations). A `clear_pantry` operation also resets `top_3_recommendations` to `[]` and `recommendation_history` to `[]` in the returned state. After a successful add/update/set operation, if the pantry is non-empty and no recommendations are active, appends a proactive recipe suggestion prompt.
- `_recipe_search_node` builds a meaningful semantic query from pantry items + cuisine preferences (not the raw user message), passes allergy/cuisine filters to `hybrid_query`, and preserves `score`/`pantry_items_used`/`missing_ingredients`/`match_percentage` through to the recommendation node. Resets `recommendation_history` and `current_recommendation_page` on each fresh search.
- `_recommendation_node` supports pagination: on a "more recommendations" request (`current_stage == "more_recommendations"`), it filters `recipe_results` to exclude already-shown recipe IDs from `recommendation_history`, then generates the next batch of 3. If the remaining pool is too small, `_run_broader_search` re-queries with relaxed parameters (`top_k=20`, `allow_missing=5`). Each batch is appended to `recommendation_history` and the page counter is incremented.
- `_route_from_orchestrator` returns `"more_recipes"` when `current_stage == "more_recommendations"`, routing directly to `_recommendation_node` (bypassing `_recipe_search_node`). The graph has a corresponding `"more_recipes": "recommendation"` conditional edge.

**Error handling:**
- `classify_and_extract()` in `_orchestrator_node` is wrapped in try/except; falls back to `query_type: "general"` with empty preferences on failure.
- `pantry.handle_query()` in `_pantry_node` is wrapped in try/except; returns a user-friendly error message.
- `sous_chef.generate_recommendations()`, `sous_chef.adapt_recipe()`, `sous_chef.converse_about_recommendations()`, and `exec_chef.respond_as_waiter()` are all wrapped in try/except with graceful fallback responses.
- All workflow node return dicts include `user_preferences` to prevent silent preference drops when the UI reads the result.

**Recipe vector database:**
- Primary: Milvus/Zilliz Cloud (`pymilvus`), collection `recipes`
- Fallback: local Qdrant (`./qdrant_data/`)
- `RecipeKnowledgeAgent.setup_milvus()` is **lazy** — called on first recipe query, not at startup, to avoid loading the ~22 MB ONNX model during cold start
