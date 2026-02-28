# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Internal Update Loop

Whenever a change is made to any agent, workflow node, prompt, or architectural pattern in this repo, this file should be updated to reflect the new behavior. After any significant update, re-read this file in full to ensure the documented architecture still matches the actual code. If a discrepancy is found, correct this file first before continuing. This creates a self-reinforcing loop: code changes trigger doc updates, and doc updates guide future code changes accurately.

## Commands

```bash
# Run the Streamlit web app
streamlit run streamlit_app.py

# Run the CLI test mode (hardcoded test conversation)
python main.py

# Run tests
python tests/test_pantry_agent_comprehensive.py   # 26 pantry agent tests
python tests/test_hybrid_search.py                # hybrid search tests

# Database utilities
python scripts/validate_pantry.py   # inspect pantry contents
python scripts/clear_pantry.py      # reset pantry database

# Ingest recipe data (required before recipe search works)
python scripts/ingest_recipes_milvus.py --input assets/full_dataset.csv --outdir data --build-milvus
python scripts/ingest_recipes_qdrant.py   # local alternative

# RAG evaluation
python scripts/evaluate_rag.py                         # 20 samples, top-10 results
python scripts/evaluate_rag.py --sample 50 --k 5      # custom evaluation
python scripts/evaluate_rag.py --output results.json   # save results
```

## Environment Variables

```env
OPENAI_API_KEY=...                    # Required
ZILLIZ_CLUSTER_ENDPOINT=...          # Recommended (Milvus cloud)
ZILLIZ_TOKEN=...
LANGCHAIN_TRACING_V2=true            # Optional LangSmith tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=leftovr-app
```

## Architecture

### Layer Separation

- **`streamlit_app.py`** — pure UI, no business logic. Caches `LeftovrWorkflow` with `@st.cache_resource` and passes state between turns.
- **`main.py`** — all orchestration logic via `LeftovrWorkflow` / `RecipeWorkflowState`. Exposes `invoke()` and `async ainvoke()`.
- **`agents/`** — four specialized agents instantiated in `LeftovrWorkflow.__init__`.
- **`database/pantry_storage.py`** — `PantryDatabase` SQLite wrapper. DB stored at `~/.leftovr/pantry.db`.
- **`mcp/server.py`** — standalone MCP server spawned as a subprocess by `PantryAgent`. Communicates via JSON-RPC over stdio (logs to stderr).

### LangGraph Workflow Orchestration (`main.py`)

`RecipeWorkflowState` (extends `MessagesState`) tracks: `query_type`, `current_stage`, `pantry_inventory`, `expiring_items`, `recipe_results`, `top_3_recommendations`, `user_recipe_selection`, `customized_recipe`, `response`, `coordination_log`.

```
User message
    │
    ▼
orchestrator (ExecutiveChefAgent)
    │  classify_query() → "pantry" | "recipe" | "general" | "selection"
    │  extract_preferences() → merged into user_preferences state
    │
    ├─── "pantry" ──────────────────► pantry node (PantryAgent)
    │                                     handle_query() → NLP parse + MCP calls
    │                                     └── needs_clarification? → set current_stage
    │                                                                  = "awaiting_quantity_clarification"
    │                                                                  (next turn skips classify, routes straight to pantry)
    │
    ├─── "recipe" ──────────────────► recipe_search node (RecipeKnowledgeAgent)
    │                                     hybrid_query() → top-10 candidates
    │                                         │
    │                                         ▼
    │                                     recommendation node (SousChefAgent)
    │                                         generate_recommendations() → top-3
    │
    ├─── "selection" ───────────────► customization node (SousChefAgent)
    │                                     adapt_recipe() + format_recipe_for_user()
    │
    └─── "general" ─────────────────► general_response node (ExecutiveChefAgent)
                                          respond_as_waiter()
```

Three LLM instances are defined at module level in `main.py` and passed explicitly into agent methods — agents do not own their LLMs:

| Instance | Temperature | Mode | Used by |
|---|---|---|---|
| `llm` | 0.7 | standard | `general_response` node |
| `llm_classifier` | 0.0 | JSON mode | `orchestrator` node |
| `llm_creative` | 0.8 | standard | `recommendation`, `customization` nodes |

### Agent Tools

#### ExecutiveChefAgent (`agents/executive_chef_agent.py`)

Stateless LLM wrapper — all methods take an `llm` argument and return structured data. Called directly by `main.py` nodes.

| Method | LLM | Returns | Used in node |
|---|---|---|---|
| `classify_query(llm, messages)` | `llm_classifier` | `{"query_type": "pantry"\|"recipe"\|"general"}` | `orchestrator` |
| `extract_preferences(llm, messages)` | `llm_classifier` | `{"allergies", "restrictions", "cuisines", "diet", "skill"}` | `orchestrator` |
| `respond_as_waiter(llm, user_input)` | `llm` | formatted response str | `general_response` |
| `extract_ingredients(llm, user_message)` | any | `{"ingredients": [{name, quantity, unit}]}` | available |
| `pantry_info_sufficient(llm, user_text)` | any | `{"sufficient_info": bool}` | available |
| `perform_quality_check(llm, recipe_text, user_prefs, messages)` | any | `{passed, issues, suggestion}` | available |
| `synthesize_recommendations(llm, agent_responses, user_preferences)` | any | formatted str | available |

> The `delegate_to_*` methods (`delegate_to_pantry`, `delegate_to_sous_chef`, `delegate_to_recipe_knowledge`, `delegate_to_quality_control`) create structured delegation packets and log them to `self.delegation_log`, but are **not wired into the LangGraph workflow** — routing is handled by `_route_from_orchestrator` in `main.py`.

#### PantryAgent (`agents/pantry_agent.py`)

MCP client — all data operations go through JSON-RPC to `mcp/server.py`. Each public method has a sync wrapper over an async `_*_async` implementation.

| Method | MCP Tool Called | Description |
|---|---|---|
| `handle_query(user_msg)` *(async)* | multiple | Main NLP entry point — uses OpenAI to detect intent, then calls appropriate MCP tools. Returns `PantryItemsResponse` or `{"needs_clarification": True, "pending_items": [...]}` |
| `add_or_update_ingredient(name, qty, unit, expire_date)` | `add_food_item` | Adds item; default expiry = 14 days |
| `get_inventory()` | `get_all_food_items` | Returns list of all pantry items |
| `get_expiring_soon(days_threshold)` | `get_expiring_soon` | Returns items expiring within N days |
| `remove_ingredient(ingredient_id)` | `delete_food_item` | Deletes by normalized ID |
| `clear_pantry()` | `delete_food_item` (×N) | Deletes all items |
| `update_quantity(ingredient_id, new_quantity, mode)` | `update_food_item` | Mode `"absolute"` sets value; `"delta"` adds/subtracts |

`handle_query` uses `_is_food_item()` (OpenAI call) to reject non-food items, `_detect_items_without_quantity()` to trigger clarification flow, and `_is_quantity_response()` to detect follow-up answers.

#### RecipeKnowledgeAgent (`agents/recipe_knowledge_agent.py`)

Stateless search engine wrapping Milvus. Has a `pantry_agent` reference injected via `set_pantry_agent()` for live inventory access.

| Method | Description |
|---|---|
| `hybrid_query(pantry_items, query_text, top_k, allow_missing, use_semantic)` | **Primary search** — combines `pantry_candidates()` (exact array filtering) + `semantic_search()` (vector ANN). Returns `[(recipe_metadata, score, num_used, missing)]` |
| `pantry_candidates(pantry_items, allow_missing, top_k)` | Milvus `array_contains` filter search. Scores = `(unique_pantry_items_used × 100) + (1000 if 0 missing) - recipe_size` |
| `semantic_search(query, pantry_items, k)` | Embeds query with `all-MiniLM-L6-v2`, ANN search in Milvus. Semantic score boosted by ×50 on top of pantry score |
| `get_recipe_by_id(recipe_id)` | Single Milvus query by ID |
| `get_recipes_by_ids(recipe_ids)` | Batch Milvus query |
| `feasibility_with_pantry(recipe_meta, allow_missing)` | Checks recipe feasibility against live pantry |
| `get_pantry_items()` | Pulls from injected `pantry_agent.get_inventory()` |
| `load_directions(path)` | Optional — loads cooking steps from `data/recipe_metadata.jsonl` into `directions_cache` |

If `pantry_items=None` in `hybrid_query`, it auto-pulls from the injected `PantryAgent`.

#### SousChefAgent (`agents/sous_chef_agent.py`)

LLM-driven ranker and recipe adapter. Has a `recipe_knowledge_agent` reference for fallback recipe fetching.

| Method | LLM | Description |
|---|---|---|
| `generate_recommendations(llm, pantry_summary, user_preferences, expiring_items, recipe_results)` | `llm_creative` | Ranks pre-fetched results into top-3; if `recipe_results` is empty, calls `recipe_knowledge_agent.hybrid_query()` internally as fallback |
| `adapt_recipe(llm, recipe, user_preferences, pantry_inventory)` | `llm_creative` | Adapts selected recipe to user's pantry and dietary constraints; suggests substitutions |
| `format_recipe_for_user(recipe, preferences)` | `llm_creative` | Formats the adapted recipe into a readable response |
| `present_recommendations(llm, top_3, expiring_items, user_preferences)` | any | Formats the top-3 list for display |
| `check_ingredient_availability(recipe, pantry_inventory)` | — | Pure Python: computes available/missing ingredients |
| `request_recipes_from_knowledge_agent(user_ingredients, user_preferences)` | — | Direct call to `recipe_knowledge_agent.hybrid_query()` |
| `converse_about_recommendations(llm, user_message, top_3, pantry, preferences)` | any | Handles follow-up Q&A about recommendations |

### MCP Communication (`agents/pantry_agent.py` ↔ `mcp/server.py`)

`PantryAgent` spawns `mcp/server.py` as a subprocess and communicates via JSON-RPC over stdio. `ensure_connected()` / `disconnect()` manage the subprocess lifecycle. The server exposes six tools: `get_all_food_items`, `get_expiring_soon`, `get_food_item`, `add_food_item`, `update_food_item` (supports `absolute`/`delta` modes), `delete_food_item`.

The async event-loop compatibility pattern (get running loop → use `ThreadPoolExecutor` if one exists, else `asyncio.run()`) appears in both `LeftovrWorkflow.__init__` and `_pantry_node`.

### Item ID Normalization

Both `pantry_agent.py` and `mcp/server.py` define `normalize_food_id()`: singularize → lowercase → strip → hyphenate. Item IDs are deterministic (e.g., `"chicken breast"` → `"chicken-breast"`). This is the key deduplication mechanism.

### Recipe Search (`agents/recipe_knowledge_agent.py`)

- **Primary**: Milvus/Zilliz Cloud — collection `recipes`, embedding model `sentence-transformers/all-MiniLM-L6-v2` (dim=384).
- **Fallback**: local Qdrant stored in `./qdrant_data/`.
- Requires `data/ingredient_index.json` and `data/recipe_metadata.jsonl` to be present (generated by ingestion scripts).
- If `RecipeKnowledgeAgent` fails to connect, it is set to `None` in `LeftovrWorkflow` and recipe search returns a "not available" message.
