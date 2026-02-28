"""
Leftovr - Refactored LangGraph Workflow
Clean separation: Streamlit = Frontend | LangGraph = Backend Logic
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Annotated
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, add_messages, END
from langgraph.types import Command

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith tracing configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "leftovr-app")

# Enable LangSmith tracing if configured
if LANGCHAIN_TRACING_V2.lower() == "true" and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print(f"✅ LangSmith tracing enabled for project: {LANGCHAIN_PROJECT}")
else:
    print("ℹ️  LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")

# Initialize OpenAI client with GPT-4o for optimal performance
# NOTE: JSON mode only used for llm_classifier (structured data extraction)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,  # Default for general use
    api_key=OPENAI_API_KEY
)

# Specialized LLM instances for different tasks
llm_classifier = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,  # Deterministic for classification
    api_key=OPENAI_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}}  # JSON mode for structured outputs
)

llm_creative = ChatOpenAI(
    model="gpt-4o",
    temperature=0.8,  # Higher creativity for recommendations
    api_key=OPENAI_API_KEY
    # NO JSON mode - creative outputs should be natural text
)

from agents.recipe_knowledge_agent import RecipeKnowledgeAgent
from agents.executive_chef_agent import ExecutiveChefAgent
from agents.pantry_agent import PantryAgent
from agents.sous_chef_agent import SousChefAgent


# ============================================
# SIMPLIFIED STATE SCHEMA
# ============================================

class RecipeWorkflowState(MessagesState):
    """
    Simplified state for recipe workflow.
    Only essential fields - no duplicate logic.
    """
    # User context
    user_message: str  # Current user input
    user_preferences: Dict[str, Any]  # {allergies, diet, cuisines, skill_level}

    # Workflow control
    query_type: Optional[Literal["pantry", "recipe", "general", "selection", "off_topic", "preference"]]  # What type of request
    current_stage: str  # Track workflow stage

    # Pantry data
    pantry_inventory: List[Dict[str, Any]]  # Available ingredients
    expiring_items: List[Dict[str, Any]]  # Ingredients expiring soon

    # Recipe search results
    recipe_results: List[Dict[str, Any]]  # Top-k recipes from search (e.g., 10)
    top_3_recommendations: List[Dict[str, Any]]  # Sous Chef's top 3 picks

    # User selection & final recipe
    user_recipe_selection: Optional[int]  # 1, 2, or 3
    selected_recipe_data: Optional[Dict[str, Any]]  # Full recipe details
    customized_recipe: Optional[Dict[str, Any]]  # Final adapted recipe

    # Response
    response: Optional[str]  # Text response for general queries

    # Classifier metadata (from classify_and_extract)
    preference_action: Optional[str]  # "view" | "update" | None

    # Coordination log
    coordination_log: Annotated[List[str], operator.add]  # Workflow tracking


# ============================================
# LANGGRAPH WORKFLOW - SIMPLIFIED
# ============================================

class LeftovrWorkflow:
    """
    Clean LangGraph workflow with specialized nodes.
    Streamlit handles UI, this handles all logic.
    """

    def __init__(self):
        # Initialize agents
        self.exec_chef = ExecutiveChefAgent(name="Maison D'Être")
        self.pantry = PantryAgent(name="Pantry Manager")

        # Connect pantry agent to MCP server
        import asyncio
        try:
            # Try to connect synchronously
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop - safe to use asyncio.run()
                asyncio.run(self.pantry.ensure_connected())
            else:
                # Already in a loop - use run_in_executor with a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.pantry.ensure_connected())
                    future.result()
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to MCP server: {e}")
            print("   Make sure mcp/server.py is available")

        # Initialize Recipe Knowledge Agent lazily — setup_milvus() is deferred to first use
        # This avoids loading the ~22 MB ONNX model at startup (saves ~2-4 GB RAM on torch stack)
        self.recipe_agent = RecipeKnowledgeAgent(data_dir='data')
        self._recipe_agent_initialized = False
        self._warmup_started = False
        self._warmup_future = None

        self.sous_chef = SousChefAgent(name="Sous Chef", recipe_knowledge_agent=self.recipe_agent)

        # Build workflow graph
        self.graph = self._build_graph()

    def _ensure_recipe_agent_ready(self) -> bool:
        """
        Lazy-initialise RecipeKnowledgeAgent on first use.
        Returns True if the agent is ready (milvus_client connected), False otherwise.
        Thread-safe: if a background warmup is in progress, waits for it.
        """
        if self._warmup_future is not None and not self._recipe_agent_initialized:
            print("⏳ Waiting for background warmup to finish...")
            self._warmup_future.result(timeout=60)
            return self.recipe_agent.milvus_client is not None

        if self._recipe_agent_initialized:
            return self.recipe_agent.milvus_client is not None

        self._recipe_agent_initialized = True
        try:
            self.recipe_agent.setup_milvus()

            try:
                self.recipe_agent.load_directions()
            except Exception as e:
                print(f"   ℹ️  Directions not loaded (optional): {e}")

            if self.recipe_agent.milvus_client:
                print("✅ Recipe Knowledge Agent initialized with Milvus cloud search")
                self.recipe_agent.set_pantry_agent(self.pantry)
                self.sous_chef.recipe_knowledge_agent = self.recipe_agent
                return True
            else:
                print("⚠️  Recipe Knowledge Agent: Milvus connection failed")
                print("   Run: python scripts/ingest_recipes_milvus.py --input assets/full_dataset.csv --outdir data --build-milvus")
                return False
        except Exception as e:
            print(f"⚠️  Recipe agent init failed: {e}")
            return False

    def start_background_warmup(self) -> None:
        """
        Kick off Zilliz/fastembed initialization in a background thread so the
        first recipe query doesn't block.  Safe to call multiple times — only
        the first call actually spawns work.
        """
        if self._warmup_started:
            return
        self._warmup_started = True

        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._warmup_future = executor.submit(self._ensure_recipe_agent_ready)
        executor.shutdown(wait=False)
        print("🔄 Background warmup started for Recipe Knowledge Agent")

    def __del__(self):
        """Cleanup: disconnect from MCP server when workflow is destroyed"""
        try:
            if hasattr(self, 'pantry') and self.pantry._connected:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(self.pantry.disconnect())
                else:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.pantry.disconnect())
                        future.result()
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def _build_graph(self) -> StateGraph:
        """Build the simplified LangGraph workflow"""
        workflow = StateGraph(RecipeWorkflowState)

        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("pantry", self._pantry_node)
        workflow.add_node("recipe_search", self._recipe_search_node)
        workflow.add_node("recommendation", self._recommendation_node)
        workflow.add_node("customization", self._customization_node)
        workflow.add_node("general_response", self._general_response_node)

        # Set entry point
        workflow.set_entry_point("orchestrator")

        # Conditional routing from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "pantry": "pantry",
                "recipe": "recipe_search",
                "general": "general_response",
                "selection": "customization"  # User selected a recipe
            }
        )

        # Simple edges
        workflow.add_edge("pantry", END)
        workflow.add_edge("recipe_search", "recommendation")
        workflow.add_edge("recommendation", END)
        workflow.add_edge("customization", END)
        workflow.add_edge("general_response", END)

        return workflow.compile()

    # ============================================
    # NODE 1: ORCHESTRATOR (Executive Chef)
    # ============================================

    def _is_exit_or_override(self, user_msg: str, current_stage: str) -> bool:
        """Detect if the user wants to cancel/override the current multi-turn flow."""
        msg = user_msg.lower().strip()
        EXIT_PHRASES = [
            "never mind", "nevermind", "nvm", "forget it", "forget that",
            "cancel", "start over", "ignore that", "scratch that",
        ]
        if any(p in msg for p in EXIT_PHRASES):
            return True
        if current_stage == "awaiting_quantity_clarification":
            RECIPE_SWITCH_SIGNALS = [
                "recipe", "cook", "make with", "what can i", "show me",
                "find me", "suggest", "recommend"
            ]
            if any(p in msg for p in RECIPE_SWITCH_SIGNALS):
                return True
        return False

    def _orchestrator_node(self, state: RecipeWorkflowState) -> Dict[str, Any]:
        """
        Single entry point - classify query and decide routing.
        This is the Executive Chef making decisions.
        """
        print("\n🎯 [ORCHESTRATOR] Analyzing user request...")

        user_msg = state.get("user_message", "")
        messages = state.get("messages", [])
        current_stage = state.get("current_stage", "initial")

        # Build message list for classification
        classify_messages = messages + [{"role": "user", "content": user_msg}] if user_msg else messages

        # One LLM call: classify query AND extract preferences simultaneously
        # Called early so new fields (wants_to_exit_flow, etc.) are available for bypass checks
        combined = self.exec_chef.classify_and_extract(llm_classifier, classify_messages)
        query_type = combined.get("query_type", "general")

        # CRITICAL: Check if we're in the middle of a multi-turn conversation
        # If awaiting clarification, route back to pantry — unless the user wants to exit
        BYPASS_STAGES = {"awaiting_quantity_clarification"}
        if current_stage in BYPASS_STAGES:
            if self._is_exit_or_override(user_msg, current_stage) or combined.get("wants_to_exit_flow"):
                print("🚪 [ORCHESTRATOR] User exiting clarification flow — reclassifying")
                self.pantry.pending_items = []  # clear stale pending items
                # fall through to full routing below
            else:
                print("🔄 [ORCHESTRATOR] Continuing quantity clarification flow -> routing to pantry")
                return {
                    "query_type": "pantry",
                    "user_preferences": state.get("user_preferences", {}),
                    "current_stage": "continuing_clarification",
                    "coordination_log": ["Continuing quantity clarification conversation"]
                }

        # Merge extracted preferences for recipe, general, selection, and preference queries.
        # For pantry queries, the LLM tends to misclassify ingredients as allergies.
        # Supports dynamic updates: add, remove specific items, clear a category, or clear all.
        current_prefs = state.get("user_preferences", {})
        if query_type in ["recipe", "general", "selection", "preference"]:
            updated_prefs = {**current_prefs}

            # "clear all" wipes everything first; individual category clears/adds/removes still apply
            if combined.get("clear_all_preferences"):
                updated_prefs = {"allergies": [], "restrictions": [], "cuisines": [],
                                 "diet": None, "skill": None}
                print("🗑️  [ORCHESTRATOR] Cleared ALL preferences")

            # Apply add/remove diff for each list-type preference
            for key in ("allergies", "restrictions", "cuisines"):
                clear_key = f"clear_{key}"
                remove_key = f"removed_{key}"

                # Category-level clear (runs after global clear so any additions below still apply)
                if combined.get(clear_key):
                    updated_prefs[key] = []
                    print(f"🗑️  [ORCHESTRATOR] Cleared all {key}")

                existing = [x.lower() for x in updated_prefs.get(key, [])]
                add_items = [x.lower() for x in combined.get(key, []) if x.strip()]
                drop_items = {x.lower() for x in combined.get(remove_key, []) if x.strip()}

                merged = list(dict.fromkeys(existing + add_items))   # dedupe, preserve order
                merged = [x for x in merged if x not in drop_items]

                if merged != existing:
                    updated_prefs[key] = merged
                    if drop_items:
                        print(f"🗑️  [ORCHESTRATOR] Removed from {key}: {drop_items}")
                    if add_items:
                        print(f"➕ [ORCHESTRATOR] Added to {key}: {add_items}")

            # Diet: explicit clear trumps a new value; new value overrides existing
            if combined.get("clear_diet"):
                updated_prefs["diet"] = None
                print("🗑️  [ORCHESTRATOR] Cleared diet preference")
            elif combined.get("diet"):
                updated_prefs["diet"] = combined["diet"]

            # Skill: always override if provided
            if combined.get("skill"):
                updated_prefs["skill"] = combined["skill"]
        else:
            # For pantry queries, keep existing preferences unchanged
            updated_prefs = current_prefs

        # Use LLM-detected selection
        if query_type == "selection":
            recipe_num = combined.get("selected_recipe_number")
            top_3 = state.get("top_3_recommendations", [])
            if recipe_num and 1 <= recipe_num <= len(top_3):
                print(f"✅ [ORCHESTRATOR] LLM-detected selection: recipe #{recipe_num}")
                return {
                    "query_type": "selection",
                    "user_preferences": updated_prefs,
                    "user_recipe_selection": recipe_num,
                    "current_stage": "customization",
                    "coordination_log": [f"LLM-detected selection: recipe #{recipe_num}"]
                }
            else:
                # No valid context — fall to general for a "please pick 1-3" reply
                query_type = "general"

        # Fallback: bare digit while in presenting_options OR awaiting_selection
        if (user_msg.strip() in ("1", "2", "3")
                and state.get("current_stage") in ("presenting_options", "awaiting_selection")
                and state.get("top_3_recommendations")):
            num = int(user_msg.strip())
            print(f"✅ [ORCHESTRATOR] Bare digit selection: recipe #{num}")
            return {
                "query_type": "selection",
                "user_preferences": updated_prefs,
                "user_recipe_selection": num,
                "current_stage": "customization",
                "coordination_log": [f"Bare digit selection: recipe #{num}"]
            }

        # Context guard: while recipe cards are on screen, keep follow-up questions
        # in Path B (recipe Q&A) rather than triggering a whole new search.
        # Only break out of context when user clearly wants a fresh search.
        if (current_stage == "presenting_options"
                and state.get("top_3_recommendations")
                and query_type == "recipe"):
            is_new_search = combined.get("is_new_recipe_search", False)
            NEW_SEARCH_SIGNALS = [
                "search for", "find me", "search again", "different recipe",
                "something else", "other recipe", "start over", "new recipe",
                "look for something", "try something else", "show me other",
                "different options", "other options",
            ]
            has_new_search_signal = any(sig in user_msg.lower() for sig in NEW_SEARCH_SIGNALS)
            if not is_new_search and not has_new_search_signal:
                print("🔄 [ORCHESTRATOR] Recipe follow-up while presenting options — routing to Q&A (Path B)")
                query_type = "general"

        print(f"📋 [ORCHESTRATOR] Query type: {query_type}")
        print(f"👤 [ORCHESTRATOR] Preferences: {updated_prefs}")

        return {
            "query_type": query_type,
            "user_preferences": updated_prefs,
            "preference_action": combined.get("preference_action"),
            "current_stage": f"routing_to_{query_type}",
            "coordination_log": [f"Orchestrator classified as: {query_type}"]
        }

    def _route_from_orchestrator(self, state: RecipeWorkflowState) -> str:
        """Decide which node to route to based on query type"""
        query_type = state.get("query_type", "general")

        # If user selected a recipe, go to customization
        if state.get("user_recipe_selection"):
            return "selection"

        # Route based on query type
        routing = {
            "pantry": "pantry",
            "ingredient": "pantry",
            "recipe": "recipe",
            "selection": "selection",   # direct to customization node
            "off_topic": "general",     # politely declined in general_response node
            "preference": "general",    # preference mgmt handled in general_response Path D
            "general": "general"
        }

        return routing.get(query_type, "general")

    # ============================================
    # NODE 2: PANTRY (Pantry Agent)
    # ============================================

    def _pantry_node(self, state: RecipeWorkflowState) -> Dict[str, Any]:
        """
        Handle pantry operations: add/update/remove ingredients.
        Uses PantryAgent's natural language handler for intelligent operation detection.
        """
        print("\n🥬 [PANTRY] Processing inventory operation...")

        user_msg = state.get("user_message", "")
        conversation_history = state.get("messages", [])

        import asyncio
        result = asyncio.run(self.pantry.handle_query(user_msg, conversation_history=conversation_history))

        # Error (e.g., non-food item rejection)
        if isinstance(result, dict) and result.get("error") and not result.get("needs_clarification"):
            error_msg = result.get("error")
            print(f"❌ [PANTRY] Error: {error_msg}")
            return {
                "response": error_msg,
                "current_stage": "error",
                "coordination_log": [f"Error: {error_msg}"]
            }

        # User cancelled during clarification ("never mind", topic switch)
        if isinstance(result, dict) and result.get("cancelled"):
            self.pantry.pending_items = []
            msg = result.get("message", "No worries! What else can I help with?")
            print(f"🚪 [PANTRY] User cancelled clarification")
            return {
                "response": msg,
                "current_stage": "pantry_complete",
                "coordination_log": ["User cancelled quantity clarification"]
            }

        # Clarification needed (ask_for_quantity was called)
        if isinstance(result, dict) and result.get("needs_clarification"):
            pending_items = result.get("pending_items", [])
            error_msg = result.get("error")

            if error_msg:
                response = f"❓ {error_msg}"
            else:
                response = self._generate_quantity_question(pending_items)

            print(f"❓ [PANTRY] Asking for clarification: {pending_items}")

            return {
                "response": response,
                "current_stage": "awaiting_quantity_clarification",
                "coordination_log": [f"Awaiting quantity for: {', '.join(pending_items)}"]
            }

        # Extract operations and typed result from the new return format
        if isinstance(result, dict) and "operations" in result:
            operations = result["operations"]
            typed_result = result["result"]
        else:
            operations = []
            typed_result = result

        # Get updated inventory
        async def _fetch_pantry_state():
            inv = await self.pantry._get_inventory_async()
            exp = await self.pantry._get_expiring_soon_async(days_threshold=3)
            return inv, exp

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _fetch_pantry_state())
                inventory, expiring = future.result()
        else:
            inventory, expiring = asyncio.run(_fetch_pantry_state())

        response = self._format_pantry_response_smart(typed_result, inventory, expiring, operations)

        print(f"✅ [PANTRY] Updated inventory: {len(inventory)} items")

        next_stage = (
            "presenting_options"
            if state.get("top_3_recommendations")
            else "pantry_complete"
        )

        return {
            "pantry_inventory": inventory,
            "expiring_items": expiring,
            "response": response,
            "current_stage": next_stage,
            "coordination_log": [f"Pantry updated via natural language"]
        }

    def _generate_quantity_question(self, items: List[str]) -> str:
        """
        Generate a natural question asking for quantities of items.

        Args:
            items: List of item names

        Returns:
            Question string
        """
        if not items:
            return "❓ How many items do you have?"

        if len(items) == 1:
            return f"❓ How many {items[0]}{'s' if not items[0].endswith('s') else ''} do you have?"

        elif len(items) == 2:
            items_str = f"{items[0]} and {items[1]}"
            return f"❓ How many {items_str} do you have?"

        else:
            # More than 2 items
            items_str = ", ".join(items[:-1]) + f", and {items[-1]}"
            return f"❓ How many of each do you have? ({items_str})"

    def _format_pantry_response_smart(self, result, inventory: List, expiring: List, operations: List[Dict]) -> str:
        """
        Format pantry operation result based on actual operations performed.

        Args:
            result: PantryItemsResponse from handle_query()
            inventory: Current inventory
            expiring: Expiring items
            operations: List of operation dicts from handle_query, e.g. [{"type": "add_food_item", "item": "egg", "quantity": 3}]

        Returns:
            Formatted response string
        """
        response = ""
        op_types = {op["type"] for op in operations} if operations else set()

        if "clear_pantry" in op_types:
            count = next((op["quantity"] for op in operations if op["type"] == "clear_pantry"), 0)
            if count:
                response = f"✅ I've cleared your pantry and removed {count} items.\n\n"
            else:
                response = "✅ Your pantry is now empty.\n\n"

        elif "delete_food_item" in op_types:
            names = [op["item"] for op in operations if op["type"] == "delete_food_item"]
            response = f"✅ I've removed {', '.join(names)} from your pantry.\n\n"

        elif "adjust_food_quantity" in op_types:
            names = [op["item"] for op in operations if op["type"] == "adjust_food_quantity"]
            response = f"✅ I've adjusted {', '.join(names)} in your pantry.\n\n"

        elif "set_food_quantity" in op_types:
            descs = [f"{op['quantity']} {op['item']}" for op in operations if op["type"] == "set_food_quantity"]
            response = f"✅ I've updated your pantry: {', '.join(descs)}.\n\n"

        elif "add_food_item" in op_types:
            descs = [f"{op['quantity']} {op['item']}" for op in operations if op["type"] == "add_food_item"]
            response = f"✅ I've added {', '.join(descs)} to your pantry.\n\n"

        elif "get_all_food_items" in op_types or "get_expiring_soon" in op_types:
            response = ""

        else:
            response = "✅ I've updated your pantry.\n\n"

        if len(inventory) == 0:
            response += "📦 **Your pantry is now empty.**"
        else:
            response += f"📦 **Your pantry now has {len(inventory)} items.**"

        if expiring:
            response += f"\n⚠️  {len(expiring)} items expiring soon: "
            response += ", ".join([item.get("ingredient_name", item.get("name", "")) for item in expiring[:3]])

        return response

    def _format_pantry_response(self, added_items: List[str], inventory: List, expiring: List) -> str:
        """Format pantry operation result for user (legacy)"""
        response = ""

        if added_items:
            response = f"✅ I've added {', '.join(added_items)} to your pantry.\n\n"
        else:
            response = "✅ I've updated your pantry.\n\n"

        response += f"📦 **Your pantry now has {len(inventory)} items.**"

        if expiring:
            response += f"\n⚠️  {len(expiring)} items expiring soon: "
            response += ", ".join([item.get("ingredient_name", item.get("name", "")) for item in expiring[:3]])

        return response

    # ============================================
    # NODE 3: RECIPE SEARCH (Recipe Knowledge Agent)
    # ============================================

    def _recipe_search_node(self, state: RecipeWorkflowState) -> Dict[str, Any]:
        """
        Search for recipes using hybrid search.
        Returns top-k results (e.g., 10 recipes).
        """
        print("\n🔍 [RECIPE SEARCH] Searching for recipes...")

        if not self._ensure_recipe_agent_ready():
            return {
                "response": "⚠️  Recipe search is not available. Please run the ingestion script first.",
                "current_stage": "error",
                "coordination_log": ["Recipe search unavailable - no data loaded"]
            }

        user_msg = state.get("user_message", "")
        preferences = state.get("user_preferences", {})
        inventory = state.get("pantry_inventory", [])

        # Extract pantry items as ingredient names
        pantry_items = [item.get("ingredient_name", item.get("name", "")) for item in inventory] if inventory else None

        # Perform hybrid query (keyword + semantic)
        try:
            # hybrid_query returns list of (recipe_metadata, score, num_used, missing)
            results = self.recipe_agent.hybrid_query(
                pantry_items=pantry_items,
                query_text=user_msg,
                top_k=10,
                allow_missing=2,
                use_semantic=True
            )

            # Extract recipe metadata from results
            recipe_results = [recipe_meta for recipe_meta, score, num_used, missing in results]

            print(f"✅ [RECIPE SEARCH] Found {len(recipe_results)} recipes")

            return {
                "recipe_results": recipe_results,
                "current_stage": "recipe_search_complete",
                "coordination_log": [f"Found {len(recipe_results)} recipes via hybrid search"]
            }

        except Exception as e:
            print(f"❌ [RECIPE SEARCH] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"❌ Sorry, I encountered an error searching recipes: {str(e)}",
                "current_stage": "error",
                "coordination_log": [f"Recipe search error: {str(e)}"]
            }

    # ============================================
    # NODE 4: RECOMMENDATION (Sous Chef - Rank)
    # ============================================

    def _recommendation_node(self, state: RecipeWorkflowState) -> Dict[str, Any]:
        """
        Sous Chef analyzes top-k recipes and selects best 3.
        Considers: ingredient match, expiring items, user skill level.
        """
        print("\n👨‍🍳 [RECOMMENDATION] Sous Chef selecting top 3...")

        recipe_results = state.get("recipe_results", [])
        preferences = state.get("user_preferences", {})
        inventory = state.get("pantry_inventory", [])
        expiring = state.get("expiring_items", [])

        if not recipe_results:
            if not inventory:
                return {
                    "response": (
                        "Your pantry is empty, so I can't match recipes to your ingredients yet. "
                        "Try telling me what you have — for example: "
                        "*\"I have chicken, garlic, and pasta\"* — and I'll suggest recipes right away!"
                    ),
                    "current_stage": "no_results",
                    "coordination_log": ["No recipes found — pantry empty"]
                }
            return {
                "response": "😕 I couldn't find any recipes matching your criteria. Try broadening your search or adding more ingredients to your pantry.",
                "current_stage": "no_results",
                "coordination_log": ["No recipes found to recommend"]
            }

        # Use Sous Chef's existing generate_recommendations method with creative LLM
        pantry_summary = {
            "inventory": inventory,
            "total_ingredients": len(inventory)
        }

        # Soft notice when pantry is empty but semantic search still returned results
        empty_pantry_note = ""
        if not inventory:
            empty_pantry_note = (
                "\n\n💡 *Your pantry is empty — these are general suggestions. "
                "Tell me what ingredients you have and I'll find recipes that use them!*"
            )

        top_3 = self.sous_chef.generate_recommendations(
            llm=llm_creative,  # Use creative LLM for recommendations
            pantry_summary=pantry_summary,
            user_preferences=preferences,
            expiring_items=expiring,
            recipe_results=recipe_results  # Pass the search results
        )

        # Format response
        response = self._format_recommendations(top_3, expiring) + empty_pantry_note

        print(f"✅ [RECOMMENDATION] Selected top 3 recipes")

        return {
            "top_3_recommendations": top_3,
            "response": response,
            "current_stage": "presenting_options",
            "coordination_log": [f"Sous Chef recommended {len(top_3)} recipes"]
        }

    def _format_preferences_response(self, prefs: Dict[str, Any], user_msg: str, preference_action: Optional[str] = None) -> str:
        """
        Format a human-readable summary of current preferences for display.
        Used by _general_response_node Path D (preference management queries).
        Shows what changed when edits were made, or just displays current state.
        """
        allergies = prefs.get("allergies") or []
        restrictions = prefs.get("restrictions") or []
        cuisines = prefs.get("cuisines") or []
        diet = prefs.get("diet")
        skill = prefs.get("skill")

        is_empty = not any([allergies, restrictions, cuisines, diet, skill])

        # Detect if the user was asking to see preferences vs making a change
        view_signals = ["show", "what are", "display", "list", "tell me", "my preference", "what's my", "my setting"]
        is_view_query = (preference_action == "view") or any(sig in user_msg.lower() for sig in view_signals)

        if is_empty:
            header = "You don't have any preferences saved yet."
            tip = " Tell me your allergies, preferred cuisines, or dietary needs and I'll remember them!"
            return f"⚙️ **Your Preferences**\n\n{header}{tip}"

        lines = ["⚙️ **Your Current Preferences**\n"]

        if allergies:
            lines.append(f"🚫 **Allergies:** {', '.join(a.title() for a in allergies)}")
        if restrictions:
            lines.append(f"⛔ **Dietary Restrictions:** {', '.join(r.title() for r in restrictions)}")
        if cuisines:
            lines.append(f"🌍 **Preferred Cuisines:** {', '.join(c.title() for c in cuisines)}")
        if diet:
            lines.append(f"🥗 **Diet:** {diet.title()}")
        if skill:
            lines.append(f"👨‍🍳 **Skill Level:** {skill.title()}")

        if not is_view_query:
            lines.append("\n✅ Preferences updated! These will be applied to your next recipe search.")
        else:
            lines.append("\n💡 You can update these anytime — just tell me what to change or remove.")

        return "\n".join(lines)

    def _format_recommendations(self, top_3: List[Dict], expiring: List) -> str:
        """Format top 3 recommendations for user"""
        response = "🍽️ **Here are my top 3 recipe recommendations:**\n\n"

        for i, recipe in enumerate(top_3, 1):
            response += f"**{i}. {recipe.get('title', 'Unknown Recipe')}**\n"

            # Show ingredient count
            ingredients = recipe.get('ner', []) or recipe.get('ingredients', [])
            if ingredients:
                response += f"   🥘 {len(ingredients)} ingredients\n"

            # Show timing and servings
            ready_time = recipe.get('readyInMinutes', 'N/A')
            servings = recipe.get('servings', 'N/A')
            if ready_time != 'N/A' or servings != 'N/A':
                response += f"   ⏱️ {ready_time} min | 👥 {servings} servings\n"

            # Show match percentage if available
            match_pct = recipe.get("match_percentage", recipe.get("score", 0))
            if match_pct:
                response += f"   🎯 {match_pct}% ingredient match\n"

            # Show recipe link
            link = recipe.get('link', '')
            if link:
                # Make sure link has protocol
                if not link.startswith('http'):
                    link = f"https://{link}"
                response += f"   🔗 [View Recipe]({link})\n"

            # Show why recommended
            reason = recipe.get("recommendation_reason", recipe.get("reasoning", "Great recipe!"))
            response += f"   💡 {reason}\n\n"

        if expiring:
            expiring_names = [item.get('ingredient_name') or item.get('name', '') for item in expiring[:3]]
            response += f"\n⚠️  Using expiring items: {', '.join(expiring_names)}"

        response += "\n\n✨ **Which recipe would you like to try?** (Reply with 1, 2, or 3)"

        return response

    # ============================================
    # NODE 6: GENERAL RESPONSE (Executive Chef)
    # ============================================

    def _customization_node(self, state: RecipeWorkflowState) -> Dict[str, Any]:
        """
        Sous Chef adapts selected recipe to user's pantry and preferences.
        Handles substitutions, adjustments, and formatting.
        """
        print("\n🎨 [CUSTOMIZATION] Adapting recipe...")

        selection = state.get("user_recipe_selection")
        top_3 = state.get("top_3_recommendations", [])
        inventory = state.get("pantry_inventory", [])
        preferences = state.get("user_preferences", {})

        if not selection or not top_3:
            return {
                "response": "😕 I need you to select a recipe first (1, 2, or 3).",
                "current_stage": "awaiting_selection",
                "coordination_log": ["No recipe selection provided"]
            }

        # Validate selection
        if selection < 1 or selection > len(top_3):
            return {
                "response": f"Please select a valid option (1-{len(top_3)}).",
                "current_stage": "awaiting_selection",
                "coordination_log": ["Invalid recipe selection"]
            }

        # Get selected recipe
        selected = top_3[selection - 1]

        # Use Sous Chef's existing adapt_recipe method with creative LLM
        customized = self.sous_chef.adapt_recipe(
            llm=llm_creative,  # Use creative LLM for recipe adaptation
            recipe=selected,
            user_preferences=preferences,
            pantry_inventory=inventory
        )

        # Format final recipe using existing method
        formatted = self.sous_chef.format_recipe_for_user(customized, preferences)

        print(f"✅ [CUSTOMIZATION] Recipe customized: {selected.get('title', 'Unknown')}")

        return {
            "selected_recipe_data": selected,
            "customized_recipe": customized,
            "response": formatted,
            "current_stage": "final_recipe",
            "coordination_log": [f"Customized recipe #{selection}: {selected.get('title', 'Unknown')}"]
        }

    # ============================================
    # NODE 6: GENERAL RESPONSE (Executive Chef)
    # ============================================

    def _general_response_node(self, state: RecipeWorkflowState) -> Dict[str, Any]:
        """
        Handle general conversation - three paths:
          A) off_topic: politely decline and redirect
          B) presenting_options + top_3: recipe Q&A with context preservation
          C) default: waiter-mode general response
        """
        print("\n💬 [GENERAL] Handling general query...")

        user_msg = state.get("user_message", "")
        current_stage = state.get("current_stage", "")
        top_3 = state.get("top_3_recommendations", [])
        query_type = state.get("query_type", "general")

        # Path A: Off-topic — politely decline
        if query_type == "off_topic":
            print("🚫 [GENERAL] Off-topic query — declining")
            response = (
                "I'm your kitchen assistant! I can help you manage your pantry, "
                "find recipes based on what you have, and guide you through cooking. "
                "I'm not able to help with that, but feel free to ask me anything food-related!"
            )
            return {
                "response": response,
                "current_stage": current_stage,   # don't disturb existing flow
                "coordination_log": ["Off-topic query declined"]
            }

        # Path B: Recipe browsing Q&A — stay in context
        if current_stage == "presenting_options" and top_3:
            print("💬 [GENERAL] Recipe Q&A while browsing options")
            result = self.sous_chef.converse_about_recommendations(
                llm_creative,
                top_3,
                user_msg,
                state.get("user_preferences", {})
            )
            detected_selection = result.get("selection")
            if detected_selection and 1 <= detected_selection <= len(top_3):
                return {
                    "response": result.get("reply", ""),
                    "top_3_recommendations": top_3,
                    "user_recipe_selection": detected_selection,
                    "current_stage": "customization",
                    "coordination_log": [f"Recipe Q&A; selection {detected_selection} detected"]
                }
            return {
                "response": result.get("reply", ""),
                "top_3_recommendations": top_3,   # preserve cards in Streamlit
                "current_stage": "presenting_options",
                "coordination_log": ["Recipe Q&A, no selection"]
            }

        # Path D: Preference management — show current prefs or confirm changes
        if query_type == "preference":
            print("⚙️  [GENERAL] Preference management query")
            prefs = state.get("user_preferences", {})
            preference_action = state.get("preference_action")
            response = self._format_preferences_response(prefs, user_msg, preference_action)
            return {
                "response": response,
                "current_stage": current_stage,   # don't disturb existing flow
                "coordination_log": ["Preference management query handled"]
            }

        # Path C: Default general response (waiter mode)
        response = self.exec_chef.respond_as_waiter(llm, user_msg)

        print(f"✅ [GENERAL] Responded to general query")

        return {
            "response": response,
            "current_stage": "general_complete",
            "coordination_log": ["Handled general query"]
        }

    # ============================================
    # PUBLIC INTERFACE
    # ============================================

    def get_current_inventory(self) -> List[Dict[str, Any]]:
        """
        Get current pantry inventory from MCP database.
        Used for initialization and UI updates.

        Returns:
            List of dicts with keys: id, name, quantity, expire_date
        """
        try:
            return self.pantry.get_inventory()
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch pantry inventory: {e}")
            return []

    async def ainvoke(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async invoke the workflow.
        Called by Streamlit frontend.
        """
        # Ensure required fields
        if "user_message" not in input_state:
            raise ValueError("user_message is required")

        # Initialize state if needed
        if "coordination_log" not in input_state:
            input_state["coordination_log"] = []
        if "current_stage" not in input_state:
            input_state["current_stage"] = "initial"

        # Run workflow
        result = await self.graph.ainvoke(input_state)

        return result

    def invoke(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync invoke the workflow.
        """
        # Ensure required fields
        if "user_message" not in input_state:
            raise ValueError("user_message is required")

        # Initialize state if needed
        if "coordination_log" not in input_state:
            input_state["coordination_log"] = []
        if "current_stage" not in input_state:
            input_state["current_stage"] = "initial"

        # Run workflow
        result = self.graph.invoke(input_state)

        return result


# ============================================
# INITIALIZATION
# ============================================

def create_workflow() -> LeftovrWorkflow:
    """Create and return the workflow instance"""
    return LeftovrWorkflow()


# For testing
if __name__ == "__main__":
    print("🚀 Initializing Leftovr Workflow...")
    workflow = create_workflow()

    # Test 1: Pantry addition
    print("\n" + "="*60)
    print("TEST 1: Add pantry items")
    print("="*60)
    result1 = workflow.invoke({
        "user_message": "I have 2 chicken breasts, tomatoes, and pasta",
        "user_preferences": {},
        "pantry_inventory": []
    })
    print(f"\n📤 Response:\n{result1.get('response', 'No response')}")

    # Test 2: Recipe search
    print("\n" + "="*60)
    print("TEST 2: Search recipes")
    print("="*60)
    result2 = workflow.invoke({
        "user_message": "What can I make? I'm vegetarian",
        "user_preferences": {"dietary_restrictions": ["vegetarian"]},
        "pantry_inventory": result1.get("pantry_inventory", [])
    })
    print(f"\n📤 Response:\n{result2.get('response', 'No response')}")
