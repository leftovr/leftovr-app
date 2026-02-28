#!/usr/bin/env python3
# ============================================================================
# PANTRY AGENT (MCP Client)
# ============================================================================
#
# This module provides the PantryAgent that IS an MCP client.
# It manages pantry operations through proper MCP client-server communication.
#
# Architecture:
#   PantryAgent (MCP Client) → [JSON-RPC/stdio] → MCP Server → Database
#
# The agent has NO direct access to the database and must communicate
# through the MCP protocol for all operations.
#
# ============================================================================

import asyncio
import json
import os
import subprocess
import sys
import threading
import queue
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import openai
import inflect
from pydantic import BaseModel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")

# Inflect engine for pluralization
p = inflect.engine()


def normalize_food_id(name: str) -> str:
    """
    Normalize a food name for deterministic IDs:
    - singularize
    - lowercase
    - strip spaces
    - replace spaces with hyphens
    """
    if not name:
        return ""
    singular = p.singular_noun(name)  # Returns False if already singular
    singular_name = singular if singular else name
    return singular_name.lower().strip().replace(' ', '-')


class SingleItemResponse(BaseModel):
    """Represents a single food item in the pantry."""
    id: str
    name: str
    quantity: int
    expire_date: str  # Keep as string "YYYY-MM-DD"


class PantryItemsResponse(BaseModel):
    """Represents multiple food items in the pantry."""
    items: List[SingleItemResponse]


def convert_items(raw_items: list) -> PantryItemsResponse:
    """
    Converts a list of raw food items (dicts) into strongly typed PantryItemsResponse.
    """
    typed_items = [SingleItemResponse(**item) for item in raw_items]
    return PantryItemsResponse(items=typed_items)


class PantryAgent:
    """
    PantryAgent - MCP Client for pantry operations.

    This agent IS an MCP client that communicates with the MCP server via JSON-RPC.

    Architecture (Proper MCP):
    ┌──────────────────────────────────────────┐
    │         PantryAgent                      │
    │  (IS THE MCP CLIENT - No DB Access)      │
    │                                          │
    │  Methods:                                │
    │  • add_or_update_ingredient()            │
    │  • get_inventory()                       │
    │  • remove_ingredient()                   │
    │  • handle_query() [AI-powered]           │
    │                                          │
    │  Communicates via JSON-RPC ↓             │
    └──────────────┬───────────────────────────┘
                   ↓ (JSON-RPC via stdio)
    ┌──────────────────────────────────────────┐
    │        MCP Server                         │
    │    (Separate Process - Owns Database)     │
    │                                          │
    │  Tools:                                  │
    │  • add_food_item                         │
    │  • get_all_food_items                    │
    │  • update_food_item                      │
    │  • delete_food_item                      │
    │  • get_expiring_soon                     │
    └──────────────┬───────────────────────────┘
                   ↓
    ┌──────────────────────────────────────────┐
    │        PantryDatabase                     │
    │        (SQLite - single source of truth)  │
    └──────────────────────────────────────────┘

    Note: MCP server runs as a separate process and can be accessed by
    multiple clients (Claude Desktop, this agent, external tools, etc.)
    """

    def __init__(self, name: str = "Pantry Agent", server_script_path: Optional[str] = None):
        """
        Initialize the PantryAgent as an MCP client.

        Args:
            name: Name of the agent
            server_script_path: Path to MCP server script (auto-detected if None)
        """
        self.name = name

        # MCP client state - NO direct database access!
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._response_queue = queue.Queue()
        self._reader_thread = None
        self._connected = False

        # Conversation state for multi-turn clarifications
        self.pending_items = []  # Items waiting for quantity clarification

        # Determine server script path
        if server_script_path is None:
            # Auto-detect: assume we're in agents/, server is in mcp/server.py
            current_dir = Path(__file__).parent.parent
            server_script_path = str(current_dir / "mcp" / "server.py")

        self.server_script_path = server_script_path

        # OpenAI client for natural language interpretation
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

        self.system_prompt = """You are an expert Food Pantry AI Assistant with access to real-time inventory data.

Your role:
- Help users manage pantry inventory (FOOD ITEMS ONLY).
- Interpret natural language statements about food usage, consumption, or restocking.
- Use your natural language understanding to parse user input.
- Handle filler phrases like "as well", "too", "also" naturally.
- Extract food items and their quantities from context.

Tools you can use:
- get_all_food_items: View current inventory
- get_expiring_soon: Get items expiring within N days
- add_food_item: Add NEW items to the pantry (item does NOT already exist)
- set_food_quantity: Set the EXACT quantity for an item that ALREADY exists
- adjust_food_quantity: Adjust quantity by delta (positive/negative)
- delete_food_item: Remove a specific item completely
- clear_pantry: Delete ALL items from the pantry at once
- ask_for_quantity: Ask user to clarify quantity when ambiguous

Guidelines:
- ONLY accept FOOD and BEVERAGE items. Reject non-food items (laptop, book, phone, etc.).
- If the user mentions multiple items in one statement, produce one tool call per item.
- For new items without expire_date, a 14-day default will be assigned.

SEMANTIC GUIDANCE — ADDING ITEMS:
- "I have X [item]" / "I bought X" / "add X" / "got X" when item is NOT in pantry:
  * WITH explicit number ("I have 2 chickens") → add_food_item with quantity
  * WITH "a/an" article ("I have a tomato") → add_food_item with quantity=1
  * PLURAL without number ("I have oysters") → ask_for_quantity
  * SINGULAR without "a/an" ("I have tomato") → ask_for_quantity (ambiguous)
  * UNCOUNTABLE nouns ("milk", "rice", "flour") → ask_for_quantity
- "I have X [item]" when item ALREADY EXISTS in pantry → set_food_quantity
  (user is stating their CURRENT total, not adding more)
- "I bought X more" / "I got X more" / "add X more" → add_food_item (explicit addition)

SEMANTIC GUIDANCE — UPDATING ITEMS:
- "Set to X" / "Update to X" / "Change to X" → set_food_quantity
- "I have X [item]" when item exists → set_food_quantity (restating current amount)

SEMANTIC GUIDANCE — REMOVING ITEMS:
- "Remove X" / "Delete X" / "Toss X" / "Throw away X" / "Threw away X" /
  "Get rid of X" / "Ditch X" / "I don't have X anymore" / "Clear all the X"
  → delete_food_item (remove that SPECIFIC item)
- "Remove N X" / "Take out N X" / "Use N X" (WITH quantity)
  → adjust_food_quantity with negative delta

SEMANTIC GUIDANCE — CONSUMPTION:
- "I ate N X" / "I used N X" / "consumed N X" (WITH quantity)
  → adjust_food_quantity with negative delta
- "I ate the X" / "I cooked with the X" / "consumed the X" (WITHOUT quantity)
  → delete_food_item (assume ALL was used)

SEMANTIC GUIDANCE — CLEARING PANTRY:
- "Clear my pantry" / "Clear everything" / "Empty my pantry" /
  "Delete all items" / "Remove everything"
  → clear_pantry (empties the ENTIRE pantry)

⚠️ CRITICAL DISTINCTION — CLEAR ITEM vs CLEAR PANTRY:
  "clear all the eggs" → delete_food_item("egg")  [removes a SPECIFIC item]
  "clear the chicken" → delete_food_item("chicken")  [removes a SPECIFIC item]
  "clear everything" / "clear my pantry" → clear_pantry  [empties ENTIRE pantry]
  The word "clear" followed by a specific food item = delete that item, NOT clear_pantry.

CONTEXT-DEPENDENT REFERENCES (use conversation history):
- When the user says something like "nvm i have 20 in hand", "actually there are 15",
  "make that 20", "no wait, 5" WITHOUT naming a specific food item, look at the
  conversation history to determine which item they are referring to. Use the most
  recently discussed food item as the referent.
- "nvm i have 20 in hand" after discussing eggs → set_food_quantity(name="egg", quantity=20)
- "actually there are 15" after adding tomatoes → set_food_quantity(name="tomato", quantity=15)
- "make that 5" after adding chicken → set_food_quantity(name="chicken", quantity=5)
- The prefix "nvm" / "never mind" means the user is CORRECTING their previous statement,
  not cancelling. Resolve the item from context and call the appropriate tool.

QUANTITY CLARIFICATION RULES:
1. "a/an [item]" = quantity 1
2. PLURAL without numbers = ask_for_quantity
3. SINGULAR without "a/an" = ask_for_quantity (ambiguous)
4. UNCOUNTABLE nouns = ask_for_quantity
5. EXPLICIT numbers = use that quantity
6. When user responds with numbers after being asked → add_food_item with that quantity

FOOD VALIDATION RULES:
1. ONLY accept food and beverage items
2. REJECT non-food items (laptop, book, phone, shirt, car, furniture, electronics)
3. If user tries to add non-food items, DO NOT call any tools — respond explaining this is a food pantry

Examples:
✅ CORRECT:
- "I have 2 eggs" (NEW) → add_food_item(name="egg", quantity=2)
- "I have 11 eggs" (egg ALREADY in pantry with qty 1) → set_food_quantity(name="egg", quantity=11)
- "I have a tomato" → add_food_item(name="tomato", quantity=1)
- "I have oysters" → ask_for_quantity(items=["oyster"])
- "I have tomato" → ask_for_quantity(items=["tomato"])
- "I have milk" → ask_for_quantity(items=["milk"])
- "clear all the eggs" → delete_food_item(name="egg")
- "toss the milk" → delete_food_item(name="milk")
- "I threw away the bread" → delete_food_item(name="bread")
- "I cooked with the chicken" → delete_food_item(name="chicken")
- "I ate 2 apples" → adjust_food_quantity(name="apple", quantity=-2)
- "clear my pantry" → clear_pantry

❌ WRONG (NEVER DO THIS):
- "I have 11 eggs" (egg exists) → add_food_item(quantity=11) [BAD! Would increment, not set]
- "clear all the eggs" → clear_pantry [BAD! User wants to remove eggs, not empty pantry]
- "I have tomato" → add_food_item(quantity=1) [BAD! No "a/an", ambiguous]
- "I have a laptop" → add_food_item [BAD! Not food]

Always respond with structured tool calls when users want to modify inventory.
Only accept food and beverage items in the pantry."""

    # ============================================
    # MCP CLIENT IMPLEMENTATION
    # Core protocol communication methods
    # ============================================

    def _run_sync(self, coro):
        """
        Helper to run async coroutines synchronously.
        Handles both cases: running event loop and no event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(coro)
        else:
            # Already in a loop - use run_in_executor with a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()

    def _read_responses(self):
        """Background thread to read JSON-RPC responses from server"""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break

                line = line.decode('utf-8').strip()
                if line:
                    try:
                        response = json.loads(line)
                        self._response_queue.put(response)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse response: {e}")
        except Exception as e:
            print(f"Error reading responses: {e}")

    async def ensure_connected(self):
        """Connect to MCP server by starting it as subprocess"""
        if self._connected:
            return

        try:
            # Start server as subprocess
            self.process = subprocess.Popen(
                [sys.executable, self.server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            # Start background thread to read responses
            self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
            self._reader_thread.start()

            # Give server a moment to start
            await asyncio.sleep(0.1)

            self._connected = True
            print(f"✅ {self.name} connected to MCP server")

        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {str(e)}")
            raise

    async def disconnect(self):
        """Disconnect from MCP server and cleanup"""
        if not self._connected:
            return

        try:
            if self.process:
                # Close stdin to signal server to shut down
                if self.process.stdin:
                    self.process.stdin.close()

                # Wait for process to terminate
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

                self.process = None

            self._connected = False
            print(f"👋 {self.name} disconnected from MCP server")

        except Exception as e:
            print(f"Error disconnecting: {str(e)}")

    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send JSON-RPC request to server and wait for response.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            Result dictionary
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Not connected to MCP server. Call ensure_connected() first.")

        try:
            # Generate request
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params or {}
            }

            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line.encode('utf-8'))
            self.process.stdin.flush()

            # Wait for response (with timeout)
            timeout = 5.0
            start_time = asyncio.get_event_loop().time()

            while True:
                try:
                    response = self._response_queue.get(timeout=0.1)
                    if response.get("id") == self._request_id:
                        if "error" in response:
                            error = response["error"]
                            raise RuntimeError(f"Server error: {error.get('message', 'Unknown error')}")
                        return response.get("result", {})
                    else:
                        # Put it back if it's not our response
                        self._response_queue.put(response)
                except queue.Empty:
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        raise TimeoutError(f"No response received for request {self._request_id}")
                    await asyncio.sleep(0.05)

        except Exception as e:
            print(f"Error sending request: {str(e)}")
            raise

    # ============================================
    # METHODS USING MCP CLIENT
    # All operations go through proper MCP protocol
    # ============================================

    def add_or_update_ingredient(
        self,
        ingredient_name: str,
        quantity: int,
        unit: str = "pieces",
        expire_date: str = None
    ) -> Dict[str, Any]:
        """
        Add or update an ingredient in the pantry via MCP.

        Args:
            ingredient_name: Name of the ingredient
            quantity: Quantity to add (will increment if exists)
            unit: Unit of measurement (stored as metadata, currently unused)
            expire_date: Optional expiration date in YYYY-MM-DD format

        Returns:
            Dict with success status, item_id, action, and quantity
        """
        return self._run_sync(self._add_or_update_ingredient_async(
            ingredient_name, quantity, unit, expire_date
        ))

    async def _add_or_update_ingredient_async(
        self,
        ingredient_name: str,
        quantity: int,
        unit: str = "pieces",
        expire_date: str = None
    ) -> Dict[str, Any]:
        """Async implementation of add_or_update_ingredient"""
        await self.ensure_connected()

        try:
            # Generate deterministic ID
            item_id = normalize_food_id(ingredient_name)

            # Default expiry date if not provided (14 days from now)
            exp_date = expire_date
            if not exp_date:
                exp_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

            # Call through MCP protocol
            args = {
                "name": ingredient_name,
                "quantity": quantity,
                "expire_date": exp_date
            }
            result = await self._send_request("tools/call", {
                "name": "add_food_item",
                "arguments": args
            })

            # Return consistent format
            if result.get("success"):
                return {
                    "success": True,
                    "item_id": item_id,
                    "action": result.get("action", "added"),
                    "quantity": quantity,
                    "data": result.get("data")
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error adding ingredient"),
                    "item_id": item_id
                }
        except Exception as e:
            print(f"❌ Error in add_or_update_ingredient: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "item_id": item_id if 'item_id' in locals() else None
            }

    def get_inventory(self) -> List[Dict[str, Any]]:
        """
        Get all pantry items via MCP.

        Returns:
            List of dicts with keys: id, name, quantity, expire_date
        """
        return self._run_sync(self._get_inventory_async())

    async def _get_inventory_async(self) -> List[Dict[str, Any]]:
        """Async implementation of get_inventory"""
        await self.ensure_connected()

        try:
            result = await self._send_request("tools/call", {
                "name": "get_all_food_items",
                "arguments": {}
            })

            if result.get("success"):
                return result.get("data", [])
            else:
                print(f"⚠️  Error getting inventory: {result.get('error')}")
                return []
        except Exception as e:
            print(f"❌ Error in get_inventory: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def get_expiring_soon(self, days_threshold: int = 7) -> List[Dict[str, Any]]:
        """
        Get items expiring within the specified number of days via MCP.

        Args:
            days_threshold: Number of days to check (default: 7)

        Returns:
            List of dicts containing expiring food items
        """
        return self._run_sync(self._get_expiring_soon_async(days_threshold))

    async def _get_expiring_soon_async(self, days_threshold: int = 7) -> List[Dict[str, Any]]:
        """Async implementation of get_expiring_soon"""
        await self.ensure_connected()

        try:
            result = await self._send_request("tools/call", {
                "name": "get_expiring_soon",
                "arguments": {"days": days_threshold}
            })

            if result.get("success"):
                return result.get("data", [])
            else:
                print(f"⚠️  Error getting expiring items: {result.get('error')}")
                return []
        except Exception as e:
            print(f"❌ Error in get_expiring_soon: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def remove_ingredient(self, ingredient_id: str) -> Dict[str, Any]:
        """
        Remove an ingredient from the pantry via MCP.

        Args:
            ingredient_id: ID or name of the ingredient to remove

        Returns:
            Dict with success status and item_id
        """
        return self._run_sync(self._remove_ingredient_async(ingredient_id))

    async def _remove_ingredient_async(self, ingredient_id: str) -> Dict[str, Any]:
        """Async implementation of remove_ingredient"""
        await self.ensure_connected()

        try:
            # Normalize ID (handles both IDs and names)
            item_id = normalize_food_id(ingredient_id)

            # Call through MCP protocol
            result = await self._send_request("tools/call", {
                "name": "delete_food_item",
                "arguments": {"id": item_id}
            })

            return {
                "success": result.get("success", False),
                "item_id": item_id,
                "action": result.get("action", "deleted"),
                "message": result.get("message"),
                "error": result.get("error"),
                "data": result.get("data")  # Include deleted item data
            }
        except Exception as e:
            print(f"❌ Error in remove_ingredient: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "item_id": item_id if 'item_id' in locals() else None
            }

    def clear_pantry(self) -> List[Dict[str, Any]]:
        """
        Clear all items from the pantry via MCP.

        Returns:
            List of items that were deleted
        """
        return self._run_sync(self._clear_pantry_async())

    async def _clear_pantry_async(self) -> List[Dict[str, Any]]:
        """
        Async implementation of clear_pantry.
        Gets all items and deletes them one by one.

        Returns:
            List of deleted items (with their data before deletion)
        """
        await self.ensure_connected()

        try:
            # Get all items in the pantry
            inventory = await self._get_inventory_async()

            if not inventory:
                print("📭 Pantry is already empty")
                return []

            print(f"🔥 Clearing {len(inventory)} items from pantry...")

            # Delete each item
            deleted_items = []
            for item in inventory:
                result = await self._remove_ingredient_async(item['id'])
                if result.get('success'):
                    deleted_items.append(item)
                    print(f"   ✓ Deleted: {item['name']}")
                else:
                    print(f"   ✗ Failed to delete: {item['name']} - {result.get('error')}")

            print(f"✅ Cleared {len(deleted_items)}/{len(inventory)} items")
            return deleted_items

        except Exception as e:
            print(f"❌ Error clearing pantry: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def update_quantity(self, ingredient_id: str, new_quantity: int, mode: str = "absolute") -> Dict[str, Any]:
        """
        Update quantity for an ingredient via MCP.

        Args:
            ingredient_id: ID or name of the ingredient
            new_quantity: Quantity value
            mode: "absolute" (set to exact value) or "delta" (add/subtract)

        Returns:
            Dict with success status and action taken
        """
        return self._run_sync(self._update_quantity_async(ingredient_id, new_quantity, mode))

    async def _update_quantity_async(self, ingredient_id: str, new_quantity: int, mode: str = "absolute") -> Dict[str, Any]:
        """Async implementation of update_quantity"""
        await self.ensure_connected()

        try:
            item_id = normalize_food_id(ingredient_id)

            # If quantity is 0 or negative in absolute mode, delete the item
            if mode == "absolute" and new_quantity <= 0:
                return await self._remove_ingredient_async(item_id)

            # Call through MCP protocol with mode parameter
            args = {
                "id": item_id,
                "quantity": new_quantity,
                "mode": mode
            }
            result = await self._send_request("tools/call", {
                "name": "update_food_item",
                "arguments": args
            })

            # If item doesn't exist and we're in absolute mode, add it instead
            if not result.get("success") and mode == "absolute" and "not found" in result.get("error", "").lower():
                print(f"📝 Item '{item_id}' not found, adding it instead")
                return await self._add_or_update_ingredient_async(
                    ingredient_name=ingredient_id,
                    quantity=new_quantity
                )

            return {
                "success": result.get("success", False),
                "action": result.get("action", "updated"),
                "item_id": item_id,
                "data": result.get("data"),
                "error": result.get("error"),
                "message": result.get("message")
            }
        except Exception as e:
            print(f"❌ Error in update_quantity: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "item_id": item_id if 'item_id' in locals() else None
            }

    # ============================================
    # AI NATURAL LANGUAGE INTERFACE
    # Async method using OpenAI for query interpretation
    # ============================================

    async def handle_query(self, user_query: str, conversation_history: list = None) -> Any:
        """
        Process natural language queries using OpenAI.
        All operations go through MCP client.

        Examples:
            - "I ate 2 eggs"
            - "What's in my pantry?"
            - "Add 5 oranges"
            - "I bought 3 tomatoes and 2 onions"
            - "Clear the pantry"

        Args:
            user_query: Natural language query from user
            conversation_history: Optional list of prior message dicts with "role" and "content"

        Returns:
            Dict with "result" (PantryItemsResponse) and "operations" list, or special dicts for clarification/cancel/error
        """
        await self.ensure_connected()

        if not self.openai_client:
            print("⚠️  OpenAI client not initialized. Please set OPENAI_API_KEY.")
            return None

        # Define tools for OpenAI function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ask_for_quantity",
                    "description": "Ask user for quantity when: (1) PLURAL without numbers ('oysters', 'eggs'), (2) SINGULAR without 'a/an' article ('tomato', 'garlic' - ambiguous), (3) UNCOUNTABLE nouns ('milk', 'rice', 'flour'). DO NOT use for items with 'a/an' (e.g., 'a tomato' = 1). Examples: 'I have oysters' → ask, 'I have milk' → ask, 'I have tomato' → ask, but 'I have a tomato' → add_food_item(quantity=1).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of item names needing quantity clarification (use singular form)"
                            }
                        },
                        "required": ["items"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_food_item",
                    "description": "Add a NEW food/beverage item to the pantry. Use ONLY when the item does NOT already exist in the pantry. Use when: (1) User states explicit number ('2 chickens', '5 tomatoes'), (2) User uses singular with 'a/an' ('a tomato' = quantity 1). If the item ALREADY EXISTS, use set_food_quantity instead. DO NOT use for plural without number (use ask_for_quantity) or non-food items (reject).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the FOOD/BEVERAGE item (singular form, e.g., 'tomato', 'oyster')"},
                            "quantity": {"type": "integer", "description": "Quantity: explicit number from user OR 1 if singular form with 'a/an'"}
                        },
                        "required": ["name", "quantity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_food_items",
                    "description": "Get all food items in the pantry with quantity and expiration date",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_food_quantity",
                    "description": "Set the exact quantity of a food item. Use when: (1) user says 'set to X', 'update to X', 'change to X', or (2) user says 'I have X [item]' and the item ALREADY exists in the pantry (they are stating their current total, not adding more).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the food item"},
                            "quantity": {"type": "integer", "description": "Exact quantity to set"}
                        },
                        "required": ["name", "quantity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "adjust_food_quantity",
                    "description": "Adjust quantity by a delta amount (use when user says 'I ate X', 'I used X', 'consumed X')",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the food item"},
                            "quantity": {"type": "integer", "description": "Quantity delta (negative for consumption, positive for addition)"}
                        },
                        "required": ["name", "quantity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_food_item",
                    "description": "Delete/remove a specific food item completely from the pantry. Use when user says 'remove X', 'delete X', 'toss X', 'throw away X', 'clear all the X', 'I don't have X anymore', or 'I cooked with X' (no quantity). Do NOT confuse with clear_pantry.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the food item to delete"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_expiring_soon",
                    "description": "Get food items expiring within a given number of days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "Number of days to check (default 7)"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_pantry",
                    "description": "Delete ALL items from the pantry at once. ONLY use when the user wants to empty the ENTIRE pantry (e.g. 'clear my pantry', 'empty everything', 'remove all items'). Do NOT use when user says 'clear all the [specific item]' — that means delete_food_item for that item.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

        try:
            # Build system content with dynamic context
            system_content = self.system_prompt

            # Inject current pantry inventory so the LLM can distinguish add vs set
            inventory = await self._get_inventory_async()
            if inventory:
                inv_lines = [f"- {item['name']}: {item['quantity']}" for item in inventory]
                system_content += (
                    "\n\nCURRENT PANTRY INVENTORY:\n" + "\n".join(inv_lines)
                    + "\n\nIMPORTANT: When the user says 'I have X [item]' and that item "
                    "ALREADY EXISTS in the pantry, use set_food_quantity (not add_food_item) "
                    "because the user is stating their CURRENT total, not adding more. "
                    "Only use add_food_item for items NOT already in the pantry, or when "
                    "the user explicitly says 'add', 'bought', or 'got more'."
                )
            else:
                system_content += "\n\nCURRENT PANTRY INVENTORY: Empty (no items)"

            # Inject pending-items context so LLM can handle quantity follow-ups
            if self.pending_items:
                pending_str = ", ".join(self.pending_items)
                pending_numbered = "; ".join(
                    f"{i+1}. {name}" for i, name in enumerate(self.pending_items)
                )
                system_content += (
                    f"\n\nPENDING QUANTITY CLARIFICATION:\n"
                    f"The user was asked to provide quantities for EXACTLY these items "
                    f"(in this order): [{pending_numbered}].\n"
                    f"RULES (follow strictly):\n"
                    f"1. Call add_food_item ONLY for items in the pending list above. "
                    f"Do NOT add, modify, or set quantities for ANY other item — even if "
                    f"that item was mentioned in earlier conversation messages.\n"
                    f"2. If the user gives MULTIPLE numbers (e.g., '3 and 2', '3 sushi, 2 broccoli', "
                    f"'3, 2'), map them to the pending items IN ORDER: first number → first pending "
                    f"item, second number → second pending item, etc.\n"
                    f"3. If the user gives a SINGLE number and there is only ONE pending item, "
                    f"apply it to that item.\n"
                    f"4. If the user gives a SINGLE number but there are MULTIPLE pending items, "
                    f"apply that number to ALL pending items.\n"
                    f"5. Extra words like 'from leftovers', 'I think', 'approximately' are just "
                    f"qualifiers — extract only the numbers.\n"
                    f"6. ONLY ignore the pending items if the user is clearly NOT providing "
                    f"quantities (asking about recipes, cancelling, changing topic, saying "
                    f"'never mind', etc.).\n"
                    f"Examples:\n"
                    f"- pending=[1. broccoli], user says '3' → add_food_item(name='broccoli', quantity=3)\n"
                    f"- pending=[1. sushi; 2. broccoli], user says '5 and 2' → "
                    f"add_food_item(name='sushi', quantity=5) AND add_food_item(name='broccoli', quantity=2)\n"
                    f"- pending=[1. sushi; 2. broccoli], user says '3' → "
                    f"add_food_item(name='sushi', quantity=3) AND add_food_item(name='broccoli', quantity=3)"
                )

            # Build messages list with optional conversation history
            messages = [{"role": "system", "content": system_content}]
            if conversation_history:
                _type_to_role = {"human": "user", "ai": "assistant"}
                recent = conversation_history[-6:]
                for msg in recent:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                    elif hasattr(msg, "type") and hasattr(msg, "content"):
                        role = _type_to_role.get(msg.type, "user")
                        content = msg.content or ""
                    else:
                        continue
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_query})

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            tool_results = []
            affected_items = []
            operations = []
            clarification_needed = False
            # Track items the LLM tried to add without a quantity so we can
            # merge them into pending_items for the clarification flow.
            items_missing_qty: List[str] = []

            for tool_call in getattr(message, "tool_calls", []) or []:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"🔧 Tool Call: {func_name}({args})")

                if func_name == "add_food_item":
                    item_name = args.get("name", "")

                    if "quantity" not in args:
                        print(f"⚠️  Warning: add_food_item called without quantity for '{item_name}'. Will ask for quantity.")
                        items_missing_qty.append(item_name)
                        continue

                    quantity = args.get("quantity", 0)
                    if quantity <= 0:
                        print(f"⚠️  Warning: Invalid quantity {quantity} for {item_name}. Will ask for quantity.")
                        items_missing_qty.append(item_name)
                        continue

                    result = await self._add_or_update_ingredient_async(
                        ingredient_name=item_name,
                        quantity=args["quantity"]
                    )
                    if result.get("success") and result.get("data"):
                        affected_items.append(result["data"])
                    operations.append({"type": func_name, "item": item_name, "quantity": args["quantity"]})

                elif func_name == "set_food_quantity":
                    item_id = normalize_food_id(args["name"])
                    result = await self._update_quantity_async(
                        ingredient_id=item_id,
                        new_quantity=args["quantity"],
                        mode="absolute"
                    )
                    if result.get("success") and result.get("data"):
                        affected_items.append(result["data"])
                    operations.append({"type": func_name, "item": args["name"], "quantity": args["quantity"]})

                elif func_name == "adjust_food_quantity":
                    item_id = normalize_food_id(args["name"])
                    result = await self._update_quantity_async(
                        ingredient_id=item_id,
                        new_quantity=args["quantity"],
                        mode="delta"
                    )
                    if result.get("success") and result.get("data"):
                        affected_items.append(result["data"])
                    operations.append({"type": func_name, "item": args["name"], "quantity": args["quantity"]})

                elif func_name == "delete_food_item":
                    result = await self._remove_ingredient_async(args["name"])
                    if result.get("success") and result.get("data"):
                        affected_items.append(result["data"])
                    operations.append({"type": func_name, "item": args["name"], "quantity": None})

                elif func_name == "get_all_food_items":
                    items = await self._get_inventory_async()
                    affected_items.extend(items)
                    result = {"success": True, "count": len(items)}
                    operations.append({"type": func_name, "item": None, "quantity": None})

                elif func_name == "get_expiring_soon":
                    days = args.get("days", 7)
                    items = await self._get_expiring_soon_async(days_threshold=days)
                    affected_items.extend(items)
                    result = {"success": True, "count": len(items)}
                    operations.append({"type": func_name, "item": None, "quantity": days})

                elif func_name == "clear_pantry":
                    items = await self._clear_pantry_async()
                    affected_items.extend(items)
                    result = {"success": True, "count": len(items), "message": f"Cleared {len(items)} items"}
                    operations.append({"type": func_name, "item": None, "quantity": len(items)})

                elif func_name == "ask_for_quantity":
                    items_list = args.get("items", [])
                    self.pending_items = items_list
                    clarification_needed = True
                    result = {"success": True, "needs_clarification": True, "items": items_list}

                tool_results.append({"tool_name": func_name, "result": result})

            # Merge any items that were skipped (no quantity) into pending_items
            # so the clarification question covers ALL of them.
            if items_missing_qty:
                existing = set(self.pending_items)
                for name in items_missing_qty:
                    if name not in existing:
                        self.pending_items.append(name)
                        existing.add(name)
                if not clarification_needed:
                    clarification_needed = True

            if clarification_needed:
                return {"needs_clarification": True, "pending_items": self.pending_items}

            # No tools called — LLM responded with text only
            if not tool_results and message.content:
                had_pending = bool(self.pending_items)
                self.pending_items = []

                if had_pending:
                    return {
                        "needs_clarification": False,
                        "cancelled": True,
                        "message": message.content,
                    }
                else:
                    return {
                        "needs_clarification": False,
                        "error": message.content,
                        "rejected": True
                    }

            # Clear pending items after successful tool execution
            if self.pending_items and operations:
                self.pending_items = []

            if affected_items:
                typed_result = convert_items(affected_items)
            else:
                typed_result = PantryItemsResponse(items=[])

            print(f"\n✅ Query processed: {len(tool_results)} tool(s) executed, operations: {[op['type'] for op in operations]}")
            return {"result": typed_result, "operations": operations}

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return None

    # ============================================
    # UTILITY METHODS
    # ============================================

    def identify_expiring_items(
        self,
        inventory: Optional[PantryItemsResponse] = None
    ) -> List[SingleItemResponse]:
        """
        Identify items expiring within 7 days.
        Can work with typed inventory or fetch fresh data.

        Args:
            inventory: Optional pre-fetched inventory

        Returns:
            List of SingleItemResponse items expiring soon
        """
        if inventory is None:
            raw_items = self.get_expiring_soon(days_threshold=7)
            return [SingleItemResponse(**item) for item in raw_items]

        expiring_items = []
        today = date.today()

        for item in inventory.items:
            expire_date = date.fromisoformat(item.expire_date)
            days_to_expiry = (expire_date - today).days
            if days_to_expiry <= 7:
                expiring_items.append(item)

        return expiring_items

    # ============================================
    # CONTEXT MANAGER SUPPORT
    # ============================================

    async def __aenter__(self):
        """Context manager entry - ensures MCP connection"""
        await self.ensure_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup MCP connection"""
        await self.disconnect()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the PantryAgent with proper MCP architecture.
    """
    import asyncio

    async def main():
        # Use context manager for automatic connection/cleanup
        async with PantryAgent() as agent:
            print("=== PantryAgent with MCP Client ===\n")

            # Example 1: Add items
            print("1. Adding items...")
            result = agent.add_or_update_ingredient("Apple", 5)
            print(f"   Result: {result}\n")

            # Example 2: Get inventory
            print("2. Getting inventory...")
            inventory = agent.get_inventory()
            print(f"   Items: {len(inventory)}\n")

            # Example 3: Natural language query
            print("3. Processing natural language query...")
            result = await agent.handle_query("I ate 2 apples")
            print(f"   Result: {result}\n")

            # Example 4: Get expiring items
            print("4. Getting expiring items...")
            expiring = agent.get_expiring_soon(7)
            print(f"   Expiring soon: {len(expiring)}\n")

    asyncio.run(main())
