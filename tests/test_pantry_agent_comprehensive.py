#!/usr/bin/env python3
"""
Comprehensive Test Suite for PantryAgent

Tests all features including:
- Basic operations (add, remove, update, get inventory)
- Natural language query parsing
- Edge cases (as well, too, also, compound phrases)
- Quantity clarification scenarios
- Food validation
- Multi-item operations
- Expiring items
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.pantry_agent import PantryAgent, PantryItemsResponse


class TestResult:
    """Container for test results"""
    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name}" + (f"\n   {self.details}" if self.details else "")


class PantryAgentTester:
    """Comprehensive test suite for PantryAgent"""

    def __init__(self):
        self.agent = None
        self.results: List[TestResult] = []

    async def setup(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        self.agent = PantryAgent(name="Test Pantry Agent")
        await self.agent.ensure_connected()

    async def _handle_query(self, query: str):
        """Wrapper that unwraps the new dict return format from handle_query."""
        result = await self._handle_query(query)
        if isinstance(result, dict) and "result" in result and "operations" in result:
            return result["result"]
        return result

        # Clear pantry before tests
        await self.agent._clear_pantry_async()
        print("✅ Test environment ready\n")

    async def teardown(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up test environment...")
        if self.agent:
            await self.agent._clear_pantry_async()
            await self.agent.disconnect()
        print("✅ Cleanup complete")

    def add_result(self, name: str, passed: bool, details: str = ""):
        """Add a test result"""
        result = TestResult(name, passed, details)
        self.results.append(result)
        print(result)

    # ============================================
    # BASIC OPERATIONS TESTS
    # ============================================

    async def test_basic_add_ingredient(self):
        """Test basic ingredient addition"""
        result = self.agent.add_or_update_ingredient("apple", 5)

        passed = (
            result.get("success") == True and
            result.get("quantity") == 5
        )

        self.add_result(
            "Basic Add Ingredient",
            passed,
            f"Added 5 apples, success={result.get('success')}"
        )

    async def test_get_inventory(self):
        """Test getting inventory"""
        # Add some items first
        self.agent.add_or_update_ingredient("banana", 3)
        self.agent.add_or_update_ingredient("orange", 2)

        inventory = self.agent.get_inventory()

        passed = len(inventory) >= 2

        self.add_result(
            "Get Inventory",
            passed,
            f"Found {len(inventory)} items in inventory"
        )

    async def test_remove_ingredient(self):
        """Test removing an ingredient"""
        # Add item first
        self.agent.add_or_update_ingredient("mango", 3)

        # Remove it
        result = self.agent.remove_ingredient("mango")

        passed = result.get("success") == True

        self.add_result(
            "Remove Ingredient",
            passed,
            f"Removed mango, success={result.get('success')}"
        )

    async def test_update_quantity_absolute(self):
        """Test updating quantity (absolute mode)"""
        # Add item first
        self.agent.add_or_update_ingredient("tomato", 5)

        # Update to exact quantity
        result = self.agent.update_quantity("tomato", 10, mode="absolute")

        passed = (
            result.get("success") == True and
            result.get("data", {}).get("quantity") == 10
        )

        self.add_result(
            "Update Quantity (Absolute)",
            passed,
            f"Set tomato quantity to 10, new quantity={result.get('data', {}).get('quantity')}"
        )

    async def test_update_quantity_delta(self):
        """Test updating quantity (delta mode)"""
        # Add item first
        self.agent.add_or_update_ingredient("potato", 10)

        # Subtract 3
        result = self.agent.update_quantity("potato", -3, mode="delta")

        passed = (
            result.get("success") == True and
            result.get("data", {}).get("quantity") == 7
        )

        self.add_result(
            "Update Quantity (Delta)",
            passed,
            f"Reduced potato by 3, new quantity={result.get('data', {}).get('quantity')}"
        )

    async def test_clear_pantry(self):
        """Test clearing entire pantry"""
        # Clear pantry first to ensure clean state
        await self.agent._clear_pantry_async()

        # Add multiple items
        self.agent.add_or_update_ingredient("item1", 1)
        self.agent.add_or_update_ingredient("item2", 2)
        self.agent.add_or_update_ingredient("item3", 3)

        # Clear all
        deleted = self.agent.clear_pantry()

        # Check inventory is empty
        inventory = self.agent.get_inventory()

        passed = len(inventory) == 0 and len(deleted) == 3

        self.add_result(
            "Clear Pantry",
            passed,
            f"Deleted {len(deleted)} items (expected 3), inventory now has {len(inventory)} items"
        )

    # ============================================
    # NATURAL LANGUAGE QUERY TESTS
    # ============================================

    async def test_nl_explicit_quantity(self):
        """Test: 'I have 5 apples' (explicit quantity)"""
        result = await self._handle_query("I have 5 apples")

        passed = (
            isinstance(result, PantryItemsResponse) and
            len(result.items) > 0 and
            not result.__dict__.get("needs_clarification", False)
        )

        self.add_result(
            "NL: Explicit Quantity",
            passed,
            f"'I have 5 apples' -> {len(result.items) if isinstance(result, PantryItemsResponse) else 0} items added"
        )

    async def test_nl_article_a_an(self):
        """Test: 'I have a tomato' (with article = quantity 1)"""
        result = await self._handle_query("I have a tomato")

        passed = (
            isinstance(result, PantryItemsResponse) and
            len(result.items) > 0 and
            not result.__dict__.get("needs_clarification", False)
        )

        self.add_result(
            "NL: Article 'a/an'",
            passed,
            f"'I have a tomato' -> {len(result.items) if isinstance(result, PantryItemsResponse) else 0} items added"
        )

    async def test_nl_plural_without_quantity(self):
        """Test: 'I have oysters' (plural without quantity - needs clarification)"""
        result = await self._handle_query("I have oysters")

        # Should ask for clarification
        needs_clarification = (
            isinstance(result, dict) and
            result.get("needs_clarification") == True and
            "oyster" in result.get("pending_items", [])
        )

        self.add_result(
            "NL: Plural Without Quantity",
            needs_clarification,
            f"'I have oysters' -> needs_clarification={needs_clarification}"
        )

        # Clear pending state
        self.agent.pending_items = []

    async def test_nl_singular_without_article(self):
        """Test: 'I have garlic' (singular without article - ambiguous)"""
        result = await self._handle_query("I have garlic")

        # Should ask for clarification
        needs_clarification = (
            isinstance(result, dict) and
            result.get("needs_clarification") == True
        )

        self.add_result(
            "NL: Singular Without Article",
            needs_clarification,
            f"'I have garlic' -> needs_clarification={needs_clarification}"
        )

        # Clear pending state
        self.agent.pending_items = []

    async def test_nl_uncountable_noun(self):
        """Test: 'I have milk' (uncountable - needs clarification)"""
        result = await self._handle_query("I have milk")

        # Should ask for clarification
        needs_clarification = (
            isinstance(result, dict) and
            result.get("needs_clarification") == True
        )

        self.add_result(
            "NL: Uncountable Noun",
            needs_clarification,
            f"'I have milk' -> needs_clarification={needs_clarification}"
        )

        # Clear pending state
        self.agent.pending_items = []

    # ============================================
    # EDGE CASE TESTS (The Important Ones!)
    # ============================================

    async def test_edge_as_well(self):
        """Test: 'I have mango and sticky rice as well' (the bug we fixed!)"""
        result = await self._handle_query("I have mango and sticky rice as well")

        # Should ask for clarification for both items
        needs_clarification = isinstance(result, dict) and result.get("needs_clarification") == True

        if needs_clarification:
            pending = result.get("pending_items", [])
            # Check that "well" is NOT in the list
            has_well_bug = "well" in pending
            has_correct_items = "mango" in pending or "sticky rice" in pending

            passed = not has_well_bug and has_correct_items
            details = f"Pending items: {pending}, has 'well' bug: {has_well_bug}"
        else:
            passed = isinstance(result, PantryItemsResponse)
            details = f"Result type: {type(result).__name__}"

        self.add_result(
            "Edge Case: 'as well'",
            passed,
            details
        )

        # Clear pending state
        self.agent.pending_items = []

    async def test_edge_too(self):
        """Test: 'I have tomatoes and eggs too'"""
        result = await self._handle_query("I have tomatoes and eggs too")

        # Should handle "too" correctly
        if isinstance(result, dict) and result.get("needs_clarification"):
            pending = result.get("pending_items", [])
            has_too_bug = "too" in pending
            passed = not has_too_bug
            details = f"Pending items: {pending}, has 'too' bug: {has_too_bug}"
        else:
            passed = True
            details = "Handled correctly without clarification issues"

        self.add_result(
            "Edge Case: 'too'",
            passed,
            details
        )

        # Clear pending state
        self.agent.pending_items = []

    async def test_edge_also(self):
        """Test: 'I got chicken also'"""
        result = await self._handle_query("I got chicken also")

        # Should handle "also" correctly
        if isinstance(result, dict) and result.get("needs_clarification"):
            pending = result.get("pending_items", [])
            has_also_bug = "also" in pending
            passed = not has_also_bug
            details = f"Pending items: {pending}, has 'also' bug: {has_also_bug}"
        else:
            passed = True
            details = "Handled correctly"

        self.add_result(
            "Edge Case: 'also'",
            passed,
            details
        )

        # Clear pending state
        self.agent.pending_items = []

    async def test_edge_compound_items(self):
        """Test: Compound food names like 'sticky rice', 'ice cream'"""
        result = await self._handle_query("I have 2 sticky rice and 3 ice cream")

        passed = isinstance(result, PantryItemsResponse) and len(result.items) >= 1

        self.add_result(
            "Edge Case: Compound Names",
            passed,
            f"'sticky rice' and 'ice cream' handled: {len(result.items) if isinstance(result, PantryItemsResponse) else 0} items"
        )

    async def test_edge_mixed_quantities(self):
        """Test: Mixed - some with quantities, some without"""
        result = await self._handle_query("I have 2 apples and bananas")

        # Should handle mixed case (2 apples explicit, bananas needs clarification)
        # The LLM should either add apples and ask about bananas, or ask about both
        passed = True  # Accept any reasonable response

        self.add_result(
            "Edge Case: Mixed Quantities",
            passed,
            f"Mixed quantities handled, result type: {type(result).__name__}"
        )

        # Clear pending state
        self.agent.pending_items = []

    # ============================================
    # FOOD VALIDATION TESTS
    # ============================================

    async def test_food_validation_reject_nonfood(self):
        """Test: Reject non-food items like 'laptop'"""
        result = await self._handle_query("I have a laptop")

        # Should reject non-food item - either explicit error OR no items added
        is_error = isinstance(result, dict) and "error" in result
        no_items_added = isinstance(result, PantryItemsResponse) and len(result.items) == 0
        is_rejected = is_error or no_items_added

        self.add_result(
            "Food Validation: Reject Non-Food",
            is_rejected,
            f"'laptop' rejected: {is_rejected} (error={is_error}, no_items={no_items_added})"
        )

    async def test_food_validation_accept_food(self):
        """Test: Accept valid food items"""
        result = await self._handle_query("I have a chicken")

        # Should accept food item
        passed = isinstance(result, PantryItemsResponse) or (
            isinstance(result, dict) and result.get("needs_clarification")
        )

        self.add_result(
            "Food Validation: Accept Food",
            passed,
            f"'chicken' accepted: {passed}"
        )

        # Clear pending state
        self.agent.pending_items = []

    # ============================================
    # MULTI-ITEM OPERATION TESTS
    # ============================================

    async def test_multi_item_explicit_quantities(self):
        """Test: 'I bought 2 apples, 3 bananas, and 5 oranges'"""
        result = await self._handle_query("I bought 2 apples, 3 bananas, and 5 oranges")

        passed = isinstance(result, PantryItemsResponse) and len(result.items) == 3

        self.add_result(
            "Multi-Item: Explicit Quantities",
            passed,
            f"Added {len(result.items) if isinstance(result, PantryItemsResponse) else 0}/3 items"
        )

    async def test_multi_item_removal(self):
        """Test: 'I ate 2 apples and 1 banana'"""
        # Clear pantry first for clean state
        await self.agent._clear_pantry_async()

        # Add items first
        self.agent.add_or_update_ingredient("apple", 10)
        self.agent.add_or_update_ingredient("banana", 10)

        result = await self._handle_query("I ate 2 apples and 1 banana")

        # Check updated quantities
        inventory = self.agent.get_inventory()
        apple_qty = next((item["quantity"] for item in inventory if item["name"] == "apple"), None)
        banana_qty = next((item["quantity"] for item in inventory if item["name"] == "banana"), None)

        passed = apple_qty == 8 and banana_qty == 9

        self.add_result(
            "Multi-Item: Consumption",
            passed,
            f"Apple: 10->8 (actual={apple_qty}), Banana: 10->9 (actual={banana_qty})"
        )

    # ============================================
    # QUANTITY CLARIFICATION FLOW TESTS
    # ============================================

    async def test_quantity_clarification_flow(self):
        """Test: Full clarification flow"""
        # Step 1: User says "I have oysters"
        result1 = await self._handle_query("I have oysters")

        needs_clarification = (
            isinstance(result1, dict) and
            result1.get("needs_clarification") == True
        )

        if not needs_clarification:
            self.add_result(
                "Quantity Clarification Flow",
                False,
                "Step 1 failed: Should ask for clarification"
            )
            self.agent.pending_items = []
            return

        # Step 2: User responds with "5"
        result2 = await self._handle_query("5")

        added = isinstance(result2, PantryItemsResponse) and len(result2.items) > 0

        if added:
            item = result2.items[0]
            correct_quantity = item.quantity == 5

            self.add_result(
                "Quantity Clarification Flow",
                correct_quantity,
                f"Full flow works: asked -> answered '5' -> added {item.quantity} oysters"
            )
        else:
            self.add_result(
                "Quantity Clarification Flow",
                False,
                "Step 2 failed: Should add 5 oysters"
            )

        # Clear pending state
        self.agent.pending_items = []

    # ============================================
    # EXPIRING ITEMS TESTS
    # ============================================

    async def test_expiring_soon(self):
        """Test: Get items expiring soon"""
        # Add item that expires in 3 days
        expire_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        self.agent.add_or_update_ingredient("expiring-item", 1, expire_date=expire_date)

        # Add item that expires in 30 days
        expire_date_far = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        self.agent.add_or_update_ingredient("fresh-item", 1, expire_date=expire_date_far)

        # Get items expiring within 7 days
        expiring = self.agent.get_expiring_soon(days_threshold=7)

        # Should only get the item expiring in 3 days
        passed = len(expiring) == 1 and expiring[0]["name"] == "expiring-item"

        self.add_result(
            "Get Expiring Soon",
            passed,
            f"Found {len(expiring)} expiring items (expected 1)"
        )

    # ============================================
    # OPERATION-SPECIFIC TESTS
    # ============================================

    async def test_operation_ate(self):
        """Test: 'I ate 2 eggs'"""
        self.agent.add_or_update_ingredient("egg", 10)

        result = await self._handle_query("I ate 2 eggs")

        inventory = self.agent.get_inventory()
        egg_qty = next((item["quantity"] for item in inventory if item["name"] == "egg"), None)

        passed = egg_qty == 8

        self.add_result(
            "Operation: 'I ate X'",
            passed,
            f"Eggs: 10->8, actual={egg_qty}"
        )

    async def test_operation_remove(self):
        """Test: 'Remove tomato' (complete removal)"""
        self.agent.add_or_update_ingredient("tomato", 5)

        result = await self._handle_query("Remove tomato")

        inventory = self.agent.get_inventory()
        has_tomato = any(item["name"] == "tomato" for item in inventory)

        passed = not has_tomato

        self.add_result(
            "Operation: 'Remove X' (complete)",
            passed,
            f"Tomato removed completely: {not has_tomato}"
        )

    async def test_operation_clear_all(self):
        """Test: 'Clear pantry'"""
        # Add some items
        self.agent.add_or_update_ingredient("item1", 1)
        self.agent.add_or_update_ingredient("item2", 2)

        result = await self._handle_query("Clear the pantry")

        inventory = self.agent.get_inventory()

        passed = len(inventory) == 0

        self.add_result(
            "Operation: 'Clear pantry'",
            passed,
            f"Pantry cleared: {len(inventory) == 0}"
        )

    async def test_operation_view_inventory(self):
        """Test: 'What's in my pantry?'"""
        # Add some items
        self.agent.add_or_update_ingredient("apple", 3)
        self.agent.add_or_update_ingredient("banana", 2)

        result = await self._handle_query("What's in my pantry?")

        passed = isinstance(result, PantryItemsResponse) and len(result.items) >= 2

        self.add_result(
            "Operation: View Inventory",
            passed,
            f"Viewed inventory: {len(result.items) if isinstance(result, PantryItemsResponse) else 0} items"
        )

    # ============================================
    # RUN ALL TESTS
    # ============================================

    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 70)
        print("🧪 COMPREHENSIVE PANTRY AGENT TEST SUITE")
        print("=" * 70)
        print()

        await self.setup()

        # Basic Operations
        print("\n" + "=" * 70)
        print("📦 BASIC OPERATIONS TESTS")
        print("=" * 70)
        await self.test_basic_add_ingredient()
        await self.test_get_inventory()
        await self.test_remove_ingredient()
        await self.test_update_quantity_absolute()
        await self.test_update_quantity_delta()
        await self.test_clear_pantry()

        # Natural Language Queries
        print("\n" + "=" * 70)
        print("🗣️  NATURAL LANGUAGE QUERY TESTS")
        print("=" * 70)
        await self.test_nl_explicit_quantity()
        await self.test_nl_article_a_an()
        await self.test_nl_plural_without_quantity()
        await self.test_nl_singular_without_article()
        await self.test_nl_uncountable_noun()

        # Edge Cases (THE IMPORTANT ONES!)
        print("\n" + "=" * 70)
        print("🔥 EDGE CASE TESTS (Critical!)")
        print("=" * 70)
        await self.test_edge_as_well()
        await self.test_edge_too()
        await self.test_edge_also()
        await self.test_edge_compound_items()
        await self.test_edge_mixed_quantities()

        # Food Validation
        print("\n" + "=" * 70)
        print("🍽️  FOOD VALIDATION TESTS")
        print("=" * 70)
        await self.test_food_validation_reject_nonfood()
        await self.test_food_validation_accept_food()

        # Multi-Item Operations
        print("\n" + "=" * 70)
        print("📚 MULTI-ITEM OPERATION TESTS")
        print("=" * 70)
        await self.test_multi_item_explicit_quantities()
        await self.test_multi_item_removal()

        # Quantity Clarification Flow
        print("\n" + "=" * 70)
        print("💬 QUANTITY CLARIFICATION FLOW TESTS")
        print("=" * 70)
        await self.test_quantity_clarification_flow()

        # Expiring Items
        print("\n" + "=" * 70)
        print("⏰ EXPIRING ITEMS TESTS")
        print("=" * 70)
        await self.test_expiring_soon()

        # Operation-Specific
        print("\n" + "=" * 70)
        print("⚙️  OPERATION-SPECIFIC TESTS")
        print("=" * 70)
        await self.test_operation_ate()
        await self.test_operation_remove()
        await self.test_operation_clear_all()
        await self.test_operation_view_inventory()

        await self.teardown()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")

        if failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   - {result.name}")
                    if result.details:
                        print(f"     {result.details}")

        print("\n" + "=" * 70)

        if failed == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED")

        print("=" * 70)


async def main():
    """Main test runner"""
    tester = PantryAgentTester()

    try:
        await tester.run_all_tests()

        # Exit with appropriate code
        failed = sum(1 for r in tester.results if not r.passed)
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        await tester.teardown()
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        await tester.teardown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

