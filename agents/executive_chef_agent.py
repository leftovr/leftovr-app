import json
from typing import Dict, List, Any, Optional, Literal, Tuple
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage


class ExecutiveChefAgent:
    """
    Executive Chef Agent - Unified Orchestrator & User Interface.

    This agent serves as BOTH the user-facing interface (Waiter) AND the backend orchestrator,
    eliminating redundant communication layers for a streamlined architecture.

    DUAL RESPONSIBILITIES:

    🎭 USER INTERFACE (Waiter Role):
    - Greet users and establish rapport
    - Collect dietary preferences, allergies, and constraints
    - Classify query types (recipe, pantry, general)
    - Present recommendations and final recipes
    - Perform quality assurance with user context
    - Handle conversational interaction

    🧠 ORCHESTRATION (Executive Chef Role):
    - Analyze request complexity
    - Decompose complex queries into subtasks
    - Delegate tasks to specialized agents (Pantry, Sous Chef, Recipe Knowledge)
    - Coordinate multi-agent workflows
    - Synthesize agent responses into coherent recommendations
    - Make strategic decisions about recipe selection
    - Optimize for food waste reduction
    """

    def __init__(self, name: str = "Maison D'Être"):
        self.name = name
        self.task_history: List[Dict[str, Any]] = []
        self.delegation_log: List[Dict[str, Any]] = []

    # ==================== ORCHESTRATION METHODS ====================

    def build_orchestration_prompt(self) -> str:
        """Return the orchestration-focused system prompt for backend reasoning."""
        return """
        <system_prompt>
        YOU ARE "MAISON D'ÊTRE" — A UNIFIED AI CULINARY ASSISTANT THAT COMBINES USER INTERACTION
        WITH INTELLIGENT ORCHESTRATION. YOU HANDLE BOTH THE FRIENDLY USER INTERFACE AND THE
        STRATEGIC BACKEND COORDINATION OF SPECIALIZED COOKING AGENTS.

        ###OBJECTIVE###
        DELIVER OPTIMAL RECIPE RECOMMENDATIONS THAT REDUCE FOOD WASTE AND SATISFY USER PREFERENCES
        BY INTELLIGENTLY COORDINATING SPECIALIZED AGENTS WHILE MAINTAINING WARM, HELPFUL INTERACTION.

        ###RESPONSIBILITIES###
        1. **TASK ANALYSIS**: Evaluate incoming requests for complexity and requirements
        2. **DECOMPOSITION**: Break complex queries into manageable subtasks
        3. **DELEGATION**: Assign tasks to appropriate specialized agents
        4. **COORDINATION**: Manage workflow and inter-agent communication
        5. **QUALITY ASSURANCE**: Review outputs from subagents for completeness
        6. **STRATEGIC DECISIONS**: Choose between recipe options based on multiple factors
        7. **OPTIMIZATION**: Prioritize ingredients nearing expiration to reduce waste
        8. **CONSTRAINT SATISFACTION**: Ensure all user preferences and restrictions are met

        ###AGENT ECOSYSTEM###
        You coordinate these specialized agents:

        **Pantry Agent**:
        - Manages ingredient inventory via Google Sheets
        - Checks availability and quantities
        - Tracks expiration dates
        - Updates inventory after recipe preparation

        **Sous Chef Agent**:
        - Generates recipe suggestions based on available ingredients
        - Adapts recipes to user skill level
        - Provides step-by-step cooking instructions
        - Handles recipe Q&A dialogue

        **Recipe Knowledge Agent**:
        - Retrieves recipes from vector database (Qdrant)
        - Performs semantic and hybrid search for recipe matching
        - Provides nutritional information and cooking tips

        ###DECISION FRAMEWORK###

        **Query Type Classification**:
        1. **Simple Ingredient Query**: "What can I make with chicken?"
           → Delegate to: Pantry Agent → Sous Chef Agent

        2. **Recipe Request**: "I want pasta recipes"
           → Delegate to: Recipe Knowledge Agent → Quality Control Agent

        3. **Complex Multi-Constraint**: "Vegan recipes using ingredients expiring soon"
           → Delegate to: Pantry Agent → Recipe Knowledge Agent → Sous Chef Agent → Quality Control

        4. **Inventory Management**: "Update inventory after making carbonara"
           → Delegate to: Pantry Agent only

        **Complexity Assessment Criteria**:
        - Number of constraints (diet, allergies, cuisines, skill level)
        - Number of ingredients involved
        - Need for recipe customization
        - Expiration urgency
        - Shopping requirements

        **Prioritization Strategy**:
        1. **Expiration Priority**: Use ingredients expiring within 3 days first
        2. **Dietary Compliance**: Never suggest recipes violating restrictions
        3. **Skill Alignment**: Match recipe complexity to user skill level
        4. **Preference Matching**: Favor user's preferred cuisines
        5. **Ingredient Efficiency**: Minimize waste and maximize usage

        ###WORKFLOW###
        1. **GREET** user warmly and establish context
        2. **COLLECT** user preferences (diet, allergies, skill level, cuisines)
        3. **CLASSIFY** query type (recipe, pantry, general)
        4. **ANALYZE** query complexity level (simple, medium, complex)
        5. **CONSULT** Pantry Agent for current inventory and expiring items
        6. **DETERMINE** optimal strategy (ingredient-first vs. recipe-first)
        7. **DELEGATE** to appropriate agents in correct sequence
        8. **COLLECT** and synthesize responses from subagents
        9. **PRESENT** recommendations to user with clear options
        10. **VALIDATE** final recipe for safety and constraint satisfaction
        11. **OPTIMIZE** for food waste reduction throughout

        ###CHAIN OF THOUGHTS###
        1. **UNDERSTAND**: What is the user really asking for?
        2. **ASSESS**: How complex is this request? (simple/medium/complex)
        3. **INVENTORY**: What ingredients are available? What's expiring?
        4. **CONSTRAINTS**: What are the hard requirements? (allergies, diet, skill)
        5. **OPTIONS**: What recipe strategies are feasible?
        6. **PRIORITIZE**: Which option best reduces waste and satisfies user?
        7. **DELEGATE**: Which agents need to be involved and in what order?
        8. **INTEGRATE**: How do I combine subagent outputs into a solution?
        9. **VALIDATE**: Does this solution satisfy all constraints?
        10. **COMMUNICATE**: How do I present this clearly to the user?

        ###DELEGATION PATTERNS###

        **Pattern 1: Ingredient-First Recipe**
        User wants recipe using specific ingredients
        Flow: Pantry Check → Recipe Search → Sous Chef Adaptation → Quality Check

        **Pattern 2: Recipe-First Approach**
        User wants specific recipe type
        Flow: Recipe Retrieval → Ingredient Check → Substitution Planning → Quality Check

        **Pattern 3: Waste-Reduction Mode**
        Focus on using expiring ingredients
        Flow: Expiration Check → Ingredient Prioritization → Recipe Match → Validation

        **Pattern 4: Full Discovery**
        User has no specific request
        Flow: Inventory Analysis → Preference Matching → Multiple Options → User Choice

        ###QUALITY CRITERIA###
        A successful recommendation must:
        - ✅ Satisfy all dietary restrictions and allergies
        - ✅ Match user's cooking skill level
        - ✅ Use available ingredients OR provide clear shopping list
        - ✅ Prioritize ingredients nearing expiration
        - ✅ Include clear, actionable cooking instructions
        - ✅ Have valid nutritional information
        - ✅ Be culturally and culinarily appropriate

        ###WHAT NOT TO DO###
        - DO NOT SUGGEST RECIPES WITH USER'S ALLERGENS — this is dangerous
        - DO NOT SKIP PANTRY CHECK — always know what's available
        - DO NOT IGNORE EXPIRING INGREDIENTS — waste reduction is a core goal
        - DO NOT DELEGATE TO NON-EXISTENT AGENTS — stay within the system
        - DO NOT MAKE ASSUMPTIONS ABOUT USER SKILL — respect their level
        - DO NOT PROVIDE INCOMPLETE RECIPES — ensure all steps are included
        - DO NOT FORGET TO UPDATE INVENTORY — track consumption

        ###EXAMPLE SCENARIOS###

        **Scenario 1: Simple Request**
        Input: "What can I cook tonight?"
        Analysis: Open-ended, medium complexity
        Actions:
        1. Check pantry for available ingredients
        2. Identify expiring items (spinach, milk)
        3. Query Recipe Knowledge for spinach + milk recipes
        4. Filter by user skill level (home cook)
        5. Suggest: Spinach and Cheese Quiche
        Delegation: Pantry → Recipe Knowledge → Sous Chef → Quality Control

        **Scenario 2: Complex Constraint**
        Input: "Vegan dinner, no soy, beginner level, using tomatoes"
        Analysis: High complexity, multiple constraints
        Actions:
        1. Verify tomato availability and quantity
        2. Search vegan, soy-free recipes for beginners
        3. Cross-reference with available ingredients
        4. Generate 2-3 options
        5. Present with ingredient gaps and substitutions
        Delegation: Pantry → Recipe Knowledge → Sous Chef → Quality Control

        **Scenario 3: Expiration Alert**
        Input: System detects milk expires tomorrow
        Analysis: Proactive waste reduction
        Actions:
        1. Alert about expiring milk
        2. Find recipes prominently featuring milk
        3. Filter by user preferences
        4. Suggest: Creamy Pasta or Pancakes
        Delegation: Pantry → Recipe Knowledge → Sous Chef

        ###COMMUNICATION PROTOCOL###
        When delegating to agents, provide:
        - Clear task description
        - Relevant context (user preferences, constraints)
        - Expected output format
        - Priority level (urgent/normal/low)

        When receiving from agents, validate:
        - Completeness of information
        - Constraint satisfaction
        - Data quality and consistency

        ###OPTIMIZATION PRINCIPLES###
        1. **Minimize API Calls**: Cache agent responses when possible
        2. **Parallel Processing**: Query independent agents simultaneously
        3. **Early Filtering**: Apply hard constraints (allergies) first
        4. **Graceful Degradation**: Provide alternatives if ideal solution unavailable
        5. **User-Centric**: Always prioritize user safety and satisfaction

        </system_prompt>
        """

    def analyze_request_complexity(
        self,
        llm,
        user_preferences: Dict[str, Any],
        query_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the complexity of a user request and determine processing strategy.

        Returns:
            Dict with 'complexity', 'strategy', 'required_agents', 'reasoning'
        """
        system_prompt = self.build_orchestration_prompt()

        analysis_instruction = """
        Analyze the following user request and preferences to determine:
        1. Complexity level: "simple", "medium", or "complex"
        2. Optimal processing strategy
        3. Which agents to involve and in what order
        4. Reasoning for your decision

        Return ONLY valid JSON:
        {
            "complexity": "simple|medium|complex",
            "strategy": "ingredient_first|recipe_first|waste_reduction|full_discovery",
            "required_agents": ["agent1", "agent2", ...],
            "agent_sequence": ["first_agent", "second_agent", ...],
            "reasoning": "explanation",
            "priority_factors": ["factor1", "factor2", ...],
            "estimated_steps": number
        }
        """

        user_info = f"""
        User Preferences:
        {json.dumps(user_preferences, indent=2)}

        Query Context: {query_context or "General recipe request"}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{analysis_instruction}\n\n{user_info}")
        ]

        try:
            response = llm.invoke(messages)
            analysis = json.loads(response.content)

            # Log the analysis
            self.task_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'complexity_analysis',
                'analysis': analysis
            })

            return analysis

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                'complexity': 'medium',
                'strategy': 'ingredient_first',
                'required_agents': ['pantry', 'sous_chef', 'quality_control'],
                'agent_sequence': ['pantry', 'sous_chef', 'quality_control'],
                'reasoning': 'Default analysis due to parsing error',
                'priority_factors': ['availability', 'preferences'],
                'estimated_steps': 3
            }

    def create_task_plan(
        self,
        llm,
        user_preferences: Dict[str, Any],
        complexity_analysis: Dict[str, Any],
        pantry_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a detailed execution plan with subtasks for each agent.

        Returns:
            Dict with 'tasks', 'delegation_order', 'success_criteria'
        """
        system_prompt = self.build_orchestration_prompt()

        planning_instruction = """
        Create a detailed execution plan for fulfilling this user request.

        Return ONLY valid JSON:
        {
            "tasks": [
                {
                    "agent": "agent_name",
                    "action": "specific_action",
                    "input": "what to provide to agent",
                    "expected_output": "what agent should return",
                    "priority": "high|medium|low"
                }
            ],
            "delegation_order": ["agent1", "agent2", ...],
            "success_criteria": ["criterion1", "criterion2", ...],
            "expected_duration": "estimate in minutes",
            "fallback_strategy": "what to do if primary plan fails"
        }
        """

        context = f"""
        User Preferences:
        {json.dumps(user_preferences, indent=2)}

        Complexity Analysis:
        {json.dumps(complexity_analysis, indent=2)}
        """

        if pantry_context:
            context += f"\n\nPantry Context:\n{json.dumps(pantry_context, indent=2)}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{planning_instruction}\n\n{context}")
        ]

        try:
            response = llm.invoke(messages)
            plan = json.loads(response.content)

            # Log the plan
            self.task_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'task_planning',
                'plan': plan
            })

            return plan

        except json.JSONDecodeError:
            # Fallback plan
            return {
                'tasks': [
                    {
                        'agent': 'pantry',
                        'action': 'check_inventory',
                        'input': user_preferences,
                        'expected_output': 'available ingredients list',
                        'priority': 'high'
                    },
                    {
                        'agent': 'sous_chef',
                        'action': 'suggest_recipes',
                        'input': 'inventory + preferences',
                        'expected_output': 'recipe suggestions',
                        'priority': 'high'
                    }
                ],
                'delegation_order': ['pantry', 'sous_chef', 'quality_control'],
                'success_criteria': ['recipe_suggested', 'constraints_met'],
                'expected_duration': '5-10 minutes',
                'fallback_strategy': 'Suggest recipes with shopping list'
            }

    def decide_query_type(
        self,
        user_preferences: Dict[str, Any],
        pantry_available: bool = True,
        recipe_db_available: bool = True
    ) -> Literal["ingredient", "recipe"]:
        """
        Decide whether to use ingredient-first or recipe-first approach.

        This is a critical decision that affects the workflow routing.
        """
        # If pantry is not available, must use recipe-first
        if not pantry_available:
            return "recipe"

        # If recipe DB is not available, must use ingredient-first
        if not recipe_db_available:
            return "ingredient"

        # Check for explicit recipe requests in cuisines or preferences
        cuisines = user_preferences.get('cuisines', [])
        if cuisines and len(cuisines) > 0:
            # User has specific cuisine preferences - prefer recipe search
            return "recipe"

        # Check for dietary restrictions that might benefit from recipe DB
        restrictions = user_preferences.get('restrictions', [])
        allergies = user_preferences.get('allergies', [])
        if len(restrictions) + len(allergies) > 2:
            # Complex constraints - recipe DB better for filtering
            return "recipe"

        # Default to ingredient-first (use what you have - reduce waste)
        return "ingredient"

    def delegate_to_pantry(
        self,
        action: Literal["check_inventory", "check_expiring", "check_feasibility", "update_inventory"],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a delegation packet for Pantry Agent.

        Returns:
            Dict with delegation details to be processed by pantry agent
        """
        delegation = {
            'agent': 'pantry',
            'action': action,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'delegated_by': self.name
        }

        self.delegation_log.append(delegation)

        return delegation

    def delegate_to_sous_chef(
        self,
        action: Literal["suggest_recipes", "adapt_recipe", "generate_instructions"],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a delegation packet for Sous Chef Agent.
        """
        delegation = {
            'agent': 'sous_chef',
            'action': action,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'delegated_by': self.name
        }

        self.delegation_log.append(delegation)

        return delegation

    def delegate_to_recipe_knowledge(
        self,
        action: Literal["search_recipes", "get_recipe_details", "semantic_search"],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a delegation packet for Recipe Knowledge Agent.
        """
        delegation = {
            'agent': 'recipe_knowledge',
            'action': action,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'delegated_by': self.name
        }

        self.delegation_log.append(delegation)

        return delegation

    def delegate_to_quality_control(
        self,
        action: Literal["validate_recipe", "check_allergens", "verify_instructions"],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a delegation packet for Quality Control Agent.
        """
        delegation = {
            'agent': 'quality_control',
            'action': action,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'delegated_by': self.name
        }

        self.delegation_log.append(delegation)

        return delegation

    def synthesize_recommendations(
        self,
        llm,
        agent_responses: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """
        Synthesize responses from multiple agents into a coherent recommendation.

        Args:
            agent_responses: Dict mapping agent names to their responses
            user_preferences: Original user preferences

        Returns:
            Formatted recommendation text
        """
        system_prompt = self.build_orchestration_prompt()

        synthesis_instruction = """
        You are synthesizing responses from multiple specialized agents into a
        coherent, user-friendly recommendation. Create a warm, helpful message that:

        1. Acknowledges what ingredients are available
        2. Highlights any items expiring soon (to encourage their use)
        3. Presents 1-3 recipe options with:
           - Recipe name and brief description
           - Required ingredients (with availability status)
           - Cooking time and difficulty
           - Key steps overview
        4. Provides a shopping list if needed
        5. Offers alternatives or substitutions

        Be conversational, encouraging, and focused on reducing food waste.
        """

        context = f"""
        User Preferences:
        {json.dumps(user_preferences, indent=2)}

        Agent Responses:
        {json.dumps(agent_responses, indent=2, default=str)}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{synthesis_instruction}\n\n{context}")
        ]

        response = llm.invoke(messages)

        # Log synthesis
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'synthesis',
            'output': response.content
        })

        return response.content

    def orchestrate_full_workflow(
        self,
        llm,
        user_preferences: Dict[str, Any],
        pantry_agent,
        query_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate the complete workflow from user request to final recommendation.
        This is the main entry point for the Executive Chef.

        Args:
            llm: Language model for reasoning
            user_preferences: User's dietary preferences and constraints
            pantry_agent: Instance of PantryAgent
            query_context: Optional additional context about the user's request

        Returns:
            Dict with 'success', 'recommendation', 'metadata', 'issues'
        """
        print(f"\n🔷 {self.name}: Initiating workflow orchestration")

        # Step 1: Analyze complexity
        print(f"   Analyzing request complexity...")
        complexity = self.analyze_request_complexity(llm, user_preferences, query_context)
        print(f"   Complexity: {complexity['complexity']} | Strategy: {complexity['strategy']}")

        # Step 2: Check pantry status
        print(f"   Consulting Pantry Agent...")
        pantry_summary = pantry_agent.get_pantry_summary()
        expiring_items = pantry_agent.get_expiring_soon()

        print(f"   Pantry: {pantry_summary['total_ingredients']} ingredients, "
              f"{len(expiring_items)} expiring soon")

        if expiring_items:
            print(f"   ⚠️  Priority items: {', '.join([item.get('ingredient_name', item.get('name', 'Unknown')) for item in expiring_items[:3]])}")

        # Step 3: Create task plan
        print(f"   Creating execution plan...")
        plan = self.create_task_plan(
            llm,
            user_preferences,
            complexity,
            {'summary': pantry_summary, 'expiring': expiring_items}
        )

        # Step 4: Collect agent responses
        agent_responses = {
            'pantry': {
                'summary': pantry_summary,
                'expiring_items': expiring_items,
                'inventory': pantry_agent.get_inventory()
            },
            'complexity_analysis': complexity,
            'execution_plan': plan
        }

        # Step 5: Synthesize recommendation
        print(f"   Synthesizing recommendation...")
        recommendation = self.synthesize_recommendations(llm, agent_responses, user_preferences)

        # Step 6: Return final result (quality check happens during final presentation)
        print(f"   ✅ Orchestration complete - preparing for quality validation")

        result = {
            'success': True,
            'recommendation': recommendation,
            'metadata': {
                'complexity': complexity,
                'plan': plan,
                'pantry_summary': pantry_summary,
                'expiring_items': expiring_items,
                'task_history': self.task_history
            }
        }

        print(f"🔷 {self.name}: Workflow complete\n")

        return result

    def get_delegation_log(self) -> List[Dict[str, Any]]:
        """Return the full delegation log for debugging/monitoring."""
        return self.delegation_log

    def get_task_history(self) -> List[Dict[str, Any]]:
        """Return the full task history for debugging/monitoring."""
        return self.task_history

    def clear_logs(self):
        """Clear logs (useful for starting fresh workflow)."""
        self.task_history = []
        self.delegation_log = []

    # ==================== USER INTERFACE METHODS ====================
    # These methods handle direct user interaction: greeting, classification,
    # preference extraction, and conversational responses.

    def build_user_interface_prompt(self, context: str = "general") -> str:
        """Return the user interface prompt for conversation handling."""
        if context == "general":
            return (
                "You are a friendly kitchen assistant. "
                "You ONLY answer questions about food, cooking, recipes, nutrition, pantry management, and dining. "
                "If asked about anything unrelated to food or cooking, politely decline and redirect the user "
                "to ask a food-related question. "
                "Do not ask about diet, allergies, or cuisines unless the user brings it up. "
                "DO NOT PROVIDE RECIPES."
            )
        if context == "pantry":
            return (
                """
                You are "Maison D'Être — Pantry Assistant," a warm, friendly food concierge focused on helping users manage their virtual pantry. Your role is to assist users in keeping track of ingredients by allowing them to **add, view, update, or remove items** from their pantry.

                ### OBJECTIVE ###
                1. Interpret user input as pantry management commands (CRUD: Create, Read, Update, Delete).
                2. Validate user input to ensure pantry actions are clear and safe.
                3. Confirm actions back to the user in a friendly and concise manner.
                4. Handle unclear or ambiguous input by asking one clarifying question at a time.

                ### INSTRUCTIONS ###
                - **Add Items**: If the user wants to add ingredients, ask for quantities and optional categories (e.g., "3 tomatoes, vegetables").
                - **View Items**: If the user wants to see the pantry, provide a neatly formatted list.
                - **Update Items**: If the user wants to change quantities or details, confirm the item and the new values.
                - **Delete Items**: Confirm before removing items to prevent mistakes.
                - **Stay Friendly**: Use cheerful, approachable language.
                - **Do Not Give Recipes** unless explicitly requested.

                ### RESPONSE FORMAT ###
                Always respond in **plain text** that is:
                - Clear
                - Short
                - Confirms the action taken or asks a clarifying question if needed

                ### EXAMPLES ###

                1. **Add Items**
                User: "Add 2 eggs and 1 carton of milk to my pantry."
                Agent: "Got it! I've added 2 eggs and 1 carton of milk to your pantry."

                2. **View Pantry**
                User: "What's currently in my pantry?"
                Agent: "Here's what you have:
                - Eggs: 2
                - Milk: 1 carton
                - Tomatoes: 5"

                3. **Update Items**
                User: "Change the number of tomatoes to 10."
                Agent: "Sure! I've updated your tomatoes count to 10."

                4. **Delete Items**
                User: "Remove the milk from my pantry."
                Agent: "Okay! I've removed the milk from your pantry."

                5. **Ambiguous Input**
                User: "Add some veggies."
                Agent: "Which vegetables would you like to add, and how many of each?"

                ### TONE ###
                Friendly, concise, helpful, and focused purely on pantry management. Avoid recipe suggestions unless explicitly requested.
                """
            )

        if context == "recipe":
            return (
                """
                <system_prompt>
                YOU ARE "MAISON D'ÊTRE" — A WARM, FRIENDLY, AND ATTENTIVE FOOD CONCIERGE AGENT WITHIN A MULTI-AGENT SYSTEM DEDICATED TO HELPING USERS DISCOVER, DISCUSS, AND ENJOY FOOD IN ALL ITS FORMS. YOUR PRIMARY ROLE IS TO GREET USERS, MAKE THEM FEEL WELCOME, AND GENTLY COLLECT ESSENTIAL INFORMATION ABOUT THEIR FOOD PREFERENCES, DIETARY RESTRICTIONS, AND ALLERGIES BEFORE PASSING THEM TO THE NEXT AGENT (THE RECIPE EXPERT OR CULINARY CREATOR).

                ###OBJECTIVE###
                YOUR GOAL IS TO CREATE A COMFORTABLE AND ENGAGING ATMOSPHERE WHILE GATHERING CRUCIAL USER DETAILS THAT WILL ENABLE THE NEXT AGENT TO PROVIDE HIGHLY PERSONALIZED AND SAFE RECIPE RECOMMENDATIONS.

                ###INSTRUCTIONS###
                1. **WELCOME THE USER** with a warm and engaging introduction. Establish a friendly tone and express enthusiasm about helping them explore delicious food options.
                2. **ASK ESSENTIAL QUESTIONS** about:
                - ALLERGIES (e.g., nuts, shellfish, gluten)
                - DIETARY RESTRICTIONS (e.g., vegetarian, vegan, pescatarian, omnivore, halal, kosher, lactose intolerance)
                3. **CONFIRM UNDERSTANDING** by restating key preferences to ensure accuracy.
                4. **PREPARE HANDOFF**: Once all essential information is gathered, SUMMARIZE the details clearly and POLITELY INFORM the user that their preferences will be shared with the next agent for tailored recipe recommendations.
                5. **MAINTAIN A CONSISTENT PERSONA**: You are polite, conversational, knowledgeable about food culture, and naturally curious about people's tastes.

                ###CHAIN OF THOUGHTS###
                FOLLOW THIS STRUCTURED REASONING PROCESS TO ENSURE A CONSISTENT AND EFFECTIVE CONVERSATION FLOW:

                1. **UNDERSTAND** the user's initial greeting or request — identify if they want to talk about food, recipes, or preferences.
                2. **BASICS** — determine what essential dietary information is missing to create a complete food profile.
                3. **BREAK DOWN** the conversation into small, friendly questions that make the user feel comfortable.
                4. **ANALYZE** their responses to infer personality cues (e.g., adventurous eater vs. comfort food lover).
                5. **BUILD** a concise summary of their preferences (dietary restrictions, allergies).
                6. **EDGE CASES** — handle users who refuse to share certain information by politely offering general options instead.
                7. **FINAL ANSWER** — deliver a warm closing message, confirming that their information will be passed to the next culinary agent.

                ###WHAT NOT TO DO###
                - DO NOT BE COLD, ROBOTIC, OR FORMAL — YOU MUST SOUND HUMAN AND FRIENDLY.
                - DO NOT JUMP TO RECIPE RECOMMENDATIONS — THAT IS THE NEXT AGENT'S ROLE.
                - DO NOT SKIP ASKING ABOUT ALLERGIES OR RESTRICTIONS — THIS INFORMATION IS ESSENTIAL.
                - DO NOT PRESS USERS FOR INFORMATION THEY DECLINE TO SHARE — RESPECT THEIR CHOICES.
                - DO NOT USE TECHNICAL OR CLINICAL LANGUAGE — KEEP THE CONVERSATION NATURAL AND WARM.
                - DO NOT PROVIDE MEDICAL ADVICE OR NUTRITIONAL PRESCRIPTIONS — FOCUS ON FOOD PREFERENCES ONLY.

                ###FEW-SHOT EXAMPLES###

                **Example 1 (Desired Behavior)**
                User: "Hey there! I'm looking for something new to cook."
                Agent: "Bonjour! I'm delighted to help. Before we begin, could you share a little about what you enjoy eating — and if you have any dietary restrictions or allergies I should keep in mind?"

                **Example 2 (Confirming Understanding)**
                User: "I'm vegan, and I'm allergic to peanuts."
                Agent: "Perfect, thank you! So, vegan and peanut-free — got it. Do you have a favorite cuisine, or should I note that you're open to exploring a variety?"

                **Example 3 (Smooth Handoff)**
                Agent: "Thank you for sharing that! I've noted your preferences — vegan, peanut-free, and you love spicy Asian flavors. I'll pass this to our culinary expert who'll find you the perfect recipes!"

                ###OPTIMIZATION STRATEGY###
                For **gpt-4o-mini**, USE CLEAR, SIMPLE LANGUAGE and FRIENDLY SENTENCES. AVOID OVERLY LONG QUESTIONS. USE NATURAL TRANSITIONS AND POSITIVE EMOTION TO CREATE A WELCOMING TONE.

                </system_prompt>

                """
            )

    def run_waiter(self, llm, context: str = "general") -> str:
        """Generate initial greeting based on context."""
        if context == "recipe":
            return "Bonjour! I'm your culinary assistant. Tell me a bit about what you like to eat and any dietary restrictions you have."
        elif context == "pantry":
            return "Hello! What would you like to do with your pantry today? "
        else:  # general
            return "Hi there! I'm your Waiter — here to help with recipes, pantry ideas, and meal planning."

    def respond_as_waiter(self, llm, user_input: str, context: str = "general") -> str:
        """Generate an interactive response given user input using the user interface prompt."""
        prompt = self.build_user_interface_prompt(context)
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_input)
        ])
        return response.content

    def extract_ingredients(self, llm, user_message: str) -> dict:
        """
        Extract ingredients from user message for pantry operations.

        Args:
            llm: Language model
            user_message: User's message about ingredients

        Returns:
            Dict with 'ingredients' list containing {name, quantity, unit} objects
        """
        schema_instruction = (
            "Return ONLY valid JSON matching this schema (no extra text):\n"
            "{\n"
            "  \"ingredients\": [\n"
            "    {\"name\": \"ingredient_name\", \"quantity\": number, \"unit\": \"unit_string\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Examples:\n"
            "- 'I have 3 apples' → {\"ingredients\": [{\"name\": \"apple\", \"quantity\": 3, \"unit\": \"pieces\"}]}\n"
            "- 'I got 2 lbs of chicken and 1 cup of rice' → {\"ingredients\": [{\"name\": \"chicken\", \"quantity\": 2, \"unit\": \"lbs\"}, {\"name\": \"rice\", \"quantity\": 1, \"unit\": \"cup\"}]}\n"
        )

        sys = (
            "You extract ingredients from user messages into structured JSON. "
            "Parse ingredient names, quantities, and units. "
            "If no unit is specified, use 'pieces'. "
            "If no quantity is specified, use 1. "
            "Normalize ingredient names to lowercase singular forms."
        )

        resp = llm.invoke([
            SystemMessage(content=sys),
            HumanMessage(content=f"{schema_instruction}\n\nUser message:\n{user_message}")
        ])

        try:
            # Handle JSON wrapped in code blocks
            content = resp.content.strip()
            if content.startswith("```"):
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()

            data = json.loads(content)
            return {"ingredients": data.get("ingredients", [])}
        except Exception as e:
            print(f"⚠️ extract_ingredients parse failed: {e}")
            return {"ingredients": []}

    def extract_preferences(self, llm, messages: list) -> dict:
        """
        Parse messages into structured preferences.
        Returns dict with keys: allergies, restrictions, cuisines, diet, skill.

        Args:
            llm: Language model
            messages: List of message dicts with 'role' and 'content'
        """
        schema_instruction = (
            "Return ONLY valid JSON matching this schema (no extra text):\n"
            "{\n"
            "  \"allergies\": string[] | [],\n"
            "  \"restrictions\": string[] | [],\n"
            "  \"cuisines\": string[] | [],\n"
            "  \"diet\": string | null,\n"
            "  \"skill\": string | null\n"
            "}"
        )
        sys = (
            "You extract user food preferences from a conversation history into a strict JSON object. "
            "Look for mentions of allergies, dietary restrictions (vegan, vegetarian, halal, kosher, etc.), "
            "preferred cuisines, diet type, and cooking skill level."
        )

        # Normalize messages to text format
        normalized_msgs = []
        for m in messages:
            if isinstance(m, dict):
                normalized_msgs.append(m)
            elif hasattr(m, "content") and hasattr(m, "type"):
                role = m.type if hasattr(m, "type") else "assistant"
                normalized_msgs.append({"role": role, "content": m.content})
            else:
                normalized_msgs.append({"role": "unknown", "content": str(m)})

        chat_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in normalized_msgs)

        resp = llm.invoke([
            SystemMessage(content=sys),
            HumanMessage(content=f"{schema_instruction}\n\nConversation:\n{chat_text}")
        ])
        try:
            data = json.loads(resp.content)
        except Exception:
            return {"allergies": [], "restrictions": [], "cuisines": [], "diet": None, "skill": None}

        # Normalize types
        def to_list(v):
            if v is None:
                return []
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return [str(v).strip()] if str(v).strip() else []

        return {
            "allergies": to_list(data.get("allergies")),
            "restrictions": to_list(data.get("restrictions")),
            "cuisines": to_list(data.get("cuisines")),
            "diet": data.get("diet"),
            "skill": data.get("skill")
        }

    def classify_query(self, llm, messages: list) -> dict:
        """
        Classify query into 'pantry', 'recipe', or 'general'.
        messages: list of dicts OR LangChain Message objects
        """
        schema_instruction = (
            "Return ONLY valid JSON matching this schema (no extra text):\n"
            "{\n"
            "  \"query_type\": \"pantry\" | \"recipe\" | \"general\"\n"
            "}"
        )
        sys = (
            "You classify the user's query strictly as one of three types: "
            "'pantry', 'recipe', or 'general'. "
            "Focus primarily on the most recent messages, but consider earlier messages "
            "to maintain ongoing context (e.g., if a recipe request was started previously). "
            "Return only the JSON object and nothing else."
        )

        # Normalize messages to dicts
        normalized_msgs = []
        for m in messages:
            if isinstance(m, dict):
                normalized_msgs.append(m)
            elif hasattr(m, "content") and hasattr(m, "type"):  # LangChain messages
                role = m.type if hasattr(m, "type") else "assistant"
                normalized_msgs.append({"role": role, "content": m.content})
            else:
                # fallback
                normalized_msgs.append({"role": "unknown", "content": str(m)})

        # Flatten for LLM input
        chat_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in normalized_msgs)

        resp = llm.invoke([
            SystemMessage(content=sys),
            HumanMessage(content=f"{schema_instruction}\n\nChat history:\n{chat_text}")
        ])

        # normalize and parse JSON
        raw_content = resp.content if isinstance(resp.content, str) else str(resp.content)
        try:
            data = json.loads(raw_content)
            qtype = data.get("query_type", "general")
        except Exception as e:
            print(f"⚠️ classify_query parse failed: {e}\nRaw content:\n{raw_content}")
            qtype = "general"

        return {"query_type": qtype}

    def classify_and_extract(self, llm, messages: list) -> dict:
        """
        One LLM call combining classify_query() + extract_preferences().

        Returns a dict with keys:
            query_type: "pantry"|"recipe"|"general"|"selection"|"off_topic"|"preference"
            selected_recipe_number: 1 | 2 | 3 | None
            allergies: list[str]             — allergies to ADD
            removed_allergies: list[str]     — specific allergies to REMOVE
            clear_allergies: bool            — clear the entire allergies list
            restrictions: list[str]          — restrictions to ADD
            removed_restrictions: list[str]  — specific restrictions to REMOVE
            clear_restrictions: bool         — clear the entire restrictions list
            cuisines: list[str]              — cuisines to ADD/prefer
            removed_cuisines: list[str]      — specific cuisines to REMOVE/avoid
            clear_cuisines: bool             — clear the entire cuisines list
            diet: str | None                 — set diet (None = no change)
            clear_diet: bool                 — True if user explicitly removes their diet
            skill: str | None
            clear_all_preferences: bool      — wipe all preference fields at once
            wants_more_recommendations: bool — user wants next batch from same search
            wants_previous_recommendations: bool — user wants to revisit an earlier batch
            previous_batch_selection: 1|2|3|None — recipe pick from that earlier batch

        Falls back to {"query_type": "general", all lists empty, all flags false} on parse failure.
        """
        schema_instruction = (
            "Return ONLY valid JSON matching this schema (no extra text):\n"
            "{\n"
            "  \"query_type\": \"pantry\" | \"recipe\" | \"general\" | \"selection\" | \"off_topic\" | \"preference\",\n"
            "  \"selected_recipe_number\": 1 | 2 | 3 | null,\n"
            "  \"allergies\": string[] | [],\n"
            "  \"removed_allergies\": string[] | [],\n"
            "  \"clear_allergies\": true | false,\n"
            "  \"restrictions\": string[] | [],\n"
            "  \"removed_restrictions\": string[] | [],\n"
            "  \"clear_restrictions\": true | false,\n"
            "  \"cuisines\": string[] | [],\n"
            "  \"removed_cuisines\": string[] | [],\n"
            "  \"clear_cuisines\": true | false,\n"
            "  \"excluded_food_types\": string[] | [],\n"
            "  \"removed_excluded_food_types\": string[] | [],\n"
            "  \"clear_excluded_food_types\": true | false,\n"
            "  \"diet\": string | null,\n"
            "  \"clear_diet\": true | false,\n"
            "  \"skill\": string | null,\n"
            "  \"clear_all_preferences\": true | false,\n"
            "  \"wants_to_exit_flow\": true | false,\n"
            "  \"is_new_recipe_search\": true | false,\n"
            "  \"wants_more_recommendations\": true | false,\n"
            "  \"wants_previous_recommendations\": true | false,\n"
            "  \"previous_batch_selection\": 1 | 2 | 3 | null,\n"
            "  \"preference_action\": \"view\" | \"update\" | null\n"
            "}"
        )
        sys = (
            "You analyze a conversation and return a single JSON object with two pieces of information:\n\n"
            "(1) Classify the user's most recent query into one of these types:\n"
            "  - 'pantry': adding, removing, or viewing ingredients in their pantry\n"
            "  - 'recipe': searching for recipes, asking what they can cook, or discussing recipes\n"
            "  - 'selection': user is CHOOSING one of the numbered recipe options that were presented "
            "(e.g., 'I'll try recipe 2', 'give me option 1', 'let's make the second one', "
            "'go with number 3', 'the first one please', 'I'd like option 2'). "
            "Only classify as 'selection' if the conversation history shows recipe options were "
            "presented AND the user is now choosing one. Set selected_recipe_number to 1, 2, or 3. "
            "Map ordinal words: 'first'=1, 'second'=2, 'third'=3.\n"
            "  - 'off_topic': the query has nothing to do with food, cooking, pantry, or nutrition "
            "(e.g., 'how do I use ChatGPT', 'what's the weather', 'help me with my taxes', "
            "'write me a poem'). If in doubt, do NOT classify as off_topic.\n"
            "  - 'preference': the user is explicitly managing their stored preferences — "
            "viewing them, editing them, removing specific items, or clearing them entirely. "
            "Use this type when the intent is about preferences themselves, not about a recipe search. "
            "(e.g., 'show my preferences', 'what are my dietary settings', "
            "'remove my Asian preference', 'clear all my restrictions', "
            "'reset my food preferences', 'I want to update my allergies').\n"
            "  - 'general': anything food/cooking related that doesn't fit the above "
            "(cooking tips, ingredient questions, greetings, app help)\n\n"
            "PANTRY CLASSIFICATION PRIORITY RULES:\n"
            "Rule 1 — INGREDIENT DECLARATIONS ARE ALWAYS PANTRY:\n"
            "When the user says 'I have [food items]', 'i got [food items]', "
            "'there is [food] in my fridge', or any variation that lists specific food "
            "ingredients they possess, ALWAYS classify as 'pantry' — even if the previous "
            "conversation was about recipes, even if the bot just said 'your pantry is empty'. "
            "The user is telling you what they have; that is a pantry operation.\n"
            "Examples that MUST be 'pantry':\n"
            "  - 'i have chicken, garlic and pasta' → pantry\n"
            "  - 'I've got 3 eggs and some milk' → pantry\n"
            "  - 'there's broccoli and rice in my fridge' → pantry\n"
            "Only classify as 'recipe' if the user ALSO explicitly asks for a recipe in "
            "the same message (e.g., 'i have chicken, what can I cook?' → recipe).\n\n"
            "Rule 2 — CONTEXT-DEPENDENT QUANTITY CORRECTIONS:\n"
            "When the user refers to a quantity WITHOUT naming a specific food item "
            "(e.g., 'nvm i have 20 in hand', 'actually there are 15', 'make that 20', "
            "'no wait, 5') RIGHT AFTER a pantry-related message in the conversation, "
            "this is still a PANTRY operation — the user is correcting or updating the "
            "quantity of the item discussed in previous turns. Classify as 'pantry', NOT 'general'. "
            "The prefix 'nvm'/'never mind' in this context means the user is correcting "
            "their previous statement, not exiting a flow. Set wants_to_exit_flow to false.\n\n"
            "(2) Extract and DIFF the user's food preferences from the LATEST message. "
            "Preferences can be ADDED, REMOVED, or CLEARED entirely. Use this logic:\n"
            "  - 'allergies': new allergies the user mentions they HAVE\n"
            "  - 'removed_allergies': specific allergies the user says they DON'T have or no longer have\n"
            "  - 'clear_allergies': true if user wants to remove ALL allergies "
            "(e.g., 'I have no allergies', 'remove all my allergies', 'clear allergies')\n"
            "  - 'cuisines': cuisines the user WANTS or prefers\n"
            "  - 'removed_cuisines': specific cuisines the user says they DON'T want\n"
            "  - 'clear_cuisines': true if user wants to remove ALL cuisine preferences "
            "(e.g., 'I'll eat any cuisine', 'clear my cuisine preferences', 'remove all cuisines')\n"
            "  - 'restrictions': dietary restrictions the user HAS\n"
            "  - 'removed_restrictions': specific restrictions the user says they no longer have\n"
            "  - 'clear_restrictions': true if user wants to remove ALL dietary restrictions\n"
            "  - 'excluded_food_types': food/dish categories the user does NOT want "
            "(e.g. 'dessert', 'appetizer', 'soup', 'salad', 'beverage', 'bread', 'snack', 'sauce', 'side dish')\n"
            "  - 'removed_excluded_food_types': food types the user is OK with again "
            "(e.g. 'actually I do want desserts' → removed_excluded_food_types: ['dessert'])\n"
            "  - 'clear_excluded_food_types': true if user wants to allow ALL food types again "
            "(e.g. 'show me any type of recipe', 'no food type restrictions')\n"
            "  - 'diet': the user's diet type as a string (e.g. 'vegan', 'keto'), or null if unchanged\n"
            "  - 'clear_diet': true ONLY if the user explicitly says they no longer follow their diet\n"
            "  - 'skill': cooking skill level if mentioned, or null if unchanged\n"
            "  - 'clear_all_preferences': true if user wants to wipe EVERYTHING "
            "(e.g., 'reset all preferences', 'clear everything', 'start fresh with no preferences')\n\n"
            "EXAMPLES:\n"
            "  'show my preferences' → query_type: 'preference', no changes\n"
            "  'I don't want Asian, I want Western' → removed_cuisines: ['asian'], cuisines: ['western']\n"
            "  'remove my Asian preference' → query_type: 'preference', removed_cuisines: ['asian']\n"
            "  'clear all my allergies' → query_type: 'preference', clear_allergies: true\n"
            "  'I have no dietary restrictions' → clear_restrictions: true\n"
            "  'reset all my preferences' → query_type: 'preference', clear_all_preferences: true\n"
            "  'I'm not vegan anymore' → clear_diet: true\n"
            "  'I changed my mind, not Italian, give me Japanese' → removed_cuisines: ['italian'], cuisines: ['japanese']\n"
            "  'I'm not allergic to shellfish anymore' → removed_allergies: ['shellfish']\n"
            "  'no desserts please' → excluded_food_types: ['dessert']\n"
            "  'skip soups and salads' → excluded_food_types: ['soup', 'salad']\n"
            "  'actually I want desserts' → removed_excluded_food_types: ['dessert']\n"
            "  'I don't want appetizers or beverages' → excluded_food_types: ['appetizer', 'beverage']\n"
            "  'show me any type of food' → clear_excluded_food_types: true\n\n"
            "IMPLICIT REPLACEMENT (critical — 'only', 'just', 'switch to' imply clearing first):\n"
            "  'nevermind I only want western' → clear_cuisines: true, cuisines: ['western']\n"
            "  'actually just Italian' → clear_cuisines: true, cuisines: ['italian']\n"
            "  'switch to keto' → clear_diet: true, diet: 'keto'\n"
            "  'I only like Thai and Japanese' → clear_cuisines: true, cuisines: ['thai', 'japanese']\n"
            "  'forget my old preferences, I'm vegan now' → clear_all_preferences: true, diet: 'vegan'\n"
            "  'instead of Asian, give me Mexican' → removed_cuisines: ['asian'], cuisines: ['mexican']\n"
            "  'no more restrictions, I eat everything' → clear_restrictions: true, clear_diet: true\n\n"
            "KEY RULE: 'only X', 'just X', 'switch to X' = clear the ENTIRE category + set new value(s).\n"
            "'also X', 'add X' = append without clearing.\n\n"
            "- 'wants_to_exit_flow': true when the user wants to cancel, abort, or abandon "
            "the current multi-turn flow (e.g., 'never mind', 'cancel', 'forget it', "
            "'start over', 'scratch that', or pivoting to a completely different topic "
            "mid-flow like asking for recipes while in quantity clarification). "
            "false in all other cases.\n\n"
            "- 'is_new_recipe_search': true ONLY when the user wants a completely FRESH search "
            "with different criteria (e.g., 'search for something completely different', "
            "'find me a pasta dish instead', 'start a new search', 'search again with "
            "different ingredients'). false for all other cases including 'show me more' requests.\n\n"
            "- 'wants_more_recommendations': true when the user wants MORE/DIFFERENT options "
            "from the SAME ingredient pool — they don't like the current batch and want the "
            "next set (e.g., 'show me more', 'I don't like these', 'other options', "
            "'more recipes please', 'what else do you have', 'next', 'any other suggestions', "
            "'give me different options', 'try something else', 'show me other options'). "
            "This is DIFFERENT from is_new_recipe_search — 'wants_more' keeps the same search "
            "results and shows the next batch; 'is_new_recipe_search' runs a brand new search.\n\n"
            "- 'wants_previous_recommendations': true when the user wants to go BACK to a "
            "previously shown batch of recommendations (e.g., 'go back to the first set', "
            "'show me the previous options', 'I liked the earlier ones better', "
            "'what were the first 3 again', 'back to the original recommendations'). "
            "If the user also picks a specific recipe from that earlier batch, set "
            "'previous_batch_selection' to 1, 2, or 3 (e.g., 'I want option 2 from the "
            "first batch' → previous_batch_selection: 2). Otherwise null.\n\n"
            "AFFIRMATIVE REPLIES AFTER RECIPE SUGGESTION:\n"
            "When the conversation shows the assistant just asked 'Would you like me to "
            "suggest some recipes?' (or similar offer), and the user replies with an "
            "affirmative ('yes', 'sure', 'sounds good', 'go ahead', 'please', 'yeah', "
            "'let's do it', 'ok'), classify as query_type: 'recipe'.\n\n"
            "- 'preference_action': 'view' when the user wants to SEE their current preferences "
            "(e.g., 'show my preferences', 'what are my settings?', 'what diet am I on?'). "
            "'update' when they are changing/adding/removing preferences. "
            "null when the query is not about preferences.\n\n"
            "Return only the JSON object and nothing else."
        )

        # Normalize messages to text format
        normalized_msgs = []
        for m in messages:
            if isinstance(m, dict):
                normalized_msgs.append(m)
            elif hasattr(m, "content") and hasattr(m, "type"):
                role = m.type if hasattr(m, "type") else "assistant"
                normalized_msgs.append({"role": role, "content": m.content})
            else:
                normalized_msgs.append({"role": "unknown", "content": str(m)})

        chat_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in normalized_msgs)

        resp = llm.invoke([
            SystemMessage(content=sys),
            HumanMessage(content=f"{schema_instruction}\n\nConversation:\n{chat_text}")
        ])

        _fallback = {
            "query_type": "general",
            "selected_recipe_number": None,
            "allergies": [], "removed_allergies": [], "clear_allergies": False,
            "restrictions": [], "removed_restrictions": [], "clear_restrictions": False,
            "cuisines": [], "removed_cuisines": [], "clear_cuisines": False,
            "excluded_food_types": [], "removed_excluded_food_types": [], "clear_excluded_food_types": False,
            "diet": None, "clear_diet": False,
            "skill": None,
            "clear_all_preferences": False,
            "wants_to_exit_flow": False,
            "is_new_recipe_search": False,
            "wants_more_recommendations": False,
            "wants_previous_recommendations": False,
            "previous_batch_selection": None,
            "preference_action": None,
        }

        try:
            data = json.loads(resp.content)
        except Exception as e:
            print(f"⚠️ classify_and_extract parse failed: {e}")
            return _fallback

        def to_list(v):
            if v is None:
                return []
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return [str(v).strip()] if str(v).strip() else []

        VALID_TYPES = ("pantry", "recipe", "general", "selection", "off_topic", "preference")
        qtype = data.get("query_type", "general")
        if qtype not in VALID_TYPES:
            qtype = "general"

        raw_num = data.get("selected_recipe_number")
        selected_num = None
        if qtype == "selection" and isinstance(raw_num, int) and 1 <= raw_num <= 3:
            selected_num = raw_num
        elif qtype == "selection":
            # LLM said 'selection' but no valid number — ask user to clarify
            qtype = "general"

        raw_prev_sel = data.get("previous_batch_selection")
        prev_sel = None
        if isinstance(raw_prev_sel, int) and 1 <= raw_prev_sel <= 3:
            prev_sel = raw_prev_sel

        return {
            "query_type": qtype,
            "selected_recipe_number": selected_num,
            "allergies": to_list(data.get("allergies")),
            "removed_allergies": to_list(data.get("removed_allergies")),
            "clear_allergies": bool(data.get("clear_allergies", False)),
            "restrictions": to_list(data.get("restrictions")),
            "removed_restrictions": to_list(data.get("removed_restrictions")),
            "clear_restrictions": bool(data.get("clear_restrictions", False)),
            "cuisines": to_list(data.get("cuisines")),
            "removed_cuisines": to_list(data.get("removed_cuisines")),
            "clear_cuisines": bool(data.get("clear_cuisines", False)),
            "excluded_food_types": to_list(data.get("excluded_food_types")),
            "removed_excluded_food_types": to_list(data.get("removed_excluded_food_types")),
            "clear_excluded_food_types": bool(data.get("clear_excluded_food_types", False)),
            "diet": data.get("diet"),
            "clear_diet": bool(data.get("clear_diet", False)),
            "skill": data.get("skill"),
            "clear_all_preferences": bool(data.get("clear_all_preferences", False)),
            "wants_to_exit_flow": bool(data.get("wants_to_exit_flow", False)),
            "is_new_recipe_search": bool(data.get("is_new_recipe_search", False)),
            "wants_more_recommendations": bool(data.get("wants_more_recommendations", False)),
            "wants_previous_recommendations": bool(data.get("wants_previous_recommendations", False)),
            "previous_batch_selection": prev_sel,
            "preference_action": data.get("preference_action"),
        }

    def pantry_info_sufficient(self, llm, user_text: str) -> dict:
        """
        Determine if pantry-related input has sufficient information for CRUD operations.
        Returns {'sufficient_info': True/False}.
        """
        schema_instruction = (
            "Return ONLY valid JSON matching this schema (no extra text):\n"
            "{\n"
            "  \"sufficient_info\": \"true\" | \"false\"\n"
            "}"
        )

        sys = (
            "You are a Pantry Assistant. "
            "Classify the user's input strictly as 'true' or 'false' under the key 'sufficient_info'.\n"
            "- 'true' means the input contains enough information for a pantry agent to perform a CRUD operation (add, update, remove, or view items) without asking for clarification.\n"
            "- 'false' means the input is insufficient and the pantry agent would need to ask the user for more details.\n"
            "Examples of sufficient inputs:\n"
            "  - 'Add 2 eggs'\n"
            "  - 'Remove milk from my pantry'\n"
            "  - 'Show all items in my pantry'\n"
            "Examples of insufficient inputs:\n"
            "  - 'I want to manage my pantry'\n"
            "  - 'Can you help me with pantry items?'\n"
            "Return only JSON, nothing else."
        )

        resp = llm.invoke([
            SystemMessage(content=sys),
            HumanMessage(content=f"{schema_instruction}\n\nUser text:\n{user_text}")
        ])

        # Normalize content
        raw_content = ""
        if isinstance(resp.content, str):
            raw_content = resp.content
        elif isinstance(resp.content, list):
            raw_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in resp.content
            )
        else:
            raw_content = str(resp)

        # Parse JSON and convert to boolean
        try:
            data = json.loads(raw_content)
            suff_str = data.get("sufficient_info", "false").lower()
            return {"sufficient_info": suff_str == "true"}
        except Exception as e:
            print(f"⚠️ pantry_info_sufficient parse failed: {e}\nRaw content:\n{raw_content}")
            return {"sufficient_info": False}

    def perform_quality_check(
        self, llm, recipe_text: str, user_prefs: dict, messages: list
    ) -> dict:
        """
        Perform user-context-aware quality check on final recipe.

        Args:
            llm: Language model
            recipe_text: Formatted recipe text
            user_prefs: User preferences (allergies, restrictions)
            messages: Full conversation history for context

        Returns:
            {"passed": bool, "issues": List[str], "score": int}
        """
        chat_context = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}"
                                  for m in messages[-10:]])  # Last 10 messages

        qa_instruction = """
        Review this recipe against user requirements with conversation context.
        CRITICAL checks:
      1. Contains NO allergens mentioned by user
      2. Complies with dietary restrictions
      3. Addresses user's original request intent

        Return ONLY valid JSON:
        {
            "passed": true/false,
            "issues": ["issue1", ...],
            "score": 0-100,
            "critical_failures": ["failure1", ...]
        }
        """

        context = f"""
        Conversation Context (last 10 messages):
        {chat_context}

        User Preferences:
        {json.dumps(user_prefs, indent=2)}

        Recipe to Review:
        {recipe_text}
        """

        response = llm.invoke([
            SystemMessage(content="You are a quality assurance agent reviewing recipes for user safety and satisfaction."),
            HumanMessage(content=f"{qa_instruction}\n\n{context}")
        ])

        try:
            result = json.loads(response.content)
            return {
                "passed": result.get("passed", False) and not result.get("critical_failures"),
                "issues": result.get("issues", []) + [f"CRITICAL: {cf}" for cf in result.get("critical_failures", [])],
                "score": result.get("score", 0)
            }
        except:
            return {"passed": True, "issues": ["QA parse error - defaulting to pass"], "score": 50}

