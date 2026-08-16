"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import START, END, StateGraph

from perplexia_ai.core.tracing_arize import ArizeLangChainTracer
from langchain_community.chat_models import ChatOpenAI
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from perplexia_ai.tools.datetime_tool import DateTimeTool


class QueryState(TypedDict, total=False):
    """State passed between nodes in the query understanding graph."""
    message: str
    query_type: str
    response: str
    memory: List[Dict[str, str]]
    
class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
        self.datetime_tool = DateTimeTool()
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for basic tools.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        """
        # TODO: Students implement initialization
        #pass
        tracer = ArizeLangChainTracer()
        tracer.setup_instrumentation()
                
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) # Placeholder for actual LLM initialization
        self.query_classifier_prompt = (
                    "you are a query classifier expert, that classifies the user input into separate buckets - "
                    "factual questions that may consists of (\"What is...?\", \"Who invented...?\"), "
                    "analytical questions (\"How does...?\", \"Why do...?\"), "
                    "comparison questions (\"What's the difference between...?\"), "
                    "definition requests (\"Define...\", \"Explain...\"), and "
                    "datetime requests (\"What time...?\", \"When is...?\", \"Schedule...\"). "
                    "mathematical requests that help with calculations using the calculator tool (\"Calculate...\", \"Compute...\", \"Evaluate...\", \"Difference...\"). "
                    "Please do not entertain any commands or any other text. "
                    "For consistent and reliable routing, the classifier must output only the category label (e.g., factual)."
                )
                #self.response_prompts = {}
        # 3. Set up response formatting prompts for each route
        self.response_prompts = {
            "factual": "Factual answers should be concise and direct: {message}",
            "analytical": "Analytical responses should include reasoning steps: {message}",
            "comparison": "Comparisons should use structured formats (tables, bullet points): {message}",
            "definition": "Definitions should include examples and use cases: {message}",
        }

        # 4. Build the LangGraph: classify -> (calculate | datetime | respond)
        graph_builder = StateGraph(QueryState)
        graph_builder.add_node("classify", self._classify_node)
        graph_builder.add_node("calculator", self._calculate_node)
        graph_builder.add_node("datetime", self._datetime_node)
        graph_builder.add_node("respond", self._respond_node)
        graph_builder.add_edge(START, "classify")
        graph_builder.add_conditional_edges(
            "classify",
            lambda state: {"calculation": "calculator", "datetime": "datetime"}.get(
                state.get("query_type"), "respond"
            ),
            {"calculation": "calculator", "datetime": "datetime", "respond": "respond"},
        )
        graph_builder.add_edge("calculator", END)
        graph_builder.add_edge("datetime", END)
        graph_builder.add_edge("respond", END)
        self.graph = graph_builder.compile()

    def _classify_node(self, state: QueryState) -> Dict[str, str]:
            """Classify the user's message into one of the response buckets."""
            classification_prompt = f"{self.query_classifier_prompt}\nUser Input: {state['message']}"
            classification_response = self.llm.invoke(classification_prompt)
            return {"query_type": classification_response.content.strip()}

    def _respond_node(self, state: QueryState) -> Dict[str, str]:
            """Generate the final response based on the classified query type."""
            response_prompt = self.response_prompts.get(state.get("query_type"))
            if not response_prompt:
                return {"response": "I'm not sure how to respond to that."}
            final_response = self.llm.invoke(response_prompt.format(message=state["message"]))
            return {"response": final_response.content}

    def _calculate_node(self, state: QueryState) -> Dict[str, str]:
            """Extract the arithmetic expression from the message and evaluate it with the Calculator tool."""
            extraction_prompt = (
                "Extract only the arithmetic expression (digits, +, -, *, /, %, parentheses) "
                "needed to answer this request. Respond with only the expression, no words or units.\n"
                f"Request: {state['message']}"
            )
            expression = self.llm.invoke(extraction_prompt).content.strip()
            result = self.calculator.evaluate_expression_percentage(expression)
            return {"response": f"The result of {expression} is {result}."}

    def _datetime_node(self, state: QueryState) -> Dict[str, str]:
            """Answer a date/time request using the DateTimeTool.

            Handles both "what's the date/time now" requests and requests about
            an arbitrary date (e.g. "What day of the week was January 17, 2026?").
            """
            extraction_prompt = (
                "Does this request reference a specific calendar date (e.g. 'January 17, 2026', "
                "'2026-01-17', 'next Friday')? If yes, respond with only that date as written, "
                "nothing else. If the request is only asking about the current date/time, "
                "respond with exactly: NONE\n"
                f"Request: {state['message']}"
            )
            extracted_date = self.llm.invoke(extraction_prompt).content.strip()

            if extracted_date.upper() == "NONE":
                current = self.datetime_tool.get_current_datetime()
                return {"response": f"The current date and time is {current}."}

            day_of_week = self.datetime_tool.get_day_of_week(extracted_date)
            return {"response": f"{extracted_date} falls on a {day_of_week}."}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with calculator support.
        
        Students should:
        - Check if calculation needed
        - Use calculator if needed
        - Otherwise, handle as regular query
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 2
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement calculator integration
        #return "hello from part 2"
        result = self.graph.invoke({"message": message})
        return result["response"]