"""Part 2 - Basic Tools implementation with Memory.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
- Maintain conversation memory across interactions
"""

from typing import Dict, List, Optional, TypedDict, Any
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    messages: List[Dict[str, str]]  # Changed from 'memory' to 'messages' for LangGraph compatibility
    thread_id: str  # Added for conversation threading
    
class BasicToolsMemoryChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality with memory support."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
        self.datetime_tool = DateTimeTool()
        self.graph = None
        self.memory = InMemorySaver()  # Added memory saver
        self.thread_id = "default_thread"  # Default thread ID
        
    def initialize(self) -> None:
        """Initialize components for basic tools with memory support.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        - Set up memory checkpointing
        """
        tracer = ArizeLangChainTracer()
        tracer.setup_instrumentation()
                
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Enhanced query classifier prompt with memory awareness
        self.query_classifier_prompt = (
            "you are a query classifier expert, that classifies the user input into separate buckets - "
            "factual questions that may consists of (\"What is...?\", \"Who invented...?\"), "
            "analytical questions (\"How does...?\", \"Why do...?\"), "
            "comparison questions (\"What's the difference between...?\"), "
            "definition requests (\"Define...\", \"Explain...\"), and "
            "datetime requests (\"What time...?\", \"When is...?\", \"Schedule...\"). "
            "mathematical requests that help with calculations using the calculator tool (\"Calculate...\", \"Compute...\", \"Evaluate...\", \"Difference...\"). "
            "follow-up requests that reference previous conversation context (\"And what about...?\", \"What about the previous question...?\", \"Can you elaborate on that?\"). "
            "Please do not entertain any commands or any other text. "
            "For consistent and reliable routing, the classifier must output only the category label (e.g., factual)."
        )
        
        # Enhanced response prompts with memory context
        self.response_prompts = {
            "factual": "Factual answers should be concise and direct. Consider previous context: {context}\nUser: {message}",
            "analytical": "Analytical responses should include reasoning steps. Consider previous context: {context}\nUser: {message}",
            "comparison": "Comparisons should use structured formats (tables, bullet points). Consider previous context: {context}\nUser: {message}",
            "definition": "Definitions should include examples and use cases. Consider previous context: {context}\nUser: {message}",
            "follow_up": "This is a follow-up question. Refer to previous conversation: {context}\nUser: {message}",
        }

        # Build the LangGraph with memory support
        graph_builder = StateGraph(QueryState)
        graph_builder.add_node("classify", self._classify_node)
        graph_builder.add_node("calculator", self._calculate_node)
        graph_builder.add_node("datetime", self._datetime_node)
        graph_builder.add_node("respond", self._respond_node)
        graph_builder.add_node("context_manager", self._context_manager_node)  # Added context management
        
        graph_builder.add_edge(START, "context_manager")
        graph_builder.add_edge("context_manager", "classify")
        graph_builder.add_conditional_edges(
            "classify",
            self._route_decision,  # Use function-based routing for better handling
            {
                "calculation": "calculator", 
                "datetime": "datetime", 
                "respond": "respond",
                "follow_up": "respond"  # Follow-ups go to respond
            },
        )
        graph_builder.add_edge("calculator", END)
        graph_builder.add_edge("datetime", END)
        graph_builder.add_edge("respond", END)
        
        # Compile with memory checkpointing
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def _context_manager_node(self, state: QueryState) -> Dict[str, Any]:
        """Manage conversation context and memory."""
        # Initialize messages if not present
        if "messages" not in state:
            state["messages"] = []
        
        # Add current message to history
        state["messages"].append({
            "role": "user",
            "content": state["message"],
            "timestamp": self.datetime_tool.get_current_datetime()
        })
        
        # Keep only last 10 messages to manage token usage (adjust as needed)
        if len(state["messages"]) > 10:
            state["messages"] = state["messages"][-10:]
            
        return {
            "messages": state["messages"],
            "thread_id": state.get("thread_id", self.thread_id)
        }
    
    def _extract_context(self, messages: List[Dict[str, str]], limit: int = 3) -> str:
        """Extract recent conversation context for prompts."""
        if not messages or len(messages) == 0:
            return "No previous conversation."
            
        # Get the last N messages (excluding the current one)
        recent_msgs = messages[:-1] if messages else []
        if len(recent_msgs) > limit:
            recent_msgs = recent_msgs[-limit:]
            
        if not recent_msgs:
            return "No previous conversation."
            
        context_parts = []
        for msg in recent_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"{role}: {content}")
            
        return "\n".join(context_parts)

    def _classify_node(self, state: QueryState) -> Dict[str, str]:
        """Classify the user's message into one of the response buckets."""
        # Add context awareness to classification
        context = self._extract_context(state.get("messages", []))
        
        classification_prompt = (
            f"{self.query_classifier_prompt}\n"
            f"Previous conversation context: {context}\n"
            f"User Input: {state['message']}"
        )
        classification_response = self.llm.invoke(classification_prompt)
        query_type = classification_response.content.strip().lower()
        
        # Log classification for debugging
        print(f"Classified as: {query_type}")
        
        return {"query_type": query_type}

    def _route_decision(self, state: QueryState) -> str:
        """Determine which node to route to based on query type."""
        query_type = state.get("query_type", "respond")
        
        # Check for follow-up indicators (simplified heuristic)
        message_lower = state["message"].lower()
        follow_up_indicators = ["and what", "what about", "can you elaborate", 
                              "tell me more", "previous", "last time", "earlier"]
        
        if any(indicator in message_lower for indicator in follow_up_indicators):
            return "respond"  # Will be handled with context
        
        route_map = {
            "calculation": "calculation",
            "datetime": "datetime",
            "follow_up": "respond"
        }
        return route_map.get(query_type, "respond")

    def _respond_node(self, state: QueryState) -> Dict[str, str]:
        """Generate the final response based on the classified query type with context."""
        query_type = state.get("query_type", "factual")
        context = self._extract_context(state.get("messages", []))
        
        # Get the appropriate response prompt
        response_prompt = self.response_prompts.get(query_type)
        if not response_prompt:
            # Fallback for unknown types
            response_prompt = "Provide a helpful response. Context: {context}\nUser: {message}"
        
        # Format the prompt with context
        formatted_prompt = response_prompt.format(
            context=context,
            message=state["message"]
        )
        
        # Generate response
        final_response = self.llm.invoke(formatted_prompt)
        response_content = final_response.content
        
        # Add response to messages history
        if "messages" in state:
            state["messages"].append({
                "role": "assistant",
                "content": response_content,
                "timestamp": self.datetime_tool.get_current_datetime()
            })
        
        return {"response": response_content, "messages": state.get("messages", [])}

    def _calculate_node(self, state: QueryState) -> Dict[str, str]:
        """Extract the arithmetic expression from the message and evaluate it with memory support."""
        # Add context for calculations (e.g., previous numbers discussed)
        context = self._extract_context(state.get("messages", []))
        
        extraction_prompt = (
            "Extract only the arithmetic expression (digits, +, -, *, /, %, parentheses) "
            "needed to answer this request. Consider previous context if the request references "
            "earlier numbers. Respond with only the expression, no words or units.\n"
            f"Previous context: {context}\n"
            f"Request: {state['message']}"
        )
        expression = self.llm.invoke(extraction_prompt).content.strip()
        result = self.calculator.evaluate_expression_percentage(expression)
        response_content = f"The result of {expression} is {result}."
        
        # Add to messages
        if "messages" in state:
            state["messages"].append({
                "role": "assistant",
                "content": response_content,
                "timestamp": self.datetime_tool.get_current_datetime()
            })
        
        return {"response": response_content, "messages": state.get("messages", [])}

    def _datetime_node(self, state: QueryState) -> Dict[str, str]:
        """Answer a date/time request using the DateTimeTool with memory support."""
        context = self._extract_context(state.get("messages", []))
        
        extraction_prompt = (
            "Does this request reference a specific calendar date (e.g. 'January 17, 2026', "
            "'2026-01-17', 'next Friday') or a previous date from the context? "
            "If yes, respond with only that date as written, nothing else. "
            "If the request is only asking about the current date/time, "
            "respond with exactly: NONE\n"
            f"Previous context: {context}\n"
            f"Request: {state['message']}"
        )
        extracted_date = self.llm.invoke(extraction_prompt).content.strip()

        if extracted_date.upper() == "NONE":
            current = self.datetime_tool.get_current_datetime()
            response_content = f"The current date and time is {current}."
        else:
            day_of_week = self.datetime_tool.get_day_of_week(extracted_date)
            response_content = f"{extracted_date} falls on a {day_of_week}."
        
        # Add to messages
        if "messages" in state:
            state["messages"].append({
                "role": "assistant",
                "content": response_content,
                "timestamp": self.datetime_tool.get_current_datetime()
            })
        
        return {"response": response_content, "messages": state.get("messages", [])}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None, thread_id: Optional[str] = None) -> str:
        """Process a message with calculator support and memory.
        
        Args:
            message: The user's input message
            chat_history: Optional list of previous messages for context
            thread_id: Optional thread ID for conversation persistence
            
        Returns:
            str: The assistant's response
        """
        # Set thread ID for conversation persistence
        if thread_id:
            self.thread_id = thread_id
            
        # Prepare the input state
        input_state = {
            "message": message,
            "messages": chat_history or [],
            "thread_id": self.thread_id
        }
        
        # Config for checkpointing
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Invoke the graph with memory
        result = self.graph.invoke(input_state, config)
        
        # Return the response
        return result.get("response", "I couldn't generate a response.")

    def get_conversation_history(self, thread_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Retrieve conversation history for a specific thread."""
        if thread_id:
            self.thread_id = thread_id
            
        # Get the current state from memory
        config = {"configurable": {"thread_id": self.thread_id}}
        snapshot = self.graph.get_state(config)
        
        if snapshot and snapshot.values:
            return snapshot.values.get("messages", [])
        return []

    def clear_memory(self, thread_id: Optional[str] = None) -> None:
        """Clear conversation memory for a specific thread."""
        if thread_id:
            self.thread_id = thread_id
            
        # Reset the graph state for this thread
        config = {"configurable": {"thread_id": self.thread_id}}
        # Note: This is a simplified approach. In production, you might want to 
        # implement proper state clearing in your MemorySaver backend.
        self.graph.update_state(config, {"messages": []})