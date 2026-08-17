"""Part 2 - Basic Tools implementation with Memory and OPA Guardrails.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
- Maintain conversation memory across interactions
- Guardrail tool calls deterministically using Open Policy Agent (OPA)
"""

from ast import expr
import re
from typing import Dict, List, Optional, TypedDict, Any
import requests  # Added for HTTP requests to the OPA Server
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from perplexia_ai.core.tracing_arize import ArizeLangChainTracer
from langchain_openai import ChatOpenAI
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from perplexia_ai.tools.datetime_tool import DateTimeTool


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, including percentages (e.g. "120 * 15%").

    Args:
        expression: A math expression using digits, +, -, *, /, %, and parentheses.
    """
    return str(Calculator().evaluate_expression_percentage(expression))


@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return DateTimeTool.get_current_datetime()


@tool
def get_day_of_week(date_str: str) -> str:
    """Get the day of the week for a given calendar date.

    Args:
        date_str: A date as text, e.g. "January 17, 2026" or "2026-01-17".
    """
    return DateTimeTool.get_day_of_week(date_str)


class QueryState(TypedDict, total=False):
    """State passed between nodes in the query understanding graph."""
    message: str
    query_type: str
    response: str
    messages: List[Dict[str, str]]  # Changed from 'memory' to 'messages' for LangGraph compatibility
    thread_id: str  # Added for conversation threading
    # --- Added OPA Guardrail State Parameters ---
    guardrail_approved: bool
    guardrail_feedback: str
    pending_tool_name: str
    pending_tool_args: dict
    
class BasicToolsMemoryGuardrailsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality with memory support and OPA."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.graph = None
        self.memory = InMemorySaver()  # Added memory saver
        self.thread_id = "default_thread"  # Default thread ID
        # Added configurable OPA endpoint URL
        self.opa_url = "http://localhost:8181/v1/data/ai/tool_guard/allow"
        
    def initialize(self) -> None:
        """Initialize components for basic tools with memory support."""
        tracer = ArizeLangChainTracer()
        tracer.setup_instrumentation()
                
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        # Tools bound so the model can decide when/how to call them, with structured args
        self.tools = [calculate, get_current_datetime, get_day_of_week]
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
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
        graph_builder.add_node("context_manager", self._context_manager_node)
        graph_builder.add_node("classify", self._classify_node)
        graph_builder.add_node("opa_guardrail", self._opa_guardrail_node)
        graph_builder.add_node("tool_call", self._tool_call_node)
        graph_builder.add_node("respond", self._respond_node)
        
        graph_builder.add_edge(START, "context_manager")
        graph_builder.add_edge("context_manager", "classify")
        
        # Route from classification node
        graph_builder.add_conditional_edges(
            "classify",
            self._route_decision,
            {
                "calculation": "opa_guardrail",
                "datetime": "opa_guardrail",
                "respond": "respond",
                "follow_up": "respond"
            },
        )
        
        # Route conditionally from OPA Guardrail node based on compliance evaluation
        graph_builder.add_conditional_edges(
            "opa_guardrail",
            self._route_guardrail_decision,
            {
                "allow": "tool_call",
                "deny": "respond"
            }
        )
        
        # FIX: Tool call returns to respond node to safely format structural data for the user
        graph_builder.add_edge("tool_call", "respond")
        graph_builder.add_edge("respond", END)
        
        # Compile with memory checkpointing
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None, thread_id: Optional[str] = None) -> str:
        """Process a message with calculator support and memory."""
        if thread_id:
            self.thread_id = thread_id
            
        # Prepare the input state dictionary matching QueryState definition
        input_state = {
            "message": message,
            "messages": chat_history or [],
            "thread_id": self.thread_id,
            "guardrail_approved": False,
            "guardrail_feedback": "",
            "pending_tool_name": "",
            "pending_tool_args": {},
            "response": ""
        }
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # FIX: Added full graph invocation handling and clean response extraction
        try:
            final_state = self.graph.invoke(input_state, config=config)
            return final_state.get("response", "No response generated.")
        except Exception as e:
            return f"Graph Execution Error: {str(e)}"

    def _context_manager_node(self, state: QueryState) -> Dict[str, Any]:
        """Manages conversation context buffers and appends history arrays."""
        messages = state.get("messages", [])
        return {"messages": messages}

    def _classify_node(self, state: QueryState) -> Dict[str, Any]:
        """Classify the user's message into one of the response buckets."""
        context = self._extract_context(state.get("messages", []))
        
        response = self.llm_with_tools.invoke(
            f"System: {self.query_classifier_prompt}\nContext: {context}\nUser Message: {state['message']}"
        )
        
        output = {"query_type": "respond", "pending_tool_name": "", "pending_tool_args": {}}
        
        # ✅ FIX: Explicitly extract the first item from the tool_calls list
        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]  # Grab the first structural tool call dictionary
            output["pending_tool_name"] = tool_call["name"]
            output["pending_tool_args"] = tool_call["args"]
            
            if tool_call["name"] == "calculate":
                output["query_type"] = "calculation"
            elif tool_call["name"] in ["get_current_datetime", "get_day_of_week"]:
                output["query_type"] = "datetime"
        else:
            raw_label = response.content.strip().lower()
            if "mathematical" in raw_label or "calculate" in raw_label:
                output["query_type"] = "calculation"
                output["pending_tool_name"] = "calculate"
                output["pending_tool_args"] = {"expression": state["message"]}
            elif "datetime" in raw_label:
                output["query_type"] = "datetime"
                output["pending_tool_name"] = "get_current_datetime"
                output["pending_tool_args"] = {}
            elif "follow-up" in raw_label:
                output["query_type"] = "follow_up"
            else:
                output["query_type"] = "respond"

        return output


    def _opa_guardrail_node(self, state: QueryState) -> Dict[str, Any]:
        """Intercept tool arguments and query OPA server before running execution steps."""
        tool_name = state.get("pending_tool_name")
        tool_args = state.get("pending_tool_args", {})

        if not tool_name:
            return {
                "guardrail_approved": True, 
                "guardrail_feedback": "",
                "pending_tool_name": "",
                "pending_tool_args": {}
            }

        payload = {
            "input": {
                "tool_name": tool_name,
                "arguments": tool_args,
                "thread_id": state.get("thread_id", self.thread_id)
            }
        }

        try:
            response = requests.post(self.opa_url, json=payload, timeout=2.0)
            opa_result = response.json().get("result", False)
        except Exception:
            opa_result = False

        if opa_result is True:
            return {
                "guardrail_approved": True, 
                "guardrail_feedback": "",
                "pending_tool_name": tool_name,
                "pending_tool_args": tool_args
            }
        else:
            # ✅ FIX: Explicitly nullify the tool properties to prevent downstream execution steps
            return {
                "guardrail_approved": False,
                "guardrail_feedback": f"Guardrail blocked tool '{tool_name}' execution due to strict OPA compliance policy constraints.",
                "pending_tool_name": "",
                "pending_tool_args": {},
                "query_type": "respond"  # Re-route the workflow to the respond text block path
            }


    def _tool_call_node(self, state: QueryState) -> Dict[str, Any]:
        """Executes the approved tool using the tool registry reference maps."""
        tool_name = state.get("pending_tool_name")
        tool_args = state.get("pending_tool_args", {})

        if not tool_name or tool_name not in self.tools_by_name:
            return {"response": f"Error: Tool '{tool_name}' missing from structural registry mapping."}

        try:
            target_tool = self.tools_by_name[tool_name]
            tool_output = target_tool.invoke(tool_args)
            return {"response": f"Tool execution result: {str(tool_output)}"}
        except Exception as e:
            return {"response": f"Tool infrastructure crash: {str(e)}"}

    def _respond_node(self, state: QueryState) -> Dict[str, Any]:
        """Constructs and formats final response string based on previous tracking context."""
        # If guardrail caught a violation, exit gracefully with feedback message
        if state.get("pending_tool_name") and not state.get("guardrail_approved", False):
            return {"response": state.get("guardrail_feedback", "Access Denied by OPA Guardrail.")}

        # If tool execution already generated a response, maintain it
        if state.get("response"):
            return {"response": state["response"]}

        # Standard structural generation prompt fallback
        context = self._extract_context(state.get("messages", []))
        q_type = state.get("query_type", "factual")
        prompt_tmpl = self.response_prompts.get(q_type, "User: {message}")
        formatted_prompt = prompt_tmpl.format(context=context, message=state["message"])
        llm_response = self.llm.invoke(formatted_prompt)
        return {"response": llm_response.content}

    def _route_decision(self, state: QueryState) -> str:
        """Determines if the flow goes to guardrails or direct response nodes."""
        q_type = state.get("query_type", "respond")
        if q_type in ["calculation", "datetime"]:
            return q_type
        if q_type == "follow_up":
            return "follow_up"
        return "respond"

    def _route_guardrail_decision(self, state: QueryState) -> str:
        """Decides if the tool call execution is allowed based on OPA outcome state."""
        if state.get("guardrail_approved", False) is True:
            return "allow"
        return "deny"

    def _extract_context(self, messages: List[Dict[str, str]]) -> str:
        """Extract conversational history into text block summaries."""
        if not messages:
            return "No previous context available."

        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Handle content-block format: [{"text": "...", "type": "text"}, ...]
                text = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                text = str(content)

            lines.append(f"{role}: {text}")

        return "\n".join(lines)

    def _sanitize_expression(self, expr: str) -> str:
        # Strip currency symbols, commas, and any word characters/punctuation
        # that aren't part of a valid arithmetic expression.
        cleaned = expr.replace("$", "").replace(",", "")
        cleaned = re.sub(r"[^\d\.\+\-\*/%\(\)\s]", "", cleaned)
        return cleaned.strip()