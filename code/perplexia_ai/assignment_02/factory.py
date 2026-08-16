"""Factory for creating Assignment 02 chat implementations."""

from enum import Enum
from typing import Type

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.assignment_02.part1 import QueryUnderstandingChat
from perplexia_ai.assignment_02.part1LangGraph import QueryUnderstandingLangGraphChat
from perplexia_ai.assignment_02.part2 import BasicToolsChat
from perplexia_ai.assignment_02.part2memory import BasicToolsMemoryChat
from perplexia_ai.assignment_02.part3 import MemoryChat

class Assignment02Mode(Enum):
    """Modes corresponding to the three parts of Assignment 02."""
    PART1_QUERY_UNDERSTANDING = "part1"  # Query classification and response formatting
    PART1_LANG_GRAPH = "part1LangGraph"  # Query understanding with language graph
    PART2_BASIC_TOOLS = "part2"
    PART2_BASIC_TOOLS_MEMORY = "part2memory"         # Adding calculator functionality
    PART3_MEMORY = "part3"              # Adding conversation memory

def create_chat_implementation(mode: Assignment02Mode) -> ChatInterface:
    """Create and return the appropriate chat implementation.
    
    Args:
        mode: Which part of Assignment 02 to run
        
    Returns:
        ChatInterface: The appropriate chat implementation
    
    Raises:
        ValueError: If mode is not recognized
    """
    implementations = {
        Assignment02Mode.PART1_QUERY_UNDERSTANDING: QueryUnderstandingChat,
        Assignment02Mode.PART1_LANG_GRAPH: QueryUnderstandingLangGraphChat,
        Assignment02Mode.PART2_BASIC_TOOLS: BasicToolsChat,
        Assignment02Mode.PART2_BASIC_TOOLS_MEMORY: BasicToolsMemoryChat,
        Assignment02Mode.PART3_MEMORY: MemoryChat
    }
    
    if mode not in implementations:
        raise ValueError(f"Unknown mode: {mode}")
    
    implementation_class = implementations[mode]
    implementation = implementation_class()
    return implementation
