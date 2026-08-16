"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator

class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""
    
    def __init__(self):
        self.llm = None
        self.memory = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for memory-enabled chat.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        - Set up conversation memory
        """
        # TODO: Students implement initialization
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and tools.
        
        Students should:
        - Use chat history for context
        - Handle follow-up questions
        - Use calculator when needed
        - Format responses appropriately
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement memory integration
        return "hello from part3"
