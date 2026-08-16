"""Presentation config for the Assignment 02 Gradio demo.

Kept in the assignment package (not the shared `app.py`) so that a release only
ever contains its own assignment's titles, descriptions, and example queries.
"""

ASSIGNMENT_LABEL = "Assignment 02"

TITLES = {
    "part1": "Query Understanding",
    "part1LangGraph": "Query Understanding (LangGraph)",
    "part2": "Basic Tools",
    "part2memory": "Basic Tools (Memory)",
    "part3": "Memory",
}

DESCRIPTIONS = {
    "part1": "Your intelligent AI assistant that can understand different types of questions and format responses accordingly.",
    "part1LangGraph": "Query understanding implemented as a LangGraph classify-then-respond workflow.",
    "part2": "Your intelligent AI assistant that can answer questions, perform calculations, and format responses.",
    "part2memory": "Your intelligent AI assistant that can answer questions, perform calculations, and maintain conversation context.",
    "part3": "Your intelligent AI assistant that can answer questions, perform calculations, and maintain conversation context.",
}

_EXAMPLES = [
    "What is machine learning?",
    "Compare SQL and NoSQL databases",
    "If I have a dinner bill of $120, what would be a 15% tip?",
    "What about 20%?",
    "Hi I need help with my homework, can you help me?",
]

# Assignment 02 uses the same example set across all parts.
EXAMPLES = {
    "part1": _EXAMPLES,
    "part1LangGraph": _EXAMPLES,
    "part2": _EXAMPLES,
    "part2memory": _EXAMPLES,
    "part3": _EXAMPLES,
}
