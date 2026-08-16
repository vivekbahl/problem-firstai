import importlib
from typing import Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APP_THEME = gr.themes.Soft()


def _load_assignment(assignment: int, use_solution: bool) -> Tuple[object, object]:
    """Import the factory + presentation config for a single assignment.

    This module is shared across all assignments, so it deliberately holds NO
    assignment-specific content (titles, descriptions, example queries). Each
    assignment ships its own `app_config.py`, and only that assignment's package
    is present in its release — so a package never reveals other assignments.
    """
    pkg = f"assignment_{assignment:02d}"
    factory_pkg = (
        f"perplexia_ai.solutions.{pkg}.factory"
        if use_solution
        else f"perplexia_ai.{pkg}.factory"
    )
    try:
        factory = importlib.import_module(factory_pkg)
        # Presentation config always lives in the student package (shipped in both
        # the problem and solution releases), so it is imported the same way for both.
        config = importlib.import_module(f"perplexia_ai.{pkg}.app_config")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Assignment {assignment} is not available in this package."
        ) from exc
    return factory, config


def create_demo(assignment: int = 2, mode_str: str = "part1", use_solution: bool = False) -> gr.ChatInterface:
    """Create and return a Gradio demo with the specified assignment and mode.

    Args:
        assignment: Which assignment implementation to use
        mode_str: String representation of the mode ('part1', 'part2', 'part3')
        use_solution: If True, use solution implementation; otherwise use student code

    Returns:
        gr.ChatInterface: Configured Gradio chat interface
    """
    code_type = "Solution" if use_solution else "Student"

    factory, config = _load_assignment(assignment, use_solution)

    # Enum values are "part1"/"part2"/... so we can resolve the mode by value,
    # keeping this file free of any per-assignment mode names.
    mode_enum_cls = getattr(factory, f"Assignment{assignment:02d}Mode")
    try:
        mode = mode_enum_cls(mode_str)
    except ValueError as exc:
        valid = [m.value for m in mode_enum_cls]
        raise ValueError(f"Unknown mode: {mode_str}. Choose from: {valid}") from exc

    chat_interface = factory.create_chat_implementation(mode)
    chat_interface.initialize()

    title = f"Perplexia AI - {config.ASSIGNMENT_LABEL}: {config.TITLES[mode_str]} ({code_type})"
    description = config.DESCRIPTIONS[mode_str]
    examples = config.EXAMPLES[mode_str]

    def respond(message: str, history: List[Dict[str, str]]) -> str:
        """Process the message and return a response."""
        return chat_interface.process_message(message, history)

    demo = gr.ChatInterface(
        fn=respond,
        title=title,
        description=description,
        examples=examples,
        run_examples_on_click=True,
        chatbot=gr.Chatbot(show_label=False, height=560, layout="bubble"),
        textbox=gr.Textbox(
            show_label=False,
            placeholder="Ask a question, compare ideas, or run a quick calculation...",
            lines=1,
            max_lines=5,
            submit_btn="Send",
            stop_btn="Stop",
        ),
        fill_height=True,
        fill_width=True,
        show_progress="minimal",
    )
    return demo
