import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

# Support both the repository layout and a flattened release package.
runner_dir = Path(__file__).resolve().parent
assignments_root = runner_dir.parent
for source_dir in (runner_dir / "src", runner_dir):
    if source_dir.is_dir():
        sys.path.insert(0, str(source_dir))

for assignment in (2, 3, 4):
    for package in ("problem", "solution"):
        source_dir = assignments_root / f"assignment_{assignment:02d}" / package / "code"
        if source_dir.is_dir():
            sys.path.insert(0, str(source_dir))

# Parse command line arguments
available_assignments = [
    assignment
    for assignment in (2, 3, 4)
    if find_spec(f"perplexia_ai.assignment_{assignment:02d}") is not None
]

parser = argparse.ArgumentParser(description='Run Perplexia AI Assistant')
parser.add_argument('--assignment', type=int, choices=available_assignments,
                    default=available_assignments[0],
                    help='Which packaged assignment to run')
parser.add_argument('--mode', type=str, choices=['part1', 'part1LangGraph', 'part2', 'part2memory', 'part3'],
                    default='part1', help='Which part of the selected assignment to run')
parser.add_argument('--solution', '--solutions', dest='solution', action='store_true',
                    help='Run solution code instead of student code')
args = parser.parse_args()

# Import and run the app
from perplexia_ai.app import APP_THEME, create_demo

if __name__ == "__main__":
    demo = create_demo(assignment=args.assignment, mode_str=args.mode, use_solution=args.solution)
    #demo.launch(theme=APP_THEME)
    demo.launch(theme=APP_THEME, server_name="0.0.0.0", server_port=8000)
