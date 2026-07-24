"""
Full pipeline check: exercises the EXACT code paths run_experiment.py uses
(GeminiLLMClient.generate() for chat turns, .generate_json() for the
exploration-tree branches), not just a bare generate_content call.

Doesn't touch experiment_results.jsonl or fixation.db.

Run from the project root, with your venv active:
    python test_pipeline.py
"""
import sys
sys.path.insert(0, ".")

from app.config import settings
from app.llm_client import GeminiLLMClient, EXPLORATION_TREE_SCHEMA_HINT

print(f"Testing with model: {settings.gemini_model}\n")

client = GeminiLLMClient(settings.gemini_api_key, settings.gemini_model)

# --- 1. Multi-turn chat generate() ---
print("=== Testing generate() (chat, multi-turn) ===")
history = [
    {"role": "user", "content": "A detective finds a locked door in an old mansion."},
    {"role": "assistant", "content": "The detective examined the door, noting its unusual bronze hinges."},
    {"role": "user", "content": "What does she do next?"},
]
try:
    reply = client.generate(history)
    print("SUCCESS")
    print("Reply:", reply[:200])
except Exception as e:
    print("FAILED:", e)

# --- 2. Structured JSON generate_json() for exploration tree ---
print("\n=== Testing generate_json() (structured exploration tree) ===")
tree_prompt = (
    "Conversation so far:\n"
    "User: A detective finds a locked door in an old mansion.\n"
    "Assistant: The detective examined the door, noting its unusual bronze hinges.\n\n"
    + EXPLORATION_TREE_SCHEMA_HINT
)
try:
    tree = client.generate_json([{"role": "user", "content": tree_prompt}])
    print("SUCCESS — got keys:", list(tree.keys()))
    for k, v in tree.items():
        print(f"  {k}: {v.get('title', '?')}")
except Exception as e:
    print("FAILED:", e)
