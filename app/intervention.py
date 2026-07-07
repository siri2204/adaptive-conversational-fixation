"""
When a strategy decides to intervene, this module produces the actual content:
a structured exploration tree with five branch categories (per the proposal):
abstract reframing, contradictory perspective, adjacent domain, unconventional
alternative, and speculative future direction.

The tree is a dict keyed by category -> {title, prompt}. The frontend / caller
shows the user all five and lets them pick one; the selected branch's `prompt`
becomes the seed for the next conversational turn, redirecting the trajectory
without discarding the existing history.
"""
from __future__ import annotations

from app.llm_client import LLMClient, EXPLORATION_TREE_SCHEMA_HINT

BRANCH_CATEGORIES = [
    "abstract_reframing",
    "contradictory_perspective",
    "adjacent_domain",
    "unconventional_alternative",
    "speculative_future",
]


class ExplorationTreeGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, conversation_history: list[dict], task_type: str) -> dict:
        """conversation_history: list of {"role": "user"|"assistant", "content": str}"""
        system_context = (
            f"You are assisting a creative co-creation task of type '{task_type}'. "
            "The conversation below may be converging on one narrow idea. Your job is to "
            "propose five genuinely different directions the user could take instead, "
            "each clearly distinct from the others and from where the conversation is now. "
            + EXPLORATION_TREE_SCHEMA_HINT
        )
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in conversation_history[-10:])
        messages = [
            {"role": "user", "content": system_context + "\n\nConversation so far:\n" + transcript}
        ]
        tree = self.llm_client.generate_json(messages)
        # Defensive normalization in case the model omits a key or adds extras.
        normalized = {}
        for cat in BRANCH_CATEGORIES:
            node = tree.get(cat) or {}
            normalized[cat] = {
                "title": node.get("title", cat.replace("_", " ").title()),
                "prompt": node.get("prompt", f"Explore a {cat.replace('_', ' ')} direction."),
            }
        return normalized
