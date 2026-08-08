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
        tree = _coerce_tree(tree)

        normalized = {}
        for cat in BRANCH_CATEGORIES:
            node = tree.get(cat) or {}
            normalized[cat] = {
                "title": _pick(node, ("title", "name", "heading", "label",
                                     "direction", "summary"))
                         or cat.replace("_", " ").title(),
                "prompt": _pick(node, ("prompt", "description", "text",
                                      "detail", "details", "content",
                                      "idea", "body"))
                          or f"Explore a {cat.replace(chr(95), chr(32))} direction.",
            }
        return normalized


def _pick(node, keys):
    """First non-empty string value among `keys`, else None."""
    if not isinstance(node, dict):
        return str(node).strip() or None if node else None
    for k in keys:
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _coerce_tree(tree):
    """Normalize whatever the model returned into {category: node}.

    Handles: the expected dict keyed by category; a bare list of branch
    objects; and a wrapper dict such as {"branches": [...]} or
    {"directions": {...}}. Falls back to positional assignment.
    """
    if isinstance(tree, str):
        try:
            import json as _json
            tree = _json.loads(tree)
        except Exception:
            return {}

    # already keyed by category?
    if isinstance(tree, dict) and any(c in tree for c in BRANCH_CATEGORIES):
        return tree

    # wrapper dict: unwrap the first list, or the first dict of dicts
    if isinstance(tree, dict):
        for v in tree.values():
            if isinstance(v, list) and v:
                tree = v
                break
            if isinstance(v, dict) and any(c in v for c in BRANCH_CATEGORIES):
                return v
        else:
            # dict of five arbitrary keys -> take values positionally
            vals = [v for v in tree.values() if isinstance(v, dict)]
            if vals:
                tree = vals

    if isinstance(tree, list):
        coerced = {}
        for idx, item in enumerate(tree):
            cat = None
            if isinstance(item, dict):
                cat = _pick(item, ("category", "type", "id", "key"))
            if cat not in BRANCH_CATEGORIES:
                cat = (BRANCH_CATEGORIES[idx]
                       if idx < len(BRANCH_CATEGORIES) else None)
            if cat:
                coerced[cat] = item if isinstance(item, dict) else {}
        return coerced

    return {}
