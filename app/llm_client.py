"""
LLM client abstraction so the rest of the app never talks to Gemini directly.

  - GeminiLLMClient: thin wrapper around google-generativeai.
  - MockLLMClient: canned/echo-style responses so you can develop and unit
    test the FastAPI app, fixation detection, and intervention strategies
    with zero network calls and zero API cost.

generate() -> plain string reply (for normal conversation turns)
generate_json() -> parsed dict (used for structured exploration-tree output)
"""
from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod

from app.config import settings

EXPLORATION_TREE_SCHEMA_HINT = """
Return ONLY valid JSON (no markdown fences, no commentary) with exactly these keys,
each mapping to an object with "title" (<=8 words) and "prompt" (a 1-3 sentence
continuation prompt written to the user, inviting them to explore that direction):

{
  "abstract_reframing": {"title": "...", "prompt": "..."},
  "contradictory_perspective": {"title": "...", "prompt": "..."},
  "adjacent_domain": {"title": "...", "prompt": "..."},
  "unconventional_alternative": {"title": "...", "prompt": "..."},
  "speculative_future": {"title": "...", "prompt": "..."}
}
"""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        ...

    def generate_json(self, messages: list[dict]) -> dict:
        raw = self.generate(messages)
        return _parse_json_relaxed(raw)


def _parse_json_relaxed(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class GeminiLLMClient(LLMClient):
    """Uses the current `google-genai` SDK (pip install google-genai).

    Note: the older `google-generativeai` package (import google.generativeai as genai)
    is deprecated as of Nov 2025 — don't use it. This client uses the new
    `from google import genai` / `client.models.generate_content(...)` API.

    Free-tier Gemini API keys are rate-limited to a handful of requests per
    minute (e.g. 5 RPM for gemini-2.5-flash as of mid-2026). Rather than
    crashing on the first 429, this client automatically waits and retries,
    respecting the server's suggested retryDelay when it provides one.
    """

    def __init__(self, api_key: str, model_name: str, max_retries: int = 6):
        from google import genai  # lazy import

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def _with_retry(self, fn):
        import time
        from google.genai import errors as genai_errors

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except genai_errors.ClientError as e:
                last_exc = e
                is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
                if not is_rate_limit:
                    raise
                if _is_daily_quota_exhausted(e):
                    # Retrying within the same day cannot succeed — a per-minute
                    # backoff is pointless against a per-day cap. Fail immediately
                    # with a clear message instead of hanging for minutes.
                    raise RuntimeError(
                        "Gemini daily request quota exhausted for this API key/model. "
                        "This will not resolve by retrying — it resets on Google's daily "
                        "cycle (roughly 24h from your first request today). Switch "
                        "FIXATION_LLM_BACKEND=mock to keep developing today, or enable "
                        "billing on your Google Cloud project to remove the daily cap. "
                        f"Original error: {e}"
                    ) from e
                wait_seconds = _extract_retry_delay(e) or (2 ** attempt) * 5
                print(
                    f"[GeminiLLMClient] Rate limited (attempt {attempt + 1}/{self.max_retries}). "
                    f"Waiting {wait_seconds:.0f}s before retrying..."
                )
                time.sleep(wait_seconds)
        raise last_exc

    def generate(self, messages: list[dict]) -> str:
        from google.genai import types  # lazy import

        def _call():
            history = []
            for m in messages[:-1]:
                role = "model" if m["role"] == "assistant" else "user"
                history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
            chat = self.client.chats.create(model=self.model_name, history=history)
            response = chat.send_message(messages[-1]["content"])
            return response.text

        return self._with_retry(_call)

    def generate_json(self, messages: list[dict]) -> dict:
        """Single-shot call (not chat) with JSON mode enabled, since the
        exploration-tree prompt already embeds the full transcript itself."""
        from google.genai import types  # lazy import

        def _call():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages[-1]["content"],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return _parse_json_relaxed(response.text)

        return self._with_retry(_call)


def _is_daily_quota_exhausted(exc) -> bool:
    """Distinguishes a per-day quota cap (retrying is futile until tomorrow)
    from a per-minute rate limit (retrying shortly will work). Google's error
    body includes a quotaId like 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'
    for the former."""
    return "PerDay" in str(exc)


def _extract_retry_delay(exc) -> float | None:
    """Best-effort parse of the server-suggested retry delay (e.g. '47s') out
    of a google-genai ClientError's details, so we wait exactly as long as
    asked instead of guessing."""
    import re

    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        return float(match.group(1)) + 1.0  # small buffer
    return None


class MockLLMClient(LLMClient):
    """Deterministic offline stand-in. Good enough to exercise the full
    request/response cycle, strategy triggers, and branch-selection flow."""

    def generate(self, messages: list[dict]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            f"[mock-response] Building on '{last_user[:60]}', here's a refined continuation "
            f"that stays close to the current idea."
        )

    def generate_json(self, messages: list[dict]) -> dict:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "the topic")
        seed = last_user[:30]
        return {
            "abstract_reframing": {
                "title": "Zoom out to principles",
                "prompt": f"Instead of the concrete version of '{seed}', what underlying principle is it really an instance of?",
            },
            "contradictory_perspective": {
                "title": "Argue the opposite",
                "prompt": f"What would the strongest counter-argument to '{seed}' look like, and does it reveal a better idea?",
            },
            "adjacent_domain": {
                "title": "Borrow from elsewhere",
                "prompt": f"How would a completely different field solve the same underlying problem as '{seed}'?",
            },
            "unconventional_alternative": {
                "title": "Break a core assumption",
                "prompt": f"What if the main constraint behind '{seed}' didn't exist — what becomes possible?",
            },
            "speculative_future": {
                "title": "Jump ahead 10 years",
                "prompt": f"If '{seed}' had already succeeded wildly and evolved for a decade, what would it look like now?",
            },
        }


_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    if settings.llm_backend == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("FIXATION_GEMINI_API_KEY is not set but llm_backend='gemini'")
        _client_instance = GeminiLLMClient(settings.gemini_api_key, settings.gemini_model)
    else:
        _client_instance = MockLLMClient()
    return _client_instance
