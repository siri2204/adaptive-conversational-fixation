"""
Quick standalone check: which Gemini models are reachable with the CURRENT
key in your .env? Doesn't touch the experiment pipeline or any results file.
Each call is a 1-token throwaway prompt, so this barely dents your daily quota.

Run from the project root (same folder as .env), with your venv active:
    python test_models.py
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("FIXATION_GEMINI_API_KEY")
if not api_key:
    raise SystemExit("FIXATION_GEMINI_API_KEY not found in .env")

from google import genai

client = genai.Client(api_key=api_key)

# Candidates worth checking: current Gemini 3.x lineup (should be accessible
# to new keys) plus gemini-2.5-pro, to see if the block is flash-tier-only
# or the whole 2.5 generation.
MODELS_TO_TEST = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
]

results = {}
for model in MODELS_TO_TEST:
    print(f"Testing {model} ...")
    try:
        response = client.models.generate_content(
            model=model,
            contents="Say 'ok' and nothing else.",
        )
        results[model] = f"SUCCESS — {response.text.strip()!r}"
    except Exception as e:
        results[model] = f"FAILED — {e}"
    time.sleep(2)  # small gap between calls, be polite to rate limits

print("\n=== Summary ===")
for model, outcome in results.items():
    print(f"{model}: {outcome}")
