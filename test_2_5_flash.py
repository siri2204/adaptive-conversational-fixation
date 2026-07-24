"""
Quick standalone check: is gemini-2.5-flash reachable with the CURRENT key
in your .env? Doesn't touch the experiment pipeline or any results file.

Run from the project root (same folder as .env), with your venv active:
    python test_2_5_flash.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("FIXATION_GEMINI_API_KEY")
if not api_key:
    raise SystemExit("FIXATION_GEMINI_API_KEY not found in .env")

from google import genai

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'ok' and nothing else.",
    )
    print("SUCCESS — gemini-2.5-flash is reachable with this key.")
    print("Response:", response.text)
except Exception as e:
    print("FAILED — gemini-2.5-flash is NOT reachable with this key.")
    print("Error:", e)
