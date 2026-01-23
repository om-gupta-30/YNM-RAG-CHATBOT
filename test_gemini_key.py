#!/usr/bin/env python3
"""
Test that GEMINI_API_KEY is set and the Gemini API is reachable.
Loads .env from project root. Run from project root: python test_gemini_key.py
"""
import os
import sys

# load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        print("FAIL: GEMINI_API_KEY is not set. Add it to .env or export it.")
        sys.exit(1)

    key = key.strip()
    if key.startswith("AIza") and len(key) > 20:
        print("OK: GEMINI_API_KEY is set (looks like a Google API key)")
    else:
        print("WARN: GEMINI_API_KEY format unexpected; still attempting API call.")

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        r = model.generate_content("Reply with exactly: OK")
        text = (r.text or "").strip()
        if "OK" in text or len(text) > 0:
            print("OK: Gemini API responded successfully.")
            return
        print("WARN: Gemini responded but not as expected:", repr(text)[:80])
    except Exception as e:
        print("FAIL: Gemini API error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
