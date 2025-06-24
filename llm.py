from google import genai
from google.genai import types

import config
import re

# ---------------------------------------------------------------------------
# Gemini Client (singleton)
# ---------------------------------------------------------------------------
client = genai.Client(api_key=config.GOOGLE_API_KEY)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def enhance_prompt(prompt: str) -> str:
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL_FLASH,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=config.ENHANCER_PROMPT,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return resp.text


def generate_reaction(
    location: str, human: str, boat: str, fish: str, size: str
) -> str:
    prompt = f"""장소: {location}
    캐릭터: {human}
    배: {boat}
    물고기: {fish}
    크기: {size}"""

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL_FLASH,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=config.REACTION_PROMPT,
            max_output_tokens=30,
            temperature=1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    reaction = resp.text.strip()
    return reaction
