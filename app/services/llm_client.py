"""
app/services/llm_client.py

Thin wrapper around the Gemini API (free tier). Single entry point
(`create_message`) returns a normalized response shape so router.py
and analyst_service.py don't need to know which provider is behind
the wrapper.

Requires GEMINI_API_KEY in .env. Get one free at
https://aistudio.google.com/apikey -- no billing needed for the
free-tier models.
"""

import os
from dataclasses import dataclass, field

import requests

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class ContentBlock:
    type: str  # "text" or "tool_use"
    text: str | None = None
    name: str | None = None
    input: dict = field(default_factory=dict)
    id: str | None = None


@dataclass
class NormalizedResponse:
    content: list[ContentBlock]
    stop_reason: str | None = None


def create_message(
    messages: list[dict],
    system: str | None = None,
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
) -> NormalizedResponse:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env (see .env.example)."
        )

    payload = {
        "contents": [_to_gemini_content(m) for m in messages],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        payload["tools"] = [{"functionDeclarations": [_to_gemini_tool(t) for t in tools]}]

    url = f"{_BASE_URL}/{_MODEL}:generateContent?key={api_key}"
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code != 200:
        print("STATUS:", resp.status_code)
        print("BODY:", resp.text)
    resp.raise_for_status()
    data = resp.json()

    candidate = data["candidates"][0]
    parts = candidate.get("content", {}).get("parts", [])
    blocks: list[ContentBlock] = []

    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            blocks.append(
                ContentBlock(type="tool_use", name=fc.get("name"), input=fc.get("args", {}))
            )
        elif "text" in part:
            blocks.append(ContentBlock(type="text", text=part["text"]))

    return NormalizedResponse(
        content=blocks,
        stop_reason=candidate.get("finishReason"),
    )


def _to_gemini_content(message: dict) -> dict:
    """Our messages are {"role": "user", "content": "some string"}.
    Gemini uses role "model" instead of "assistant" and wraps text in
    parts -- only "user" is used today, but this handles both."""
    role = "model" if message["role"] == "assistant" else message["role"]
    return {"role": role, "parts": [{"text": message["content"]}]}


def _to_gemini_tool(anthropic_tool: dict) -> dict:
    """Anthropic tool schema -> Gemini functionDeclaration schema."""
    return {
        "name": anthropic_tool["name"],
        "description": anthropic_tool["description"],
        "parameters": anthropic_tool["input_schema"],
    }