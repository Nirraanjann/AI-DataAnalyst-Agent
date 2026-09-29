"""
app/services/llm_client.py

Thin wrapper around the Gemini API (free tier). Single entry point
(`create_message`) returns a normalized response shape so router.py
and analyst_service.py don't need to know which provider is behind
the wrapper.

Requires GEMINI_API_KEY in .env. Get one free at
https://aistudio.google.com/apikey -- no billing needed for the
free-tier models.

Phase 4 addition: basic retry/backoff around the REST call itself,
for transient network errors and Gemini's retryable HTTP status
codes (429 rate limit, 5xx server errors). This does NOT retry on
non-retryable errors (4xx other than 429) -- those fail immediately,
same as before.
"""

import os
import random
import re
import time
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

load_dotenv()

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
_BACKOFF_BASE_SECONDS = float(os.environ.get("GEMINI_BACKOFF_BASE_SECONDS", "1.0"))
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Phase 4 addition: the LangGraph agent makes up to 4 Gemini calls per
# question (planning, routing, decide, phrasing) vs. Phase 3's 2, which
# hits the free tier's 15-requests/minute limit in practice during eval
# runs. Proactively space calls out rather than only reacting after a
# 429. 4.5s -> ~13.3 calls/min, safely under the 15/min cap.
_MIN_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "4.5"))
_last_call_at = 0.0


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
    _throttle()
    resp = _post_with_retry(url, payload)
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


def _post_with_retry(url: str, payload: dict, timeout: int = 60) -> requests.Response:
    """POST with retry/backoff on transient network errors and
    retryable HTTP status codes. Raises the last exception if every
    attempt fails; returns the final response otherwise (including a
    non-retryable error response, which the caller handles via
    resp.raise_for_status())."""
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                raise
            print(
                f"Gemini request failed ({exc.__class__.__name__}), retrying "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})..."
            )
            _sleep_backoff(attempt)
            continue

        if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
            if resp.status_code == 429:
                delay = _extract_retry_delay_seconds(resp)
            else:
                delay = None

            if delay is not None:
                # Trust Google's own suggested wait over our exponential
                # guess -- a quota reset is usually ~50-60s, far longer
                # than our default backoff would ever wait on its own.
                print(
                    f"Gemini call hit the rate limit (429); waiting "
                    f"{delay:.0f}s per Google's retryDelay (attempt "
                    f"{attempt + 1}/{_MAX_RETRIES})..."
                )
                time.sleep(delay + 1.0)
            else:
                print(
                    f"Gemini call returned status {resp.status_code}, retrying "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})..."
                )
                _sleep_backoff(attempt)
            continue

        return resp

    # Unreachable in practice: the loop either returns or raises above.
    assert last_exc is not None
    raise last_exc


def _sleep_backoff(attempt: int) -> None:
    delay = _BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.25)
    time.sleep(delay)


def _extract_retry_delay_seconds(resp: requests.Response) -> float | None:
    """Gemini's 429 body includes a RetryInfo.retryDelay (e.g. '50s').
    Parse it if present; return None to fall back to exponential backoff."""
    try:
        data = resp.json()
    except ValueError:
        return None

    for detail in data.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            match = re.match(r"([\d.]+)s?", detail.get("retryDelay", ""))
            if match:
                return float(match.group(1))
    return None


def _throttle() -> None:
    """Proactively space calls out so we stay under the free-tier
    requests/minute cap, rather than only reacting after a 429."""
    global _last_call_at
    now = time.monotonic()
    wait = _MIN_INTERVAL_SECONDS - (now - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


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