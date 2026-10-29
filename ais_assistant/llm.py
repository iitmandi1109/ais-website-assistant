"""Thin client for a local Ollama server.

I call Ollama's HTTP API with `requests` instead of a framework wrapper so that
timeouts, retries and JSON parsing are explicit and in one place. Every agent
gets an `LLM` object passed in, which lets the tests swap in a fake one.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from . import config

log = logging.getLogger("llm")


class LLMError(RuntimeError):
    """The model could not be reached, timed out, or returned something unusable."""


class LLM:
    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST,
                 timeout: float = config.LLM_TIMEOUT):
        self.model = model
        self.url = host.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def chat(self, system: str, user: str, json_mode: bool = False, stop: list[str] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            # Ollama unloads an idle model after 5 min; reloading costs seconds on the next visitor's question
            "keep_alive": config.LLM_KEEP_ALIVE,
            # temperature 0: the same question should get the same answer (repeatable evaluation)
            "options": {"temperature": 0, "num_ctx": config.LLM_NUM_CTX},
        }
        if json_mode:
            payload["format"] = "json"  # Ollama constrains decoding to valid JSON
        if stop:
            payload["options"]["stop"] = stop  # generation ends as soon as one of these is written

        last_error: Exception | None = None
        for attempt in range(config.LLM_RETRIES + 1):
            try:
                start = time.perf_counter()
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                log.debug("LLM call took %.1fs", time.perf_counter() - start)
                return content
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            except (requests.HTTPError, KeyError, ValueError) as exc:
                # a 404 here usually means the model was never pulled: retrying will not help
                raise LLMError(f"Ollama returned an unusable response: {exc}") from exc
        raise LLMError(f"Ollama did not answer after {config.LLM_RETRIES + 1} attempts: {last_error}")

    def chat_json(self, system: str, user: str) -> dict:
        return parse_json(self.chat(system, user, json_mode=True))


def parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating code fences or chatter around it (small models do this)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMError(f"Model did not return JSON: {text[:200]!r}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned broken JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"Expected a JSON object, got: {text[:200]!r}")
    return data
