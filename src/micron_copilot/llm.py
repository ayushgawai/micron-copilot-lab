"""Model-agnostic LLM client.

Adapters for hosted providers use only the standard library (urllib)
so the package installs with zero dependencies. Without keys, callers
get the deterministic offline MockLLM — the default for tests/bench.
Provider HTTP shapes follow each provider's public chat/messages API;
they are intentionally thin and are not exercised in CI.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMClient:
    provider = "base"
    model = "unknown"

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class MockLLM(LLMClient):
    """Deterministic extractive mock: echoes the most relevant evidence.

    It reads ``EVIDENCE:`` blocks the agents embed in the prompt and
    returns their first sentences, preserving citations. This makes the
    full pipeline testable offline with faithful, cited output.
    """
    provider = "mock"
    model = "mock-extractive-v1"

    def complete(self, system: str, user: str) -> LLMResponse:
        blocks = re.findall(r"\[DOC-\d+\][^\n]*", user)
        sentences: list[str] = []
        for block in blocks[:4]:
            # Keep "DOC-xxx: <first sentence> [DOC-xxx]"
            body = block.split(":", 1)[-1].strip()
            first = body.split(". ")[0].rstrip(".") + "."
            doc_ref = re.search(r"DOC-\d+", block).group(0)  # type: ignore[union-attr]
            sentences.append(f"{first[:-1]} [{doc_ref}].")
        text = " ".join(sentences) if sentences else (
            "No supporting evidence was retrieved for this request.")
        return LLMResponse(text=text, model=self.model, provider=self.provider)


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


class OpenAIAdapter(LLMClient):
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def complete(self, system: str, user: str) -> LLMResponse:
        data = _post_json("https://api.openai.com/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0}, {"Authorization": f"Bearer {self._key}"})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           self.model, self.provider)


class AnthropicAdapter(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, system: str, user: str) -> LLMResponse:
        data = _post_json("https://api.anthropic.com/v1/messages", {
            "model": self.model, "max_tokens": 1024, "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0}, {"x-api-key": self._key,
                                "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return LLMResponse(text, self.model, self.provider)


class AzureOpenAIAdapter(LLMClient):
    provider = "azure-openai"

    def __init__(self, deployment: str | None = None, endpoint: str | None = None,
                 api_key: str | None = None, api_version: str = "2024-02-15-preview"):
        self.model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self._endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
                          ).rstrip("/")
        self._key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._version = api_version

    def complete(self, system: str, user: str) -> LLMResponse:
        url = (f"{self._endpoint}/openai/deployments/{self.model}"
               f"/chat/completions?api-version={self._version}")
        data = _post_json(url, {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0}, {"api-key": self._key})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           self.model, self.provider)


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    name = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if name in ("mock", "offline", ""):
        return MockLLM()
    if name == "openai":
        return OpenAIAdapter(model or "gpt-4o-mini")
    if name == "anthropic":
        return AnthropicAdapter(model or "claude-sonnet-4-5")
    if name in ("azure", "azure-openai", "azure_openai"):
        return AzureOpenAIAdapter(deployment=model)
    raise ValueError(f"Unknown LLM provider: {name!r} "
                     "(expected mock|openai|anthropic|azure)")
