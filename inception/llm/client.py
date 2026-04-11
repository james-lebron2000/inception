"""LLM client wrapper for Claude API.

Provides async methods for text completion and structured output,
used by the crossover and mutation engines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Async wrapper around the Anthropic Claude API."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a completion request to Claude.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with the generated text.
        """
        client = self._get_client()

        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.config.temperature
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def complete_code(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> str:
        """Complete a request and extract code from the response.

        Strips markdown code fences if present.
        """
        response = await self.complete(prompt, system=system, temperature=temperature)
        return extract_code_block(response.text)


def extract_code_block(text: str) -> str:
    """Extract code from markdown code fences, or return as-is if no fences."""
    lines = text.strip().split("\n")

    # Find code fence boundaries
    fence_starts = []
    fence_ends = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not fence_starts or len(fence_starts) == len(fence_ends):
                fence_starts.append(i)
            else:
                fence_ends.append(i)

    if fence_starts and fence_ends:
        # Return content between first pair of fences
        return "\n".join(lines[fence_starts[0] + 1 : fence_ends[0]])

    return text.strip()
