"""Async HTTP client for A2A protocol interactions.

Discovers remote agents, fetches their genetic cards,
and sends mate requests over the network.
"""

from __future__ import annotations

from typing import Any

import httpx

from inception.a2a.models import (
    GeneticAgentCard,
    GenomeExport,
    MateRequest,
    MateResponse,
)



class A2AClient:
    """Async client for communicating with remote A2A agents."""

    def __init__(self, timeout: float = 30.0, api_key: str = ""):
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def discover(self, url: str) -> GeneticAgentCard:
        """Fetch a remote agent's GeneticAgentCard.

        Args:
            url: Base URL of the remote agent (e.g. http://host:3000)

        Returns:
            The agent's GeneticAgentCard parsed from .well-known/agent.json
        """
        url = url.rstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{url}/.well-known/agent.json", headers=self._headers()
            )
            resp.raise_for_status()
            return GeneticAgentCard.model_validate(resp.json())

    async def get_genome(self, url: str) -> GenomeExport:
        """Fetch a remote agent's exported genome.

        Args:
            url: Base URL of the remote agent

        Returns:
            The GenomeExport (redacted per the agent's MatingPolicy)
        """
        url = url.rstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{url}/api/genome", headers=self._headers()
            )
            resp.raise_for_status()
            return GenomeExport.model_validate(resp.json())

    async def request_mate(self, url: str, request: MateRequest) -> MateResponse:
        """Send a mate request to a remote agent.

        Args:
            url: Base URL of the remote agent
            request: The MateRequest to send

        Returns:
            MateResponse indicating acceptance/rejection and optional offspring
        """
        url = url.rstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{url}/api/mate",
                json=request.model_dump(),
                headers=self._headers(),
            )
            resp.raise_for_status()
            return MateResponse.model_validate(resp.json())

    async def get_fitness(self, url: str) -> dict[str, Any]:
        """Fetch a remote agent's fitness scores.

        Args:
            url: Base URL of the remote agent

        Returns:
            Dict with 'scores', 'best', and 'latest' fitness data
        """
        url = url.rstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{url}/api/fitness", headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()

    async def check_health(self, url: str) -> bool:
        """Check if a remote agent is online and healthy.

        Args:
            url: Base URL of the remote agent

        Returns:
            True if the agent responds with a healthy status
        """
        url = url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/health")
                return resp.status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException):
            return False
