"""FastAPI-based A2A server for agent discovery and mating.

Serves a GeneticAgentCard at /.well-known/agent.json and handles
incoming mate requests at /api/mate, performing crossover server-side
when the owner's policy permits.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from inception.a2a.auth import verify_api_key
from inception.a2a.models import (
    GenomeExport,
    MateRequest,
    MateResponse,
    MatingPolicy,
)
from inception.genome.export import (
    build_agent_card,
    export_genome,
    reconstruct_mating_genome,
)
from inception.genome.schema import AgentDNA


class A2AServer:
    """A2A protocol server for a single agent.

    Serves the agent's genetic card for discovery and handles
    mate requests from remote agents.
    """

    def __init__(
        self,
        agent: AgentDNA,
        policy: MatingPolicy,
        host: str = "0.0.0.0",
        port: int = 3000,
        api_key: str = "",
        llm_api_key: str = "",
    ):
        self.agent = agent
        self.policy = policy
        self.host = host
        self.port = port
        self.api_key = api_key
        self.llm_api_key = llm_api_key
        self._mate_count_today = 0
        self._mate_lock = asyncio.Lock()
        self._last_reset = datetime.now(timezone.utc).date()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title=f"Inception A2A: {self.agent.name}")

        @app.get("/.well-known/agent.json")
        async def agent_card() -> dict[str, Any]:
            base_url = f"http://{self.host}:{self.port}"
            card = build_agent_card(self.agent, base_url, self.policy)
            return card.model_dump()

        @app.get("/api/genome")
        async def get_genome(request: Request) -> dict[str, Any]:
            self._check_auth(request)
            genome_export = export_genome(self.agent.genome, self.policy)
            return genome_export.model_dump()

        @app.post("/api/mate")
        async def handle_mate(mate_req: MateRequest, request: Request) -> dict[str, Any]:
            self._check_auth(request)
            response = await self._process_mate_request(mate_req)
            return response.model_dump()

        @app.get("/api/fitness")
        async def get_fitness() -> dict[str, Any]:
            if not self.policy.share_fitness_scores:
                return {"scores": []}
            return {
                "scores": [s.model_dump() for s in self.agent.fitness_scores],
                "best": self.agent.best_fitness,
                "latest": self.agent.latest_fitness,
            }

        @app.get("/api/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "agent": self.agent.name}

        return app

    def _check_auth(self, request: Request) -> None:
        if not self.api_key:
            return
        provided = request.headers.get("x-api-key", "")
        if not provided:
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                provided = auth[7:]
        if not verify_api_key(provided, self.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _reset_daily_counter(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset:
            self._mate_count_today = 0
            self._last_reset = today

    async def _process_mate_request(self, req: MateRequest) -> MateResponse:
        await self._reset_daily_counter()

        if not self.policy.allow_remote_mating:
            return MateResponse(
                request_id=req.request_id,
                accepted=False,
                rejection_reason="Remote mating is disabled by owner policy",
            )

        # Check blocklist
        requester_id = req.requester_card.agent_id
        if requester_id in self.policy.blocked_requester_ids:
            return MateResponse(
                request_id=req.request_id,
                accepted=False,
                rejection_reason="Requester is blocked",
            )

        # Check allowlist
        if (
            self.policy.allowed_requester_ids
            and requester_id not in self.policy.allowed_requester_ids
        ):
            return MateResponse(
                request_id=req.request_id,
                accepted=False,
                rejection_reason="Requester not in allowlist",
            )

        # Check fitness threshold
        if req.requester_card.best_fitness < self.policy.min_requester_fitness:
            return MateResponse(
                request_id=req.request_id,
                accepted=False,
                rejection_reason=f"Requester fitness {req.requester_card.best_fitness:.2f} below minimum {self.policy.min_requester_fitness:.2f}",
            )

        # Check rate limit
        async with self._mate_lock:
            if self._mate_count_today >= self.policy.max_matings_per_day:
                return MateResponse(
                    request_id=req.request_id,
                    accepted=False,
                    rejection_reason="Daily mating limit reached",
                )
            self._mate_count_today += 1

        # Build response card
        base_url = f"http://{self.host}:{self.port}"
        responder_card = build_agent_card(self.agent, base_url, self.policy)
        responder_export = export_genome(self.agent.genome, self.policy)

        # Try server-side crossover if requester shared their genome
        offspring_agent = None
        if req.requester_genome_export:
            offspring_agent = await self._attempt_crossover(req.requester_genome_export)

        return MateResponse(
            request_id=req.request_id,
            accepted=True,
            responder_card=responder_card,
            responder_genome_export=responder_export,
            offspring_agent=offspring_agent,
        )

    async def _attempt_crossover(self, remote_export: GenomeExport) -> AgentDNA | None:
        """Attempt server-side crossover with the remote genome."""
        remote_genome = reconstruct_mating_genome(remote_export)
        if remote_genome is None:
            return None

        from inception.genome.compatibility import compatibility_score
        from inception.reproduction.crossover import ChromosomeLevelCrossover

        compat = compatibility_score(self.agent.genome, remote_genome)
        if compat < 0.1:
            return None

        local_genome = copy.deepcopy(self.agent.genome)
        crossover = ChromosomeLevelCrossover()

        from inception.llm.client import LLMClient, LLMConfig

        llm = LLMClient(LLMConfig(api_key=self.llm_api_key)) if self.llm_api_key else LLMClient()

        child_genome = await crossover.crossover(local_genome, remote_genome, llm)

        child = AgentDNA(
            name=f"{self.agent.name[:4]}xRemote",
            genome=child_genome,
            generation=max(self.agent.generation, 0) + 1,
            parents=(self.agent.id, remote_export.genome_id),
        )
        return child

    def run(self) -> None:
        """Start the server with uvicorn."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)
