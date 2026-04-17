"""FastAPI-based Gene Registry server for agent matchmaking.

Provides endpoints for agent registration, task-based discovery,
and fitness leaderboards.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inception.a2a.models import GeneticAgentCard
from inception.registry.db import RegistryDB
from inception.registry.matcher import AgentMatcher


class DiscoverRequest(BaseModel):
    task: str
    requester_card: GeneticAgentCard
    top_k: int = 5


class RegistryServer:
    """Gene Registry server for agent discovery and matchmaking."""

    def __init__(self, db: RegistryDB):
        self.db = db
        self.matcher = AgentMatcher(db)
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Inception Gene Registry")

        @app.on_event("startup")
        async def startup() -> None:
            await self.db.initialize()

        @app.on_event("shutdown")
        async def shutdown() -> None:
            await self.db.close()

        @app.post("/api/register")
        async def register_agent(card: GeneticAgentCard) -> dict[str, str]:
            await self.db.register_agent(card)
            return {"status": "registered", "agent_id": card.agent_id}

        @app.put("/api/register/{agent_id}")
        async def update_agent(agent_id: str, card: GeneticAgentCard) -> dict[str, str]:
            if card.agent_id != agent_id:
                raise HTTPException(400, "Agent ID mismatch")
            await self.db.update_agent(card)
            return {"status": "updated", "agent_id": agent_id}

        @app.delete("/api/register/{agent_id}")
        async def deregister(agent_id: str) -> dict[str, str]:
            await self.db.deactivate_agent(agent_id)
            return {"status": "deregistered", "agent_id": agent_id}

        @app.get("/api/agents")
        async def list_agents(
            capability: str = "",
            min_fitness: float = 0.0,
            limit: int = 20,
        ) -> list[dict[str, Any]]:
            if capability:
                entries = await self.db.search_by_capability(
                    capability, min_fitness, limit
                )
            else:
                entries = await self.db.get_all_active(limit)
            return [e.model_dump() for e in entries]

        @app.get("/api/agents/{agent_id}")
        async def get_agent(agent_id: str) -> dict[str, Any]:
            entry = await self.db.get_agent(agent_id)
            if not entry:
                raise HTTPException(404, "Agent not found")
            return entry.model_dump()

        @app.post("/api/discover")
        async def discover(req: DiscoverRequest) -> list[dict[str, Any]]:
            matches = await self.matcher.find_matches(
                req.task, req.requester_card, req.top_k
            )
            return [
                {
                    "agent_card": m.entry.agent_card.model_dump(),
                    "score": m.score,
                    "breakdown": m.breakdown,
                }
                for m in matches
            ]

        @app.get("/api/leaderboard")
        async def leaderboard(limit: int = 20) -> list[dict[str, Any]]:
            entries = await self.db.get_leaderboard(limit)
            return [
                {
                    "name": e.agent_card.name,
                    "agent_id": e.agent_card.agent_id,
                    "best_fitness": e.agent_card.best_fitness,
                    "generation": e.agent_card.generation,
                    "gene_count": e.agent_card.gene_count,
                    "mating_count": e.mating_count,
                    "offspring_count": e.offspring_count,
                }
                for e in entries
            ]

        @app.post("/api/mating-log")
        async def log_mating(
            requester_id: str,
            responder_id: str,
            accepted: bool,
            offspring_id: str = "",
            task_description: str = "",
        ) -> dict[str, str]:
            await self.db.log_mating(
                requester_id, responder_id, accepted, offspring_id, task_description
            )
            return {"status": "logged"}

        @app.get("/api/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "service": "gene-registry"}

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
