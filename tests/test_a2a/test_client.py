"""Tests for the A2A client."""

from __future__ import annotations

import httpx
import pytest

from inception.a2a.models import GeneticAgentCard, MateRequest, MateResponse


@pytest.fixture
def mock_card_data():
    return GeneticAgentCard(
        name="MockAgent",
        url="http://mock:3000",
        agent_id="mock-123",
        generation=2,
        gene_count=8,
        capabilities={"coding": 0.9},
        best_fitness=0.75,
    ).model_dump()


@pytest.fixture
def mock_transport(mock_card_data):
    """Create a mock httpx transport that returns test data."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/.well-known/agent.json":
            return httpx.Response(200, json=mock_card_data)

        if path == "/api/genome":
            return httpx.Response(200, json={
                "genome_id": "g-mock",
                "version": 1,
                "chromosome_summaries": [],
                "capability_vector": {"coding": 0.9},
                "gene_count": 8,
                "full_chromosomes": None,
            })

        if path == "/api/mate":
            return httpx.Response(200, json={
                "request_id": "r-mock",
                "accepted": True,
                "responder_card": mock_card_data,
                "responder_genome_export": None,
            })

        if path == "/api/fitness":
            return httpx.Response(200, json={
                "scores": [],
                "best": 0.75,
                "latest": 0.70,
            })

        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})

        return httpx.Response(404)

    return httpx.MockTransport(handler)


class TestA2AClientDiscover:
    @pytest.mark.asyncio
    async def test_discover_agent(self, mock_transport):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock:3000") as http:
            resp = await http.get("/.well-known/agent.json")
            card = GeneticAgentCard.model_validate(resp.json())

        assert card.name == "MockAgent"
        assert card.agent_id == "mock-123"
        assert card.capabilities["coding"] == 0.9


class TestA2AClientMate:
    @pytest.mark.asyncio
    async def test_request_mate(self, mock_transport, mock_card_data):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock:3000") as http:
            req_card = GeneticAgentCard(name="local", url="http://me", agent_id="me1")
            mate_req = MateRequest(requester_card=req_card, task_description="test")

            resp = await http.post("/api/mate", json=mate_req.model_dump())
            mate_resp = MateResponse.model_validate(resp.json())

        assert mate_resp.accepted is True
        assert mate_resp.responder_card is not None


class TestA2AClientHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, mock_transport):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock:3000") as http:
            resp = await http.get("/api/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
