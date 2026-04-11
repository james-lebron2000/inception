"""Tests for the A2A server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inception.a2a.models import GeneticAgentCard, MateRequest, MatingPolicy
from inception.a2a.server import A2AServer
from inception.genome.export import build_agent_card, export_genome


@pytest.fixture
def mating_policy():
    return MatingPolicy(
        allow_remote_mating=True,
        share_source_code=True,
        share_fitness_scores=True,
        share_capability_vector=True,
        max_matings_per_day=5,
    )


@pytest.fixture
def a2a_server(coder_agent, mating_policy):
    return A2AServer(
        agent=coder_agent,
        policy=mating_policy,
        host="127.0.0.1",
        port=3000,
    )


@pytest.fixture
def client(a2a_server):
    return TestClient(a2a_server.app)


class TestAgentCardEndpoint:
    def test_serves_agent_card(self, client, coder_agent):
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == coder_agent.name
        assert data["protocol_version"] == "genesis/1.0"
        assert "capabilities" in data

    def test_card_has_genetic_metadata(self, client):
        resp = client.get("/.well-known/agent.json")
        data = resp.json()
        assert "gene_count" in data
        assert "generation" in data
        assert "mating_policy" in data


class TestGenomeEndpoint:
    def test_get_genome(self, client):
        resp = client.get("/api/genome")
        assert resp.status_code == 200
        data = resp.json()
        assert "genome_id" in data
        assert "chromosome_summaries" in data
        assert data["gene_count"] > 0

    def test_genome_includes_source_when_policy_allows(self, client):
        resp = client.get("/api/genome")
        data = resp.json()
        assert data["full_chromosomes"] is not None


class TestMateEndpoint:
    def _make_request(self, researcher_agent):
        requester_policy = MatingPolicy(
            allow_remote_mating=True, share_source_code=True
        )
        card = build_agent_card(researcher_agent, "http://remote:3001", requester_policy)
        genome_export = export_genome(researcher_agent.genome, requester_policy)
        return MateRequest(
            requester_card=card,
            task_description="code analysis",
            requester_genome_export=genome_export,
        )

    def test_mate_accepted(self, client, researcher_agent):
        req = self._make_request(researcher_agent)
        resp = client.post("/api/mate", json=req.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["responder_genome_export"] is not None

    def test_mate_rejected_when_disabled(self, coder_agent):
        policy = MatingPolicy(allow_remote_mating=False)
        server = A2AServer(agent=coder_agent, policy=policy)
        test_client = TestClient(server.app)

        card = GeneticAgentCard(name="req", url="http://x", agent_id="x1")
        req = MateRequest(requester_card=card)
        resp = test_client.post("/api/mate", json=req.model_dump())
        data = resp.json()
        assert data["accepted"] is False
        assert "disabled" in data["rejection_reason"].lower()

    def test_mate_rejected_blocked_requester(self, coder_agent):
        policy = MatingPolicy(
            allow_remote_mating=True,
            blocked_requester_ids=["blocked-agent"],
        )
        server = A2AServer(agent=coder_agent, policy=policy)
        test_client = TestClient(server.app)

        card = GeneticAgentCard(name="blocked", url="http://x", agent_id="blocked-agent")
        req = MateRequest(requester_card=card)
        resp = test_client.post("/api/mate", json=req.model_dump())
        data = resp.json()
        assert data["accepted"] is False
        assert "blocked" in data["rejection_reason"].lower()

    def test_mate_rejected_low_fitness(self, coder_agent):
        policy = MatingPolicy(
            allow_remote_mating=True,
            min_requester_fitness=0.5,
        )
        server = A2AServer(agent=coder_agent, policy=policy)
        test_client = TestClient(server.app)

        card = GeneticAgentCard(
            name="weak", url="http://x", agent_id="weak1", best_fitness=0.1
        )
        req = MateRequest(requester_card=card)
        resp = test_client.post("/api/mate", json=req.model_dump())
        data = resp.json()
        assert data["accepted"] is False
        assert "fitness" in data["rejection_reason"].lower()


class TestHealthEndpoint:
    def test_health(self, client, coder_agent):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["agent"] == coder_agent.name


class TestFitnessEndpoint:
    def test_fitness(self, client):
        resp = client.get("/api/fitness")
        assert resp.status_code == 200
        data = resp.json()
        assert "scores" in data
