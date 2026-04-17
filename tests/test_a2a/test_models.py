"""Tests for A2A protocol data models."""

from __future__ import annotations

from inception.a2a.models import (
    ChromosomeSummary,
    GeneticAgentCard,
    GenomeExport,
    MateRequest,
    MateResponse,
    MatingPolicy,
    RegistryEntry,
)
from inception.genome.schema import ChromosomeKind


class TestMatingPolicy:
    def test_defaults_are_private(self):
        policy = MatingPolicy()
        assert policy.allow_remote_mating is False
        assert policy.share_source_code is False
        assert policy.redact_gene_sources is True
        assert policy.require_mutual_consent is True

    def test_custom_policy(self):
        policy = MatingPolicy(
            allow_remote_mating=True,
            share_source_code=True,
            min_requester_fitness=0.5,
            max_matings_per_day=5,
        )
        assert policy.allow_remote_mating is True
        assert policy.share_source_code is True
        assert policy.min_requester_fitness == 0.5
        assert policy.max_matings_per_day == 5

    def test_blocklist(self):
        policy = MatingPolicy(
            blocked_requester_ids=["bad_agent_1", "bad_agent_2"],
        )
        assert "bad_agent_1" in policy.blocked_requester_ids
        assert len(policy.blocked_requester_ids) == 2


class TestGeneticAgentCard:
    def test_create_card(self):
        card = GeneticAgentCard(
            name="TestAgent",
            url="http://localhost:3000",
            agent_id="test-123",
            generation=5,
            gene_count=10,
            capabilities={"coding": 0.8, "math": 0.6},
            skill_names=["coding", "math"],
            best_fitness=0.85,
        )
        assert card.name == "TestAgent"
        assert card.protocol_version == "genesis/1.0"
        assert card.capabilities["coding"] == 0.8

    def test_json_round_trip(self):
        card = GeneticAgentCard(
            name="RoundTrip",
            url="http://test:3000",
            agent_id="rt-456",
        )
        json_str = card.model_dump_json()
        restored = GeneticAgentCard.model_validate_json(json_str)
        assert restored.name == "RoundTrip"
        assert restored.agent_id == "rt-456"


class TestGenomeExport:
    def test_redacted_export(self):
        export = GenomeExport(
            genome_id="g-123",
            version=1,
            gene_count=5,
            chromosome_summaries=[
                ChromosomeSummary(
                    id="c1",
                    kind=ChromosomeKind.SKILL_MODULE,
                    name="skill_a",
                    gene_count=3,
                    source_hash="abc123",
                ),
            ],
        )
        assert export.full_chromosomes is None
        assert len(export.chromosome_summaries) == 1
        assert export.chromosome_summaries[0].source_hash == "abc123"

    def test_full_export(self):
        export = GenomeExport(
            genome_id="g-456",
            version=1,
            gene_count=2,
            full_chromosomes=[{"id": "c1", "kind": "skill_module", "genes": []}],
        )
        assert export.full_chromosomes is not None
        assert len(export.full_chromosomes) == 1


class TestMateRequestResponse:
    def test_mate_request_has_id(self):
        card = GeneticAgentCard(name="req", url="http://a", agent_id="a1")
        req = MateRequest(requester_card=card, task_description="code stuff")
        assert req.request_id
        assert req.task_description == "code stuff"
        assert req.timestamp

    def test_mate_response_accepted(self):
        resp = MateResponse(request_id="r-1", accepted=True)
        assert resp.accepted is True
        assert resp.rejection_reason == ""

    def test_mate_response_rejected(self):
        resp = MateResponse(
            request_id="r-2",
            accepted=False,
            rejection_reason="Fitness too low",
        )
        assert resp.accepted is False
        assert "Fitness" in resp.rejection_reason


class TestRegistryEntry:
    def test_create_entry(self):
        card = GeneticAgentCard(name="entry", url="http://x", agent_id="x1")
        entry = RegistryEntry(agent_card=card)
        assert entry.mating_count == 0
        assert entry.offspring_count == 0
        assert entry.registered_at
