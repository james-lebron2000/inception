"""Tests for genome export and redaction."""

from __future__ import annotations

from inception.a2a.models import MatingPolicy
from inception.genome.export import (
    build_agent_card,
    export_genome,
    reconstruct_mating_genome,
)


class TestExportGenome:
    def test_restrictive_policy_no_source(self, coder_genome):
        policy = MatingPolicy(share_source_code=False, redact_gene_sources=True)
        export = export_genome(coder_genome, policy)

        assert export.genome_id == coder_genome.id
        assert export.full_chromosomes is None
        assert len(export.chromosome_summaries) == len(coder_genome.chromosomes)
        assert export.gene_count == coder_genome.gene_count

    def test_permissive_policy_includes_source(self, coder_genome):
        policy = MatingPolicy(share_source_code=True)
        export = export_genome(coder_genome, policy)

        assert export.full_chromosomes is not None
        assert len(export.full_chromosomes) == len(coder_genome.chromosomes)

    def test_capability_vector_shared(self, coder_genome):
        policy = MatingPolicy(share_capability_vector=True)
        export = export_genome(coder_genome, policy)
        assert len(export.capability_vector) > 0

    def test_capability_vector_hidden(self, coder_genome):
        policy = MatingPolicy(share_capability_vector=False)
        export = export_genome(coder_genome, policy)
        assert export.capability_vector == {}

    def test_chromosome_summary_has_hash(self, coder_genome):
        policy = MatingPolicy()
        export = export_genome(coder_genome, policy)
        for summary in export.chromosome_summaries:
            assert summary.source_hash
            assert len(summary.source_hash) == 16


class TestBuildAgentCard:
    def test_card_from_agent(self, coder_agent):
        policy = MatingPolicy(
            allow_remote_mating=True,
            share_fitness_scores=True,
            share_capability_vector=True,
        )
        card = build_agent_card(coder_agent, "http://localhost:3000", policy)

        assert card.name == "coder"
        assert card.url == "http://localhost:3000"
        assert card.agent_id == coder_agent.id
        assert card.gene_count == coder_agent.genome.gene_count
        assert card.mating_policy.allow_remote_mating is True

    def test_card_hides_fitness_when_policy_denies(self, coder_agent):
        policy = MatingPolicy(share_fitness_scores=False)
        card = build_agent_card(coder_agent, "http://test", policy)
        assert card.best_fitness == 0.0
        assert card.latest_fitness == 0.0


class TestReconstructGenome:
    def test_reconstruct_from_full_export(self, coder_genome):
        policy = MatingPolicy(share_source_code=True)
        export = export_genome(coder_genome, policy)
        reconstructed = reconstruct_mating_genome(export)

        assert reconstructed is not None
        assert len(reconstructed.chromosomes) == len(coder_genome.chromosomes)
        assert reconstructed.gene_count == coder_genome.gene_count

    def test_reconstruct_fails_without_source(self, coder_genome):
        policy = MatingPolicy(share_source_code=False)
        export = export_genome(coder_genome, policy)
        reconstructed = reconstruct_mating_genome(export)

        assert reconstructed is None
