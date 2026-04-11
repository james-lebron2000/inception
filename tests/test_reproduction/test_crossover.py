"""Tests for crossover strategies."""

import pytest

from inception.genome.schema import ChromosomeKind
from inception.llm.client import LLMClient
from inception.reproduction.crossover import ChromosomeLevelCrossover


class TestChromosomeLevelCrossover:
    @pytest.mark.asyncio
    async def test_crossover_produces_offspring(self, coder_genome, researcher_genome):
        crossover = ChromosomeLevelCrossover()
        llm = LLMClient()  # Not actually used for chromosome-level

        offspring = await crossover.crossover(coder_genome, researcher_genome, llm)

        assert offspring is not None
        assert offspring.id != coder_genome.id
        assert offspring.id != researcher_genome.id
        assert len(offspring.chromosomes) > 0

    @pytest.mark.asyncio
    async def test_offspring_has_lineage(self, coder_genome, researcher_genome):
        crossover = ChromosomeLevelCrossover()
        llm = LLMClient()

        offspring = await crossover.crossover(coder_genome, researcher_genome, llm)

        assert coder_genome.id in offspring.lineage.parent_ids
        assert researcher_genome.id in offspring.lineage.parent_ids
        assert offspring.lineage.generation == 1

    @pytest.mark.asyncio
    async def test_offspring_inherits_chromosomes(self, coder_genome, researcher_genome):
        crossover = ChromosomeLevelCrossover(unpaired_inheritance_prob=1.0)
        llm = LLMClient()

        offspring = await crossover.crossover(coder_genome, researcher_genome, llm)

        # With 100% inheritance prob, offspring should have chromosomes
        assert offspring.gene_count > 0

    @pytest.mark.asyncio
    async def test_crossover_same_parents(self, coder_genome):
        crossover = ChromosomeLevelCrossover()
        llm = LLMClient()

        offspring = await crossover.crossover(coder_genome, coder_genome, llm)

        # Should still produce valid offspring
        assert offspring is not None
        assert offspring.gene_count > 0
