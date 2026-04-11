"""Tests for evolution loop (using chromosome-level crossover, no LLM needed)."""

import pytest

from inception.evolution.config import EvolutionConfig
from inception.evolution.history import EvolutionHistory
from inception.evolution.loop import EvolutionLoop
from inception.genome.schema import FitnessScore


class TestEvolutionConfig:
    def test_default_config(self):
        config = EvolutionConfig()
        assert config.population_size == 20
        assert config.max_generations == 50
        assert config.crossover_rate == 0.7
        assert config.mutation.mutation_rate == 0.1


class TestEvolutionHistory:
    def test_empty_history(self):
        history = EvolutionHistory()
        assert len(history.generations) == 0
        summary = history.summary()
        assert "No generations" in summary

    def test_record_generation(self, coder_agent, researcher_agent):
        from inception.population.population import Population

        coder_agent.fitness_scores = [FitnessScore(overall=0.7)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.5)]

        pop = Population(agents=[coder_agent, researcher_agent], generation=0)
        history = EvolutionHistory()
        history.record_generation(pop)

        assert len(history.generations) == 1
        assert len(history.all_agents) == 2

    def test_fitness_over_time(self, coder_agent, researcher_agent):
        from inception.population.population import Population

        history = EvolutionHistory()

        coder_agent.fitness_scores = [FitnessScore(overall=0.5)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.3)]
        pop = Population(agents=[coder_agent, researcher_agent], generation=0)
        history.record_generation(pop)

        fitness = history.fitness_over_time()
        assert len(fitness) == 1
        assert fitness[0]["best"] == 0.5

    def test_export_dot(self, coder_agent):
        coder_agent.fitness_scores = [FitnessScore(overall=0.7)]
        from inception.population.population import Population

        history = EvolutionHistory()
        pop = Population(agents=[coder_agent], generation=0)
        history.record_generation(pop)

        dot = history.export_genealogy_dot()
        assert "digraph" in dot
        assert "Genealogy" in dot


class TestEvolutionLoop:
    @pytest.mark.asyncio
    async def test_evolution_runs(self, coder_agent, researcher_agent):
        """Test that evolution loop runs for a few generations."""
        config = EvolutionConfig(
            population_size=4,
            offspring_count=2,
            elite_count=1,
            max_generations=3,
            crossover_strategy="chromosome",
            crossover_rate=0.7,
            mutation=type(config.mutation)(mutation_rate=0.0) if hasattr(config := EvolutionConfig(), 'mutation') else EvolutionConfig().mutation,
            verbose=False,
        )
        # Disable mutations for deterministic testing
        config.mutation.mutation_rate = 0.0

        loop = EvolutionLoop(config)
        result = await loop.run(
            seed_population=[coder_agent, researcher_agent],
            num_generations=2,
        )

        assert result is not None
        assert result.generations_run >= 0
        assert result.final_population.size > 0
        assert len(result.history.generations) > 0
