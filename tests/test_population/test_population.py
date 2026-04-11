"""Tests for population management."""

import pytest

from inception.genome.schema import AgentDNA, FitnessScore
from inception.population.population import Population, Species
from inception.population.selection import ElitistSelector, TruncationSelector


class TestPopulation:
    def test_create_population(self, coder_agent, researcher_agent):
        pop = Population(agents=[coder_agent, researcher_agent])
        assert pop.size == 2

    def test_best_agent(self, coder_agent, researcher_agent):
        coder_agent.fitness_scores = [FitnessScore(overall=0.8)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.6)]

        pop = Population(agents=[coder_agent, researcher_agent])
        best = pop.best_agent()
        assert best.id == coder_agent.id

    def test_average_fitness(self, coder_agent, researcher_agent):
        coder_agent.fitness_scores = [FitnessScore(overall=0.8)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.6)]

        pop = Population(agents=[coder_agent, researcher_agent])
        assert pop.average_fitness() == pytest.approx(0.7)

    def test_empty_population(self):
        pop = Population(agents=[])
        assert pop.size == 0
        assert pop.best_agent() is None
        assert pop.average_fitness() == 0.0

    def test_diversity_index_no_species(self, coder_agent):
        pop = Population(agents=[coder_agent])
        assert pop.diversity_index() == 0.0

    def test_get_stats(self, coder_agent, researcher_agent):
        coder_agent.fitness_scores = [FitnessScore(overall=0.8)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.6)]

        pop = Population(agents=[coder_agent, researcher_agent], generation=5)
        stats = pop.get_stats()

        assert stats.generation == 5
        assert stats.population_size == 2
        assert stats.best_fitness == 0.8
        assert stats.average_fitness == pytest.approx(0.7)


class TestElitistSelector:
    def test_select_keeps_elite(self, coder_agent, researcher_agent):
        coder_agent.fitness_scores = [FitnessScore(overall=0.9)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.3)]

        selector = ElitistSelector(elite_count=1)
        survivors = selector.select([coder_agent, researcher_agent], target_size=1)

        assert len(survivors) == 1
        assert survivors[0].id == coder_agent.id

    def test_select_all_when_target_large(self, coder_agent, researcher_agent):
        selector = ElitistSelector(elite_count=1)
        survivors = selector.select([coder_agent, researcher_agent], target_size=5)
        assert len(survivors) == 2


class TestTruncationSelector:
    def test_truncation(self, coder_agent, researcher_agent):
        coder_agent.fitness_scores = [FitnessScore(overall=0.9)]
        researcher_agent.fitness_scores = [FitnessScore(overall=0.3)]

        selector = TruncationSelector()
        survivors = selector.select([coder_agent, researcher_agent], target_size=1)

        assert len(survivors) == 1
        assert survivors[0].id == coder_agent.id
