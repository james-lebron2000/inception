"""Survival selection strategies — who lives to the next generation.

After offspring are produced, the combined pool of parents + offspring
is culled back to the target population size. Selection pressure
drives evolution by preferring fitter individuals.

Biological analog: natural selection / survival of the fittest.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from inception.genome.schema import AgentDNA


class SurvivalSelector(ABC):
    """Abstract base for survival selection."""

    @abstractmethod
    def select(self, agents: list[AgentDNA], target_size: int) -> list[AgentDNA]:
        """Select agents to survive to the next generation."""
        ...


class ElitistSelector(SurvivalSelector):
    """Top N agents always survive. The rest are selected by tournament.

    Ensures the best solutions are never lost.
    """

    def __init__(self, elite_count: int = 2, tournament_size: int = 3):
        self.elite_count = elite_count
        self.tournament_size = tournament_size

    def select(self, agents: list[AgentDNA], target_size: int) -> list[AgentDNA]:
        if len(agents) <= target_size:
            return agents

        # Sort by fitness
        sorted_agents = sorted(agents, key=lambda a: a.latest_fitness, reverse=True)

        # Elite always survive
        elite = sorted_agents[: self.elite_count]
        remaining_pool = sorted_agents[self.elite_count :]
        remaining_needed = target_size - len(elite)

        # Fill rest by tournament
        selected: list[AgentDNA] = list(elite)
        for _ in range(remaining_needed):
            if remaining_pool:
                k = min(self.tournament_size, len(remaining_pool))
                contestants = random.sample(remaining_pool, k)
                winner = max(contestants, key=lambda a: a.latest_fitness)
                selected.append(winner)
                remaining_pool.remove(winner)

        return selected


class TruncationSelector(SurvivalSelector):
    """Simply keep the top N agents by fitness.

    The simplest and most aggressive selection strategy.
    """

    def select(self, agents: list[AgentDNA], target_size: int) -> list[AgentDNA]:
        sorted_agents = sorted(agents, key=lambda a: a.latest_fitness, reverse=True)
        return sorted_agents[:target_size]


class DiversityPreservingSelector(SurvivalSelector):
    """Keep the best agents while maintaining species diversity.

    Each species gets at least one survivor slot, then remaining
    slots go to the best agents overall.
    """

    def __init__(self, min_per_species: int = 1):
        self.min_per_species = min_per_species

    def select(self, agents: list[AgentDNA], target_size: int) -> list[AgentDNA]:
        if len(agents) <= target_size:
            return agents

        # Group by species (using generation as a proxy if species not assigned)
        from inception.population.speciation import SpeciationEngine

        engine = SpeciationEngine()
        species_groups = engine.group_by_similarity(agents)

        selected: list[AgentDNA] = []
        selected_ids: set[str] = set()

        # Each species gets its best representative
        for group in species_groups:
            best = max(group, key=lambda a: a.latest_fitness)
            if best.id not in selected_ids:
                selected.append(best)
                selected_ids.add(best.id)

        # Fill remaining slots with best overall
        remaining_needed = target_size - len(selected)
        all_sorted = sorted(agents, key=lambda a: a.latest_fitness, reverse=True)
        for agent in all_sorted:
            if remaining_needed <= 0:
                break
            if agent.id not in selected_ids:
                selected.append(agent)
                selected_ids.add(agent.id)
                remaining_needed -= 1

        return selected[:target_size]
