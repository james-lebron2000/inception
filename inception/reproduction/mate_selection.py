"""Mate selection strategies — choosing who mates with whom.

Biological analog: sexual selection. Beyond raw fitness, agents are
matched based on complementary capabilities, ensuring offspring
inherit a diverse skill set.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from inception.genome.compatibility import compatibility_score
from inception.genome.schema import AgentDNA


class MateSelector(ABC):
    """Abstract base for mate selection strategies."""

    @abstractmethod
    def select_pairs(
        self, agents: list[AgentDNA], num_pairs: int
    ) -> list[tuple[AgentDNA, AgentDNA]]:
        """Select parent pairs for mating."""
        ...


class TournamentMateSelector(MateSelector):
    """Tournament selection: random groups compete, winners mate.

    Biological analog: male-male competition for mating rights.
    """

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select_pairs(
        self, agents: list[AgentDNA], num_pairs: int
    ) -> list[tuple[AgentDNA, AgentDNA]]:
        pairs: list[tuple[AgentDNA, AgentDNA]] = []
        for _ in range(num_pairs):
            parent_a = self._tournament(agents)
            # Ensure parent_b is different from parent_a
            remaining = [a for a in agents if a.id != parent_a.id]
            if not remaining:
                remaining = agents
            parent_b = self._tournament(remaining)
            pairs.append((parent_a, parent_b))
        return pairs

    def _tournament(self, agents: list[AgentDNA]) -> AgentDNA:
        k = min(self.tournament_size, len(agents))
        contestants = random.sample(agents, k)
        return max(contestants, key=lambda a: a.latest_fitness)


class FitnessProportionalSelector(MateSelector):
    """Roulette wheel selection: probability proportional to fitness.

    Biological analog: organisms with higher fitness have more offspring.
    """

    def select_pairs(
        self, agents: list[AgentDNA], num_pairs: int
    ) -> list[tuple[AgentDNA, AgentDNA]]:
        if not agents:
            return []

        # Ensure all weights are positive
        fitnesses = [max(0.01, a.latest_fitness) for a in agents]
        pairs: list[tuple[AgentDNA, AgentDNA]] = []

        for _ in range(num_pairs):
            parent_a = random.choices(agents, weights=fitnesses, k=1)[0]
            parent_b = random.choices(agents, weights=fitnesses, k=1)[0]
            # Retry if same parent selected
            attempts = 0
            while parent_b.id == parent_a.id and attempts < 10:
                parent_b = random.choices(agents, weights=fitnesses, k=1)[0]
                attempts += 1
            pairs.append((parent_a, parent_b))

        return pairs


class ComplementarySelector(MateSelector):
    """Prefer mating partners with complementary (non-overlapping) capabilities.

    Biological analog: sexual selection for diverse traits.
    Agents with different skill sets are more likely to mate,
    producing offspring with broader capabilities.
    """

    def __init__(self, optimal_compatibility: float = 0.5, tournament_size: int = 3):
        self.optimal_compatibility = optimal_compatibility
        self.tournament_size = tournament_size

    def select_pairs(
        self, agents: list[AgentDNA], num_pairs: int
    ) -> list[tuple[AgentDNA, AgentDNA]]:
        pairs: list[tuple[AgentDNA, AgentDNA]] = []

        for _ in range(num_pairs):
            # First parent by tournament
            parent_a = self._tournament_select(agents)

            # Second parent: find best complement
            candidates = [a for a in agents if a.id != parent_a.id]
            if not candidates:
                candidates = agents

            # Score candidates by how close their compatibility is to optimal
            scored = []
            for candidate in candidates:
                compat = compatibility_score(parent_a.genome, candidate.genome)
                distance = abs(compat - self.optimal_compatibility)
                # Combine with fitness
                score = candidate.latest_fitness * (1.0 - distance)
                scored.append((candidate, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            parent_b = scored[0][0]
            pairs.append((parent_a, parent_b))

        return pairs

    def _tournament_select(self, agents: list[AgentDNA]) -> AgentDNA:
        k = min(self.tournament_size, len(agents))
        contestants = random.sample(agents, k)
        return max(contestants, key=lambda a: a.latest_fitness)


class HybridSelector(MateSelector):
    """Combines tournament selection with complementary preference.

    First filters by fitness (tournament), then by compatibility.
    """

    def __init__(self, tournament_size: int = 3):
        self.tournament = TournamentMateSelector(tournament_size)
        self.complementary = ComplementarySelector(tournament_size=tournament_size)

    def select_pairs(
        self, agents: list[AgentDNA], num_pairs: int
    ) -> list[tuple[AgentDNA, AgentDNA]]:
        # Half from tournament, half from complementary
        n_tournament = num_pairs // 2
        n_complementary = num_pairs - n_tournament

        pairs = self.tournament.select_pairs(agents, n_tournament)
        pairs.extend(self.complementary.select_pairs(agents, n_complementary))
        return pairs
