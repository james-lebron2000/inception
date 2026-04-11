"""Population container and diversity tracking.

Manages a collection of agents within a single generation,
tracks diversity metrics, and provides population-level queries.

Biological analog: a breeding population of organisms in an ecosystem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from inception.genome.schema import AgentDNA, AgentStatus


@dataclass
class Species:
    """A group of genetically similar agents.

    Biological analog: a species — a reproductively compatible group.
    """

    id: str
    name: str
    representative_id: str  # Agent ID used as the species archetype
    member_ids: list[str] = field(default_factory=list)
    avg_fitness: float = 0.0

    @property
    def size(self) -> int:
        return len(self.member_ids)


@dataclass
class GenerationStats:
    """Statistics for one generation."""

    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    population_size: int
    species_count: int
    diversity_index: float
    gene_count_avg: float


class Population:
    """Container for a generation of agents."""

    def __init__(self, agents: list[AgentDNA], generation: int = 0):
        self.agents = agents
        self.generation = generation
        self.species: list[Species] = []

    @property
    def size(self) -> int:
        return len(self.agents)

    @property
    def alive_agents(self) -> list[AgentDNA]:
        return [a for a in self.agents if a.status == AgentStatus.ALIVE]

    def best_agent(self) -> AgentDNA | None:
        if not self.agents:
            return None
        return max(self.agents, key=lambda a: a.latest_fitness)

    def average_fitness(self) -> float:
        if not self.agents:
            return 0.0
        return sum(a.latest_fitness for a in self.agents) / len(self.agents)

    def worst_fitness(self) -> float:
        if not self.agents:
            return 0.0
        return min(a.latest_fitness for a in self.agents)

    def diversity_index(self) -> float:
        """Shannon diversity index across species.

        Higher values = more diverse population.
        A population with all agents in one species has index 0.
        """
        if not self.species or self.size == 0:
            return 0.0

        proportions = [s.size / self.size for s in self.species if s.size > 0]
        if not proportions:
            return 0.0

        return -sum(p * math.log(p) for p in proportions if p > 0)

    def get_stats(self) -> GenerationStats:
        """Compute statistics for this generation."""
        gene_counts = [a.genome.gene_count for a in self.agents]
        return GenerationStats(
            generation=self.generation,
            best_fitness=self.best_agent().latest_fitness if self.best_agent() else 0.0,
            average_fitness=self.average_fitness(),
            worst_fitness=self.worst_fitness(),
            population_size=self.size,
            species_count=len(self.species),
            diversity_index=self.diversity_index(),
            gene_count_avg=sum(gene_counts) / max(len(gene_counts), 1),
        )

    def get_agent(self, agent_id: str) -> AgentDNA | None:
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    def add_agent(self, agent: AgentDNA) -> None:
        self.agents.append(agent)

    def remove_agent(self, agent_id: str) -> None:
        self.agents = [a for a in self.agents if a.id != agent_id]
