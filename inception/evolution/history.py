"""Evolution history and lineage tracking.

Records the full genealogy tree, per-generation statistics,
and enables analysis of evolutionary trajectories.

Biological analog: fossil record + phylogenetic tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from inception.genome.schema import AgentDNA
from inception.population.population import GenerationStats, Population


class EvolutionHistory:
    """Tracks the complete evolutionary history."""

    def __init__(self):
        self.generations: list[GenerationStats] = []
        self.all_agents: dict[str, AgentDNA] = {}  # All agents ever created
        self.genealogy: nx.DiGraph = nx.DiGraph()  # Parent -> child edges

    def record_generation(self, population: Population) -> None:
        """Record a generation's statistics and agents."""
        stats = population.get_stats()
        self.generations.append(stats)

        for agent in population.agents:
            self.all_agents[agent.id] = agent
            self.genealogy.add_node(agent.id, name=agent.name, generation=agent.generation)

            if agent.parents:
                for parent_id in agent.parents:
                    if parent_id in self.all_agents:
                        self.genealogy.add_edge(parent_id, agent.id)

    def record_agent(self, agent: AgentDNA) -> None:
        """Record a single agent (e.g., newly born offspring)."""
        self.all_agents[agent.id] = agent
        self.genealogy.add_node(agent.id, name=agent.name, generation=agent.generation)
        if agent.parents:
            for parent_id in agent.parents:
                self.genealogy.add_edge(parent_id, agent.id)

    def get_ancestry(self, agent_id: str) -> list[AgentDNA]:
        """Trace an agent's lineage back to seed agents."""
        ancestors = []
        visited = set()
        queue = [agent_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            agent = self.all_agents.get(current)
            if agent:
                ancestors.append(agent)
                if agent.parents:
                    queue.extend(agent.parents)

        return ancestors

    def get_descendants(self, agent_id: str) -> list[AgentDNA]:
        """Find all descendants of an agent."""
        descendants = []
        if agent_id in self.genealogy:
            for desc_id in nx.descendants(self.genealogy, agent_id):
                agent = self.all_agents.get(desc_id)
                if agent:
                    descendants.append(agent)
        return descendants

    def fitness_over_time(self) -> list[dict[str, float]]:
        """Best, average, worst fitness per generation."""
        return [
            {
                "generation": g.generation,
                "best": g.best_fitness,
                "average": g.average_fitness,
                "worst": g.worst_fitness,
            }
            for g in self.generations
        ]

    def species_diversity_over_time(self) -> list[dict[str, Any]]:
        """Diversity metrics per generation."""
        return [
            {
                "generation": g.generation,
                "species_count": g.species_count,
                "diversity_index": g.diversity_index,
            }
            for g in self.generations
        ]

    def export_genealogy_dot(self) -> str:
        """Export genealogy as a Graphviz DOT string."""
        lines = ["digraph Genealogy {", "  rankdir=TB;"]

        for node_id in self.genealogy.nodes:
            agent = self.all_agents.get(node_id)
            if agent:
                label = f"{agent.name}\\ngen={agent.generation}\\nfit={agent.latest_fitness:.2f}"
                color = "green" if agent.latest_fitness > 0.7 else "orange" if agent.latest_fitness > 0.4 else "red"
                lines.append(f'  "{node_id[:8]}" [label="{label}", color="{color}"];')

        for u, v in self.genealogy.edges:
            lines.append(f'  "{u[:8]}" -> "{v[:8]}";')

        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary of evolution history."""
        if not self.generations:
            return "No generations recorded yet."

        latest = self.generations[-1]
        best_ever = max(self.all_agents.values(), key=lambda a: a.best_fitness) if self.all_agents else None

        parts = [
            f"Evolution History Summary",
            f"========================",
            f"Generations: {len(self.generations)}",
            f"Total agents created: {len(self.all_agents)}",
            f"Current generation: {latest.generation}",
            f"  Population: {latest.population_size}",
            f"  Best fitness: {latest.best_fitness:.4f}",
            f"  Avg fitness: {latest.average_fitness:.4f}",
            f"  Species: {latest.species_count}",
            f"  Diversity: {latest.diversity_index:.4f}",
        ]

        if best_ever:
            parts.extend([
                f"",
                f"Best agent ever: {best_ever.name}",
                f"  Fitness: {best_ever.best_fitness:.4f}",
                f"  Generation: {best_ever.generation}",
                f"  Genes: {best_ever.genome.gene_count}",
            ])

        return "\n".join(parts)
