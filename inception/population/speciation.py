"""Speciation — grouping agents by genetic similarity.

Prevents a single dominant strategy from taking over the population.
Each species competes internally, giving novel approaches time to develop.

Biological analog: speciation events creating reproductively isolated groups.
Inspired by the NEAT algorithm's speciation mechanism.
"""

from __future__ import annotations

import uuid
from typing import Any

from inception.genome.compatibility import compatibility_score
from inception.genome.schema import AgentDNA
from inception.population.population import Species


class SpeciationEngine:
    """NEAT-style speciation: group agents by genome similarity."""

    def __init__(self, compatibility_threshold: float = 0.4):
        self.compatibility_threshold = compatibility_threshold

    def assign_species(
        self,
        agents: list[AgentDNA],
        existing_species: list[Species] | None = None,
    ) -> list[Species]:
        """Assign each agent to a species based on genome compatibility.

        Algorithm:
        1. For each existing species, keep its representative
        2. For each agent, compare to each species representative
        3. If compatible (score > threshold), join that species
        4. If not compatible with any, create a new species
        """
        species_list: list[Species] = []
        representatives: dict[str, AgentDNA] = {}

        # Seed with existing species representatives
        if existing_species:
            for sp in existing_species:
                rep = next((a for a in agents if a.id == sp.representative_id), None)
                if rep:
                    species_list.append(Species(
                        id=sp.id,
                        name=sp.name,
                        representative_id=rep.id,
                        member_ids=[],
                    ))
                    representatives[sp.id] = rep

        # Assign each agent
        for agent in agents:
            assigned = False
            for sp in species_list:
                rep = representatives.get(sp.id)
                if rep is None:
                    continue

                score = compatibility_score(agent.genome, rep.genome)
                if score >= self.compatibility_threshold:
                    sp.member_ids.append(agent.id)
                    assigned = True
                    break

            if not assigned:
                # Create new species with this agent as representative
                sp_id = str(uuid.uuid4())[:8]
                new_species = Species(
                    id=sp_id,
                    name=f"species_{sp_id}",
                    representative_id=agent.id,
                    member_ids=[agent.id],
                )
                species_list.append(new_species)
                representatives[sp_id] = agent

        # Compute average fitness per species
        agent_map = {a.id: a for a in agents}
        for sp in species_list:
            members = [agent_map[mid] for mid in sp.member_ids if mid in agent_map]
            if members:
                sp.avg_fitness = sum(a.latest_fitness for a in members) / len(members)

        # Remove empty species
        species_list = [sp for sp in species_list if sp.size > 0]

        return species_list

    def group_by_similarity(self, agents: list[AgentDNA]) -> list[list[AgentDNA]]:
        """Group agents into clusters by genome similarity.

        Simple greedy clustering (no existing species context).
        Returns list of groups.
        """
        if not agents:
            return []

        groups: list[list[AgentDNA]] = []
        assigned: set[str] = set()

        for agent in agents:
            if agent.id in assigned:
                continue

            # Start a new group
            group = [agent]
            assigned.add(agent.id)

            # Find all compatible agents
            for other in agents:
                if other.id in assigned:
                    continue
                score = compatibility_score(agent.genome, other.genome)
                if score >= self.compatibility_threshold:
                    group.append(other)
                    assigned.add(other.id)

            groups.append(group)

        return groups

    def adjusted_fitness(self, agent: AgentDNA, species: Species) -> float:
        """Compute fitness adjusted by species size (fitness sharing).

        Agents in smaller species get a fitness boost, encouraging diversity.
        """
        if species.size == 0:
            return agent.latest_fitness
        return agent.latest_fitness / species.size
