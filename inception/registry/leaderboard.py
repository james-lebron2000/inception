"""Leaderboard queries for the Gene Registry."""

from __future__ import annotations

from inception.registry.db import RegistryDB


async def get_fitness_leaderboard(
    db: RegistryDB, limit: int = 20
) -> list[dict[str, object]]:
    """Get top agents ranked by best fitness."""
    entries = await db.get_leaderboard(limit)
    return [
        {
            "rank": i + 1,
            "name": e.agent_card.name,
            "agent_id": e.agent_card.agent_id,
            "best_fitness": e.agent_card.best_fitness,
            "generation": e.agent_card.generation,
            "gene_count": e.agent_card.gene_count,
        }
        for i, e in enumerate(entries)
    ]


async def get_most_prolific(
    db: RegistryDB, limit: int = 20
) -> list[dict[str, object]]:
    """Get agents with the most offspring."""
    entries = await db.get_all_active(limit=200)
    sorted_entries = sorted(entries, key=lambda e: e.offspring_count, reverse=True)
    return [
        {
            "rank": i + 1,
            "name": e.agent_card.name,
            "agent_id": e.agent_card.agent_id,
            "offspring_count": e.offspring_count,
            "mating_count": e.mating_count,
        }
        for i, e in enumerate(sorted_entries[:limit])
    ]
