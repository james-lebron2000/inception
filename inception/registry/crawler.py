"""Agent card URL crawler for the Gene Registry.

Periodically refreshes agent cards from registered URLs
and detects stale/offline agents.
"""

from __future__ import annotations

import asyncio

from inception.a2a.client import A2AClient
from inception.registry.db import RegistryDB


async def crawl_agents(db: RegistryDB, timeout: float = 10.0) -> dict[str, str]:
    """Refresh all registered agent cards.

    Returns a dict mapping agent_id to status ("updated", "offline", "error").
    """
    client = A2AClient(timeout=timeout)
    entries = await db.get_all_active(limit=1000)
    results: dict[str, str] = {}

    for entry in entries:
        agent_id = entry.agent_card.agent_id
        url = entry.agent_card.url
        try:
            card = await client.discover(url)
            await db.update_agent(card)
            results[agent_id] = "updated"
        except Exception:
            # Agent is offline or unreachable
            await db.deactivate_agent(agent_id)
            results[agent_id] = "offline"

    return results


async def run_crawler_loop(
    db: RegistryDB, interval_seconds: int = 300, timeout: float = 10.0
) -> None:
    """Run the crawler in a loop, refreshing agents periodically."""
    while True:
        await crawl_agents(db, timeout)
        await asyncio.sleep(interval_seconds)
