"""Demo: Full A2A remote mating cycle.

Shows how two agents can exchange genetic material over the network
and produce an optimized offspring.

Usage:
    # Terminal 1: Start agent A
    inception serve --agent examples/seed_agents/coder_agent --port 3001 --allow-mating --share-source

    # Terminal 2: Start agent B
    inception serve --agent examples/seed_agents/researcher_agent --port 3002 --allow-mating --share-source

    # Terminal 3: Run this demo
    python examples/demo_remote_mate.py
"""

from __future__ import annotations

import asyncio

from inception.a2a.client import A2AClient
from inception.a2a.models import MatingPolicy
from inception.a2a.transport import RemoteMatingOrchestrator
from inception.genome.serialization import load_seed_agent_from_directory, save_agent_yaml
from pathlib import Path


async def main():
    # Load the local agent (parent A)
    parent_a = load_seed_agent_from_directory(Path("examples/seed_agents/coder_agent"))
    print(f"Parent A: {parent_a.name} (genes={parent_a.genome.gene_count})")

    # Configure mating policy
    policy = MatingPolicy(
        allow_remote_mating=True,
        share_source_code=True,
        share_capability_vector=True,
    )

    # Create the orchestrator
    orchestrator = RemoteMatingOrchestrator(
        local_agent=parent_a,
        policy=policy,
    )

    # Discover the remote agent
    client = A2AClient()
    target_url = "http://localhost:3002"

    print(f"\nDiscovering agent at {target_url}...")
    try:
        card = await client.discover(target_url)
        print(f"Found: {card.name} (gen={card.generation}, fitness={card.best_fitness:.3f})")
        print(f"Skills: {', '.join(card.skill_names)}")
    except Exception as e:
        print(f"Could not discover agent: {e}")
        print("Make sure to start agent B first:")
        print("  inception serve --agent examples/seed_agents/researcher_agent --port 3002 --allow-mating --share-source")
        return

    # Mate!
    print("\nInitiating mating...")
    offspring = await orchestrator.mate_with_remote(
        target_url=target_url,
        task="analyze code and research solutions",
        crossover_strategy="chromosome",
    )

    if offspring:
        print(f"\nOffspring born: {offspring.name}")
        print(f"  Generation: {offspring.generation}")
        print(f"  Genes: {offspring.genome.gene_count}")
        print(f"  Chromosomes: {len(offspring.genome.chromosomes)}")

        output = Path("offspring_demo.yaml")
        save_agent_yaml(offspring, output)
        print(f"\nSaved to: {output}")
    else:
        print("\nMating failed — no offspring produced.")


if __name__ == "__main__":
    asyncio.run(main())
