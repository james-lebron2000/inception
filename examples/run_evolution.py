"""End-to-end evolution demo.

Loads 3 seed agents (coder, researcher, planner), runs evolution for
several generations, and displays the results.

This demo uses CodeQualityEvaluator (no LLM API needed for fitness).
Crossover and mutation require an ANTHROPIC_API_KEY environment variable,
but the demo falls back to chromosome-level crossover if no key is set.

Usage:
    python examples/run_evolution.py
    python examples/run_evolution.py --generations 10
    python examples/run_evolution.py --api-key sk-ant-...
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from inception.evolution.config import EvolutionConfig
from inception.evolution.loop import EvolutionLoop
from inception.genome.serialization import load_seed_agent_from_directory

console = Console()


def load_seed_agents() -> list:
    """Load all seed agents from the examples/seed_agents directory."""
    seed_dir = Path(__file__).parent / "seed_agents"
    agents = []

    for agent_dir in sorted(seed_dir.iterdir()):
        if agent_dir.is_dir() and (agent_dir / "genome.yaml").exists():
            agent = load_seed_agent_from_directory(agent_dir)
            agents.append(agent)
            console.print(f"  Loaded: [cyan]{agent.name}[/cyan] "
                         f"({agent.genome.gene_count} genes, "
                         f"{len(agent.genome.chromosomes)} chromosomes)")

    return agents


def display_agent_tree(agent, title: str = "Agent") -> None:
    """Display an agent's genome as a tree."""
    tree = Tree(f"[bold]{title}: {agent.name}[/bold] (fitness={agent.latest_fitness:.4f})")

    for chrom in agent.genome.chromosomes:
        branch = tree.add(f"[cyan]{chrom.kind.value}[/cyan]: {chrom.name}")
        for gene in chrom.genes:
            gene_info = f"[green]{gene.kind.value}[/green]: {gene.name}"
            if gene.kind.value == "function":
                lines = gene.source.count("\n") + 1
                gene_info += f" ({lines} lines)"
            branch.add(gene_info)

    console.print(tree)


async def main():
    """Run the evolution demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Inception Evolution Demo")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=10, help="Population size")
    parser.add_argument("--api-key", type=str, default="", help="Anthropic API key")
    parser.add_argument("--output-dir", type=str, default="evolution_output", help="Output directory")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]🧬 Inception: AI Agent Code Evolution Through Mating[/bold green]\n"
        "Evolving AI agents by exchanging code like biological organisms exchange genetic material.",
        title="Welcome",
    ))

    # Load seed agents
    console.print("\n[bold]Loading seed agents...[/bold]")
    seed_agents = load_seed_agents()

    if not seed_agents:
        console.print("[red]No seed agents found! Check examples/seed_agents/[/red]")
        return

    console.print(f"\n[bold]Seed population: {len(seed_agents)} agents[/bold]")
    for agent in seed_agents:
        display_agent_tree(agent, "Seed")

    # Configure evolution
    config = EvolutionConfig(
        population_size=args.population_size,
        offspring_count=max(2, args.population_size // 2),
        elite_count=min(2, len(seed_agents)),
        max_generations=args.generations,
        crossover_strategy="chromosome",  # No LLM needed for chromosome-level
        crossover_rate=0.7,
        selection_strategy="elitist",
        mate_selection="hybrid",
        llm_api_key=args.api_key,
        output_dir=args.output_dir,
        save_every_n_generations=2,
        verbose=True,
    )

    # Use LLM-driven crossover if API key is available
    if args.api_key:
        config.crossover_strategy = "adaptive"
        console.print("\n[green]API key provided — using LLM-driven adaptive crossover[/green]")
    else:
        console.print("\n[yellow]No API key — using chromosome-level crossover (no LLM needed)[/yellow]")

    # Run evolution
    loop = EvolutionLoop(config)
    result = await loop.run(seed_agents, num_generations=args.generations)

    # Display results
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        "[bold green]Evolution Complete![/bold green]",
        title="Results",
    ))

    console.print(f"\nGenerations run: {result.generations_run}")
    console.print(f"Termination: {result.termination_reason}")
    console.print(f"Total agents created: {len(result.history.all_agents)}")

    if result.best_agent:
        console.print(f"\n[bold]Best agent:[/bold]")
        display_agent_tree(result.best_agent, "Champion")

        # Show ancestry
        ancestry = result.history.get_ancestry(result.best_agent.id)
        if len(ancestry) > 1:
            console.print(f"\n[bold]Lineage ({len(ancestry)} ancestors):[/bold]")
            for ancestor in ancestry:
                gen = ancestor.generation
                fit = ancestor.latest_fitness
                console.print(f"  Gen {gen}: {ancestor.name} (fitness={fit:.4f})")

    # Fitness over time
    fitness_history = result.history.fitness_over_time()
    if fitness_history:
        console.print("\n[bold]Fitness over time:[/bold]")
        for entry in fitness_history:
            gen = entry["generation"]
            best = entry["best"]
            avg = entry["average"]
            bar_best = "█" * int(best * 30)
            bar_avg = "▒" * int(avg * 30)
            console.print(f"  Gen {gen:3d}: Best={best:.3f} {bar_best}")
            console.print(f"           Avg ={avg:.3f} {bar_avg}")

    console.print(f"\n{result.history.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
