"""Command-line interface for the Inception evolution engine.

Usage:
    inception evolve --seed-dir examples/seed_agents --generations 10
    inception inspect --agent path/to/agent.yaml
    inception mate --parent-a path/to/a.yaml --parent-b path/to/b.yaml
    inception export --agent path/to/agent.yaml --format openclaw --output ./exported/
    inception serve --agent examples/seed_agents/coder_agent --port 3000 --allow-mating
    inception discover --task "data analysis" --registry http://localhost:8080
    inception register --agent examples/seed_agents/coder_agent --registry http://localhost:8080
    inception remote-mate --target http://localhost:3001 --agent examples/seed_agents/coder_agent
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Inception — AI Agent Code Evolution Through Mating.

    Evolve AI agents by exchanging code like biological organisms
    exchange genetic material through sexual reproduction.
    """
    pass


@main.command()
@click.option("--seed-dir", type=click.Path(exists=True), required=True,
              help="Directory containing seed agent subdirectories")
@click.option("--generations", "-g", type=int, default=10, help="Number of generations")
@click.option("--population-size", "-p", type=int, default=10, help="Population size")
@click.option("--api-key", type=str, default="", help="Anthropic API key for LLM-driven crossover")
@click.option("--output-dir", "-o", type=str, default="evolution_output", help="Output directory")
@click.option("--crossover", type=click.Choice(["chromosome", "gene", "semantic", "adaptive"]),
              default="chromosome", help="Crossover strategy")
def evolve(seed_dir: str, generations: int, population_size: int,
           api_key: str, output_dir: str, crossover: str):
    """Run evolution on a population of seed agents."""
    from inception.evolution.config import EvolutionConfig
    from inception.evolution.loop import EvolutionLoop
    from inception.genome.serialization import load_seed_agent_from_directory

    console.print(Panel.fit(
        "[bold green]Inception Evolution Engine[/bold green]",
        title="Starting",
    ))

    # Load seed agents
    seed_path = Path(seed_dir)
    agents = []
    for agent_dir in sorted(seed_path.iterdir()):
        if agent_dir.is_dir() and (agent_dir / "genome.yaml").exists():
            agent = load_seed_agent_from_directory(agent_dir)
            agents.append(agent)
            console.print(f"  Loaded: [cyan]{agent.name}[/cyan]")

    if not agents:
        console.print("[red]No seed agents found![/red]")
        return

    config = EvolutionConfig(
        population_size=population_size,
        offspring_count=max(2, population_size // 2),
        max_generations=generations,
        crossover_strategy=crossover,
        llm_api_key=api_key,
        output_dir=output_dir,
    )

    loop = EvolutionLoop(config)
    result = asyncio.run(loop.run(agents, num_generations=generations))

    if result.best_agent:
        console.print(f"\n[bold green]Best agent: {result.best_agent.name}[/bold green]")
        console.print(f"Fitness: {result.best_agent.latest_fitness:.4f}")


@main.command()
@click.argument("agent_path", type=click.Path(exists=True))
def inspect(agent_path: str):
    """Inspect an agent's genome structure."""
    from inception.genome.serialization import load_agent_yaml, load_seed_agent_from_directory

    path = Path(agent_path)

    if path.is_dir():
        agent = load_seed_agent_from_directory(path)
    else:
        agent = load_agent_yaml(path)

    tree = Tree(f"[bold]{agent.name}[/bold] (gen={agent.generation})")

    for chrom in agent.genome.chromosomes:
        branch = tree.add(f"[cyan]{chrom.kind.value}[/cyan]: {chrom.name}")
        if chrom.interface:
            branch.add(f"[dim]capability: {chrom.interface.capability}[/dim]")
        for gene in chrom.genes:
            gene_label = f"[green]{gene.kind.value}[/green]: {gene.name}"
            if gene.kind.value == "function":
                lines = gene.source.count("\n") + 1
                gene_label += f" ({lines} lines)"
            branch.add(gene_label)

    console.print(tree)
    console.print(f"\nTotal genes: {agent.genome.gene_count}")
    console.print(f"Chromosomes: {len(agent.genome.chromosomes)}")
    console.print(f"Fitness: {agent.latest_fitness:.4f}")

    caps = agent.genome.capability_vector()
    if caps:
        console.print("\nCapabilities:")
        for cap, strength in sorted(caps.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(strength * 20)
            console.print(f"  {cap:30s} {bar} {strength:.2f}")


@main.command()
@click.option("--parent-a", type=click.Path(exists=True), required=True)
@click.option("--parent-b", type=click.Path(exists=True), required=True)
@click.option("--output", "-o", type=click.Path(), default="offspring.yaml")
@click.option("--api-key", type=str, default="", help="Anthropic API key")
def mate(parent_a: str, parent_b: str, output: str, api_key: str):
    """Mate two agents to produce an offspring."""
    from inception.genome.compatibility import compatibility_score
    from inception.genome.serialization import load_seed_agent_from_directory, save_agent_yaml
    from inception.llm.client import LLMClient, LLMConfig
    from inception.reproduction.crossover import ChromosomeLevelCrossover, GeneLevelCrossover

    path_a, path_b = Path(parent_a), Path(parent_b)
    agent_a = load_seed_agent_from_directory(path_a) if path_a.is_dir() else None
    agent_b = load_seed_agent_from_directory(path_b) if path_b.is_dir() else None

    if not agent_a or not agent_b:
        console.print("[red]Could not load both parents[/red]")
        return

    compat = compatibility_score(agent_a.genome, agent_b.genome)
    console.print(f"Compatibility: {compat:.4f}")

    if api_key:
        llm = LLMClient(LLMConfig(api_key=api_key))
        crossover = GeneLevelCrossover()
    else:
        llm = LLMClient()
        crossover = ChromosomeLevelCrossover()

    child_genome = asyncio.run(crossover.crossover(agent_a.genome, agent_b.genome, llm))

    from inception.genome.schema import AgentDNA
    child = AgentDNA(
        name=f"{agent_a.name[:4]}x{agent_b.name[:4]}",
        genome=child_genome,
        generation=1,
        parents=(agent_a.id, agent_b.id),
    )

    save_agent_yaml(child, Path(output))
    console.print(f"[green]Offspring saved to {output}[/green]")
    console.print(f"Genes: {child.genome.gene_count}")


@main.command()
@click.argument("agent_path", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["openclaw", "hermes"]), required=True)
@click.option("--output", "-o", type=click.Path(), required=True)
def export(agent_path: str, fmt: str, output: str):
    """Export an agent to OpenClaw or Hermes format."""
    from inception.genome.serialization import load_agent_yaml, load_seed_agent_from_directory

    path = Path(agent_path)
    if path.is_dir():
        agent = load_seed_agent_from_directory(path)
    else:
        agent = load_agent_yaml(path)

    output_path = Path(output)

    if fmt == "openclaw":
        from inception.agent.adapters.openclaw import OpenClawAdapter
        adapter = OpenClawAdapter()
        adapter.genome_to_openclaw(agent.genome, output_path)
        console.print(f"[green]Exported to OpenClaw format: {output_path}[/green]")
    elif fmt == "hermes":
        from inception.agent.adapters.hermes import HermesAdapter
        adapter = HermesAdapter()
        adapter.genome_to_hermes_file(agent.genome, output_path / "config.json")
        console.print(f"[green]Exported to Hermes format: {output_path}[/green]")


def _register_a2a_commands() -> None:
    """Register A2A networking commands (lazy import)."""
    from inception.a2a.cli_commands import (
        discover,
        fitness_cmd,
        pool,
        register,
        registry_start,
        remote_mate,
        serve,
    )
    main.add_command(serve)
    main.add_command(discover)
    main.add_command(register)
    main.add_command(remote_mate, name="remote-mate")
    main.add_command(pool)
    main.add_command(registry_start, name="registry")
    main.add_command(fitness_cmd, name="fitness")


_register_a2a_commands()


if __name__ == "__main__":
    main()
