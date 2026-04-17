"""CLI commands for the A2A networking layer.

Adds serve, discover, register, remote-mate, pool, and registry
commands to the Inception CLI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command()
@click.option("--agent", type=click.Path(exists=True), required=True,
              help="Path to agent directory or YAML file")
@click.option("--port", "-p", type=int, default=3000, help="Port to serve on")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to")
@click.option("--api-key", type=str, default="", help="API key for authentication")
@click.option("--llm-key", type=str, default="", help="Anthropic API key for LLM crossover")
@click.option("--allow-mating", is_flag=True, help="Allow remote mating requests")
@click.option("--share-source", is_flag=True, help="Share source code in genome exports")
@click.option("--max-matings", type=int, default=10, help="Max matings per day")
def serve(agent: str, port: int, host: str, api_key: str, llm_key: str,
          allow_mating: bool, share_source: bool, max_matings: int):
    """Start an A2A server to expose this agent for discovery and mating."""
    from inception.a2a.models import MatingPolicy
    from inception.a2a.server import A2AServer
    from inception.genome.serialization import load_agent_yaml, load_seed_agent_from_directory

    path = Path(agent)
    if path.is_dir():
        agent_dna = load_seed_agent_from_directory(path)
    else:
        agent_dna = load_agent_yaml(path)

    policy = MatingPolicy(
        allow_remote_mating=allow_mating,
        share_source_code=share_source,
        max_matings_per_day=max_matings,
        share_fitness_scores=True,
        share_capability_vector=True,
    )

    console.print(Panel.fit(
        f"[bold green]Serving agent: {agent_dna.name}[/bold green]\n"
        f"Port: {port} | Mating: {'enabled' if allow_mating else 'disabled'}\n"
        f"Source sharing: {'yes' if share_source else 'no'}",
        title="A2A Server",
    ))

    server = A2AServer(
        agent=agent_dna,
        policy=policy,
        host=host,
        port=port,
        api_key=api_key,
        llm_api_key=llm_key,
    )
    server.run()


@click.command()
@click.option("--task", "-t", type=str, default="", help="Task description for matchmaking")
@click.option("--registry", type=str, default="", help="Registry URL for discovery")
@click.option("--url", type=str, default="", help="Direct agent URL to discover")
@click.option("--top", type=int, default=5, help="Number of results to show")
def discover(task: str, registry: str, url: str, top: int):
    """Discover compatible agents for a task."""
    async def _discover():
        from inception.a2a.client import A2AClient

        client = A2AClient()

        if url:
            card = await client.discover(url)
            table = Table(title="Discovered Agent")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Name", card.name)
            table.add_row("URL", card.url)
            table.add_row("Generation", str(card.generation))
            table.add_row("Genes", str(card.gene_count))
            table.add_row("Best Fitness", f"{card.best_fitness:.4f}")
            table.add_row("Skills", ", ".join(card.skill_names))
            table.add_row("Remote Mating", str(card.mating_policy.allow_remote_mating))
            console.print(table)

        elif registry and task:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as http:
                # Build a minimal requester card for the query
                from inception.a2a.models import GeneticAgentCard
                requester = GeneticAgentCard(
                    name="requester", url="", agent_id="requester"
                )
                resp = await http.post(
                    f"{registry.rstrip('/')}/api/discover",
                    json={
                        "task": task,
                        "requester_card": requester.model_dump(),
                        "top_k": top,
                    },
                )
                resp.raise_for_status()
                matches = resp.json()

            table = Table(title=f"Matches for: {task}")
            table.add_column("#", style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("URL", style="blue")
            table.add_column("Score", style="green")
            table.add_column("Fitness", style="yellow")
            table.add_column("Skills")

            for i, m in enumerate(matches):
                card = m["agent_card"]
                table.add_row(
                    str(i + 1),
                    card["name"],
                    card["url"],
                    f"{m['score']:.3f}",
                    f"{card['best_fitness']:.3f}",
                    ", ".join(card.get("skill_names", [])),
                )
            console.print(table)
        else:
            console.print("[red]Provide --url for direct discovery or --registry + --task for matchmaking[/red]")

    asyncio.run(_discover())


@click.command()
@click.option("--agent", type=click.Path(exists=True), required=True,
              help="Path to agent directory or YAML")
@click.option("--registry", type=str, required=True, help="Registry URL")
@click.option("--serve-url", type=str, default="", help="URL where this agent is served")
def register(agent: str, registry: str, serve_url: str):
    """Register an agent with a Gene Registry."""
    async def _register():
        import httpx
        from inception.a2a.models import MatingPolicy
        from inception.genome.export import build_agent_card
        from inception.genome.serialization import load_agent_yaml, load_seed_agent_from_directory

        path = Path(agent)
        if path.is_dir():
            agent_dna = load_seed_agent_from_directory(path)
        else:
            agent_dna = load_agent_yaml(path)

        policy = MatingPolicy(allow_remote_mating=True, share_capability_vector=True)
        card = build_agent_card(agent_dna, serve_url, policy)

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                f"{registry.rstrip('/')}/api/register",
                json=card.model_dump(),
            )
            resp.raise_for_status()
            result = resp.json()

        console.print(f"[green]Registered: {agent_dna.name}[/green]")
        console.print(f"Agent ID: {result['agent_id']}")

    asyncio.run(_register())


@click.command("remote-mate")
@click.option("--agent", type=click.Path(exists=True), required=True,
              help="Path to local agent directory or YAML")
@click.option("--target", type=str, default="", help="Target agent URL")
@click.option("--registry", type=str, default="", help="Registry URL for auto-discovery")
@click.option("--task", "-t", type=str, default="", help="Task description")
@click.option("--crossover", type=click.Choice(["chromosome", "gene", "semantic", "adaptive"]),
              default="adaptive", help="Crossover strategy")
@click.option("--output", "-o", type=click.Path(), default="offspring.yaml",
              help="Output path for offspring")
@click.option("--llm-key", type=str, default="", help="Anthropic API key")
def remote_mate(agent: str, target: str, registry: str, task: str,
                crossover: str, output: str, llm_key: str):
    """Mate with a remote agent to produce offspring."""
    async def _mate():
        from inception.a2a.models import MatingPolicy
        from inception.a2a.transport import RemoteMatingOrchestrator
        from inception.genome.serialization import (
            load_agent_yaml,
            load_seed_agent_from_directory,
            save_agent_yaml,
        )

        path = Path(agent)
        if path.is_dir():
            agent_dna = load_seed_agent_from_directory(path)
        else:
            agent_dna = load_agent_yaml(path)

        policy = MatingPolicy(
            allow_remote_mating=True,
            share_source_code=True,
            share_capability_vector=True,
        )

        orchestrator = RemoteMatingOrchestrator(
            local_agent=agent_dna,
            policy=policy,
            llm_api_key=llm_key,
        )

        console.print(f"[cyan]Mating {agent_dna.name}...[/cyan]")

        if target:
            offspring = await orchestrator.mate_with_remote(
                target, task, crossover
            )
        elif registry and task:
            offspring = await orchestrator.discover_and_mate(
                registry, task, crossover_strategy=crossover
            )
        else:
            console.print("[red]Provide --target or --registry + --task[/red]")
            return

        if offspring:
            save_agent_yaml(offspring, Path(output))
            console.print(f"[green]Offspring saved: {output}[/green]")
            console.print(f"  Name: {offspring.name}")
            console.print(f"  Generation: {offspring.generation}")
            console.print(f"  Genes: {offspring.genome.gene_count}")
        else:
            console.print("[yellow]Mating failed — no compatible partner found[/yellow]")

    asyncio.run(_mate())


@click.command()
@click.option("--dir", "directory", type=click.Path(exists=True),
              default="evolution_output", help="Directory to scan")
def pool(directory: str):
    """List bred agents in a local pool directory."""
    from inception.genome.serialization import load_agent_yaml

    pool_path = Path(directory)
    yaml_files = sorted(pool_path.glob("*.yaml")) + sorted(pool_path.glob("*.yml"))

    if not yaml_files:
        console.print(f"[yellow]No agents found in {directory}[/yellow]")
        return

    table = Table(title=f"Agent Pool: {directory}")
    table.add_column("Name", style="cyan")
    table.add_column("Gen", style="green")
    table.add_column("Genes", style="yellow")
    table.add_column("Fitness", style="magenta")
    table.add_column("File", style="dim")

    for f in yaml_files:
        try:
            agent = load_agent_yaml(f)
            table.add_row(
                agent.name,
                str(agent.generation),
                str(agent.genome.gene_count),
                f"{agent.latest_fitness:.4f}",
                f.name,
            )
        except Exception:
            table.add_row("?", "?", "?", "?", f.name)

    console.print(table)


@click.command("registry")
@click.option("--port", "-p", type=int, default=8080, help="Port for registry")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to")
@click.option("--db", type=str, default="gene_registry.db", help="SQLite database path")
def registry_start(port: int, host: str, db: str):
    """Start a Gene Registry server for agent discovery."""
    from inception.registry.db import RegistryDB
    from inception.registry.registry import RegistryServer

    console.print(Panel.fit(
        f"[bold green]Gene Registry[/bold green]\n"
        f"Port: {port} | DB: {db}",
        title="Registry Server",
    ))

    registry_db = RegistryDB(db_path=db)
    server = RegistryServer(db=registry_db)
    server.run(host=host, port=port)


@click.command("fitness")
@click.argument("agent_path", type=click.Path(exists=True))
def fitness_cmd(agent_path: str):
    """Evaluate an agent's fitness."""
    async def _fitness():
        from inception.fitness.evaluator import CodeQualityEvaluator
        from inception.genome.serialization import load_agent_yaml, load_seed_agent_from_directory

        path = Path(agent_path)
        if path.is_dir():
            agent = load_seed_agent_from_directory(path)
        else:
            agent = load_agent_yaml(path)

        evaluator = CodeQualityEvaluator()
        score = await evaluator.evaluate(agent)

        console.print(Panel.fit(
            f"[bold]{agent.name}[/bold] — Fitness Report",
            title="Fitness Evaluation",
        ))
        console.print(f"Overall: [green]{score.overall:.4f}[/green]")
        for dim, val in score.dimensions.items():
            bar = "[green]" + "█" * int(val * 20) + "[/green]"
            console.print(f"  {dim:20s} {bar} {val:.3f}")

    asyncio.run(_fitness())
