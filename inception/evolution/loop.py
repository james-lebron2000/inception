"""Main evolution loop — the top-level orchestrator.

Ties together crossover, mutation, fitness evaluation, selection,
and speciation into a generational evolutionary cycle.

Biological analog: the cycle of life — birth, growth, selection, reproduction, death.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from inception.evolution.config import EvolutionConfig
from inception.evolution.history import EvolutionHistory
from inception.fitness.evaluator import CodeQualityEvaluator, CompositeEvaluator, FitnessEvaluator
from inception.fitness.task_runner import TaskRunner
from inception.genome.schema import AgentDNA, AgentStatus
from inception.genome.serialization import save_agent_yaml
from inception.llm.client import LLMClient, LLMConfig
from inception.population.population import Population
from inception.population.selection import (
    DiversityPreservingSelector,
    ElitistSelector,
    SurvivalSelector,
    TruncationSelector,
)
from inception.population.speciation import SpeciationEngine
from inception.reproduction.crossover import (
    AdaptiveCrossover,
    ChromosomeLevelCrossover,
    CrossoverStrategy,
    GeneLevelCrossover,
    SemanticCrossover,
)
from inception.reproduction.mate_selection import (
    ComplementarySelector,
    FitnessProportionalSelector,
    HybridSelector,
    MateSelector,
    TournamentMateSelector,
)
from inception.reproduction.mutation import MutationEngine

console = Console()


@dataclass
class EvolutionResult:
    """Result of an evolution run."""

    best_agent: AgentDNA | None
    final_population: Population
    history: EvolutionHistory
    generations_run: int
    termination_reason: str


class EvolutionLoop:
    """Main evolution loop orchestrator."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.llm = LLMClient(LLMConfig(
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
        ))
        self.crossover = self._build_crossover()
        self.mutation = MutationEngine(self.llm, config.mutation)
        self.evaluator = self._build_evaluator()
        self.selector = self._build_selector()
        self.mate_selector = self._build_mate_selector()
        self.speciation = SpeciationEngine(config.compatibility_threshold)
        self.history = EvolutionHistory()
        self.task_runner = TaskRunner(self.evaluator)

    def _build_crossover(self) -> CrossoverStrategy:
        strategies = {
            "chromosome": ChromosomeLevelCrossover,
            "gene": lambda: GeneLevelCrossover(merge_rate=0.3),
            "semantic": SemanticCrossover,
            "adaptive": AdaptiveCrossover,
        }
        factory = strategies.get(self.config.crossover_strategy, AdaptiveCrossover)
        return factory()

    def _build_evaluator(self) -> FitnessEvaluator:
        return CodeQualityEvaluator()

    def _build_selector(self) -> SurvivalSelector:
        selectors = {
            "elitist": lambda: ElitistSelector(
                elite_count=self.config.elite_count,
                tournament_size=self.config.tournament_size,
            ),
            "truncation": TruncationSelector,
            "diversity": DiversityPreservingSelector,
        }
        factory = selectors.get(self.config.selection_strategy, lambda: ElitistSelector())
        return factory()

    def _build_mate_selector(self) -> MateSelector:
        selectors = {
            "tournament": lambda: TournamentMateSelector(self.config.tournament_size),
            "proportional": FitnessProportionalSelector,
            "complementary": ComplementarySelector,
            "hybrid": lambda: HybridSelector(self.config.tournament_size),
        }
        factory = selectors.get(self.config.mate_selection, lambda: HybridSelector())
        return factory()

    async def run(
        self, seed_population: list[AgentDNA], num_generations: int | None = None
    ) -> EvolutionResult:
        """Run the full evolution loop.

        Args:
            seed_population: Initial population of agents.
            num_generations: Override max_generations from config.

        Returns:
            EvolutionResult with the best agent, final population, and history.
        """
        max_gen = num_generations or self.config.max_generations
        population = Population(agents=seed_population, generation=0)
        stagnation_counter = 0
        best_fitness_ever = 0.0
        termination_reason = "max_generations"

        if self.config.verbose:
            console.print("\n[bold green]🧬 Inception Evolution Engine Started[/bold green]")
            console.print(f"Population size: {population.size}")
            console.print(f"Max generations: {max_gen}")
            console.print(f"Crossover strategy: {self.config.crossover_strategy}")
            console.print()

        for gen in range(max_gen):
            if self.config.verbose:
                console.print(f"\n[bold cyan]━━━ Generation {gen} ━━━[/bold cyan]")

            # 1. Evaluate fitness
            await self.task_runner.evaluate_population(population.agents)

            # 2. Assign species
            population.species = self.speciation.assign_species(
                population.agents, population.species
            )

            # 3. Record history
            self.history.record_generation(population)

            # 4. Display stats
            if self.config.verbose:
                self._display_generation(population)

            # 5. Check termination
            best = population.best_agent()
            if best and best.latest_fitness >= self.config.fitness_target:
                termination_reason = f"fitness_target_reached ({best.latest_fitness:.4f})"
                break

            if best:
                if best.latest_fitness > best_fitness_ever:
                    best_fitness_ever = best.latest_fitness
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            if stagnation_counter >= self.config.stagnation_limit:
                termination_reason = f"stagnation ({stagnation_counter} generations)"
                break

            # 6. Select parents
            parent_pairs = self.mate_selector.select_pairs(
                population.agents, self.config.offspring_count
            )

            # 7. Produce offspring
            offspring: list[AgentDNA] = []
            for parent_a, parent_b in parent_pairs:
                child = await self._produce_offspring(parent_a, parent_b, gen + 1)
                offspring.append(child)
                self.history.record_agent(child)

            # 8. Evaluate offspring
            await self.task_runner.evaluate_population(offspring)

            # 9. Select survivors
            all_agents = population.agents + offspring
            survivors = self.selector.select(all_agents, self.config.population_size)

            population = Population(agents=survivors, generation=gen + 1)

            # 10. Save checkpoint
            if (gen + 1) % self.config.save_every_n_generations == 0:
                self._save_checkpoint(population, gen + 1)

        # Final evaluation
        await self.task_runner.evaluate_population(population.agents)
        self.history.record_generation(population)

        if self.config.verbose:
            console.print(f"\n[bold green]Evolution complete![/bold green]")
            console.print(f"Reason: {termination_reason}")
            console.print(self.history.summary())

        return EvolutionResult(
            best_agent=population.best_agent(),
            final_population=population,
            history=self.history,
            generations_run=population.generation,
            termination_reason=termination_reason,
        )

    async def _produce_offspring(
        self, parent_a: AgentDNA, parent_b: AgentDNA, generation: int
    ) -> AgentDNA:
        """Mate two parents to produce one offspring."""
        import random

        # Crossover or clone
        if random.random() < self.config.crossover_rate:
            child_genome = await self.crossover.crossover(
                parent_a.genome, parent_b.genome, self.llm
            )
        else:
            # Clone the fitter parent
            fitter = parent_a if parent_a.latest_fitness >= parent_b.latest_fitness else parent_b
            child_genome = fitter.genome.model_copy(deep=True)
            child_genome.id = str(uuid.uuid4())

        # Apply mutations
        child_genome, mutation_records = await self.mutation.mutate(child_genome)

        child = AgentDNA(
            name=f"{parent_a.name[:4]}x{parent_b.name[:4]}_g{generation}",
            genome=child_genome,
            generation=generation,
            parents=(parent_a.id, parent_b.id),
        )

        return child

    def _display_generation(self, population: Population) -> None:
        """Display generation statistics using Rich."""
        stats = population.get_stats()

        table = Table(title=f"Generation {stats.generation}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Population", str(stats.population_size))
        table.add_row("Best Fitness", f"{stats.best_fitness:.4f}")
        table.add_row("Avg Fitness", f"{stats.average_fitness:.4f}")
        table.add_row("Worst Fitness", f"{stats.worst_fitness:.4f}")
        table.add_row("Species", str(stats.species_count))
        table.add_row("Diversity", f"{stats.diversity_index:.4f}")
        table.add_row("Avg Genes", f"{stats.gene_count_avg:.1f}")

        best = population.best_agent()
        if best:
            table.add_row("Best Agent", best.name)

        console.print(table)

    def _save_checkpoint(self, population: Population, generation: int) -> None:
        """Save population checkpoint to disk."""
        output_dir = Path(self.config.output_dir) / f"gen_{generation:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for agent in population.agents:
            save_agent_yaml(agent, output_dir / f"{agent.name}.yaml")
