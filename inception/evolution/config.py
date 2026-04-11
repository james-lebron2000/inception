"""Evolution hyperparameter configuration.

All tunable parameters for the evolutionary process, with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inception.reproduction.mutation import MutationConfig


@dataclass
class EvolutionConfig:
    """Master configuration for the evolution loop."""

    # Population
    population_size: int = 20
    offspring_count: int = 10  # Children per generation
    elite_count: int = 2  # Always survive

    # Reproduction
    crossover_rate: float = 0.7  # Probability of crossover vs. cloning
    crossover_strategy: str = "adaptive"  # "chromosome", "gene", "semantic", "adaptive"

    # Mutation
    mutation: MutationConfig = field(default_factory=MutationConfig)

    # Selection
    selection_strategy: str = "elitist"  # "elitist", "truncation", "diversity"
    tournament_size: int = 3
    mate_selection: str = "hybrid"  # "tournament", "proportional", "complementary", "hybrid"

    # Speciation
    compatibility_threshold: float = 0.4

    # LLM
    llm_model: str = "claude-sonnet-4-20250514"
    llm_temperature: float = 0.7
    llm_api_key: str = ""

    # Fitness
    fitness_evaluator: str = "code_quality"  # "code_quality", "llm_judge", "composite"

    # Termination
    max_generations: int = 50
    fitness_target: float = 0.95  # Stop if any agent reaches this
    stagnation_limit: int = 10  # Stop if no improvement for N generations

    # Output
    output_dir: str = "evolution_output"
    save_every_n_generations: int = 5
    verbose: bool = True
