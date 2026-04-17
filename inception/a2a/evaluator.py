"""Offspring fitness evaluation and parent comparison.

Evaluates offspring agents against their parents to determine
whether the offspring represents an improvement worth deploying.
"""

from __future__ import annotations

from inception.fitness.evaluator import CodeQualityEvaluator
from inception.genome.schema import AgentDNA


async def evaluate_offspring(
    offspring: AgentDNA,
    parents: list[AgentDNA],
    threshold: float = 0.95,
) -> dict[str, object]:
    """Evaluate offspring fitness and compare against parents.

    Returns a report dict with fitness scores and a recommendation
    on whether to deploy the offspring.
    """
    evaluator = CodeQualityEvaluator()

    offspring_score = await evaluator.evaluate(offspring)
    parent_scores = [await evaluator.evaluate(p) for p in parents]

    best_parent_fitness = max(s.overall for s in parent_scores) if parent_scores else 0.0
    offspring_fitness = offspring_score.overall

    beats_parents = offspring_fitness > best_parent_fitness * threshold

    return {
        "offspring_name": offspring.name,
        "offspring_fitness": offspring_fitness,
        "offspring_dimensions": dict(offspring_score.dimensions),
        "parent_fitnesses": [s.overall for s in parent_scores],
        "best_parent_fitness": best_parent_fitness,
        "threshold": threshold,
        "beats_parents": beats_parents,
        "recommendation": "deploy" if beats_parents else "archive",
    }
