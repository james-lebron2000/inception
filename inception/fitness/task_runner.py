"""Task runner — executes agents against benchmark tasks.

Coordinates agent instantiation, task execution, and result collection.
"""

from __future__ import annotations

from typing import Any

from inception.genome.schema import AgentDNA, FitnessScore, GeneKind
from inception.fitness.evaluator import FitnessEvaluator, CodeQualityEvaluator


class TaskRunner:
    """Run an agent through a suite of evaluation tasks."""

    def __init__(self, evaluator: FitnessEvaluator | None = None):
        self.evaluator = evaluator or CodeQualityEvaluator()

    async def evaluate_agent(self, agent: AgentDNA) -> FitnessScore:
        """Evaluate a single agent and record its fitness score."""
        score = await self.evaluator.evaluate(agent)
        agent.fitness_scores.append(score)
        return score

    async def evaluate_population(self, agents: list[AgentDNA]) -> list[FitnessScore]:
        """Evaluate all agents in a population."""
        scores = []
        for agent in agents:
            score = await self.evaluate_agent(agent)
            scores.append(score)
        return scores

    @staticmethod
    def extract_agent_code(agent: AgentDNA) -> str:
        """Extract all function code from an agent's genome."""
        code_parts = []
        for chrom in agent.genome.chromosomes:
            for gene in chrom.genes:
                if gene.kind == GeneKind.FUNCTION:
                    code_parts.append(f"# Skill: {gene.name}\n{gene.source}")
        return "\n\n".join(code_parts)

    @staticmethod
    def extract_system_prompt(agent: AgentDNA) -> str:
        """Extract the system prompt from an agent's genome."""
        prompt_parts = []
        for chrom in agent.genome.get_system_prompt_chromosomes():
            for gene in chrom.genes:
                if gene.kind == GeneKind.PROMPT_SECTION:
                    prompt_parts.append(gene.source)
        return "\n\n".join(prompt_parts)
