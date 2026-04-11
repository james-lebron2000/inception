"""Fitness evaluation framework — measuring agent phenotype quality.

Evaluators measure how well an agent performs, driving natural selection.
Multiple evaluators can be composed with weights.

Biological analog: environmental fitness — survival and reproductive success.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from inception.genome.schema import AgentDNA, FitnessScore
from inception.llm.client import LLMClient
from inception.llm.prompts import FITNESS_JUDGE


class Task(ABC):
    """A task that an agent can be evaluated on."""

    id: str
    description: str
    difficulty: float  # 0.0 to 1.0
    required_capabilities: list[str]
    evaluation_criteria: str

    @abstractmethod
    async def run(self, agent_code: str) -> dict[str, Any]:
        """Execute the task with the given agent code and return results."""
        ...


class CodingTask(Task):
    """A coding task for evaluating code generation skills."""

    def __init__(
        self,
        task_id: str,
        description: str,
        test_cases: list[dict[str, Any]],
        difficulty: float = 0.5,
    ):
        self.id = task_id
        self.description = description
        self.test_cases = test_cases
        self.difficulty = difficulty
        self.required_capabilities = ["code_generation"]
        self.evaluation_criteria = "Code must pass all test cases"

    async def run(self, agent_code: str) -> dict[str, Any]:
        """Execute agent code against test cases in a sandboxed environment."""
        from inception.agent.sandbox import execute_code_safely

        results = []
        for tc in self.test_cases:
            try:
                output = await execute_code_safely(
                    code=agent_code,
                    function_name=tc.get("function", "solve"),
                    args=tc.get("input", []),
                    timeout=tc.get("timeout", 10),
                )
                passed = output == tc.get("expected")
                results.append({"passed": passed, "output": output, "expected": tc.get("expected")})
            except Exception as e:
                results.append({"passed": False, "error": str(e)})

        passed_count = sum(1 for r in results if r.get("passed"))
        return {
            "passed": passed_count,
            "total": len(self.test_cases),
            "accuracy": passed_count / max(len(self.test_cases), 1),
            "details": results,
        }


class FitnessEvaluator(ABC):
    """Abstract fitness evaluator."""

    @abstractmethod
    async def evaluate(self, agent: AgentDNA) -> FitnessScore:
        """Evaluate an agent's fitness."""
        ...


class CodeQualityEvaluator(FitnessEvaluator):
    """Evaluate agent fitness based on code quality metrics.

    Checks: syntax validity, complexity, function count, documentation.
    """

    async def evaluate(self, agent: AgentDNA) -> FitnessScore:
        import ast

        total_score = 0.0
        dimensions: dict[str, float] = {}
        gene_count = 0
        valid_count = 0
        documented_count = 0
        total_complexity = 0

        for chrom in agent.genome.chromosomes:
            for gene in chrom.genes:
                if gene.kind.value == "function":
                    gene_count += 1
                    try:
                        tree = ast.parse(gene.source)
                        valid_count += 1

                        # Check for docstrings
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if (
                                    node.body
                                    and isinstance(node.body[0], ast.Expr)
                                    and isinstance(node.body[0].value, ast.Constant)
                                    and isinstance(node.body[0].value.value, str)
                                ):
                                    documented_count += 1

                        # Simple complexity: count branches
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                                total_complexity += 1
                    except SyntaxError:
                        pass

        if gene_count > 0:
            dimensions["syntax_validity"] = valid_count / gene_count
            dimensions["documentation"] = min(1.0, documented_count / max(valid_count, 1))
            # Lower complexity is better (up to a point)
            avg_complexity = total_complexity / max(valid_count, 1)
            dimensions["simplicity"] = max(0.0, 1.0 - avg_complexity / 20.0)
            dimensions["capability_breadth"] = min(1.0, gene_count / 5.0)

            # Overall is average of dimensions, capped at 1.0
            total_score = min(1.0, sum(dimensions.values()) / len(dimensions))

        return FitnessScore(
            benchmark_id="code_quality",
            overall=total_score,
            dimensions=dimensions,
        )


class LLMJudgeEvaluator(FitnessEvaluator):
    """Use an LLM to judge agent output quality."""

    def __init__(self, llm: LLMClient, tasks: list[dict[str, str]]):
        self.llm = llm
        self.tasks = tasks  # Each has 'description', 'evaluation_criteria', 'agent_output'

    async def evaluate(self, agent: AgentDNA) -> FitnessScore:
        all_dimensions: dict[str, list[float]] = {}
        overall_scores: list[float] = []

        for task in self.tasks:
            prompt = FITNESS_JUDGE.render(
                task_description=task["description"],
                evaluation_criteria=task["evaluation_criteria"],
                agent_output=task.get("agent_output", "(no output)"),
            )

            response = await self.llm.complete(prompt, temperature=0.3)
            try:
                data = json.loads(response.text.strip().strip("```json").strip("```"))
                for dim, score in data.items():
                    if isinstance(score, (int, float)) and dim != "overall":
                        all_dimensions.setdefault(dim, []).append(float(score))
                overall_scores.append(float(data.get("overall", 0.5)))
            except (json.JSONDecodeError, ValueError):
                overall_scores.append(0.5)

        avg_dimensions = {k: sum(v) / len(v) for k, v in all_dimensions.items()}
        avg_overall = sum(overall_scores) / max(len(overall_scores), 1)

        return FitnessScore(
            benchmark_id="llm_judge",
            overall=avg_overall,
            dimensions=avg_dimensions,
        )


class CompositeEvaluator(FitnessEvaluator):
    """Combine multiple evaluators with weights.

    Like biological fitness being determined by multiple environmental factors.
    """

    def __init__(self, evaluators: list[tuple[float, FitnessEvaluator]]):
        self.evaluators = evaluators  # (weight, evaluator) pairs

    async def evaluate(self, agent: AgentDNA) -> FitnessScore:
        total_weight = sum(w for w, _ in self.evaluators)
        weighted_overall = 0.0
        all_dimensions: dict[str, float] = {}
        all_details: dict[str, Any] = {}

        for weight, evaluator in self.evaluators:
            score = await evaluator.evaluate(agent)
            normalized_weight = weight / total_weight
            weighted_overall += score.overall * normalized_weight

            for dim, val in score.dimensions.items():
                key = f"{score.benchmark_id}.{dim}"
                all_dimensions[key] = val

            all_details[score.benchmark_id] = {
                "overall": score.overall,
                "dimensions": score.dimensions,
            }

        return FitnessScore(
            benchmark_id="composite",
            overall=weighted_overall,
            dimensions=all_dimensions,
            details=all_details,
        )
