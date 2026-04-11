"""Tests for fitness evaluators."""

import pytest

from inception.fitness.evaluator import CodeQualityEvaluator
from inception.genome.schema import FitnessScore


class TestCodeQualityEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_coder_agent(self, coder_agent):
        evaluator = CodeQualityEvaluator()
        score = await evaluator.evaluate(coder_agent)

        assert isinstance(score, FitnessScore)
        assert 0.0 <= score.overall <= 1.0
        assert score.benchmark_id == "code_quality"

    @pytest.mark.asyncio
    async def test_evaluate_researcher_agent(self, researcher_agent):
        evaluator = CodeQualityEvaluator()
        score = await evaluator.evaluate(researcher_agent)

        assert isinstance(score, FitnessScore)
        assert 0.0 <= score.overall <= 1.0

    @pytest.mark.asyncio
    async def test_dimensions_present(self, coder_agent):
        evaluator = CodeQualityEvaluator()
        score = await evaluator.evaluate(coder_agent)

        assert "syntax_validity" in score.dimensions
        assert score.dimensions["syntax_validity"] == 1.0  # Our test code is valid

    @pytest.mark.asyncio
    async def test_seed_agent_quality(self):
        """Test that loaded seed agents get reasonable fitness scores."""
        from pathlib import Path
        from inception.genome.serialization import load_seed_agent_from_directory

        seed_dir = Path(__file__).parent.parent.parent / "examples" / "seed_agents" / "coder_agent"
        if not seed_dir.exists():
            return

        agent = load_seed_agent_from_directory(seed_dir)
        evaluator = CodeQualityEvaluator()
        score = await evaluator.evaluate(agent)

        assert score.overall > 0.3  # Seed agents should have decent quality
