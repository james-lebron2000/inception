"""Tests for the task-based agent matcher."""

from __future__ import annotations

import pytest

from inception.a2a.models import GeneticAgentCard, MatingPolicy
from inception.registry.db import RegistryDB
from inception.registry.matcher import AgentMatcher, _extract_keywords, _keyword_overlap


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self):
        keywords = _extract_keywords("scrape websites and generate SQL reports")
        assert "scrape" in keywords
        assert "sql" in keywords
        assert "reports" in keywords
        assert "and" not in keywords

    def test_empty_string(self):
        assert _extract_keywords("") == set()

    def test_removes_stop_words(self):
        keywords = _extract_keywords("the quick brown fox in a box")
        assert "the" not in keywords
        assert "in" not in keywords
        assert "a" not in keywords
        assert "quick" in keywords


class TestKeywordOverlap:
    def test_full_overlap(self):
        score = _keyword_overlap({"coding", "math"}, {"coding", "math"})
        assert score == 1.0

    def test_no_overlap(self):
        score = _keyword_overlap({"coding"}, {"research"})
        assert score == 0.0

    def test_partial_overlap(self):
        score = _keyword_overlap({"coding", "math"}, {"coding", "research"})
        assert score == 0.5

    def test_empty_keywords(self):
        score = _keyword_overlap(set(), {"coding"})
        assert score == 0.5


@pytest.fixture
async def populated_db():
    db = RegistryDB(db_path=":memory:")
    await db.initialize()

    cards = [
        GeneticAgentCard(
            name="Coder",
            url="http://coder:3000",
            agent_id="coder-1",
            capabilities={"coding": 0.9, "debugging": 0.7},
            skill_names=["coding", "debugging"],
            best_fitness=0.85,
            mating_policy=MatingPolicy(allow_remote_mating=True),
        ),
        GeneticAgentCard(
            name="Researcher",
            url="http://researcher:3000",
            agent_id="researcher-1",
            capabilities={"research": 0.9, "summarize": 0.8},
            skill_names=["research", "summarize"],
            best_fitness=0.72,
            mating_policy=MatingPolicy(allow_remote_mating=True),
        ),
        GeneticAgentCard(
            name="Analyst",
            url="http://analyst:3000",
            agent_id="analyst-1",
            capabilities={"data_analysis": 0.8, "coding": 0.6},
            skill_names=["data_analysis", "coding"],
            best_fitness=0.78,
            mating_policy=MatingPolicy(allow_remote_mating=False),
        ),
    ]
    for card in cards:
        await db.register_agent(card)

    yield db
    await db.close()


class TestAgentMatcher:
    @pytest.mark.asyncio
    async def test_find_matches_for_coding_task(self, populated_db):
        matcher = AgentMatcher(populated_db)
        requester = GeneticAgentCard(
            name="Me", url="http://me", agent_id="me-1",
            capabilities={"research": 0.5},
        )
        matches = await matcher.find_matches("write Python code", requester, top_k=3)

        assert len(matches) > 0
        # Coder should score highest for a coding task
        names = [m.entry.agent_card.name for m in matches]
        assert "Coder" in names

    @pytest.mark.asyncio
    async def test_excludes_self(self, populated_db):
        matcher = AgentMatcher(populated_db)
        requester = GeneticAgentCard(
            name="Coder", url="http://coder:3000", agent_id="coder-1",
        )
        matches = await matcher.find_matches("coding", requester, top_k=5)
        agent_ids = [m.entry.agent_card.agent_id for m in matches]
        assert "coder-1" not in agent_ids

    @pytest.mark.asyncio
    async def test_match_result_has_breakdown(self, populated_db):
        matcher = AgentMatcher(populated_db)
        requester = GeneticAgentCard(
            name="Me", url="http://me", agent_id="me-1",
        )
        matches = await matcher.find_matches("research and summarize", requester, top_k=1)
        assert len(matches) == 1
        assert "coverage" in matches[0].breakdown
        assert "complementarity" in matches[0].breakdown
        assert "fitness" in matches[0].breakdown
        assert "availability" in matches[0].breakdown

    @pytest.mark.asyncio
    async def test_respects_top_k(self, populated_db):
        matcher = AgentMatcher(populated_db)
        requester = GeneticAgentCard(
            name="Me", url="http://me", agent_id="me-1",
        )
        matches = await matcher.find_matches("anything", requester, top_k=1)
        assert len(matches) <= 1
