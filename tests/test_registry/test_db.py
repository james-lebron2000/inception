"""Tests for the Gene Registry database."""

from __future__ import annotations

import pytest

from inception.a2a.models import GeneticAgentCard, MatingPolicy
from inception.registry.db import RegistryDB


@pytest.fixture
async def db():
    """Create an in-memory database for testing."""
    registry_db = RegistryDB(db_path=":memory:")
    await registry_db.initialize()
    yield registry_db
    await registry_db.close()


@pytest.fixture
def sample_card():
    return GeneticAgentCard(
        name="TestAgent",
        url="http://localhost:3000",
        agent_id="test-agent-1",
        generation=3,
        gene_count=12,
        capabilities={"coding": 0.8, "math": 0.6},
        skill_names=["coding", "math"],
        best_fitness=0.85,
        mating_policy=MatingPolicy(allow_remote_mating=True),
    )


@pytest.fixture
def second_card():
    return GeneticAgentCard(
        name="ResearchBot",
        url="http://localhost:3001",
        agent_id="test-agent-2",
        generation=1,
        gene_count=8,
        capabilities={"research": 0.9, "summarize": 0.7},
        skill_names=["research", "summarize"],
        best_fitness=0.72,
        mating_policy=MatingPolicy(allow_remote_mating=True),
    )


class TestRegistryDB:
    @pytest.mark.asyncio
    async def test_register_and_retrieve(self, db, sample_card):
        await db.register_agent(sample_card)
        entry = await db.get_agent("test-agent-1")

        assert entry is not None
        assert entry.agent_card.name == "TestAgent"
        assert entry.agent_card.best_fitness == 0.85
        assert entry.mating_count == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, db):
        entry = await db.get_agent("nonexistent")
        assert entry is None

    @pytest.mark.asyncio
    async def test_deactivate_agent(self, db, sample_card):
        await db.register_agent(sample_card)
        await db.deactivate_agent("test-agent-1")
        entry = await db.get_agent("test-agent-1")
        assert entry is None

    @pytest.mark.asyncio
    async def test_search_by_capability(self, db, sample_card, second_card):
        await db.register_agent(sample_card)
        await db.register_agent(second_card)

        results = await db.search_by_capability("coding")
        assert len(results) == 1
        assert results[0].agent_card.name == "TestAgent"

    @pytest.mark.asyncio
    async def test_search_by_capability_partial_match(self, db, sample_card):
        await db.register_agent(sample_card)
        results = await db.search_by_capability("cod")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_all_active(self, db, sample_card, second_card):
        await db.register_agent(sample_card)
        await db.register_agent(second_card)
        all_agents = await db.get_all_active()
        assert len(all_agents) == 2

    @pytest.mark.asyncio
    async def test_log_mating(self, db, sample_card, second_card):
        await db.register_agent(sample_card)
        await db.register_agent(second_card)
        await db.log_mating(
            requester_id="test-agent-1",
            responder_id="test-agent-2",
            accepted=True,
            offspring_id="offspring-1",
            task_description="code analysis",
        )

        entry = await db.get_agent("test-agent-1")
        assert entry is not None
        assert entry.mating_count == 1
        assert entry.offspring_count == 1

    @pytest.mark.asyncio
    async def test_leaderboard_ordered_by_fitness(self, db, sample_card, second_card):
        await db.register_agent(sample_card)
        await db.register_agent(second_card)
        leaders = await db.get_leaderboard(limit=10)
        assert len(leaders) == 2
        assert leaders[0].agent_card.best_fitness >= leaders[1].agent_card.best_fitness

    @pytest.mark.asyncio
    async def test_update_agent(self, db, sample_card):
        await db.register_agent(sample_card)
        sample_card.best_fitness = 0.95
        await db.update_agent(sample_card)
        entry = await db.get_agent("test-agent-1")
        assert entry is not None
        assert entry.agent_card.best_fitness == 0.95
