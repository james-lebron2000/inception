"""Task-based agent compatibility matching for the Gene Registry.

Finds the best partner agent for a given task by scoring candidates
on capability coverage, complementarity, fitness, and availability.
"""

from __future__ import annotations

from dataclasses import dataclass

from inception.a2a.models import GeneticAgentCard, RegistryEntry
from inception.registry.db import RegistryDB


@dataclass
class MatchResult:
    """Result of matching a candidate agent against a task."""

    entry: RegistryEntry
    score: float
    breakdown: dict[str, float]


class AgentMatcher:
    """Finds the best mate candidates for a given task and requester."""

    def __init__(self, db: RegistryDB):
        self.db = db

    async def find_matches(
        self,
        task: str,
        requester_card: GeneticAgentCard,
        top_k: int = 5,
    ) -> list[MatchResult]:
        """Find the top-k best agent matches for a task.

        Scores each candidate on:
        - Capability coverage (40%): match against task keywords
        - Complementarity (30%): prefer diverse skills the requester lacks
        - Fitness (20%): higher fitness is better
        - Availability (10%): respects mating policy
        """
        all_agents = await self.db.get_all_active(limit=200)

        # Don't match with self
        candidates = [
            entry for entry in all_agents
            if entry.agent_card.agent_id != requester_card.agent_id
        ]

        task_keywords = _extract_keywords(task)
        requester_caps = set(requester_card.capabilities.keys())

        results = []
        for entry in candidates:
            score, breakdown = self._score_candidate(
                entry, task_keywords, requester_caps
            )
            results.append(MatchResult(entry=entry, score=score, breakdown=breakdown))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _score_candidate(
        self,
        entry: RegistryEntry,
        task_keywords: set[str],
        requester_caps: set[str],
    ) -> tuple[float, dict[str, float]]:
        card = entry.agent_card

        # Capability coverage: how well the candidate's skills match the task
        candidate_caps = set(card.capabilities.keys())
        candidate_skills = set(card.skill_names)
        all_candidate_terms = candidate_caps | candidate_skills | set(card.species_tags)

        coverage = _keyword_overlap(task_keywords, all_candidate_terms)

        # Complementarity: prefer agents with skills the requester doesn't have
        if requester_caps or candidate_caps:
            unique_to_candidate = candidate_caps - requester_caps
            total = len(requester_caps | candidate_caps) or 1
            complementarity = len(unique_to_candidate) / total
        else:
            complementarity = 0.5

        # Fitness bonus
        fitness = min(card.best_fitness, 1.0)

        # Availability: does mating policy allow remote mating?
        availability = 1.0 if card.mating_policy.allow_remote_mating else 0.0

        score = (
            0.4 * coverage
            + 0.3 * complementarity
            + 0.2 * fitness
            + 0.1 * availability
        )

        breakdown = {
            "coverage": coverage,
            "complementarity": complementarity,
            "fitness": fitness,
            "availability": availability,
        }
        return score, breakdown


def _extract_keywords(task: str) -> set[str]:
    """Extract meaningful keywords from a task description."""
    stop_words = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "that",
        "this", "it", "i", "me", "my", "we", "our",
    }
    words = task.lower().replace(",", " ").replace(".", " ").split()
    return {w.strip() for w in words if w.strip() and w not in stop_words}


def _keyword_overlap(keywords: set[str], terms: set[str]) -> float:
    """Compute overlap between task keywords and candidate terms."""
    if not keywords:
        return 0.5

    lower_terms = {t.lower().replace("_", " ").replace("-", " ") for t in terms}
    expanded_terms: set[str] = set()
    for t in lower_terms:
        expanded_terms.update(t.split())
        expanded_terms.add(t)

    matches = sum(1 for kw in keywords if kw in expanded_terms)
    return matches / len(keywords)
