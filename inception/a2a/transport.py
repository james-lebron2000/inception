"""Remote mating orchestration spanning client and server.

Provides high-level workflows for discovering agents, sending
mate requests, and producing offspring through remote crossover.
"""

from __future__ import annotations

import copy

from inception.a2a.client import A2AClient
from inception.a2a.models import (
    GeneticAgentCard,
    GenomeExport,
    MateRequest,
    MatingPolicy,
)
from inception.genome.export import build_agent_card, export_genome, reconstruct_mating_genome
from inception.genome.schema import AgentDNA


class RemoteMatingOrchestrator:
    """Orchestrates the full remote mating workflow."""

    def __init__(
        self,
        local_agent: AgentDNA,
        policy: MatingPolicy,
        llm_api_key: str = "",
    ):
        self.local_agent = local_agent
        self.policy = policy
        self.llm_api_key = llm_api_key
        self.client = A2AClient()

    async def mate_with_remote(
        self,
        target_url: str,
        task: str = "",
        crossover_strategy: str = "adaptive",
    ) -> AgentDNA | None:
        """Full remote mating workflow with a specific agent.

        1. Discover the target agent
        2. Check compatibility
        3. Send mate request with our genome
        4. Handle the response — use server-provided offspring or do local crossover
        """
        # Discover target
        target_card = await self.client.discover(target_url)

        # Check basic compatibility
        if not target_card.mating_policy.allow_remote_mating:
            return None

        # Build our card and genome export
        local_card = build_agent_card(self.local_agent, "", self.policy)
        local_export = export_genome(self.local_agent.genome, self.policy)

        # Send mate request
        request = MateRequest(
            requester_card=local_card,
            task_description=task,
            preferred_crossover=crossover_strategy,
            requester_genome_export=local_export,
        )

        response = await self.client.request_mate(target_url, request)

        if not response.accepted:
            return None

        # If server provided an offspring, use it
        if response.offspring_agent:
            return response.offspring_agent

        # Otherwise, try local crossover with the responder's genome
        if response.responder_genome_export:
            return await self._local_crossover(
                response.responder_genome_export, target_card, crossover_strategy
            )

        return None

    async def discover_and_mate(
        self,
        registry_url: str,
        task: str,
        top_k: int = 3,
        crossover_strategy: str = "adaptive",
    ) -> AgentDNA | None:
        """Find the best match via registry, then mate.

        Tries each candidate in order until one succeeds.
        """
        import httpx

        local_card = build_agent_card(self.local_agent, "", self.policy)

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                f"{registry_url.rstrip('/')}/api/discover",
                json={
                    "task": task,
                    "requester_card": local_card.model_dump(),
                    "top_k": top_k,
                },
            )
            resp.raise_for_status()
            matches = resp.json()

        for match in matches:
            target_url = match["agent_card"]["url"]
            offspring = await self.mate_with_remote(
                target_url, task, crossover_strategy
            )
            if offspring:
                return offspring

        return None

    async def _local_crossover(
        self,
        remote_export: GenomeExport,
        remote_card: GeneticAgentCard,
        strategy: str,
    ) -> AgentDNA | None:
        """Perform crossover locally using the remote genome export."""
        remote_genome = reconstruct_mating_genome(remote_export)
        if remote_genome is None:
            return None

        from inception.genome.compatibility import compatibility_score
        from inception.llm.client import LLMClient, LLMConfig
        from inception.reproduction.crossover import (
            AdaptiveCrossover,
            ChromosomeLevelCrossover,
        )

        compat = compatibility_score(self.local_agent.genome, remote_genome)
        if compat < 0.1:
            return None

        local_genome = copy.deepcopy(self.local_agent.genome)

        if self.llm_api_key and strategy != "chromosome":
            llm = LLMClient(LLMConfig(api_key=self.llm_api_key))
            crossover = AdaptiveCrossover()
        else:
            llm = LLMClient()
            crossover = ChromosomeLevelCrossover()

        child_genome = await crossover.crossover(local_genome, remote_genome, llm)

        child = AgentDNA(
            name=f"{self.local_agent.name[:4]}x{remote_card.name[:4]}",
            genome=child_genome,
            generation=max(self.local_agent.generation, remote_card.generation) + 1,
            parents=(self.local_agent.id, remote_export.genome_id),
        )
        return child
