"""LLM-driven mutation engine — introducing genetic novelty.

Mutations introduce variation, preventing the population from converging
on a single solution. Unlike traditional genetic algorithms that perform
random bit flips, this engine uses an LLM to make semantically meaningful
code modifications.

Biological analog: DNA mutation during replication.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import uuid
from dataclasses import dataclass, field

from inception.genome.schema import (
    Chromosome,
    ChromosomeKind,
    Gene,
    GeneKind,
    Genome,
    MutationRecord,
    MutationType,
    SkillInterface,
)
from inception.llm.client import LLMClient
from inception.llm.prompts import (
    INSERTION_MUTATION,
    POINT_MUTATION,
    REGULATORY_MUTATION,
    SKILL_SYNTHESIS,
)


@dataclass
class MutationConfig:
    """Configuration for the mutation engine."""

    mutation_rate: float = 0.1  # Per-gene probability
    mutation_type_weights: dict[str, float] = field(default_factory=lambda: {
        MutationType.POINT_MUTATION.value: 0.40,
        MutationType.INSERTION.value: 0.15,
        MutationType.DELETION.value: 0.10,
        MutationType.DUPLICATION.value: 0.05,
        MutationType.REGULATORY.value: 0.15,
        MutationType.SKILL_SYNTHESIS.value: 0.15,
    })
    max_mutations_per_genome: int = 3


def _validate_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _select_mutation_type(weights: dict[str, float]) -> MutationType:
    """Select a mutation type by weighted random choice."""
    types = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    chosen = random.choices(types, weights=probs, k=1)[0]
    return MutationType(chosen)


class MutationEngine:
    """Applies LLM-driven mutations to agent genomes."""

    def __init__(self, llm: LLMClient, config: MutationConfig | None = None):
        self.llm = llm
        self.config = config or MutationConfig()

    async def mutate(self, genome: Genome) -> tuple[Genome, list[MutationRecord]]:
        """Apply mutations to a genome.

        For each gene, roll against mutation_rate. If mutating, select
        a mutation type by weighted random and use the LLM to perform it.

        Returns:
            (mutated_genome, list_of_mutation_records)
        """
        mutated = genome.model_copy(deep=True)
        mutated.id = str(uuid.uuid4())
        mutated.version = genome.version + 1
        records: list[MutationRecord] = []

        mutation_count = 0

        # Gene-level mutations
        for chrom in mutated.chromosomes:
            for gene in chrom.genes:
                if mutation_count >= self.config.max_mutations_per_genome:
                    break
                if random.random() < self.config.mutation_rate:
                    mutation_type = _select_mutation_type(self.config.mutation_type_weights)
                    record = await self._apply_mutation(mutation_type, gene, chrom, mutated)
                    if record:
                        records.append(record)
                        mutation_count += 1

        # Genome-level mutations (skill synthesis — requires LLM)
        if mutation_count < self.config.max_mutations_per_genome and self._has_api_key():
            if random.random() < self.config.mutation_rate:
                record = await self._apply_skill_synthesis(mutated)
                if record:
                    records.append(record)

        mutated.lineage.mutations_applied.extend(records)
        return mutated, records

    def _has_api_key(self) -> bool:
        """Check if an LLM API key is configured."""
        return bool(self.llm.config.api_key)

    async def _apply_mutation(
        self,
        mutation_type: MutationType,
        gene: Gene,
        chromosome: Chromosome,
        genome: Genome,
    ) -> MutationRecord | None:
        """Apply a specific mutation type to a gene."""
        # LLM-requiring mutations fall back to non-LLM if no API key
        llm_types = {
            MutationType.POINT_MUTATION,
            MutationType.INSERTION,
            MutationType.REGULATORY,
        }
        if mutation_type in llm_types and not self._has_api_key():
            mutation_type = MutationType.DUPLICATION

        if mutation_type == MutationType.POINT_MUTATION:
            return await self._point_mutation(gene)
        elif mutation_type == MutationType.INSERTION:
            return await self._insertion_mutation(gene, chromosome)
        elif mutation_type == MutationType.DELETION:
            return self._deletion_mutation(gene, chromosome)
        elif mutation_type == MutationType.DUPLICATION:
            return self._duplication_mutation(gene, chromosome)
        elif mutation_type == MutationType.REGULATORY:
            return await self._regulatory_mutation(gene)
        return None

    async def _point_mutation(self, gene: Gene) -> MutationRecord | None:
        """Small, targeted improvement to a function gene."""
        if gene.kind != GeneKind.FUNCTION:
            return None

        prompt = POINT_MUTATION.render(current_code=gene.source)
        mutated_code = await self.llm.complete_code(prompt, temperature=0.7)

        if _validate_python(mutated_code) and mutated_code != gene.source:
            old_source = gene.source
            gene.source = mutated_code
            gene.metadata["mutated"] = True
            return MutationRecord(
                mutation_type=MutationType.POINT_MUTATION,
                target_gene_id=gene.id,
                description="Point mutation: small targeted improvement",
                diff=f"--- old\n+++ new\n-{old_source[:100]}...\n+{mutated_code[:100]}...",
            )
        return None

    async def _insertion_mutation(
        self, gene: Gene, chromosome: Chromosome
    ) -> MutationRecord | None:
        """Add a new helper function to a skill module."""
        if gene.kind != GeneKind.FUNCTION:
            return None

        capability = chromosome.interface.capability if chromosome.interface else chromosome.name
        prompt = INSERTION_MUTATION.render(
            current_code=gene.source,
            capability=capability,
        )
        expanded_code = await self.llm.complete_code(prompt, temperature=0.8)

        if _validate_python(expanded_code):
            gene.source = expanded_code
            gene.metadata["insertion_mutation"] = True
            return MutationRecord(
                mutation_type=MutationType.INSERTION,
                target_gene_id=gene.id,
                description=f"Insertion: added helper function to {capability}",
            )
        return None

    def _deletion_mutation(
        self, gene: Gene, chromosome: Chromosome
    ) -> MutationRecord | None:
        """Remove a gene from a chromosome (simplification)."""
        if len(chromosome.genes) <= 1:
            return None  # Don't delete the last gene

        chromosome.genes.remove(gene)
        return MutationRecord(
            mutation_type=MutationType.DELETION,
            target_gene_id=gene.id,
            description=f"Deletion: removed gene '{gene.name}' for simplification",
        )

    def _duplication_mutation(
        self, gene: Gene, chromosome: Chromosome
    ) -> MutationRecord | None:
        """Duplicate a gene — the copy can later diverge via point mutations."""
        new_gene = gene.model_copy(deep=True)
        new_gene.id = str(uuid.uuid4())[:8]
        new_gene.name = f"{gene.name}_dup"
        new_gene.metadata["duplicated_from"] = gene.id

        chromosome.genes.append(new_gene)
        return MutationRecord(
            mutation_type=MutationType.DUPLICATION,
            target_gene_id=gene.id,
            description=f"Duplication: cloned gene '{gene.name}' as '{new_gene.name}'",
        )

    async def _regulatory_mutation(self, gene: Gene) -> MutationRecord | None:
        """Modify a prompt section or config block."""
        if gene.kind == GeneKind.PROMPT_SECTION:
            prompt = REGULATORY_MUTATION.render(current_prompt=gene.source)
            mutated_text = (await self.llm.complete(prompt, temperature=0.7)).text
            if mutated_text and mutated_text != gene.source:
                gene.source = mutated_text
                return MutationRecord(
                    mutation_type=MutationType.REGULATORY,
                    target_gene_id=gene.id,
                    description="Regulatory mutation: modified system prompt section",
                )
        return None

    async def _apply_skill_synthesis(self, genome: Genome) -> MutationRecord | None:
        """Generate an entirely new skill chromosome via LLM."""
        capabilities = list(genome.capability_vector().keys())

        # Summarize the system prompt
        prompt_chroms = genome.get_system_prompt_chromosomes()
        prompt_summary = ""
        if prompt_chroms:
            prompt_summary = " ".join(
                g.source[:200] for c in prompt_chroms for g in c.genes
            )[:500]

        prompt = SKILL_SYNTHESIS.render(
            capabilities=capabilities or ["general"],
            prompt_summary=prompt_summary or "A general-purpose AI agent",
        )

        response = await self.llm.complete(prompt, temperature=0.9)

        try:
            data = json.loads(response.text.strip().strip("```json").strip("```"))
        except json.JSONDecodeError:
            return None

        skill_name = data.get("skill_name", "new_skill")
        source = data.get("source", "")
        capability = data.get("capability", skill_name)
        tags = data.get("tags", [])

        if not source or not _validate_python(source):
            return None

        new_chrom = Chromosome(
            kind=ChromosomeKind.SKILL_MODULE,
            name=skill_name,
            genes=[Gene(
                kind=GeneKind.FUNCTION,
                name=skill_name,
                source=source,
                metadata={"mutation": "skill_synthesis"},
            )],
            interface=SkillInterface(
                capability=capability,
                tags=tags,
            ),
        )

        genome.chromosomes.append(new_chrom)
        return MutationRecord(
            mutation_type=MutationType.SKILL_SYNTHESIS,
            target_gene_id=new_chrom.id,
            description=f"Skill synthesis: generated new skill '{skill_name}' ({capability})",
        )
