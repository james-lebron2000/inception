"""LLM-driven crossover engine — the core of agent 'mating'.

Three crossover strategies mirroring biological mechanisms:
1. ChromosomeLevelCrossover: swap entire skill modules (like meiotic chromosome exchange)
2. GeneLevelCrossover: LLM merges individual functions from both parents
3. SemanticCrossover: LLM reads both parents holistically and creates a novel blend

Biological analog: sexual recombination producing offspring with traits from both parents.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import uuid
from abc import ABC, abstractmethod

from inception.genome.compatibility import find_homologous_pairs
from inception.genome.schema import (
    Chromosome,
    ChromosomeKind,
    Gene,
    GeneKind,
    Genome,
    Lineage,
    SkillInterface,
)
from inception.llm.client import LLMClient
from inception.llm.prompts import GENE_LEVEL_CROSSOVER, PROMPT_CROSSOVER, SEMANTIC_CROSSOVER


def _validate_python(source: str) -> bool:
    """Check if source code is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _deep_copy_chromosome(chrom: Chromosome) -> Chromosome:
    """Create a deep copy of a chromosome with a new ID."""
    new_chrom = chrom.model_copy(deep=True)
    new_chrom.id = str(uuid.uuid4())[:8]
    return new_chrom


class CrossoverStrategy(ABC):
    """Abstract base for crossover strategies."""

    @abstractmethod
    async def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        llm: LLMClient,
    ) -> Genome:
        """Produce an offspring genome from two parents."""
        ...


class ChromosomeLevelCrossover(CrossoverStrategy):
    """Swap entire chromosomes (skills) between parents.

    Analogous to: swapping entire chromosomes during meiosis.
    Fast and cheap (no LLM calls), but coarse-grained.
    """

    def __init__(self, unpaired_inheritance_prob: float = 0.5):
        self.unpaired_inheritance_prob = unpaired_inheritance_prob

    async def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        llm: LLMClient,
    ) -> Genome:
        paired, unpaired_a, unpaired_b = find_homologous_pairs(parent_a, parent_b)

        child_chromosomes: list[Chromosome] = []
        crossover_points: list[str] = []

        # For each homologous pair, randomly pick one parent's version
        for ca, cb in paired:
            if random.random() < 0.5:
                child_chromosomes.append(_deep_copy_chromosome(ca))
            else:
                child_chromosomes.append(_deep_copy_chromosome(cb))
                crossover_points.append(ca.name)

        # Unpaired chromosomes: inherit with probability
        for chrom in unpaired_a:
            if random.random() < self.unpaired_inheritance_prob:
                child_chromosomes.append(_deep_copy_chromosome(chrom))

        for chrom in unpaired_b:
            if random.random() < self.unpaired_inheritance_prob:
                child_chromosomes.append(_deep_copy_chromosome(chrom))

        return Genome(
            id=str(uuid.uuid4()),
            version=1,
            lineage=Lineage(
                parent_ids=[parent_a.id, parent_b.id],
                generation=max(parent_a.lineage.generation, parent_b.lineage.generation) + 1,
                crossover_points=crossover_points,
            ),
            chromosomes=child_chromosomes,
        )


class GeneLevelCrossover(CrossoverStrategy):
    """LLM merges individual genes (functions) from homologous chromosomes.

    Analogous to: intrachromosomal crossover (crossing over within a chromosome).
    Uses the LLM to semantically merge two implementations of the same skill.
    """

    def __init__(self, merge_rate: float = 0.3, unpaired_inheritance_prob: float = 0.5):
        self.merge_rate = merge_rate
        self.unpaired_inheritance_prob = unpaired_inheritance_prob

    async def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        llm: LLMClient,
    ) -> Genome:
        paired, unpaired_a, unpaired_b = find_homologous_pairs(parent_a, parent_b)

        child_chromosomes: list[Chromosome] = []
        crossover_points: list[str] = []

        for ca, cb in paired:
            if random.random() < self.merge_rate and ca.kind == ChromosomeKind.SKILL_MODULE:
                # LLM-driven merge of skill code
                merged = await self._merge_skill_chromosomes(ca, cb, llm)
                child_chromosomes.append(merged)
                crossover_points.append(f"merged:{ca.name}+{cb.name}")
            elif random.random() < self.merge_rate and ca.kind == ChromosomeKind.SYSTEM_PROMPT:
                # LLM-driven merge of system prompts
                merged = await self._merge_prompt_chromosomes(ca, cb, llm)
                child_chromosomes.append(merged)
                crossover_points.append(f"merged_prompt:{ca.name}+{cb.name}")
            else:
                # Simple inheritance
                chosen = ca if random.random() < 0.5 else cb
                child_chromosomes.append(_deep_copy_chromosome(chosen))

        for chrom in unpaired_a:
            if random.random() < self.unpaired_inheritance_prob:
                child_chromosomes.append(_deep_copy_chromosome(chrom))

        for chrom in unpaired_b:
            if random.random() < self.unpaired_inheritance_prob:
                child_chromosomes.append(_deep_copy_chromosome(chrom))

        return Genome(
            id=str(uuid.uuid4()),
            version=1,
            lineage=Lineage(
                parent_ids=[parent_a.id, parent_b.id],
                generation=max(parent_a.lineage.generation, parent_b.lineage.generation) + 1,
                crossover_points=crossover_points,
            ),
            chromosomes=child_chromosomes,
        )

    async def _merge_skill_chromosomes(
        self, ca: Chromosome, cb: Chromosome, llm: LLMClient
    ) -> Chromosome:
        """Use LLM to merge two skill chromosomes at the gene level."""
        source_a = "\n\n".join(g.source for g in ca.genes if g.kind == GeneKind.FUNCTION)
        source_b = "\n\n".join(g.source for g in cb.genes if g.kind == GeneKind.FUNCTION)

        prompt = GENE_LEVEL_CROSSOVER.render(
            capability=ca.interface.capability if ca.interface else ca.name,
            parent_a_code=source_a,
            parent_b_code=source_b,
            interface=ca.interface,
        )

        merged_code = await llm.complete_code(prompt, temperature=0.7)

        # Validate the merged code
        if not _validate_python(merged_code):
            # Fallback: pick the longer parent's code
            merged_code = source_a if len(source_a) >= len(source_b) else source_b

        merged_gene = Gene(
            kind=GeneKind.FUNCTION,
            name=f"{ca.name}_x_{cb.name}",
            source=merged_code,
            metadata={"crossover": "gene_level", "parents": [ca.id, cb.id]},
        )

        return Chromosome(
            kind=ChromosomeKind.SKILL_MODULE,
            name=f"{ca.name}_x_{cb.name}",
            genes=[merged_gene],
            interface=ca.interface or cb.interface,
        )

    async def _merge_prompt_chromosomes(
        self, ca: Chromosome, cb: Chromosome, llm: LLMClient
    ) -> Chromosome:
        """Use LLM to merge two system prompt chromosomes."""
        prompt_a = "\n\n".join(g.source for g in ca.genes if g.kind == GeneKind.PROMPT_SECTION)
        prompt_b = "\n\n".join(g.source for g in cb.genes if g.kind == GeneKind.PROMPT_SECTION)

        prompt = PROMPT_CROSSOVER.render(
            parent_a_prompt=prompt_a,
            parent_b_prompt=prompt_b,
        )

        merged_text = (await llm.complete(prompt, temperature=0.7)).text

        merged_gene = Gene(
            kind=GeneKind.PROMPT_SECTION,
            name="merged_prompt",
            source=merged_text,
            metadata={"crossover": "prompt_merge", "parents": [ca.id, cb.id]},
        )

        return Chromosome(
            kind=ChromosomeKind.SYSTEM_PROMPT,
            name="system_prompt",
            genes=[merged_gene],
        )


class SemanticCrossover(CrossoverStrategy):
    """LLM reads both parents holistically and generates a novel blend.

    Analogous to: gene conversion with blending. The most expensive but
    highest-quality crossover strategy. The LLM understands both agents'
    complete designs and produces a coherent offspring.
    """

    async def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        llm: LLMClient,
    ) -> Genome:
        # Gather parent information
        prompt_a = self._extract_prompt(parent_a)
        prompt_b = self._extract_prompt(parent_b)
        skills_a = self._extract_skills(parent_a)
        skills_b = self._extract_skills(parent_b)

        prompt = SEMANTIC_CROSSOVER.render(
            parent_a_name="Parent_A",
            parent_b_name="Parent_B",
            parent_a_prompt=prompt_a,
            parent_b_prompt=prompt_b,
            parent_a_skills=skills_a,
            parent_b_skills=skills_b,
        )

        response = await llm.complete(prompt, temperature=0.8)

        # Parse the JSON response
        try:
            data = json.loads(response.text.strip().strip("```json").strip("```"))
        except json.JSONDecodeError:
            # Fallback to gene-level crossover
            fallback = GeneLevelCrossover()
            return await fallback.crossover(parent_a, parent_b, llm)

        chromosomes: list[Chromosome] = []

        # Build system prompt chromosome
        if "system_prompt" in data:
            chromosomes.append(Chromosome(
                kind=ChromosomeKind.SYSTEM_PROMPT,
                name="system_prompt",
                genes=[Gene(
                    kind=GeneKind.PROMPT_SECTION,
                    name="main_prompt",
                    source=data["system_prompt"],
                    metadata={"crossover": "semantic"},
                )],
            ))

        # Build skill chromosomes
        for skill_name, source in data.get("skills", {}).items():
            if _validate_python(source):
                chromosomes.append(Chromosome(
                    kind=ChromosomeKind.SKILL_MODULE,
                    name=skill_name,
                    genes=[Gene(
                        kind=GeneKind.FUNCTION,
                        name=skill_name,
                        source=source,
                        metadata={"crossover": "semantic"},
                    )],
                    interface=SkillInterface(capability=skill_name, tags=[skill_name]),
                ))

        return Genome(
            id=str(uuid.uuid4()),
            version=1,
            lineage=Lineage(
                parent_ids=[parent_a.id, parent_b.id],
                generation=max(parent_a.lineage.generation, parent_b.lineage.generation) + 1,
                crossover_points=["semantic_full"],
            ),
            chromosomes=chromosomes,
        )

    @staticmethod
    def _extract_prompt(genome: Genome) -> str:
        prompts = genome.get_system_prompt_chromosomes()
        if not prompts:
            return "(no system prompt)"
        return "\n\n".join(g.source for c in prompts for g in c.genes)

    @staticmethod
    def _extract_skills(genome: Genome) -> list[Gene]:
        skills = genome.get_skill_chromosomes()
        genes = []
        for chrom in skills:
            for gene in chrom.genes:
                if gene.kind == GeneKind.FUNCTION:
                    genes.append(gene)
        return genes


class AdaptiveCrossover(CrossoverStrategy):
    """Dynamically selects crossover strategy based on parent compatibility.

    - High compatibility (>0.6): chromosome-level (parents are similar, fine-grained merge not needed)
    - Medium compatibility (0.3-0.6): gene-level (sweet spot for blending)
    - Low compatibility (<0.3): semantic (need LLM to make sense of very different agents)
    """

    def __init__(self):
        self.chromosome_level = ChromosomeLevelCrossover()
        self.gene_level = GeneLevelCrossover(merge_rate=0.5)
        self.semantic = SemanticCrossover()

    async def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        llm: LLMClient,
    ) -> Genome:
        from inception.genome.compatibility import compatibility_score

        score = compatibility_score(parent_a, parent_b)

        if score > 0.6:
            return await self.chromosome_level.crossover(parent_a, parent_b, llm)
        elif score > 0.3:
            return await self.gene_level.crossover(parent_a, parent_b, llm)
        else:
            return await self.semantic.crossover(parent_a, parent_b, llm)
