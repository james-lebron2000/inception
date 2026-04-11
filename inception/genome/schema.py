"""Core data models for AI Agent genomes — the 'DNA' of agents.

Maps biological evolution concepts to AI agent code:
- Gene: smallest heritable unit (function, prompt section, config block)
- Chromosome: group of related genes (skill module, system prompt)
- Genome: complete genetic material of one agent
- AgentDNA: genome + fitness history + lineage metadata
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────


class GeneKind(str, Enum):
    """Type of genetic unit."""

    FUNCTION = "function"  # A Python function body (skill implementation)
    PROMPT_SECTION = "prompt_section"  # A section of a system prompt
    CONFIG_BLOCK = "config_block"  # Configuration (model params, tool settings)
    TOOL_BINDING = "tool_binding"  # A tool definition (name, description, schema)


class ChromosomeKind(str, Enum):
    """Type of chromosome — groups of related genes."""

    SKILL_MODULE = "skill_module"  # Complete skill (e.g., write_python.py)
    SYSTEM_PROMPT = "system_prompt"  # Agent personality/instructions (SOUL.md)
    TOOL_CONFIG = "tool_config"  # Tool definitions and bindings
    MEMORY_STRATEGY = "memory_strategy"  # How the agent stores/retrieves memories


class MutationType(str, Enum):
    """Types of mutations that can be applied to genes."""

    POINT_MUTATION = "point"  # Small change to one gene
    INSERTION = "insertion"  # Add a new gene to a chromosome
    DELETION = "deletion"  # Remove a gene (simplification)
    DUPLICATION = "duplication"  # Copy a gene and diverge it
    INVERSION = "inversion"  # Restructure control flow
    TRANSPOSITION = "transposition"  # Move a gene between chromosomes
    REGULATORY = "regulatory"  # Change prompt/config (not code)
    SKILL_SYNTHESIS = "skill_synthesis"  # LLM generates entirely new skill


class AgentStatus(str, Enum):
    """Lifecycle status of an agent."""

    ALIVE = "alive"
    RETIRED = "retired"
    FAILED = "failed"


# ─── Core Models ─────────────────────────────────────────────────────────────


class Gene(BaseModel):
    """Smallest heritable unit — a single function, config block, or prompt section.

    Biological analog: a gene encoding a specific trait.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind: GeneKind
    name: str
    source: str  # The actual code/text content
    metadata: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)  # IDs of required genes
    locus: str = ""  # Position identifier (which chromosome, what slot)


class SkillInterface(BaseModel):
    """Contract for a skill — enables allele-like substitution.

    Two skills with the same capability and compatible interfaces
    can be swapped like alleles at the same locus.
    """

    capability: str  # What this skill does (e.g., "code_generation")
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class Chromosome(BaseModel):
    """A group of related genes — maps to a skill module or prompt file.

    Biological analog: a chromosome carrying multiple linked genes.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind: ChromosomeKind
    name: str
    genes: list[Gene] = Field(default_factory=list)
    interface: SkillInterface | None = None  # For SKILL_MODULE: inputs/outputs contract


class MutationRecord(BaseModel):
    """Record of a single mutation event."""

    mutation_type: MutationType
    target_gene_id: str
    description: str  # What changed
    diff: str = ""  # Unified diff
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Lineage(BaseModel):
    """Tracks ancestry of a genome."""

    parent_ids: list[str] = Field(default_factory=list)  # 0 for seed, 2 for offspring
    generation: int = 0
    crossover_points: list[str] = Field(default_factory=list)  # Chromosome boundaries crossed
    mutations_applied: list[MutationRecord] = Field(default_factory=list)


class Genome(BaseModel):
    """Complete genetic material of one agent.

    Biological analog: the full genome (all chromosomes) of an organism.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    lineage: Lineage = Field(default_factory=Lineage)
    chromosomes: list[Chromosome] = Field(default_factory=list)

    @property
    def gene_count(self) -> int:
        return sum(len(c.genes) for c in self.chromosomes)

    def get_chromosomes_by_kind(self, kind: ChromosomeKind) -> list[Chromosome]:
        return [c for c in self.chromosomes if c.kind == kind]

    def get_gene(self, gene_id: str) -> Gene | None:
        for chrom in self.chromosomes:
            for gene in chrom.genes:
                if gene.id == gene_id:
                    return gene
        return None

    def capability_vector(self) -> dict[str, float]:
        """Build a capability vector for compatibility scoring.

        Returns a dict mapping capability names to strength scores (0-1).
        """
        capabilities: dict[str, float] = {}
        for chrom in self.chromosomes:
            if chrom.interface and chrom.interface.capability:
                cap = chrom.interface.capability
                # More genes in a capability = higher strength
                strength = min(1.0, len(chrom.genes) / 5.0)
                capabilities[cap] = max(capabilities.get(cap, 0.0), strength)
            for tag in (chrom.interface.tags if chrom.interface else []):
                capabilities[tag] = max(capabilities.get(tag, 0.0), 0.5)
        return capabilities

    def get_skill_chromosomes(self) -> list[Chromosome]:
        return self.get_chromosomes_by_kind(ChromosomeKind.SKILL_MODULE)

    def get_system_prompt_chromosomes(self) -> list[Chromosome]:
        return self.get_chromosomes_by_kind(ChromosomeKind.SYSTEM_PROMPT)


class FitnessScore(BaseModel):
    """Result of one fitness evaluation.

    Biological analog: reproductive success / survival probability.
    """

    benchmark_id: str = ""
    overall: float = 0.0  # 0.0 to 1.0
    dimensions: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)


class AgentDNA(BaseModel):
    """Top-level agent identity: genome + phenotype metadata + fitness history.

    This is the complete representation of an agent individual in the population.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    genome: Genome
    fitness_scores: list[FitnessScore] = Field(default_factory=list)
    generation: int = 0
    parents: tuple[str, str] | None = None  # Parent agent IDs
    birth_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: AgentStatus = AgentStatus.ALIVE

    @property
    def best_fitness(self) -> float:
        if not self.fitness_scores:
            return 0.0
        return max(s.overall for s in self.fitness_scores)

    @property
    def latest_fitness(self) -> float:
        if not self.fitness_scores:
            return 0.0
        return self.fitness_scores[-1].overall
