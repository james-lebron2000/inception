"""Pydantic models for the A2A (Agent-to-Agent) genetic exchange protocol.

Defines the data structures for agent discovery, mating policy,
genome export, and the mate request/response protocol.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from inception.genome.schema import AgentDNA, ChromosomeKind


class MatingPolicy(BaseModel):
    """Owner-defined controls on what genetic material can be shared remotely.

    Privacy by default: source code is never shared unless explicitly opted in.
    """

    allow_remote_mating: bool = False
    require_mutual_consent: bool = True
    share_source_code: bool = False
    redact_gene_sources: bool = True
    share_fitness_scores: bool = True
    share_capability_vector: bool = True
    share_lineage: bool = False
    allowed_requester_ids: list[str] = Field(default_factory=list)
    blocked_requester_ids: list[str] = Field(default_factory=list)
    max_matings_per_day: int = 10
    min_requester_fitness: float = 0.0
    preferred_crossover: str = "chromosome"


class ChromosomeSummary(BaseModel):
    """Redacted chromosome for network sharing — no source code by default."""

    id: str
    kind: ChromosomeKind
    name: str
    gene_count: int
    capability: str = ""
    tags: list[str] = Field(default_factory=list)
    source_hash: str = ""


class GenomeExport(BaseModel):
    """A redacted genome safe for network transmission.

    If the mating policy allows source sharing, full_chromosomes
    contains the complete chromosome data. Otherwise, only
    chromosome_summaries are provided.
    """

    genome_id: str
    version: int
    chromosome_summaries: list[ChromosomeSummary] = Field(default_factory=list)
    capability_vector: dict[str, float] = Field(default_factory=dict)
    gene_count: int = 0
    full_chromosomes: list[dict[str, Any]] | None = None


class GeneticAgentCard(BaseModel):
    """A2A Agent Card with genetic metadata, served at .well-known/agent.json.

    Extends the standard A2A Agent Card with genetic information
    to enable discovery and compatibility matching.
    """

    name: str
    description: str = ""
    url: str
    version: str = "0.1.0"
    protocol_version: str = "genesis/1.0"
    agent_id: str
    generation: int = 0
    chromosome_count: int = 0
    gene_count: int = 0
    capabilities: dict[str, float] = Field(default_factory=dict)
    skill_names: list[str] = Field(default_factory=list)
    best_fitness: float = 0.0
    latest_fitness: float = 0.0
    fitness_dimensions: dict[str, float] = Field(default_factory=dict)
    species_tags: list[str] = Field(default_factory=list)
    mating_policy: MatingPolicy = Field(default_factory=MatingPolicy)
    created_at: str = ""
    lineage_depth: int = 0


class MateRequest(BaseModel):
    """Request to mate with a remote agent via the A2A protocol."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requester_card: GeneticAgentCard
    task_description: str = ""
    preferred_crossover: str = "adaptive"
    requester_genome_export: GenomeExport | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MateResponse(BaseModel):
    """Response to a mate request."""

    request_id: str
    accepted: bool
    rejection_reason: str = ""
    responder_card: GeneticAgentCard | None = None
    responder_genome_export: GenomeExport | None = None
    offspring_genome_export: GenomeExport | None = None
    offspring_agent: AgentDNA | None = None


class RegistryEntry(BaseModel):
    """An agent's entry in the Gene Registry."""

    agent_card: GeneticAgentCard
    registered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    fitness_history: list[float] = Field(default_factory=list)
    mating_count: int = 0
    offspring_count: int = 0
