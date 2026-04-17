"""Genome export with privacy-first redaction.

Exports agent genomes for network sharing according to the owner's
MatingPolicy. Sensitive data (source code, credentials) is redacted
by default — only hashes and structural metadata are shared.
"""

from __future__ import annotations

import hashlib

from inception.a2a.models import (
    ChromosomeSummary,
    GeneticAgentCard,
    GenomeExport,
    MatingPolicy,
)
from inception.genome.schema import AgentDNA, Chromosome, Genome
from inception.genome.serialization import genome_to_dict


def _hash_source(source: str) -> str:
    """SHA-256 hash of gene source code for identity without exposure."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _summarize_chromosome(chrom: Chromosome) -> ChromosomeSummary:
    """Create a redacted summary of a chromosome."""
    combined_source = "".join(g.source for g in chrom.genes)
    return ChromosomeSummary(
        id=chrom.id,
        kind=chrom.kind,
        name=chrom.name,
        gene_count=len(chrom.genes),
        capability=chrom.interface.capability if chrom.interface else "",
        tags=list(chrom.interface.tags) if chrom.interface else [],
        source_hash=_hash_source(combined_source),
    )


def export_genome(genome: Genome, policy: MatingPolicy) -> GenomeExport:
    """Export a genome with redaction according to MatingPolicy.

    If policy.share_source_code is False (the default), only
    chromosome summaries are included — no raw source code.
    """
    summaries = [_summarize_chromosome(c) for c in genome.chromosomes]

    full_chromosomes = None
    if policy.share_source_code:
        genome_dict = genome_to_dict(genome)
        full_chromosomes = genome_dict.get("chromosomes", [])

    cap_vector = genome.capability_vector() if policy.share_capability_vector else {}

    return GenomeExport(
        genome_id=genome.id,
        version=genome.version,
        chromosome_summaries=summaries,
        capability_vector=cap_vector,
        gene_count=genome.gene_count,
        full_chromosomes=full_chromosomes,
    )


def build_agent_card(
    agent: AgentDNA, base_url: str, policy: MatingPolicy
) -> GeneticAgentCard:
    """Build a GeneticAgentCard from an AgentDNA for network discovery."""
    genome = agent.genome
    caps = genome.capability_vector()

    skill_names = []
    for chrom in genome.get_skill_chromosomes():
        if chrom.interface and chrom.interface.capability:
            skill_names.append(chrom.interface.capability)

    species_tags = []
    for chrom in genome.chromosomes:
        if chrom.interface:
            species_tags.extend(chrom.interface.tags)
    species_tags = list(set(species_tags))

    fitness_dims = {}
    if policy.share_fitness_scores and agent.fitness_scores:
        latest = agent.fitness_scores[-1]
        fitness_dims = dict(latest.dimensions)

    lineage_depth = agent.genome.lineage.generation

    return GeneticAgentCard(
        name=agent.name,
        description=f"Generation {agent.generation} agent with {genome.gene_count} genes",
        url=base_url,
        agent_id=agent.id,
        generation=agent.generation,
        chromosome_count=len(genome.chromosomes),
        gene_count=genome.gene_count,
        capabilities=caps if policy.share_capability_vector else {},
        skill_names=skill_names,
        best_fitness=agent.best_fitness if policy.share_fitness_scores else 0.0,
        latest_fitness=agent.latest_fitness if policy.share_fitness_scores else 0.0,
        fitness_dimensions=fitness_dims,
        species_tags=species_tags,
        mating_policy=policy,
        created_at=agent.birth_timestamp.isoformat(),
        lineage_depth=lineage_depth,
    )


def reconstruct_mating_genome(export: GenomeExport) -> Genome | None:
    """Reconstruct a Genome from a GenomeExport for crossover.

    Returns None if the export doesn't include full chromosomes
    (i.e., the source code was redacted by the owner's policy).
    """
    if not export.full_chromosomes:
        return None

    from inception.genome.serialization import genome_from_dict

    genome_dict = {
        "id": export.genome_id,
        "version": export.version,
        "chromosomes": export.full_chromosomes,
    }
    return genome_from_dict(genome_dict)
