"""Genome compatibility scoring for mate selection.

Determines how compatible two genomes are for crossover.
Too similar (>0.7): crossover yields little novelty.
Too different (<0.3): offspring likely non-viable.
Sweet spot: 0.3 - 0.7.

Biological analog: species compatibility for successful mating.
"""

from __future__ import annotations

from inception.genome.schema import Chromosome, ChromosomeKind, Genome


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _capability_tags(genome: Genome) -> set[str]:
    """Extract all capability tags from a genome."""
    tags: set[str] = set()
    for chrom in genome.chromosomes:
        if chrom.interface:
            tags.add(chrom.interface.capability)
            tags.update(chrom.interface.tags)
    return tags


def _structural_similarity(a: Genome, b: Genome) -> float:
    """Compare structural properties: chromosome count, gene count ratios."""
    chrom_ratio = min(len(a.chromosomes), len(b.chromosomes)) / max(
        len(a.chromosomes), len(b.chromosomes), 1
    )
    gene_ratio = min(a.gene_count, b.gene_count) / max(a.gene_count, b.gene_count, 1)
    return (chrom_ratio + gene_ratio) / 2.0


def _interface_compatibility(a: Genome, b: Genome) -> float:
    """How many skills have matching interfaces (same capability)?"""
    caps_a = {
        c.interface.capability
        for c in a.chromosomes
        if c.interface and c.kind == ChromosomeKind.SKILL_MODULE
    }
    caps_b = {
        c.interface.capability
        for c in b.chromosomes
        if c.interface and c.kind == ChromosomeKind.SKILL_MODULE
    }
    return _jaccard_similarity(caps_a, caps_b)


def compatibility_score(a: Genome, b: Genome) -> float:
    """Compute overall compatibility score between two genomes.

    Returns a float from 0.0 (incompatible) to 1.0 (identical).

    Components (weighted):
    - Capability tag overlap (Jaccard): 40%
    - Structural similarity: 30%
    - Interface compatibility: 30%
    """
    cap_sim = _jaccard_similarity(_capability_tags(a), _capability_tags(b))
    struct_sim = _structural_similarity(a, b)
    iface_sim = _interface_compatibility(a, b)

    return 0.4 * cap_sim + 0.3 * struct_sim + 0.3 * iface_sim


def is_compatible_for_mating(a: Genome, b: Genome, low: float = 0.3, high: float = 0.7) -> bool:
    """Check if two genomes are in the optimal compatibility range for mating."""
    score = compatibility_score(a, b)
    return low <= score <= high


def find_homologous_pairs(
    a: Genome, b: Genome
) -> tuple[list[tuple[Chromosome, Chromosome]], list[Chromosome], list[Chromosome]]:
    """Find homologous chromosome pairs between two genomes.

    Returns:
        (paired, unpaired_a, unpaired_b)
        - paired: list of (chrom_a, chrom_b) pairs sharing same capability
        - unpaired_a: chromosomes unique to genome A
        - unpaired_b: chromosomes unique to genome B
    """
    paired: list[tuple[Chromosome, Chromosome]] = []
    used_b: set[str] = set()

    unpaired_a: list[Chromosome] = []

    for ca in a.chromosomes:
        matched = False
        if ca.interface and ca.kind == ChromosomeKind.SKILL_MODULE:
            for cb in b.chromosomes:
                if cb.id in used_b:
                    continue
                if (
                    cb.interface
                    and cb.kind == ChromosomeKind.SKILL_MODULE
                    and cb.interface.capability == ca.interface.capability
                ):
                    paired.append((ca, cb))
                    used_b.add(cb.id)
                    matched = True
                    break

        # Also pair system prompts, tool configs, etc. by kind
        if not matched and ca.kind != ChromosomeKind.SKILL_MODULE:
            for cb in b.chromosomes:
                if cb.id in used_b:
                    continue
                if cb.kind == ca.kind:
                    paired.append((ca, cb))
                    used_b.add(cb.id)
                    matched = True
                    break

        if not matched:
            unpaired_a.append(ca)

    unpaired_b = [cb for cb in b.chromosomes if cb.id not in used_b]

    return paired, unpaired_a, unpaired_b
