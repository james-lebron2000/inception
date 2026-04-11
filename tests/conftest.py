"""Shared test fixtures for Inception tests."""

from __future__ import annotations

import pytest

from inception.genome.schema import (
    AgentDNA,
    Chromosome,
    ChromosomeKind,
    Gene,
    GeneKind,
    Genome,
    Lineage,
    SkillInterface,
)


@pytest.fixture
def simple_gene():
    """A simple function gene."""
    return Gene(
        id="gene_1",
        kind=GeneKind.FUNCTION,
        name="add_numbers",
        source='def add_numbers(a, b):\n    """Add two numbers."""\n    return a + b\n',
    )


@pytest.fixture
def prompt_gene():
    """A system prompt gene."""
    return Gene(
        id="prompt_1",
        kind=GeneKind.PROMPT_SECTION,
        name="identity",
        source="You are a helpful Python developer.",
    )


@pytest.fixture
def skill_chromosome(simple_gene):
    """A skill module chromosome with one gene."""
    return Chromosome(
        id="chrom_1",
        kind=ChromosomeKind.SKILL_MODULE,
        name="math_skill",
        genes=[simple_gene],
        interface=SkillInterface(
            capability="arithmetic",
            tags=["math", "numbers"],
        ),
    )


@pytest.fixture
def prompt_chromosome(prompt_gene):
    """A system prompt chromosome."""
    return Chromosome(
        id="chrom_2",
        kind=ChromosomeKind.SYSTEM_PROMPT,
        name="system_prompt",
        genes=[prompt_gene],
    )


@pytest.fixture
def coder_genome(skill_chromosome, prompt_chromosome):
    """A coder agent genome with one skill and one prompt."""
    return Genome(
        id="genome_coder",
        chromosomes=[skill_chromosome, prompt_chromosome],
        lineage=Lineage(generation=0),
    )


@pytest.fixture
def researcher_genome():
    """A researcher agent genome with different skills."""
    return Genome(
        id="genome_researcher",
        chromosomes=[
            Chromosome(
                id="chrom_search",
                kind=ChromosomeKind.SKILL_MODULE,
                name="search_skill",
                genes=[Gene(
                    id="gene_search",
                    kind=GeneKind.FUNCTION,
                    name="search",
                    source='def search(query):\n    """Search for info."""\n    return [query]\n',
                )],
                interface=SkillInterface(
                    capability="information_retrieval",
                    tags=["search", "research"],
                ),
            ),
            Chromosome(
                id="chrom_summarize",
                kind=ChromosomeKind.SKILL_MODULE,
                name="summarize_skill",
                genes=[Gene(
                    id="gene_summarize",
                    kind=GeneKind.FUNCTION,
                    name="summarize",
                    source='def summarize(text):\n    """Summarize text."""\n    return text[:100]\n',
                )],
                interface=SkillInterface(
                    capability="text_summarization",
                    tags=["summarize", "nlp"],
                ),
            ),
            Chromosome(
                id="chrom_rprompt",
                kind=ChromosomeKind.SYSTEM_PROMPT,
                name="researcher_prompt",
                genes=[Gene(
                    id="prompt_r",
                    kind=GeneKind.PROMPT_SECTION,
                    name="identity",
                    source="You are a thorough research analyst.",
                )],
            ),
        ],
        lineage=Lineage(generation=0),
    )


@pytest.fixture
def coder_agent(coder_genome):
    """A coder AgentDNA."""
    return AgentDNA(
        id="agent_coder",
        name="coder",
        genome=coder_genome,
        generation=0,
    )


@pytest.fixture
def researcher_agent(researcher_genome):
    """A researcher AgentDNA."""
    return AgentDNA(
        id="agent_researcher",
        name="researcher",
        genome=researcher_genome,
        generation=0,
    )
