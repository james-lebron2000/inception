"""Hermes agent framework adapter.

Maps between Inception's genome model and Hermes's configuration format:
- Agent personality/config -> SYSTEM_PROMPT chromosome
- Skill definitions -> SKILL_MODULE chromosomes
- Learning loop config -> REGULATORY genes in MEMORY_STRATEGY chromosome
- Tool integrations -> TOOL_CONFIG chromosome

Hermes is notable for its self-improvement learning loop,
which maps naturally to our mutation/evolution mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inception.genome.schema import (
    Chromosome,
    ChromosomeKind,
    Gene,
    GeneKind,
    Genome,
    Lineage,
    SkillInterface,
)


class HermesAdapter:
    """Bidirectional adapter between Inception genomes and Hermes agents."""

    def genome_from_hermes(self, config: dict[str, Any]) -> Genome:
        """Import a Hermes agent configuration as an Inception Genome.

        Expected config structure:
        {
            "name": "agent_name",
            "personality": "...",
            "instructions": "...",
            "skills": {
                "skill_name": {
                    "description": "...",
                    "source": "python code...",
                    "tags": [...],
                }
            },
            "learning": {
                "enabled": true,
                "memory_strategy": "...",
                "skill_creation": true,
            },
            "tools": [...]
        }
        """
        chromosomes: list[Chromosome] = []

        # Personality + Instructions -> System Prompt
        prompt_genes: list[Gene] = []
        if "personality" in config:
            prompt_genes.append(Gene(
                kind=GeneKind.PROMPT_SECTION,
                name="personality",
                source=config["personality"],
                metadata={"source_format": "hermes"},
            ))
        if "instructions" in config:
            prompt_genes.append(Gene(
                kind=GeneKind.PROMPT_SECTION,
                name="instructions",
                source=config["instructions"],
                metadata={"source_format": "hermes"},
            ))

        if prompt_genes:
            chromosomes.append(Chromosome(
                kind=ChromosomeKind.SYSTEM_PROMPT,
                name="hermes_prompt",
                genes=prompt_genes,
            ))

        # Skills -> Skill Module chromosomes
        for skill_name, skill_def in config.get("skills", {}).items():
            source = skill_def.get("source", "")
            if not source:
                continue

            interface = SkillInterface(
                capability=skill_name,
                tags=skill_def.get("tags", [skill_name]),
            )

            chromosomes.append(Chromosome(
                kind=ChromosomeKind.SKILL_MODULE,
                name=skill_name,
                genes=[Gene(
                    kind=GeneKind.FUNCTION,
                    name=skill_name,
                    source=source,
                    metadata={
                        "source_format": "hermes_skill",
                        "description": skill_def.get("description", ""),
                    },
                )],
                interface=interface,
            ))

        # Learning config -> Memory Strategy chromosome
        learning = config.get("learning", {})
        if learning:
            memory_genes: list[Gene] = []

            if learning.get("memory_strategy"):
                memory_genes.append(Gene(
                    kind=GeneKind.CONFIG_BLOCK,
                    name="memory_strategy",
                    source=json.dumps(learning, indent=2),
                    metadata={"source_format": "hermes_learning"},
                ))

            # Hermes's learning loop maps to a REGULATORY gene
            if learning.get("enabled"):
                memory_genes.append(Gene(
                    kind=GeneKind.PROMPT_SECTION,
                    name="learning_loop",
                    source=(
                        "After each interaction:\n"
                        "1. Reflect on what went well and what could be improved\n"
                        "2. Extract reusable patterns as new skills\n"
                        "3. Update memory with key learnings\n"
                        "4. Adjust approach based on feedback"
                    ),
                    metadata={"hermes_feature": "learning_loop"},
                ))

            if memory_genes:
                chromosomes.append(Chromosome(
                    kind=ChromosomeKind.MEMORY_STRATEGY,
                    name="hermes_learning",
                    genes=memory_genes,
                ))

        # Tools -> Tool Config chromosome
        tools = config.get("tools", [])
        if tools:
            tool_genes = []
            for tool in tools:
                if isinstance(tool, dict):
                    tool_genes.append(Gene(
                        kind=GeneKind.TOOL_BINDING,
                        name=tool.get("name", "unnamed"),
                        source=json.dumps(tool, indent=2),
                        metadata={"source_format": "hermes_tool"},
                    ))
            if tool_genes:
                chromosomes.append(Chromosome(
                    kind=ChromosomeKind.TOOL_CONFIG,
                    name="hermes_tools",
                    genes=tool_genes,
                ))

        return Genome(
            lineage=Lineage(generation=0),
            chromosomes=chromosomes,
        )

    def genome_to_hermes(self, genome: Genome) -> dict[str, Any]:
        """Export an Inception Genome as a Hermes agent configuration."""
        config: dict[str, Any] = {}

        # System Prompt -> personality + instructions
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.SYSTEM_PROMPT):
            for gene in chrom.genes:
                if gene.name == "personality":
                    config["personality"] = gene.source
                elif gene.name == "instructions":
                    config["instructions"] = gene.source
                else:
                    # Concatenate other prompt sections as instructions
                    config.setdefault("instructions", "")
                    config["instructions"] += f"\n\n## {gene.name}\n{gene.source}"

        # Skill Modules -> skills
        skills: dict[str, Any] = {}
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.SKILL_MODULE):
            source_parts = [g.source for g in chrom.genes if g.kind == GeneKind.FUNCTION]
            if source_parts:
                skills[chrom.name] = {
                    "description": chrom.interface.capability if chrom.interface else chrom.name,
                    "source": "\n\n".join(source_parts),
                    "tags": chrom.interface.tags if chrom.interface else [chrom.name],
                }
        config["skills"] = skills

        # Memory Strategy -> learning config
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.MEMORY_STRATEGY):
            for gene in chrom.genes:
                if gene.kind == GeneKind.CONFIG_BLOCK:
                    try:
                        config["learning"] = json.loads(gene.source)
                    except json.JSONDecodeError:
                        config["learning"] = {"enabled": True}

        # Tool Config -> tools
        tools: list[dict[str, Any]] = []
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.TOOL_CONFIG):
            for gene in chrom.genes:
                if gene.kind == GeneKind.TOOL_BINDING:
                    try:
                        tools.append(json.loads(gene.source))
                    except json.JSONDecodeError:
                        tools.append({"name": gene.name, "raw": gene.source})
        config["tools"] = tools

        return config

    def genome_from_hermes_file(self, config_path: Path) -> Genome:
        """Load a Hermes agent from a JSON config file."""
        with open(config_path) as f:
            config = json.load(f)
        return self.genome_from_hermes(config)

    def genome_to_hermes_file(self, genome: Genome, output_path: Path) -> None:
        """Save a genome as a Hermes JSON config file."""
        config = self.genome_to_hermes(genome)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
