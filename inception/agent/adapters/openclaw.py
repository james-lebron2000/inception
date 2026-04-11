"""OpenClaw framework adapter.

Maps between Inception's genome model and OpenClaw's file-based format:
- SOUL.md <-> SYSTEM_PROMPT chromosome
- skills/*.py <-> SKILL_MODULE chromosomes
- Channel/Brain/Body architecture <-> meta-chromosomes

This enables importing existing OpenClaw agents as seed genomes
and exporting evolved agents back to OpenClaw format.
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


class OpenClawAdapter:
    """Bidirectional adapter between Inception genomes and OpenClaw agents."""

    def genome_from_openclaw(self, agent_dir: Path) -> Genome:
        """Import an OpenClaw agent directory as an Inception Genome.

        Expected OpenClaw structure:
            agent_dir/
            ├── SOUL.md          -> SYSTEM_PROMPT chromosome
            ├── skills/
            │   ├── SKILL.md     -> skill metadata
            │   └── *.py         -> SKILL_MODULE chromosomes
            └── config.json      -> TOOL_CONFIG chromosome
        """
        chromosomes: list[Chromosome] = []

        # Parse SOUL.md -> System Prompt chromosome
        soul_path = agent_dir / "SOUL.md"
        if soul_path.exists():
            soul_content = soul_path.read_text()
            sections = self._parse_markdown_sections(soul_content)

            prompt_genes = []
            for section_name, section_text in sections.items():
                prompt_genes.append(Gene(
                    kind=GeneKind.PROMPT_SECTION,
                    name=section_name,
                    source=section_text,
                    metadata={"source_format": "openclaw_soul_md"},
                ))

            chromosomes.append(Chromosome(
                kind=ChromosomeKind.SYSTEM_PROMPT,
                name="soul",
                genes=prompt_genes,
            ))

        # Parse skill files -> Skill Module chromosomes
        skills_dir = agent_dir / "skills"
        if skills_dir.exists():
            # Load skill metadata if available
            skill_meta = {}
            skill_md = skills_dir / "SKILL.md"
            if skill_md.exists():
                skill_meta = self._parse_skill_md(skill_md.read_text())

            for skill_file in sorted(skills_dir.glob("*.py")):
                skill_name = skill_file.stem
                source = skill_file.read_text()

                meta = skill_meta.get(skill_name, {})
                interface = SkillInterface(
                    capability=meta.get("capability", skill_name),
                    tags=meta.get("tags", [skill_name]),
                    input_schema=meta.get("input_schema", {}),
                    output_schema=meta.get("output_schema", {}),
                )

                chromosomes.append(Chromosome(
                    kind=ChromosomeKind.SKILL_MODULE,
                    name=skill_name,
                    genes=[Gene(
                        kind=GeneKind.FUNCTION,
                        name=skill_name,
                        source=source,
                        metadata={"source_format": "openclaw_skill"},
                    )],
                    interface=interface,
                ))

        # Parse config.json -> Tool Config chromosome
        config_path = agent_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            tool_genes = []

            for tool_name, tool_def in config.get("tools", {}).items():
                tool_genes.append(Gene(
                    kind=GeneKind.TOOL_BINDING,
                    name=tool_name,
                    source=json.dumps(tool_def, indent=2),
                    metadata={"source_format": "openclaw_config"},
                ))

            if tool_genes:
                chromosomes.append(Chromosome(
                    kind=ChromosomeKind.TOOL_CONFIG,
                    name="openclaw_tools",
                    genes=tool_genes,
                ))

        return Genome(
            lineage=Lineage(generation=0),
            chromosomes=chromosomes,
        )

    def genome_to_openclaw(self, genome: Genome, output_dir: Path) -> None:
        """Export an Inception Genome as an OpenClaw agent directory.

        Creates:
            output_dir/
            ├── SOUL.md
            ├── skills/
            │   └── *.py
            └── config.json
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write SOUL.md from system prompt chromosomes
        prompt_parts: list[str] = []
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.SYSTEM_PROMPT):
            for gene in chrom.genes:
                section = f"## {gene.name}\n\n{gene.source}"
                prompt_parts.append(section)

        if prompt_parts:
            soul_content = "# SOUL\n\n" + "\n\n".join(prompt_parts)
            (output_dir / "SOUL.md").write_text(soul_content)

        # Write skill files
        skills_dir = output_dir / "skills"
        skills_dir.mkdir(exist_ok=True)

        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.SKILL_MODULE):
            source_parts = [g.source for g in chrom.genes if g.kind == GeneKind.FUNCTION]
            if source_parts:
                (skills_dir / f"{chrom.name}.py").write_text("\n\n".join(source_parts))

        # Write config.json from tool config chromosomes
        tools: dict[str, Any] = {}
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.TOOL_CONFIG):
            for gene in chrom.genes:
                if gene.kind == GeneKind.TOOL_BINDING:
                    try:
                        tools[gene.name] = json.loads(gene.source)
                    except json.JSONDecodeError:
                        tools[gene.name] = {"raw": gene.source}

        if tools:
            (output_dir / "config.json").write_text(json.dumps({"tools": tools}, indent=2))

    @staticmethod
    def _parse_markdown_sections(content: str) -> dict[str, str]:
        """Parse markdown into sections keyed by heading."""
        sections: dict[str, str] = {}
        current_heading = "main"
        current_content: list[str] = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_content:
                    sections[current_heading] = "\n".join(current_content).strip()
                current_heading = line[3:].strip().lower().replace(" ", "_")
                current_content = []
            elif line.startswith("# "):
                if current_content:
                    sections[current_heading] = "\n".join(current_content).strip()
                current_heading = line[2:].strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_heading] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def _parse_skill_md(content: str) -> dict[str, dict[str, Any]]:
        """Parse SKILL.md metadata for skill definitions."""
        skills: dict[str, dict[str, Any]] = {}
        current_skill: str | None = None

        for line in content.split("\n"):
            if line.startswith("## "):
                current_skill = line[3:].strip().lower().replace(" ", "_")
                skills[current_skill] = {}
            elif current_skill and ": " in line:
                key, value = line.split(": ", 1)
                key = key.strip().lower()
                skills[current_skill][key] = value.strip()

        return skills
