"""Serialize and deserialize agent genomes to/from YAML files.

Supports round-trip: Genome -> YAML -> Genome with full fidelity.
Also handles loading seed agents from directory structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

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


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    """Convert a Genome to a plain dict for serialization."""
    return json.loads(genome.model_dump_json())


def genome_from_dict(data: dict[str, Any]) -> Genome:
    """Reconstruct a Genome from a plain dict."""
    return Genome.model_validate(data)


def save_genome_yaml(genome: Genome, path: Path) -> None:
    """Save a genome to a YAML file."""
    data = genome_to_dict(genome)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_genome_yaml(path: Path) -> Genome:
    """Load a genome from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return genome_from_dict(data)


def save_agent_yaml(agent: AgentDNA, path: Path) -> None:
    """Save a complete AgentDNA to a YAML file."""
    data = json.loads(agent.model_dump_json())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_agent_yaml(path: Path) -> AgentDNA:
    """Load a complete AgentDNA from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentDNA.model_validate(data)


def load_seed_agent_from_directory(agent_dir: Path) -> AgentDNA:
    """Load a seed agent from a directory structure.

    Expected structure:
        agent_dir/
        ├── genome.yaml       # Metadata: name, interface definitions
        └── skills/
            ├── skill1.py     # Each .py file becomes a SKILL_MODULE chromosome
            ├── skill2.py
            └── ...

    The genome.yaml contains agent metadata and optional chromosome configs.
    Each .py file in skills/ is loaded as a SKILL_MODULE chromosome with
    a single FUNCTION gene containing the file's source code.
    """
    genome_path = agent_dir / "genome.yaml"
    if not genome_path.exists():
        raise FileNotFoundError(f"No genome.yaml found in {agent_dir}")

    with open(genome_path) as f:
        meta = yaml.safe_load(f)

    agent_name = meta.get("name", agent_dir.name)
    chromosomes: list[Chromosome] = []

    # Load system prompt if defined in genome.yaml
    if "system_prompt" in meta:
        prompt_genes = []
        prompt_data = meta["system_prompt"]
        if isinstance(prompt_data, str):
            prompt_genes.append(Gene(
                kind=GeneKind.PROMPT_SECTION,
                name="main_prompt",
                source=prompt_data,
            ))
        elif isinstance(prompt_data, dict):
            for section_name, section_text in prompt_data.items():
                prompt_genes.append(Gene(
                    kind=GeneKind.PROMPT_SECTION,
                    name=section_name,
                    source=str(section_text),
                ))
        chromosomes.append(Chromosome(
            kind=ChromosomeKind.SYSTEM_PROMPT,
            name=f"{agent_name}_prompt",
            genes=prompt_genes,
        ))

    # Load skill files
    skills_dir = agent_dir / "skills"
    if skills_dir.exists():
        for skill_file in sorted(skills_dir.glob("*.py")):
            source = skill_file.read_text()
            skill_name = skill_file.stem

            # Extract interface from genome.yaml if available
            interface = None
            skill_meta = meta.get("skills", {}).get(skill_name, {})
            if skill_meta:
                interface = SkillInterface(
                    capability=skill_meta.get("capability", skill_name),
                    input_schema=skill_meta.get("input_schema", {}),
                    output_schema=skill_meta.get("output_schema", {}),
                    tags=skill_meta.get("tags", []),
                )
            else:
                interface = SkillInterface(
                    capability=skill_name,
                    tags=[skill_name],
                )

            gene = Gene(
                kind=GeneKind.FUNCTION,
                name=skill_name,
                source=source,
                locus=f"{agent_name}.{skill_name}",
            )

            chromosomes.append(Chromosome(
                kind=ChromosomeKind.SKILL_MODULE,
                name=skill_name,
                genes=[gene],
                interface=interface,
            ))

    # Load tool config if present
    if "tools" in meta:
        tool_genes = []
        for tool_name, tool_def in meta["tools"].items():
            tool_genes.append(Gene(
                kind=GeneKind.TOOL_BINDING,
                name=tool_name,
                source=json.dumps(tool_def, indent=2),
            ))
        chromosomes.append(Chromosome(
            kind=ChromosomeKind.TOOL_CONFIG,
            name=f"{agent_name}_tools",
            genes=tool_genes,
        ))

    genome = Genome(
        lineage=Lineage(generation=0),
        chromosomes=chromosomes,
    )

    return AgentDNA(
        name=agent_name,
        genome=genome,
        generation=0,
    )
