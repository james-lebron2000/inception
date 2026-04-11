"""Agent runtime — instantiate and run agents from their genomes.

Compiles a Genome into a runnable agent by:
1. Assembling skill functions into callable modules
2. Building the system prompt from prompt chromosomes
3. Configuring tools from tool config chromosomes

Biological analog: gene expression — DNA -> mRNA -> protein -> organism behavior.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

from inception.genome.schema import AgentDNA, ChromosomeKind, GeneKind, Genome


class AgentInstance:
    """A running agent compiled from a genome.

    Contains:
    - system_prompt: assembled personality and instructions
    - skills: dict of skill_name -> callable module
    - tools: list of tool configurations
    - metadata: agent identity info
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        skills: dict[str, types.ModuleType],
        tools: list[dict[str, Any]],
        genome: Genome,
    ):
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.skills = skills
        self.tools = tools
        self.genome = genome

    def get_skill(self, name: str) -> types.ModuleType | None:
        """Get a skill module by name."""
        return self.skills.get(name)

    def list_capabilities(self) -> list[str]:
        """List all available skill names."""
        return list(self.skills.keys())

    def describe(self) -> str:
        """Human-readable description of this agent."""
        return (
            f"Agent: {self.name} (id={self.agent_id[:8]})\n"
            f"Skills: {', '.join(self.list_capabilities())}\n"
            f"Tools: {len(self.tools)}\n"
            f"Prompt length: {len(self.system_prompt)} chars"
        )


class AgentRuntime:
    """Compiles genomes into runnable agent instances."""

    def instantiate(self, agent: AgentDNA) -> AgentInstance:
        """Compile an AgentDNA into a runnable AgentInstance.

        This is the 'gene expression' step — turning genotype into phenotype.
        """
        genome = agent.genome

        # 1. Assemble system prompt
        system_prompt = self._assemble_system_prompt(genome)

        # 2. Compile skill modules
        skills = self._compile_skills(genome)

        # 3. Extract tool configurations
        tools = self._extract_tools(genome)

        return AgentInstance(
            agent_id=agent.id,
            name=agent.name,
            system_prompt=system_prompt,
            skills=skills,
            tools=tools,
            genome=genome,
        )

    def _assemble_system_prompt(self, genome: Genome) -> str:
        """Build the complete system prompt from prompt chromosomes."""
        prompt_parts: list[str] = []
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.SYSTEM_PROMPT):
            for gene in chrom.genes:
                if gene.kind == GeneKind.PROMPT_SECTION:
                    prompt_parts.append(gene.source)
        return "\n\n".join(prompt_parts)

    def _compile_skills(self, genome: Genome) -> dict[str, types.ModuleType]:
        """Compile skill chromosomes into importable Python modules.

        Each SKILL_MODULE chromosome becomes a Python module with
        its function genes as module-level functions.
        """
        skills: dict[str, types.ModuleType] = {}

        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.SKILL_MODULE):
            source_parts = []
            for gene in chrom.genes:
                if gene.kind == GeneKind.FUNCTION:
                    source_parts.append(gene.source)

            if not source_parts:
                continue

            full_source = "\n\n".join(source_parts)

            # Compile into a module
            module = self._compile_module(chrom.name, full_source)
            if module:
                skills[chrom.name] = module

        return skills

    def _compile_module(self, name: str, source: str) -> types.ModuleType | None:
        """Compile source code into a Python module."""
        try:
            # Create a temporary file for the module
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix=f"inception_{name}_", delete=False
            ) as f:
                f.write(source)
                temp_path = f.name

            spec = importlib.util.spec_from_file_location(
                f"inception.runtime.{name}", temp_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception:
            pass
        return None

    def _extract_tools(self, genome: Genome) -> list[dict[str, Any]]:
        """Extract tool configurations from tool config chromosomes."""
        import json

        tools: list[dict[str, Any]] = []
        for chrom in genome.get_chromosomes_by_kind(ChromosomeKind.TOOL_CONFIG):
            for gene in chrom.genes:
                if gene.kind == GeneKind.TOOL_BINDING:
                    try:
                        tool_def = json.loads(gene.source)
                        tools.append(tool_def)
                    except json.JSONDecodeError:
                        tools.append({"name": gene.name, "raw": gene.source})
        return tools
