"""Tests for genome serialization."""

import tempfile
from pathlib import Path

from inception.genome.schema import AgentDNA, Genome
from inception.genome.serialization import (
    genome_from_dict,
    genome_to_dict,
    load_agent_yaml,
    load_genome_yaml,
    load_seed_agent_from_directory,
    save_agent_yaml,
    save_genome_yaml,
)


class TestGenomeSerialization:
    def test_genome_to_dict_roundtrip(self, coder_genome):
        data = genome_to_dict(coder_genome)
        restored = genome_from_dict(data)
        assert restored.gene_count == coder_genome.gene_count
        assert len(restored.chromosomes) == len(coder_genome.chromosomes)

    def test_save_load_genome_yaml(self, coder_genome):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "genome.yaml"
            save_genome_yaml(coder_genome, path)

            assert path.exists()

            loaded = load_genome_yaml(path)
            assert loaded.gene_count == coder_genome.gene_count
            assert len(loaded.chromosomes) == len(coder_genome.chromosomes)

    def test_save_load_agent_yaml(self, coder_agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.yaml"
            save_agent_yaml(coder_agent, path)

            assert path.exists()

            loaded = load_agent_yaml(path)
            assert loaded.name == coder_agent.name
            assert loaded.genome.gene_count == coder_agent.genome.gene_count


class TestSeedAgentLoading:
    def test_load_seed_agent(self):
        seed_dir = Path(__file__).parent.parent.parent / "examples" / "seed_agents" / "coder_agent"
        if not seed_dir.exists():
            return  # Skip if seed agents not available

        agent = load_seed_agent_from_directory(seed_dir)
        assert agent.name == "coder_agent"
        assert agent.genome.gene_count > 0
        assert len(agent.genome.get_skill_chromosomes()) > 0

    def test_load_all_seed_agents(self):
        seed_base = Path(__file__).parent.parent.parent / "examples" / "seed_agents"
        if not seed_base.exists():
            return

        agents = []
        for agent_dir in seed_base.iterdir():
            if agent_dir.is_dir() and (agent_dir / "genome.yaml").exists():
                agent = load_seed_agent_from_directory(agent_dir)
                agents.append(agent)

        assert len(agents) == 3  # coder, researcher, planner
        names = {a.name for a in agents}
        assert "coder_agent" in names
        assert "researcher_agent" in names
        assert "planner_agent" in names
