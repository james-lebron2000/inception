"""Tests for genome data models."""

from inception.genome.schema import (
    AgentDNA,
    AgentStatus,
    Chromosome,
    ChromosomeKind,
    FitnessScore,
    Gene,
    GeneKind,
    Genome,
    Lineage,
    MutationRecord,
    MutationType,
    SkillInterface,
)


class TestGene:
    def test_create_function_gene(self):
        gene = Gene(kind=GeneKind.FUNCTION, name="test", source="def test(): pass")
        assert gene.kind == GeneKind.FUNCTION
        assert gene.name == "test"
        assert gene.source == "def test(): pass"
        assert gene.id  # Auto-generated

    def test_create_prompt_gene(self):
        gene = Gene(kind=GeneKind.PROMPT_SECTION, name="identity", source="You are helpful")
        assert gene.kind == GeneKind.PROMPT_SECTION

    def test_gene_with_dependencies(self):
        gene = Gene(
            kind=GeneKind.FUNCTION,
            name="test",
            source="pass",
            dependencies=["dep1", "dep2"],
        )
        assert len(gene.dependencies) == 2


class TestChromosome:
    def test_create_skill_chromosome(self, simple_gene):
        chrom = Chromosome(
            kind=ChromosomeKind.SKILL_MODULE,
            name="test_skill",
            genes=[simple_gene],
            interface=SkillInterface(capability="math", tags=["numbers"]),
        )
        assert chrom.kind == ChromosomeKind.SKILL_MODULE
        assert len(chrom.genes) == 1
        assert chrom.interface.capability == "math"

    def test_chromosome_without_interface(self, prompt_gene):
        chrom = Chromosome(
            kind=ChromosomeKind.SYSTEM_PROMPT,
            name="prompt",
            genes=[prompt_gene],
        )
        assert chrom.interface is None


class TestGenome:
    def test_gene_count(self, coder_genome):
        assert coder_genome.gene_count == 2  # 1 skill gene + 1 prompt gene

    def test_get_chromosomes_by_kind(self, coder_genome):
        skills = coder_genome.get_chromosomes_by_kind(ChromosomeKind.SKILL_MODULE)
        assert len(skills) == 1
        prompts = coder_genome.get_chromosomes_by_kind(ChromosomeKind.SYSTEM_PROMPT)
        assert len(prompts) == 1

    def test_get_gene(self, coder_genome):
        gene = coder_genome.get_gene("gene_1")
        assert gene is not None
        assert gene.name == "add_numbers"

    def test_get_missing_gene(self, coder_genome):
        assert coder_genome.get_gene("nonexistent") is None

    def test_capability_vector(self, coder_genome):
        caps = coder_genome.capability_vector()
        assert "arithmetic" in caps
        assert caps["arithmetic"] > 0

    def test_skill_chromosomes(self, coder_genome):
        skills = coder_genome.get_skill_chromosomes()
        assert len(skills) == 1
        assert skills[0].name == "math_skill"


class TestAgentDNA:
    def test_create_agent(self, coder_genome):
        agent = AgentDNA(name="test_agent", genome=coder_genome)
        assert agent.name == "test_agent"
        assert agent.generation == 0
        assert agent.status == AgentStatus.ALIVE
        assert agent.parents is None

    def test_best_fitness_empty(self, coder_agent):
        assert coder_agent.best_fitness == 0.0

    def test_best_fitness_with_scores(self, coder_agent):
        coder_agent.fitness_scores = [
            FitnessScore(overall=0.5),
            FitnessScore(overall=0.8),
            FitnessScore(overall=0.6),
        ]
        assert coder_agent.best_fitness == 0.8

    def test_latest_fitness(self, coder_agent):
        coder_agent.fitness_scores = [
            FitnessScore(overall=0.5),
            FitnessScore(overall=0.8),
        ]
        assert coder_agent.latest_fitness == 0.8

    def test_agent_with_parents(self, coder_genome):
        agent = AgentDNA(
            name="child",
            genome=coder_genome,
            parents=("parent_a", "parent_b"),
        )
        assert agent.parents == ("parent_a", "parent_b")


class TestLineage:
    def test_seed_lineage(self):
        lineage = Lineage()
        assert lineage.generation == 0
        assert len(lineage.parent_ids) == 0

    def test_offspring_lineage(self):
        lineage = Lineage(
            parent_ids=["p1", "p2"],
            generation=3,
            crossover_points=["skill_a"],
        )
        assert lineage.generation == 3
        assert len(lineage.parent_ids) == 2


class TestMutationRecord:
    def test_create_record(self):
        record = MutationRecord(
            mutation_type=MutationType.POINT_MUTATION,
            target_gene_id="gene_1",
            description="Improved error handling",
        )
        assert record.mutation_type == MutationType.POINT_MUTATION
        assert record.timestamp is not None


class TestFitnessScore:
    def test_create_score(self):
        score = FitnessScore(
            benchmark_id="test",
            overall=0.85,
            dimensions={"accuracy": 0.9, "speed": 0.8},
        )
        assert score.overall == 0.85
        assert score.dimensions["accuracy"] == 0.9
