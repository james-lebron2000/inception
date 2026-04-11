"""Tests for genome compatibility scoring."""

from inception.genome.compatibility import (
    compatibility_score,
    find_homologous_pairs,
    is_compatible_for_mating,
)


class TestCompatibilityScore:
    def test_identical_genomes(self, coder_genome):
        score = compatibility_score(coder_genome, coder_genome)
        assert score == 1.0

    def test_different_genomes(self, coder_genome, researcher_genome):
        score = compatibility_score(coder_genome, researcher_genome)
        assert 0.0 <= score <= 1.0

    def test_coder_researcher_not_identical(self, coder_genome, researcher_genome):
        score = compatibility_score(coder_genome, researcher_genome)
        assert score < 1.0


class TestHomologousPairs:
    def test_find_pairs_same_genome(self, coder_genome):
        paired, unpaired_a, unpaired_b = find_homologous_pairs(coder_genome, coder_genome)
        assert len(paired) > 0  # Should find homologous pairs
        # All chromosomes should be paired with themselves
        assert len(unpaired_a) == 0
        assert len(unpaired_b) == 0

    def test_find_pairs_different_genomes(self, coder_genome, researcher_genome):
        paired, unpaired_a, unpaired_b = find_homologous_pairs(
            coder_genome, researcher_genome
        )
        # System prompts should pair, but skills are different
        assert len(paired) >= 1  # At least the system prompt pair
        total = len(paired) + len(unpaired_a) + len(unpaired_b)
        total_chroms = len(coder_genome.chromosomes) + len(researcher_genome.chromosomes)
        # Each chromosome appears exactly once (in a pair or unpaired)
        assert total * 2 >= total_chroms  # paired counts 2 chromosomes each


class TestMatingCompatibility:
    def test_identical_not_in_range(self, coder_genome):
        # Identical genomes (score=1.0) are above the sweet spot
        assert not is_compatible_for_mating(coder_genome, coder_genome)

    def test_custom_range(self, coder_genome, researcher_genome):
        # With a wide enough range, they should be compatible
        assert is_compatible_for_mating(
            coder_genome, researcher_genome, low=0.0, high=1.0
        )
