"""Tests for mutation engine (non-LLM mutations only)."""

from inception.genome.schema import Chromosome, ChromosomeKind, Gene, GeneKind, MutationType
from inception.reproduction.mutation import MutationConfig, _select_mutation_type, _validate_python


class TestValidatePython:
    def test_valid_code(self):
        assert _validate_python("def foo(): pass")

    def test_invalid_code(self):
        assert not _validate_python("def foo(")

    def test_empty_code(self):
        assert _validate_python("")

    def test_complex_code(self):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        assert _validate_python(code)


class TestMutationTypeSelection:
    def test_select_mutation_type(self):
        weights = {
            MutationType.POINT_MUTATION.value: 1.0,
            MutationType.INSERTION.value: 0.0,
        }
        # With all weight on point mutation, should always get point mutation
        for _ in range(10):
            result = _select_mutation_type(weights)
            assert result == MutationType.POINT_MUTATION

    def test_all_types_possible(self):
        weights = {mt.value: 1.0 for mt in MutationType if mt != MutationType.SKILL_SYNTHESIS}
        results = set()
        for _ in range(1000):
            results.add(_select_mutation_type(weights))
        # Should eventually hit most types
        assert len(results) >= 3


class TestMutationConfig:
    def test_default_config(self):
        config = MutationConfig()
        assert config.mutation_rate == 0.1
        assert config.max_mutations_per_genome == 3
        assert len(config.mutation_type_weights) > 0

    def test_custom_config(self):
        config = MutationConfig(mutation_rate=0.5, max_mutations_per_genome=5)
        assert config.mutation_rate == 0.5
        assert config.max_mutations_per_genome == 5
