"""Tests for the Mendel mating engine."""

from __future__ import annotations

import pytest

from inception.skills.mendel import (
    MendelConfig,
    MendelEngine,
    format_tsv_row,
    TSV_HEADER,
    _merge_descriptions,
)
from inception.skills.rubric import evaluate_full, parse_skill_md


CODER_SKILL = """---
name: coder-skill
description: "Write and debug Python code. Triggers: 'code', 'python', 'debug'"
---

# Coder Skill

> A skill for writing clean, efficient Python code.

## Design Philosophy

1. Write correct, working code
2. Handle edge cases
3. Keep functions focused

## Workflow

### Phase 1: Understand

1. Read the user's request
2. Identify requirements and constraints
3. Plan the solution approach

**Output:** Solution plan

### Phase 2: Implement

1. Write the Python code
2. Add type hints and docstrings
3. If syntax error, fix immediately

**Pause:** Show code to user for review.

### Phase 3: Test

1. Run the code with sample inputs
2. Check edge cases
3. If tests fail, debug and fix

**Output:** Working, tested code

## Boundary Conditions

- If requirements are unclear, ask for clarification
- Never use global mutable state
- Always include error handling for I/O operations
- Maximum function length: 50 lines

## Examples

```python
def fibonacci(n: int) -> list[int]:
    \"\"\"Generate Fibonacci sequence.\"\"\"
    if n <= 0:
        return []
    result = [0, 1]
    for _ in range(2, n):
        result.append(result[-1] + result[-2])
    return result[:n]
```
"""

RESEARCHER_SKILL = """---
name: researcher-skill
description: "Research topics and summarize findings. Triggers: 'research', 'analyze', 'summarize'"
---

# Researcher Skill

> A skill for thorough research and analysis.

## Workflow

### Step 1: Define Scope

1. Identify the research question
2. List key topics to investigate
3. Set boundaries on scope

### Step 2: Gather Information

1. Search multiple sources
2. Cross-reference findings
3. Note conflicting information

### Step 3: Analyze

1. Identify patterns and themes
2. Evaluate source reliability
3. Draw conclusions

**Pause:** Present findings to user for validation.

### Step 4: Summarize

1. Write executive summary
2. List key findings with evidence
3. Provide recommendations

**Output:** Research report in markdown format

## Quality Standards

- Always cite sources
- Flag uncertainty explicitly
- If data is insufficient, recommend further research
- Keep summaries under 500 words unless requested otherwise
"""


@pytest.fixture
def coder_path(tmp_path):
    path = tmp_path / "coder.md"
    path.write_text(CODER_SKILL)
    return path


@pytest.fixture
def researcher_path(tmp_path):
    path = tmp_path / "researcher.md"
    path.write_text(RESEARCHER_SKILL)
    return path


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "offspring.md"


class TestMendelEngine:
    def test_breed_produces_result(self, coder_path, researcher_path, output_path):
        engine = MendelEngine()
        result = engine.breed(coder_path, researcher_path, output_path)

        assert result.parent_a.skill_name == "coder-skill"
        assert result.parent_b.skill_name == "researcher-skill"
        assert result.offspring is not None
        assert result.strategy in ("section-wise", "dimension-wise", "auto")
        assert result.status in ("keep", "revert")

    def test_offspring_has_valid_scores(self, coder_path, researcher_path, output_path):
        engine = MendelEngine()
        result = engine.breed(coder_path, researcher_path, output_path)

        assert 0 < result.offspring.total_score <= 100
        assert len(result.offspring.dimensions) == 8

    def test_offspring_inherits_from_both(self, coder_path, researcher_path, output_path):
        engine = MendelEngine()
        result = engine.breed(coder_path, researcher_path, output_path)

        sources = set(result.inherited_from.values())
        # Should inherit from at least one parent (often both)
        assert len(sources) >= 1

    def test_offspring_name_combines_parents(self, coder_path, researcher_path, output_path):
        engine = MendelEngine()
        result = engine.breed(coder_path, researcher_path, output_path)

        name = result.offspring.skill_name
        assert "x" in name  # crossover marker

    def test_section_wise_strategy(self, coder_path, researcher_path, output_path):
        config = MendelConfig(strategy="section-wise")
        engine = MendelEngine(config)
        result = engine.breed(coder_path, researcher_path, output_path)
        assert result.strategy == "section-wise"

    def test_dimension_wise_strategy(self, coder_path, researcher_path, output_path):
        config = MendelConfig(strategy="dimension-wise")
        engine = MendelEngine(config)
        result = engine.breed(coder_path, researcher_path, output_path)
        assert result.strategy == "dimension-wise"

    def test_no_mutation(self, coder_path, researcher_path, output_path):
        config = MendelConfig(enable_mutation=False)
        engine = MendelEngine(config)
        result = engine.breed(coder_path, researcher_path, output_path)
        assert result.mutations_applied == []

    def test_writes_output_on_keep(self, coder_path, researcher_path, output_path):
        engine = MendelEngine()
        result = engine.breed(coder_path, researcher_path, output_path)

        if result.succeeded:
            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0
            assert "---" in content  # has frontmatter

    def test_score_delta_correct(self, coder_path, researcher_path, output_path):
        engine = MendelEngine()
        result = engine.breed(coder_path, researcher_path, output_path)

        better_parent = max(result.parent_a.total_score, result.parent_b.total_score)
        expected_delta = result.offspring.total_score - better_parent
        assert abs(result.score_delta - expected_delta) < 0.01


class TestMergeDescriptions:
    def test_combines_triggers(self):
        desc_a = 'Write code. Triggers: "code", "python"'
        desc_b = 'Research topics. Triggers: "research", "analyze"'
        merged = _merge_descriptions(desc_a, desc_b)
        assert len(merged) > 0
        assert len(merged) <= 1024

    def test_handles_empty(self):
        assert _merge_descriptions("", "hello") == "hello"
        assert _merge_descriptions("hello", "") == "hello"

    def test_respects_length_limit(self):
        long_a = "A" * 800
        long_b = "B" * 800
        merged = _merge_descriptions(long_a, long_b)
        assert len(merged) <= 1024


class TestTSVFormatting:
    def test_header_has_columns(self):
        cols = TSV_HEADER.split("\t")
        assert "timestamp" in cols
        assert "event" in cols
        assert "skill" in cols
        assert "total" in cols
        assert "status" in cols

    def test_format_row(self, coder_path):
        doc = parse_skill_md(coder_path)
        result = evaluate_full(doc)
        row = format_tsv_row("baseline", "coder-skill", result)
        parts = row.split("\t")
        assert len(parts) == len(TSV_HEADER.split("\t"))
        assert "baseline" in row
        assert "coder-skill" in row


class TestBreedingSameParent:
    def test_breed_same_skill(self, coder_path, output_path):
        """Breeding a skill with itself should produce similar offspring."""
        engine = MendelEngine()
        result = engine.breed(coder_path, coder_path, output_path)
        # Offspring should be similar to parent
        assert abs(result.score_delta) < 15.0
