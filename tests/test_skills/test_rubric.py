"""Tests for the 8-dimension SKILL.md rubric evaluator."""

from __future__ import annotations

import pytest

from inception.skills.rubric import (
    DIMENSIONS,
    evaluate_full,
    evaluate_structure,
    parse_skill_md,
)


SAMPLE_SKILL = """---
name: sample-skill
description: "A sample skill for testing. Triggers: 'test', 'sample'"
---

# Sample Skill

> A test skill for rubric evaluation.

## Design Philosophy

1. Keep things simple
2. Be explicit
3. Handle errors gracefully

## Workflow

### Phase 1: Input

1. Read the user's input
2. Validate the format
3. Parse parameters

**Output:** Parsed parameters dict

### Phase 2: Process

1. Execute the main logic
2. If processing fails, log error and retry once
3. Otherwise, format the result

**Output:** Formatted result string

### Phase 3: Review

**Pause:** Ask user to confirm the result before saving.

1. Display result to user
2. Wait for confirmation
3. Save to `output.json`

## Boundary Conditions

- If input is empty, return error message
- If file not found, fallback to default config
- Never exceed 1000 items per batch
- Always validate before writing

## Examples

```python
result = run_skill(input="test data")
print(result)
```

```json
{"status": "ok", "output": "processed"}
```
"""

MINIMAL_SKILL = """---
name: minimal
description: "Minimal skill"
---

# Minimal

Do something.
"""


@pytest.fixture
def sample_skill_path(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(SAMPLE_SKILL)
    return path


@pytest.fixture
def minimal_skill_path(tmp_path):
    path = tmp_path / "minimal.md"
    path.write_text(MINIMAL_SKILL)
    return path


class TestParseSkillMd:
    def test_parses_frontmatter(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        assert doc.name == "sample-skill"
        assert "sample skill" in doc.description.lower()

    def test_parses_sections(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        assert len(doc.sections) > 0
        section_names = list(doc.sections.keys())
        assert any("Workflow" in s or "Phase" in s or "Design" in s for s in section_names)

    def test_content_not_empty(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        assert len(doc.content) > 100

    def test_minimal_skill(self, minimal_skill_path):
        doc = parse_skill_md(minimal_skill_path)
        assert doc.name == "minimal"
        assert "Minimal" in doc.content


class TestEvaluateStructure:
    def test_returns_6_dimensions(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        scores = evaluate_structure(doc)
        assert len(scores) == 6
        assert "frontmatter_quality" in scores
        assert "workflow_clarity" in scores
        assert "boundary_coverage" in scores
        assert "checkpoint_design" in scores
        assert "instruction_specificity" in scores
        assert "resource_integration" in scores

    def test_scores_in_valid_range(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        scores = evaluate_structure(doc)
        for dim, score in scores.items():
            assert 1.0 <= score.raw_score <= 10.0, f"{dim} score out of range: {score.raw_score}"

    def test_good_skill_scores_higher(self, sample_skill_path, minimal_skill_path):
        doc_good = parse_skill_md(sample_skill_path)
        doc_min = parse_skill_md(minimal_skill_path)
        scores_good = evaluate_structure(doc_good)
        scores_min = evaluate_structure(doc_min)

        good_total = sum(s.weighted_score for s in scores_good.values())
        min_total = sum(s.weighted_score for s in scores_min.values())
        assert good_total > min_total

    def test_weighted_scores_correct(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        scores = evaluate_structure(doc)
        for dim, score in scores.items():
            expected_weight = DIMENSIONS[dim]
            assert score.weight == expected_weight
            assert score.weighted_score == score.raw_score * expected_weight


class TestEvaluateFull:
    def test_returns_8_dimensions(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        result = evaluate_full(doc)
        assert len(result.dimensions) == 8
        assert "overall_architecture" in result.dimensions
        assert "live_performance" in result.dimensions

    def test_total_score_in_range(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        result = evaluate_full(doc)
        assert 0.0 < result.total_score <= 100.0

    def test_structure_and_effectiveness_sum(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        result = evaluate_full(doc)
        total = result.structure_score + result.effectiveness_score
        assert abs(total - result.total_score) < 0.01


class TestRubricResult:
    def test_weakest_strongest(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        result = evaluate_full(doc)
        weakest = result.weakest_dimension()
        strongest = result.strongest_dimension()
        assert weakest is not None
        assert strongest is not None
        assert result.dimensions[weakest].raw_score <= result.dimensions[strongest].raw_score

    def test_strengths_weaknesses(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        result = evaluate_full(doc)
        strengths = result.strengths(threshold=5.0)
        weaknesses = result.weaknesses(threshold=5.0)
        # Strengths and weaknesses should not overlap
        assert not set(strengths) & set(weaknesses)

    def test_tsv_scores(self, sample_skill_path):
        doc = parse_skill_md(sample_skill_path)
        result = evaluate_full(doc)
        tsv = result.to_tsv_scores()
        parts = tsv.split("\t")
        assert len(parts) == 8
        for p in parts:
            assert float(p) >= 0


class TestDimensionWeights:
    def test_weights_sum_to_100(self):
        assert sum(DIMENSIONS.values()) == 100

    def test_structure_weights_sum_to_60(self):
        structure = ["frontmatter_quality", "workflow_clarity", "boundary_coverage",
                     "checkpoint_design", "instruction_specificity", "resource_integration"]
        assert sum(DIMENSIONS[d] for d in structure) == 60

    def test_effectiveness_weights_sum_to_40(self):
        effectiveness = ["overall_architecture", "live_performance"]
        assert sum(DIMENSIONS[d] for d in effectiveness) == 40
