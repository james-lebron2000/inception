"""8-dimension evaluation rubric for SKILL.md files.

Adapted from darwin-skill's dual evaluation system:
- Structure (60 pts): frontmatter, workflow, boundaries, checkpoints, specificity, resources
- Effectiveness (40 pts): architecture, live performance

Each dimension is scored 1-10, multiplied by weight, summed, divided by 10.
Total range: 0-100.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Dimension weights (must sum to 100)
DIMENSIONS = {
    "frontmatter_quality": 8,
    "workflow_clarity": 15,
    "boundary_coverage": 10,
    "checkpoint_design": 7,
    "instruction_specificity": 15,
    "resource_integration": 5,
    "overall_architecture": 15,
    "live_performance": 25,
}

DIMENSION_NAMES = {
    "frontmatter_quality": "Frontmatter Quality",
    "workflow_clarity": "Workflow Clarity",
    "boundary_coverage": "Boundary Coverage",
    "checkpoint_design": "Checkpoint Design",
    "instruction_specificity": "Instruction Specificity",
    "resource_integration": "Resource Integration",
    "overall_architecture": "Overall Architecture",
    "live_performance": "Live Performance",
}


@dataclass
class DimensionScore:
    """Score for a single rubric dimension."""

    name: str
    raw_score: float  # 1-10
    weight: int
    reasoning: str = ""

    @property
    def weighted_score(self) -> float:
        return self.raw_score * self.weight

    @property
    def display_name(self) -> str:
        return DIMENSION_NAMES.get(self.name, self.name)


@dataclass
class RubricResult:
    """Complete rubric evaluation result."""

    skill_name: str
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    eval_mode: str = "structure_only"  # "structure_only", "full_test", "dry_run"
    notes: str = ""

    @property
    def total_score(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(d.weighted_score for d in self.dimensions.values()) / 10.0

    @property
    def structure_score(self) -> float:
        structure_dims = [
            "frontmatter_quality", "workflow_clarity", "boundary_coverage",
            "checkpoint_design", "instruction_specificity", "resource_integration",
        ]
        return sum(
            self.dimensions[d].weighted_score
            for d in structure_dims if d in self.dimensions
        ) / 10.0

    @property
    def effectiveness_score(self) -> float:
        eff_dims = ["overall_architecture", "live_performance"]
        return sum(
            self.dimensions[d].weighted_score
            for d in eff_dims if d in self.dimensions
        ) / 10.0

    def weakest_dimension(self) -> str | None:
        if not self.dimensions:
            return None
        return min(self.dimensions, key=lambda d: self.dimensions[d].raw_score)

    def strongest_dimension(self) -> str | None:
        if not self.dimensions:
            return None
        return max(self.dimensions, key=lambda d: self.dimensions[d].raw_score)

    def strengths(self, threshold: float = 7.0) -> list[str]:
        return [d for d, s in self.dimensions.items() if s.raw_score >= threshold]

    def weaknesses(self, threshold: float = 5.0) -> list[str]:
        return [d for d, s in self.dimensions.items() if s.raw_score < threshold]

    def to_tsv_scores(self) -> str:
        ordered = [
            "frontmatter_quality", "workflow_clarity", "boundary_coverage",
            "checkpoint_design", "instruction_specificity", "resource_integration",
            "overall_architecture", "live_performance",
        ]
        scores = [str(self.dimensions[d].raw_score) if d in self.dimensions else "0" for d in ordered]
        return "\t".join(scores)


@dataclass
class SkillDocument:
    """A parsed SKILL.md file."""

    path: str
    name: str = ""
    description: str = ""
    content: str = ""
    frontmatter: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


def parse_skill_md(path: str | Path) -> SkillDocument:
    """Parse a SKILL.md file into a structured SkillDocument."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")

    doc = SkillDocument(path=str(path), raw_text=raw_text)

    # Parse frontmatter (YAML between --- delimiters)
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw_text, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        for line in fm_text.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                doc.frontmatter[key.strip()] = value.strip().strip('"').strip("'")
        doc.name = doc.frontmatter.get("name", "")
        doc.description = doc.frontmatter.get("description", "")
        doc.content = raw_text[fm_match.end():]
    else:
        doc.content = raw_text

    # Parse sections by heading
    current_heading = "__intro__"
    current_lines: list[str] = []

    for line in doc.content.split("\n"):
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            if current_lines:
                doc.sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        doc.sections[current_heading] = "\n".join(current_lines).strip()

    return doc


def evaluate_structure(doc: SkillDocument) -> dict[str, DimensionScore]:
    """Evaluate the 6 structural dimensions of a SKILL.md.

    This is a static analysis — no LLM or execution required.
    """
    scores: dict[str, DimensionScore] = {}

    # D1: Frontmatter Quality (weight 8)
    fm_score = _score_frontmatter(doc)
    scores["frontmatter_quality"] = DimensionScore(
        name="frontmatter_quality", raw_score=fm_score, weight=8,
        reasoning=_frontmatter_reasoning(doc),
    )

    # D2: Workflow Clarity (weight 15)
    wf_score = _score_workflow(doc)
    scores["workflow_clarity"] = DimensionScore(
        name="workflow_clarity", raw_score=wf_score, weight=15,
        reasoning=_workflow_reasoning(doc),
    )

    # D3: Boundary Coverage (weight 10)
    bc_score = _score_boundaries(doc)
    scores["boundary_coverage"] = DimensionScore(
        name="boundary_coverage", raw_score=bc_score, weight=10,
        reasoning=_boundary_reasoning(doc),
    )

    # D4: Checkpoint Design (weight 7)
    cp_score = _score_checkpoints(doc)
    scores["checkpoint_design"] = DimensionScore(
        name="checkpoint_design", raw_score=cp_score, weight=7,
        reasoning=_checkpoint_reasoning(doc),
    )

    # D5: Instruction Specificity (weight 15)
    is_score = _score_specificity(doc)
    scores["instruction_specificity"] = DimensionScore(
        name="instruction_specificity", raw_score=is_score, weight=15,
        reasoning=_specificity_reasoning(doc),
    )

    # D6: Resource Integration (weight 5)
    ri_score = _score_resources(doc)
    scores["resource_integration"] = DimensionScore(
        name="resource_integration", raw_score=ri_score, weight=5,
        reasoning=_resource_reasoning(doc),
    )

    return scores


def evaluate_full(doc: SkillDocument) -> RubricResult:
    """Full structural evaluation (dimensions 1-6).

    Dimensions 7-8 (architecture + live performance) require LLM
    or sub-agent evaluation and are set to a default of 5.0.
    Use evaluate_with_llm() for full 8-dimension scoring.
    """
    scores = evaluate_structure(doc)

    # Default scores for effectiveness dimensions (require LLM)
    scores["overall_architecture"] = DimensionScore(
        name="overall_architecture", raw_score=5.0, weight=15,
        reasoning="Default score — requires LLM evaluation for accurate scoring",
    )
    scores["live_performance"] = DimensionScore(
        name="live_performance", raw_score=5.0, weight=25,
        reasoning="Default score — requires live test execution for accurate scoring",
    )

    return RubricResult(
        skill_name=doc.name or doc.path,
        dimensions=scores,
        eval_mode="structure_only",
    )


# ─── Scoring Functions ──────────────────────────────────────────────────────

def _score_frontmatter(doc: SkillDocument) -> float:
    score = 3.0  # base

    if doc.frontmatter:
        score += 1.5
    if doc.name:
        score += 1.0
        if re.match(r"^[a-z][a-z0-9-]*$", doc.name):
            score += 0.5  # kebab-case
    if doc.description:
        score += 1.5
        if len(doc.description) <= 1024:
            score += 0.5
        # Check for trigger words
        triggers = re.findall(r'"[^"]+"', doc.description)
        if triggers:
            score += 1.0
    if not doc.frontmatter:
        score = 2.0

    return min(10.0, max(1.0, score))


def _score_workflow(doc: SkillDocument) -> float:
    content = doc.content
    score = 3.0

    # Check for numbered steps
    numbered_steps = re.findall(r"^\d+\.\s", content, re.MULTILINE)
    if len(numbered_steps) >= 3:
        score += 2.0
    elif len(numbered_steps) >= 1:
        score += 1.0

    # Check for phase/step headings
    phase_headings = re.findall(r"#{1,3}\s+(?:Phase|Step|Stage)\s", content, re.IGNORECASE)
    if phase_headings:
        score += 1.5

    # Check for input/output markers
    io_markers = len(re.findall(r"\b(?:input|output|returns?|produces?)\b", content, re.IGNORECASE))
    if io_markers >= 3:
        score += 1.5
    elif io_markers >= 1:
        score += 0.5

    # Check for code blocks (explicit examples)
    code_blocks = len(re.findall(r"```", content))
    if code_blocks >= 4:
        score += 1.0
    elif code_blocks >= 2:
        score += 0.5

    return min(10.0, max(1.0, score))


def _score_boundaries(doc: SkillDocument) -> float:
    content = doc.content
    score = 3.0

    # Error handling keywords
    error_kw = len(re.findall(
        r"\b(?:error|exception|fail|fallback|edge case|boundary|if.*fails?|otherwise)\b",
        content, re.IGNORECASE,
    ))
    if error_kw >= 5:
        score += 3.0
    elif error_kw >= 2:
        score += 1.5

    # Conditional logic
    conditionals = len(re.findall(r"\b(?:if|else|when|unless|otherwise)\b", content, re.IGNORECASE))
    if conditionals >= 5:
        score += 2.0
    elif conditionals >= 2:
        score += 1.0

    # Constraint rules
    constraint_kw = len(re.findall(
        r"\b(?:must|never|always|constraint|rule|limit|max|min)\b",
        content, re.IGNORECASE,
    ))
    if constraint_kw >= 3:
        score += 1.5

    return min(10.0, max(1.0, score))


def _score_checkpoints(doc: SkillDocument) -> float:
    content = doc.content
    score = 3.0

    # Pause/confirm keywords
    pause_kw = len(re.findall(
        r"\b(?:pause|confirm|approval|ask user|human|checkpoint|wait|review)\b",
        content, re.IGNORECASE,
    ))
    if pause_kw >= 4:
        score += 3.5
    elif pause_kw >= 2:
        score += 2.0
    elif pause_kw >= 1:
        score += 1.0

    # Critical decision markers
    decision_kw = len(re.findall(
        r"\b(?:critical|important|decision|choose|select)\b",
        content, re.IGNORECASE,
    ))
    if decision_kw >= 2:
        score += 1.5

    # Bold/emphasis markers for important points
    emphasis = len(re.findall(r"\*\*[^*]+\*\*", content))
    if emphasis >= 5:
        score += 1.0

    return min(10.0, max(1.0, score))


def _score_specificity(doc: SkillDocument) -> float:
    content = doc.content
    score = 3.0

    # Concrete examples (code blocks)
    code_blocks = len(re.findall(r"```[\s\S]*?```", content))
    if code_blocks >= 4:
        score += 2.0
    elif code_blocks >= 2:
        score += 1.0

    # Specific formats/parameters
    format_kw = len(re.findall(
        r"\b(?:format|parameter|schema|json|yaml|path|file|directory)\b",
        content, re.IGNORECASE,
    ))
    if format_kw >= 5:
        score += 1.5
    elif format_kw >= 2:
        score += 0.5

    # Tables (structured data)
    tables = len(re.findall(r"\|.*\|.*\|", content))
    if tables >= 5:
        score += 1.5
    elif tables >= 2:
        score += 0.5

    # Inline code references
    inline_code = len(re.findall(r"`[^`]+`", content))
    if inline_code >= 10:
        score += 1.5
    elif inline_code >= 5:
        score += 0.5

    return min(10.0, max(1.0, score))


def _score_resources(doc: SkillDocument) -> float:
    content = doc.content
    score = 5.0  # Base is higher for this dimension (many skills don't need resources)

    # File path references
    paths = len(re.findall(r"[`\"][\w./\\-]+\.\w+[`\"]", content))
    if paths >= 3:
        score += 2.0
    elif paths >= 1:
        score += 1.0

    # URL references
    urls = len(re.findall(r"https?://", content))
    if urls >= 1:
        score += 1.0

    # Script references
    scripts = len(re.findall(r"\b(?:script|command|run|execute|install)\b", content, re.IGNORECASE))
    if scripts >= 2:
        score += 1.0

    return min(10.0, max(1.0, score))


# ─── Reasoning Generators ───────────────────────────────────────────────────

def _frontmatter_reasoning(doc: SkillDocument) -> str:
    parts = []
    if doc.frontmatter:
        parts.append("Has YAML frontmatter")
    else:
        parts.append("Missing YAML frontmatter")
    if doc.name:
        parts.append(f"name='{doc.name}'")
    if doc.description:
        parts.append(f"description length={len(doc.description)}")
    return "; ".join(parts)


def _workflow_reasoning(doc: SkillDocument) -> str:
    steps = len(re.findall(r"^\d+\.\s", doc.content, re.MULTILINE))
    phases = len(re.findall(r"#{1,3}\s+(?:Phase|Step|Stage)\s", doc.content, re.IGNORECASE))
    return f"{steps} numbered steps, {phases} phase headings"


def _boundary_reasoning(doc: SkillDocument) -> str:
    errors = len(re.findall(r"\b(?:error|fail|fallback)\b", doc.content, re.IGNORECASE))
    conds = len(re.findall(r"\b(?:if|else|when|unless)\b", doc.content, re.IGNORECASE))
    return f"{errors} error handling mentions, {conds} conditional keywords"


def _checkpoint_reasoning(doc: SkillDocument) -> str:
    pauses = len(re.findall(r"\b(?:pause|confirm|ask user|checkpoint)\b", doc.content, re.IGNORECASE))
    return f"{pauses} checkpoint/pause markers"


def _specificity_reasoning(doc: SkillDocument) -> str:
    codes = len(re.findall(r"```[\s\S]*?```", doc.content))
    tables = len(re.findall(r"\|.*\|.*\|", doc.content))
    return f"{codes} code blocks, {tables} table rows"


def _resource_reasoning(doc: SkillDocument) -> str:
    paths = len(re.findall(r"[`\"][\w./\\-]+\.\w+[`\"]", doc.content))
    return f"{paths} file path references"
