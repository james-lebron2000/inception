"""Mendel mating engine — breed two SKILL.md parents into an optimized offspring.

Implements the Mendel Skill lifecycle:
1. Parse parent SKILL.md files
2. Evaluate baseline fitness (8-dimension rubric)
3. Crossover: section-wise, dimension-wise, or LLM semantic
4. Optional mutation
5. Evaluate offspring
6. Natural selection: keep only if offspring beats better parent

Integrates with Inception's genome model and darwin-skill's ratchet mechanism.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from inception.skills.rubric import (
    DIMENSIONS,
    RubricResult,
    SkillDocument,
    evaluate_full,
    parse_skill_md,
)


@dataclass
class BreedingResult:
    """Result of a Mendel breeding attempt."""

    parent_a: RubricResult
    parent_b: RubricResult
    offspring: RubricResult
    offspring_doc: SkillDocument
    strategy: str
    status: str  # "keep" or "revert"
    score_delta: float
    inherited_from: dict[str, str] = field(default_factory=dict)
    mutations_applied: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def succeeded(self) -> bool:
        return self.status == "keep"


@dataclass
class MendelConfig:
    """Configuration for Mendel breeding."""

    strategy: str = "auto"  # "section-wise", "dimension-wise", "semantic", "auto"
    enable_mutation: bool = True
    max_mutations: int = 2
    size_limit_ratio: float = 1.3  # offspring <= 130% of larger parent
    min_improvement: float = 0.0  # must strictly beat better parent


class MendelEngine:
    """Mendel mating engine for SKILL.md breeding."""

    def __init__(self, config: MendelConfig | None = None):
        self.config = config or MendelConfig()

    def breed(
        self,
        parent_a_path: str | Path,
        parent_b_path: str | Path,
        output_path: str | Path | None = None,
    ) -> BreedingResult:
        """Breed two parent skills to produce offspring.

        1. Parse parents
        2. Evaluate baselines
        3. Select crossover strategy
        4. Perform crossover
        5. Optional mutation
        6. Evaluate offspring
        7. Natural selection
        """
        # Parse parents
        doc_a = parse_skill_md(parent_a_path)
        doc_b = parse_skill_md(parent_b_path)

        # Evaluate baselines
        result_a = evaluate_full(doc_a)
        result_b = evaluate_full(doc_b)

        # Select strategy
        strategy = self._select_strategy(result_a, result_b)

        # Crossover
        offspring_doc, inherited = self._crossover(doc_a, doc_b, result_a, result_b, strategy)

        # Mutation
        mutations: list[str] = []
        if self.config.enable_mutation:
            offspring_doc, mutations = self._mutate(offspring_doc, result_a, result_b)

        # Enforce size limit
        self._enforce_size_limit(offspring_doc, doc_a, doc_b)

        # Evaluate offspring
        offspring_result = evaluate_full(offspring_doc)

        # Natural selection
        better_parent_score = max(result_a.total_score, result_b.total_score)
        offspring_score = offspring_result.total_score
        score_delta = offspring_score - better_parent_score

        if score_delta > self.config.min_improvement:
            status = "keep"
        else:
            status = "revert"

        # Write output if keeping
        if status == "keep" and output_path:
            self._write_offspring(offspring_doc, Path(output_path))

        return BreedingResult(
            parent_a=result_a,
            parent_b=result_b,
            offspring=offspring_result,
            offspring_doc=offspring_doc,
            strategy=strategy,
            status=status,
            score_delta=score_delta,
            inherited_from=inherited,
            mutations_applied=mutations,
        )

    def _select_strategy(self, result_a: RubricResult, result_b: RubricResult) -> str:
        """Auto-select crossover strategy based on parent characteristics."""
        if self.config.strategy != "auto":
            return self.config.strategy

        # Count how many dimensions each parent wins
        a_wins = 0
        b_wins = 0
        for dim in DIMENSIONS:
            score_a = result_a.dimensions.get(dim)
            score_b = result_b.dimensions.get(dim)
            if score_a and score_b:
                if score_a.raw_score > score_b.raw_score:
                    a_wins += 1
                elif score_b.raw_score > score_a.raw_score:
                    b_wins += 1

        # If one parent dominates most dimensions, use section-wise
        if a_wins >= 6 or b_wins >= 6:
            return "section-wise"

        # If parents have complementary strengths, use dimension-wise
        if abs(a_wins - b_wins) <= 2:
            return "dimension-wise"

        return "section-wise"

    def _crossover(
        self,
        doc_a: SkillDocument,
        doc_b: SkillDocument,
        result_a: RubricResult,
        result_b: RubricResult,
        strategy: str,
    ) -> tuple[SkillDocument, dict[str, str]]:
        """Perform crossover between two parent skills."""
        if strategy == "dimension-wise":
            return self._dimension_wise_crossover(doc_a, doc_b, result_a, result_b)
        else:
            return self._section_wise_crossover(doc_a, doc_b, result_a, result_b)

    def _section_wise_crossover(
        self,
        doc_a: SkillDocument,
        doc_b: SkillDocument,
        result_a: RubricResult,
        result_b: RubricResult,
    ) -> tuple[SkillDocument, dict[str, str]]:
        """Take each section from the parent scoring higher on the relevant dimension."""
        inherited: dict[str, str] = {}

        # Merge frontmatter — take from overall better parent, combine triggers
        if result_a.total_score >= result_b.total_score:
            fm = dict(doc_a.frontmatter)
            inherited["frontmatter"] = "parent_a"
        else:
            fm = dict(doc_b.frontmatter)
            inherited["frontmatter"] = "parent_b"

        # Name the offspring
        name_a = doc_a.name or "a"
        name_b = doc_b.name or "b"
        offspring_name = f"{name_a[:8]}x{name_b[:8]}"
        fm["name"] = offspring_name

        # Merge descriptions (combine trigger words from both)
        desc_a = doc_a.description or ""
        desc_b = doc_b.description or ""
        fm["description"] = _merge_descriptions(desc_a, desc_b)

        # Merge sections — pick best from each parent per dimension mapping
        dim_to_section_keywords = {
            "workflow_clarity": ["workflow", "lifecycle", "phase", "step", "process"],
            "boundary_coverage": ["boundary", "error", "constraint", "rule", "limit"],
            "checkpoint_design": ["checkpoint", "pause", "confirm", "human"],
            "instruction_specificity": ["usage", "example", "format", "integration"],
            "overall_architecture": ["architecture", "design", "philosophy", "overview"],
        }

        merged_sections: dict[str, str] = {}
        all_section_names = set(doc_a.sections.keys()) | set(doc_b.sections.keys())

        for section_name in all_section_names:
            # Determine which dimension this section maps to
            best_dim = _match_section_to_dimension(section_name, dim_to_section_keywords)

            if best_dim:
                score_a = result_a.dimensions.get(best_dim)
                score_b = result_b.dimensions.get(best_dim)
                if score_a and score_b and score_b.raw_score > score_a.raw_score:
                    source = "parent_b"
                else:
                    source = "parent_a"
            else:
                # Default: take from whoever has this section, prefer A
                source = "parent_a" if section_name in doc_a.sections else "parent_b"

            if source == "parent_a" and section_name in doc_a.sections:
                merged_sections[section_name] = doc_a.sections[section_name]
            elif source == "parent_b" and section_name in doc_b.sections:
                merged_sections[section_name] = doc_b.sections[section_name]
            elif section_name in doc_a.sections:
                merged_sections[section_name] = doc_a.sections[section_name]
                source = "parent_a"
            elif section_name in doc_b.sections:
                merged_sections[section_name] = doc_b.sections[section_name]
                source = "parent_b"

            inherited[section_name] = source

        # Assemble offspring
        offspring = _assemble_skill_doc(offspring_name, fm, merged_sections)
        return offspring, inherited

    def _dimension_wise_crossover(
        self,
        doc_a: SkillDocument,
        doc_b: SkillDocument,
        result_a: RubricResult,
        result_b: RubricResult,
    ) -> tuple[SkillDocument, dict[str, str]]:
        """Inherit content contributing to each parent's strongest dimensions."""
        inherited: dict[str, str] = {}

        # Start with parent A as base, overlay parent B's stronger sections
        merged_sections = dict(doc_a.sections)
        for section_name in doc_a.sections:
            inherited[section_name] = "parent_a"

        # For sections in B but not A, add them
        for section_name, content in doc_b.sections.items():
            if section_name not in merged_sections:
                merged_sections[section_name] = content
                inherited[section_name] = "parent_b"

        # For shared sections, pick from dimension-winner
        for section_name in set(doc_a.sections.keys()) & set(doc_b.sections.keys()):
            dim = _match_section_to_dimension(section_name, {
                "workflow_clarity": ["workflow", "lifecycle", "phase", "step"],
                "boundary_coverage": ["boundary", "error", "constraint", "rule"],
                "checkpoint_design": ["checkpoint", "pause", "confirm"],
                "instruction_specificity": ["usage", "example", "format"],
            })
            if dim:
                sa = result_a.dimensions.get(dim)
                sb = result_b.dimensions.get(dim)
                if sa and sb and sb.raw_score > sa.raw_score:
                    merged_sections[section_name] = doc_b.sections[section_name]
                    inherited[section_name] = "parent_b"

        # Name offspring
        name_a = doc_a.name or "a"
        name_b = doc_b.name or "b"
        offspring_name = f"{name_a[:8]}x{name_b[:8]}"

        fm = dict(doc_a.frontmatter)
        fm["name"] = offspring_name
        fm["description"] = _merge_descriptions(doc_a.description, doc_b.description)
        inherited["frontmatter"] = "mixed"

        offspring = _assemble_skill_doc(offspring_name, fm, merged_sections)
        return offspring, inherited

    def _mutate(
        self, doc: SkillDocument, result_a: RubricResult, result_b: RubricResult
    ) -> tuple[SkillDocument, list[str]]:
        """Apply targeted mutations to improve weakest areas."""
        mutations: list[str] = []
        result = evaluate_full(doc)
        weakest = result.weakest_dimension()

        if not weakest:
            return doc, mutations

        # Only mutate if the weakest dimension is below 6
        weak_score = result.dimensions[weakest].raw_score if weakest in result.dimensions else 5.0
        if weak_score >= 6.0:
            return doc, mutations

        # Apply a regulatory mutation to frontmatter if it's weak
        if weakest == "frontmatter_quality" and doc.description:
            if len(doc.description) > 1024:
                doc.description = doc.description[:1020] + "..."
                doc.frontmatter["description"] = doc.description
                mutations.append("regulatory: trimmed description to <=1024 chars")

        # Apply a checkpoint mutation if checkpoints are weak
        if weakest == "checkpoint_design":
            if "Pause" not in doc.content and "pause" not in doc.content:
                # Add a pause note to the content
                doc.content += "\n\n**Pause:** Review results before proceeding.\n"
                mutations.append("insertion: added checkpoint pause marker")

        # Apply boundary mutation if boundaries are weak
        if weakest == "boundary_coverage":
            if "fallback" not in doc.content.lower() and "error" not in doc.content.lower():
                doc.content += "\n\n## Error Handling\n\nIf any step fails, revert and log the error before proceeding.\n"
                mutations.append("insertion: added error handling section")

        # Rebuild raw_text after mutations
        if mutations:
            doc = _rebuild_raw_text(doc)

        return doc, mutations

    def _enforce_size_limit(
        self, offspring: SkillDocument, doc_a: SkillDocument, doc_b: SkillDocument
    ) -> None:
        """Ensure offspring doesn't exceed size limit."""
        max_parent_size = max(len(doc_a.raw_text), len(doc_b.raw_text))
        limit = int(max_parent_size * self.config.size_limit_ratio)

        if len(offspring.raw_text) > limit:
            # Truncate content to fit (rough approach — better to trim sections)
            overflow = len(offspring.raw_text) - limit
            offspring.content = offspring.content[:len(offspring.content) - overflow]
            offspring = _rebuild_raw_text(offspring)

    def _write_offspring(self, doc: SkillDocument, path: Path) -> None:
        """Write offspring SKILL.md to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.raw_text, encoding="utf-8")


def format_breeding_report(result: BreedingResult) -> str:
    """Format a breeding result as a human-readable report."""
    lines = [
        "=" * 60,
        "  Mendel Breeding Report",
        "=" * 60,
        "",
        f"  Parent A: {result.parent_a.skill_name}  (score: {result.parent_a.total_score:.1f})",
        f"  Parent B: {result.parent_b.skill_name}  (score: {result.parent_b.total_score:.1f})",
        f"  Strategy: {result.strategy}",
        "",
        f"  Offspring: {result.offspring.skill_name}  (score: {result.offspring.total_score:.1f})",
        f"  Status: {result.status.upper()}  (delta: {result.score_delta:+.1f})",
        "",
        "  Dimension Scores:",
        f"  {'Dimension':<25s} {'Parent A':>10s} {'Parent B':>10s} {'Offspring':>10s} {'From':>10s}",
        "  " + "-" * 65,
    ]

    ordered_dims = [
        "frontmatter_quality", "workflow_clarity", "boundary_coverage",
        "checkpoint_design", "instruction_specificity", "resource_integration",
        "overall_architecture", "live_performance",
    ]

    for dim in ordered_dims:
        da = result.parent_a.dimensions.get(dim)
        db = result.parent_b.dimensions.get(dim)
        do = result.offspring.dimensions.get(dim)
        source = result.inherited_from.get(dim, "?")
        lines.append(
            f"  {dim:<25s} "
            f"{da.raw_score:>10.1f}" if da else f"{'?':>10s}"
            + f" {db.raw_score:>10.1f}" if db else f" {'?':>10s}"
            + f" {do.raw_score:>10.1f}" if do else f" {'?':>10s}"
            + f" {source:>10s}"
        )

    if result.mutations_applied:
        lines.append("\n  Mutations Applied:")
        for m in result.mutations_applied:
            lines.append(f"    - {m}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_tsv_row(
    event: str,
    skill_name: str,
    result: RubricResult,
    parent_a: str = "-",
    parent_b: str = "-",
    status: str = "baseline",
    strategy: str = "-",
    note: str = "",
) -> str:
    """Format a single TSV row for mendel-results.tsv."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    scores = result.to_tsv_scores()
    total = f"{result.total_score:.1f}"
    return f"{ts}\t{event}\t{skill_name}\t{parent_a}\t{parent_b}\t{scores}\t{total}\t{status}\t{strategy}\t{note}"


TSV_HEADER = "timestamp\tevent\tskill\tparent_a\tparent_b\td1\td2\td3\td4\td5\td6\td7\td8\ttotal\tstatus\tstrategy\tnote"


# ─── Helper Functions ──────────────────────────────────────────────────────


def _merge_descriptions(desc_a: str, desc_b: str) -> str:
    """Merge two skill descriptions, combining trigger words."""
    if not desc_a:
        return desc_b
    if not desc_b:
        return desc_a

    # Extract trigger words from both
    triggers_a = set(re.findall(r'"([^"]+)"', desc_a))
    triggers_b = set(re.findall(r'"([^"]+)"', desc_b))
    all_triggers = triggers_a | triggers_b

    # Take the shorter core description
    core_a = re.sub(r'触发词.*$', '', desc_a).strip().rstrip('。，,.')
    core_b = re.sub(r'触发词.*$', '', desc_b).strip().rstrip('。，,.')

    core = core_a if len(core_a) <= len(core_b) else core_b
    if not core:
        core = desc_a[:100]

    if all_triggers:
        trigger_str = ", ".join(f'"{t}"' for t in sorted(all_triggers)[:8])
        result = f"{core}. Triggers: {trigger_str}"
    else:
        result = core

    return result[:1024]


def _match_section_to_dimension(
    section_name: str, mapping: dict[str, list[str]]
) -> str | None:
    """Match a section heading to a rubric dimension based on keywords."""
    name_lower = section_name.lower()
    for dim, keywords in mapping.items():
        if any(kw in name_lower for kw in keywords):
            return dim
    return None


def _assemble_skill_doc(
    name: str, frontmatter: dict[str, str], sections: dict[str, str]
) -> SkillDocument:
    """Assemble a SkillDocument from components."""
    # Build frontmatter
    fm_lines = ["---"]
    for key, value in frontmatter.items():
        if "\n" in str(value):
            fm_lines.append(f'{key}: "{value}"')
        else:
            fm_lines.append(f"{key}: {value}")
    fm_lines.append("---")
    fm_text = "\n".join(fm_lines)

    # Build content from sections
    content_parts = []
    for heading, body in sections.items():
        if heading == "__intro__":
            content_parts.append(body)
        else:
            content_parts.append(f"\n## {heading}\n\n{body}")

    content = "\n".join(content_parts)
    raw_text = f"{fm_text}\n\n{content}\n"

    doc = SkillDocument(
        path="",
        name=name,
        description=frontmatter.get("description", ""),
        content=content,
        frontmatter=frontmatter,
        sections=sections,
        raw_text=raw_text,
    )
    return doc


def _rebuild_raw_text(doc: SkillDocument) -> SkillDocument:
    """Rebuild raw_text from frontmatter + content."""
    fm_lines = ["---"]
    for key, value in doc.frontmatter.items():
        fm_lines.append(f"{key}: {value}")
    fm_lines.append("---")
    doc.raw_text = "\n".join(fm_lines) + "\n\n" + doc.content + "\n"
    return doc
