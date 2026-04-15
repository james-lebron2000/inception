---
name: mendel
description: "通过agent交配实现skill进化和代码优化。将两个parent skill交叉重组产生offspring，用8维评分体系评估适应度，仅保留改进。触发词：孟德尔、mendel、skill交配、skill mating、breed skills、进化skill、evolve skills"
---

# Mendel Skill — Agent Mating for Skill Evolution

> Inspired by Gregor Mendel's laws of inheritance. Where Darwin optimizes a single
> skill through hill-climbing, Mendel breeds two parent skills to produce offspring
> that inherit the best traits from both. Keep only improvements. Discard regressions.

---

## Design Philosophy

1. **Two Parents, One Offspring** — Each breeding takes two SKILL.md files as parents,
   crosses their sections to produce a hybrid offspring skill
2. **8-Dimension Fitness** — Structure (60pts) + Effectiveness (40pts), same rubric as
   Darwin. Offspring must beat the better parent to survive
3. **Git Ratchet** — Every edit is a git commit. Score down = `git revert HEAD`. No decay
4. **Independent Evaluation** — The agent that breeds NEVER scores the offspring.
   Spawn a fresh sub-agent for evaluation
5. **Mendelian Inheritance** — Dominant traits (high-scoring dimensions) are preferentially
   inherited; recessive traits (low-scoring) are candidates for crossover from the other parent

---

## 8-Dimension Evaluation Rubric (Total: 100)

### Structure Dimensions (60 pts)

| # | Dimension | Weight | Criteria |
|---|-----------|--------|----------|
| 1 | Frontmatter Quality | 8 | Name convention, description has what+when+triggers, ≤1024 chars |
| 2 | Workflow Clarity | 15 | Steps are numbered, explicit, executable, clear I/O per step |
| 3 | Boundary Coverage | 10 | Exception handling, fallback paths, error recovery |
| 4 | Checkpoint Design | 7 | Critical decisions pause for user confirmation |
| 5 | Instruction Specificity | 15 | Concrete parameters, formats, examples; not vague |
| 6 | Resource Integration | 5 | Referenced scripts/assets exist and are correct |

### Effectiveness Dimensions (40 pts)

| # | Dimension | Weight | Criteria |
|---|-----------|--------|----------|
| 7 | Overall Architecture | 15 | Hierarchy clarity, no redundancy, aligned with ecosystem |
| 8 | Live Performance | 25 | Execute 2-3 test prompts; output quality vs baseline |

### Scoring

Each dimension: score 1-10, multiply by weight, sum, divide by 10.

```
total = (d1×8 + d2×15 + d3×10 + d4×7 + d5×15 + d6×5 + d7×15 + d8×25) / 10
```

---

## Mendel Lifecycle

### Phase M0: Select Parents

1. User provides two parent skills (SKILL.md paths or agent genome references)
2. If not provided, auto-select the two highest-fitness skills from the local pool
3. Create git branch: `mendel/YYYYMMDD-HHMM`
4. Initialize tracking file `mendel-results.tsv`

**Output:** Two parent skill files loaded and displayed

### Phase M1: Baseline Evaluation

1. Score Parent A across all 8 dimensions
2. Score Parent B across all 8 dimensions
3. Record baselines to `mendel-results.tsv`
4. Identify each parent's strengths (dimensions scoring > 7) and weaknesses (< 5)

**Pause:** Show parent scorecards side by side. Ask user to confirm mating.

### Phase M2: Design Test Prompts

For the offspring skill, design 2-3 test prompts that cover:
- Happy path (typical use case combining both parents' strengths)
- Edge case (scenario testing inherited boundary handling)
- Novel scenario (something neither parent handles alone)

Save to `test-prompts.json`:
```json
[
  {"id": 1, "prompt": "...", "expected": "..."},
  {"id": 2, "prompt": "...", "expected": "..."}
]
```

**Pause:** Show test prompts for user approval.

### Phase M3: Crossover (Breeding)

Select crossover strategy based on parent compatibility:

**Strategy A — Section-wise Crossover** (when parents have similar structure)
- For each major section (frontmatter, workflow, boundaries, checkpoints...):
  - Take the section from the parent scoring higher on the corresponding dimension
  - Stitch sections together into coherent offspring

**Strategy B — Dimension-wise Crossover** (when parents have complementary strengths)
- Identify dominant traits: Parent A's top 4 dimensions vs Parent B's top 4
- Inherit content that produces each parent's highest scores
- Merge into unified offspring

**Strategy C — LLM Semantic Crossover** (when parents are structurally different)
- Feed both parent SKILL.md files to LLM
- Prompt: "Create a new skill that combines the best traits of both parents.
  Inherit Parent A's strength in [dims] and Parent B's strength in [dims].
  The offspring must be a valid SKILL.md file."

After crossover:
```bash
git add SKILL.md
git commit -m "breed: {parent_a} × {parent_b} → {offspring_name}"
```

### Phase M4: Mutation (Optional)

Apply 0-2 targeted mutations to the offspring:
- **Regulatory mutation:** Adjust frontmatter triggers/description
- **Point mutation:** Improve the lowest-scoring dimension with a specific fix
- **Insertion mutation:** Add a missing section (e.g., error handling)

Each mutation is a separate git commit:
```bash
git commit -m "mutate {offspring}: {mutation_type} - {description}"
```

### Phase M5: Evaluate Offspring

**Critical: Spawn INDEPENDENT sub-agent for evaluation.**

1. Structure scoring (dimensions 1-7): Fresh agent reads offspring SKILL.md
2. Effectiveness scoring (dimension 8): Execute test prompts with offspring skill
   - Run same prompts with each parent for comparison
   - Score offspring vs parents

Record to `mendel-results.tsv`

### Phase M6: Natural Selection

```
if offspring_score > max(parent_a_score, parent_b_score):
    status = "keep"         # Offspring survives
    → Update pool with new skill
    → Show breeding success card
else:
    status = "revert"       # Offspring dies
    → git revert HEAD       # Revert mutation commits
    → git revert HEAD       # Revert crossover commit
    → Show breeding failure card
```

**Pause:** Show offspring scorecard vs parents. Ask user to confirm.

### Phase M7: Iterate or Report

If breeding succeeded:
- Ask: "Breed again with another partner? Or optimize offspring with Darwin?"
- Can chain: Mendel breeding → Darwin hill-climbing for fine-tuning

If breeding failed:
- Suggest alternative parent pairing
- Or switch to Darwin single-skill optimization

### Phase M8: Lineage Report

Generate visual report:
- Ancestry tree (Parent A + Parent B → Offspring)
- Score comparison table (all 8 dimensions, before/after)
- Inherited traits map (which dimension came from which parent)
- Git history of the breeding

---

## Results Tracking

Append-only TSV file: `mendel-results.tsv`

```tsv
timestamp	event	skill	parent_a	parent_b	d1	d2	d3	d4	d5	d6	d7	d8	total	status	strategy	note
2026-04-15T10:00	baseline	coder-skill	-	-	8	7	6	5	8	4	7	6	67.5	baseline	-	Initial
2026-04-15T10:01	baseline	research-skill	-	-	6	8	7	7	6	5	8	7	70.0	baseline	-	Initial
2026-04-15T10:05	breed	coder-research-v1	coder-skill	research-skill	8	8	7	6	8	5	8	7	74.0	keep	section-wise	Inherited workflow from B
2026-04-15T10:10	mutate	coder-research-v1	-	-	8	8	7	7	8	5	8	7	75.0	keep	point	Improved checkpoints
```

---

## Crossover Rules

1. **Offspring must be a valid SKILL.md** — Proper frontmatter, coherent workflow
2. **No Frankenstein monsters** — Sections must flow logically, not just concatenated
3. **Respect both parents' intent** — Don't change the core purpose of either parent
4. **Size constraint** — Offspring ≤ 130% of larger parent's size
5. **One breeding per iteration** — Evaluate before breeding again
6. **Git everything** — Every crossover and mutation is a separate commit
7. **Revert, don't delete** — Use `git revert`, preserve full history

---

## Integration with Inception

This skill bridges the SKILL.md world with Inception's genetic engine:

- **SKILL.md → Genome:** Parse skill sections into chromosomes and genes
- **8-dim rubric → FitnessScore:** Map rubric scores to Inception's fitness model
- **Crossover strategies → CrossoverStrategy:** Section-wise, dimension-wise, semantic
- **Git ratchet → Selection:** Only keep improvements, auto-revert regressions
- **mendel-results.tsv → EvolutionHistory:** Track lineage across generations

---

## Example Usage

```bash
# Breed two skills
inception mendel \
    --parent-a skills/coder.md \
    --parent-b skills/researcher.md \
    --strategy section-wise \
    --output offspring.md

# Auto-select best parents from pool and breed
inception mendel --auto-select --pool ./skills/ --output offspring.md

# Breed then optimize (Mendel + Darwin pipeline)
inception mendel --parent-a a.md --parent-b b.md --output offspring.md
inception darwin --skill offspring.md --rounds 3
```
