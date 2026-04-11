"""Prompt templates for LLM-driven crossover, mutation, and evaluation.

All prompts are Jinja2 templates that receive structured context
about parent agents, their code, and the desired operation.
"""

from __future__ import annotations

from jinja2 import Template

# ─── Crossover Prompts ───────────────────────────────────────────────────────

GENE_LEVEL_CROSSOVER = Template("""\
You are performing genetic crossover on two AI agent skill implementations.
Your goal is to create a child implementation that combines the best aspects of both parents.

## Parent A's implementation of "{{ capability }}":
```python
{{ parent_a_code }}
```

## Parent B's implementation of "{{ capability }}":
```python
{{ parent_b_code }}
```

{% if interface %}
## Interface contract (must be preserved):
- Capability: {{ interface.capability }}
- Input schema: {{ interface.input_schema }}
- Output schema: {{ interface.output_schema }}
{% endif %}

## Instructions:
Create a child implementation that:
1. Combines the best aspects of both parents
2. Maintains the interface contract if specified
3. Is syntactically valid Python
4. Preserves error handling from either parent
5. Does not introduce new dependencies not present in either parent
6. Is a genuine blend, not just one parent copied

Return ONLY the merged Python code, no explanations.""")


SEMANTIC_CROSSOVER = Template("""\
You are an evolutionary operator performing semantic crossover on two AI agent implementations.

## Parent A — "{{ parent_a_name }}":
### System prompt:
{{ parent_a_prompt }}

### Skills:
{% for skill in parent_a_skills %}
#### {{ skill.name }}:
```python
{{ skill.source }}
```
{% endfor %}

## Parent B — "{{ parent_b_name }}":
### System prompt:
{{ parent_b_prompt }}

### Skills:
{% for skill in parent_b_skills %}
#### {{ skill.name }}:
```python
{{ skill.source }}
```
{% endfor %}

## Instructions:
Create an offspring agent that genuinely blends capabilities from both parents.
Produce a JSON object with:
{
  "system_prompt": "the merged system prompt",
  "skills": {
    "skill_name": "python source code",
    ...
  }
}

The offspring should:
1. Have a coherent personality merging both parents' strengths
2. Retain the most useful skills from each parent
3. Where both parents have similar skills, produce a merged version
4. Be a viable, self-consistent agent

Return ONLY the JSON object.""")


PROMPT_CROSSOVER = Template("""\
You are merging two AI agent system prompts to create an offspring agent's personality.

## Parent A's system prompt:
{{ parent_a_prompt }}

## Parent B's system prompt:
{{ parent_b_prompt }}

## Instructions:
Create a merged system prompt that:
1. Combines the strongest personality traits from both parents
2. Blends their instruction styles coherently
3. Preserves critical constraints from both
4. Creates a unified, non-contradictory personality
5. Is roughly the same length as the longer parent

Return ONLY the merged system prompt text.""")


# ─── Mutation Prompts ────────────────────────────────────────────────────────

POINT_MUTATION = Template("""\
You are applying a point mutation to an AI agent's skill code.

## Current implementation:
```python
{{ current_code }}
```

## Mutation directive:
Apply a small, targeted improvement. Choose ONE of:
- Improve error handling
- Optimize performance
- Add edge case handling
- Improve code clarity
- Add input validation

## Constraints:
- Keep the same function signature
- Keep the same general approach
- Change should be small (1-10 lines affected)
- Must remain syntactically valid Python

Return ONLY the mutated Python code.""")


INSERTION_MUTATION = Template("""\
You are applying an insertion mutation — adding a new helper function to an existing skill.

## Current skill module:
```python
{{ current_code }}
```

## Skill capability: {{ capability }}

## Instructions:
Add ONE new helper function that enhances this skill's capability.
The helper should:
1. Complement the existing functions
2. Be useful for the skill's stated capability
3. Be self-contained (no new external dependencies)
4. Include a docstring

Return the COMPLETE module with the new function added.""")


REGULATORY_MUTATION = Template("""\
You are applying a regulatory mutation to an AI agent's system prompt.

## Current system prompt:
{{ current_prompt }}

## Instructions:
Make a small but meaningful modification to the agent's personality or behavior.
Choose ONE of:
- Adjust the agent's communication style slightly
- Add or refine a behavioral constraint
- Strengthen or soften a personality trait
- Add a new area of expertise or focus

The change should be subtle — not a complete rewrite.

Return ONLY the modified system prompt.""")


SKILL_SYNTHESIS = Template("""\
You are synthesizing an entirely new skill for an AI agent through mutation.

## Current agent capabilities:
{% for cap in capabilities %}
- {{ cap }}
{% endfor %}

## Agent's system prompt summary:
{{ prompt_summary }}

## Instructions:
Generate a NEW skill module that would complement this agent's existing capabilities.
The skill should:
1. Fill a gap in the agent's current skill set
2. Be a complete, working Python module
3. Include at least one main function with a clear docstring
4. Have a clear capability name

Respond with a JSON object:
{
  "skill_name": "name_of_new_skill",
  "capability": "what_it_does",
  "tags": ["tag1", "tag2"],
  "source": "the complete python source code"
}

Return ONLY the JSON object.""")


# ─── Fitness Evaluation Prompts ──────────────────────────────────────────────

FITNESS_JUDGE = Template("""\
You are evaluating an AI agent's output quality for a given task.

## Task description:
{{ task_description }}

## Expected behavior:
{{ evaluation_criteria }}

## Agent's output:
{{ agent_output }}

## Scoring dimensions:
Rate each dimension from 0.0 to 1.0:
- accuracy: How correct/relevant is the output?
- completeness: Does it fully address the task?
- efficiency: Is the approach efficient?
- robustness: Does it handle edge cases?

Respond with a JSON object:
{
  "accuracy": <float>,
  "completeness": <float>,
  "efficiency": <float>,
  "robustness": <float>,
  "overall": <float>,
  "reasoning": "brief explanation"
}

Return ONLY the JSON object.""")


# ─── Compatibility Analysis ──────────────────────────────────────────────────

COMPATIBILITY_ANALYSIS = Template("""\
Analyze the compatibility of these two AI agents for "mating" (code crossover).

## Agent A — "{{ agent_a_name }}":
Capabilities: {{ agent_a_capabilities }}
Personality: {{ agent_a_prompt_summary }}

## Agent B — "{{ agent_b_name }}":
Capabilities: {{ agent_b_capabilities }}
Personality: {{ agent_b_prompt_summary }}

## Questions:
1. How complementary are their skill sets? (0-1)
2. How compatible are their personalities? (0-1)
3. What would an offspring agent likely excel at?
4. What potential conflicts might arise in the offspring?

Respond with a JSON object:
{
  "skill_complementarity": <float>,
  "personality_compatibility": <float>,
  "offspring_strengths": ["strength1", "strength2"],
  "potential_conflicts": ["conflict1"],
  "recommendation": "breed" | "skip" | "high_mutation"
}

Return ONLY the JSON object.""")
