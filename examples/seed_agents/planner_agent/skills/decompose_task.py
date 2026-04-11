"""Task decomposition skill — break complex tasks into subtasks."""

from typing import Any


def decompose(task: str, max_depth: int = 3) -> list[dict[str, Any]]:
    """Break a complex task into manageable subtasks.

    Args:
        task: Description of the task to decompose.
        max_depth: Maximum nesting depth for subtasks.

    Returns:
        List of subtask dicts with 'name', 'description', 'dependencies', 'estimated_effort'.
    """
    # Analyze task complexity
    words = task.lower().split()
    complexity = _estimate_complexity(words)

    subtasks = []

    if complexity <= 1:
        # Simple task — no decomposition needed
        subtasks.append({
            "name": task,
            "description": task,
            "dependencies": [],
            "estimated_effort": "small",
        })
    elif complexity <= 3:
        # Medium task — basic decomposition
        subtasks.extend([
            {"name": "Analyze requirements", "description": f"Understand what '{task}' entails",
             "dependencies": [], "estimated_effort": "small"},
            {"name": "Implement solution", "description": f"Build the solution for: {task}",
             "dependencies": ["Analyze requirements"], "estimated_effort": "medium"},
            {"name": "Verify results", "description": "Test and validate the implementation",
             "dependencies": ["Implement solution"], "estimated_effort": "small"},
        ])
    else:
        # Complex task — detailed decomposition
        subtasks.extend([
            {"name": "Research & analysis", "description": "Gather requirements and context",
             "dependencies": [], "estimated_effort": "medium"},
            {"name": "Design approach", "description": "Plan the solution architecture",
             "dependencies": ["Research & analysis"], "estimated_effort": "medium"},
            {"name": "Core implementation", "description": "Build the main functionality",
             "dependencies": ["Design approach"], "estimated_effort": "large"},
            {"name": "Edge case handling", "description": "Handle boundary conditions and errors",
             "dependencies": ["Core implementation"], "estimated_effort": "medium"},
            {"name": "Testing", "description": "Write and run tests",
             "dependencies": ["Core implementation"], "estimated_effort": "medium"},
            {"name": "Review & refine", "description": "Code review and optimization",
             "dependencies": ["Testing", "Edge case handling"], "estimated_effort": "small"},
        ])

    return subtasks


def identify_dependencies(subtasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a dependency graph from subtasks.

    Returns:
        Dict mapping task name to list of tasks it depends on.
    """
    return {task["name"]: task.get("dependencies", []) for task in subtasks}


def _estimate_complexity(words: list[str]) -> int:
    """Estimate task complexity from word analysis."""
    complexity_indicators = {
        "and": 1, "with": 1, "including": 1, "multiple": 2,
        "integrate": 2, "optimize": 2, "refactor": 2,
        "architecture": 3, "distributed": 3, "concurrent": 3,
    }
    score = 1
    for word in words:
        score += complexity_indicators.get(word, 0)
    return min(score, 5)
