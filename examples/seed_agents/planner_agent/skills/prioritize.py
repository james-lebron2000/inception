"""Task prioritization skill — rank tasks by importance and urgency."""

from typing import Any


def prioritize(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank tasks by priority using impact/effort scoring.

    Args:
        tasks: List of task dicts with 'name' and optional 'effort', 'impact', 'dependencies'.

    Returns:
        Tasks sorted by priority score (highest first), with 'priority_score' added.
    """
    scored_tasks = []
    for task in tasks:
        score = _compute_priority_score(task, tasks)
        task_copy = dict(task)
        task_copy["priority_score"] = round(score, 2)
        scored_tasks.append(task_copy)

    # Sort by priority score (highest first), then by dependency count (fewer deps first)
    scored_tasks.sort(
        key=lambda t: (t["priority_score"], -len(t.get("dependencies", []))),
        reverse=True,
    )

    return scored_tasks


def categorize_priority(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Categorize tasks into priority buckets.

    Returns:
        Dict with 'critical', 'high', 'medium', 'low' lists.
    """
    ranked = prioritize(tasks)
    categories: dict[str, list[dict[str, Any]]] = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
    }

    for task in ranked:
        score = task["priority_score"]
        if score >= 0.8:
            categories["critical"].append(task)
        elif score >= 0.6:
            categories["high"].append(task)
        elif score >= 0.3:
            categories["medium"].append(task)
        else:
            categories["low"].append(task)

    return categories


def _compute_priority_score(task: dict[str, Any], all_tasks: list[dict[str, Any]]) -> float:
    """Compute priority score for a single task.

    Factors:
    - Impact (user-specified or inferred): 40%
    - Effort (inverse — quick wins score higher): 30%
    - Dependency importance (tasks that unblock others score higher): 30%
    """
    impact = task.get("impact", 0.5)
    effort_map = {"small": 0.8, "medium": 0.5, "large": 0.2}
    effort_score = effort_map.get(task.get("effort", task.get("estimated_effort", "medium")), 0.5)

    # How many other tasks depend on this one?
    task_name = task.get("name", "")
    dependents = sum(
        1 for t in all_tasks
        if task_name in t.get("dependencies", [])
    )
    dependency_score = min(1.0, dependents / max(len(all_tasks), 1) * 3)

    return 0.4 * impact + 0.3 * effort_score + 0.3 * dependency_score
