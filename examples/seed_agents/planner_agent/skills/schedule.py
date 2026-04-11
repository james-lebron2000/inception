"""Task scheduling skill — create timelines and execution plans."""

from typing import Any


def schedule(
    tasks: list[dict[str, Any]],
    parallel_capacity: int = 2,
) -> list[dict[str, Any]]:
    """Create an execution schedule respecting dependencies.

    Uses topological sort with parallel execution where possible.

    Args:
        tasks: List of task dicts with 'name', 'dependencies', 'estimated_effort'.
        parallel_capacity: Max tasks that can run in parallel.

    Returns:
        List of phase dicts, each containing tasks that can run in parallel.
    """
    # Build dependency graph
    task_map = {t["name"]: t for t in tasks}
    deps = {t["name"]: set(t.get("dependencies", [])) for t in tasks}

    phases: list[dict[str, Any]] = []
    completed: set[str] = set()
    remaining = set(deps.keys())

    phase_num = 1
    while remaining:
        # Find tasks whose dependencies are all completed
        ready = [
            name for name in remaining
            if deps[name].issubset(completed)
        ]

        if not ready:
            # Circular dependency detected
            phases.append({
                "phase": phase_num,
                "tasks": list(remaining),
                "note": "WARNING: circular dependency detected, scheduling remaining tasks",
            })
            break

        # Limit parallel tasks
        batch = ready[:parallel_capacity]

        effort_values = {"small": 1, "medium": 2, "large": 4}
        batch_effort = max(
            effort_values.get(task_map[name].get("estimated_effort", "medium"), 2)
            for name in batch
        )

        phases.append({
            "phase": phase_num,
            "tasks": [task_map[name] for name in batch],
            "estimated_effort": batch_effort,
            "parallel": len(batch) > 1,
        })

        completed.update(batch)
        remaining -= set(batch)
        phase_num += 1

    return phases


def estimate_total_effort(phases: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate total effort from a schedule.

    Returns:
        Dict with 'total_phases', 'total_effort_units', 'critical_path_length'.
    """
    effort_values = {"small": 1, "medium": 2, "large": 4}
    total_effort = sum(
        phase.get("estimated_effort", 2)
        for phase in phases
    )

    return {
        "total_phases": len(phases),
        "total_effort_units": total_effort,
        "critical_path_length": len(phases),
    }


def find_critical_path(tasks: list[dict[str, Any]]) -> list[str]:
    """Find the critical path — longest chain of dependent tasks.

    Returns:
        List of task names forming the critical path.
    """
    task_map = {t["name"]: t for t in tasks}
    deps = {t["name"]: t.get("dependencies", []) for t in tasks}

    # Find all paths using DFS
    effort_values = {"small": 1, "medium": 2, "large": 4}

    def path_length(name: str, visited: set[str] | None = None) -> tuple[int, list[str]]:
        if visited is None:
            visited = set()
        if name in visited:
            return 0, []
        visited.add(name)

        effort = effort_values.get(
            task_map.get(name, {}).get("estimated_effort", "medium"), 2
        )

        dependents = [
            t_name for t_name, t_deps in deps.items()
            if name in t_deps
        ]

        if not dependents:
            return effort, [name]

        best_length = 0
        best_path: list[str] = []
        for dep in dependents:
            length, path = path_length(dep, visited.copy())
            if length > best_length:
                best_length = length
                best_path = path

        return effort + best_length, [name] + best_path

    # Find the root tasks (no dependencies)
    roots = [name for name, d in deps.items() if not d]

    longest_path: list[str] = []
    longest_length = 0
    for root in roots:
        length, path = path_length(root)
        if length > longest_length:
            longest_length = length
            longest_path = path

    return longest_path
