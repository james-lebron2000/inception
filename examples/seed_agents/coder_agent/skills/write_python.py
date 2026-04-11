"""Code generation skill — writes Python functions from descriptions."""


def generate_function(description: str) -> str:
    """Generate a Python function based on a natural language description.

    Args:
        description: What the function should do.

    Returns:
        A string containing the Python function source code.
    """
    # Template-based generation for common patterns
    templates = {
        "sort": _generate_sort,
        "search": _generate_search,
        "filter": _generate_filter,
        "transform": _generate_transform,
    }

    for keyword, generator in templates.items():
        if keyword in description.lower():
            return generator(description)

    return _generate_generic(description)


def _generate_sort(description: str) -> str:
    return '''def sort_items(items: list) -> list:
    """Sort items in ascending order."""
    if not items:
        return []
    return sorted(items)
'''


def _generate_search(description: str) -> str:
    return '''def search(items: list, target) -> int:
    """Binary search for target in sorted list. Returns index or -1."""
    left, right = 0, len(items) - 1
    while left <= right:
        mid = (left + right) // 2
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
'''


def _generate_filter(description: str) -> str:
    return '''def filter_items(items: list, predicate) -> list:
    """Filter items using a predicate function."""
    return [item for item in items if predicate(item)]
'''


def _generate_transform(description: str) -> str:
    return '''def transform(items: list, func) -> list:
    """Apply a transformation function to each item."""
    return [func(item) for item in items]
'''


def _generate_generic(description: str) -> str:
    return f'''def solve(data):
    """Auto-generated function for: {description}"""
    result = data
    return result
'''
