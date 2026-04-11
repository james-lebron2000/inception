"""Data analysis skill — statistical analysis and pattern detection."""

from typing import Any
import math


def analyze(data: list[float | int]) -> dict[str, Any]:
    """Perform statistical analysis on numerical data.

    Args:
        data: List of numerical values.

    Returns:
        Dict with statistical measures.
    """
    if not data:
        return {"error": "Empty dataset", "count": 0}

    n = len(data)
    sorted_data = sorted(data)
    mean = sum(data) / n

    # Variance and standard deviation
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)

    # Median
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    # Mode
    freq: dict[float | int, int] = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    mode = max(freq, key=lambda k: freq[k])

    return {
        "count": n,
        "mean": round(mean, 4),
        "median": median,
        "mode": mode,
        "std_dev": round(std_dev, 4),
        "variance": round(variance, 4),
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
    }


def detect_outliers(data: list[float | int], threshold: float = 2.0) -> list[float | int]:
    """Detect outliers using z-score method.

    Args:
        data: List of numerical values.
        threshold: Z-score threshold for outlier detection.

    Returns:
        List of outlier values.
    """
    if len(data) < 3:
        return []

    mean = sum(data) / len(data)
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

    if std_dev == 0:
        return []

    return [x for x in data if abs((x - mean) / std_dev) > threshold]


def find_trend(data: list[float | int]) -> str:
    """Detect the overall trend in sequential data.

    Returns: 'increasing', 'decreasing', 'stable', or 'fluctuating'.
    """
    if len(data) < 2:
        return "insufficient_data"

    increases = sum(1 for i in range(1, len(data)) if data[i] > data[i - 1])
    decreases = sum(1 for i in range(1, len(data)) if data[i] < data[i - 1])
    total = len(data) - 1

    if increases / total > 0.7:
        return "increasing"
    elif decreases / total > 0.7:
        return "decreasing"
    elif increases / total > 0.4 and decreases / total > 0.4:
        return "fluctuating"
    else:
        return "stable"
