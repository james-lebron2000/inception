"""Built-in coding task benchmarks for fitness evaluation.

A collection of programming tasks used to evaluate agent code generation skills.
Tasks range from simple (string manipulation) to complex (algorithm design).
"""

from __future__ import annotations

from typing import Any


# Each task is a dict with:
# - id: unique identifier
# - description: what the task asks
# - function: the function name to test
# - test_cases: list of {input, expected} pairs
# - difficulty: 0.0 to 1.0

CODING_BENCHMARKS: list[dict[str, Any]] = [
    {
        "id": "fizzbuzz",
        "description": "Implement fizzbuzz: return 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else the number as string.",
        "function": "fizzbuzz",
        "difficulty": 0.2,
        "test_cases": [
            {"input": [1], "expected": "1"},
            {"input": [3], "expected": "Fizz"},
            {"input": [5], "expected": "Buzz"},
            {"input": [15], "expected": "FizzBuzz"},
            {"input": [7], "expected": "7"},
        ],
    },
    {
        "id": "reverse_string",
        "description": "Reverse a string without using built-in reverse functions.",
        "function": "reverse_string",
        "difficulty": 0.1,
        "test_cases": [
            {"input": ["hello"], "expected": "olleh"},
            {"input": [""], "expected": ""},
            {"input": ["a"], "expected": "a"},
            {"input": ["racecar"], "expected": "racecar"},
        ],
    },
    {
        "id": "fibonacci",
        "description": "Return the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.",
        "function": "fibonacci",
        "difficulty": 0.3,
        "test_cases": [
            {"input": [0], "expected": 0},
            {"input": [1], "expected": 1},
            {"input": [5], "expected": 5},
            {"input": [10], "expected": 55},
        ],
    },
    {
        "id": "is_palindrome",
        "description": "Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric).",
        "function": "is_palindrome",
        "difficulty": 0.3,
        "test_cases": [
            {"input": ["racecar"], "expected": True},
            {"input": ["hello"], "expected": False},
            {"input": ["A man a plan a canal Panama"], "expected": True},
            {"input": [""], "expected": True},
        ],
    },
    {
        "id": "flatten_list",
        "description": "Flatten a nested list of arbitrary depth into a single list.",
        "function": "flatten_list",
        "difficulty": 0.5,
        "test_cases": [
            {"input": [[1, [2, [3, 4], 5]]], "expected": [1, 2, 3, 4, 5]},
            {"input": [[]], "expected": []},
            {"input": [[1, 2, 3]], "expected": [1, 2, 3]},
            {"input": [[[1], [[2]], [[[3]]]]], "expected": [1, 2, 3]},
        ],
    },
    {
        "id": "two_sum",
        "description": "Given a list of numbers and a target, return indices of two numbers that add up to target.",
        "function": "two_sum",
        "difficulty": 0.4,
        "test_cases": [
            {"input": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"input": [[3, 2, 4], 6], "expected": [1, 2]},
        ],
    },
    {
        "id": "merge_sorted",
        "description": "Merge two sorted lists into one sorted list.",
        "function": "merge_sorted",
        "difficulty": 0.4,
        "test_cases": [
            {"input": [[1, 3, 5], [2, 4, 6]], "expected": [1, 2, 3, 4, 5, 6]},
            {"input": [[], [1, 2]], "expected": [1, 2]},
            {"input": [[1], []], "expected": [1]},
        ],
    },
]


def get_benchmark_by_id(benchmark_id: str) -> dict[str, Any] | None:
    """Get a specific benchmark by ID."""
    for b in CODING_BENCHMARKS:
        if b["id"] == benchmark_id:
            return b
    return None


def get_benchmarks_by_difficulty(
    min_difficulty: float = 0.0, max_difficulty: float = 1.0
) -> list[dict[str, Any]]:
    """Filter benchmarks by difficulty range."""
    return [
        b for b in CODING_BENCHMARKS
        if min_difficulty <= b["difficulty"] <= max_difficulty
    ]
