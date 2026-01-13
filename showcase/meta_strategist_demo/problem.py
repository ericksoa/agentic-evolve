"""
Sorting Algorithm Optimization Problem
======================================

Goal: Evolve the fastest sorting implementation for arrays of 1000 integers.

This problem is ideal for demonstrating Meta-Strategist because:
- Multiple mutation types have very different effectiveness
- parameter_tweak: Low impact (threshold tuning, etc.)
- structural: Medium impact (loop optimizations)
- algorithm_swap: High impact (switching sort algorithms)

Without Meta-Strategist: Wastes generations on low-impact mutations
With Meta-Strategist: Shifts focus to high-impact strategies
"""

import random
import time
from typing import Callable


def generate_test_data(n: int = 1000, seed: int = 42) -> list[int]:
    """Generate reproducible test data."""
    random.seed(seed)
    return [random.randint(0, 10000) for _ in range(n)]


def evaluate_sort(sort_func: Callable, iterations: int = 100) -> float:
    """Evaluate sorting function speed (ops/sec)."""
    test_data = generate_test_data()

    start = time.perf_counter()
    for _ in range(iterations):
        data = test_data.copy()
        sort_func(data)
    elapsed = time.perf_counter() - start

    return iterations / elapsed


# Baseline: Simple bubble sort (intentionally slow)
def baseline_sort(arr: list[int]) -> list[int]:
    """Bubble sort - O(n²) baseline."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# Parameter tweak: Adjusted threshold (minimal improvement)
def param_tweak_sort(arr: list[int]) -> list[int]:
    """Bubble sort with early termination flag."""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


# Structural: Better loop structure (moderate improvement)
def structural_sort(arr: list[int]) -> list[int]:
    """Insertion sort - better than bubble, still O(n²)."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# Algorithm swap: Completely different algorithm (huge improvement)
def algorithm_swap_sort(arr: list[int]) -> list[int]:
    """Quick sort - O(n log n) average."""
    if len(arr) <= 1:
        return arr

    def quicksort(items: list[int], low: int, high: int) -> None:
        if low < high:
            pivot = items[high]
            i = low - 1
            for j in range(low, high):
                if items[j] <= pivot:
                    i += 1
                    items[i], items[j] = items[j], items[i]
            items[i + 1], items[high] = items[high], items[i + 1]
            pi = i + 1
            quicksort(items, low, pi - 1)
            quicksort(items, pi + 1, high)

    quicksort(arr, 0, len(arr) - 1)
    return arr


# Champion: Optimized quicksort with insertion sort for small arrays
def champion_sort(arr: list[int]) -> list[int]:
    """Hybrid quicksort - uses insertion sort for small partitions."""
    THRESHOLD = 16

    def insertion_sort(items: list[int], low: int, high: int) -> None:
        for i in range(low + 1, high + 1):
            key = items[i]
            j = i - 1
            while j >= low and items[j] > key:
                items[j + 1] = items[j]
                j -= 1
            items[j + 1] = key

    def quicksort(items: list[int], low: int, high: int) -> None:
        if high - low < THRESHOLD:
            insertion_sort(items, low, high)
            return

        # Median-of-three pivot
        mid = (low + high) // 2
        if items[low] > items[mid]:
            items[low], items[mid] = items[mid], items[low]
        if items[low] > items[high]:
            items[low], items[high] = items[high], items[low]
        if items[mid] > items[high]:
            items[mid], items[high] = items[high], items[mid]

        pivot = items[mid]
        items[mid], items[high - 1] = items[high - 1], items[mid]

        i, j = low, high - 1
        while True:
            i += 1
            while items[i] < pivot:
                i += 1
            j -= 1
            while items[j] > pivot:
                j -= 1
            if i >= j:
                break
            items[i], items[j] = items[j], items[i]

        items[i], items[high - 1] = items[high - 1], items[i]
        quicksort(items, low, i - 1)
        quicksort(items, i + 1, high)

    if len(arr) > 1:
        quicksort(arr, 0, len(arr) - 1)
    return arr


MUTATIONS = {
    "baseline": baseline_sort,
    "param_tweak": param_tweak_sort,
    "structural": structural_sort,
    "algorithm_swap": algorithm_swap_sort,
    "champion": champion_sort,
}


if __name__ == "__main__":
    print("Sorting Algorithm Benchmarks")
    print("=" * 50)

    for name, func in MUTATIONS.items():
        ops = evaluate_sort(func, iterations=10)
        print(f"{name:20s}: {ops:,.1f} ops/sec")
