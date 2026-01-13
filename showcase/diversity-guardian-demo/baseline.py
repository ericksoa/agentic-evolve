"""Baseline solution for the deceptive landscape problem.

Problem: Find the optimal bit string that maximizes a deceptive fitness function.
The function has a global optimum and a deceptive local optimum that "traps"
greedy hill-climbing approaches.

This demonstrates premature convergence - without diversity monitoring,
evolution gets stuck in the local optimum.
"""


def solve(n: int = 20) -> str:
    """Generate a solution bit string of length n.

    This baseline returns all zeros, which is NOT optimal but serves
    as a starting point for evolution.

    The fitness function rewards:
    - Local optimum: mostly 1s (e.g., "11111111110000000000") - fitness ~75
    - Global optimum: alternating pattern (e.g., "10101010101010101010") - fitness 100

    The trap: greedy mutations toward "more 1s" lead to the local optimum,
    while the global optimum requires a non-obvious alternating structure.

    Args:
        n: Length of the bit string

    Returns:
        A bit string solution
    """
    # Baseline: all zeros (suboptimal, fitness ~50)
    return "0" * n


if __name__ == "__main__":
    result = solve()
    print(f"Solution: {result}")
    print(f"Length: {len(result)}")
