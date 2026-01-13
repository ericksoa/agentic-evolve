"""
Regex Golf Problem Definition
=============================

Goal: Find the shortest regex that matches all GOOD strings and rejects all BAD strings.

This problem is ideal for testing Phase 1 agents because:
- Debugger: Regex syntax is fragile - invalid patterns crash immediately
- Plateau Breaker: Easy to get "good enough" regex, hard to find optimal

Problem sets are ordered by difficulty.
"""

# Problem Set 1: Simple prefix matching
# Optimal solution: ~5 chars
PROBLEM_1 = {
    "name": "foo_prefix",
    "description": "Match words starting with 'foo'",
    "good": ["foo", "food", "fool", "foot", "football", "foolish"],
    "bad": ["bar", "baz", "far", "foe", "fo", "oof", "afoo"],
    "known_optimal": r"^foo",
    "optimal_length": 4,
}

# Problem Set 2: Character class needed
# Optimal solution: ~8 chars
PROBLEM_2 = {
    "name": "vowel_sandwich",
    "description": "Match consonant-vowel-consonant patterns",
    "good": ["bat", "bet", "bit", "bot", "but", "cat", "cot", "cut", "dog", "dig"],
    "bad": ["aa", "bee", "sea", "the", "art", "eat", "oat", "eel", "add", "egg"],
    "known_optimal": r"^[b-df-hj-np-tv-z][aeiou][b-df-hj-np-tv-z]$",
    "optimal_length": 35,  # This is hard to golf
}

# Problem Set 3: Alternation needed
# Optimal solution: ~12 chars
PROBLEM_3 = {
    "name": "programming_langs",
    "description": "Match programming language names",
    "good": ["python", "ruby", "rust", "go", "java", "scala"],
    "bad": ["english", "french", "german", "spanish", "latin", "greek"],
    "known_optimal": r"^(python|ruby|rust|go|java|scala)$",
    "optimal_length": 34,
}

# Problem Set 4: Repetition patterns
# Tests quantifiers
PROBLEM_4 = {
    "name": "repeated_chars",
    "description": "Match strings with repeated characters",
    "good": ["aaa", "bbb", "abab", "aaab", "abbb", "aabb", "ababab"],
    "bad": ["abc", "ab", "a", "abcd", "abca", "xyz", "hello"],
    "known_optimal": r"^(.)\1|^(..)\2",
    "optimal_length": 16,
}

# Problem Set 5: The classic - Star Wars vs Star Trek
# A well-known regex golf problem
PROBLEM_5 = {
    "name": "wars_vs_trek",
    "description": "Match Star Wars movies, reject Star Trek",
    "good": [
        "A New Hope",
        "The Empire Strikes Back",
        "Return of the Jedi",
        "The Phantom Menace",
        "Attack of the Clones",
        "Revenge of the Sith",
    ],
    "bad": [
        "The Motion Picture",
        "The Wrath of Khan",
        "The Search for Spock",
        "The Voyage Home",
        "The Final Frontier",
        "The Undiscovered Country",
    ],
    # Pattern that matches: Hope, Back, Jedi, Menace, Clones, Sith
    # but not Trek movies (verified to work)
    "known_optimal": r"ope|edi|nac|ith|ack|lon",
    "optimal_length": 23,
}

# Problem Set 6: Numeric patterns
PROBLEM_6 = {
    "name": "valid_numbers",
    "description": "Match valid decimal numbers",
    "good": ["123", "0", "3.14", "0.5", "100.00", "42", "7.0"],
    "bad": ["12.34.56", ".5", "1.", "abc", "1a2", "12..34", ""],
    "known_optimal": r"^\d+\.?\d*$",
    "optimal_length": 12,
}

# Default problem for showcase
DEFAULT_PROBLEM = PROBLEM_5

ALL_PROBLEMS = [
    PROBLEM_1,
    PROBLEM_2,
    PROBLEM_3,
    PROBLEM_4,
    PROBLEM_5,
    PROBLEM_6,
]


def get_problem(name: str = None) -> dict:
    """Get a problem by name, or the default."""
    if name is None:
        return DEFAULT_PROBLEM
    for p in ALL_PROBLEMS:
        if p["name"] == name:
            return p
    raise ValueError(f"Unknown problem: {name}")
