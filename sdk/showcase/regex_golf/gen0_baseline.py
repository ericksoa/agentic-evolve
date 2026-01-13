"""
Generation 0: Baseline Solution
================================

A naive approach that matches any Star Wars title by looking for common words.
This is intentionally suboptimal to leave room for evolution.

Fitness: Will be low due to length
"""

# Naive approach: match common words from Star Wars titles
# This is long but correct
PATTERN = r"Hope|Empire|Jedi|Phantom|Clones|Sith"

# Length: 39 characters
# Correct: Yes (matches all good, rejects all bad)
# Fitness: 1000 - 39 = 961 (baseline)
