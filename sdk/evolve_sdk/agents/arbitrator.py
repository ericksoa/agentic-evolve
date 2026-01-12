"""Arbitrator agent - provides neutral analysis for human escalation decisions.

The Arbitrator agent acts as a balanced advisor that:
1. Explains what the adversary's flags actually mean
2. Assesses realistic risk levels (not just paranoid rejection)
3. Presents the risk/reward tradeoffs clearly
4. Helps humans make informed decisions
5. Acknowledges uncertainty where it exists
"""

from ..skills import get_mode_guidance

ARBITRATOR_SYSTEM = """You are a neutral Arbitrator agent helping a human make trust decisions in evolutionary optimization.

Your role: Provide balanced, objective analysis when the trust system has flagged a candidate for human review.
You are NOT the adversary (skeptic) or the advocate (promoter). You are neutral.

## Core Responsibilities

1. **Explain Flags Clearly**: What does each flag actually mean? Is it a serious concern or a minor technicality?
2. **Assess Real Risk**: Not paranoid worst-case, but realistic assessment of what could go wrong
3. **Quantify Upside**: What's the potential benefit if this candidate is legitimate?
4. **Present Tradeoffs**: Help the human understand accept vs reject consequences
5. **Acknowledge Uncertainty**: Be honest about what you can and cannot determine

## Risk Level Guidelines

**LOW RISK** (likely safe to accept):
- Flag is about process/parsing rather than actual code problems
- Improvement is consistent with the algorithmic change made
- Similar improvements have been seen in legitimate candidates
- Code is readable and changes are clear

**MEDIUM RISK** (requires judgment):
- Large improvement but plausible explanation exists
- Some flags are concerning but others are benign
- Trade-off between exploration (accept) and safety (reject)

**HIGH RISK** (lean toward rejection):
- Multiple serious red flags
- Code changes don't explain the improvement
- Signs of overfitting or hardcoding
- Would become new champion with uncertain validity

## Output Format

Provide your analysis in a structured format that a human can quickly scan:

```
ARBITRATOR ANALYSIS
==================

SUMMARY: [One sentence - accept/reject/uncertain and why]

RISK LEVEL: [LOW / MEDIUM / HIGH]

FLAG BREAKDOWN:
- [flag1]: [SERIOUS/MINOR/TECHNICAL] - [what it means]
- [flag2]: [SERIOUS/MINOR/TECHNICAL] - [what it means]

POTENTIAL UPSIDE:
- [What the candidate offers if legitimate]

POTENTIAL DOWNSIDE:
- [What could go wrong if it's problematic]

KEY QUESTIONS:
- [Question human should consider]
- [Question human should consider]

MY ASSESSMENT:
[Your balanced recommendation with confidence level]
```

Remember: Your job is to INFORM, not to decide. Present the information clearly so the human can make the final call.
"""


def get_arbitrator_prompt(
    candidate_file: str,
    fitness: float,
    champion_fitness: float,
    trust_score: float,
    flags: list[str],
    adversary_analysis: str,
    mode: str,
    generation: int,
    candidate_code: str | None = None,
    parent_code: str | None = None,
) -> str:
    """Generate the prompt for an arbitrator to analyze a human escalation case.

    Args:
        candidate_file: Path to the candidate solution
        fitness: The candidate's fitness score
        champion_fitness: Current champion's fitness
        trust_score: Trust score assigned by adversary
        flags: List of flags raised by adversary
        adversary_analysis: The adversary's detailed analysis
        mode: Optimization mode (size, perf, ml)
        generation: Current generation number
        candidate_code: The candidate's source code (if available)
        parent_code: The parent's source code for comparison (if available)
    """

    # Calculate improvement stats
    improvement = 0
    improvement_pct = 0
    is_champion_candidate = fitness > champion_fitness

    if champion_fitness > 0:
        improvement = fitness - champion_fitness
        improvement_pct = (improvement / champion_fitness) * 100

    status = "POTENTIAL NEW CHAMPION" if is_champion_candidate else "Below current champion"

    flags_section = ""
    if flags:
        flags_section = "## Flags Raised\n"
        for i, flag in enumerate(flags, 1):
            flags_section += f"{i}. {flag}\n"
    else:
        flags_section = "## Flags Raised\n(No flags)\n"

    code_section = ""
    if candidate_code:
        # Truncate if too long
        if len(candidate_code) > 3000:
            candidate_code = candidate_code[:3000] + "\n... (truncated)"
        code_section = f"""
## Candidate Code
```python
{candidate_code}
```
"""

    diff_note = ""
    if parent_code and candidate_code:
        diff_note = "\nNote: Parent code is available - you can compare to understand what changed."

    # Get mode-specific context
    mode_guidance = get_mode_guidance(mode, "arbitrator") or ""
    mode_section = ""
    if mode_guidance:
        mode_section = f"""
## Mode Context ({mode})
{mode_guidance}
"""

    return f"""## Arbitrator Review - Human Escalation Case

A candidate solution has been flagged for human review. The trust system would reject it,
but it may be a legitimate improvement. Please provide balanced analysis to help the human decide.

## The Situation

- **Generation**: {generation}
- **Candidate**: {candidate_file}
- **Fitness**: {fitness:.4f}
- **Champion Fitness**: {champion_fitness:.4f}
- **Improvement**: {improvement:+.4f} ({improvement_pct:+.1f}%)
- **Status**: {status}
- **Trust Score**: {trust_score:.2f} (rejection threshold is typically 0.4)

{flags_section}

## Adversary's Analysis
{adversary_analysis if adversary_analysis else "(No detailed analysis provided)"}
{mode_section}
{code_section}
{diff_note}

## Your Task

1. **Review each flag** - Is it a serious concern or a technicality?
2. **Assess the code** (if provided) - Does the improvement make sense?
3. **Weigh risk vs reward** - What's at stake either way?
4. **Provide clear guidance** - Help the human make an informed decision

Be balanced. The adversary tends to be paranoid. Your job is to give a realistic assessment.
Don't rubber-stamp acceptance, but don't pile on more fear either.

Provide your analysis in the format specified in your system prompt."""
