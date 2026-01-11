"""
Evolvable Chess Prompt Template

This file defines the prompt configuration that will be evolved
to minimize ACPL (Average Centipawn Loss).

EVOLUTION TARGETS:
- system_prompt: The system message defining the model's role
- user_template: How we format the position and moves for the model
- few_shot_examples: Example position/move pairs to guide output
- format_hint: Instructions about output format
- temperature: Sampling temperature (0.0 = greedy)
- force_prefix: What to prepend to generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class FewShotExample:
    """A single few-shot example"""
    fen: str
    moves: List[str]
    best_move: str
    rationale: str


@dataclass
class PromptConfig:
    """Evolvable prompt configuration"""

    # Core prompts
    system_prompt: str = "You are a chess grandmaster. Analyze positions and select the best move."

    user_template: str = "FEN: {fen}\nMoves: {moves}\nBest?"

    # Output format guidance
    format_hint: str = ""  # Added after user content

    # Few-shot examples (0-3 recommended)
    few_shot_examples: List[FewShotExample] = field(default_factory=list)

    # Generation parameters
    temperature: float = 0.0  # 0.0 = greedy (deterministic)
    top_p: float = 1.0

    # Force model to start with this prefix
    force_prefix: str = "<uci_move>"

    # Max tokens to generate
    max_new_tokens: int = 50

    def to_dict(self) -> dict:
        """Serialize to dict for saving"""
        return {
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "format_hint": self.format_hint,
            "few_shot_examples": [
                {"fen": ex.fen, "moves": ex.moves, "best_move": ex.best_move, "rationale": ex.rationale}
                for ex in self.few_shot_examples
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "force_prefix": self.force_prefix,
            "max_new_tokens": self.max_new_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptConfig":
        """Deserialize from dict"""
        few_shots = [
            FewShotExample(**ex) for ex in d.get("few_shot_examples", [])
        ]
        return cls(
            system_prompt=d.get("system_prompt", cls.system_prompt),
            user_template=d.get("user_template", cls.user_template),
            format_hint=d.get("format_hint", ""),
            few_shot_examples=few_shots,
            temperature=d.get("temperature", 0.0),
            top_p=d.get("top_p", 1.0),
            force_prefix=d.get("force_prefix", "<uci_move>"),
            max_new_tokens=d.get("max_new_tokens", 50),
        )

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PromptConfig":
        """Load config from JSON file"""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ============================================================
# BASELINE CONFIGURATION (Generation 0)
# ============================================================

BASELINE = PromptConfig(
    system_prompt="You are a chess grandmaster. Analyze positions and select the best move.",
    user_template="FEN: {fen}\nMoves: {moves}\nBest?",
    format_hint="",
    few_shot_examples=[],
    temperature=0.0,
    force_prefix="<uci_move>",
)


# ============================================================
# MUTATION CANDIDATES - Ideas to evolve
# ============================================================

SYSTEM_PROMPT_VARIANTS = [
    # Original
    "You are a chess grandmaster. Analyze positions and select the best move.",

    # More specific
    "You are a chess grandmaster with 2600+ Elo. Given a position, identify the strongest continuation.",

    # Task-focused
    "You are a chess engine. Output only the best move in UCI format.",

    # Analytical
    "You are a chess analyst. Evaluate the position and select the move that maximizes advantage.",

    # Concise
    "Expert chess player. Pick the best move.",

    # Stockfish-style
    "You are Stockfish. Analyze deeply and output the optimal move.",

    # Teaching style
    "You are a chess teacher helping a student find the best move. Be precise.",
]

USER_TEMPLATE_VARIANTS = [
    # Original
    "FEN: {fen}\nMoves: {moves}\nBest?",

    # More structured
    "Position (FEN): {fen}\nLegal moves: {moves}\nSelect the best move:",

    # Minimal
    "{fen}\n{moves}\nBest:",

    # Analytical
    "Analyze this chess position:\nFEN: {fen}\nAvailable moves: {moves}\nWhich move is strongest?",

    # Direct
    "FEN: {fen}\nChoose the best move from: {moves}",
]

FORMAT_HINT_VARIANTS = [
    # None
    "",

    # Explicit format
    "\nRespond with <uci_move>MOVE</uci_move>",

    # With rationale
    "\nFormat: <uci_move>MOVE</uci_move> then <rationale>WHY</rationale>",

    # Minimal
    "\nOutput format: <uci_move>move</uci_move>",
]

# Good few-shot examples (from training data, known correct)
CANDIDATE_FEW_SHOTS = [
    FewShotExample(
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        moves=["e7e5", "e7e6", "c7c5", "d7d5", "g8f6", "b8c6"],
        best_move="e7e5",
        rationale="Classical response to 1.e4, fighting for the center"
    ),
    FewShotExample(
        fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        moves=["b8c6", "g8f6", "d7d6", "f7f6"],
        best_move="b8c6",
        rationale="Develops knight to natural square, defends e5 pawn"
    ),
    FewShotExample(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        moves=["d2d3", "c2c3", "b1c3", "d2d4", "e1g1"],
        best_move="c2c3",
        rationale="Prepares d4 advance, solid Italian Game setup"
    ),
]


if __name__ == "__main__":
    # Test serialization
    config = BASELINE
    print("Baseline config:")
    print(json.dumps(config.to_dict(), indent=2))

    # Save baseline
    config.save(".evolve/chess-prompts/mutations/gen0_baseline.json")
    print("\nSaved to gen0_baseline.json")
