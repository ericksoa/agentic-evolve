#!/bin/bash
# =============================================================================
# Global Chess Challenge 2025 - Full Training Pipeline
# =============================================================================
#
# Complete pipeline that:
#   1. Extracts training data from PGN files
#   2. Trains the model
#   3. Evaluates against Stockfish
#   4. Saves all results
#
# Usage:
#   ./run_full_pipeline.sh --pgn /path/to/games.pgn
#   ./run_full_pipeline.sh --data /path/to/existing.jsonl  # Skip extraction
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
PGN_PATH=""
DATA_PATH=""
OUTPUT_NAME="chess_llm_$(date +%Y%m%d_%H%M%S)"
MAX_EXAMPLES=50000
STOCKFISH_DEPTH=18
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE="1e-5"
LORA_RANK=64
EVAL_POSITIONS=100
SKIP_EVAL=false

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --pgn)
            PGN_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --stockfish-depth)
            STOCKFISH_DEPTH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --eval-positions)
            EVAL_POSITIONS="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 --pgn <path> OR --data <path> [options]"
            echo ""
            echo "Data Source (one required):"
            echo "  --pgn PATH              Path to PGN file for data extraction"
            echo "  --data PATH             Path to existing training JSONL"
            echo ""
            echo "Data Generation:"
            echo "  --max-examples N        Max training examples (default: 50000)"
            echo "  --stockfish-depth N     Stockfish analysis depth (default: 18)"
            echo ""
            echo "Training:"
            echo "  --output NAME           Output name (default: timestamped)"
            echo "  --epochs N              Training epochs (default: 3)"
            echo "  --batch-size N          Batch size (default: 4)"
            echo "  --lr RATE               Learning rate (default: 1e-5)"
            echo "  --lora-rank N           LoRA rank (default: 64)"
            echo ""
            echo "Evaluation:"
            echo "  --eval-positions N      Number of positions to evaluate (default: 100)"
            echo "  --skip-eval             Skip evaluation step"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate Arguments
# -----------------------------------------------------------------------------
if [ -z "$PGN_PATH" ] && [ -z "$DATA_PATH" ]; then
    echo "Error: Either --pgn or --data is required"
    echo "Usage: $0 --pgn <path> OR --data <path>"
    exit 1
fi

# -----------------------------------------------------------------------------
# Setup Directories
# -----------------------------------------------------------------------------
PROJECT_DIR="${HOME}/chess-training"
DATA_DIR="${PROJECT_DIR}/data"
MODELS_DIR="${PROJECT_DIR}/models"
LOGS_DIR="${PROJECT_DIR}/logs"
RESULTS_DIR="${PROJECT_DIR}/results"

mkdir -p "${DATA_DIR}" "${MODELS_DIR}" "${LOGS_DIR}" "${RESULTS_DIR}"

OUTPUT_DIR="${MODELS_DIR}/${OUTPUT_NAME}"
RESULTS_FILE="${RESULTS_DIR}/${OUTPUT_NAME}_results.json"
LOG_FILE="${LOGS_DIR}/${OUTPUT_NAME}.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start logging
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Global Chess Challenge 2025 - Full Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Data Extraction (if PGN provided)
# -----------------------------------------------------------------------------
if [ -n "$PGN_PATH" ]; then
    echo "=============================================="
    echo "STEP 1: Extracting Training Data from PGN"
    echo "=============================================="
    echo "PGN file: ${PGN_PATH}"
    echo "Max examples: ${MAX_EXAMPLES}"
    echo "Stockfish depth: ${STOCKFISH_DEPTH}"
    echo ""

    if [ ! -f "$PGN_PATH" ]; then
        echo "Error: PGN file not found: $PGN_PATH"
        exit 1
    fi

    DATA_PATH="${DATA_DIR}/${OUTPUT_NAME}_training.jsonl"

    # Create extraction script
    python3 << PYEOF
import chess
import chess.pgn
import chess.engine
import json
import sys
from pathlib import Path

PGN_PATH = "${PGN_PATH}"
OUTPUT_PATH = "${DATA_PATH}"
MAX_EXAMPLES = ${MAX_EXAMPLES}
STOCKFISH_DEPTH = ${STOCKFISH_DEPTH}

print(f"Opening PGN: {PGN_PATH}")
print(f"Output: {OUTPUT_PATH}")

# Find Stockfish
import shutil
STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"
print(f"Using Stockfish: {STOCKFISH}")

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
engine.configure({"Threads": 4})

examples = []
games_processed = 0
positions_skipped = 0

with open(PGN_PATH, 'r') as pgn_file:
    while len(examples) < MAX_EXAMPLES:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

        games_processed += 1
        if games_processed % 100 == 0:
            print(f"Processed {games_processed} games, {len(examples)} examples...")

        # Get Elo ratings
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            if white_elo < 1800 or black_elo < 1800:
                continue
        except ValueError:
            continue

        board = game.board()
        for move_num, node in enumerate(game.mainline()):
            if len(examples) >= MAX_EXAMPLES:
                break

            move = node.move

            # Skip early moves (opening book)
            if move_num < 8:
                board.push(move)
                continue

            # Skip forced moves
            legal_moves = list(board.legal_moves)
            if len(legal_moves) <= 1:
                board.push(move)
                positions_skipped += 1
                continue

            # Skip very long endgames
            if move_num > 80:
                board.push(move)
                continue

            try:
                # Analyze with Stockfish
                analysis = engine.analyse(
                    board,
                    chess.engine.Limit(depth=STOCKFISH_DEPTH)
                )
                best_move = analysis["pv"][0]
                score = analysis["score"].white()

                if score.is_mate():
                    cp_score = 10000 if score.mate() > 0 else -10000
                else:
                    cp_score = score.score()

                # Determine difficulty
                if abs(cp_score) > 300:
                    difficulty = "easy"
                elif abs(cp_score) > 100:
                    difficulty = "medium"
                else:
                    difficulty = "hard"

                # Determine game phase
                if move_num <= 15:
                    phase = "opening"
                elif move_num <= 35:
                    phase = "middlegame"
                else:
                    phase = "endgame"

                # Generate rationale
                san = board.san(best_move)
                if board.is_capture(best_move):
                    captured = board.piece_at(best_move.to_square)
                    rationale = f"Captures on {chess.square_name(best_move.to_square)}"
                elif board.gives_check(best_move):
                    rationale = f"Plays {san} with check"
                else:
                    piece = board.piece_at(best_move.from_square)
                    piece_name = chess.piece_name(piece.piece_type).title()
                    rationale = f"Develops {piece_name} to {chess.square_name(best_move.to_square)}"

                # Create training example in chat format
                fen = board.fen()
                side = "White" if board.turn else "Black"
                legal_uci = [m.uci() for m in legal_moves]

                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a chess grandmaster. Given a position and legal moves, select the best move."
                        },
                        {
                            "role": "user",
                            "content": f"Position (FEN): {fen}\\nSide to move: {side}\\nLegal moves: {', '.join(legal_uci[:20])}\\n\\nSelect the best move."
                        },
                        {
                            "role": "assistant",
                            "content": f"<uci_move>{best_move.uci()}</uci_move>\\n<rationale>{rationale}</rationale>"
                        }
                    ],
                    "metadata": {
                        "fen": fen,
                        "best_move": best_move.uci(),
                        "cp_score": cp_score,
                        "difficulty": difficulty,
                        "phase": phase
                    }
                }

                examples.append(example)

            except Exception as e:
                pass

            board.push(move)

engine.quit()

print(f"\\nExtraction complete!")
print(f"  Games processed: {games_processed}")
print(f"  Examples generated: {len(examples)}")
print(f"  Positions skipped: {positions_skipped}")

# Save examples
with open(OUTPUT_PATH, 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\\n')

print(f"  Saved to: {OUTPUT_PATH}")

# Print phase distribution
phases = {}
difficulties = {}
for ex in examples:
    p = ex["metadata"]["phase"]
    d = ex["metadata"]["difficulty"]
    phases[p] = phases.get(p, 0) + 1
    difficulties[d] = difficulties.get(d, 0) + 1

print(f"\\nPhase distribution:")
for p, c in sorted(phases.items()):
    print(f"  {p}: {c} ({100*c/len(examples):.1f}%)")

print(f"\\nDifficulty distribution:")
for d, c in sorted(difficulties.items()):
    print(f"  {d}: {c} ({100*c/len(examples):.1f}%)")
PYEOF

    echo ""
    echo "Data extraction complete: ${DATA_PATH}"
    echo ""
else
    echo "=============================================="
    echo "STEP 1: Using Existing Training Data"
    echo "=============================================="
    echo "Data file: ${DATA_PATH}"

    if [ ! -f "$DATA_PATH" ]; then
        echo "Error: Data file not found: $DATA_PATH"
        exit 1
    fi

    NUM_EXAMPLES=$(wc -l < "$DATA_PATH")
    echo "Examples: ${NUM_EXAMPLES}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 2: Model Training
# -----------------------------------------------------------------------------
echo "=============================================="
echo "STEP 2: Training Model"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "LoRA rank: ${LORA_RANK}"
echo ""

# Run training script
"${SCRIPT_DIR}/train.sh" \
    --data "${DATA_PATH}" \
    --output "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --lora-rank "${LORA_RANK}"

echo ""
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}/final"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Evaluation
# -----------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
    echo "=============================================="
    echo "STEP 3: Evaluating Model"
    echo "=============================================="
    echo "Positions: ${EVAL_POSITIONS}"
    echo ""

    # Create evaluation script
    python3 << PYEOF
import chess
import chess.engine
import json
import random
import re
import sys
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "${OUTPUT_DIR}/final"
NUM_POSITIONS = ${EVAL_POSITIONS}
RESULTS_FILE = "${RESULTS_FILE}"

print(f"Loading model from: {MODEL_PATH}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Find Stockfish
STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"
print(f"Using Stockfish: {STOCKFISH}")
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

def generate_test_positions(n, seed=42):
    """Generate diverse test positions."""
    random.seed(seed)
    positions = []

    while len(positions) < n:
        board = chess.Board()
        num_moves = random.randint(10, 60)

        for _ in range(num_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))

        if not board.is_game_over() and len(list(board.legal_moves)) > 1:
            positions.append(board.copy())

    return positions

def get_model_move(board, model, tokenizer):
    """Get move from the fine-tuned model."""
    fen = board.fen()
    side = "White" if board.turn else "Black"
    legal_moves = [m.uci() for m in board.legal_moves]

    prompt = f"""<|im_start|>system
You are a chess grandmaster. Given a position and legal moves, select the best move.<|im_end|>
<|im_start|>user
Position (FEN): {fen}
Side to move: {side}
Legal moves: {', '.join(legal_moves[:20])}

Select the best move.<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse UCI move
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
    if match:
        return match.group(1)

    match = re.search(r'\\b([a-h][1-8][a-h][1-8][qrbn]?)\\b', response)
    if match:
        return match.group(1)

    return None

def evaluate_move(board, model_move_uci, engine, depth=20):
    """Evaluate model's move against Stockfish."""
    # Get Stockfish's best move
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    best_move = result["pv"][0]
    best_score = result["score"].white()

    if best_score.is_mate():
        best_cp = 10000 if best_score.mate() > 0 else -10000
    else:
        best_cp = best_score.score()

    # Check if model's move is legal
    try:
        model_move = chess.Move.from_uci(model_move_uci)
        is_legal = model_move in board.legal_moves
    except (ValueError, TypeError):
        is_legal = False

    if not is_legal:
        return {
            "is_legal": False,
            "is_best": False,
            "cp_loss": 10000,
            "best_move": best_move.uci(),
            "model_move": str(model_move_uci)
        }

    # Evaluate model's move
    board.push(model_move)
    model_result = engine.analyse(board, chess.engine.Limit(depth=depth))
    model_score = model_result["score"].white()
    board.pop()

    if model_score.is_mate():
        model_cp = 10000 if model_score.mate() > 0 else -10000
    else:
        model_cp = model_score.score()

    # Calculate loss
    if board.turn == chess.WHITE:
        cp_loss = best_cp - model_cp
    else:
        cp_loss = model_cp - best_cp

    cp_loss = max(0, cp_loss)

    return {
        "is_legal": True,
        "is_best": model_move == best_move,
        "cp_loss": cp_loss,
        "best_move": best_move.uci(),
        "model_move": model_move.uci()
    }

print(f"\\nGenerating {NUM_POSITIONS} test positions...")
positions = generate_test_positions(NUM_POSITIONS)

print(f"Evaluating model...")
results = []
for i, board in enumerate(positions):
    if (i + 1) % 10 == 0:
        print(f"  Position {i+1}/{NUM_POSITIONS}")

    model_move = get_model_move(board, model, tokenizer)
    result = evaluate_move(board, model_move, engine)
    result["fen"] = board.fen()
    results.append(result)

engine.quit()

# Calculate metrics
legal_results = [r for r in results if r["is_legal"]]
acpl = sum(r["cp_loss"] for r in results) / len(results)
legal_rate = len(legal_results) / len(results)
best_rate = sum(1 for r in results if r["is_best"]) / len(results)

print(f"\\n{'='*50}")
print("EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Positions evaluated: {len(results)}")
print(f"Average Centipawn Loss (ACPL): {acpl:.1f}")
print(f"Legal Move Rate: {legal_rate*100:.1f}%")
print(f"Best Move Rate: {best_rate*100:.1f}%")

# Save results
output = {
    "model_path": MODEL_PATH,
    "num_positions": len(results),
    "acpl": acpl,
    "legal_rate": legal_rate,
    "best_move_rate": best_rate,
    "results": results
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\\nResults saved to: {RESULTS_FILE}")
PYEOF

    echo ""
else
    echo "=============================================="
    echo "STEP 3: Evaluation Skipped"
    echo "=============================================="
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Training data: ${DATA_PATH}"
echo "  Model: ${OUTPUT_DIR}/final"
if [ "$SKIP_EVAL" = false ]; then
    echo "  Results: ${RESULTS_FILE}"
fi
echo "  Log: ${LOG_FILE}"
echo ""
echo "To download the model:"
echo "  scp -r user@instance:${OUTPUT_DIR}/final ./chess_model/"
echo ""
echo "To use the model for inference:"
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  model = AutoModelForCausalLM.from_pretrained('${OUTPUT_DIR}/final')"
echo ""
