#!/usr/bin/env python3
"""
Auto-download and test checkpoint-1000 when ready.
Results saved to checkpoint_1000_results.txt
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Lightning.ai credentials
os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

from lightning_sdk import Studio

LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")

RESULTS_FILE = Path(__file__).parent / "checkpoint_1000_results.txt"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(RESULTS_FILE, "a") as f:
        f.write(line + "\n")

def main():
    log("=" * 60)
    log("Auto Checkpoint-1000 Download & Test")
    log("=" * 60)

    studio = Studio(
        name="chess-llm-v4",
        teamspace=LIGHTNING_TEAMSPACE,
        user=LIGHTNING_USERNAME,
    )

    # Wait for checkpoint-1000
    log("Waiting for checkpoint-1000...")
    while True:
        result = studio.run("ls ~/chess-training/models-v2/checkpoint-1000 2>/dev/null && echo 'FOUND' || echo 'NOT_FOUND'")
        if "FOUND" in result:
            log("Checkpoint-1000 found!")
            break
        time.sleep(30)  # Check every 30 seconds

    # Download checkpoint
    log("Downloading checkpoint-1000...")
    local_dir = Path(__file__).parent / "models" / "checkpoint-1000"
    local_dir.mkdir(parents=True, exist_ok=True)

    files_result = studio.run("ls ~/chess-training/models-v2/checkpoint-1000/")
    files = [f.strip() for f in files_result.strip().split("\n") if f.strip()]
    log(f"Files to download: {files}")

    for filename in files:
        log(f"  Downloading {filename}...")
        try:
            studio.download_file(f"chess-training/models-v2/checkpoint-1000/{filename}", str(local_dir / filename))
        except Exception as e:
            log(f"    Error: {e}")

    log(f"Downloaded to: {local_dir}")

    # Also get current training status
    log("\nCurrent training status:")
    status = studio.run("tail -20 ~/train_chess_v2.log 2>/dev/null | grep -E '(loss|eval_loss|%\\|)'  | tail -5")
    log(status)

    # Run inference test
    log("\n" + "=" * 60)
    log("Running Inference Test")
    log("=" * 60)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    CHECKPOINT_PATH = local_dir
    BASE_MODEL = "Qwen/Qwen2.5-3B"

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log(f"Device: {device}")

    # Load tokenizer
    log(f"Loading tokenizer from: {CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_PATH))

    # Test positions
    test_positions = [
        {
            "name": "After 1.e4 (Black to move)",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "moves": ["a7a6", "b7b6", "c7c6", "d7d6", "e7e6", "f7f6", "g7g6", "h7h6",
                      "a7a5", "b7b5", "c7c5", "d7d5", "e7e5", "f7f5", "g7g5", "h7h5",
                      "b8a6", "b8c6", "g8f6", "g8h6"]
        },
        {
            "name": "Italian Game (White to move)",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "moves": ["a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "d2d3", "d2d4", "g2g3", "h2h3", "h2h4",
                      "b1a3", "b1c3", "c4b3", "c4d5", "c4b5", "c4f7", "c4d3", "c4e2", "c4f1",
                      "f3g1", "f3d4", "f3e5", "f3g5", "f3h4", "e1f1", "e1g1", "e1e2",
                      "h1f1", "h1g1", "d1e2"]
        },
        {
            "name": "Sicilian Defense (White to move)",
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "moves": ["a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4", "f2f3", "f2f4",
                      "g2g3", "g2g4", "h2h3", "h2h4", "b1a3", "b1c3", "g1f3", "g1h3", "g1e2",
                      "f1e2", "f1d3", "f1c4", "f1b5", "f1a6", "d1e2", "d1f3", "d1g4", "d1h5",
                      "e1e2"]
        }
    ]

    # Load model
    log(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map={"": device} if device in ["mps", "cpu"] else "auto",
        trust_remote_code=True
    )

    log(f"Loading LoRA adapter from: {CHECKPOINT_PATH}")
    model = PeftModel.from_pretrained(model, str(CHECKPOINT_PATH))
    model.eval()

    import re

    results = []
    for pos in test_positions:
        log(f"\n--- {pos['name']} ---")
        log(f"FEN: {pos['fen']}")

        messages = [
            {"role": "system", "content": "You are a chess grandmaster. Analyze positions and select the best move."},
            {"role": "user", "content": f"FEN: {pos['fen']}\nMoves: {', '.join(pos['moves'])}\nBest?"}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)

        log(f"Response: {response[:200]}...")

        # Check for move
        match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
        if match:
            move = match.group(1)
            is_legal = move in pos['moves']
            log(f"Extracted move: {move} ({'LEGAL' if is_legal else 'ILLEGAL'})")
            results.append({"pos": pos['name'], "move": move, "legal": is_legal})
        else:
            log("No <uci_move> tag found - model still learning format")
            results.append({"pos": pos['name'], "move": None, "legal": False})

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    valid_moves = sum(1 for r in results if r['move'] is not None)
    legal_moves = sum(1 for r in results if r['legal'])
    log(f"Positions tested: {len(results)}")
    log(f"Valid format (has <uci_move>): {valid_moves}/{len(results)}")
    log(f"Legal moves: {legal_moves}/{len(results)}")

    if valid_moves == 0:
        log("\nModel still learning output format. Check again at checkpoint-1500 or final.")
    elif legal_moves == len(results):
        log("\nModel is producing legal chess moves! Ready for ACPL evaluation.")
    else:
        log("\nModel is learning but making some illegal moves.")

    log("\nDone! Results saved to checkpoint_1000_results.txt")

if __name__ == "__main__":
    main()
