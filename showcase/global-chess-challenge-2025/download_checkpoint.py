#!/usr/bin/env python3
"""
Download a checkpoint from Lightning.ai and optionally run quick evaluation.

Usage:
    python3 download_checkpoint.py                    # Download latest checkpoint
    python3 download_checkpoint.py --checkpoint 500   # Download specific checkpoint
    python3 download_checkpoint.py --list             # List available checkpoints
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Lightning.ai credentials
os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

from lightning_sdk import Studio

LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")

def get_studio():
    """Connect to Lightning.ai studio"""
    return Studio(
        name="chess-llm-v4",
        teamspace=LIGHTNING_TEAMSPACE,
        user=LIGHTNING_USERNAME,
    )

def list_checkpoints(studio):
    """List available checkpoints"""
    print("Available checkpoints:")
    print("-" * 40)
    result = studio.run("ls -la ~/chess-training/models-v2/ 2>/dev/null || echo 'No checkpoints'")
    print(result)
    return result

def download_checkpoint(studio, checkpoint_step=None):
    """Download a checkpoint from Lightning.ai"""

    # Determine which checkpoint to download
    if checkpoint_step:
        remote_path = f"chess-training/models-v2/checkpoint-{checkpoint_step}"
        local_name = f"checkpoint-{checkpoint_step}"
    else:
        # Check for 'final' first, then latest checkpoint
        result = studio.run("ls ~/chess-training/models-v2/final 2>/dev/null && echo 'HAS_FINAL' || echo 'NO_FINAL'")
        if "HAS_FINAL" in result:
            remote_path = "chess-training/models-v2/final"
            local_name = "final"
        else:
            # Find latest checkpoint
            result = studio.run("ls -d ~/chess-training/models-v2/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
            if not result.strip():
                print("No checkpoints found!")
                return None
            checkpoint_dir = result.strip().split("/")[-1]
            remote_path = f"chess-training/models-v2/{checkpoint_dir}"
            local_name = checkpoint_dir

    # Create local directory
    local_dir = Path(__file__).parent / "models" / local_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {remote_path}")
    print(f"To: {local_dir}")
    print()

    # List files to download
    files_result = studio.run(f"ls ~/{remote_path}/ 2>/dev/null")
    if not files_result.strip():
        print(f"Checkpoint not found: {remote_path}")
        return None

    files = files_result.strip().split("\n")
    print(f"Files to download: {files}")

    # Download each file
    for filename in files:
        filename = filename.strip()
        if not filename:
            continue
        print(f"  Downloading {filename}...")
        try:
            studio.download_file(f"{remote_path}/{filename}", str(local_dir / filename))
        except Exception as e:
            print(f"    Error: {e}")

    print()
    print(f"Downloaded to: {local_dir}")

    # Verify download
    downloaded_files = list(local_dir.iterdir())
    print(f"Local files: {[f.name for f in downloaded_files]}")

    return local_dir

def quick_test(checkpoint_path):
    """Run a quick inference test on downloaded checkpoint"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print()
    print("=" * 60)
    print("Quick Inference Test")
    print("=" * 60)

    BASE_MODEL = "Qwen/Qwen2.5-3B"

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"Loading base model: {BASE_MODEL}")

    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map={"": device} if device in ["mps", "cpu"] else "auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    # Test prompt - opening position after 1.e4
    test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    test_moves = ["a7a6", "b7b6", "c7c6", "d7d6", "e7e6", "f7f6", "g7g6", "h7h6",
                  "a7a5", "b7b5", "c7c5", "d7d5", "e7e5", "f7f5", "g7g5", "h7h5",
                  "b8a6", "b8c6", "g8f6", "g8h6"]

    messages = [
        {"role": "system", "content": "You are a chess grandmaster. Analyze positions and select the best move."},
        {"role": "user", "content": f"FEN: {test_fen}\nMoves: {', '.join(test_moves)}\nBest?"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print()
    print("Test position: 1.e4 (Black to move)")
    print(f"FEN: {test_fen}")
    print()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("Model response:")
    print("-" * 40)
    print(response)
    print("-" * 40)

    # Check if response contains a valid move (training format uses <uci_move>)
    import re
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
    if match:
        move = match.group(1)
        print(f"\nExtracted move: {move}")
        if move in test_moves:
            print("Move is LEGAL")
        else:
            print("Move is ILLEGAL")
    else:
        print("\nNo UCI move found in response")

def main():
    parser = argparse.ArgumentParser(description="Download and test checkpoints")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--checkpoint", type=int, help="Specific checkpoint step to download")
    parser.add_argument("--test", action="store_true", help="Run quick inference test after download")
    args = parser.parse_args()

    studio = get_studio()

    if args.list:
        list_checkpoints(studio)
        return

    checkpoint_path = download_checkpoint(studio, args.checkpoint)

    if checkpoint_path and args.test:
        quick_test(checkpoint_path)

if __name__ == "__main__":
    main()
