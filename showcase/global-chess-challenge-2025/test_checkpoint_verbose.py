#!/usr/bin/env python3
"""Verbose test of checkpoint inference"""
import torch
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).parent / "models" / "checkpoint-500"
BASE_MODEL = "Qwen/Qwen2.5-3B"

print("=" * 60)
print("Verbose Checkpoint Test")
print("=" * 60)

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

# Load tokenizer from checkpoint
from transformers import AutoTokenizer
print(f"\nLoading tokenizer from: {CHECKPOINT_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_PATH))
print(f"Tokenizer type: {type(tokenizer)}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

# Test chat template
messages = [
    {"role": "system", "content": "You are a chess grandmaster. Analyze positions and select the best move."},
    {"role": "user", "content": "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\nMoves: a7a6, b7b6, c7c6, d7d6, e7e6, f7f6, g7g6, h7h6, a7a5, b7b5, c7c5, d7d5, e7e5, f7f5, g7g5, h7h5, b8a6, b8c6, g8f6, g8h6\nBest?"}
]

print("\n--- Chat Template ---")
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
print("--- End Template ---")

# Tokenize
inputs = tokenizer(text, return_tensors="pt")
print(f"\nInput shape: {inputs['input_ids'].shape}")
print(f"Input tokens (first 50): {inputs['input_ids'][0][:50].tolist()}")

# Load model
from transformers import AutoModelForCausalLM
from peft import PeftModel

print(f"\nLoading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map={"": device} if device in ["mps", "cpu"] else "auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapter from: {CHECKPOINT_PATH}")
model = PeftModel.from_pretrained(model, str(CHECKPOINT_PATH))
model.eval()

print(f"\nModel dtype: {model.dtype}")
print(f"Model device: {next(model.parameters()).device}")

# Generate
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\nGenerating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(f"Output shape: {outputs.shape}")
new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
print(f"New tokens ({len(new_tokens)}): {new_tokens.tolist()[:50]}")

# Decode different ways
print("\n--- Decoded (skip_special_tokens=False) ---")
response_raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
print(response_raw)

print("\n--- Decoded (skip_special_tokens=True) ---")
response_clean = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response_clean)

# Check for move
import re
match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response_raw)
if match:
    print(f"\nExtracted move: {match.group(1)}")
else:
    print("\nNo <uci_move> tag found")
