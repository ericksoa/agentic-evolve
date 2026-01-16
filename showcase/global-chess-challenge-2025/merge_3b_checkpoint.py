#!/usr/bin/env python3
"""
Merge Qwen 3B LoRA checkpoint and upload to HuggingFace.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent
ADAPTER_PATH = PROJECT_DIR / "models" / "qwen3b-checkpoint-12000"
MERGED_OUTPUT = PROJECT_DIR / "models" / "qwen3b-12k-merged"
HF_REPO = "ericksoa/chess-qwen3b-12k"
BASE_MODEL = "Qwen/Qwen2.5-3B"


def main():
    print("=" * 60)
    print("Qwen 3B LoRA Merge & Upload")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Output: {MERGED_OUTPUT}")
    print(f"HF Repo: {HF_REPO}")
    print()

    print("[1/4] Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("\n[2/4] Loading and merging LoRA adapter...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model = model.merge_and_unload()

    print("\n[3/4] Saving merged model locally...")
    MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_OUTPUT), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_OUTPUT))

    # Update tokenizer_config.json with correct eos_token_id
    import json
    tokenizer_config_path = MERGED_OUTPUT / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path) as f:
            tc = json.load(f)
        # Fix eos_token_id to point to <|im_end|> (151645) not <|endoftext|> (151643)
        tc["eos_token_id"] = 151645
        with open(tokenizer_config_path, "w") as f:
            json.dump(tc, f, indent=2)
        print("  Fixed eos_token_id to 151645 (<|im_end|>)")

    # Verify config.json
    config_path = MERGED_OUTPUT / "config.json"
    if config_path.exists():
        print("  config.json: OK")
    else:
        print("  WARNING: config.json not found!")

    print("\n[4/4] Uploading to HuggingFace...")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in environment")
        print("Set it with: export HF_TOKEN=your_token")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO, exist_ok=True, private=False)

    api.upload_folder(
        folder_path=str(MERGED_OUTPUT),
        repo_id=HF_REPO,
        commit_message="Qwen2.5-3B checkpoint-12000 merged (186 ACPL local)"
    )

    print()
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Uploaded to: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
