#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and upload to HuggingFace.

The AIcrowd competition's vLLM setup expects a complete model with config.json,
not just a LoRA adapter. This script merges our V6 adapter into Qwen2.5-7B.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_DIR = Path(__file__).parent
ADAPTER_PATH = PROJECT_DIR / "models" / "v6_checkpoint_4000"
MERGED_OUTPUT = PROJECT_DIR / "models" / "v6_merged"
HF_REPO = "ericksoa/chess-v6-aicrowd"
BASE_MODEL = "Qwen/Qwen2.5-7B"


def main():
    print("=" * 60)
    print("LoRA Merge & Upload Script")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Output: {MERGED_OUTPUT}")
    print(f"HF Repo: {HF_REPO}")
    print()

    # Check for MPS (Apple Silicon) or CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("Using CPU")

    print("\n[1/4] Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load in float16 to save memory
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

    # Verify config.json exists
    config_path = MERGED_OUTPUT / "config.json"
    if config_path.exists():
        print(f"  config.json: OK")
    else:
        print(f"  WARNING: config.json not found!")

    print("\n[4/4] Uploading to HuggingFace...")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in environment")
        return

    api = HfApi(token=token)

    # Create/verify repo
    api.create_repo(repo_id=HF_REPO, exist_ok=True, private=False)

    # Upload all files
    api.upload_folder(
        folder_path=str(MERGED_OUTPUT),
        repo_id=HF_REPO,
        commit_message="V6 merged model (Qwen2.5-7B + LoRA, 192.3 ACPL)"
    )

    print()
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Merged model uploaded to: https://huggingface.co/{HF_REPO}")
    print()
    print("Files uploaded:")
    for f in api.list_repo_files(repo_id=HF_REPO):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
