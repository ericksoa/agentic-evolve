#!/usr/bin/env python3
"""
Chess Model Checkpoint Pipeline

COMPLETE WORKFLOW: Download LoRA -> Merge -> Test -> Upload -> Submit

This script ensures the correct process is ALWAYS followed:
1. Download LoRA adapter from Lightning.ai cloud
2. Merge with base model locally
3. Run local ACPL evaluation
4. Upload to HuggingFace (only if eval passes threshold)
5. Submit to AIcrowd contest

Usage:
    # Full pipeline for V6 7B checkpoint
    python checkpoint_pipeline.py --studio chess-llm-v5 --checkpoint 8000 --base-model Qwen/Qwen2.5-7B

    # Full pipeline for 3B checkpoint
    python checkpoint_pipeline.py --studio chess-llm-v4 --checkpoint 13000 --base-model Qwen/Qwen2.5-3B

    # Skip to specific step (if previous steps completed)
    python checkpoint_pipeline.py --skip-to merge --checkpoint 8000
    python checkpoint_pipeline.py --skip-to eval --checkpoint 8000
    python checkpoint_pipeline.py --skip-to upload --checkpoint 8000

    # Dry run (no actual upload/submit)
    python checkpoint_pipeline.py --checkpoint 8000 --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

# Studio -> training run mapping
STUDIO_RUNS = {
    "chess-llm-v5": "7b_v6_r128_e2_20260114_0320",  # V6 7B r128
    "chess-llm-v4": "qwen3b_sf3_500k",              # Qwen 3B (update if different)
}

# Base model mapping
BASE_MODELS = {
    "chess-llm-v5": "Qwen/Qwen2.5-7B",
    "chess-llm-v4": "Qwen/Qwen2.5-3B",
}

# ACPL thresholds
ACPL_BASELINE = 146  # Submission 307752 got ~146 ACPL
ACPL_THRESHOLD = 200  # Don't upload if worse than this

# Chess grandmaster chat template - MUST handle case where no system message provided!
# Copied from working ericksoa/chess-v6-aicrowd - DO NOT SIMPLIFY
# Instead of hardcoding, load from the golden template file
CHESS_CHAT_TEMPLATE_FILE = Path(__file__).parent / "golden_chat_template.jinja"


# =============================================================================
# STEP 1: DOWNLOAD LORA FROM CLOUD
# =============================================================================

def download_lora(studio_name: str, checkpoint: int, run_name: str = None) -> Path:
    """Download LoRA adapter from Lightning.ai studio"""
    print()
    print("=" * 60)
    print(f"STEP 1: DOWNLOAD LORA (checkpoint-{checkpoint})")
    print("=" * 60)

    os.environ['LIGHTNING_USER_ID'] = os.getenv('LIGHTNING_USER_ID')
    os.environ['LIGHTNING_API_KEY'] = os.getenv('LIGHTNING_API_KEY')

    from lightning_sdk import Studio

    username = os.getenv('LIGHTNING_AI_USERNAME')
    teamspace = os.getenv('LIGHTNING_TEAMSPACE')

    print(f"Connecting to {studio_name}...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=username)

    # Determine run name
    if run_name is None:
        run_name = STUDIO_RUNS.get(studio_name)
        if run_name is None:
            # Try to find it
            result = studio.run("ls ~/chess-training/ | head -5")
            print(f"Available runs: {result}")
            raise ValueError(f"Unknown run for studio {studio_name}. Set --run-name explicitly.")

    remote_path = f"chess-training/{run_name}/model/checkpoint-{checkpoint}"

    # Check if checkpoint exists
    result = studio.run(f"ls ~/{remote_path} 2>/dev/null && echo 'EXISTS' || echo 'NOT_FOUND'")
    if "NOT_FOUND" in result:
        print(f"Checkpoint-{checkpoint} not found!")
        print("Available checkpoints:")
        print(studio.run(f"ls ~/chess-training/{run_name}/model/ | grep checkpoint"))
        return None

    # Create local directory
    local_dir = MODELS_DIR / f"{studio_name}_checkpoint_{checkpoint}"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download files
    files = ['adapter_config.json', 'adapter_model.safetensors']
    for f in files:
        print(f"  Downloading {f}...")
        studio.download_file(f"{remote_path}/{f}", str(local_dir / f))

    # Verify
    adapter_size = (local_dir / "adapter_model.safetensors").stat().st_size
    print(f"  Downloaded: {adapter_size / 1e6:.1f} MB")

    print(f"\nLoRA saved to: {local_dir}")
    return local_dir


# =============================================================================
# STEP 2: MERGE WITH BASE MODEL
# =============================================================================

def merge_model(lora_path: Path, base_model: str, output_name: str = None) -> Path:
    """Merge LoRA adapter with base model"""
    print()
    print("=" * 60)
    print("STEP 2: MERGE WITH BASE MODEL")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if output_name is None:
        output_name = f"{lora_path.name}_merged"
    output_path = MODELS_DIR / output_name

    print(f"Base model: {base_model}")
    print(f"LoRA path: {lora_path}")
    print(f"Output: {output_path}")
    print()

    # Check if already merged
    if output_path.exists() and (output_path / "config.json").exists():
        print(f"Merged model already exists at {output_path}")
        response = input("Re-merge? [y/N]: ").strip().lower()
        if response != 'y':
            return output_path
        shutil.rmtree(output_path)

    print("[1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("[2/4] Loading and merging LoRA...")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()

    print("[3/4] Saving merged model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print("[4/4] Fixing tokenizer configs for ChatML...")
    fix_tokenizer_configs(output_path)

    print(f"\nMerged model saved to: {output_path}")
    return output_path


def fix_tokenizer_configs(model_path: Path):
    """Fix tokenizer and generation configs for proper ChatML termination"""

    # Fix tokenizer_config.json - eos_token must be <|im_end|>
    tokenizer_config = model_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)

        # Set eos_token to <|im_end|> for ChatML
        tc["eos_token"] = "<|im_end|>"

        with open(tokenizer_config, "w") as f:
            json.dump(tc, f, indent=2)
        print("  Fixed tokenizer_config.json: eos_token = <|im_end|>")

    # Fix generation_config.json - eos_token_id must be 151645 (<|im_end|>)
    gen_config = model_path / "generation_config.json"
    gen_data = {
        "eos_token_id": 151645,  # <|im_end|>, NOT 151643 (<|endoftext|>)
        "pad_token_id": 151643,
        "bos_token_id": 151643,
    }
    with open(gen_config, "w") as f:
        json.dump(gen_data, f, indent=2)
    print("  Created generation_config.json: eos_token_id = 151645")

    # Create/fix chat_template.jinja with chess grandmaster prompt
    # CRITICAL: Use the golden template, not a simplified version!
    chat_template = model_path / "chat_template.jinja"
    if CHESS_CHAT_TEMPLATE_FILE.exists():
        with open(CHESS_CHAT_TEMPLATE_FILE) as src:
            template_content = src.read()
        with open(chat_template, "w") as dst:
            dst.write(template_content)
        print(f"  Copied chat_template.jinja from {CHESS_CHAT_TEMPLATE_FILE}")
    else:
        raise FileNotFoundError(f"CRITICAL: {CHESS_CHAT_TEMPLATE_FILE} not found! Cannot proceed without golden template.")
    print("  Created chat_template.jinja with chess grandmaster prompt")


# =============================================================================
# STEP 3: LOCAL EVALUATION
# =============================================================================

def run_evaluation(model_path: Path, n_positions: int = 50, depth: int = 15) -> dict:
    """Run local ACPL evaluation"""
    print()
    print("=" * 60)
    print("STEP 3: LOCAL ACPL EVALUATION")
    print("=" * 60)

    eval_script = PROJECT_DIR / "evaluate_final_model.py"
    if not eval_script.exists():
        print(f"ERROR: {eval_script} not found!")
        return None

    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"eval_{model_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Model: {model_path}")
    print(f"Positions: {n_positions}")
    print(f"Stockfish depth: {depth}")
    print()

    cmd = [
        sys.executable,
        str(eval_script),
        "--model-path", str(model_path),
        "--positions", str(n_positions),
        "--depth", str(depth),
        "--output", str(results_file),
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Evaluation failed!")
        return None

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        acpl = results.get("acpl_legal_only", results.get("acpl", float("inf")))
        legal_rate = results.get("legal_move_rate", 0)

        print()
        print("=" * 40)
        print(f"RESULTS: ACPL = {acpl:.1f}, Legal = {legal_rate*100:.1f}%")
        print(f"BASELINE: ACPL = {ACPL_BASELINE} (submission 307752)")
        if acpl < ACPL_BASELINE:
            print(f"IMPROVEMENT: {ACPL_BASELINE - acpl:.1f} centipawns better!")
        else:
            print(f"REGRESSION: {acpl - ACPL_BASELINE:.1f} centipawns worse")
        print("=" * 40)

        return results

    return None


# =============================================================================
# STEP 4: UPLOAD TO HUGGINGFACE
# =============================================================================

def upload_to_huggingface(model_path: Path, repo_id: str, commit_message: str = None) -> bool:
    """Upload merged model to HuggingFace"""
    print()
    print("=" * 60)
    print("STEP 4: UPLOAD TO HUGGINGFACE")
    print("=" * 60)

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set!")
        return False

    print(f"Model: {model_path}")
    print(f"Repo: {repo_id}")
    print()

    api = HfApi(token=token)

    # Create repo if needed
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

    # Upload
    if commit_message is None:
        commit_message = f"Upload {model_path.name}"

    print("Uploading...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message=commit_message,
    )

    print(f"\nUploaded to: https://huggingface.co/{repo_id}")
    return True


# =============================================================================
# STEP 5: SUBMIT TO AICROWD
# =============================================================================

def verify_hf_repo(repo_id: str) -> bool:
    """Run mandatory pre-submission checks on HF repo"""
    print()
    print("PRE-SUBMISSION VERIFICATION")
    print("-" * 40)

    import requests

    base_url = f"https://huggingface.co/{repo_id}/raw/main"

    # Check 1: chat_template.jinja has chess grandmaster
    print("1. Checking chat_template.jinja...")
    try:
        r = requests.get(f"{base_url}/chat_template.jinja")
        if "chess grandmaster" in r.text.lower():
            print("   PASS: Found 'chess grandmaster' in template")
        else:
            print("   FAIL: 'chess grandmaster' not found!")
            return False
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    # Check 2: tokenizer_config.json eos_token
    print("2. Checking tokenizer_config.json...")
    try:
        r = requests.get(f"{base_url}/tokenizer_config.json")
        tc = r.json()
        eos = tc.get("eos_token")
        if eos == "<|im_end|>":
            print(f"   PASS: eos_token = {eos}")
        else:
            print(f"   FAIL: eos_token = {eos}, expected <|im_end|>")
            return False
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    # Check 3: generation_config.json eos_token_id
    print("3. Checking generation_config.json...")
    try:
        r = requests.get(f"{base_url}/generation_config.json")
        gc = r.json()
        eos_id = gc.get("eos_token_id")
        if eos_id == 151645:
            print(f"   PASS: eos_token_id = {eos_id}")
        else:
            print(f"   FAIL: eos_token_id = {eos_id}, expected 151645")
            return False
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    print()
    print("ALL CHECKS PASSED")
    return True


def submit_to_aicrowd(repo_id: str, dry_run: bool = False) -> bool:
    """Submit to AIcrowd contest"""
    print()
    print("=" * 60)
    print("STEP 5: SUBMIT TO AICROWD")
    print("=" * 60)

    # Verify HF repo first
    if not verify_hf_repo(repo_id):
        print("\nPre-submission verification FAILED. Fix issues before submitting.")
        return False

    if dry_run:
        print("\nDRY RUN - Would submit with these settings:")
        print(f"  HF Repo: {repo_id}")
        print("  --num-games 1")
        print("  --neuron.model-type qwen2")
        print("  --vllm.max-model-len 4096")
        return True

    # Determine model type based on repo name
    model_type = "qwen2"  # Qwen2.5 uses qwen2 backend

    prompt_template = PROJECT_DIR / "starter-kit" / "player_agents" / "v6_chess_template.jinja"
    if not prompt_template.exists():
        print(f"WARNING: Prompt template not found: {prompt_template}")
        prompt_template = None

    cmd = [
        "aicrowd", "submit-model",
        "--challenge", "global-chess-challenge-2025",
        "--hf-repo", repo_id,
        "--hf-repo-tag", "main",
        "--neuron.model-type", model_type,
        "--num-games", "1",
        "--vllm.max-model-len", "4096",
        "--vllm-inference.max-tokens", "64",
    ]

    if prompt_template:
        cmd.extend(["--prompt-template-path", str(prompt_template)])

    print("\nSubmitting with command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)
    return result.returncode == 0


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chess Model Checkpoint Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--studio", default="chess-llm-v5",
                        help="Lightning.ai studio name (default: chess-llm-v5)")
    parser.add_argument("--checkpoint", type=int, required=True,
                        help="Checkpoint number to process")
    parser.add_argument("--base-model",
                        help="Base model (default: auto-detect from studio)")
    parser.add_argument("--run-name",
                        help="Training run name (default: auto-detect)")
    parser.add_argument("--hf-repo",
                        help="HuggingFace repo ID (default: ericksoa/chess-{studio}-{checkpoint})")

    parser.add_argument("--skip-to", choices=["download", "merge", "eval", "upload", "submit"],
                        help="Skip to specific step (assumes previous steps done)")
    parser.add_argument("--eval-positions", type=int, default=50,
                        help="Number of positions for evaluation (default: 50)")
    parser.add_argument("--eval-depth", type=int, default=15,
                        help="Stockfish depth for evaluation (default: 15)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually upload or submit")
    parser.add_argument("--force", action="store_true",
                        help="Skip confirmation prompts")

    args = parser.parse_args()

    # Defaults
    base_model = args.base_model or BASE_MODELS.get(args.studio, "Qwen/Qwen2.5-7B")
    hf_repo = args.hf_repo or f"ericksoa/chess-{args.studio.replace('chess-llm-', '')}-{args.checkpoint}"

    print("=" * 60)
    print("CHESS MODEL CHECKPOINT PIPELINE")
    print("=" * 60)
    print(f"Studio: {args.studio}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Base model: {base_model}")
    print(f"HF repo: {hf_repo}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    skip_to = args.skip_to or "download"
    steps = ["download", "merge", "eval", "upload", "submit"]
    start_idx = steps.index(skip_to)

    lora_path = MODELS_DIR / f"{args.studio}_checkpoint_{args.checkpoint}"
    merged_path = MODELS_DIR / f"{args.studio}_checkpoint_{args.checkpoint}_merged"

    # STEP 1: Download
    if start_idx <= 0:
        lora_path = download_lora(args.studio, args.checkpoint, args.run_name)
        if lora_path is None:
            print("\nDownload failed!")
            return 1
    else:
        print(f"\nSkipping download, using: {lora_path}")
        if not lora_path.exists():
            print(f"ERROR: {lora_path} does not exist!")
            return 1

    # STEP 2: Merge
    if start_idx <= 1:
        merged_path = merge_model(lora_path, base_model)
        if merged_path is None:
            print("\nMerge failed!")
            return 1
    else:
        print(f"\nSkipping merge, using: {merged_path}")
        if not merged_path.exists():
            print(f"ERROR: {merged_path} does not exist!")
            return 1

    # STEP 3: Evaluate
    if start_idx <= 2:
        results = run_evaluation(merged_path, args.eval_positions, args.eval_depth)
        if results is None:
            print("\nEvaluation failed!")
            return 1

        acpl = results.get("acpl_legal_only", results.get("acpl", float("inf")))

        if acpl > ACPL_THRESHOLD:
            print(f"\nACPL ({acpl:.1f}) exceeds threshold ({ACPL_THRESHOLD})")
            if not args.force:
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Stopping pipeline.")
                    return 1

    # STEP 4: Upload
    if start_idx <= 3:
        if args.dry_run:
            print(f"\nDRY RUN: Would upload {merged_path} to {hf_repo}")
        else:
            if not args.force:
                response = input(f"\nUpload to {hf_repo}? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Skipping upload.")
                    return 0

            success = upload_to_huggingface(merged_path, hf_repo)
            if not success:
                print("\nUpload failed!")
                return 1

    # STEP 5: Submit
    if start_idx <= 4:
        if not args.force and not args.dry_run:
            response = input(f"\nSubmit {hf_repo} to AIcrowd? [y/N]: ").strip().lower()
            if response != 'y':
                print("Skipping submission.")
                return 0

        success = submit_to_aicrowd(hf_repo, args.dry_run)
        if not success:
            print("\nSubmission failed!")
            return 1

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
