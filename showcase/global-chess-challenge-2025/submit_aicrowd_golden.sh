#!/bin/bash
# =============================================================================
# GOLDEN PATH SUBMISSION SCRIPT - AIcrowd Chess Challenge 2025
# =============================================================================
#
# THIS IS THE ONLY WAY TO SUBMIT. NO EXCEPTIONS.
#
# Created: 2026-01-20 after wasting 1+ hour debugging encoding mismatch
#
# =============================================================================
# CRITICAL FACTS (DO NOT FORGET):
# =============================================================================
#
# 1. MODEL: Qwen3-0.6B with EXTENDED VOCAB (151746 tokens)
#    - Trained on special token encoding: <chess_position><a1><blank>...
#    - NOT Qwen2.5, NOT standard prompts
#
# 2. TOKENIZER: Custom chess tokenizer with special tokens
#    - Located: aicrowd_baselines/data_preparation/chess_tokenizer_qwen3/
#    - Tokens: <chess_position>, <a1>-<h8>, <White_Pawn>, <blank>, etc.
#
# 3. PROMPT TEMPLATE: special_tokens_prompt.jinja (MUST USE THIS)
#    - Encodes FEN to special token format
#    - Located: starter-kit/player_agents/special_tokens_prompt.jinja
#
# 4. MODEL TYPE: qwen3 (NOT qwen2!)
#
# 5. OUTPUT FORMAT: <think>...</think>\n<uci_move>xxxx</uci_move>
#
# =============================================================================
# USAGE:
# =============================================================================
#
#   ./submit_aicrowd_golden.sh <checkpoint_number>
#
# Examples:
#   ./submit_aicrowd_golden.sh 160000
#   ./submit_aicrowd_golden.sh 170000
#   ./submit_aicrowd_golden.sh 180000
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION - DO NOT CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING
# =============================================================================

CHALLENGE="global-chess-challenge-2025"
PROMPT_TEMPLATE="starter-kit/player_agents/special_tokens_prompt.jinja"
MODEL_TYPE="qwen3"
MAX_MODEL_LEN="4096"
MAX_TOKENS="150"
NUM_GAMES="1"

# Lightning.ai studio where checkpoints live
STUDIO_NAME="chess-llm-v4"
CHECKPOINT_BASE_PATH="checkpoints/aicrowd_full_v1"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

if [ -z "$1" ]; then
    echo -e "${RED}ERROR: Checkpoint number required${NC}"
    echo ""
    echo "Usage: $0 <checkpoint_number>"
    echo ""
    echo "Examples:"
    echo "  $0 160000"
    echo "  $0 170000"
    echo ""
    exit 1
fi

CHECKPOINT_NUM="$1"
CHECKPOINT_PATH="${CHECKPOINT_BASE_PATH}/checkpoint-${CHECKPOINT_NUM}"
HF_REPO="ericksoa/chess-qwen3-${CHECKPOINT_NUM}"

echo "============================================"
echo -e "${GREEN}GOLDEN PATH SUBMISSION${NC}"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT_NUM}"
echo "HF Repo: ${HF_REPO}"
echo "Model Type: ${MODEL_TYPE}"
echo "Prompt Template: ${PROMPT_TEMPLATE}"
echo "============================================"
echo ""

# =============================================================================
# STEP 1: VERIFY PROMPT TEMPLATE EXISTS
# =============================================================================

echo -e "${YELLOW}[1/4] Verifying prompt template...${NC}"

if [ ! -f "$PROMPT_TEMPLATE" ]; then
    echo -e "${RED}ERROR: Prompt template not found: $PROMPT_TEMPLATE${NC}"
    exit 1
fi

# Verify it's the special tokens template
if ! grep -q "<chess_position>" "$PROMPT_TEMPLATE"; then
    echo -e "${RED}ERROR: Prompt template missing <chess_position> - wrong template!${NC}"
    exit 1
fi

echo -e "${GREEN}  OK: special_tokens_prompt.jinja verified${NC}"
echo ""

# =============================================================================
# STEP 2: UPLOAD CHECKPOINT TO HUGGINGFACE
# =============================================================================

echo -e "${YELLOW}[2/4] Uploading checkpoint to HuggingFace...${NC}"

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

python3 << PYTHON
import os
import sys

# Load env
env_path = '.env'
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from lightning_sdk import Studio
from huggingface_hub import HfApi

CHECKPOINT_NUM = "${CHECKPOINT_NUM}"
CHECKPOINT_PATH = "${CHECKPOINT_PATH}"
HF_REPO = "${HF_REPO}"
STUDIO_NAME = "${STUDIO_NAME}"

print(f"Connecting to studio {STUDIO_NAME}...")
studio = Studio(
    name=STUDIO_NAME,
    teamspace=os.environ.get('LIGHTNING_TEAMSPACE'),
    user=os.environ.get('LIGHTNING_AI_USERNAME'),
)

# Verify checkpoint exists
result = studio.run(f"ls /teamspace/studios/this_studio/{CHECKPOINT_PATH}/model.safetensors 2>/dev/null && echo EXISTS || echo MISSING")
if "MISSING" in result:
    print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"Checkpoint verified on studio")

# Create temp dir and download checkpoint files
import tempfile
import shutil

with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Downloading checkpoint files...")

    # Files needed for inference (NOT optimizer, scheduler, etc.)
    files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]

    for f in files:
        remote_path = f"/teamspace/studios/this_studio/{CHECKPOINT_PATH}/{f}"
        local_path = os.path.join(tmpdir, f)
        print(f"  Downloading {f}...")
        try:
            studio.download_file(remote_path.replace("/teamspace/studios/this_studio/", ""), local_path)
        except Exception as e:
            print(f"  Warning: Could not download {f}: {e}")

    # Upload to HuggingFace
    print(f"Uploading to HuggingFace: {HF_REPO}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=HF_REPO, exist_ok=True, private=False)

    api.upload_folder(
        folder_path=tmpdir,
        repo_id=HF_REPO,
        commit_message=f"Upload checkpoint-{CHECKPOINT_NUM} for AIcrowd submission"
    )

    print(f"Uploaded to: https://huggingface.co/{HF_REPO}")

PYTHON

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Upload failed${NC}"
    exit 1
fi

echo -e "${GREEN}  OK: Uploaded to HuggingFace${NC}"
echo ""

# =============================================================================
# STEP 3: VERIFY HUGGINGFACE REPO
# =============================================================================

echo -e "${YELLOW}[3/4] Verifying HuggingFace repo...${NC}"

# Check model.safetensors exists
MODEL_CHECK=$(curl -s -o /dev/null -w "%{http_code}" "https://huggingface.co/${HF_REPO}/resolve/main/model.safetensors")
if [ "$MODEL_CHECK" != "302" ] && [ "$MODEL_CHECK" != "200" ]; then
    echo -e "${RED}ERROR: model.safetensors not found in HF repo${NC}"
    exit 1
fi
echo -e "${GREEN}  OK: model.safetensors exists${NC}"

# Check config.json has qwen3
CONFIG_CHECK=$(curl -s "https://huggingface.co/${HF_REPO}/raw/main/config.json" | grep -o '"model_type":\s*"[^"]*"')
echo "  Config: $CONFIG_CHECK"
if [[ "$CONFIG_CHECK" != *"qwen3"* ]]; then
    echo -e "${RED}ERROR: config.json does not have model_type=qwen3${NC}"
    exit 1
fi
echo -e "${GREEN}  OK: model_type is qwen3${NC}"

# Check tokenizer has chess tokens
TOKENIZER_CHECK=$(curl -s "https://huggingface.co/${HF_REPO}/raw/main/added_tokens.json" | grep -c "chess_position" || true)
if [ "$TOKENIZER_CHECK" -lt 1 ]; then
    echo -e "${RED}ERROR: Tokenizer missing chess special tokens${NC}"
    exit 1
fi
echo -e "${GREEN}  OK: Chess special tokens present${NC}"

echo ""

# =============================================================================
# STEP 4: SUBMIT TO AICROWD
# =============================================================================

echo -e "${YELLOW}[4/4] Submitting to AIcrowd...${NC}"
echo ""
echo "Command:"
echo "  aicrowd submit-model \\"
echo "    --challenge ${CHALLENGE} \\"
echo "    --hf-repo ${HF_REPO} \\"
echo "    --hf-repo-tag main \\"
echo "    --prompt-template-path ${PROMPT_TEMPLATE} \\"
echo "    --neuron.model-type ${MODEL_TYPE} \\"
echo "    --num-games ${NUM_GAMES} \\"
echo "    --vllm.max-model-len ${MAX_MODEL_LEN} \\"
echo "    --vllm-inference.max-tokens ${MAX_TOKENS}"
echo ""

read -p "Proceed with submission? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
fi

aicrowd submit-model \
    --challenge "${CHALLENGE}" \
    --hf-repo "${HF_REPO}" \
    --hf-repo-tag main \
    --prompt-template-path "${PROMPT_TEMPLATE}" \
    --neuron.model-type "${MODEL_TYPE}" \
    --num-games "${NUM_GAMES}" \
    --vllm.max-model-len "${MAX_MODEL_LEN}" \
    --vllm-inference.max-tokens "${MAX_TOKENS}"

echo ""
echo "============================================"
echo -e "${GREEN}SUBMISSION COMPLETE${NC}"
echo "============================================"
echo ""
echo "REMEMBER: This model uses SPECIAL TOKEN ENCODING"
echo "  - <chess_position><a1><blank>..."
echo "  - NOT standard FEN prompts"
echo "  - Output: <think>...</think><uci_move>xxxx</uci_move>"
echo ""
