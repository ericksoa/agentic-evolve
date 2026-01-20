#!/bin/bash
# =============================================================================
# SFT UCI Format Chess Model Submission Script
# =============================================================================
#
# This model uses special token encoding for INPUT and outputs <uci_move>e2e4</uci_move>
# Model: Qwen3-0.6B trained on 30k elite positions (corrected output format)
#
# =============================================================================

set -e

# Configuration
CHALLENGE="global-chess-challenge-2025"
HF_REPO="ericksoa/chess-sft-uci-v1"
HF_REPO_TAG="main"
PROMPT_TEMPLATE="aicrowd_baselines/data_preparation/chess_encode_special_tokens.jinja"

# Neuron/vLLM settings
MODEL_TYPE="qwen2"
MAX_MODEL_LEN="4096"
NUM_GAMES="1"
MAX_TOKENS="64"

echo "============================================"
echo "MANDATORY PRE-SUBMISSION VERIFICATION"
echo "============================================"

echo ""
echo "1. Checking chat_template.jinja system prompt..."
SYSTEM_PROMPT=$(curl -s "https://huggingface.co/$HF_REPO/raw/main/chat_template.jinja" | grep -o "You are a [^<]*" | head -1)
echo "   Found: $SYSTEM_PROMPT"
if [[ "$SYSTEM_PROMPT" != *"chess grandmaster"* ]]; then
    echo ""
    echo "   ❌ FAILED: System prompt is NOT chess grandmaster!"
    exit 1
fi
echo "   ✓ PASSED"

echo ""
echo "2. Checking tokenizer_config.json eos_token..."
EOS_TOKEN=$(curl -s "https://huggingface.co/$HF_REPO/raw/main/tokenizer_config.json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('eos_token'))")
echo "   Found: $EOS_TOKEN"
if [[ "$EOS_TOKEN" != "<|im_end|>" ]]; then
    echo ""
    echo "   ❌ FAILED: eos_token is NOT <|im_end|>!"
    exit 1
fi
echo "   ✓ PASSED"

echo ""
echo "3. Checking generation_config.json eos_token_id..."
EOS_TOKEN_ID=$(curl -s "https://huggingface.co/$HF_REPO/raw/main/generation_config.json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('eos_token_id'))")
echo "   Found: $EOS_TOKEN_ID"
if [[ "$EOS_TOKEN_ID" != "151645" ]]; then
    echo ""
    echo "   ❌ FAILED: eos_token_id is NOT 151645 (<|im_end|>)!"
    exit 1
fi
echo "   ✓ PASSED"

echo ""
echo "============================================"
echo "ALL PRE-SUBMISSION CHECKS PASSED"
echo "============================================"
echo ""
echo "============================================"
echo "SFT UCI Format Model Submission"
echo "============================================"
echo "HF Repo: $HF_REPO"
echo "Prompt: $PROMPT_TEMPLATE (SPECIAL TOKEN FORMAT)"
echo "Model Type: $MODEL_TYPE"
echo "Output Format: <uci_move>e2e4</uci_move>"
echo ""
echo "Settings:"
echo "  neuron.model-type: $MODEL_TYPE"
echo "  num-games: $NUM_GAMES"
echo "  vllm.max-model-len: $MAX_MODEL_LEN"
echo "  vllm-inference.max-tokens: $MAX_TOKENS"
echo "============================================"
echo ""
read -p "Proceed with submission? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
fi

# Submit
aicrowd submit-model \
    --challenge "$CHALLENGE" \
    --hf-repo "$HF_REPO" \
    --hf-repo-tag "$HF_REPO_TAG" \
    --prompt-template-path "$PROMPT_TEMPLATE" \
    --neuron.model-type "$MODEL_TYPE" \
    --num-games "$NUM_GAMES" \
    --vllm.max-model-len "$MAX_MODEL_LEN" \
    --vllm-inference.max-tokens "$MAX_TOKENS"

echo ""
echo "============================================"
echo "Submission complete!"
echo "============================================"
echo "Check status at: https://www.aicrowd.com/challenges/global-chess-challenge-2025/submissions"
