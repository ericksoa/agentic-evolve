#!/bin/bash
# =============================================================================
# Qwen 3B Chess Model Submission Script
# =============================================================================
#
# ALWAYS USE THIS SCRIPT FOR ALL CHESS MODEL SUBMISSIONS
# DO NOT MANUALLY RUN aicrowd submit-model - USE THIS SCRIPT
#
# =============================================================================
# CRITICAL FLAGS - Each one was learned from a failed submission:
# =============================================================================
#
# --num-games 1              : MOST CRITICAL! Prevents Neuron concurrency
#                              corruption that causes garbage output.
#                              Submission 307752 (V6) got ~146 ACPL with this.
#                              Without it: binary garbage like '<fastcall8820ah...'
#
# --vllm.max-num-seqs 1      : Also prevents concurrency issues
#
# --vllm.max-model-len 512   : Prevents memory/compilation issues on Trainium
#
# --vllm.enforce-eager true  : More reliable execution, avoids graph issues
#
# --vllm.dtype bfloat16      : Standard precision for Neuron/Trainium
#
# --vllm-inference.max-tokens 64 : Sufficient for chess move output
#
# --neuron.model-type qwen2  : Qwen2.5 uses qwen2 backend (not qwen2.5!)
#
# =============================================================================
# PRE-SUBMISSION CHECKS - Script verifies HuggingFace repo before submitting:
# =============================================================================
#
# 1. chat_template.jinja must have "chess grandmaster" system prompt
# 2. tokenizer_config.json eos_token must be "<|im_end|>"
# 3. generation_config.json eos_token_id must be 151645 (not 151643!)
#
# =============================================================================
# REFERENCE: Submission 307752 got ~146 ACPL with these exact flags
# =============================================================================

set -e

# =============================================================================
# MANDATORY PRE-SUBMISSION VERIFICATION
# These checks MUST pass before submission. No exceptions.
# =============================================================================

echo "============================================"
echo "MANDATORY PRE-SUBMISSION VERIFICATION"
echo "============================================"

# Configuration - UPDATE THESE FOR EACH SUBMISSION
CHALLENGE="global-chess-challenge-2025"
HF_REPO="ericksoa/chess-qwen3b-12k"  # <-- UPDATE: HuggingFace repo with merged model

echo ""
echo "1. Checking chat_template.jinja system prompt..."
SYSTEM_PROMPT=$(curl -s "https://huggingface.co/$HF_REPO/raw/main/chat_template.jinja" | grep -o "You are a [^<]*" | head -1)
echo "   Found: $SYSTEM_PROMPT"
if [[ "$SYSTEM_PROMPT" != *"chess grandmaster"* ]]; then
    echo ""
    echo "   ❌ FAILED: System prompt is NOT chess grandmaster!"
    echo "   Expected: 'You are a chess grandmaster...'"
    echo "   Got: '$SYSTEM_PROMPT'"
    echo ""
    echo "   FIX: Update chat_template.jinja in $HF_REPO"
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
    echo "   Expected: '<|im_end|>'"
    echo "   Got: '$EOS_TOKEN'"
    echo ""
    echo "   FIX: Update tokenizer_config.json in $HF_REPO"
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
    echo "   Expected: '151645'"
    echo "   Got: '$EOS_TOKEN_ID'"
    echo "   NOTE: 151643 is <|endoftext|> (WRONG), 151645 is <|im_end|> (CORRECT)"
    echo ""
    echo "   FIX: Update generation_config.json in $HF_REPO"
    exit 1
fi
echo "   ✓ PASSED"

echo ""
echo "============================================"
echo "ALL PRE-SUBMISSION CHECKS PASSED"
echo "============================================"
echo ""

# Note: Login check removed - aicrowd CLI doesn't have list command
# If not logged in, the submit-model command will fail with auth error

# Additional config
HF_REPO_TAG="main"
PROMPT_TEMPLATE="starter-kit/player_agents/v6_chess_template.jinja"

# Neuron/vLLM settings - DO NOT CHANGE THESE
MODEL_TYPE="qwen2"  # Qwen2.5 uses qwen2 backend
MAX_MODEL_LEN="512"
MAX_BATCHED_TOKENS="512"
MAX_NUM_SEQS="1"
NUM_GAMES="1"  # CRITICAL: Prevents concurrency corruption on Neuron
DTYPE="bfloat16"
ENFORCE_EAGER="true"
MAX_TOKENS="64"

echo "============================================"
echo "Qwen 3B Chess Model Submission"
echo "============================================"
echo "HF Repo: $HF_REPO"
echo "HF Tag: $HF_REPO_TAG"
echo "Prompt: $PROMPT_TEMPLATE"
echo "Model Type: $MODEL_TYPE (Neuron)"
echo ""
echo "vLLM Settings (DO NOT CHANGE):"
echo "  max-model-len: $MAX_MODEL_LEN"
echo "  max-num-seqs: $MAX_NUM_SEQS"
echo "  dtype: $DTYPE"
echo "  enforce-eager: $ENFORCE_EAGER"
echo "  max-tokens: $MAX_TOKENS"
echo "============================================"
echo ""
read -p "Proceed with submission? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
fi

# Submit with ALL required flags
aicrowd submit-model \
    --challenge "$CHALLENGE" \
    --hf-repo "$HF_REPO" \
    --hf-repo-tag "$HF_REPO_TAG" \
    --prompt-template-path "$PROMPT_TEMPLATE" \
    --neuron.model-type "$MODEL_TYPE" \
    --num-games "$NUM_GAMES" \
    --vllm.max-model-len "$MAX_MODEL_LEN" \
    --vllm.max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --vllm.max-num-seqs "$MAX_NUM_SEQS" \
    --vllm.dtype "$DTYPE" \
    --vllm.enforce-eager "$ENFORCE_EAGER" \
    --vllm-inference.max-tokens "$MAX_TOKENS"

echo ""
echo "============================================"
echo "Submission complete!"
echo "============================================"
echo "Check status at: https://www.aicrowd.com/challenges/global-chess-challenge-2025/submissions"
