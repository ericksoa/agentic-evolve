#!/usr/bin/env python3
"""
Comprehensive HuggingFace repo audit for chess model submissions.

Run this BEFORE any submission to catch ALL known issues at once.
"""
import sys
import json
import requests

def audit_repo(repo_id: str) -> bool:
    base_url = f"https://huggingface.co/{repo_id}/raw/main"
    errors = []
    warnings = []
    
    print(f"Auditing HuggingFace repo: {repo_id}")
    print("=" * 60)
    
    # 1. Check chat_template.jinja
    print("\n1. chat_template.jinja - system prompt")
    try:
        r = requests.get(f"{base_url}/chat_template.jinja")
        if r.status_code == 200:
            if "chess grandmaster" in r.text.lower():
                print("   ✓ Contains 'chess grandmaster'")
            elif "helpful assistant" in r.text.lower():
                errors.append("chat_template.jinja has 'helpful assistant' instead of 'chess grandmaster'")
                print("   ✗ Contains 'helpful assistant' - WRONG!")
            else:
                warnings.append("chat_template.jinja has unknown system prompt")
                print("   ? Unknown system prompt - check manually")
        else:
            print("   - No chat_template.jinja file (may use tokenizer_config)")
    except Exception as e:
        warnings.append(f"Could not check chat_template.jinja: {e}")
    
    # 2. Check tokenizer_config.json - eos_token
    print("\n2. tokenizer_config.json - eos_token")
    try:
        r = requests.get(f"{base_url}/tokenizer_config.json")
        if r.status_code == 200:
            config = r.json()
            eos = config.get("eos_token")
            if eos == "<|im_end|>":
                print(f"   ✓ eos_token = {eos}")
            elif eos == "<|endoftext|>":
                errors.append(f"tokenizer_config.json eos_token={eos}, should be <|im_end|>")
                print(f"   ✗ eos_token = {eos} - WRONG! Should be <|im_end|>")
            else:
                warnings.append(f"tokenizer_config.json eos_token={eos} - verify this is correct")
                print(f"   ? eos_token = {eos} - verify manually")
    except Exception as e:
        errors.append(f"Could not check tokenizer_config.json: {e}")
    
    # 3. Check generation_config.json - eos_token_id
    print("\n3. generation_config.json - eos_token_id")
    try:
        r = requests.get(f"{base_url}/generation_config.json")
        if r.status_code == 200:
            config = r.json()
            eos_id = config.get("eos_token_id")
            # For Qwen2.5: 151643=<|endoftext|>, 151645=<|im_end|>
            if eos_id == 151645:
                print(f"   ✓ eos_token_id = {eos_id} (<|im_end|>)")
            elif eos_id == 151643:
                errors.append(f"generation_config.json eos_token_id={eos_id} (<|endoftext|>), should be 151645 (<|im_end|>)")
                print(f"   ✗ eos_token_id = {eos_id} (<|endoftext|>) - WRONG! Should be 151645")
            else:
                warnings.append(f"generation_config.json eos_token_id={eos_id} - verify this is correct for your model")
                print(f"   ? eos_token_id = {eos_id} - verify manually")
        else:
            warnings.append("No generation_config.json found")
            print("   - No generation_config.json file")
    except Exception as e:
        warnings.append(f"Could not check generation_config.json: {e}")
    
    # 4. Check config.json - model_type
    print("\n4. config.json - model_type")
    try:
        r = requests.get(f"{base_url}/config.json")
        if r.status_code == 200:
            config = r.json()
            model_type = config.get("model_type")
            if model_type == "qwen2":
                print(f"   ✓ model_type = {model_type}")
            else:
                warnings.append(f"config.json model_type={model_type} - ensure --neuron.model-type matches")
                print(f"   ? model_type = {model_type} - ensure submission flags match")
    except Exception as e:
        warnings.append(f"Could not check config.json: {e}")
    
    # 5. Check for required files
    print("\n5. Required files")
    required = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
    for f in required:
        r = requests.head(f"{base_url}/{f}")
        if r.status_code == 200:
            print(f"   ✓ {f}")
        else:
            # Check alternative names
            if f == "model.safetensors.index.json":
                r2 = requests.head(f"{base_url}/model.safetensors")
                if r2.status_code == 200:
                    print(f"   ✓ model.safetensors (single file)")
                    continue
            errors.append(f"Missing required file: {f}")
            print(f"   ✗ {f} - MISSING!")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"AUDIT FAILED - {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nDO NOT SUBMIT until errors are fixed!")
        return False
    elif warnings:
        print(f"AUDIT PASSED with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ? {w}")
        print("\nReview warnings before submitting.")
        return True
    else:
        print("AUDIT PASSED - all checks OK")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_hf_repo.py <repo_id>")
        print("Example: python audit_hf_repo.py ericksoa/chess-qwen3b-12k")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    success = audit_repo(repo_id)
    sys.exit(0 if success else 1)
