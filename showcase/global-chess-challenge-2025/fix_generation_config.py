#!/usr/bin/env python3
"""Fix generation_config.json for ericksoa/chess-sft-special-v1."""
from huggingface_hub import hf_hub_download, HfApi
import json

# Download current config
config_path = hf_hub_download("ericksoa/chess-sft-special-v1", "generation_config.json")
with open(config_path) as f:
    config = json.load(f)

print("Current config:", config)

# Fix eos_token_id and do_sample
config["eos_token_id"] = 151645  # Single value, not list
config["do_sample"] = False  # No sampling for deterministic play

print("Fixed config:", config)

# Save locally
with open("/tmp/generation_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Upload
api = HfApi()
api.upload_file(
    path_or_fileobj="/tmp/generation_config.json",
    path_in_repo="generation_config.json",
    repo_id="ericksoa/chess-sft-special-v1",
)
print("Uploaded fixed generation_config.json")
