#!/usr/bin/env python3
"""Fix chess tokens special flag and upload to HuggingFace."""
from huggingface_hub import hf_hub_download, HfApi
import json
import re

# Download current config
path = hf_hub_download('ericksoa/chess-sft-special-v1', 'tokenizer_config.json')
with open(path) as f:
    config = json.load(f)

decoder = config.get('added_tokens_decoder', {})

# Chess tokens pattern: squares, pieces, blank
chess_pattern = re.compile(r'^<([a-h][1-8]|White_|Black_|blank).*>$')

changed = 0
for token_id, token_config in decoder.items():
    content = token_config.get('content', '')
    if chess_pattern.match(content):
        if token_config.get('special', False):
            token_config['special'] = False
            changed += 1

print(f'Changed {changed} chess tokens from special=True to special=False')

# Save locally
with open('/tmp/tokenizer_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Upload to HuggingFace
api = HfApi()
api.upload_file(
    path_or_fileobj='/tmp/tokenizer_config.json',
    path_in_repo='tokenizer_config.json',
    repo_id='ericksoa/chess-sft-special-v1',
)
print('Uploaded fixed tokenizer_config.json to HuggingFace')
