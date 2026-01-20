#!/usr/bin/env python3
"""Upload the UCI-format trained model to HuggingFace."""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).parent / '.env')

from lightning_sdk import Studio
from huggingface_hub import HfApi

# Config
STUDIO_NAME = "chess-llm-v4"
CHECKPOINT_PATH = "checkpoints/sft_uci_v1/final"
HF_REPO = "ericksoa/chess-sft-uci-v1"
LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "chess-sft-uci-v1"

# Golden chat template
GOLDEN_TEMPLATE = '''{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a chess grandmaster. Analyze positions and select the best move.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are a chess grandmaster. Analyze positions and select the best move.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
'''


def main():
    print("=" * 60)
    print("UPLOAD UCI MODEL TO HUGGINGFACE")
    print("=" * 60)

    # Connect to studio
    print("\n1. Connecting to Lightning studio...")
    studio = Studio(
        name=STUDIO_NAME,
        teamspace=os.environ.get('LIGHTNING_TEAMSPACE'),
        user=os.environ.get('LIGHTNING_AI_USERNAME'),
    )
    print(f"   Connected to {STUDIO_NAME}")

    # Fix files on studio
    print("\n2. Fixing chat_template.jinja on studio...")
    studio.run(f"cat > {CHECKPOINT_PATH}/chat_template.jinja << 'EOF'\n{GOLDEN_TEMPLATE}\nEOF")

    # Verify grandmaster is in template
    result = studio.run(f"grep -c 'grandmaster' {CHECKPOINT_PATH}/chat_template.jinja")
    if '2' not in result:
        print(f"   WARNING: grandmaster not found enough times in template: {result}")
    else:
        print(f"   OK: grandmaster found in template")

    print("\n3. Fixing generation_config.json...")
    studio.run(f'''python3 -c "
import json
with open('{CHECKPOINT_PATH}/generation_config.json') as f:
    config = json.load(f)
config['eos_token_id'] = 151645
config['do_sample'] = False
with open('{CHECKPOINT_PATH}/generation_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"''')
    result = studio.run(f"cat {CHECKPOINT_PATH}/generation_config.json")
    print(f"   {result.strip()}")

    # Download model locally
    print("\n4. Downloading model from studio...")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # List files
    files = studio.run(f"ls {CHECKPOINT_PATH}").strip().split()
    print(f"   Files to download: {files}")

    # Download each file
    for filename in files:
        if filename.endswith('.bin') and 'training' in filename:
            continue  # Skip training_args.bin
        print(f"   Downloading {filename}...")
        content = studio.run(f"cat {CHECKPOINT_PATH}/{filename}")
        with open(LOCAL_MODEL_DIR / filename, 'wb' if filename.endswith('.safetensors') else 'w') as f:
            if filename.endswith('.safetensors'):
                # Binary file - need different approach
                pass
            else:
                f.write(content)

    # For safetensors, we need to use lightning's file download
    print("\n   Safetensors needs direct download. Using rsync...")

    # Actually, let's just upload directly from the studio to HF
    print("\n5. Uploading to HuggingFace from studio...")

    # Upload via studio
    upload_script = f'''
from huggingface_hub import HfApi, upload_folder
import os

api = HfApi()
api.create_repo("{HF_REPO}", exist_ok=True)

upload_folder(
    folder_path="{CHECKPOINT_PATH}",
    repo_id="{HF_REPO}",
    repo_type="model",
)
print("Upload complete!")
'''

    print("   Running upload on studio...")
    result = studio.run(f"python3 -c '{upload_script}'")
    print(f"   {result}")

    print("\n" + "=" * 60)
    print(f"Model uploaded to: https://huggingface.co/{HF_REPO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
