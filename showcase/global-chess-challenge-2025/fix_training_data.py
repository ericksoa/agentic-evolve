#!/usr/bin/env python3
"""
Fix training data to use <uci_move> format instead of special token format.

Converts: <e2><e4> -> <uci_move>e2e4</uci_move>
"""
import json
import re

INPUT_FILE = "data_special/training_30k_special.jsonl"
OUTPUT_FILE = "data_special/training_30k_uci.jsonl"

def convert_special_to_uci(completion: str) -> str:
    """Convert <e2><e4> format to <uci_move>e2e4</uci_move> format."""
    # Pattern: <from><to> with optional promotion <White_Queen> etc
    pattern = r'<([a-h][1-8])><([a-h][1-8])>(?:<(?:White|Black)_(Queen|Rook|Bishop|Knight)>)?'
    match = re.match(pattern, completion.strip())

    if not match:
        return completion  # Return as-is if doesn't match

    from_sq = match.group(1)
    to_sq = match.group(2)
    promotion = match.group(3)

    uci = from_sq + to_sq
    if promotion:
        promo_map = {'Queen': 'q', 'Rook': 'r', 'Bishop': 'b', 'Knight': 'n'}
        uci += promo_map.get(promotion, '')

    return f"<uci_move>{uci}</uci_move>"

# Process file
converted = 0
failed = 0

with open(INPUT_FILE) as f_in, open(OUTPUT_FILE, 'w') as f_out:
    for line in f_in:
        data = json.loads(line)

        old_completion = data.get('completion', '')
        new_completion = convert_special_to_uci(old_completion)

        if new_completion != old_completion:
            converted += 1
        else:
            failed += 1

        data['completion'] = new_completion
        f_out.write(json.dumps(data) + '\n')

print(f"Converted: {converted}")
print(f"Unchanged: {failed}")
print(f"Output: {OUTPUT_FILE}")

# Show samples
print("\nSample conversions:")
with open(OUTPUT_FILE) as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"  {data['completion']}")
