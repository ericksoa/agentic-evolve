# Task 22eb0ac0

## Pattern
Fill rows where left and right edge markers match with that color.

## Algorithm
Check each row's first and last elements. If they're equal, fill the entire row
with that value; otherwise keep the row unchanged. The grid is always 10 columns
wide, so we can hardcode `*10` instead of using `len(r)`.

## Key Tricks
- `r[0]==r[-1]` checks both endpoints match (including 0==0 case which is fine since filling with 0s doesn't change an all-zero row)
- `*10` hardcoded grid width instead of `*len(r)` saves 4 bytes
- Lambda expression for minimal overhead
- Conditional expression `[...]*10if...else r` for inline selection

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 57 | Initial lambda solution |
