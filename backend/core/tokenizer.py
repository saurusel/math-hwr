# Tokenizer & normalization with GROUPED TOKENS
# Groups visual structures like "0^8" into single tokens instead of "0", "^", "(", "8", ")"
#
# Vocabulary now includes:
#  - Base digits: 0-9
#  - Base vars: x, y, z
#  - Grouped powers: 0^0, 0^1, ..., z^z (all combinations)
#  - Operators: + - * ÷ =
#  - Parentheses: ( )
#
# Total: 198 unique tokens
#
# Load grouped vocabulary from file
import os
_vocab_path = os.path.join(os.path.dirname(__file__), "..", "..", "grouped_vocab.txt")
if os.path.exists(_vocab_path):
    with open(_vocab_path, "r", encoding="utf-8") as f:
        TOKEN_LIST = [line.strip() for line in f if line.strip()]
else:
    # Fallback to basic vocab if file not found
    TOKEN_LIST = [
        "0","1","2","3","4","5","6","7","8","9",
        "x","y","z",
        "+","-","*","÷","=",
        "(",")",
        "^"
    ]

TOK2ID = {t:i for i,t in enumerate(TOKEN_LIST)}
ID2TOK = {i:t for t,i in TOK2ID.items()}

_MAP = {
    "·": "*", "⋅": "*", "•": "*", "×": "*", "В·": "*",
    "/": "÷", "÷": "÷", "Г·": "÷",
}

def canonicalize_token(t: str) -> str:
    t = t.strip()
    if not t:
        return t
    if t in TOK2ID:
        return t
    if t in _MAP:
        return _MAP[t]
    return t  # keep as-is; caller will validate

def canonicalize_tokens(tokens):
    return [canonicalize_token(t) for t in tokens]

def normalize_tokens(tokens):
    # Join tokens back into our space-separated canonical form
    return " ".join(tokens)
