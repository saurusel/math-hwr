# Tokenizer & normalization (Grammar v2: no 'frac', exponent has single-token inside parentheses)
# Vocab:
#  digits: 0-9
#  vars:   x y z
#  ops:    + - * ÷ =
#  paren:  ( )
#  pow:    ^
#
#  Additional normalization accepted in inputs/labels and mapped to canonical tokens:
#    '·', '⋅', '•', '×', 'В·' -> '*'
#    '/', '÷', 'Г·'          -> '÷'
#    whitespace variations are ignored (split by spaces)
#
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
