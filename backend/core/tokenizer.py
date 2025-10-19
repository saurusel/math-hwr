# Unified vocabulary and simple helpers
TOKEN_LIST = [
    "0","1","2","3","4","5","6","7","8","9",
    "x","y","z",
    "+","-","·","=",
    "(",")","{","}",",",
    "^","f","r","a","c"
]
TOK2ID = {t:i for i,t in enumerate(TOKEN_LIST)}
ID2TOK = {i:t for t,i in TOK2ID.items()}

def normalize_tokens(tokens):
    # join tokens with spaces where appropriate (minimal normalizer)
    return " ".join(tokens)
