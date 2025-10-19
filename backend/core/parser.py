# Minimal stub for validity checks & future AST parsing
# For now: a very light-weight checker that ensures tokens are from vocab and brackets count roughly match.
from .tokenizer import TOKEN_LIST
def is_valid_token_stream(tokens):
    if not all(t in TOKEN_LIST for t in tokens):
        return False
    stack = []
    for t in tokens:
        if t in ("(", "{"): stack.append(t)
        if t == ")":
            if not stack or stack.pop() != "(": return False
        if t == "}":
            if not stack or stack.pop() != "{": return False
    return len(stack)==0
