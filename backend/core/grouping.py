"""
Token grouping for visual structure matching.
Groups mathematical expressions into visual units (e.g., "0^8" instead of "0", "^", "(", "8", ")").
"""
from typing import List


def group_tokens(tokens: List[str]) -> List[str]:
    """
    Group tokens based on visual mathematical structure.

    Rules:
    - base^(power) → base^power (single group token)
    - Other tokens remain separate

    Examples:
        ["0", "^", "(", "8", ")"] → ["0^8"]
        ["x", "^", "(", "2", ")"] → ["x^2"]
        ["3", "+", "4"] → ["3", "+", "4"]
        ["x", "^", "(", "2", ")", "+", "1"] → ["x^2", "+", "1"]

    Args:
        tokens: List of space-separated tokens

    Returns:
        List of grouped tokens
    """
    if not tokens:
        return []

    grouped = []
    i = 0

    while i < len(tokens):
        # Check for exponent pattern: token ^ ( power )
        if i + 4 < len(tokens) and tokens[i+1] == "^" and tokens[i+2] == "(" and tokens[i+4] == ")":
            # Found exponent pattern
            base = tokens[i]
            power = tokens[i+3]
            grouped_token = f"{base}^{power}"
            grouped.append(grouped_token)
            i += 5  # Skip all 5 tokens
        else:
            # Regular token
            grouped.append(tokens[i])
            i += 1

    return grouped


def ungroup_tokens(grouped_tokens: List[str]) -> List[str]:
    """
    Reverse operation: expand grouped tokens back to individual tokens.

    Examples:
        ["0^8"] → ["0", "^", "(", "8", ")"]
        ["x^2", "+", "1"] → ["x", "^", "(", "2", ")", "+", "1"]

    Args:
        grouped_tokens: List of potentially grouped tokens

    Returns:
        List of ungrouped tokens
    """
    ungrouped = []

    for token in grouped_tokens:
        # Check if token contains ^
        if "^" in token and len(token) >= 3:
            parts = token.split("^")
            if len(parts) == 2:
                base, power = parts
                # Expand: base^power → base ^ ( power )
                ungrouped.extend([base, "^", "(", power, ")"])
            else:
                # Malformed, keep as-is
                ungrouped.append(token)
        else:
            # Regular token
            ungrouped.append(token)

    return ungrouped


def is_grouped_token(token: str) -> bool:
    """Check if a token is a grouped structure."""
    return "^" in token and len(token) >= 3


def build_grouped_vocab(expressions: List[str]) -> List[str]:
    """
    Build vocabulary of all unique grouped tokens from a list of expressions.

    Args:
        expressions: List of space-separated token sequences

    Returns:
        Sorted list of unique tokens (grouped)
    """
    unique_tokens = set()

    for expr in expressions:
        tokens = expr.split()
        grouped = group_tokens(tokens)
        unique_tokens.update(grouped)

    # Sort for consistency
    return sorted(list(unique_tokens))
