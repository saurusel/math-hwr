from typing import List

def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n==0: return m
    if m==0: return n
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return dp[m]

def cer(ref: str, hyp: str) -> float:
    a = list(ref)
    b = list(hyp)
    dist = _levenshtein(a,b)
    return dist / max(1,len(a))

def wer(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    dist = _levenshtein(ref_tokens, hyp_tokens)
    return dist / max(1,len(ref_tokens))
