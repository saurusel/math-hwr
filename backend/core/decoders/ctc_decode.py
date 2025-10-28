import torch
from typing import List
def ctc_greedy_decode(logits, blank_id: int) -> List[List[int]]:
    # logits: (T,B,C) log-probs or raw scores
    # returns list of sequences (per batch), collapsed & without blanks
    with torch.no_grad():
        best = logits.argmax(dim=-1)  # (T,B)
        T, B = best.size(0), best.size(1)
        out = []
        for b in range(B):
            seq = []
            prev = None
            for t in range(T):
                k = int(best[t,b].item())
                if k != blank_id and k != prev:
                    seq.append(k)
                prev = k
            out.append(seq)
        return out
