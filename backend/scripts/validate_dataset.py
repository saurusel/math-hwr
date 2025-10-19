# --- path bootstrap ---
import os, sys, json, re
from typing import List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

OPS = {"+","-","*","÷","="}

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def tokens_ok(tokens: List[str]) -> bool:
    if tokens[0] in OPS or tokens[-1] in OPS:
        return False
    if tokens.count("=") > 1:
        return False
    for i in range(len(tokens)-1):
        a, b = tokens[i], tokens[i+1]
        if a in "0123456789xyz" and b in "xyz":
            return False
        if a in "xyz" and b in "0123456789":
            return False
    return True

def main(base="data/synth"):
    bad = 0
    total = 0
    for split in ("train","val","test"):
        jpath = os.path.join(base, split, "labels.jsonl")
        if not os.path.isfile(jpath):
            print(f"[WARN] missing {jpath}")
            continue
        for row in read_jsonl(jpath):
            total += 1
            toks = row["target"].split()
            if not tokens_ok(toks):
                bad += 1
    if bad == 0:
        print(f"OK: {total} samples pass constraints.")
    else:
        print(f"FAIL: {bad}/{total} violate constraints.")
        raise SystemExit(1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="data/synth")
    args = ap.parse_args()
    main(args.base)
