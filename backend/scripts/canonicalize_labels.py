import json, argparse, sys, collections

CANON = {
    "·": "*", "⋅": "*", "•": "*", "×": "*", "В·": "*",
    "/": "÷", "÷": "÷", "Г·": "÷",
}
VOCAB = set([
    "0","1","2","3","4","5","6","7","8","9",
    "x","y","z",
    "+","-","*","÷","=",
    "(" ,")",
    "^"
])

def canon_token(t: str) -> str:
    t = t.strip()
    if t in CANON: return CANON[t]
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    bad = collections.Counter()
    kept = 0
    with open(args.inp, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            toks = [canon_token(t) for t in row["target"].split()]
            if all(t in VOCAB for t in toks):
                row["target"] = " ".join(toks)
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                for t in toks:
                    if t not in VOCAB:
                        bad[t]+=1
    print(f"wrote {kept} rows to {args.out}")
    if bad:
        print("UNKNOWN TOKENS (top20):", bad.most_common(20))

if __name__ == "__main__":
    main()
