# --- path bootstrap ---
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
# -----------------------------------------------

import os, json, argparse, random, re
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): 
        return x

from backend.core.grammar import gen_expression, serialize
from backend.core.synth_assets import SymbolBank
from backend.core.rasterizer import render_expr

OPS = {"+","-","*","÷","="}

def ensure_dirs(base):
    os.makedirs(os.path.join(base,"train","images"), exist_ok=True)
    os.makedirs(os.path.join(base,"val","images"), exist_ok=True)
    os.makedirs(os.path.join(base,"test","images"), exist_ok=True)

def split_counts(n):
    n_train = int(n*0.8)
    n_val = int(n*0.1)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def tokens_ok(tokens: list[str]) -> bool:
    # 1) cannot start or end with operator
    if tokens[0] in OPS or tokens[-1] in OPS:
        return False
    # 2) equality at most once
    if tokens.count("=") > 1:
        return False
    # 3) no implicit concatenation like '5x' in tokens — must appear as '5','*','x'
    #    (Our generator already enforces BinOp('*'), here just double-check around digits/vars adjacency)
    for i in range(len(tokens)-1):
        a, b = tokens[i], tokens[i+1]
        if a in "0123456789xyz" and b in "xyz":
            return False
        if a in "xyz" and b in "0123456789":
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_dir", type=str, required=True, help="Путь к папкам классов Kaggle (0..9, plus, minus, times, divide, equals, x, y, z)")
    ap.add_argument("--out", type=str, default="data/synth", help="куда сохранить датасет")
    ap.add_argument("--n", type=int, default=100000, help="сколько сэмплов сгенерировать")
    ap.add_argument("--seed", type=int, default=20251019)
    ap.add_argument("--img_h", type=int, default=64)
    ap.add_argument("--img_w_max", type=int, default=512)
    ap.add_argument("--len_min", type=int, default=5)
    ap.add_argument("--len_max", type=int, default=20)
    ap.add_argument("--eq_prob", type=float, default=0.25, help="доля уравнений с '='")
    args = ap.parse_args()

    random.seed(args.seed)

    bank = SymbolBank(args.symbols_dir)
    for tok in list("0123456789xyz+-*÷="):
        if not bank.has_token(tok):
            print(f"[WARN] Нет образцов для токена '{tok}' — будет простой шрифт-заглушка.")

    ensure_dirs(args.out)
    n_train, n_val, n_test = split_counts(args.n)

    def gen_valid_tokens():
        # regenerate until tokens satisfy constraints and length bounds
        for _ in range(10000):
            expr = gen_expression(max_depth=2, eq_prob=args.eq_prob)
            tokens = serialize(expr)
            if not (args.len_min <= len(tokens) <= args.len_max):
                continue
            if not tokens_ok(tokens):
                continue
            return tokens, expr
        raise RuntimeError("Не удалось сгенерировать валидное выражение за разумное число попыток. Увеличь len_max или ослабь ограничения.")

    def gen_split(split_name, n_split):
        rows = []
        img_dir = os.path.join(args.out, split_name, "images")
        for i in tqdm(range(n_split), desc=f"gen {split_name}"):
            tokens, expr = gen_valid_tokens()
            img = render_expr(expr, bank, img_h=args.img_h, max_w=args.img_w_max)
            img_id = f"{i:08d}"
            img_path = os.path.join(img_dir, f"{img_id}.png")
            img.save(img_path)
            rows.append({
                "id": img_id,
                "image": f"images/{img_id}.png",
                "target": " ".join(tokens),
                "len_tokens": len(tokens)
            })
        write_jsonl(os.path.join(args.out, split_name, "labels.jsonl"), rows)

    gen_split("train", n_train)
    gen_split("val", n_val)
    gen_split("test", n_test)

    for split in ("data/hand/val","data/hand/test"):
        os.makedirs(split, exist_ok=True)
        os.makedirs(os.path.join(split, "images"), exist_ok=True)
        with open(os.path.join(split, "labels.jsonl"), "w", encoding="utf-8") as f:
            pass

    print("Done. Примеры и метки сохранены.")

if __name__ == "__main__":
    main()
