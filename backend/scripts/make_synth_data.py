
# --- path bootstrap ---
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
# -----------------------------------------------
import os, json, argparse, random
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): 
        return x

from backend.core.grammar import gen_expression, serialize
from backend.core.synth_assets import SymbolBank
from backend.core.rasterizer import render_expr

TOKS_FROM_DS = list("0123456789") + ["x","y","z","+","-","=","*","÷","·"]
OPS_ALL = {"+","-","·","*","÷","="}

def ensure_dirs(base):
    os.makedirs(os.path.join(base,"train","images"), exist_ok=True)
    os.makedirs(os.path.join(base,"val","images"), exist_ok=True)
    os.makedirs(os.path.join(base,"test","images"), exist_ok=True)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20251019)
    ap.add_argument("--img_h", type=int, default=64)
    ap.add_argument("--img_w_max", type=int, default=512)
    ap.add_argument("--len_min", type=int, default=5)
    ap.add_argument("--len_max", type=int, default=20)
    ap.add_argument("--eq_prob", type=float, default=0.25)
    ap.add_argument("--ink_rel_pad", type=float, default=0.08)
    ap.add_argument("--mask_thresh", type=int, default=12, help="порог бинаризации маски (6..24)")
    ap.add_argument("--enable_mul", type=str, default="dot,star")
    ap.add_argument("--enable_div", type=str, default="frac,sign")
    ap.add_argument("--op_dot_h_ratio", type=float, default=0.16)
    ap.add_argument("--op_star_h_ratio", type=float, default=0.50)
    ap.add_argument("--op_dot_max_ink", type=float, default=0.10)
    ap.add_argument("--op_star_max_ink", type=float, default=0.20)
    ap.add_argument("--op_generic_max_ink", type=float, default=0.28)
    args = ap.parse_args()

    random.seed(args.seed)

    bank = SymbolBank(args.symbols_dir)
    for tok in TOKS_FROM_DS:
        if not bank.has_token(tok):
            print(f"[WARN] Нет образцов для токена '{tok}' — будет простой шрифт-заглушка.")

    mul_opts = {s.strip() for s in args.enable_mul.split(",") if s.strip()}
    div_opts = {s.strip() for s in args.enable_div.split(",") if s.strip()}
    allowed_ops = ["+","-"]
    if "dot" in mul_opts and bank.has_token("·"):  allowed_ops.append("·")
    if "star" in mul_opts and bank.has_token("*"): allowed_ops.append("*")
    if "sign" in div_opts and bank.has_token("÷"): allowed_ops.append("÷")

    def tokens_ok(tokens: list[str]) -> bool:
        if tokens[0] in OPS_ALL or tokens[-1] in OPS_ALL:
            return False
        if tokens.count("=") > 1:
            return False
        for i in range(len(tokens)-1):
            a, b = tokens[i], tokens[i+1]
            if a in "0123456789xyz" and b in "xyz": return False
            if a in "xyz" and b in "0123456789": return False
        for t in tokens:
            if t in {"+","-","·","*","÷"} and t not in allowed_ops:
                return False
        return True

    def gen_valid_expr_and_tokens():
        for _ in range(20000):
            expr = gen_expression(max_depth=2, eq_prob=args.eq_prob, allowed_ops=allowed_ops)
            tokens = serialize(expr)
            if not (args.len_min <= len(tokens) <= args.len_max): continue
            if not tokens_ok(tokens): continue
            return expr, tokens
        raise RuntimeError("Не удалось сгенерировать валидное выражение за разумное число попыток.")

    def gen_split(split_name, n_split):
        rows = []
        img_dir = os.path.join(args.out, split_name, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i in tqdm(range(n_split), desc=f"gen {split_name}"):
            expr, tokens = gen_valid_expr_and_tokens()
            img = render_expr(
                expr, bank, img_h=args.img_h, max_w=args.img_w_max,
                ink_rel_pad=args.ink_rel_pad, mask_thresh=args.mask_thresh,
                op_dot_h_ratio=args.op_dot_h_ratio, op_star_h_ratio=args.op_star_h_ratio,
                op_dot_max_ink=args.op_dot_max_ink, op_star_max_ink=args.op_star_max_ink,
                op_generic_max_ink=args.op_generic_max_ink
            )
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

    n_train = int(args.n*0.8); n_val = int(args.n*0.1); n_test = args.n - n_train - n_val
    for split, nsplit in (("train", n_train), ("val", n_val), ("test", n_test)):
        gen_split(split, nsplit)

    for split in ("data/hand/val","data/hand/test"):
        os.makedirs(split, exist_ok=True)
        os.makedirs(os.path.join(split, "images"), exist_ok=True)
        with open(os.path.join(split, "labels.jsonl"), "w", encoding="utf-8") as f:
            pass

    print("Done. Примеры и метки сохранены.")

if __name__ == "__main__":
    main()
