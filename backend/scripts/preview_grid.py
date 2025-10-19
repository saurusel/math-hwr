import os, argparse
from PIL import Image

def make_grid(image_paths, cols=8, cell_bg=255):
    imgs = [Image.open(p).convert('L') for p in image_paths]
    if not imgs:
        raise SystemExit("No images found")
    h = max(im.height for im in imgs)
    w = max(im.width for im in imgs)
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new('L', (cols*w, rows*h), color=cell_bg)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas.paste(im, (c*w, r*h))
    return canvas

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="Например data/synth/train/images")
    ap.add_argument("--out", type=str, default="grid.png")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    files = [os.path.join(args.dir, fn) for fn in os.listdir(args.dir) if fn.lower().endswith(".png")]
    files = sorted(files)[:args.n]
    grid = make_grid(files)
    grid.save(args.out)
    print("Saved:", args.out)
