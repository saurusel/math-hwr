from __future__ import annotations
from typing import Tuple
from PIL import Image, ImageOps, ImageDraw

from .synth_assets import SymbolBank
from .grammar import Num, Var, BinOp, Pow, Frac, Expr

def _trim_glyph(img: Image.Image, pad: int = 1) -> Image.Image:
    # Find ink bbox (any non-white after invert), crop and add small padding
    inv = ImageOps.invert(img)
    bbox = inv.getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad); y1 = min(img.height, y1 + pad)
    return img.crop((x0, y0, x1, y1))

def _resize_to_height(img: Image.Image, h: int) -> Image.Image:
    w = max(1, int(img.width * (h / max(1, img.height))))
    return img.resize((w, h), Image.BILINEAR)

def _ink_mask(glyph: Image.Image) -> Image.Image:
    # Mask is strong where ink is dark; white bg -> mask ~ 0
    return ImageOps.invert(glyph)

def _paste_ink(canvas: Image.Image, glyph: Image.Image, x: int, y: int):
    # Paste glyph using its ink mask so white backgrounds never erase previous ink
    mask = _ink_mask(glyph)
    canvas.paste(glyph, (x, y), mask)
    return canvas

def render_expr(expr: Expr, bank: SymbolBank, img_h: int = 64, max_w: int = 512) -> Image.Image:
    base_h = max(12, int(img_h * 0.6))
    margin = max(2, img_h//16)
    vpad   = max(2, img_h//16)

    def render_node(node: Expr) -> Tuple[Image.Image, int]:
        if isinstance(node, Num):
            pieces = []
            total_w = 0
            for ch in node.s:
                g = bank.sample(ch)
                if g is None:
                    g = Image.new("L", (base_h//2, base_h), 255)
                    ImageDraw.Draw(g).text((2,2), ch, fill=0)
                g = _trim_glyph(g)
                g = _resize_to_height(g, base_h)
                pieces.append(g)
                total_w += g.width + 2
            total_w = max(1, total_w + 2)
            img = Image.new("L", (total_w, img_h), 255)
            x = 0
            baseline = img_h//2 + base_h//2 - vpad
            for g in pieces:
                y = baseline - g.height
                _paste_ink(img, g, x, y)
                x += g.width + 2
            return img, baseline

        if isinstance(node, Var):
            g = bank.sample(node.s)
            if g is None:
                g = Image.new("L", (base_h//2, base_h), 255)
                ImageDraw.Draw(g).text((2,2), node.s, fill=0)
            g = _trim_glyph(g)
            g = _resize_to_height(g, base_h)
            img = Image.new("L", (g.width+2, img_h), 255)
            baseline = img_h//2 + base_h//2 - vpad
            y = baseline - g.height
            _paste_ink(img, g, 0, y)
            return img, baseline

        if isinstance(node, BinOp):
            L, bL = render_node(node.left)
            R, bR = render_node(node.right)
            op_g = bank.sample(node.op)
            if op_g is None:
                op_g = Image.new("L", (base_h//2, base_h), 255)
                ImageDraw.Draw(op_g).text((2,2), node.op, fill=0)
            op_g = _trim_glyph(op_g)
            op_g = _resize_to_height(op_g, base_h)
            gap = 4
            w = L.width + gap + op_g.width + gap + R.width
            h = max(L.height, R.height, img_h)
            img = Image.new("L", (w, h), 255)
            baseline = h//2 + base_h//2 - vpad
            # paste left
            _paste_ink(img, L, 0, h - L.height)
            # operator
            yOp = baseline - op_g.height
            _paste_ink(img, op_g, L.width + gap, yOp)
            # right
            _paste_ink(img, R, L.width + gap + op_g.width + gap, h - R.height)
            return img, baseline

        if isinstance(node, Pow):
            base_img, base_b = render_node(node.base)
            # exponent is single symbol by grammar; render then shrink
            exp_img, exp_b = render_node(node.exp)
            scale = 0.65
            exp_h_px = max(8, int(exp_img.height * scale))
            exp_w_px = max(1, int(exp_img.width * scale))
            exp_img = exp_img.resize((exp_w_px, exp_h_px), Image.BILINEAR)

            pad = 2
            w = base_img.width + pad + exp_img.width
            h = max(base_img.height, base_img.height//2 + exp_img.height)
            img = Image.new("L", (w, h), 255)
            _paste_ink(img, base_img, 0, h - base_img.height)
            x_exp = base_img.width - exp_img.width//3
            y_exp = h - base_img.height - exp_img.height + base_img.height//3
            _paste_ink(img, exp_img, x_exp, max(0, y_exp))
            baseline = h - (base_img.height - base_b)
            return img, baseline

        if isinstance(node, Frac):
            num_img, nb = render_node(node.num)
            den_img, db = render_node(node.den)
            pad = 4
            w = max(num_img.width, den_img.width) + 2*pad
            line_h = 2
            h = num_img.height//2 + den_img.height//2 + line_h + 2*pad
            img = Image.new("L", (w, h), 255)
            draw = ImageDraw.Draw(img)
            # numerator
            nx = (w - num_img.width)//2
            ny = pad
            _paste_ink(img, num_img, nx, ny)
            # fraction bar
            ly = ny + num_img.height + 2
            draw.rectangle([pad, ly, w-pad, ly+line_h], fill=0)
            # denominator
            dx = (w - den_img.width)//2
            dy = ly + line_h + 2
            _paste_ink(img, den_img, dx, dy)
            baseline = dy + den_img.height//2 + 4
            return img, baseline

        raise TypeError("unknown node")

    img, baseline = render_node(expr)
    # add side margins and constrain width
    w = img.width + 2*margin
    if w > max_w:
        scale = (max_w - 2*margin) / max(1, img.width)
        img = img.resize((max(1, int(img.width*scale)), max(1, int(img.height*scale))), Image.BILINEAR)
        w = img.width + 2*margin
    canvas = Image.new("L", (w, img_h), 255)
    y = (img_h - img.height)//2
    _paste_ink(canvas, img, margin, y)
    return canvas
