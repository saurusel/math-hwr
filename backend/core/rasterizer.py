from __future__ import annotations

import math
from typing import Tuple, List, Iterable

import numpy as np
from PIL import Image, ImageOps

from .synth_assets import SymbolBank
from .grammar import Num, Var, BinOp, Pow, Frac, Expr


# =========================
# Mask & geometry helpers
# =========================

def _binary_mask(img: Image.Image, thresh: int) -> Image.Image:
    """
    Build a binary (0/255) mask of 'ink' from a grayscale glyph image.
    Invert first (ink -> high), then threshold.
    """
    arr = np.asarray(img, dtype=np.uint8)
    inv = 255 - arr
    m = (inv > thresh).astype(np.uint8) * 255
    return Image.fromarray(m, mode="L")


def _trim_glyph(img: Image.Image, pad_px: int, rel_pad: float, thresh: int) -> Image.Image:
    """
    Trim by ink bbox computed on binary mask, then add relative padding.
    """
    m = _binary_mask(img, thresh)
    bbox = m.getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    pad_x = max(pad_px, int(w * rel_pad))
    pad_y = max(pad_px, int(h * rel_pad))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.width, x1 + pad_x)
    y1 = min(img.height, y1 + pad_y)
    return img.crop((x0, y0, x1, y1))


def _resize_ink_height(img: Image.Image, target_h: int) -> Image.Image:
    """
    Resize glyph so that its trimmed height ~= target_h.
    """
    if img.height == 0:
        return img
    scale = target_h / max(1, img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


# ---------- mask postprocessing ----------

def _remove_bottom_hairline_rows(mask_img: Image.Image,
                                 band_rel: float = 0.35,
                                 row_thick_max: int = 2,
                                 fill_ratio: float = 0.40) -> Image.Image:
    """
    Убираем 1–2 сплошных строки чернил в нижней части (антиалиас-полоска).
    Работает на записываемой копии массива.
    """
    m = np.asarray(mask_img, dtype=np.uint8).copy()
    h, w = m.shape
    start = int(h * (1.0 - band_rel))
    rows_on = (m > 0).astype(np.uint8)

    run = 0
    to_zero: List[int] = []
    for y in range(start, h):
        fill = rows_on[y].sum() / float(w)
        if fill >= fill_ratio:
            run += 1
            to_zero.append(y)
        else:
            if not (0 < run <= row_thick_max):
                to_zero = []
            run = 0

    if not (0 < run <= row_thick_max):
        to_zero = []

    if to_zero:
        m[to_zero, :] = 0

    return Image.fromarray(m, mode="L")


def _ccs(mask_bin: np.ndarray) -> Iterable[Tuple[List[Tuple[int,int]], Tuple[int,int,int,int]]]:
    """
    8-связные компоненты для бинарной маски (0/255 или 0/1).
    Возвращает (список пикселей, bbox=(y0,x0,y1,x1)) для каждой компоненты.
    """
    m = (mask_bin > 0).astype(np.uint8)
    H, W = m.shape
    visited = np.zeros_like(m, dtype=bool)

    def neigh(y, x):
        for ny in (y-1, y, y+1):
            for nx in (x-1, x, x+1):
                if ny == y and nx == x:
                    continue
                if 0 <= ny < H and 0 <= nx < W:
                    yield ny, nx

    for y in range(H):
        for x in range(W):
            if m[y, x] == 1 and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                coords = []
                y0 = y1 = y
                x0 = x1 = x
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    if cy < y0: y0 = cy
                    if cy > y1: y1 = cy
                    if cx < x0: x0 = cx
                    if cx > x1: x1 = cx
                    for ny, nx in neigh(cy, cx):
                        if m[ny, nx] == 1 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                yield coords, (y0, x0, y1+1, x1+1)  # y1/x1 exclusive


def _remove_bottom_hairline_cc(mask_img: Image.Image,
                               band_rel: float = 0.35,
                               max_h_px: int = 2,
                               min_w_rel: float = 0.50) -> Image.Image:
    """
    Удаляем горизонтальные низкие компоненты в нижней зоне:
    - высота <= max_h_px,
    - ширина >= min_w_rel * ширины маски,
    - bbox целиком в нижних band_rel маски.
    Это не тронет точки у '÷' и '·' (узкие), но снимет «подложку».
    """
    m = np.asarray(mask_img, dtype=np.uint8).copy()
    H, W = m.shape
    band_y0 = int(H * (1.0 - band_rel))

    for coords, (y0, x0, y1, x1) in _ccs(m):
        h = y1 - y0
        w = x1 - x0
        if h <= max_h_px and w >= int(min_w_rel * W) and y0 >= band_y0:
            # зануляем всю компоненту
            for (yy, xx) in coords:
                m[yy, xx] = 0

    return Image.fromarray(m, mode="L")


def _remove_small_components(mask_img: Image.Image,
                             min_area_px: int = 3,
                             min_area_rel: float = 0.0008) -> Image.Image:
    """
    Удаляем мелкие связные компоненты («пылинки»).
    Порог = max(min_area_px, min_area_rel * H * W).
    """
    m = (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.uint8)
    H, W = m.shape
    thresh = max(min_area_px, int(round(min_area_rel * H * W)))
    visited = np.zeros_like(m, dtype=bool)

    def neigh(y, x):
        for ny in (y-1, y, y+1):
            for nx in (x-1, x, x+1):
                if ny == y and nx == x:
                    continue
                if 0 <= ny < H and 0 <= nx < W:
                    yield ny, nx

    for y in range(H):
        for x in range(W):
            if m[y, x] == 1 and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                coords = []
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for ny, nx in neigh(cy, cx):
                        if m[ny, nx] == 1 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(coords) < thresh:
                    for (yy, xx) in coords:
                        m[yy, xx] = 0

    return Image.fromarray((m * 255).astype(np.uint8), mode="L")


def _prep_mask(glyph_img: Image.Image, thresh: int) -> Image.Image:
    """
    Полный конвейер подготовки маски: бинаризация -> срез нижних рядов ->
    удаление нижних горизонтальных компонент -> удаление пыли.
    """
    m = _binary_mask(glyph_img, thresh)
    m = _remove_bottom_hairline_rows(m)          # быстрый случай (1–2 ряда)
    m = _remove_bottom_hairline_cc(m)            # CC-фильтр для «хитрых» полос
    m = _remove_small_components(m)              # пылинки
    return m


def _paste_by_mask(canvas: Image.Image, glyph_img: Image.Image, x: int, y: int, thresh: int) -> None:
    """
    Composite glyph onto canvas using a crisp binary mask with
    hairline cleanup and speckle removal.
    """
    mask = _prep_mask(glyph_img, thresh)
    patch = Image.new("L", (mask.width, mask.height), 0)  # solid black
    canvas.paste(patch, (x, y), mask)


def _ink_ratio(img: Image.Image, thresh: int) -> float:
    """
    Ratio of ink pixels in a glyph image (after thresholding).
    """
    m = _binary_mask(img, thresh)
    ink = np.count_nonzero(np.asarray(m, dtype=np.uint8))
    return ink / float(m.width * m.height)


def _cap_operator_width(w: int, base_h: int, Lw: int, Rw: int, ratio_h: float = 1.2) -> int:
    """
    Cap operator width relative to line height and neighbour widths
    to avoid super-long bars from 'equals'/'minus' etc.
    """
    cap1 = int(base_h * ratio_h)
    cap2 = int(0.28 * (Lw + Rw))
    return max(1, min(w, cap1, cap2))


# =========================
# Main render
# =========================

def render_expr(
    expr: Expr,
    bank: SymbolBank,
    img_h: int = 64,
    max_w: int = 512,
    *,
    ink_rel_pad: float = 0.08,
    mask_thresh: int = 14,
    # operator height (relative to base line height)
    op_dot_h_ratio: float = 0.16,
    op_star_h_ratio: float = 0.48,
    # operator ink-coverage clamps (max allowed ink area share)
    op_dot_max_ink: float = 0.10,
    op_star_max_ink: float = 0.18,
    op_generic_max_ink: float = 0.26,
    # layout
    center_tokens: bool = True,
) -> Image.Image:
    """
    Render expression to a grayscale image (white background, black ink).
    Uses only dataset glyphs; no procedural drawing of symbols.

    - Tokens (digits/vars/operators) are vertically centered by default.
    - Exponent and fraction have special layout rules.
    - All composition is via binary masks; hairlines and speckles are removed.
    """
    base_h = max(12, int(img_h * 0.6))
    margin = max(2, img_h // 16)

    def render_node(node: Expr) -> Tuple[Image.Image, int]:
        # ---------- number ----------
        if isinstance(node, Num):
            pieces = []
            total_w = 0
            for ch in node.s:
                g = bank.sample(ch)
                if g is None:
                    raise RuntimeError(f"No glyph for token '{ch}'")
                g = _trim_glyph(g, pad_px=1, rel_pad=ink_rel_pad, thresh=mask_thresh)
                g = _resize_ink_height(g, base_h)
                pieces.append(g)
                total_w += g.width + 2

            total_w = max(1, total_w + 2)
            img = Image.new("L", (total_w, img_h), 255)
            x = 0
            for g in pieces:
                y = (img_h - g.height) // 2 if center_tokens else img_h // 2 + base_h // 2 - g.height - (img_h // 16)
                _paste_by_mask(img, g, x, y, mask_thresh)
                x += g.width + 2

            baseline = img_h // 2 + base_h // 2
            return img, baseline

        # ---------- variable ----------
        if isinstance(node, Var):
            g = bank.sample(node.s)
            if g is None:
                raise RuntimeError(f"No glyph for token '{node.s}'")
            g = _trim_glyph(g, pad_px=1, rel_pad=ink_rel_pad, thresh=mask_thresh)
            g = _resize_ink_height(g, base_h)

            img = Image.new("L", (g.width + 2, img_h), 255)
            y = (img_h - g.height) // 2 if center_tokens else img_h // 2 + base_h // 2 - g.height - (img_h // 16)
            _paste_by_mask(img, g, 0, y, mask_thresh)

            baseline = img_h // 2 + base_h // 2
            return img, baseline

        # ---------- binary operator ----------
        if isinstance(node, BinOp):
            L, bL = render_node(node.left)
            R, bR = render_node(node.right)
            gap = 4

            op_g = bank.sample(node.op)
            if op_g is None:
                raise RuntimeError(f"No glyph for operator '{node.op}'")
            op_g = _trim_glyph(op_g, pad_px=1, rel_pad=ink_rel_pad, thresh=mask_thresh)

            # target height by operator type
            if node.op == "·":
                target_h = max(2, int(base_h * op_dot_h_ratio))
                max_ink = op_dot_max_ink
                align_center = True
            elif node.op == "*":
                target_h = max(2, int(base_h * op_star_h_ratio))
                max_ink = op_star_max_ink
                align_center = True
            else:
                target_h = base_h
                max_ink = op_generic_max_ink
                align_center = center_tokens

            op_g = _resize_ink_height(op_g, target_h)

            # if operator still looks 'fat' (high ink coverage), shrink a bit
            for _ in range(3):
                ink_ratio = _ink_ratio(op_g, mask_thresh)
                if ink_ratio <= max_ink:
                    break
                scale = math.sqrt(max_ink / (ink_ratio + 1e-6))
                new_h = max(2, int(round(op_g.height * scale)))
                new_w = max(1, int(round(op_g.width * scale)))
                op_g = op_g.resize((new_w, new_h), Image.BILINEAR)

            # cap operator width w.r.t. neighbours
            cap_w = _cap_operator_width(op_g.width, base_h, L.width, R.width, ratio_h=1.1)
            if cap_w != op_g.width:
                new_h = max(1, int(round(op_g.height * (cap_w / op_g.width))))
                op_g = op_g.resize((cap_w, new_h), Image.BILINEAR)

            # compose
            w = L.width + gap + op_g.width + gap + R.width
            h = max(L.height, R.height, img_h)
            img = Image.new("L", (w, h), 255)

            # left
            _paste_by_mask(img, L, 0, (h - L.height) // 2 if center_tokens else h - L.height, mask_thresh)

            # operator
            yOp = (h - op_g.height) // 2 if align_center else h // 2 + base_h // 2 - op_g.height - (h // 16)
            _paste_by_mask(img, op_g, L.width + gap, yOp, mask_thresh)

            # right
            _paste_by_mask(img, R, L.width + gap + op_g.width + gap,
                           (h - R.height) // 2 if center_tokens else h - R.height, mask_thresh)

            baseline = h // 2 + base_h // 2
            return img, baseline

        # ---------- power ----------
        if isinstance(node, Pow):
            base_img, base_b = render_node(node.base)
            exp_img, exp_b = render_node(node.exp)

            scale = 0.65
            exp_h_px = max(8, int(exp_img.height * scale))
            exp_w_px = max(1, int(exp_img.width * scale))
            exp_img = exp_img.resize((exp_w_px, exp_h_px), Image.BILINEAR)

            pad = 2
            w = base_img.width + pad + exp_img.width
            h = max(base_img.height, base_img.height // 2 + exp_img.height)
            img = Image.new("L", (w, h), 255)

            _paste_by_mask(img, base_img, 0, (h - base_img.height) // 2, mask_thresh)

            x_exp = base_img.width - exp_img.width // 3
            y_exp = max(0, (h - base_img.height) // 2 - exp_img.height // 2 + base_img.height // 3)
            _paste_by_mask(img, exp_img, x_exp, y_exp, mask_thresh)

            baseline = h // 2 + base_img.height // 2
            return img, baseline

        # ---------- fraction ----------
        if isinstance(node, Frac):
            num_img, nb = render_node(node.num)
            den_img, db = render_node(node.den)
            pad = 4

            w = max(num_img.width, den_img.width) + 2 * pad
            line_h = max(1, int(base_h * 0.06))
            h = num_img.height // 2 + den_img.height // 2 + line_h + 2 * pad

            img = Image.new("L", (w, h), 255)

            nx = (w - num_img.width) // 2
            ny = pad
            _paste_by_mask(img, num_img, nx, ny, mask_thresh)

            line = Image.new("L", (w - 2 * pad, line_h), 0)
            img.paste(line, (pad, ny + num_img.height + 2))

            dx = (w - den_img.width) // 2
            dy = ny + num_img.height + 2 + line_h + 2
            _paste_by_mask(img, den_img, dx, dy, mask_thresh)

            baseline = dy + den_img.height // 2 + 4
            return img, baseline

        raise TypeError("unknown node type")

    # Render full expression
    img, _ = render_node(expr)

    # Fit to max width (keep height, then center on final canvas)
    w = img.width + 2 * margin
    if w > max_w:
        scale = (max_w - 2 * margin) / max(1, img.width)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.BILINEAR)
        w = img.width + 2 * margin

    canvas = Image.new("L", (w, img_h), 255)
    y = (img_h - img.height) // 2
    _paste_by_mask(canvas, img, margin, y, mask_thresh)
    return canvas
