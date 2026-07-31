"""Redraw a qualitative figure as a PDF whose text and shapes are vector, not pixels.

`viz_qual_pretty.py --dump-layout DIR` writes the geometry of each figure it renders; this
script turns one of those JSON files into a PDF. Only the video frames stay raster (they are
photographs); every box, ring, card, chip and glyph is a vector object, so the figure stays
sharp at any zoom and the type does not soften when the publisher rescales it.

The layout is recomputed with the same formulas and the same PIL font metrics as the PNG
path, so the PDF is the same figure, not an approximation.

  python scripts/viz_qual/render_qual_vector.py layout/sg_O1V0P1F0_idx83_*.json \
      --font scripts/viz_qual/Inter.ttf --out fig_qual.pdf [--type-scale 1.6]

--type-scale multiplies every font size (and only that). At \\linewidth the default design
puts the question at roughly 6 pt on the page, which is small for a paper; 1.5-1.8 brings it
to 9-11 pt without touching the frames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # embed TrueType; keep text selectable
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.font_manager import FontProperties                         # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle           # noqa: E402
from PIL import Image, ImageDraw, ImageFont                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_common import (C_GAZE, eyebrow, footer,       # noqa: E402
                           legend_entries, resolve_strip)

# design tokens, shared with the PNG renderer
INK, INK_80, INK_48 = (29, 29, 31), (51, 51, 51), (122, 122, 122)
HAIRLINE, WHITE = (224, 224, 224), (255, 255, 255)
BLUE, OK_FG, BAD_FG = (0, 102, 204), (26, 127, 55), (193, 58, 52)


def hexc(rgb, a=1.0):
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a)


class Type:
    """PIL metrics for layout, matplotlib FontProperties for drawing, one size table."""

    def __init__(self, path, scale=1.0):
        self.path, self.scale = path, scale
        self._pil, self._mpl = {}, {}

    def pil(self, size, weight="Regular"):
        key = (size, weight)
        if key not in self._pil:
            f = ImageFont.truetype(self.path, int(round(size * self.scale)))
            try:
                f.set_variation_by_name(weight)
            except Exception:
                pass
            self._pil[key] = f
        return self._pil[key]

    def mpl(self, size, weight="Regular"):
        key = (size, weight)
        if key not in self._mpl:
            fp = FontProperties(fname=self.path, size=size * self.scale)
            # Inter.ttf is a variable font and matplotlib draws its default instance, so a
            # heavier weight is approximated by the same face; the PDF stays vector either way
            self._mpl[key] = fp
        return self._mpl[key]


def wrap(d, text, font, maxw):
    """Word wrap that also breaks a single over-wide word, like the PNG composer's _wrap.

    StreamGaze's options are set-notation strings with almost no spaces
    (`{lettuce,bag,hands}>{cucumber,knife,plate,lettuce}>{onion,mesh bag}`), so a space-only
    wrap leaves one unbreakable token that runs past the column — visibly so at
    --type-scale 1.6, which the bundle README recommends for papers.
    """
    out = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            while d.textlength(w, font=font) > maxw and len(w) > 1:
                lo, hi = 1, len(w)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if d.textlength(w[:mid], font=font) <= maxw:
                        lo = mid
                    else:
                        hi = mid - 1
                if cur:
                    out.append(cur); cur = ""
                out.append(w[:lo]); w = w[lo:]
            t = (cur + " " + w).strip()
            if d.textlength(t, font=font) <= maxw or not cur:
                cur = t
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out or [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--font", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type-scale", type=float, default=1.0)
    ap.add_argument("--frames-root", default=None,
                    help="where the strip's frames live on this machine (see layout_common.py)")
    ap.add_argument("--png-dpi", type=int, default=0,
                    help="also write a PNG next to the PDF at this dpi (0 = skip)")
    args = ap.parse_args()

    L = resolve_strip(json.load(open(args.layout)), args.layout, args.frames_root)
    T = Type(args.font, args.type_scale)
    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    opts, gt = L["options"], L["answer"]
    letters = [chr(65 + i) for i in range(len(opts))]
    strip, rows = L["strip"], L["rows"]
    n = len(strip)

    # ── geometry, in design units, mirroring compose() ──
    dw = L["disp_w"]
    im0 = Image.open(strip[0]["path"])
    fw, fh = dw, int(dw * im0.height / im0.width)
    M, Lw, Rw, fgap, colgap = 46, 252, 340, 10, 26
    strip_w = n * fw + (n - 1) * fgap
    Wc = M + Lw + strip_w + colgap + Rw + M
    inner = Wc - 2 * M

    fq, fol, folb = T.pil(23, "SemiBold"), T.pil(16), T.pil(16, "SemiBold")
    fsub, fpred = T.pil(12), T.pil(15, "SemiBold")
    q_lh = int(fq.getmetrics()[0] * 1.34)
    opt_lh = int(fol.getmetrics()[0] * 1.42)
    sub_lh = int(fsub.getmetrics()[0] * 1.32)
    pred_lh = int(fpred.getmetrics()[0] * 1.42)

    qlines = wrap(d0, L["question"], fq, inner)
    blocks = [(wrap(d0, o.strip(), folb if letters[i] == gt else fol, inner - 16),
               letters[i] == gt) for i, o in enumerate(opts)]
    opts_h = sum(len(b) * opt_lh for b, _ in blocks) + (len(blocks) - 1) * 9
    y_q = M
    y_opts = y_q + len(qlines) * q_lh + 16
    y_legend = y_opts + opts_h + 18
    head = y_legend + 22 + 20

    sub_lines = [wrap(d0, r["sub"], fsub, Lw - 48) for r in rows]
    label_h = 28 + 6 + max(len(s) for s in sub_lines) * sub_lh + 10 + 18
    pred_texts = [(opts[letters.index(r["pred_letter"])].strip()
                   if r["pred_letter"] in letters else r["pred_letter"]) for r in rows]
    pred_lines = [wrap(d0, p, fpred, Rw - 8) for p in pred_texts]
    pred_h = 20 + max(len(p) for p in pred_lines) * pred_lh
    band_h = max(fh, pred_h, label_h + 20) + 32
    bands_end = head + len(rows) * band_h + (len(rows) - 1) * 12
    # the footer grew a clause per row here and runs off the page at --type-scale 1.6, so it
    # wraps to the content width and the canvas grows to fit however many lines that takes
    ffoot = T.pil(12)
    foot_lh = int(ffoot.getmetrics()[0] * 1.36)
    foot_lines = wrap(d0, footer(L), ffoot, inner)
    Hc = bands_end + 14 + len(foot_lines) * foot_lh + 10 + M

    # 1 design unit = 1 pt, so a font size in design units is that many points on the page
    fig = plt.figure(figsize=(Wc / 72, Hc / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, Wc)
    ax.set_ylim(Hc, 0)
    fig.patch.set_facecolor("white")

    def text(x, y, s, size, weight="Regular", color=INK, va="top", ha="left"):
        ax.text(x, y, s, fontproperties=T.mpl(size, weight), color=hexc(color),
                va=va, ha=ha)

    def rrect(x0, y0, x1, y1, r, fc=None, ec=None, lw=1.0, z=1):
        # matplotlib draws images at the zorder given to imshow (2 here), so anything meant
        # to sit ON a frame needs a higher zorder or it is painted over
        ax.add_patch(FancyBboxPatch(
            (x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r,
            boxstyle=f"round,pad={r}", linewidth=lw, zorder=z,
            facecolor=hexc(fc) if fc else "none",
            edgecolor=hexc(ec) if ec else "none"))

    # ── header ──
    text(M, M - 26, eyebrow(L), 12, "SemiBold", BLUE)
    y = y_q
    for ln in qlines:
        text(M, y, ln, 23, "SemiBold", INK)
        y += q_lh
    y = y_opts
    for lines, is_gt in blocks:
        blk = len(lines) * opt_lh
        if is_gt:
            ax.add_patch(Rectangle((M, y + 2), 4, blk - 6, facecolor=hexc(BLUE), lw=0))
        yy = y
        for ln in lines:
            text(M + 16, yy, ln, 16, "SemiBold" if is_gt else "Regular",
                 BLUE if is_gt else INK_80)
            yy += opt_lh
        y += blk + 9
    lx, cy = M, y_legend + 10.5
    for rgb, lab, is_ring in legend_entries(L):
        if is_ring:
            ax.add_patch(Circle((lx + 7.5, cy), 7.5, fill=False, ec=hexc(rgb), lw=2.5))
        else:
            rrect(lx, y_legend + 3, lx + 15, y_legend + 18, 4, fc=rgb)
        text(lx + 22, cy, lab, 17, "Medium", INK, va="center")
        lx += 22 + d0.textlength(lab, font=T.pil(17, "Medium")) + 26

    # ── rows ──
    s_h, s_w = L["grid"]
    by = head
    for ri, row in enumerate(rows):
        rrect(M - 10, by, Wc - M + 10, by + band_h, 16, fc=WHITE, ec=HAIRLINE, lw=0.8)
        vc = OK_FG if row["correct"] else BAD_FG
        cx0, cy0, cx1, cy1 = M, by + 12, M + Lw - 18, by + band_h - 12
        rrect(cx0, cy0, cx1, cy1, 12,
              fc=tuple(int(c * 0.16 + 255 * 0.84) for c in vc),
              ec=tuple(int(c * 0.55 + 255 * 0.45) for c in vc), lw=0.8)
        blk = 28 + 6 + len(sub_lines[ri]) * sub_lh + 10 + 18
        ty = cy0 + ((cy1 - cy0) - blk) / 2
        # step the name down a size until it clears the card — the bundle's own names were
        # short enough ("Ours", "VisionZip") that this never came up
        nsz = 20
        for cand in (20, 18, 16, 14):
            nsz = cand
            if d0.textlength(row["name"], font=T.pil(cand, "SemiBold")) <= (cx1 - cx0) - 30:
                break
        text(cx0 + 15, ty, row["name"], nsz, "SemiBold", BLUE if row["is_ours"] else INK)
        yy = ty + 34
        for ln in sub_lines[ri]:
            text(cx0 + 15, yy, ln, 12, "Regular", INK_80)
            yy += sub_lh
        text(cx0 + 15, yy + 6, "Correct" if row["correct"] else "Wrong", 12, "SemiBold", vc)

        fy = by + (band_h - fh) / 2
        fx = M + Lw
        # a row may override the shared strip with its own frame files (see
        # layout_common.resolve_strip); geometry is identical, only the pixels differ
        for si, cell in enumerate(row.get("strip") or strip):
            img = Image.open(cell["path"]).convert("RGB")
            ax.imshow(img, extent=(fx, fx + fw, fy + fh, fy), zorder=2,
                      interpolation="antialiased")
            pw, ph = fw / s_w, fh / s_h
            for grp in row["groups"][si]:
                rgb, lw = tuple(grp["rgb"]), max(0.6, grp["width_u"])
                for r, c in grp["cells"]:
                    rrect(fx + c * pw + 1, fy + r * ph + 1,
                          fx + (c + 1) * pw - 1, fy + (r + 1) * ph - 1,
                          max(1.0, pw * 0.18),
                          fc=(rgb if grp["fill_alpha"] else None), ec=rgb, lw=lw, z=3)
                    if grp["fill_alpha"]:                      # translucent fill like the PNG
                        ax.patches[-1].set_facecolor(hexc(rgb, grp["fill_alpha"] / 255))
            if cell["gaze"]:
                gx, gy = fx + cell["gaze"][0] * fw, fy + cell["gaze"][1] * fh
                ax.add_patch(Circle((gx, gy), 11, fill=False, ec=(1, 1, 1, 0.82), lw=3,
                                    zorder=3))
                ax.add_patch(Circle((gx, gy), 9, fill=False, ec=hexc(C_GAZE), lw=3, zorder=4))
            lbl = f"t{cell['t']}" + (" · final fixation" if cell["query_moment"] else "")
            tw = d0.textlength(lbl, font=T.pil(11, "SemiBold"))
            rrect(fx + 7, fy + fh - 29, fx + 7 + tw + 14, fy + fh - 7, 9,
                  fc=(BLUE if cell["query_moment"] else (0, 0, 0)), z=5)
            ax.patches[-1].set_facecolor(
                hexc(BLUE, 0.92) if cell["query_moment"] else hexc((0, 0, 0), 0.55))
            ax.patches[-1].set_zorder(5)
            text(fx + 14, fy + fh - 26, lbl, 11, "SemiBold", (255, 255, 255))
            ax.texts[-1].set_zorder(6)
            fx += fw + fgap

        px = fx + colgap
        pblk = 20 + len(pred_lines[ri]) * pred_lh
        py = by + (band_h - pblk) / 2
        text(px, py, "PREDICTION", 12, "SemiBold", INK_48)
        yy = py + 20
        for ln in pred_lines[ri]:
            text(px, yy, ln, 15, "SemiBold", OK_FG if row["correct"] else BAD_FG)
            yy += pred_lh
        by += band_h + 12

    fyy = bands_end + 14
    for ln in foot_lines:
        text(M, fyy, ln, 12, "Regular", INK_48)
        fyy += foot_lh

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, format="pdf", facecolor="white")
    print(f"-> {args.out}  ({Wc / 72:.1f} x {Hc / 72:.1f} in, text is vector)")
    if args.png_dpi:
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=args.png_dpi, facecolor="white")
        print(f"-> {png}  ({int(Wc / 72 * args.png_dpi)}px wide)")


if __name__ == "__main__":
    main()
