"""What the two offline renderers share: where the frames are, and the figure's chrome.

Part 1, frame paths: a dump records absolute paths from the machine that made it, so they
have to be re-found here (`resolve_strip`).

Part 2, chrome: the eyebrow, the legend and the footer describe *our* four methods and *our*
benchmarks. A layout from another project overrides them with the optional `title`, `legend`
and `footer` keys rather than editing the renderers:

    "title":  "QUALITATIVE  ·  MyBench",
    "legend": [{"rgb": [52,199,89], "label": "kept 10%"},
               {"rgb": [255,209,26], "label": "gaze", "shape": "ring"}],
    "footer": "same budget, same frozen backbone"

`note` is still appended to the footer, so the frame-selection caption stays separate from
the claim about the comparison.

## Finding the video frames of a layout dump on another machine

`viz_qual_pretty.py --dump-layout` records each strip frame as the absolute path it had on
the rendering machine (`/workspace/datasets/StreamGaze_v2/frames/...`). That path is right
on that machine and meaningless anywhere else, so the two offline renderers resolve it in
this order, first hit wins:

  1. `--frames-root DIR`, joined with the recorded path both as-is and by basename, so a
     directory that mirrors the dataset tree and a flat directory of JPEGs both work;
  2. the recorded path itself;
  3. the same path taken as relative to the layout JSON's own directory, which is what a
     shared bundle uses (`frames/<clip>/frame_000556.jpg` next to the .json);
  4. `frames/<basename>` beside the layout, or one level up.

Rewriting the paths inside the JSON works too and needs no flag; this exists so a dump can be
copied around untouched.
"""
from __future__ import annotations

import os

# the four legend colours, shared with both renderers' drawing code
C_CONTENT, C_COMP = (52, 199, 89), (255, 45, 190)
C_BASE, C_GAZE = (0, 178, 224), (255, 209, 26)

DEFAULT_LEGEND = [(C_CONTENT, "content-based selection 7%", False),
                  (C_COMP, "gaze/hand complement 3%", False),
                  (C_BASE, "baseline kept 10%", False),
                  (C_GAZE, "gaze", True)]
DEFAULT_FOOTER = ("Same 10% visual-token budget · frozen Qwen2.5-VL-7B backbone · "
                  "each method uses its own selector and its own LoRA adapter")


def eyebrow(layout: dict) -> str:
    if layout.get("title"):
        return layout["title"]
    src = "EgoGazeVQA" if layout.get("source") == "eg" else "StreamGaze"
    return f"QUALITATIVE  ·  {src}"


def legend_entries(layout: dict) -> list[tuple[tuple[int, int, int], str, bool]]:
    """[(rgb, label, is_ring)]. A ring is drawn hollow, a swatch filled."""
    if not layout.get("legend"):
        return list(DEFAULT_LEGEND)
    return [(tuple(e["rgb"]), e["label"], e.get("shape", "swatch") == "ring")
            for e in layout["legend"]]


def footer(layout: dict) -> str:
    foot = layout.get("footer", DEFAULT_FOOTER)
    if layout.get("note"):
        foot += f" · {layout['note']}"
    return foot


def resolve_frame(path: str, layout_path: str, root: str | None = None) -> str:
    base = os.path.basename(path)
    here = os.path.dirname(os.path.abspath(layout_path))
    cands = []
    if root:
        cands += [os.path.join(root, path.lstrip("/")), os.path.join(root, base)]
    cands += [path,
              os.path.join(here, path.lstrip("/")),
              os.path.join(here, "frames", base),
              os.path.join(os.path.dirname(here), "frames", base)]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"frame not found: {path}\n  tried:\n    " + "\n    ".join(cands)
        + "\n  pass --frames-root DIR pointing at the frames, or rewrite the paths in the "
          "layout JSON")


def resolve_strip(layout: dict, layout_path: str, root: str | None = None) -> dict:
    """Replace every strip entry's `path` with one that exists here. Returns the layout.

    Local extension to the bundle schema: a row may carry its own `strip`, in which case that
    row is drawn over different image files than the shared one. We need it because the
    overlay-free student is preprocessed from the marker-free frame tree (`original`/`no_gaze`)
    while the teacher and the overlay student read `viz`/`gaze` — same moments, different
    pixels. Rows without a `strip` fall back to the shared one, so a layout written by the
    original pipeline is unaffected.
    """
    for cell in layout["strip"]:
        cell["path"] = resolve_frame(cell["path"], layout_path, root)
    for row in layout.get("rows", []):
        for cell in row.get("strip") or []:
            cell["path"] = resolve_frame(cell["path"], layout_path, root)
    return layout
