"""
Visualize Talk2DINO frame-question similarity as per-QA GIFs.

Each QA pair produces ONE GIF showing:
  - Top: full question text (word-wrapped)
  - Middle: uniform (blue) and weighted (red) similarity bars
  - Bottom: frame image with gaze/hand markers

Usage:
  python visualize_similarity_gif.py \
      --results talk2dino_sim_sample.json \
      --data_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
      --output_dir viz_gifs --skip 3 --fps 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from PIL import Image, ImageDraw


# ── Colors ────────────────────────────────────────────────────────────────────

BG_DARK  = (25, 25, 30)
BG_BAR   = (35, 35, 40)
CLR_UNIF = (60, 130, 230)
CLR_WGHT = (220, 65, 65)
CLR_GAZE = (0, 230, 80)
CLR_HL   = (255, 230, 0)
CLR_HR   = (255, 170, 0)
CLR_TEXT = (210, 210, 210)
CLR_DIM  = (140, 140, 140)


# ── Helpers ───────────────────────────────────────────────────────────────────

def word_wrap(text: str, chars_per_line: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= chars_per_line:
            current = (current + " " + word) if current else word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_bar(draw: ImageDraw.Draw, x0: int, y0: int, width: int, height: int,
             value: float, vmin: float, vmax: float, color: tuple,
             label: str):
    bar_w = width - 4
    ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin + 1e-8)))
    fill_w = int(bar_w * ratio)

    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=BG_BAR,
                   outline=(60, 60, 65))
    if fill_w > 0:
        draw.rectangle([x0 + 2, y0 + 2, x0 + 2 + fill_w, y0 + height - 2],
                       fill=color)
    draw.text((x0 + 6, y0 + 2), f"{label}: {value:.3f}", fill=CLR_TEXT)


def draw_markers(draw: ImageDraw.Draw, gaze, hand_left, hand_right,
                 orig_w: int, orig_h: int, scale: float):
    if gaze is not None:
        px = int(gaze[0] * orig_w * scale)
        py = int(gaze[1] * orig_h * scale)
        r = 7
        draw.ellipse([px - r, py - r, px + r, py + r],
                     fill=CLR_GAZE, outline=(255, 255, 255), width=1)

    for hand, color in [(hand_left, CLR_HL), (hand_right, CLR_HR)]:
        if hand is not None:
            hx, hy = int(hand[0] * scale), int(hand[1] * scale)
            r = 6
            draw.ellipse([hx - r, hy - r, hx + r, hy + r],
                         fill=color, outline=(255, 255, 255), width=1)


def load_gaze_mapping(data_dir, dataset, video_id, clip_name):
    csv_path = os.path.join(data_dir, dataset, "gaze_mapping", video_id,
                            f"{clip_name}_mapping.csv")
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fnum = int(row["gaze_frame_num"])
                mapping[fnum] = (float(row["gaze_x"]), float(row["gaze_y"]))
            except (ValueError, KeyError):
                pass
    return mapping


def load_hand_data(data_dir, dataset, video_id):
    path = os.path.join(data_dir, dataset, "hand_locations",
                        f"{video_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── GIF creation ──────────────────────────────────────────────────────────────

def create_qa_gif(
    qa_result: dict,
    data_dir: str,
    output_path: str,
    target_w: int = 480,
    fps: int = 4,
    skip: int = 3,
):
    parts = qa_result["file_name"].strip().split("/")
    dataset, video_id = parts[0], parts[1]
    clip_name = parts[2].replace(".mp4", "")

    frame_dir = os.path.join(data_dir, dataset, "no_gaze", video_id)
    if not os.path.isdir(frame_dir):
        print(f"  SKIP: {frame_dir} not found")
        return

    frame_names = qa_result["frame_names"]
    n_frames = len(frame_names)
    uniform_sims = qa_result["uniform_sims"]
    weighted_sims = qa_result["weighted_sims"]

    # Load spatial data
    gaze_map = load_gaze_mapping(data_dir, dataset, video_id, clip_name)
    hand_data = load_hand_data(data_dir, dataset, video_id)

    # Get original image resolution
    first_path = os.path.join(frame_dir, frame_names[0])
    if not os.path.exists(first_path):
        print(f"  SKIP: {first_path} not found")
        return
    with Image.open(first_path) as tmp:
        orig_w, orig_h = tmp.size
    scale = target_w / orig_w
    target_h_img = int(orig_h * scale)

    # Sim range for bars
    vmin = min(min(uniform_sims), min(weighted_sims))
    vmax = max(max(uniform_sims), max(weighted_sims))

    # Word-wrap question text
    line_h = 16
    chars_per_line = int(target_w / 6.5)
    q_lines = word_wrap(f"Q{qa_result['qa_index']}: {qa_result['question']}",
                        chars_per_line)
    qa_text_h = len(q_lines) * line_h + 8

    # Layout
    bar_h = 20
    bars_h = bar_h * 2 + 4
    footer_h = 20
    total_h = qa_text_h + bars_h + target_h_img + footer_h

    # Build GIF frames
    gif_frames = []
    indices = list(range(0, n_frames, skip))

    for idx in indices:
        fname = frame_names[idx]
        fpath = os.path.join(frame_dir, fname)
        if not os.path.exists(fpath):
            continue

        img = Image.open(fpath).convert("RGB")
        img = img.resize((target_w, target_h_img), Image.LANCZOS)

        # Draw markers
        fnum = int(fname.replace(".jpg", "").rsplit("_", 1)[-1])
        gaze = gaze_map.get(fnum)
        hd = hand_data.get(fname, {})
        hl_raw = hd.get("left")
        hr_raw = hd.get("right")

        draw_img = ImageDraw.Draw(img)
        draw_markers(draw_img, gaze, hl_raw, hr_raw, orig_w, orig_h, scale)

        # Compose canvas
        canvas = Image.new("RGB", (target_w, total_h), BG_DARK)
        draw = ImageDraw.Draw(canvas)
        y = 4

        # Question text
        for line in q_lines:
            draw.text((6, y), line, fill=CLR_TEXT)
            y += line_h
        y += 4

        # Bars
        draw_bar(draw, 2, y, target_w - 4, bar_h,
                 uniform_sims[idx], vmin, vmax, CLR_UNIF, "Uniform")
        y += bar_h
        draw_bar(draw, 2, y, target_w - 4, bar_h,
                 weighted_sims[idx], vmin, vmax, CLR_WGHT, "Weighted")
        y += bar_h + 4

        # Frame image
        canvas.paste(img, (0, y))
        y += target_h_img

        # Footer
        draw.text((target_w - 120, y + 2),
                  f"frame {idx+1}/{n_frames}", fill=CLR_DIM)
        draw.ellipse([6, y + 4, 12, y + 10], fill=CLR_GAZE)
        draw.text((15, y + 2), "gaze", fill=CLR_DIM)
        draw.ellipse([56, y + 4, 62, y + 10], fill=CLR_HL)
        draw.text((65, y + 2), "handL", fill=CLR_DIM)
        draw.ellipse([110, y + 4, 116, y + 10], fill=CLR_HR)
        draw.text((119, y + 2), "handR", fill=CLR_DIM)

        gif_frames.append(canvas)

    if not gif_frames:
        print(f"  SKIP: no frames for QA{qa_result['qa_index']}")
        return

    duration = int(1000 / fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {os.path.basename(output_path)} "
          f"({len(gif_frames)} frames, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    print(f"Generating {len(data['results'])} GIFs (1 per QA)...\n")

    for r in data["results"]:
        parts = r["file_name"].strip().split("/")
        dataset = parts[0]
        clip_name = parts[2].replace(".mp4", "")

        out_name = f"qa{r['qa_index']}_{dataset}_{clip_name}.gif"
        out_path = os.path.join(args.output_dir, out_name)

        print(f"QA{r['qa_index']} ({dataset}/{clip_name}):")
        create_qa_gif(
            qa_result=r,
            data_dir=args.data_dir,
            output_path=out_path,
            target_w=args.width,
            fps=args.fps,
            skip=args.skip,
        )

    print(f"\nDone. {len(data['results'])} GIFs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
