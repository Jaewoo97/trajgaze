"""Build example overlay montages + GIFs for EgoGazeVQA and HD-EPIC,
exactly as the VLM now sees them (gaze marker drawn in pixels)."""
import os, glob, collections
from PIL import Image, ImageDraw

OUT = "/tmp"


def montage(frames, path, tile_w, title, cols=8):
    imgs = []
    for fp, lab in frames:
        im = Image.open(fp).convert("RGB")
        w, h = im.size
        tile_h = int(tile_w * h / w)
        im = im.resize((tile_w, tile_h))
        d = ImageDraw.Draw(im)
        d.text((4, 4), lab, fill=(255, 255, 0))
        imgs.append(im)
    rows = (len(imgs) + cols - 1) // cols
    tw, th = imgs[0].size
    M = Image.new("RGB", (tw * cols, th * rows + 20), (24, 24, 24))
    for i, im in enumerate(imgs):
        M.paste(im, ((i % cols) * tw, (i // cols) * th))
    ImageDraw.Draw(M).text((4, th * rows + 4), title, fill=(255, 255, 255))
    M.save(path)
    print("saved", path, M.size)


def gif(frames, path, w):
    imgs = []
    for fp, _ in frames:
        im = Image.open(fp).convert("RGB")
        ww, hh = im.size
        imgs.append(im.resize((w, int(w * hh / ww))))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=160, loop=0)
    print("saved", path, len(imgs), "frames")


# ---------------- EgoGazeVQA (egtea/gaze) ----------------
eg_dir = "/workspace/EgoGazeVQA/egtea/gaze/OP01-R01-PastaSalad"
files = [os.path.basename(f) for f in glob.glob(f"{eg_dir}/*.jpg")]
# group by subclip prefix (everything before the last underscore), pick biggest
groups = collections.defaultdict(list)
for f in files:
    groups[f.rsplit("_", 1)[0]].append(f)
sub = max(groups, key=lambda k: len(groups[k]))
seq = sorted(groups[sub], key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
print(f"EG subclip {sub}: {len(seq)} frames")
step = max(1, len(seq) // 24)
pick = seq[::step][:24]
eg_frames = [(os.path.join(eg_dir, f), f.rsplit("_", 1)[1].split(".")[0]) for f in pick]
montage(eg_frames[:8], f"{OUT}/eg_overlay_montage.png", 240,
        f"EgoGazeVQA egtea {sub} — red ring = gaze (upstream 'gaze' set)")
gif(eg_frames, f"{OUT}/eg_overlay.gif", 360)

# ---------------- HD-EPIC (frames_gaze) ----------------
hd_stem = "P09-20240621-093545"
hd_dir = f"/workspace/HD-EPIC/frames_gaze/{hd_stem.split('-')[0]}/{hd_stem}"
hd_all = sorted(glob.glob(f"{hd_dir}/frame_*.jpg"))
# take a 24-frame window from the middle where there's motion
mid = len(hd_all) // 2
win = hd_all[mid: mid + 24 * 30: 30][:24]   # every 30th frame for visible motion
hd_frames = [(f, os.path.basename(f).replace("frame_", "").replace(".jpg", ""))
             for f in win]
print(f"HD {hd_stem}: {len(hd_all)} frames, window {len(win)}")
montage(hd_frames[:8], f"{OUT}/hd_overlay_montage.png", 224,
        f"HD-EPIC {hd_stem} — red ring + green dot = projected gaze")
gif(hd_frames, f"{OUT}/hd_overlay.gif", 320)
print("DONE")
