"""Option-permutation marginalization for M1 (MC inference hygiene, no retrain).

M1 has a measured option-POSITION bias (predicts A/C/D > B/E; accuracy when the
answer sits at B=55.6 vs D=68.3). This is orthogonal to gaze: marginalize it out
by re-asking each item under all n_opt CYCLIC option orderings, mapping each
letter's probability back to the ORIGINAL option, averaging the softmax probs,
then argmax. ONE model, no training; costs n_opt forward passes/item.

Reuses select_complementary with the exact M1 config so the s=0 (identity) pass
reproduces m1.jsonl (validity), and the marginalized pred is the debiased answer.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import build_merged_inputs, forward_logits, get_option_ids
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item)
from TrajGazeMerge.training.train_visionzip_complement_lora import select_complementary
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"
_LET = re.compile(r"^\s*[A-Ea-e]\s*[.):]\s*")


def item_key(item) -> str:
    s = "|".join([str(item.get("task", "")), str(item.get("question", "")),
                  "||".join(item.get("options", [])), str(item.get("answer", ""))])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def reletter(options, shift):
    """Cyclic-shift options by `shift`, re-prefix with fresh A.. letters.
    Returns (new_options, orig_index_at_position): position j holds original
    option (j+shift) % n."""
    n = len(options)
    stripped = [_LET.sub("", o).strip() for o in options]
    new, orig = [], []
    for j in range(n):
        oi = (j + shift) % n
        new.append(f"{chr(65 + j)}. {stripped[oi]}")
        orig.append(oi)
    return new, orig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio", type=float, default=0.03)
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--include-hdepic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0, alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)

    print(f"[optmarg] loading {args.ckpt}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    qwen.load_state_dict(ckpt["lora_state"], strict=False)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)
    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for prm in encoder.parameters():
        prm.requires_grad_(False)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=args.n_frames,
                              n_traj_frames=args.n_frames, include_hdepic=args.include_hdepic)
    print(f"[optmarg] n_items={len(ds)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    c_marg = c_base = total = 0
    by_task = {}
    with open(args.dump, "w") as fout, torch.no_grad():
        for idx in range(len(ds)):
            try:
                item = ds[idx]
                if item is None:
                    continue
                opts = item["options"]; n_opt = len(opts)
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                gt = letters.index(item["answer"])
                probs = torch.zeros(n_opt)
                base_pred = None
                for s in range(n_opt):                       # cyclic permutations
                    perm_opts, orig = reletter(opts, s)
                    cached = preprocess_visionzip_item(
                        processor, base_qwen, item["vlm_frame_paths"],
                        item["question"], perm_opts, device)
                    if cached is None:
                        continue
                    sel, recv = select_complementary(
                        cached, item, device, "learned", encoder, hp,
                        args.content_ratio, args.traj_ratio, complement_mode="topk")
                    logits = forward_logits(qwen, build_merged_inputs(base_qwen, cached, sel, recv))
                    p = torch.softmax(logits[option_ids[:n_opt]].float(), dim=0).cpu()
                    for j in range(n_opt):                   # letter j → original option orig[j]
                        probs[orig[j]] += p[j]
                    if s == 0:
                        base_pred = int(p.argmax())          # identity == standard eval
                if base_pred is None:
                    continue
                marg_pred = int(probs.argmax())
                ok_m = int(marg_pred == gt); ok_b = int(base_pred == gt)
                c_marg += ok_m; c_base += ok_b; total += 1
                by_task.setdefault(item["task"], []).append((ok_b, ok_m))
                fout.write(json.dumps({
                    "key": item_key(item), "task": item["task"],
                    "pred": marg_pred, "pred_base": base_pred, "gt": gt,
                    "ok": ok_m, "ok_base": ok_b, "n_opt": n_opt,
                }) + "\n")
                fout.flush()
                if total % 20 == 0:
                    print(f"  [{total}] base={100*c_base/total:.2f} marg={100*c_marg/total:.2f}", flush=True)
            except Exception as e:
                print(f"[optmarg] idx={idx} ERR {e!r}", flush=True)
    print(f"\n[optmarg] base={100*c_base/max(1,total):.2f}  marg={100*c_marg/max(1,total):.2f}  "
          f"(n={total}, Δ={100*(c_marg-c_base)/max(1,total):+.2f})  → {args.dump}", flush=True)
    for t, v in sorted(by_task.items()):
        b = 100*sum(x[0] for x in v)/len(v); m = 100*sum(x[1] for x in v)/len(v)
        print(f"    {t:42s} n={len(v):4d}  base={b:6.2f} marg={m:6.2f} Δ={m-b:+.2f}", flush=True)


if __name__ == "__main__":
    main()
