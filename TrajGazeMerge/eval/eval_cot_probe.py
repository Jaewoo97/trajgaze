"""Direction A spot-check: does step-by-step REASONING move spatial/temporal off
chance, WITHIN the exact M1 pipeline (compressed 10% tokens + LoRA, 128 frames)?

Ceiling #2 says M1's locked errors are reasoning-bound (spatial 42, temporal 43 ≈
chance); token selection can't fix them. This probes the orthogonal lever: keep
M1's tokens, but let the model REASON before answering.

Two conditions per item (same compressed visual tokens):
  direct : standard single-logit answer (reproduces M1)            [forward_logits]
  cot    : "think step by step, end with 'Answer: <letter>'" then  [manual greedy
           autoregressive generation; parse the final letter]       decode]

Decisive read: cot >> direct on spatial/temporal → reasoning is the unlock →
invest (72B-rationale SFT). cot ≈ direct → 7B capability wall, drop A.

Manual greedy decode (not HF generate) so we control mrope position continuation
explicitly over the merged inputs_embeds: text tokens after the multimodal prefix
advance all 3 mrope dims by +1 each step.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import build_merged_inputs, forward_logits, get_option_ids
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, VIDEO_KWARGS)
from TrajGazeMerge.training.train_visionzip_complement_lora import select_complementary
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"
TARGET_TASKS = {"spatial", "temporal"}
DIRECT_INSTR = "Answer with a single letter ({L})."
COT_INSTR = ("Reason in at most 3 short steps, then on a new line write exactly "
             "'Answer: X' where X is one of {L}. Keep it brief.")


def item_key(item):
    s = "|".join([str(item.get("task", "")), str(item.get("question", "")),
                  "||".join(item.get("options", [])), str(item.get("answer", ""))])
    return hashlib.md5(s.encode()).hexdigest()


def preprocess_instr(processor, base_qwen, frame_paths, question, options, device, instruction):
    """Same as preprocess_visionzip_item but with a custom answer instruction."""
    from qwen_vl_utils import process_vision_info
    options_text = "\n".join(options)
    letters = [chr(65 + i) for i in range(len(options))]
    lstr = ", ".join(letters[:-1]) + (f", or {letters[-1]}" if len(letters) > 1 else "")
    prompt = f"{question}\nOptions:\n{options_text}\n" + instruction.format(L=lstr)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frame_paths, "max_pixels": VIDEO_KWARGS["max_pixels"],
         "min_pixels": VIDEO_KWARGS["min_pixels"], "fps": VIDEO_KWARGS["fps"]},
        {"type": "text", "text": prompt}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           **video_kwargs, return_tensors="pt")
    except Exception:
        return None
    if "pixel_values_videos" not in inputs:
        return None
    emb_dev = base_qwen.get_input_embeddings().weight.device
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device
    input_ids = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw = inputs["video_grid_thw"].to(vis_dev)
    with torch.no_grad():
        video_embeds, attn_scores, attn_key = base_qwen.visual(pv_vid, grid_thw=grid_thw)
        video_embeds = video_embeds.to(emb_dev); attn_scores = attn_scores.to(emb_dev)
        attn_key = attn_key.to(emb_dev)
        position_ids, rope_deltas = base_qwen.get_rope_index(
            input_ids=input_ids, video_grid_thw=grid_thw, attention_mask=attention_mask)
    video_token_id = base_qwen.config.video_token_id
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "position_ids": position_ids, "rope_deltas": rope_deltas, "grid_thw": grid_thw,
            "video_embeds": video_embeds, "video_positions": video_positions,
            "attn_scores": attn_scores, "attn_key": attn_key, "query_emb": None, "emb_dev": emb_dev}


@torch.no_grad()
def greedy_decode(qwen, base_qwen, merged, eos_ids, tokenizer, max_new=448):
    emb = base_qwen.get_input_embeddings()
    ie = merged["inputs_embeds"]; attn = merged["attention_mask"]; pos = merged["position_ids"]
    dev = ie.device
    out = qwen(inputs_embeds=ie, attention_mask=attn, position_ids=pos,
               rope_deltas=merged["rope_deltas"], use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :]
    next_pos = int(pos[0, 0, -1].item()) + 1
    gen = []
    answered_at = None
    for _ in range(max_new):
        tok = int(logits.argmax())
        gen.append(tok)
        if tok in eos_ids:
            break
        # early stop: 'Answer: X' emitted → allow a couple tokens then break
        if answered_at is None and len(gen) >= 4:
            tail = tokenizer.decode(gen[-12:], skip_special_tokens=True)
            if _ANS.search(tail):
                answered_at = len(gen)
        if answered_at is not None and len(gen) >= answered_at + 2:
            break
        te = emb(torch.tensor([[tok]], device=dev))
        attn = torch.cat([attn, torch.ones((1, 1), device=dev, dtype=attn.dtype)], dim=1)
        p = torch.full((3, 1, 1), next_pos, device=dev, dtype=torch.long)
        out = qwen(inputs_embeds=te, attention_mask=attn, position_ids=p,
                   past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        next_pos += 1
    return gen


_ANS = re.compile(r"answer\s*[:\-]?\s*\(?\s*([A-Ea-e])", re.I)


def parse_letter(text, n_opt):
    valid = set(chr(65 + i) for i in range(n_opt))
    m = list(_ANS.finditer(text))
    if m:
        c = m[-1].group(1).upper()
        if c in valid:
            return ord(c) - 65
    for ch in reversed(text):                       # fallback: last standalone valid letter
        if ch.upper() in valid:
            return ord(ch.upper()) - 65
    return -1


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
    p.add_argument("--max-new", type=int, default=448)
    p.add_argument("--all-tasks", action="store_true", help="probe all tasks, not just spatial/temporal")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0, alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)

    print(f"[cot] loading {args.ckpt}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    qwen.load_state_dict(ckpt["lora_state"], strict=False)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)
    tok = processor.tokenizer
    eos_ids = {tok.eos_token_id}
    try:
        eos_ids.add(tok.convert_tokens_to_ids("<|im_end|>"))
    except Exception:
        pass
    eos_ids = {e for e in eos_ids if e is not None}

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for prm in encoder.parameters():
        prm.requires_grad_(False)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=args.n_frames,
                              n_traj_frames=args.n_frames, include_hdepic=False)
    print(f"[cot] n_items={len(ds)} target={'all' if args.all_tasks else sorted(TARGET_TASKS)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    cd = cc = total = 0
    by_task = {}
    with open(args.dump, "w") as fout, torch.no_grad():
        for idx in range(len(ds)):
            try:
                item = ds[idx]
                if item is None:
                    continue
                if not args.all_tasks and item["task"] not in TARGET_TASKS:
                    continue
                opts = item["options"]; n_opt = len(opts)
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                gt = letters.index(item["answer"])

                # direct (reproduce M1)
                cdir = preprocess_instr(processor, base_qwen, item["vlm_frame_paths"],
                                        item["question"], opts, device, DIRECT_INSTR)
                if cdir is None:
                    continue
                sel, recv = select_complementary(cdir, item, device, "learned", encoder, hp,
                                                 args.content_ratio, args.traj_ratio, complement_mode="topk")
                dlog = forward_logits(qwen, build_merged_inputs(base_qwen, cdir, sel, recv))
                pred_d = int(dlog[option_ids[:n_opt]].argmax())

                # cot (reason then answer)
                ccot = preprocess_instr(processor, base_qwen, item["vlm_frame_paths"],
                                        item["question"], opts, device, COT_INSTR)
                sel2, recv2 = select_complementary(ccot, item, device, "learned", encoder, hp,
                                                   args.content_ratio, args.traj_ratio, complement_mode="topk")
                merged = build_merged_inputs(base_qwen, ccot, sel2, recv2)
                gen = greedy_decode(qwen, base_qwen, merged, eos_ids, tok, max_new=args.max_new)
                gtext = tok.decode(gen, skip_special_tokens=True)
                pred_c = parse_letter(gtext, n_opt)

                ok_d = int(pred_d == gt); ok_c = int(pred_c == gt)
                cd += ok_d; cc += ok_c; total += 1
                by_task.setdefault(item["task"], []).append((ok_d, ok_c))
                fout.write(json.dumps({
                    "key": item_key(item), "task": item["task"], "gt": gt,
                    "pred_direct": pred_d, "pred_cot": pred_c,
                    "ok_direct": ok_d, "ok_cot": ok_c, "n_opt": n_opt,
                    "cot_text": gtext[:1200],
                }) + "\n")
                fout.flush()
                if total % 10 == 0:
                    print(f"  [{total}] direct={100*cd/total:.2f} cot={100*cc/total:.2f} "
                          f"(last parse {'OK' if pred_c>=0 else 'FAIL'})", flush=True)
            except Exception as e:
                print(f"[cot] idx={idx} ERR {e!r}", flush=True)
    print(f"\n[cot] direct={100*cd/max(1,total):.2f}  cot={100*cc/max(1,total):.2f}  "
          f"(n={total}, Δ={100*(cc-cd)/max(1,total):+.2f})  → {args.dump}", flush=True)
    for t, v in sorted(by_task.items()):
        d = 100*sum(x[0] for x in v)/len(v); c = 100*sum(x[1] for x in v)/len(v)
        print(f"    {t:20s} n={len(v):4d}  direct={d:6.2f} cot={c:6.2f} Δ={c-d:+.2f}", flush=True)


if __name__ == "__main__":
    main()
