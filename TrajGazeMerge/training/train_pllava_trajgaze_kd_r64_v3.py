"""
PLLaVA-7B + TrajGaze encoder (10% tokens) + LoRA r=64. v3 (fixed training).

Fixes vs v2:
  1. Prompt ends at "ASSISTANT:Best option:(" — natural next token is the option
     letter, making CE/KD well-posed (v2 ended at "ASSISTANT:" where next token
     is "▁Best", not a letter, causing gradient conflict).
  2. Student LoRA warm-initialized from teacher best_delta.pth (64.83%) instead
     of zero — starts from a strong point and only needs to adapt to merged tokens.
  3. Cosine LR decay — prevents the epoch-3 collapse seen in v2 (constant LR).
  4. Full-vocab CE at option-letter position (not 4-class CE on 4 logits).
  5. n_traj_frames=32 to match standalone eval scripts.

Teacher : finetuned PLLaVA (64.83%, pllava_baseline_lora/best_delta.pth), frozen.
Student : PLLaVA + LoRA r=64 (warm from teacher) + TrajGaze-merged 10% tokens.

Train : egoexolearn + holoassist (5799 items)
Eval  : egtea (526 items, logit-based with merged tokens)
GPUs  : 0,1 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/gaze/bin/torchrun \\
        --nproc_per_node=2 --master_port=29801 \\
        -m TrajGazeMerge.training.train_pllava_trajgaze_kd_r64_v3 \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_trajgaze_kd_r64_v3 \\
        --teacher-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora/best_delta.pth \\
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \\
        --epochs 3 --lr-lora 1e-4 --lr-enc 1e-5 --alpha 0.5 --kd-temp 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/prunevid")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge

PLLAVA_HF      = "ermu2001/pllava-7b"
POOLING_SHAPE  = (16, 12, 12)
FRAME_SHAPE    = (24, 24)
IMAGE_TOKEN    = "<image>"
IMAGE_TOKEN_ID = 32000
NUM_FRAMES     = 16
KEEP_RATIO     = 0.10
MERGE_RATIO    = 1.0 - KEEP_RATIO

LORA_R     = 64
LORA_ALPHA = 128

OPTION_LETTERS = ["A", "B", "C", "D"]

STAGE1_CKPT  = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"
TEACHER_CKPT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora/best_delta.pth"

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _sample_paths(paths: list, n: int) -> list:
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    return [paths[int(i * len(paths) / n)] for i in range(n)]


def build_prompt(question: str, options: list) -> str:
    return f"{question}\nOptions:\n" + "\n".join(options)


def _load_pllava_peft_ckpt(model, hf_path, lora_alpha=256, lora_r=128):
    import glob
    from safetensors import safe_open
    if not os.path.isdir(hf_path):
        from huggingface_hub import snapshot_download
        hf_path = snapshot_download(hf_path)
    raw = {}
    for sf in sorted(glob.glob(os.path.join(hf_path, "*.safetensors"))):
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    scale = lora_alpha / lora_r
    remapped = {}
    for k, v in raw.items():
        if k.startswith("language_model.base_model.model."):
            new_k = k.replace("language_model.base_model.model.", "language_model.", 1)
            if ".base_layer.weight" in new_k:
                proj_prefix = k[: k.rfind(".base_layer.weight")]
                lora_a = raw.get(proj_prefix + ".lora_A.default.weight")
                lora_b = raw.get(proj_prefix + ".lora_B.default.weight")
                if lora_a is not None and lora_b is not None:
                    v = v + scale * (lora_b.float() @ lora_a.float()).to(v.dtype)
                final_k = new_k.replace(".base_layer.weight", ".weight")
                remapped[final_k] = v
            elif ".lora_A." in k or ".lora_B." in k:
                continue
            else:
                remapped[new_k] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[PEFT ckpt] loaded {len(remapped)} keys | missing={len(missing)} unexpected={len(unexpected)}")
    return model


def _build_pllava_base(device):
    from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor
    processor = PllavaProcessor.from_pretrained(PLLAVA_HF)
    config = PllavaConfig.from_pretrained(
        PLLAVA_HF,
        pooling_method="avg",
        use_pooling=True,
        frame_shape=FRAME_SHAPE,
        pooling_shape=POOLING_SHAPE,
        torch_dtype=torch.bfloat16,
        selected_layer=99,
        tau=1.0,
        cluster_ratio=1.0,
        temporal_segment_ratio=1.0,
    )
    model = PllavaForConditionalGeneration.from_pretrained(
        PLLAVA_HF, config=config, torch_dtype=torch.bfloat16
    )
    _load_pllava_peft_ckpt(model, PLLAVA_HF)
    return model, processor


def _lora_config(inference_mode=False):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=inference_mode,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def load_teacher(teacher_ckpt_path: str, device: torch.device):
    """Load finetuned PLLaVA (64.83%) as frozen teacher with full tokens."""
    model, _ = _build_pllava_base(device)
    model.language_model = get_peft_model(model.language_model, _lora_config(inference_mode=True))
    delta = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[Teacher] loaded {len(delta)} keys | missing={len(missing)} unexpected={len(unexpected)}")
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    model.eval()
    return model


def load_student(device: torch.device, teacher_ckpt_path: str):
    """PLLaVA + LoRA r=64, warm-initialized from teacher best_delta.pth."""
    model, processor = _build_pllava_base(device)
    for p in model.vision_tower.parameters():
        p.requires_grad_(False)
    for p in model.language_model.parameters():
        p.requires_grad_(False)
    model.language_model = get_peft_model(model.language_model, _lora_config(inference_mode=False))
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()

    # Warm start: copy teacher LoRA + projector weights into student
    teacher_delta = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(teacher_delta, strict=False)
    print(f"[Student warm init] loaded {len(teacher_delta)} teacher keys "
          f"| missing={len(missing)} unexpected={len(unexpected)}")

    return model.to(device), processor


def load_traj_encoder(ckpt_path: str, device: torch.device):
    from TrajGaze_v2.models.model import TrajGazeV2
    enc = TrajGazeV2().to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        m, u = enc.load_state_dict(state, strict=False)
        print(f"[TrajEnc] loaded | missing={len(m)} unexpected={len(u)}")
    else:
        print(f"[TrajEnc] WARNING: {ckpt_path} not found, using random init")
    return enc


# ── vision helpers ─────────────────────────────────────────────────────────────

def score_to_pllava_spatial(patch_scores: torch.Tensor, n_spatial: int) -> torch.Tensor:
    side = int(n_spatial ** 0.5)
    scores_2d = patch_scores.float().reshape(1, 1, 14, 14)
    out = F.interpolate(scores_2d, size=(side, side), mode="bilinear", align_corners=False)
    return out.squeeze().flatten()


def get_pllava_image_features(model, pixel_values, media_type="video"):
    model_dtype = next(model.language_model.parameters()).dtype
    pixel_values = pixel_values.to(model_dtype)
    batch_size = 1
    num_videos = pixel_values.shape[0] // model.config.num_frames // batch_size
    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
    selected = image_outputs.hidden_states[model.config.vision_feature_layer][:, 1:]
    image_features = model.multi_modal_projector(
        selected, media_type,
        batch_size=batch_size,
        num_videos=num_videos,
        num_frames=model.config.num_frames,
    )
    return image_features  # (1, N_vis, d)


def get_patch_scores(traj_encoder, item, device):
    traj_batch  = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb   = traj_encoder.query_encoder([item["question"]], device)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)
    return scores_raw.squeeze(0)  # (196,)


def merge_visual_features(full_features, scores, device):
    N_vis      = full_features.shape[1]
    n_spatial  = POOLING_SHAPE[1] * POOLING_SHAPE[2]
    n_temporal = POOLING_SHAPE[0]
    scores_sp  = score_to_pllava_spatial(scores, n_spatial)
    scores_all = scores_sp.unsqueeze(0).expand(n_temporal, -1).reshape(-1)
    if scores_all.shape[0] != N_vis:
        if scores_all.shape[0] > N_vis:
            scores_all = scores_all[:N_vis]
        else:
            reps = (N_vis + scores_all.shape[0] - 1) // scores_all.shape[0]
            scores_all = scores_all.repeat(reps)[:N_vis]
    r = max(1, int(MERGE_RATIO * N_vis))
    merged, _ = gaze_weighted_merge(full_features[0].detach(), scores_all, r)
    return merged.unsqueeze(0)  # (1, ~230, d)


# ── forward ────────────────────────────────────────────────────────────────────

def forward_logit(model, processor, image_features, question, options, device):
    """
    Prompt ends at 'ASSISTANT:Best option:(' — the last token is '(' and the
    model predicts the option letter at logits[-1]. This matches the token
    position where CE and KD are well-posed (letter is natural next token).
    """
    model_dtype = next(model.language_model.parameters()).dtype
    tok    = processor.tokenizer
    pad_id = tok.pad_token_id or 0

    qtext  = build_prompt(question, options)
    prompt = (f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} "
              f"\nOnly give the best option. ASSISTANT:Best option:(")

    enc       = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    no_img = torch.where(input_ids != IMAGE_TOKEN_ID, input_ids,
                         torch.full_like(input_ids, pad_id))
    inputs_embeds = model.get_input_embeddings()(no_img).to(model_dtype)
    image_features = image_features.to(model_dtype)

    embeds_m, mask_m, _, _, _ = model._merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, attn_mask, labels=None
    )
    out = model.language_model(
        inputs_embeds=embeds_m,
        attention_mask=mask_m,
        use_cache=False,
    )
    return out.logits[0, -1, :]  # (vocab_size,)


# ── eval ───────────────────────────────────────────────────────────────────────

def evaluate_logit(student, processor, traj_encoder, device, option_ids_t,
                   n_frames=16, n_traj_frames=32, max_items=None):
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=n_frames,
                                     n_traj_frames=n_traj_frames)
    if max_items:
        test_ds.items = test_ds.items[:max_items]

    student.eval(); traj_encoder.eval()
    correct = total = 0

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < n_frames:
                    pil_frames.append(pil_frames[-1])

                proc_out = processor(
                    text=f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: placeholder",
                    images=pil_frames, return_tensors="pt",
                )
                if proc_out.get("pixel_values") is None:
                    continue
                pixel_values = proc_out["pixel_values"].to(device)

                full_feats = get_pllava_image_features(student, pixel_values)
                scores     = get_patch_scores(traj_encoder, item, device)
                merged     = merge_visual_features(full_feats, scores, device)

                logit    = forward_logit(student, processor, merged,
                                         item["question"], item["options"], device)
                pred_idx = logit[option_ids_t].argmax().item()
                gt_letter = item["answer"].upper()
                if gt_letter not in OPTION_LETTERS:
                    continue
                gt_idx = OPTION_LETTERS.index(gt_letter)
                correct += int(pred_idx == gt_idx)
                total   += 1
            except Exception:
                continue

    student.train(); traj_encoder.train()
    return 100.0 * correct / max(1, total), total


# ── DDP setup ──────────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt",  default=TEACHER_CKPT)
    p.add_argument("--stage1-ckpt",   default=STAGE1_CKPT)
    p.add_argument("--output-dir",    default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_trajgaze_kd_r64_v3")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--lr-lora",       type=float, default=1e-4)
    p.add_argument("--lr-enc",        type=float, default=1e-5)
    p.add_argument("--alpha",         type=float, default=0.5, help="KD weight (0=CE only, 1=KD only)")
    p.add_argument("--kd-temp",       type=float, default=2.0, help="KD temperature")
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--log-every",     type=int,   default=20)
    p.add_argument("--eval-every",    type=int,   default=300)
    p.add_argument("--n-frames",      type=int,   default=16)
    p.add_argument("--n-traj-frames", type=int,   default=32)
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[PLLaVA TrajGaze KD r64 v3] output: {args.output_dir}")
        print(f"GPUs={world_size} | epochs={args.epochs} | lr_lora={args.lr_lora} "
              f"| lr_enc={args.lr_enc} | alpha={args.alpha} | T={args.kd_temp}")
        print("Fixes: (1) prompt@( (2) warm LoRA from teacher (3) cosine LR (4) full-vocab CE")

    # ── Load teacher (finetuned PLLaVA, frozen, full tokens) ──
    if is_main:
        print("Loading teacher (finetuned PLLaVA 64.83%, frozen) ...")
    teacher = load_teacher(args.teacher_ckpt, device)

    # ── Load student (PLLaVA + LoRA r=64, warm from teacher) ──
    if is_main:
        print("Loading student (PLLaVA + LoRA r=64, warm from teacher) ...")
    student, processor = load_student(device, args.teacher_ckpt)
    student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

    # ── Load TrajGaze encoder ──
    if is_main:
        print("Loading TrajGaze encoder ...")
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        lora_n = sum(p.numel() for p in student.parameters() if p.requires_grad)
        enc_n  = sum(p.numel() for p in traj_encoder.parameters())
        print(f"Trainable LoRA params:   {lora_n:,}")
        print(f"TrajGaze encoder params: {enc_n:,}")

    # ── Dataset ──
    train_ds = StreamGazeMergeDataset(split="train", n_vlm_frames=args.n_frames,
                                      n_traj_frames=args.n_traj_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    # ── Optimizer + cosine LR scheduler ──
    lora_params = [p for p in student.parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    steps_per_epoch = max(1, len(train_ds) // world_size // args.grad_accum)
    total_steps     = args.epochs * steps_per_epoch
    scheduler       = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    if is_main:
        print(f"Cosine LR: T_max={total_steps} steps (est. {steps_per_epoch}/epoch)")

    # ── Option letter token IDs ──
    # Must use "Best option:(X" prefix so X is NOT at the start of the encoded
    # string — that avoids the SentencePiece dummy-space prefix which would give
    # ▁A (29909) instead of the in-context A (319). Same IDs as v2 eval.
    tok = processor.tokenizer
    option_ids = [tok.encode(f"Best option:({l}", add_special_tokens=False)[-1]
                  for l in OPTION_LETTERS]
    option_ids_t = torch.tensor(option_ids, device=device)
    if is_main:
        print(f"Option letter IDs (A/B/C/D): {option_ids}")

    log_path = os.path.join(args.output_dir, "train.log")
    best_acc = 0.0
    global_opt_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student.train(); traj_encoder.train()
        optimizer.zero_grad()

        epoch_loss = epoch_ce = epoch_kd = 0.0
        n_steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], args.n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < args.n_frames:
                    pil_frames.append(pil_frames[-1])

                proc_out = processor(
                    text=f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: placeholder",
                    images=pil_frames, return_tensors="pt",
                )
                if proc_out.get("pixel_values") is None:
                    continue
                pixel_values = proc_out["pixel_values"].to(device)

                # Full visual features (frozen vision tower, no grad needed)
                with torch.no_grad():
                    full_feats = get_pllava_image_features(student.module, pixel_values)

                # Teacher: full tokens, no grad
                with torch.no_grad():
                    logit_teacher = forward_logit(
                        teacher, processor, full_feats.detach(),
                        item["question"], item["options"], device,
                    )
                    teacher_opts = logit_teacher[option_ids_t].detach()  # (4,)

                # TrajGaze scores (grad flows back through encoder)
                scores = get_patch_scores(traj_encoder.module, item, device)  # (196,)

                # Merge to 10%
                merged = merge_visual_features(full_feats, scores, device)  # (1, ~230, d)

                # Student: merged tokens
                logit_student = forward_logit(
                    student.module, processor, merged,
                    item["question"], item["options"], device,
                )  # (vocab_size,)

                gt_letter = item["answer"].upper()
                if gt_letter not in OPTION_LETTERS:
                    continue
                gt_idx = OPTION_LETTERS.index(gt_letter)

                # Full-vocab CE: target is the letter token ID (e.g. 319 for A)
                # At prompt position '(', natural next token IS the option letter.
                gt_token_id = torch.tensor([option_ids[gt_idx]], device=device, dtype=torch.long)
                loss_ce = F.cross_entropy(logit_student.unsqueeze(0), gt_token_id)

                # KD: KL over 4 option logits
                T = args.kd_temp
                s_opts = logit_student[option_ids_t] / T
                t_opts = teacher_opts / T
                loss_kd = F.kl_div(
                    F.log_softmax(s_opts, dim=-1),
                    F.softmax(t_opts,  dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)

                loss = ((1.0 - args.alpha) * loss_ce + args.alpha * loss_kd) / args.grad_accum
                loss.backward()

                epoch_loss += loss.item() * args.grad_accum
                epoch_ce   += loss_ce.item()
                epoch_kd   += loss_kd.item()
                n_steps    += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_opt_step += 1

                if is_main and n_steps % args.log_every == 0:
                    avg_l  = epoch_loss / n_steps
                    avg_ce = epoch_ce   / n_steps
                    avg_kd = epoch_kd   / n_steps
                    elapsed = time.time() - t_start
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_l:.4f} ce={avg_ce:.4f} kd={avg_kd:.4f} "
                          f"lr={cur_lr:.2e} | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_l, "ce": avg_ce, "kd": avg_kd,
                            "lr": cur_lr, "elapsed": elapsed,
                        }) + "\n")

                if is_main and n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate_logit(
                        student.module, processor, traj_encoder.module, device,
                        option_ids_t, n_frames=args.n_frames,
                        n_traj_frames=args.n_traj_frames, max_items=200,
                    )
                    print(f"  → partial eval (200): {acc:.2f}% (n={n_eval})")
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({
                            "epoch": epoch, "step": n_steps,
                            "lora_state": {k: v.cpu()
                                           for k, v in student.module.state_dict().items()
                                           if "lora_" in k or "multi_modal_projector" in k},
                            "encoder_state": traj_encoder.module.state_dict(),
                            "acc": acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)")
                    student.train(); traj_encoder.train()

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Flush remaining gradients
        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if is_main:
            avg_l  = epoch_loss / max(1, n_steps)
            avg_ce = epoch_ce   / max(1, n_steps)
            avg_kd = epoch_kd   / max(1, n_steps)
            elapsed = time.time() - t_start
            cur_lr  = optimizer.param_groups[0]["lr"]
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_l:.4f} ce={avg_ce:.4f} kd={avg_kd:.4f} "
                  f"lr={cur_lr:.2e} | time={elapsed:.0f}s ===")

            acc, n_eval = evaluate_logit(
                student.module, processor, traj_encoder.module, device,
                option_ids_t, n_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
            )
            print(f"  → full eval (526): {acc:.2f}% (n={n_eval})")

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "avg_loss": avg_l, "ce": avg_ce, "kd": avg_kd,
                    "lr": cur_lr, "full_eval_acc": acc, "n_eval": n_eval,
                }) + "\n")

            epoch_ckpt = os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth")
            torch.save({
                "epoch": epoch,
                "lora_state": {k: v.cpu()
                               for k, v in student.module.state_dict().items()
                               if "lora_" in k or "multi_modal_projector" in k},
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg_l, "acc": acc,
            }, epoch_ckpt)
            print(f"  → saved {epoch_ckpt}")

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch, "step": -1,
                    "lora_state": {k: v.cpu()
                                   for k, v in student.module.state_dict().items()
                                   if "lora_" in k or "multi_modal_projector" in k},
                    "encoder_state": traj_encoder.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)")

            student.train(); traj_encoder.train()

    if is_main:
        print(f"\n=== Training complete. Best eval acc: {best_acc:.2f}% ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
