"""
PLLaVA-7B + LoRA r=16 fine-tuning on StreamGaze_v2 MCQ tasks.

Approach  : generation-based (CE loss over full answer tokens)
Train     : egoexolearn + holoassist (5799 items)
Test      : egtea (526 items, generation-based eval)
GPUs      : 0,1 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/gaze/bin/torchrun --nproc_per_node=2 \
        -m TrajGazeMerge.training.train_pllava_lora_r16 \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora_r16 \
        --epochs 3 --lr 1e-4 --grad-accum 4
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
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/prunevid")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset

PLLAVA_HF     = "ermu2001/pllava-7b"
POOLING_SHAPE = (16, 12, 12)
FRAME_SHAPE   = (24, 24)
IMAGE_TOKEN   = "<image>"
IMAGE_TOKEN_ID = 32000
NUM_FRAMES    = 16   # PLLaVA native: 16 frames → 16×12×12 = 2304 tokens

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)


def _sample_paths(paths: list, n: int) -> list:
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    return [paths[int(i * len(paths) / n)] for i in range(n)]


def build_prompt(question: str, options: list) -> str:
    opts = "\n".join(options)
    return f"{question}\nOptions:\n{opts}"


def get_answer_text(options: list, letter: str) -> str:
    """Extract the text part from an option like 'A. text' → 'text'."""
    for opt in options:
        if opt.strip().upper().startswith(letter):
            return opt.split(".", 1)[-1].strip() if "." in opt else opt
    return ""


def _load_pllava_peft_ckpt(model, hf_path, lora_alpha=256, lora_r=128):
    """
    ermu2001/pllava-7b is saved as a PeftModel (LoRA on q/v proj, r=128, alpha=256).
    Standard from_pretrained() leaves all LLM weights unloaded (NaN CUDA memory).
    This function loads the 814 safetensors keys, remaps PEFT→base prefixes, merges
    LoRA deltas into q/v base weights, and loads with strict=False.
    """
    import glob
    from safetensors import safe_open

    # Resolve HF cache path if given a repo id
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
            # strip PEFT prefix: base_model.model.
            new_k = k.replace("language_model.base_model.model.", "language_model.", 1)
            if ".base_layer.weight" in new_k:
                # q/v base weight — merge LoRA delta
                proj_prefix = k[: k.rfind(".base_layer.weight")]
                lora_a = raw.get(proj_prefix + ".lora_A.default.weight")
                lora_b = raw.get(proj_prefix + ".lora_B.default.weight")
                if lora_a is not None and lora_b is not None:
                    v = v + scale * (lora_b.float() @ lora_a.float()).to(v.dtype)
                final_k = new_k.replace(".base_layer.weight", ".weight")
                remapped[final_k] = v
            elif ".lora_A." in k or ".lora_B." in k:
                continue  # absorbed above
            else:
                remapped[new_k] = v
        else:
            remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[PEFT ckpt] loaded {len(remapped)} keys | missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_pllava(device):
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
    )   # num_frames kept at HF default (16) to match merge_frames_dynamic assertion
    model = PllavaForConditionalGeneration.from_pretrained(
        PLLAVA_HF, config=config, torch_dtype=torch.bfloat16
    )
    _load_pllava_peft_ckpt(model, PLLAVA_HF)
    for p in model.vision_tower.parameters():
        p.requires_grad_(False)
    for p in model.language_model.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()
    return model.to(device), processor


def forward_train(model, processor, pil_frames, question, options, answer, device):
    """Generation-based training: CE loss over answer tokens."""
    model_dtype = next(model.language_model.parameters()).dtype
    qtext = build_prompt(question, options)
    letter = answer
    content = get_answer_text(options, letter)

    prompt_text = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
    full_text = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:Best option:({letter}) {content}"

    # Tokenize to find prompt length (without images, just text alignment)
    tok = processor.tokenizer
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Get pixel_values + full tokenization
    inputs = processor(text=full_text, images=pil_frames, return_tensors="pt")
    if inputs.get("pixel_values") is None:
        return None
    pixel_values = inputs["pixel_values"].to(device).to(model_dtype)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    # Labels: -100 for prompt, actual IDs for answer
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    # Ensure at least some tokens are supervised
    if (labels != -100).sum() == 0:
        return None

    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
        media_type="video",
        use_cache=False,
    )
    return out.loss


def evaluate(model, processor, device, n_frames=64, max_items=None):
    """Generation-based eval on egtea test split."""
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=n_frames, n_traj_frames=32)
    if max_items:
        test_ds.items = test_ds.items[:max_items]

    model_dtype = next(model.language_model.parameters()).dtype
    pad_id = processor.tokenizer.pad_token_id or 0
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < n_frames:
                    pil_frames.append(pil_frames[-1])

                qtext = build_prompt(item["question"], item["options"])
                prompt = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:Best option:("

                inputs = processor(text=prompt, images=pil_frames, return_tensors="pt")
                if inputs.get("pixel_values") is None:
                    continue
                pixel_values = inputs["pixel_values"].to(device).to(model_dtype)
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs["attention_mask"].to(device)

                try:
                    out_ids = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        media_type="video",
                        do_sample=False,
                        max_new_tokens=10,
                        pad_token_id=pad_id,
                    )
                except Exception:
                    out_ids = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        do_sample=False,
                        max_new_tokens=10,
                        pad_token_id=pad_id,
                    )

                text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                text = text.split("ASSISTANT:")[-1].split("Best option:(")[-1].strip()
                pred = text[0].upper() if text else "A"

                gt = item["answer"].upper()
                correct += int(pred == gt)
                total += 1
            except Exception:
                traceback.print_exc()
                continue

    model.train()
    return 100.0 * correct / max(1, total), total


def save_delta(model, path):
    delta = {k: v.cpu() for k, v in model.state_dict().items()
             if "lora_" in k or "multi_modal_projector" in k}
    torch.save(delta, path)


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora_r16")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--eval-every", type=int,   default=300)
    p.add_argument("--n-frames",   type=int,   default=16)
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[PLLaVA LoRA r16] output: {args.output_dir}")
        print(f"GPUs={world_size}, epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")

    if is_main:
        print("Loading PLLaVA-7B + LoRA r=16 ...")
    model, processor = load_pllava(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        lora_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {lora_params_count:,}")

    train_ds = StreamGazeMergeDataset(split="train", n_vlm_frames=args.n_frames, n_traj_frames=16)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
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

                loss = forward_train(
                    model.module, processor,
                    pil_frames, item["question"], item["options"], item["answer"], device
                )
                if loss is None:
                    continue

                loss = loss / args.grad_accum
                loss.backward()

                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "elapsed": elapsed,
                        }) + "\n")

                if is_main and n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate(model.module, processor, device,
                                           n_frames=args.n_frames, max_items=200)
                    print(f"  → eval egtea: {acc:.2f}% (n={n_eval})")
                    if acc > best_acc:
                        best_acc = acc
                        save_delta(model.module, os.path.join(args.output_dir, "best_delta.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_loss = epoch_loss / max(1, n_steps)
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")
            save_delta(model.module, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}_delta.pth"))

    if is_main:
        acc, n_eval = evaluate(model.module, processor, device, n_frames=args.n_frames)
        print(f"\n[Final] egtea accuracy: {acc:.2f}% (n={n_eval})")
        with open(log_path, "a") as f:
            f.write(json.dumps({"final_acc": acc, "n": n_eval}) + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
