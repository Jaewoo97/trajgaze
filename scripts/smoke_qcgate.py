"""Single-GPU smoke test for QC-Gate before the full 4-GPU launch.

Validates, on a handful of real items spanning all three sources (sg / eg / hd):
  1. selection produces a grad-carrying merged_video in training mode;
  2. loss is finite;
  3. the controller's OUTPUT layer (fc2) gets non-zero gradient on the very first
     backward — this is the end-to-end proof that the straight-through gate path
     (loss -> LLM -> inputs_embeds -> merged_video -> gate -> controller) is intact;
  4. after one optimizer step the INPUT layers (fc1 / q_proj) unlock (non-zero grad),
     confirming the progressive init behaves as designed;
  5. gate means start at ~(1.0, 0.7, 0.0) == VZ-traj and then move;
  6. the eval path (training=False, no gate) runs.

Emits a final 'QCGATE_SMOKE: HEALTHY' or 'QCGATE_SMOKE: FAILED <reason>' line.
"""
from __future__ import annotations

import sys
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import build_merged_inputs, forward_logits, get_option_ids
from TrajGazeMerge.models.qc_gate import QCGateController
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_qcgate_lora_3way import qcgate_select


def fail(msg):
    print(f"QCGATE_SMOKE: FAILED {msg}", flush=True)
    sys.exit(1)


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    hp = dict(sigma_g=2.0, sigma_h=3.0, sigma_v=0.05, sigma_gh=0.10)

    print("[smoke] loading VisionZip + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    controller = QCGateController().to(device)
    option_ids = get_option_ids(processor, 5)
    model.train(); controller.train()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    ctrl_params = [p for p in controller.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params + ctrl_params, lr=1e-4, weight_decay=1e-4)

    # grab one item from each UNDERLYING source (sg/eg/hd) so all three traj
    # schemas are exercised — index the sub-datasets directly (sg & eg both
    # report dataset=="egtea" in test, so dedup-by-label would miss one).
    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=True)
    picks = []
    for src in ("sg", "eg", "hd"):
        sub = ds._src.get(src)
        if sub is None:
            continue
        for li in range(len(sub)):
            it = sub[li]
            if it is not None:
                picks.append(it)
                print(f"[smoke] {src}: item {li} dataset={it['dataset']}", flush=True)
                break
    if not picks:
        fail("no_items")

    init_gate = None
    for step, item in enumerate(picks * 2):   # ~6 steps so fc2 updates then fc1 unlocks
        try:
            with torch.no_grad():
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device)
            if cached is None:
                print(f"[smoke] step {step}: cached None ({item['dataset']}), skip", flush=True)
                continue

            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters:
                continue

            sel_embeds, recv_idx, gm = qcgate_select(
                cached, item, controller, base_qwen, device, hp, training=True)

            if not sel_embeds.requires_grad:
                fail("sel_embeds_no_grad")
            if init_gate is None:
                init_gate = gm

            inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
            logits = forward_logits(model, inputs_dict)
            gt_idx = letters.index(item["answer"])
            loss = F.cross_entropy(logits[option_ids[:n_opt]].unsqueeze(0),
                                   torch.tensor([gt_idx], device=device))
            if not torch.isfinite(loss):
                fail(f"nonfinite_loss_step{step}")

            opt.zero_grad()
            loss.backward()

            mod = controller
            fc2_w_g = mod.fc2.weight.grad
            fc2_b_g = mod.fc2.bias.grad
            fc1_w_g = mod.fc1.weight.grad
            qp_w_g  = mod.q_proj.weight.grad

            fc2_norm = (fc2_w_g.norm().item() if fc2_w_g is not None else 0.0) \
                     + (fc2_b_g.norm().item() if fc2_b_g is not None else 0.0)
            fc1_norm = fc1_w_g.norm().item() if fc1_w_g is not None else 0.0
            qp_norm  = qp_w_g.norm().item()  if qp_w_g  is not None else 0.0

            print(f"[smoke] step {step} src={item['dataset']:6s} loss={loss.item():.4f} "
                  f"gates=({gm[0]:+.3f},{gm[1]:+.3f},{gm[2]:+.3f}) "
                  f"|grad| fc2={fc2_norm:.2e} fc1={fc1_norm:.2e} qproj={qp_norm:.2e}", flush=True)

            if step == 0 and fc2_norm <= 0.0:
                fail("fc2_zero_grad_step0 (straight-through path broken)")

            opt.step()

            # after fc2 has moved (step >= 1), fc1/q_proj must unlock
            if step >= 2 and (fc1_norm <= 0.0 or qp_norm <= 0.0):
                fail(f"input_layers_locked_step{step} fc1={fc1_norm:.2e} qproj={qp_norm:.2e}")
        except Exception:
            traceback.print_exc()
            fail(f"exception_step{step}")

    # init gates must equal the VZ-traj prior (1.0, 0.7, 0.0)
    if init_gate is None:
        fail("no_successful_step")
    g, h, t = init_gate
    if abs(g - 1.0) > 0.05 or abs(h - 0.7) > 0.05 or abs(t) > 0.05:
        fail(f"init_gates_off ({g:.3f},{h:.3f},{t:.3f}) != (1.0,0.7,0.0)")
    print(f"[smoke] init gates == VZ-traj prior: ({g:.3f},{h:.3f},{t:.3f}) OK", flush=True)

    # eval path (no gate)
    try:
        controller.eval(); model.eval()
        with torch.no_grad():
            item = picks[0]
            cached = preprocess_visionzip_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device)
            sel_embeds, recv_idx, _ = qcgate_select(
                cached, item, controller, base_qwen, device, hp, training=False)
            if sel_embeds.requires_grad:
                fail("eval_sel_requires_grad")
            inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
            logits = forward_logits(model, inputs_dict)
            _ = logits[option_ids[:len(item["options"])]].argmax().item()
        print("[smoke] eval path OK", flush=True)
    except Exception:
        traceback.print_exc()
        fail("eval_path_exception")

    mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[smoke] peak GPU mem: {mem:.1f} GB", flush=True)
    print("QCGATE_SMOKE: HEALTHY", flush=True)


if __name__ == "__main__":
    main()
