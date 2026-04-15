"""
Extract hand locations (left/right) for every frame in EgoGazeVQA/{ego4d,egoexo,egtea}/no_gaze
using the handobj_100K Faster-RCNN detector.

Output per clip:
  <output_root>/{dataset}/hand_locations/{clip_id}.json
  {
    "frame_name.jpg": {"left": [cx, cy], "right": [cx, cy]},
    ...
  }
  cx/cy are pixel coordinates. Values are null when hand not detected.

Usage (2 GPUs):
  CUDA_VISIBLE_DEVICES=0 python preprocess/extract_hand_locations_egogaze.py --shard 0 --n-shards 2 &
  CUDA_VISIBLE_DEVICES=1 python preprocess/extract_hand_locations_egogaze.py --shard 1 --n-shards 2 &
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

DETECTOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hand_object_detector"))
sys.path.insert(0, os.path.join(DETECTOR_ROOT, "lib"))

from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file

CFG_FILE      = os.path.join(DETECTOR_ROOT, "cfgs", "res101.yml")
PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])
THRESH_HAND   = 0.5

DATA_ROOT = "/workspace/datasets/EgoGazeVQA"
CKPT_PATH = "/workspace/EgoGazeVQA/hand_object_detector/models/res101_handobj_100k/pascal_voc/faster_rcnn_1_8_132028.pth"
DATASETS  = ["ego4d", "egoexo", "egtea"]


# ── Detector helpers (identical to extract_hand_locations.py) ──────────────────

def _get_image_blob(im: np.ndarray):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])
    processed_ims, im_scale_factors = [], []
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                             interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(resized)
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def load_model(ckpt_path: str, cuda: bool):
    cfg_from_file(CFG_FILE)
    cfg.USE_GPU_NMS = cuda
    cfg.ANCHOR_SCALES = [8, 16, 32, 64]
    cfg.ANCHOR_RATIOS = [0.5, 1, 2]
    model = resnet(PASCAL_CLASSES, 101, pretrained=False, class_agnostic=False)
    model.create_architecture()
    print(f"Loading checkpoint: {ckpt_path}")
    map_loc = None if cuda else (lambda storage, loc: storage)
    checkpoint = torch.load(ckpt_path, map_location=map_loc)
    model.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint:
        cfg.POOLING_MODE = checkpoint["pooling_mode"]
    if cuda:
        model.cuda()
        cfg.CUDA = True
    model.eval()
    print("Model loaded.")
    return model


def detect_hands(model, im_bgr: np.ndarray, cuda: bool):
    blob, im_scales = _get_image_blob(im_bgr)
    im_scale  = im_scales[0]
    im_blob   = blob
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scale]], dtype=np.float32)
    im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data   = torch.FloatTensor(1)
    im_info   = torch.FloatTensor(1)
    gt_boxes  = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    box_info  = torch.FloatTensor(1)

    if cuda:
        im_data   = im_data.cuda()
        im_info   = im_info.cuda()
        gt_boxes  = gt_boxes.cuda()
        num_boxes = num_boxes.cuda()

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()

        rois, cls_prob, bbox_pred, \
        _, _, _, _, _, loss_list = model(im_data, im_info, gt_boxes, num_boxes, box_info)

    scores = cls_prob.data
    boxes  = rois.data[:, :, 1:5]
    contact_vector = loss_list[0][0]
    offset_vector  = loss_list[1][0].detach()
    lr_vector      = loss_list[2][0].detach()

    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

    box_deltas = bbox_pred.data
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        stds  = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        if cuda:
            stds, means = stds.cuda(), means.cuda()
        box_deltas = (box_deltas.view(-1, 4) * stds + means).view(1, -1, 4 * len(PASCAL_CLASSES))

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    pred_boxes /= im_scale
    scores     = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    hand_dets = None
    hand_cls_idx = 2
    inds = torch.nonzero(scores[:, hand_cls_idx] > THRESH_HAND).view(-1)
    if inds.numel() > 0:
        cls_scores = scores[:, hand_cls_idx][inds]
        _, order   = torch.sort(cls_scores, 0, True)
        cls_boxes  = pred_boxes[inds][:, hand_cls_idx * 4:(hand_cls_idx + 1) * 4]
        cls_dets   = torch.cat(
            (cls_boxes, cls_scores.unsqueeze(1),
             contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]),
            dim=1,
        )
        cls_dets = cls_dets[order]
        keep     = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        hand_dets = cls_dets.cpu().numpy()

    return hand_dets


EDGE_MARGIN    = 0.04
MIN_AREA_RATIO = 0.003
MIN_SIDE_RATIO = 0.03


def _is_valid_det(det, img_w: int, img_h: int) -> bool:
    x1, y1, x2, y2 = det[:4]
    bw, bh = x2 - x1, y2 - y1
    if bw * bh < MIN_AREA_RATIO * img_w * img_h:
        return False
    if bw < MIN_SIDE_RATIO * img_w or bh < MIN_SIDE_RATIO * img_h:
        return False
    margin_x = EDGE_MARGIN * img_w
    margin_y = EDGE_MARGIN * img_h
    if x1 < margin_x or y1 < margin_y or x2 > img_w - margin_x or y2 > img_h - margin_y:
        return False
    return True


def parse_hands(hand_dets, img_w: int, img_h: int):
    if hand_dets is None or len(hand_dets) == 0:
        return None, None
    left_det = right_det = None
    for det in hand_dets:
        if not _is_valid_det(det, img_w, img_h):
            continue
        side = int(det[-1])
        if side == 0 and left_det is None:
            left_det = det
        elif side == 1 and right_det is None:
            right_det = det

    def center(det):
        x1, y1, x2, y2 = det[:4]
        return [round(float(np.clip((x1+x2)/2, 0, img_w)), 1),
                round(float(np.clip((y1+y2)/2, 0, img_h)), 1)]

    return (center(left_det) if left_det is not None else None,
            center(right_det) if right_det is not None else None)


# ── Clip enumeration ───────────────────────────────────────────────────────────

def collect_clips():
    """Return list of (dataset, clip_id, no_gaze_dir, out_path)."""
    clips = []
    for ds in DATASETS:
        no_gaze_root = os.path.join(DATA_ROOT, ds, "no_gaze")
        out_root     = os.path.join(DATA_ROOT, ds, "hand_locations")
        if not os.path.isdir(no_gaze_root):
            print(f"WARNING: {no_gaze_root} not found, skipping.")
            continue
        for clip_id in sorted(os.listdir(no_gaze_root)):
            clip_dir = os.path.join(no_gaze_root, clip_id)
            if not os.path.isdir(clip_dir):
                continue
            out_path = os.path.join(out_root, f"{clip_id}.json")
            clips.append((ds, clip_id, clip_dir, out_path))
    return clips


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",    type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--no-cuda",  action="store_true")
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()
    model = load_model(CKPT_PATH, cuda)

    all_clips = collect_clips()
    print(f"Total clips: {len(all_clips)}")

    # Shard by clip
    clips = all_clips[args.shard::args.n_shards]
    # Skip already done
    todo = [(ds, cid, cdir, op) for ds, cid, cdir, op in clips if not os.path.exists(op)]
    print(f"Shard {args.shard}/{args.n_shards}: {len(clips)} clips assigned, {len(todo)} remaining")

    total_frames = 0
    for ds, clip_id, clip_dir, out_path in tqdm(todo, desc=f"shard{args.shard}"):
        frames = sorted(f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg"))
        if not frames:
            continue

        results = {}
        for fname in frames:
            fpath = os.path.join(clip_dir, fname)
            im = cv2.imread(fpath)
            if im is None:
                results[fname] = {"left": None, "right": None}
                continue
            h, w = im.shape[:2]
            try:
                hand_dets = detect_hands(model, im, cuda)
                left, right = parse_hands(hand_dets, w, h)
            except Exception:
                left, right = None, None
            results[fname] = {"left": left, "right": right}

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f)
        total_frames += len(results)

    print(f"Shard {args.shard} done. {total_frames} frames processed.")


if __name__ == "__main__":
    main()
