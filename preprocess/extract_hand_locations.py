"""
Extract hand locations (left/right) for every frame in StreamGaze_v2/frames
using the handobj_100K Faster-RCNN detector.

Output per video:
  <output_dir>/{dataset}/{split}/{video_stem}.json
  {
    "frame_000001.jpg": {"left": [cx, cy], "right": null},
    "frame_000002.jpg": {"left": null,     "right": [cx, cy]},
    ...
  }

cx/cy are pixel coordinates in the original (unscaled) image.
Values are null when the hand is not detected above thresh_hand.

Usage:
  CUDA_VISIBLE_DEVICES=0 python preprocess/extract_hand_locations.py \
      --frames-dir /workspace/datasets/StreamGaze_v2/frames \
      --output     /workspace/EgoGazeVQA/preprocess/hand_locations \
      --ckpt       /workspace/EgoGazeVQA/hand_object_detector/models/res101_handobj_100k/pascal_voc/faster_rcnn_1_8_132028.pth
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

# ── Repo paths ─────────────────────────────────────────────────────────────────
DETECTOR_ROOT = os.path.join(os.path.dirname(__file__), "..", "hand_object_detector")
DETECTOR_ROOT = os.path.abspath(DETECTOR_ROOT)
sys.path.insert(0, os.path.join(DETECTOR_ROOT, "lib"))

from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file

# ── Constants ──────────────────────────────────────────────────────────────────
CFG_FILE   = os.path.join(DETECTOR_ROOT, "cfgs", "res101.yml")
PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])
THRESH_HAND = 0.5   # confidence threshold for hand detections


def _get_image_blob(im: np.ndarray):
    """Preprocess a BGR image into a Faster-RCNN input blob."""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape   = im_orig.shape
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


def load_model(ckpt_path: str, cuda: bool) -> torch.nn.Module:
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
    """
    Run detection on one BGR image.

    Returns:
        hand_dets: numpy array (N, 10+) with columns
                   [x1, y1, x2, y2, score, contact, offset..., lr]
                   or None if no hands detected above threshold.
    """
    blob, im_scales = _get_image_blob(im_bgr)
    im_scale = im_scales[0]

    im_blob     = blob
    im_info_np  = np.array([[im_blob.shape[1], im_blob.shape[2], im_scale]], dtype=np.float32)

    im_data_pt  = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
    im_info_pt  = torch.from_numpy(im_info_np)

    im_data  = torch.FloatTensor(1)
    im_info  = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    box_info  = torch.FloatTensor(1)

    if cuda:
        im_data  = im_data.cuda()
        im_info  = im_info.cuda()
        gt_boxes = gt_boxes.cuda()
        num_boxes = num_boxes.cuda()

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()

        rois, cls_prob, bbox_pred, \
        _, _, _, _, _, loss_list = model(im_data, im_info, gt_boxes, num_boxes, box_info)

    scores    = cls_prob.data
    boxes     = rois.data[:, :, 1:5]

    contact_vector = loss_list[0][0]
    offset_vector  = loss_list[1][0].detach()
    lr_vector      = loss_list[2][0].detach()

    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

    lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

    # Bounding-box regression
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
    hand_cls_idx = 2  # 'hand' is index 2 in PASCAL_CLASSES

    inds = torch.nonzero(scores[:, hand_cls_idx] > THRESH_HAND).view(-1)
    if inds.numel() > 0:
        cls_scores = scores[:, hand_cls_idx][inds]
        _, order   = torch.sort(cls_scores, 0, True)
        cls_boxes  = pred_boxes[inds][:, hand_cls_idx * 4:(hand_cls_idx + 1) * 4]

        cls_dets = torch.cat(
            (cls_boxes, cls_scores.unsqueeze(1),
             contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]),
            dim=1,
        )
        cls_dets = cls_dets[order]
        keep     = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        hand_dets = cls_dets.cpu().numpy()

    return hand_dets


EDGE_MARGIN      = 0.04   # bbox clipped if any edge within 4% of image boundary
MIN_AREA_RATIO   = 0.003  # bbox must cover at least 0.3% of image area
MIN_SIDE_RATIO   = 0.03   # both bbox width and height must be ≥ 3% of image dimension


def _is_valid_det(det, img_w: int, img_h: int) -> bool:
    """Return False for edge-clipped or too-small detections (likely partial/noise)."""
    x1, y1, x2, y2 = det[:4]
    bw = x2 - x1
    bh = y2 - y1

    # Reject tiny boxes
    if bw * bh < MIN_AREA_RATIO * img_w * img_h:
        return False
    if bw < MIN_SIDE_RATIO * img_w or bh < MIN_SIDE_RATIO * img_h:
        return False

    # Reject boxes clipped by image boundary (any side within edge margin)
    margin_x = EDGE_MARGIN * img_w
    margin_y = EDGE_MARGIN * img_h
    if x1 < margin_x or y1 < margin_y or x2 > img_w - margin_x or y2 > img_h - margin_y:
        return False

    return True


def parse_hands(hand_dets, img_w: int, img_h: int):
    """
    Given hand_dets array, return left and right hand centers.

    hand_dets columns: [x1, y1, x2, y2, score, contact, ..., lr]
    lr = 1 → right hand, lr = 0 → left hand.
    Detections clipped at image edges or with tiny bounding boxes are rejected.
    Among valid detections, the highest-score per side is returned (dets are pre-sorted).

    Returns:
        left:  [cx, cy] (floats) or None
        right: [cx, cy] (floats) or None
    """
    if hand_dets is None or len(hand_dets) == 0:
        return None, None

    left_det  = None
    right_det = None
    for det in hand_dets:
        if not _is_valid_det(det, img_w, img_h):
            continue
        side = int(det[-1])  # 0=left, 1=right
        if side == 0 and left_det is None:
            left_det = det
        elif side == 1 and right_det is None:
            right_det = det

    def center(det):
        x1, y1, x2, y2 = det[:4]
        cx = float(np.clip((x1 + x2) / 2, 0, img_w))
        cy = float(np.clip((y1 + y2) / 2, 0, img_h))
        return [round(cx, 1), round(cy, 1)]

    left  = center(left_det)  if left_det  is not None else None
    right = center(right_det) if right_det is not None else None
    return left, right


def collect_videos(frames_dir: str):
    """Walk frames_dir and return list of (dataset, split, stem, frame_dir)."""
    videos = []
    for dataset in sorted(os.listdir(frames_dir)):
        dataset_dir = os.path.join(frames_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for split in sorted(os.listdir(dataset_dir)):
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for stem in sorted(os.listdir(split_dir)):
                frame_dir = os.path.join(split_dir, stem)
                if os.path.isdir(frame_dir):
                    videos.append((dataset, split, stem, frame_dir))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True,
                        help="Root of StreamGaze_v2/frames")
    parser.add_argument("--output", required=True,
                        help="Output directory for per-video JSON files")
    parser.add_argument("--ckpt", required=True,
                        help="Path to faster_rcnn_1_8_132028.pth")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU (slow)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip videos whose JSON already exists")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Process at most N videos (for testing)")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index (0-based) for multi-GPU parallelism")
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Total number of shards")
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()
    model = load_model(args.ckpt, cuda)

    videos = collect_videos(args.frames_dir)
    print(f"Found {len(videos)} videos under {args.frames_dir}")
    if args.max_videos:
        videos = videos[:args.max_videos]
    if args.n_shards > 1:
        videos = videos[args.shard::args.n_shards]
        print(f"Shard {args.shard}/{args.n_shards}: {len(videos)} videos")

    total_frames = 0
    for dataset, split, stem, frame_dir in tqdm(videos, desc="Videos"):
        out_dir  = os.path.join(args.output, dataset, split)
        out_path = os.path.join(out_dir, f"{stem}.json")

        if args.skip_existing and os.path.exists(out_path):
            continue

        frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
        if not frames:
            continue

        results = {}
        for fname in frames:
            fpath = os.path.join(frame_dir, fname)
            im = cv2.imread(fpath)
            if im is None:
                results[fname] = {"left": None, "right": None}
                continue

            h, w = im.shape[:2]
            try:
                hand_dets = detect_hands(model, im, cuda)
                left, right = parse_hands(hand_dets, w, h)
            except Exception as e:
                left, right = None, None

            results[fname] = {"left": left, "right": right}

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f)

        total_frames += len(results)

    print(f"Done. Processed {total_frames} frames across {len(videos)} videos.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
