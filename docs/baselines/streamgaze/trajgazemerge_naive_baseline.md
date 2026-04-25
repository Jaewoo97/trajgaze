# TrajGazeMerge — Naive TokenDrop Baseline (ViT + LLM LoRA)

TrajGazeMerge 의 learned encoder + soft merge + KL distillation 조합에 대비되는 **rule-based ablation baseline**. `TrajGaze/baselines/patch_selector.py` (Talk2DINO sim + gaze/hand Gaussian) 로 visual token 을 **offline 에서 선택**해 `.pt` 로 저장하고, 학습 loop 에서는 Qwen2.5-VL ViT 내부에 token drop 을 주입한 커스텀 forward 로 **선택된 token 만** blocks/merger/LLM 에 통과시킨다. **ViT 와 LLM 양쪽에 LoRA** 를 붙여 공동 finetune.

```
 StreamGaze_v2 JPG frames (T)      qa/*.json (question)       gaze/*.json + hand/*.json
         │                                │                              │
         ▼                                ▼                              ▼
  DINOv2 ViT-L/14             Talk2DINO text enc              Gaussian prior
  (T, 1369, 1024)                 (1, 1024)              (gaze+L/R hand → L1 norm)
         │                                │                              │
         └───────────── cos sim ──────────┘                              │
                        ▼                                                 ▼
                    sim(t,p) ──────────── priority = α·norm(sim) + α·norm(w)
                                                     │
                                                     ▼
                            adaptive_avg_pool2d (37×37 → 8×8)
                                                     │
                                                     ▼  ★ pair-mean: (T_vlm, 64) → (T_vlm/2, 64)
                                                     ▼     (Qwen temporal_patch_size=2 정렬)
                                                     ▼  top-10% per pair (7/64 patches)
                                   gazing_mask_8x8   (T_merged, 8, 8) bool
                                                     │  (v2, 기본값)
                                                     │
                                                     ▼
                               {stem}/{q_hash}.pt       ←── Phase 1 outputs
                                                     │
═══════════════════════════════════════════════════════════════════════════════
                                                     │
                                                     ▼  Phase 2 (training)
  128 JPG frames ─── Qwen processor ─── pixel_values_videos (T_raw, 3, 224, 224)
                                                     │
                                                     ▼
                         Qwen Vision Tower  (FROZEN, original forward)
                         patch_embed → 32 blocks → VisionPatchMerger
                                                     │
                                                     ▼
                      full_embeds  (T_merged × 64 = 4096, d_llm)
                                                     │
                                           keep_mask_flat (bool, 4096)
                                                     ▼
                      kept_embeds  (N_kept ≈ 0.109 × 4096, d_llm)
                                                     │
                                                     ▼
                drop <video> positions (not kept) from input_ids / attn / RoPE
                                                     │
                                                     ▼
                       splice kept_embeds into remaining <video> positions
                                                     │
                                                     ▼
                                    Qwen LLM  (LoRA, trainable)
                                                     │
                                                     ▼
                                   CE loss on A/B/C/D logits
```

전체 배경 / TrajGazeMerge 본체 파이프라인은 `/home/yujin/gaze/trajgaze/TrajGazeMerge/IMPLEMENTATION.md` 참조.

---

## 1. Environment

- Conda env: `trajgaze` (`/opt/conda/envs/trajgaze`), Python 3.10, torch 2.10, transformers 4.51.3.
- 세션마다 export:
  ```bash
  export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8
  export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- 하드웨어: 2×H200 가정 (bf16, `attn_implementation="sdpa"`).
- HF cache: `lorebianchi98/Talk2DINO-ViTL` (첫 Phase 1 실행 시 자동 캐싱).

---

## 2. Data assets (read-only)

| 용도 | 경로 |
| --- | --- |
| Pre-extract JPG | `/home/yujin/dataset/StreamGaze_v2/frames/{dataset}/viz/{stem}/frame_NNNNNN.jpg` |
| Gaze JSON (pixel 좌표) | `/home/yujin/dataset/StreamGaze_v2/gaze/{dataset}/viz/{stem}.json` |
| Hand JSON | `/home/yujin/dataset/StreamGaze_v2/hand/{dataset}/viz/{stem}.json` |
| Interaction NPZ (본 베이스라인 불사용) | `/home/yujin/dataset/StreamGaze_v2/interaction/{dataset}/viz/{stem}.npz` |
| QA metadata | `/home/yujin/dataset/StreamGaze_v2/qa/*.json` |

Split 정의 (기존 `StreamGazeMergeDataset` 과 동일):
- `train` = egoexolearn + holoassist (5,799 QA, 246 unique stems)
- `test`  = egtea (526 QA, 35 unique stems)
- 8 MCQ tasks (proactive_* 제외)

Train/test 는 **source dataset 단위 disjoint** → test 는 학습에서 본 적 없는 egtea 도메인으로 generalization 측정. Path B 의 video_id group split 과 성격이 다름.

Per-task QA 수:

| Task | train | test |
|---|---:|---:|
| past_gaze_sequence_matching | 122 | 64 |
| past_non_fixated_object_identification | 570 | 68 |
| past_object_transition_prediction | 492 | **2** |
| past_scene_recall | 174 | 37 |
| present_future_action_prediction | 827 | 94 |
| present_object_attribute_recognition | 1,305 | 96 |
| present_object_identification_easy | 1,368 | 101 |
| present_object_identification_hard | 941 | 64 |
| **합계** | **5,799** | **526** |

`past_object_transition_prediction` 은 egtea test 에 2개만 있어 통계적 의미가 낮음 — per-task eval 시 제외/별도 표기 권장.

**주의**: 현재 `TrajGazeMerge/data/dataset.py:29-33` 은 `/workspace/datasets/StreamGaze_v2/` 를 하드코딩. 이 머신에서 돌리려면 5개 상수를 `/home/yujin/dataset/StreamGaze_v2/` 로 수정 필요. (본 베이스라인 구현 시 함께 수정)

---

## 3. Phase 1 — Offline precompute (1회 실행)

### 출력

```
/home/yujin/dataset/StreamGaze_v2/naive_masks_8x8/
└── {dataset}/{stem}/{md5(stem|question)[:8]}.pt
```

### `.pt` 스키마

```python
{
    # v2 (기본 --temporal-agg mean): (T_merged = T_vlm/2, 8, 8) — 정확히 budget 10%
    # v1 (--temporal-agg none): (T_vlm, 8, 8) — per-VLM-frame top-K, 학습 시 OR 필요 (~19%)
    "keep_mask_8x8": torch.bool,
    "frame_names":   list[str],         # (T_vlm=128,) 원본 frame basename (trace 용도)
    "meta": {
        "question": str,
        "options":  list[str],
        "answer":   "A"/"B"/"C"/"D",
        "task":     str,
        "dataset":  str,
        "stem":     str,
        "ts_sec":   float,
        "n_vlm_frames": int,
        "selector_cfg": dict,            # scales, patch_size, alphas, ratio, temporal_agg
    },
}
```

- `q_hash = md5(f"{stem}|{question}")[:8]`
- 디스크 사용: (128 × 8 × 8) bool ≈ 8KB + meta ≈ 10KB/QA. 수천 QA → 수십 MB.

### 파이프라인 내부

| 단계 | 요지 |
| --- | --- |
| ① QA 순회 | `StreamGazeMergeDataset` 의 QA 인덱싱 재사용 (train+test 양쪽) |
| ② Frame 샘플 | `time_stamp × 10 fps` 까지 JPG 에서 linspace 샘플 (최대 `n_vlm_frames=128`) |
| ③ Gaze/Hand 로드 | `{stem}.json` 에서 frame_name 기준 lookup → 픽셀 좌표 [0,1] 정규화 |
| ④ Selector 호출 | `BaselinePatchSelector(frame_paths, question, gaze, hl, hr)` → `priority_target` (T_vlm, 64) on 8×8 grid |
| ⑤ **Pair aggregation** (v2, 기본 `mean`) | `priority.view(T_vlm/2, 2, 64).mean(dim=1)` → (T_merged, 64). Qwen 의 `temporal_patch_size=2` 와 정렬 |
| ⑥ Top-10% per (pair or frame) | `priority.topk(k=7)` (ceil(0.10 × 64)) → (T_merged, 8, 8) bool |
| ⑦ 저장 | torch.save → `.pt` (이미 있으면 skip, resume 가능) |

### Selector 설정

```python
BaselinePatchSelectorConfig(
    scales="224", patch_size=28,       # 224/28 = 8 → 8×8 = 64 patches per frame (Qwen post-merge 1:1)
    selection_ratio=0.10,               # keep 10%
    selection_mode="per_frame",
    alpha_sim=0.5, alpha_pos=0.5,       # naive_patch_selector.md 기본값
    sigma_gaze=32/224, sigma_hand=40/224,
    eps_bg=0.01,
    device="cuda:0",
    batch_size=64,
)
```

---

## 4. Phase 2 — Training pipeline (`train_naive_tokendrop_lora.py`)

### 4.1 Vision wrapper — `QwenVisionWithMask` (post-merger slicing)

Qwen 원본 vision tower 의 forward 를 **그대로 호출**한 뒤, 반환된 post-merge embed sequence 에 `keep_mask_flat` 을 적용해 선택된 token 만 남긴다. ViT 내부 (cu_seqlens, window_index, RoPE) 는 건드리지 않음 → transformers 버전 drift 에 robust.

```python
class QwenVisionWithMask(nn.Module):
    def __init__(self, base_visual):
        self.base = base_visual   # FROZEN, original Qwen vision tower

    def forward(self, pixel_values_videos, grid_thw, keep_mask_flat):
        # 1) Run full ViT as-is (patch_embed → blocks → VisionPatchMerger)
        full_embeds = self.base(pixel_values_videos, grid_thw)   # (T_merged*64, d_llm)
        # 2) Slice kept tokens (natural spatial-temporal order)
        return full_embeds[keep_mask_flat]                       # (N_kept, d_llm)
```

특성:
- ViT FLOPs 절감은 없음 (full forward). 본 baseline 의 목적은 LLM 컨텍스트 budget 축소 효과 측정이므로 ViT compute 절감은 범위 밖.
- **Mask 정렬**: Qwen output 이 자연 spatial-temporal 순 `[frame0 r0c0 … r7c7, frame1 r0c0 …]` 이므로 precompute 한 `(T_merged, 8, 8)` mask 를 `reshape(-1)` 하면 그대로 정렬됨.
- **Temporal alignment (`make_keep_mask_flat`)**: 저장된 `(T_vlm, 8, 8)` mask 를 `(T_merged, 8, 8)` → flat `(T_merged*64,)` 로 변환. v2 (pair-mean) 에선 이미 T_vlm = T_merged 이라 pass-through, v1 (none) 에선 인접 pair OR/AND 병합 (`--temporal-mode`).

### 4.2 학습 루프 단위 흐름

```
Item = StreamGazeNaiveTokenDropDataset[i]
  ├─ vlm_frame_paths (128)
  ├─ keep_mask_8x8 (.pt 에서 로드, (T_merged=64 또는 T_vlm=128, 8, 8))
  └─ question, options, answer
        │
        ▼
preprocess_tokens_only(processor, base_qwen, ...)
  └─ processor(...) → input_ids, attention_mask, pixel_values_videos, grid_thw
     + base_qwen.get_rope_index → position_ids, rope_deltas
     (NO vision forward here)
        │
        ▼
run_vision_with_mask  (vision_with_mask + make_keep_mask_flat)
  ├─ make_keep_mask_flat(keep_mask_8x8, grid_thw, temporal_mode)
  │    → keep_mask_flat (T_merged*64,) bool
  ├─ vision_with_mask(pv_vid, grid_thw, keep_mask_flat)
  │    → kept_video_embeds (N_kept, d_llm)
        │
        ▼
build_tokendrop_inputs(base_qwen, tokenized, kept, keep_flat)
  ├─ drop un-kept <video> positions from input_ids / attention_mask / position_ids
  ├─ new_inputs_embeds = embed(new_input_ids)
  └─ splice kept_video_embeds into remaining <video> positions
        │
        ▼
forward_logits(peft_model, inputs)       # LLM LoRA only (vision frozen)
        │
        ▼
loss = CE(option_logits, gt_idx) / grad_accum
  → backward → single-group optimizer step (llm_params)
```

### 4.3 Temporal 정렬

- Qwen 내부: `temporal_patch_size=2` 로 128 VLM frame → **64 temporal tokens** (T_merged)
- **v2 기본 (`--temporal-agg mean`)**: Phase 1 에서 priority 를 인접 pair 평균으로 aggregate 한 뒤 top-K 선택 → 저장 mask shape `(64, 8, 8)`. 학습 시 `make_keep_mask_flat` 의 `T_vlm == T_merged` 분기가 바로 통과 → **budget 정확히 10.9%** (= 448/4096).
- **v1 레거시 (`--temporal-agg none`)**: VLM frame 단위 독립 top-K → mask `(128, 8, 8)` 저장 → 학습 시 인접 pair mask 를 OR 로 결합 → **budget ~19%** (`1-(1-p)² ≈ 0.19`). 비교 실험 전용.
- 두 방식 모두 Qwen 2.5-VL 의 원본 `temporal_patch_size=2` 동작 (Conv3d temporal kernel=2) 과 호환. pair-mean 은 Qwen 이 내부적으로 2 프레임을 하나로 압축하는 것과 같은 "pair = 한 단위" 관점.

### 4.4 Trainable Surface

| 모듈 | 학습 여부 | 설정 |
| --- | --- | --- |
| Vision `patch_embed`, `blocks`, `merger` | **FROZEN** | — (token drop 은 post-merger slicing 이므로 ViT 구조 변경 없음) |
| LLM (Qwen2.5) attention | **LoRA** | `r=16, α=32, dropout=0.05, target=["q_proj","k_proj","v_proj","o_proj"]` |
| 나머지 (embedding, MLP, layer norm, lm_head) | FROZEN | |

- Trainable params: **10,092,544** (0.12% of 8.3B total) — `load_qwen_dual_lora` 로그로 확인 가능
- Optimizer: AdamW, `lr_llm=1e-4`, `wd=1e-4`, `grad_accum=4`, `clip=1.0`
- DDP: `find_unused_parameters=True`
- Loss: **CE only** (teacher / KL 없음)
- TrajGazeMerge 의 `baseline_lora` 와 정확히 동일한 LoRA 구성 (선택 방식만 다름: 우리는 10% token drop, baseline 은 100% tokens).

---

## 5. 파일 레이아웃

### 신규
```
TrajGazeMerge/
├── data/
│   ├── precompute_naive_masks.py        ← Phase 1 CLI (+ pair-mean aggregation)
│   └── naive_dataset.py                 ← StreamGazeNaiveTokenDropDataset
├── models/
│   ├── vision_with_mask.py              ← QwenVisionWithMask, make_keep_mask_flat
│   └── model_tokendrop.py               ← load_qwen_dual_lora (LLM LoRA only),
│                                          preprocess_tokens_only,
│                                          build_tokendrop_inputs, forward_logits
├── training/
│   └── train_naive_tokendrop_lora.py    ← Phase 2 학습 엔트리
└── scripts/
    └── run_naive_baseline.sh            ← Phase 1 + Phase 2 wrapper (background)
```

### 수정
- `TrajGazeMerge/data/dataset.py` — `/workspace/datasets/` → `/home/yujin/dataset/` (5개 상수)
- `TrajGazeMerge/IMPLEMENTATION.md` — "Task 5: Naive TokenDrop (LLM LoRA only)" 섹션 추가

### 출력
```
/home/yujin/dataset/StreamGaze_v2/naive_masks_8x8_v2_pair_mean/{ds}/{stem}/{qhash}.pt
/home/yujin/gaze/trajgaze/TrajGazeMerge/checkpoints/naive_tokendrop_lora_v2_pair_mean/
    ├── train_log_rank0.jsonl
    ├── best.pth            (state_dict: LLM LoRA only, + epoch + step + acc)
    └── epoch_XX.pth
/home/yujin/gaze/trajgaze/TrajGazeMerge/logs_v2/
    ├── phase1_precompute.log
    ├── phase2_train.log
    └── run_status.log
```

---

## 6. 실행 커맨드

모든 커맨드는 세션 당 1회 아래 환경변수 export 가 전제입니다 (§1 참조):

```bash
export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8
export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

### 6.1 Phase 1 — Offline mask precompute

**CLI 인자** (`TrajGazeMerge/data/precompute_naive_masks.py`):

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--output-dir` | (필수) | `.pt` 저장 루트 |
| `--data-root` | `/home/yujin/dataset/StreamGaze_v2` | 정보용 (실제 경로는 dataset.py 상수에서 읽음) |
| `--n-vlm-frames` | `128` | QA 당 샘플할 VLM 프레임 수 |
| `--scales` | `"224"` | Selector scales 문자열 |
| `--patch-size` | `28` | `224/28=8` → 8×8 grid 직접 출력 |
| `--ratio` | `0.10` | top-K 비율 (per pair in v2 / per frame in v1) |
| `--temporal-agg` | `mean` | `mean`: pair-mean priority → top-K per pair → `(T_merged, 8, 8)` 정확히 10%.  `max`: pair-max priority. `none`: v1 레거시 (per-frame top-K → `(T_vlm, 8, 8)`, 학습 시 OR → ~19%) |
| `--alpha-sim` / `--alpha-pos` | `0.5` / `0.5` | Talk2DINO sim vs Gaussian prior 가중치 |
| `--sigma-gaze` / `--sigma-hand` | `32/224` / `40/224` | Gaussian σ (정규화 좌표) |
| `--eps-bg` | `0.01` | 배경 보존 상수 |
| `--device` | `cuda:0` | 단일 GPU 사용 |
| `--batch-size` | `64` | DINOv2 forward 배치 |
| `--splits` | `both` | `train` / `test` / `both` |
| `--tasks` | (전체 8 tasks) | 콤마 분리 task 이름 |
| `--max-items` | `0` | `>0` 이면 smoke test (조기 종료) |
| `--overwrite` | `false` | 기본은 기존 `.pt` skip (resume-safe) |

**Smoke (5 items, ~30초)**:
```bash
/opt/conda/envs/trajgaze/bin/python -m TrajGazeMerge.data.precompute_naive_masks \
    --output-dir /tmp/naive_masks_smoke \
    --n-vlm-frames 128 --scales 224 --patch-size 28 \
    --ratio 0.10 --alpha-sim 0.5 --alpha-pos 0.5 --temporal-agg mean \
    --device cuda:0 --batch-size 64 \
    --splits test --max-items 5
```

**Test split 만 (526 QA, 대략 5~15분)**:
```bash
/opt/conda/envs/trajgaze/bin/python -m TrajGazeMerge.data.precompute_naive_masks \
    --output-dir /home/yujin/dataset/StreamGaze_v2/naive_masks_8x8 \
    --n-vlm-frames 128 --scales 224 --patch-size 28 \
    --ratio 0.10 --alpha-sim 0.5 --alpha-pos 0.5 --temporal-agg mean \
    --device cuda:0 --batch-size 64 --splits test
```

**전체 (train+test = 5,799 + 526 = 6,325 QA, 수십 분~수 시간)**:
```bash
/opt/conda/envs/trajgaze/bin/python -m TrajGazeMerge.data.precompute_naive_masks \
    --output-dir /home/yujin/dataset/StreamGaze_v2/naive_masks_8x8 \
    --n-vlm-frames 128 --scales 224 --patch-size 28 \
    --ratio 0.10 --alpha-sim 0.5 --alpha-pos 0.5 --temporal-agg mean \
    --device cuda:0 --batch-size 64 --splits both
```

병렬 실행 팁: dataset 단위로 나눠 GPU 를 분산할 수 있음 (예: GPU 0 은 `--splits train`, GPU 1 은 `--splits test` 또는 `--tasks` 로 쪼개기).

### 6.2 Phase 2 — Training

**CLI 인자** (`TrajGazeMerge/training/train_naive_tokendrop_lora.py`):

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--mask-dir` | (필수) | Phase 1 출력 루트 |
| `--output-dir` | (필수) | 체크포인트 / 로그 저장 경로 |
| `--epochs` | `3` | |
| `--lr-vision` | `5e-5` | **(ignored)** — vision frozen, backward-compat 용 인자 |
| `--lr-llm` | `1e-4` | LLM LoRA learning rate |
| `--grad-accum` | `4` | gradient accumulation |
| `--grad-clip` | `1.0` | |
| `--log-every` | `20` | micro-step 단위 |
| `--eval-every` | `200` | `0` 이면 비활성화 |
| `--eval-max-items` | `100` | 주기 eval 아이템 수. 200 은 NCCL 10min watchdog 위험 → 100 로 안전 (~3-4분) |
| `--n-frames` | `128` | VLM 입력 프레임 수 (precompute 와 일치해야 함) |
| `--temporal-mode` | `or` | v1 (`--temporal-agg none`) mask 용 128→64 병합. v2 (pair-mean) 에선 무시됨 (T_vlm==T_merged) |
| `--resume-ckpt` | `None` | `best.pth` 경로. LoRA state 복구 (optimizer state 는 복구 안함) |
| `--max-train-steps` | `0` | `>0` 이면 smoke (rank 당) |
| `--single-gpu` | `false` | DDP 우회 (smoke) |

**Smoke (2 steps, 1 GPU, ~1분)**:
```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/trajgaze/bin/python \
    -m TrajGazeMerge.training.train_naive_tokendrop_lora \
    --mask-dir /tmp/naive_masks_smoke \
    --output-dir /tmp/naive_td_smoke \
    --epochs 1 --lr-llm 1e-4 --grad-accum 1 \
    --log-every 1 --eval-every 0 --max-train-steps 2 \
    --single-gpu
```

기대 로그:
```
[load_qwen_dual_lora] trainable params: 10,092,544 / 8,292,166,976 (0.122%) — LLM LoRA only, vision frozen
[params] llm_lora=10,092,544 (vision frozen)
Epoch 1 | step 1/... | loss=0.85 | kept=10.9% | t=30s
Epoch 1 | step 2/... | loss=1.11 | kept=10.9% | t=34s
```

**Full train (3 epochs, 2 GPUs DDP, wrapper 권장)**:

권장: `TrajGazeMerge/scripts/run_naive_baseline.sh` 사용 (Phase 1+2 순차 실행, `best.pth` 존재 시 자동 `--resume-ckpt`).
```bash
nohup setsid bash /home/yujin/gaze/trajgaze/TrajGazeMerge/scripts/run_naive_baseline.sh \
    > /dev/null 2>&1 < /dev/null &
```

직접 실행하려면:
```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/trajgaze/bin/torchrun \
    --nproc_per_node=2 --master_port=29503 \
    -m TrajGazeMerge.training.train_naive_tokendrop_lora \
    --mask-dir /home/yujin/dataset/StreamGaze_v2/naive_masks_8x8_v2_pair_mean \
    --output-dir /home/yujin/gaze/trajgaze/TrajGazeMerge/checkpoints/naive_tokendrop_lora_v2_pair_mean \
    --epochs 3 --lr-llm 1e-4 --grad-accum 4 \
    --log-every 20 --eval-every 200 --eval-max-items 100 \
    --temporal-mode or \
    2>&1 | tee /home/yujin/gaze/trajgaze/TrajGazeMerge/logs_v2/phase2_train.log
```

**체크포인트 구조**:
```
{output-dir}/
├── train_log_rank0.jsonl          ← micro-step 단위 loss/kept
├── best.pth                        ← {epoch, step, state_dict, acc}
├── epoch_01.pth
├── epoch_02.pth
└── epoch_03.pth
```

`state_dict` 는 PEFT-wrapped 모델 전체 state — 실질 trainable 은 **LLM LoRA 만** (vision 은 frozen 이라 adapter 없음). 재개/eval 시 `load_qwen_dual_lora` → `peft_model.load_state_dict(ckpt["state_dict"], strict=False)` 로 복구 (training script 의 `--resume-ckpt` 가 자동 처리).

### 6.3 빠른 end-to-end 검증 시퀀스 (권장)

```bash
# 1) test split mask (빠름)
/opt/conda/envs/trajgaze/bin/python -m TrajGazeMerge.data.precompute_naive_masks \
    --output-dir /home/yujin/dataset/StreamGaze_v2/naive_masks_8x8 \
    --splits test

# 2) train split 일부 (smoke 학습을 위해 약 50개)
/opt/conda/envs/trajgaze/bin/python -m TrajGazeMerge.data.precompute_naive_masks \
    --output-dir /home/yujin/dataset/StreamGaze_v2/naive_masks_8x8 \
    --splits train --max-items 50

# 3) smoke 학습 (10 step, 1 GPU)
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/trajgaze/bin/python \
    -m TrajGazeMerge.training.train_naive_tokendrop_lora \
    --mask-dir /home/yujin/dataset/StreamGaze_v2/naive_masks_8x8 \
    --output-dir /tmp/naive_td_check \
    --epochs 1 --grad-accum 1 --log-every 1 --eval-every 0 \
    --max-train-steps 10 --single-gpu

# 4) 검증 통과 시 전체 precompute + DDP 학습으로 전환 (위 6.1 / 6.2)
```

---

## 7. Key decisions (why)

- **왜 8×8 post-merge grid?**  Qwen 의 `spatial_merge_size=2` 와 `temporal_patch_size=2` 를 거친 후 최종 video token 은 `T_merged × 8 × 8` 배열. Selector mask 도 같은 공간에서 만들면 `.pt` → 학습 loop 로의 인덱스 변환 오버헤드 없음.

- **왜 2×2 block 단위 drop?**  Qwen 의 `Qwen2_5_VisionPatchMerger.forward` 는 `hidden.view(-1, spatial_merge_unit, d)` 로 2×2 group 을 projection. 임의 token drop 은 이 구조를 깨뜨려 merger 결과가 의미 없게 됨. 2×2 group all-True/all-False 보장이 가장 간단한 해법.

- **왜 ViT 는 frozen 인가?** (v2 기본)  우리의 token drop 은 **post-merger slicing** (ViT 전체 forward 후 output 만 slice) 이므로 ViT 내부 연산·입력 형식은 원본 그대로 유지됨. ViT 의 학습 가능성은 marginal 이득만 주고 학습 부담만 커짐. Path B (AutoGaze) 도 **vision_tower FROZEN** 전략을 택했으며, 우리도 같은 방향. 사진 config (r=16, α=32, dropout=0.05, target=q/k/v/o_proj, LLM attention only) 와 정확히 일치.

- **왜 `patch_size=28` (selector)?**  Qwen post-merge 8×8 와 1:1 정렬. 대안 `patch_size=14 → 16×16 + adaptive_pool 8×8` 도 가능하지만 첫 구현에선 직접 8×8 생성이 단순.

- **왜 CE only, teacher 없음?**  TrajGazeMerge 의 KD 부분을 제거한 ablation — "rule-based selection 만으로 얼마나 가나" 를 측정.

- **왜 pixel mask 아님?**  이전 버전 plan 에서 검토했으나 사용자 요구 "선택된 token 만 ViT 에 넣기" 에 맞지 않음. ViT 는 여전히 전체 4096 token 을 forward 하므로 개입 지점이 불분명.

- **왜 pair-mean priority aggregation (v2)?**  v1 은 VLM frame 단위 독립 top-K 로 저장해 학습 loop 에서 128→64 해상도 mismatch 를 OR 로 해결했는데, 이 OR 이 두 frame 의 서로 다른 선택을 합쳐서 budget 을 $1-(1-0.1)^2 \approx 19\%$ 로 부풀림. v2 는 priority 를 먼저 인접 pair 평균 → 64 슬롯마다 독립 top-7 → 중복 카운트 없이 **정확히 10.9%**. TrajGazeMerge (r = 0.9·n_video = 3686, keep=410, 10%) 와 공정 비교 가능. Qwen Conv3d temporal kernel=2 가 2 frame 을 하나의 embedding 으로 압축하는 것과 같은 "pair = 한 단위" 관점과도 정합.

---

## 8. Gotchas

- **Post-merger slicing 의 전제**: Qwen2_5_VisionTransformer 는 내부적으로 window attention 을 위해 토큰을 reorder 했다가 merger 이후에 **자연 spatial-temporal 순으로 되돌려서 반환**함. 따라서 `(T_merged, 8, 8)` mask 를 그대로 `reshape(-1)` 해 `full_embeds[keep_mask_flat]` 이 정합. transformers 가 이 출력 순서를 바꾸면 이 baseline 도 깨짐 (4.51.3 기준 OK).
- **All-True mask sanity**: `keep_mask_flat.all()` 상태로 돌리면 `vision_with_mask.forward` 는 원본 `base(pv, grid_thw)` 와 bit-identical 해야 함 (단순 slice). 구현 회귀 탐지용.
- **Temporal mask 결합 (128 → 64)**: v2 (`--temporal-agg mean`, 기본) 에서는 Phase 1 단계에서 pair-mean 후 top-K → 저장 shape `(64, 8, 8)` → 학습 시 `T_vlm == T_merged` 분기로 pass-through → budget 정확히 10.9%. v1 (`--temporal-agg none`) 로 돌리면 per-frame top-K 저장 (128,8,8) → 학습 loop 의 `make_keep_mask_flat` 이 `--temporal-mode or|and` 에 따라 병합 → OR 은 budget ~19%, AND 은 ~5%.
- **DDP + LoRA**: `find_unused_parameters=True` 필수. LLM LoRA 만 trainable 이라 forward graph 에 unused param (vision tower, embed, lm_head) 가 많음.
- **NCCL watchdog**: 기본 timeout 10 분은 evaluate(max_items=200) 가 초과할 수 있음 → `setup_ddp` 에서 30분으로 확장. 추가로 `dist.barrier()` 를 eval 전후에 둬 rank-1 이 미리 다음 step 으로 진입해 BROADCAST 에 블록되는 상황을 방지. `--eval-max-items` 기본값 100 (~3–4분) 유지.
- **Disk I/O**: Phase 1 에서 DINOv2 forward 가 bottleneck. `batch_size=64` 로 고정. 수천 QA → 수십 분~수 시간.
- **Checkpoint 구조**: `best.pth = {epoch, step, state_dict (LLM LoRA included), acc}`. state_dict 는 PEFT-wrapped 전체지만 실질 non-zero adapter param 은 LLM 만 (vision frozen). `--resume-ckpt` 는 optimizer state 는 복구 안 하고 LoRA weights 만 `strict=False` 로 로드.
- **Session-detach**: wrapper script 는 `nohup setsid ... &` 로 실행해야 bash exit 시 SIGHUP/SIGTERM 으로 죽지 않음. 일반 `&` 만으로는 tty 종료 시 같이 사라짐.
- **Resume**: precompute 는 이미 존재하는 `.pt` skip. `run_naive_baseline.sh` 는 Phase 2 시작 시 `$CKPT_DIR/best.pth` 가 있으면 자동으로 `--resume-ckpt` 를 붙임.

---

## 9. Pre-processed artefacts

### Split 정의 (기존 `StreamGazeMergeDataset` 그대로)
- train = egoexolearn + holoassist (8 MCQ tasks 전수)
- test  = egtea (8 MCQ tasks 전수)
- QA 수는 `StreamGazeMergeDataset` 가 init 시 print 하는 수치로 확인.

### Precompute 저장 레이아웃 (Phase 1 출력)
- 경로: `{output-dir}/{dataset}/{stem}/{qhash}.pt`
- Resume: 이미 존재하는 `.pt` 는 skip (로그 `[skip]`)
- 실패 시: stderr 로 (qhash, reason) 출력, 계속 진행

---

## 10. Eval

- 구현: TBD (학습 완료 후 `train_naive_tokendrop_lora.py` 내 `evaluate()` 재사용 또는 `TrajGazeMerge/eval/evaluate.py` 에 `naive_tokendrop_lora` condition 추가)
- 평가 대상: egtea test split, 8 MCQ tasks
- 지표: MCQ accuracy (overall + task 별)

### 학습 후 비교 예정 조건

| 조건 | Selector | ViT | LLM | Loss |
| --- | --- | --- | --- | --- |
| Baseline LoRA | 없음 (100%) | frozen | LoRA | CE |
| TrajGazeMerge LoRA | TrajGaze encoder (trainable) | frozen | LoRA | α·KL + (1−α)·CE |
| AutoGaze LoRA | AutoGaze (frozen) | frozen | LoRA | CE |
| **Naive TokenDrop (본 플랜)** | Rule-based (offline `.pt`) | **LoRA (trainable)** | LoRA | CE |

---

## 11. 관련 문서 / 코드

| 종류 | 경로 |
| --- | --- |
| 상위 파이프라인 (TrajGazeMerge 본체) | `TrajGazeMerge/IMPLEMENTATION.md` |
| Selector (heuristic) | `TrajGaze/baselines/patch_selector.py` |
| Selector config | `TrajGaze/baselines/config.py` |
| Selector 상세 문서 | `docs/baselines/naive_patch_selector.md` |
| 구조 참조 (offline precompute 패턴) | `docs/baselines/path_b_autogaze_lora_finetune.md` |
| 기존 dataset (QA 인덱싱 재사용) | `TrajGazeMerge/data/dataset.py` |
| Qwen loader (공용) | `TrajGazeMerge/models/model.py` |
| AutoGaze 학습 루프 참조 | `TrajGazeMerge/training/train_autogaze_lora.py` |
| TrajGazeMerge 학습 루프 참조 | `TrajGazeMerge/training/train_merge_lora.py` |
