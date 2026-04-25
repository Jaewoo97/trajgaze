# Naive Patch Selector — gazing_info `.pt` 생성 파이프라인

Path B / Path C 학습에 투입되는 사용자 `.pt` (gazing_info) 를 생성하는 **heuristic baseline**. AutoGaze (learned) 를 대체해 **Talk2DINO 질문-패치 유사도 + gaze/hand Gaussian prior** 를 priority score 로 결합하고, `F.adaptive_avg_pool2d` 로 SigLIP 그리드에 정렬한 뒤 top-K% 패치를 선택한다. 출력 스키마가 AutoGaze 와 동일하므로 downstream (SigLIP → NVILA) 에 그대로 투입된다.

```
 JPG frames (T)                    question (str)          gaze/hand (T, *)
      │                                 │                         │
      ▼                                 ▼                         ▼
 DINOv2 ViT-L/14              Talk2DINO text enc           Gaussian map
 (T, 1369, 1024)                   (1, 1024)             w(t,p), L1-norm
      │                                 │                         │
      └────────────┬───── cos sim ──────┘                         │
                   ▼                                              ▼
               sim(t,p) ────────────────────── priority = α_sim·norm(sim) + α_pos·norm(w)
                                                      │
                                                      ▼
                                 F.adaptive_avg_pool2d (37×37 → scales grid)
                                                      │
                                                      ▼
                                     top-K% (per_frame or global) + pad-to-max
                                                      │
                                                      ▼
                                  gazing_info = {gazing_pos, num_gazing_each_frame,
                                                 if_padded_gazing, gazing_mask, scales,
                                                 num_vision_tokens_each_frame, ...}
                                                      │
                                                      ▼
                                    .pt file keyed by md5(question)[:8]
```

전체 아키텍처·원본 다이어그램: [`TrajGaze/BASELINE_PATCH_SELECTOR.md`](../../TrajGaze/BASELINE_PATCH_SELECTOR.md) (본 문서의 상위 레퍼런스).

---

## 1. Environment

- Conda env: `trajgaze` (`/opt/conda/envs/trajgaze`), Python 3.10.
- 핵심 패키지: `torch (bf16)`, `torchvision`, `transformers`, `Talk2DINO` (로컬 레포 `Talk2DINO/`), `huggingface_hub`.
- Talk2DINO 체크포인트 `lorebianchi98/Talk2DINO-ViTL` 는 첫 실행 시 HF hub 에서 자동 캐싱 (DINOv2 ViT-L/14 + CLIP→DINO text projection).
- **세션마다 export**:
  ```bash
  export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8      # torchvision libjpeg 충돌
  export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib              # CXXABI_1.3.15 (cv2)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True          # 프래그 완화
  ```
- 하드웨어: 단일 GPU (default `cuda:0`) 로 충분. DINOv2 forward 가 bottleneck — `--batch_size 64` 로 설정됨.

---

## 2. Data assets

### 입력 (read-only)

| 용도 | 경로 |
| --- | --- |
| Pre-extract JPG (selector 가 실제로 읽는 소스) | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/{dataset}/no_gaze/{video_id}/{clip}_{frame}.jpg` |
| 프레임별 gaze 좌표 | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/{dataset}/gaze_mapping/{video_id}/{clip}_mapping.csv` |
| QA 메타 (1750 rows) | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/metadata.csv` |

Hand 좌표는 `gaze_mapping` CSV 내 칼럼에서 함께 파싱 (없으면 해당 항 Gaussian 생략).

### 출력

```
results/baseline/<run_name>/
├── config.json                       ← 사용된 BaselinePatchSelectorConfig dump
├── stats.json                        ← 처리 통계 (QA 수, 실패 수, per-dataset 분포 등)
├── gazing_info/
│   └── {dataset}/{video_id}/{md5(question)[:8]}.pt
└── vis/                              ← --save_vis 시에만
    └── {dataset}/{video_id}/{qa_hash}_frame*.png, {qa_hash}.gif
```

`.pt` 스키마 (AutoGaze 호환, CLAUDE.md §2 와 동일):

```python
{
    "gazing_pos":                   (1, N) int64,   # global patch indices across whole video
    "num_gazing_each_frame":        (T,) int64,     # uniform = max selection across frames
    "if_padded_gazing":             (1, N) bool,    # True at padded slots
    "gazing_mask":                  list[(1, T, n_scale)] bool,   # per-scale binary masks
    "scales":                       list[int],
    "num_vision_tokens_each_frame": int,
    "frame_sampling_rate":          int,             # 1 (no temporal subsample at this stage)
}
```

Global index: `frame_idx * num_vision_tokens_each_frame + local_patch_idx`.

---

## 3. Pipeline internals

| 단계 | 위치 | 요지 |
| --- | --- | --- |
| ① DINOv2 patch feature | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | 각 프레임 518px resize → Talk2DINO.`forward_features` → (T, 1369, 1024), 37×37 grid. |
| ② Text embedding | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | Talk2DINO text encoder → (1, 1024). |
| ③ Per-patch similarity | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | `cos(patch, text)` → (T, 1369). |
| ④ Gaze/hand Gaussian | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | `w(p) = ε_bg + Σ exp(-‖c(p) − μ‖²/(2σ²))` for μ ∈ {gaze, hand_L, hand_R}, L1 normalize per frame. gaze/hand 모두 부재 시 uniform fallback. |
| ⑤ Priority | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | `priority = α_sim · minmax(sim) + α_pos · minmax(w)`, per-frame `[0,1]` 로 정규화. |
| ⑥ Grid resample | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | 각 scale s 에 대해 `F.adaptive_avg_pool2d(37→s/patch_size)` 로 priority 맵 축소. Multi-scale 은 각 스케일 독립 downsample 후 concat. |
| ⑦ Selection | [`patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) | `per_frame`: 프레임별 top ⌈N·r⌉. `global`: 전체 top ⌈T·N·r⌉ + `min_patches_per_frame` 가드. |
| ⑧ gazing_info 빌드 | [`gazing_info_builder.py`](../../TrajGaze/baselines/gazing_info_builder.py) | frame 간 선택 수 정렬 → max 로 pad, `if_padded_gazing` 마킹, per-scale `gazing_mask` 생성, global index 변환. |
| Driver | [`run_baseline.py`](../../TrajGaze/baselines/run_baseline.py) | `metadata.csv` 스캔 → 각 QA 당 JPG 로딩 + Talk2DINO inference + `.pt` 저장 (+ 선택적 시각화). |
| Config | [`config.py`](../../TrajGaze/baselines/config.py) | `BaselinePatchSelectorConfig` dataclass, `__post_init__` 에서 `num_vision_tokens_each_frame = Σ (s/patch_size)²` 계산. |

---

## 4. Selection mode

- **`per_frame`**: 각 프레임에서 독립적으로 top-K%. 프레임별 패치 수 동일 → pad 불필요. 모든 프레임 표현 보장. **현재 `nvila_perframe/` 는 이 모드.**
- **`global`**: 전체 프레임 priority 를 pool 한 뒤 top-K%. 정보량 높은 프레임에 패치 예산 집중, pad 필요. `min_patches_per_frame` 로 빈 프레임 방지.

---

## 5. Key parameters — `nvila_perframe/` 실제 세팅

(CLAUDE.md §2 의 Path B/C 학습 입력 `nvila_perframe/` 에 대응. 사본: [`results/baseline/nvila_perframe/config.json`](../../results/baseline/nvila_perframe/config.json).)

| 파라미터 | 기본값 (config.py) | **nvila_perframe 실제** | 메모 |
| --- | --- | --- | --- |
| `scales` | `"224"` | **`"56+112+224+448"`** | NVILA HD-Video scale 과 정합 |
| `patch_size` | `16` | **`14`** | SigLIP 실제 patch (CLI `--patch_size 14`) |
| `selection_ratio` | `0.10` | `0.10` | top 10% |
| `selection_mode` | `"per_frame"` | `"per_frame"` | |
| `min_patches_per_frame` | `1` | `1` | per_frame 모드에선 미사용 |
| `alpha_sim` / `alpha_pos` | `0.5` / `0.5` | `0.5` / `0.5` | 독립 weight (합=1 제약 없음) |
| `sigma_gaze` | `32/224 ≈ 0.1429` | 동일 | gaze Gaussian std |
| `sigma_hand` | `40/224 ≈ 0.1786` | 동일 | hand Gaussian std (넓음) |
| `eps_bg` | `0.01` | 동일 | 배경 보존 상수 |
| `dino_resize` / `dino_grid_size` / `dino_dim` | `518` / `37` / `1024` | 동일 | Talk2DINO ViT-L/14 고정 |
| `batch_size` | `64` | `64` | DINOv2 forward |

**NV 계산 확인** (`num_vision_tokens_each_frame`):
```
patches_per_scale = [(56/14)², (112/14)², (224/14)², (448/14)²]
                  = [16, 64, 256, 1024]
NV = 16 + 64 + 256 + 1024 = 1360   ← CLAUDE.md §2 의 `nvila_perframe` NV=1360 과 일치
```

> `nvila_perframe_single/` 의 경우: `scales="392"`, `patch14` → NV = (392/14)² = 784.

---

## 6. 파일 레이아웃

### 구현

```
TrajGaze/baselines/
├── __init__.py
├── config.py                 ← BaselinePatchSelectorConfig dataclass
├── patch_selector.py         ← DINOv2 + Gaussian + adaptive_avg_pool2d + top-K
├── gazing_info_builder.py    ← AutoGaze 호환 dict 빌더
├── run_baseline.py           ← CLI driver (python -m TrajGaze.baselines.run_baseline)
├── visualize_patches.py      ← 프레임별 PNG + 클립 GIF 오버레이
└── run_all_configs.sh        ← 4-config sweep 스크립트
```

### 출력

```
results/baseline/<run_name>/
├── config.json
├── stats.json
├── gazing_info/{dataset}/{video_id}/{qa_hash}.pt
└── vis/{dataset}/{video_id}/{qa_hash}_frame0000.png, {qa_hash}.gif   # --save_vis 시
```

---

## 7. 실행 커맨드

### 현재 `nvila_perframe/` 재생성 (Path B/C 학습 입력)

```bash
cd /home/yujin/gaze/trajgaze && \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/trajgaze/bin/python -m TrajGaze.baselines.run_baseline \
    --data_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
    --output_dir results/baseline/nvila_perframe \
    --scales 56+112+224+448 --patch_size 14 \
    --selection_ratio 0.10 --selection_mode per_frame \
    --alpha_sim 0.5 --alpha_pos 0.5 \
    --device cuda:0 --save_gazing_info
```

### 4-config sweep (per_frame × {single, multi} × {global, per_frame})

```bash
bash /home/yujin/gaze/trajgaze/TrajGaze/baselines/run_all_configs.sh
```

기본 설정: `scales ∈ {"224", "32+64+112+224"}`, `patch_size=16` (NVILA 와 다름 — 시각화용·ablation 전용). 결과는 `results/baseline/{perframe_single, perframe_multi, global_single, global_multi}/`.

### Quick smoke test (5 클립)

```bash
/opt/conda/envs/trajgaze/bin/python -m TrajGaze.baselines.run_baseline \
    --data_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
    --output_dir results/baseline_smoke \
    --max_clips 5 --save_vis --vis_clips_per_dataset 5
```

---

## 8. Gotchas

- **JPG 인덱스 공간 = selector 인덱스 공간**. JPG 정렬 key 는 `filename.rsplit("_", 1)[-1]` 정수. 소비 측 (Path B/C `train_*.py`) 이 동일 정렬로 JPG 를 읽어야 `.pt` 의 global index 가 유효. mp4 pyav 디코드는 프레임 수가 미묘히 달라 불일치.
- **`.pt` 생성 후 JPG 재추출 금지**. 재추출 시 frame 수 불일치로 `.pt` 가 무효. Path B/C 의 [`tools/build_sft_jsonl.py`](../../vila_hd_work/tools/build_sft_jsonl.py) 가 이 케이스를 필터 (현재 8 QA 제외, 1367/375 로 축소).
- **질문 없는 frame → uniform fallback** (`1/1369`). `alpha_sim=0` 으로 QA 기여를 0 으로 두면 gaze/hand-only ablation 가능.
- **DINOv2 37 → SigLIP target 그리드 해상도 미스매치**. 37 이 14 로 나누어 떨어지지 않아 `adaptive_avg_pool2d` 가 non-uniform 영역 평균을 냄 (약간의 공간 부정확성). 본 baseline 한계.
- **Fixed selection ratio**. AutoGaze 는 task loss 요구치로 적응적 중단, 본 selector 는 top 10% 고정.
- **Sweep 비용**: 1750 QA × Talk2DINO forward 가 지배적. 단일 config 당 수십 분, 4-config sweep 은 수 시간.

---

## 9. 관련 문서 / 코드

| 종류 | 경로 |
| --- | --- |
| 원문 (아키텍처 완전판) | [`TrajGaze/BASELINE_PATCH_SELECTOR.md`](../../TrajGaze/BASELINE_PATCH_SELECTOR.md) |
| 소비 측 — Path B | [`path_b_autogaze_lora_finetune.md`](path_b_autogaze_lora_finetune.md) |
| 소비 측 — Path C | [`path_c_nvila_lite_8b_lora_finetune.md`](path_c_nvila_lite_8b_lora_finetune.md) |
| Config dataclass | [`TrajGaze/baselines/config.py`](../../TrajGaze/baselines/config.py) |
| 선택 로직 | [`TrajGaze/baselines/patch_selector.py`](../../TrajGaze/baselines/patch_selector.py) |
| gazing_info 빌더 | [`TrajGaze/baselines/gazing_info_builder.py`](../../TrajGaze/baselines/gazing_info_builder.py) |
| CLI driver | [`TrajGaze/baselines/run_baseline.py`](../../TrajGaze/baselines/run_baseline.py) |
| 시각화 | [`TrajGaze/baselines/visualize_patches.py`](../../TrajGaze/baselines/visualize_patches.py) |
| 4-config sweep | [`TrajGaze/baselines/run_all_configs.sh`](../../TrajGaze/baselines/run_all_configs.sh) |
| 참고 (Talk2DINO 원 스크립트) | `scripts/compute_talk2dino_similarity.py` |

> 본 문서는 데이터 생성 파이프라인 전용 — 모델 accuracy 지표는 기록하지 않는다. 각 run 의 처리 통계는 `results/baseline/<run>/stats.json` 을 참조.
