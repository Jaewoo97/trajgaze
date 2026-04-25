# Path B — NVILA-8B-HD-Video LoRA SFT on EgoGazeVQA (AutoGaze LoRA finetune)

NVILA-8B-HD-Video 를 사용자 멀티스케일 gazing_info (`.pt`) 로 LoRA SFT 하는 파이프라인. **HF processor 를 완전히 우회** 하여 사용자 데이터의 1-track 인덱스 공간을 그대로 소비한다.

```
 JPG frames (trajgaze ordering)         user .pt (gazing_info)
     │                                        │
     ▼                                        ▼
 AutoGazeImageProcessor                slice_gazing_info (K-frame remap)
     │                                        │
     └──────────────┬─────────────────────────┘
                    ▼
     vision_tower  (scales = "56+112+224+448", frozen, bf16)
                    │   hidden_states[-2]
                    ▼
     per-frame unpad (~if_padded_gazing) → pad to multiple of 9
                    ▼
     mm_projector  (full-tune)   ──►  (num_video_tokens, llm_hidden)
                    │
                    ▼
     splice into <vila/video> positions in input_ids embedding
                    ▼
     Qwen2 LLM  (LoRA, r=32, α=64, dropout=0.05)
                    ▼
     cross-entropy on assistant letter only
```

---

## 1. Environment

- Conda env: `trajgaze` (clone of `gaze`), Python 3.10.
- Python 핵심 패키지: `transformers`, `peft==0.19.1`, `autogaze` (local repo), `wandb==0.26.0` (already logged in as `yujinbae` via `~/.netrc`). TensorBoard 는 optional.
- **세션마다 반드시 export 할 환경 변수**:
  ```bash
  export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8      # torchvision 번들 libjpeg 충돌 회피
  export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib              # CXXABI_1.3.15 (cv2) 해결
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True          # 메모리 프래그먼테이션 완화
  ```
- 하드웨어: 2×H200 기준 (bf16, `attn_implementation="sdpa"`).

---

## 2. Data assets (read-only)

| 용도 | 경로 |
| --- | --- |
| 원본 QA (1750행) | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/metadata.csv` |
| 원본 클립 (mp4, 학습에는 안 씀) | `/home/yujin/dataset/EgoGazeVQA/{dataset}/{video_id}/{start}_{end}.mp4` |
| **Pre-extract JPG** (학습 로더가 실제로 읽는 소스) | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/{dataset}/no_gaze/{video_id}/{clip}_{frame}.jpg` |
| 멀티스케일 gazing_info | `/home/yujin/gaze/trajgaze/results/baseline/nvila_perframe/gazing_info/{dataset}/{video_id}/{md5(question)[:8]}.pt` |

gazing_info spec: scales `56+112+224+448`, patch14 → 1360 tok/frame, key `md5(question)[:8]`. 1750 QA 전수 매칭 (단, JPG 프레임 수 불일치 8개 빌드 단계에서 제외 → 1367/375).

---

## 3. Pre-processed artefacts (재생성 가능)

### 8:2 group split (video_id 단위, seed 42)

```bash
/opt/conda/envs/trajgaze/bin/python tools/split_egogaze.py
# → vila_hd_work/splits.json
# 1373 train (210 video_ids) / 377 test (53 video_ids)
# ※ 프레임 수 불일치로 학습용 jsonl 에서는 1367 / 375 로 줄어듦 (build 단계 필터).
```

### SFT JSONL

```bash
/opt/conda/envs/trajgaze/bin/python tools/build_sft_jsonl.py
# → vila_hd_work/data/{train,test}.jsonl  (1367 / 375)
# 엔트리 필드: qa_idx, video_path, gazing_info_path, dataset, video_id,
#            qa_type, question, options, correct_letter
# 필터 기준: (1) video_path + gi_path 존재, (2) .pt 의 frame 수 == 현재 JPG 수
# JPG 가 .pt 생성 이후 재추출된 8개 QA 는 여기서 제외.
```

---

## 4. Pipeline internals (`train_path_b.py`)

| 단계 | 함수 | 요지 |
| --- | --- | --- |
| Frame subsample | `pick_frame_indices(T_full, K)` | linspace K개. `T_full <= K` 면 마지막 프레임 repeat-pad. |
| gazing_info remap | `slice_gazing_info(gi, indices, NV=1360)` | `old_global = old_idx*NV + local` → `new_global = new_idx*NV + local` |
| Video token 개수 | `count_video_tokens(gi)` | `Σ ceil(non_pad_frame_i / 9)` (9 = NVILAMultiModalProjector TokenShuffle) |
| Vision forward | `encode_video_tokens()` | `vision_tower(..., output_hidden_states=True).hidden_states[-2]` → split by `num_gazing_each_frame` → unpad via `~if_padded_gazing` → pad-to-9 → `mm_projector` |
| LLM splice | `forward_loss()` | tokenizer 한 곳의 `<vila/video>` 를 N 회 복제한 ids → embedding 에서 해당 위치를 projected feature 로 치환 → `llm(inputs_embeds=..., labels=...)` |
| Loss mask | `_build_prompt` | `apply_chat_template` 로 user turn prefix 구해 prefix 길이 이하 labels 를 `-100`. assistant letter 만 학습 대상. |

### Trainable surface

- `vision_tower` — FROZEN.
- `mm_projector` — `requires_grad=True` (full-tune).
- `llm` — `peft.LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target=[q,k,v,o,gate,up,down]_proj, task=CAUSAL_LM)`.

---

## 5. Key decisions (why)

- **Processor bypass (옵션 B1)**: HD-Video `processing_nvila._get_gazing_info_from_videos` 는 tile+thumbnail 두 track 을 돌려 AutoGaze 를 호출 → 사용자 1-track `.pt` 인덱스 공간과 정합 불가.
- **`scales = "56+112+224+448"` override**: HD-Video 기본 `"56+112+196+392"` 와 사용자 데이터 scale 이 다름. SigLIP positional embedding 은 `image_size=448` 기준 `interpolate_pos_encoding` 으로 bilinear 보간 → 임의 scale 에서도 forward 가능. 상세 근거 → [`vila_hd_work/notes/multiscale_wiring.md`](../../vila_hd_work/notes/multiscale_wiring.md).
- **JPG loader (mp4 아님)**: 사용자 `.pt` 는 trajgaze `get_clip_frames()` 가 pre-extract 해둔 JPG 순서로 계산됨. 정렬 key 는 파일명 끝 숫자 (`filename.rsplit("_", 1)[-1]`). mp4 pyav 디코드와 frame count 가 미묘하게 달라 인덱스 공간이 어긋난다.
- **K=32 frames**: block-causal mask 메모리 O(N²). 2×H200 bf16 기준 K=32 안전, K=64 위험, K=128 OOM 확정.
- **`attn_implementation="sdpa"`**: flash-attn 2 는 "only supports causal or bidirectional" assertion 으로 block-causal 마스크와 충돌.

---

## 6. 파일 레이아웃

```
vila_hd_work/
  train_path_b.py              ← 학습 엔트리
  eval_egogaze.py              ← native / precomputed eval
  splits.json                  ← 8:2 group split
  data/{train,test}.jsonl
  notes/multiscale_wiring.md   ← scales override 근거
  out/baseline_test.jsonl      ← native baseline 결과
  tools/
    split_egogaze.py
    build_sft_jsonl.py
  runs/<run_name>/
    step_{500,1000,...}/{llm_lora/, mm_projector.pt}
    final/{llm_lora/, mm_projector.pt}
    train.log                  ← tee 로그
    tb/                        ← --logger tensorboard 사용 시
```

---

## 7. 실행 커맨드

### Smoke test — 파이프라인 무결성 검증 (10 samples × 200 micro-step, ≈ 2.5 분)

```bash
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python train_path_b.py \
  --overfit-n 10 --max-steps 200 --grad-accum 4 \
  --num-frames 32 --log-every 5 --save-every 0 \
  --output-dir runs/smoke
```

기대: loss 1.2 → 0 근처로 수렴, 0.79 s/step 안정.

### Full run — 2 epoch, wandb 로깅 (≈ 40–50 분)

```bash
mkdir -p /home/yujin/gaze/trajgaze/vila_hd_work/runs/path_b_full && \
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python train_path_b.py \
  --max-steps 2734 --grad-accum 4 --num-frames 32 \
  --log-every 5 --save-every 500 \
  --output-dir runs/path_b_full \
  --logger wandb --wandb-project trajgaze-path-b \
  --run-name path_b_full \
  2>&1 | tee runs/path_b_full/train.log
```

- 2734 micro-step × grad_accum 4 = **684 updater step** (≈ 2 epoch × 1367 train).
- Checkpoint at micro-step 500 / 1000 / 1500 / 2000 / 2500 + `final/`.
- wandb 대시보드: `trajgaze-path-b / path_b_full`.

### TensorBoard 대안

```bash
/opt/conda/envs/trajgaze/bin/pip install tensorboard
# ...train_path_b.py 커맨드에 --logger tensorboard 추가...
tensorboard --logdir /home/yujin/gaze/trajgaze/vila_hd_work/runs/path_b_full/tb --port 6006
```

---

## 8. 로깅 지표 (updater step 기준)

| 이름 | 설명 |
| --- | --- |
| `train/loss` | grad_accum 복원된 micro-batch 평균 loss |
| `train/lr` | cosine w/ warmup (warmup_ratio=0.03) |
| `train/grad_norm` | `clip_grad_norm_` 반환값 (clip 이전 norm) |
| `train/nv_tok` | per-sample 최종 video token 수 |

`--log-every 5` = updater step 5 마다 기록.

---

## 9. Eval (학습 결과 확인)

- `eval_egogaze.py --mode precomputed` 는 **TODO** 상태. train wrapper (`encode_video_tokens` + splice) 를 `generate()` 경로로 옮기는 작업이 남아있음.
- 임시 수동 경로:
  1. `PeftModel.from_pretrained(llm_base, runs/path_b_full/final/llm_lora)` 로 LLM LoRA 로드.
  2. `model.mm_projector.load_state_dict(torch.load(runs/path_b_full/final/mm_projector.pt))`.
  3. `train_path_b.encode_video_tokens` + splice 를 그대로 재사용, `llm.generate(inputs_embeds=..., max_new_tokens=8)`.
  4. 첫 `A-E` 파싱 → `eval_egogaze.ANSWER_RE` 재사용.

Baseline(native, 학습 전) 결과는 `out/baseline_test.jsonl` 에 저장되어 있으므로 학습 후 비교는 같은 test split 에서 수행.

### 결과 (test split 375 샘플)

출처: `vila_hd_work/out/path_b_test.jsonl` (eval 은 377행 전수, `data/test.jsonl` 의 qa_idx 로 375 필터링 — frame-count 미스매치 qa_idx 231/452 제외).

| 지표 | 값 |
| --- | --- |
| **Overall** | **0.5893** (221/375) |
| qa_type=causal | 0.8413 (106/126) |
| qa_type=spatial | 0.4160 (52/125) |
| qa_type=temporal | 0.5081 (63/124) |
| dataset=ego4d | 0.5373 (72/134) |
| dataset=egoexo | 0.6599 (97/147) |
| dataset=egtea | 0.5532 (52/94) |

참고: 필터 없이 raw 377행 기준 overall 은 0.5862 (221/377).

---

## 10. Gotchas

- **`num_gazing_each_frame.shape[0] != T_full`** → 데이터 오염 또는 JPG 누락. 학습 스크립트가 assert 로 잡음.
- **OOM**: `--num-frames 24` → `16` 순으로 축소. `--grad-accum` 은 loss 곡선 완만화용이지 메모리 감소 효과 없음.
- **체크포인트 디렉토리**: `save_pretrained` 가 자동 생성 — 별도 mkdir 불필요.
- **wandb 인증**: 이미 `yujinbae` 로 로그인됨. 다른 계정이면 `wandb login` 재실행 필요.
- **sdpa 고정**: `--attn-implementation` 같은 CLI 는 없음. sdpa 는 train_path_b.py 에 hard-code (flash-attn2 비호환 이유).

---

## 관련 문서

- [baseline_autogaze_zeroshot.md](baseline_autogaze_zeroshot.md) — native AutoGaze baseline.
- [path_a_autogaze_precomputed.md](path_a_autogaze_precomputed.md) — HD-Video + 사용자 단일-tile zero-shot.
- [path_c_nvila_lite_8b_lora_finetune.md](path_c_nvila_lite_8b_lora_finetune.md) — Lite-8B LoRA SFT plan.
- [`vila_hd_work/notes/multiscale_wiring.md`](../../vila_hd_work/notes/multiscale_wiring.md) — HD-Video vs 사용자 scales 분석.
- 코드: [`vila_hd_work/train_path_b.py`](../../vila_hd_work/train_path_b.py), [`vila_hd_work/eval_egogaze.py`](../../vila_hd_work/eval_egogaze.py).
