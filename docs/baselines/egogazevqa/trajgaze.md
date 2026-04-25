# trajgaze — EgoGazeVQA × NVILA-HD-Video 실험 가이드

EgoGazeVQA benchmark (1750 QA, 3 dataset) 에 대해 NVILA 계열 MLLM 의 gaze-guided 추론 성능을 baseline / Path A / Path B / Path C 네 조건에서 측정하는 연구 저장소. 본 문서는 최상위 인덱스: 디렉토리·데이터 경로 + 전체 결과 표.

---

## 1. 디렉토리 개요 (`/home/yujin/gaze/trajgaze/`)

| 경로 | 용도 |
| --- | --- |
| [`AutoGaze/`](../AutoGaze/) | VideoMAE-based gaze selection 모듈. `INTEGRATION.md` 와 SigLIP 포팅 레퍼런스 (`autogaze/vision_encoders/siglip/*.py`). Path C 포팅 원천. |
| [`TrajGaze/`](../TrajGaze/) | `baselines/run_baseline.py` — JPG 시퀀스에 Talk2DINO + gaze/hand prior 로 gazing_info 를 생성하는 파이프라인. 사용자 `.pt` 생성기. 상세: [`naive_patch_selector.md`](baselines/naive_patch_selector.md). |
| [`VILA/`](../VILA/) | NVlabs/VILA clone (`vila_hd/` 포함). HD-Video / Lite-8B inference 레퍼런스. Stage 5 공식 학습 코드는 여전히 미공개. |
| [`Talk2DINO/`](../Talk2DINO/) | DINOv2-based vision-language alignment, TrajGaze 내부에서 사용. |
| [`preprocess/`](../preprocess/) | 원본 EgoGazeVQA 전처리 스크립트. |
| [`results/baseline/`](../results/baseline/) | 사용자가 생성한 gazing_info 세트 저장소 (아래 데이터 섹션 참조). |
| [`vila_hd_work/`](../vila_hd_work/) | **본 실험의 메인 작업 디렉토리.** 학습·eval 스크립트, split, jsonl, runs, out 전부 여기. |
| [`docs/`](./) | 본 문서 + Path 별 핸드오프 문서 (`docs/baselines/*.md`). |

### `vila_hd_work/` 내부

```
vila_hd_work/
  train_path_b.py              ← HD-Video LoRA SFT (Path B, 구현 완료)
  train_path_c.py              ← Lite-8B + AutoGaze SigLIP runtime swap + LoRA SFT (Path C, 구현 완료)
  eval_path_a.py               ← NVILA-8B-Video (vanilla) zero-shot.
                                 VILA llava.load API 사용 (HF AutoModel 미지원).
  eval_path_c.py               ← Path C 체크포인트 eval (구현 완료)
  eval_egogaze.py              ← native baseline (HD-Video + AutoGaze) + Path B precomputed 전용.
  splits.json                  ← 8:2 group split (video_id 단위, seed 42)
  data/
    train.jsonl test.jsonl     ← Path B/C 멀티스케일용 (1367 / 375)
    single/train.jsonl test.jsonl  ← (구 Path A 단일-tile용, 신 정의에서 미사용 — 보존)
  notes/multiscale_wiring.md   ← HD-Video scales override 근거
  out/                         ← eval 결과 jsonl + log
  runs/<run_name>/             ← 학습 체크포인트 + tb/wandb 로그
  tools/
    split_egogaze.py           ← splits.json 생성
    build_sft_jsonl.py         ← jsonl 생성 (frame-count 검증 포함)
```

---

## 2. 데이터셋 경로

### 원본 EgoGazeVQA — `/home/yujin/dataset/EgoGazeVQA/`

| 경로 | 내용 |
| --- | --- |
| `all_gaze_v1/metadata.csv` | 1750 QA (file_name, video_id, dataset, qa_type, question, answer_options, correct_answer) |
| `{ego4d,egoexo,egtea}/<video_id>/<start>_<end>.mp4` | 클립 원본 (baseline native 모드가 직접 디코드) |
| `all_gaze_v1/{dataset}/no_gaze/<video_id>/<clip>_<frame>.jpg` | Pre-extract JPG (사용자 `.pt` 인덱스 공간 = 이 JPG 순서) |
| `all_gaze_v1/{dataset}/gaze_mapping/<video_id>/<clip>_mapping.csv` | 프레임별 gaze 좌표 (TrajGaze 가 가짜 gaze 계산 시 사용) |
| `all_gaze_v1/{dataset}_jsons/<video_id>.json` | 프레임별 gaze metadata (본 실험 불사용) |

분포: ego4d=577, egoexo=688, egtea=485 (총 1750), causal=584, spatial=584, temporal=582.

### 사용자 gazing_info — `/home/yujin/gaze/trajgaze/results/baseline/`

| 경로 | config | 용도 |
| --- | --- | --- |
| [`nvila_perframe/`](../results/baseline/nvila_perframe/) | `scales="56+112+224+448"`, patch14, NV=1360, 1750 QA 매칭 | **Path B / Path C** 학습 입력 |
| [`nvila_perframe_single/`](../results/baseline/nvila_perframe_single/) | `scales="392"`, patch14, NV=784, 1750 QA 매칭 | 구 Path A 용 (신 정의 미사용, 보존) |
| `perframe_multi/` | (확인 필요) | 현재 미사용 |

각 `.pt` 파일 키: `md5(question)[:8]`. 스키마: `{gazing_pos (1, N) int64, num_gazing_each_frame (T,) int64, if_padded_gazing (1, N) bool}`.

> `nvila_perframe/` 세트는 [`naive_patch_selector.md`](baselines/naive_patch_selector.md) 의 heuristic (Talk2DINO + gaze/hand Gaussian, `scales=56+112+224+448`, `patch14`, `ratio=0.10`, `per_frame`) 로 재생성 가능. 재생성 커맨드 및 파라미터 표는 해당 문서 §5, §7 참조.

### Split — `vila_hd_work/splits.json`

8:2 random group split (by `video_id`, seed 42, sklearn `GroupShuffleSplit`). 1373 train (210 video_ids) / 377 test (53 video_ids). 같은 클립의 모든 QA 는 동일 split 에 배치 → clip leakage 차단.

학습용 jsonl 에서는 JPG 프레임 수와 `.pt` frame 수 불일치 8개 제외 → 1367 / 375.

---

## 3. 조건별 정의 — 한 줄 요약

| 조건 | 모델 | 학습 | Gaze 신호 | Scales / NV |
| --- | --- | --- | --- | --- |
| **Baseline** (autogaze zeroshot) | NVILA-8B-HD-Video | 없음 | 매 비디오마다 **AutoGaze 를 on-the-fly** 로 실행 | 기본 `56+112+196+392` / 1060 |
| **Path A** (NVILA-8B-Video zeroshot) | NVILA-8B-Video (vanilla) | 없음 | 없음 (AutoGaze 호출 없음) | processor default |
| **Path B** (autogaze LoRA finetune) | NVILA-8B-HD-Video | LoRA SFT (2 epoch) | 사용자 **멀티스케일 `.pt`** 주입 | `56+112+224+448` / 1360 |
| **Path C** (NVILA-Lite-8B LoRA finetune) | NVILA-Lite-8B (AutoGaze SigLIP runtime swap) | 2-phase SFT (vision 상위 N 블록 + projector + LLM LoRA) | 사용자 **멀티스케일 `.pt`** 주입 | `56+112+224+448` / 1360 |

상세 문서:
- [`docs/baselines/naive_patch_selector.md`](baselines/naive_patch_selector.md) — gazing_info `.pt` 생성 파이프라인 (Path B/C 학습 입력 생성기).
- [`docs/baselines/baseline_autogaze_zeroshot.md`](baselines/baseline_autogaze_zeroshot.md)
- [`docs/baselines/path_a_nvila8b_video_zeroshot.md`](baselines/path_a_nvila8b_video_zeroshot.md)
- [`docs/baselines/path_b_autogaze_lora_finetune.md`](baselines/path_b_autogaze_lora_finetune.md)
- [`docs/baselines/path_c_nvila_lite_8b_lora_finetune.md`](baselines/path_c_nvila_lite_8b_lora_finetune.md)

---

## 4. 전체 결과 (test split)

### Overall accuracy

| 조건 | N | Overall acc | 상태 |
| --- | --: | --: | --- |
| Baseline (autogaze zeroshot) | 375 | **0.5387** (202/375) | 완료 |
| Path A (NVILA-8B-Video zeroshot) | 375 | **0.5573** (209/375) | 완료 |
| Path B (autogaze LoRA finetune) | 375 | **0.5893** (221/375) | 완료 |
| Path C (NVILA-Lite-8B LoRA) | 375 | **0.6267** (235/375) | 완료 |

### qa_type 별

| 조건 | causal | spatial | temporal |
| --- | --: | --: | --: |
| Baseline | 0.8413 (106/126) | 0.3600 (45/125) | 0.4113 (51/124) |
| Path A | 0.8175 (103/126) | 0.3680 (46/125) | 0.4839 (60/124) |
| Path B | 0.8413 (106/126) | 0.4160 (52/125) | 0.5081 (63/124) |
| Path C | 0.8413 (106/126) | 0.4160 (52/125) | 0.6210 (77/124) |

### dataset 별

| 조건 | ego4d | egoexo | egtea |
| --- | --: | --: | --: |
| Baseline | 0.5373 (72/134) | 0.5714 (84/147) | 0.4894 (46/94) |
| Path A | 0.4925 (66/134) | 0.6599 (97/147) | 0.4894 (46/94) |
| Path B | 0.5373 (72/134) | 0.6599 (97/147) | 0.5532 (52/94) |
| Path C | 0.6194 (83/134) | 0.6939 (102/147) | 0.5319 (50/94) |

> 모든 조건을 375 `qa_idx` (data/test.jsonl 기준, frame-count 미스매치 231/452 제외) 로 정렬. Baseline 은 원본 eval 이 377 이었으나 Path B/C 와 공정 비교 위해 375 로 재필터. Baseline 전수 377 수치는 [`baseline_autogaze_zeroshot.md`](baselines/baseline_autogaze_zeroshot.md) §7 참고.

---

## 5. 환경 변수 (모든 조건 공용, 세션마다 export)

```bash
export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8      # torchvision libjpeg 충돌 회피
export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib              # CXXABI_1.3.15 (cv2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True          # frag 완화
```

Conda env: `trajgaze` (`/opt/conda/envs/trajgaze`). Python 3.10, torch bf16, `attn_implementation="sdpa"`, 2×H200 기준.

Path C 추가 의존성 / 패치:
- `pip install deepspeed s2wrapper@git+https://github.com/bfshi/scaling_on_scales.git`
- `ps3` 는 **설치하지 않음** (transformers 4.49 로 다운그레이드 강제) — train_path_c.py 가 `ps3` 모듈을 dummy 로 stub.
- `fp8linearqwen2` 도 stub (transformers 4.51+ 에서 `Qwen2FlashAttention2` 제거됨).
- `no_init_weights(_enable=True)` → transformers 4.50+ 호환 wrapper.
- Lite-8B 는 단일 GPU (`device_map={"": "cuda:0"}`) 에 적재.

---

## 6. 문서 업데이트 규칙

0. **plan 또는 implementation 이 변경될 때마다 관련 `.md` 파일(들) 을 같은 변경 단위에서 동시에 갱신한다.** 예: Path A 정의를 바꾸면 본 CLAUDE.md 의 섹션 1/3/4 표·링크·트리와 `docs/baselines/path_a_*.md` 를 함께 수정. 코드 변경 (학습 파이프라인, eval 스크립트 인자, 저장 레이아웃 등) 이 생기면 해당 `docs/baselines/<path>.md` 의 실행 커맨드·파일 레이아웃 섹션도 같이 수정. docs 와 코드가 드리프트하면 조건 비교의 의미가 무너진다.

각 Path 의 eval 이 끝날 때마다:
1. 해당 `docs/baselines/<path>.md` 의 결과 섹션 (_TBD_) 를 실제 숫자로 채운다.
2. 본 CLAUDE.md 의 결과 표 3개 (overall / qa_type / dataset) 동기화.
3. 스크립트 구조 변경 (args, 저장 레이아웃 등) 이 생기면 해당 `docs/baselines/<path>.md` 의 실행 커맨드·파일 레이아웃 섹션도 동기화 (0번 규칙 참조).
