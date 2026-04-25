# Path A — NVILA-8B-Video (vanilla) native zero-shot

Baseline (NVILA-8B-HD-Video + AutoGaze on-the-fly) 과 **같은 선상에서 backbone 만** vanilla NVILA-8B-Video 로 교체해 accuracy 만 측정한다. **학습 없음, AutoGaze 호출 없음, 사용자 `.pt` 주입 없음.** HD-Video vs Video 아키텍처 차이와 AutoGaze 유무가 EgoGazeVQA 성능에 주는 영향을 분리해 관찰하는 용도.

```
 video (mp4)
    ▼
 AutoProcessor.preprocess  (vanilla Video — multi-tile/AutoGaze 없음)
    ▼
 vision_tower              (processor default scales, bf16)
    ▼
 mm_projector → Qwen2 LLM.generate → 첫 A-E letter
```

---

## 1. Environment

Baseline / Path B 와 동일. 세션마다 export:

```bash
export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8
export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Conda env `trajgaze`. 하드웨어 2×H200 (bf16, sdpa).
- wandb / peft 불필요 (학습 없음).

---

## 2. Data assets (read-only)

| 용도 | 경로 |
| --- | --- |
| 원본 QA (1750행) | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/metadata.csv` |
| 원본 클립 (mp4, processor 가 직접 디코드) | `/home/yujin/dataset/EgoGazeVQA/{dataset}/{video_id}/{start}_{end}.mp4` |
| 8:2 split | `/home/yujin/gaze/trajgaze/vila_hd_work/splits.json` (1373 train / 377 test) |

Baseline 과 동일하게 test split 377 샘플 전체에 대해 돌리며, 사용자 `.pt` / pre-extract JPG 는 **쓰지 않는다**.

---

## 3. Pipeline internals (`eval_path_a.py`)

vanilla NVILA-8B-Video (`Efficient-Large-Model/NVILA-8B-Video`) 는 HF `AutoModel` 이 model_type `llava_llama` 를 인식하지 못함. 그래서 **VILA 의 `llava.load` / `llava.Video` / `generate_content` API** 를 직접 사용한다.

| 단계 | 구현 |
| --- | --- |
| sys.path | `sys.path.insert(0, "/home/yujin/gaze/trajgaze/VILA")` → `import llava` |
| Model load | `model = llava.load("Efficient-Large-Model/NVILA-8B-Video", device_map="auto")`; `model.config.num_video_frames = 32`; `model.eval()` |
| Input | `prompt = [llava.Video(video_path), text]` — mp4 경로를 그대로 전달 (`_load_video` 가 cv2 로 32 frame uniform sample). |
| Prompt text | `"\n\n{question}\nOptions:\n{A~E}\nAnswer with the letter only."` (video_token 은 `extract_media` 가 자동 삽입하므로 텍스트에서 제외). |
| Generate | `model.generate_content(prompt, generation_config=GenerationConfig(max_new_tokens=8, do_sample=False))` |
| Answer parse | `re.search(r"\b([A-E])\b", raw)` — baseline 과 동일. |

모델은 FROZEN. `generate_content` 내부에서 `torch.inference_mode()` wrap.

---

## 4. Key decisions (why)

- **Backbone 만 교체**: baseline 의 파이프라인 코드(`eval_egogaze.py --mode native`) 를 그대로 쓰고 `--model-path` 만 `Efficient-Large-Model/NVILA-8B-Video` 로 교체. 다른 eval 변수 (split, prompt, regex, K) 는 전부 baseline 과 문자 일치 → 모델 아키텍처 효과만 분리.
- **AutoGaze 미호출**: vanilla Video processor 는 AutoGaze 통합 훅이 없다. 따라서 이 조건은 "모델 + AutoGaze" 대신 "모델만" 을 측정한다. baseline 과의 차이 = (HD-Video - Video) + (AutoGaze - no AutoGaze) 두 요소를 복합적으로 관찰.
- **scales / tile / thumbnail kwargs 제외**: vanilla processor 는 HD 전용 kwargs 를 모른다. 기본값을 그대로 사용.
- **K=32 고정**: 참고 논문 `https://arxiv.org/pdf/2603.12254` (AutoGaze 논문) 은 short-clip VQA 에 대한 K 를 명시하지 않는다 — EgoGazeVQA 자체도 평가 대상이 아님. baseline.md 와 동일한 K=32 로 맞춰 공정 비교.
- **bf16 + sdpa**: 위 논문의 "FP32 + flash-attn 비활성화" 구성은 2×H200 메모리로 OOM. 실무적 선택으로 baseline 의 bf16/sdpa 설정을 유지.
- **`attn_implementation="sdpa"`**: flash-attn 2 block-causal 비호환 가능성 회피.

---

## 5. 파일 레이아웃 (Path A 관련분만)

```
vila_hd_work/
  eval_path_a.py                     ← VILA llava.load API 기반 vanilla Video eval
  data/test.jsonl                    ← 375 `qa_idx` (Path B/C 와 공유)
  out/path_a_test.jsonl              ← 375 샘플 per-sample 결과
  out/path_a_test.log                ← stdout tee
```

※ `data/single/*.jsonl`, `results/baseline/nvila_perframe_single/` 등 구 Path A 산출물은 신 정의에서 사용하지 않는다. 파일 자체는 추후 재활용 여지가 있어 보존.

---

## 6. 실행 커맨드

### Smoke test (5 샘플)

```bash
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python eval_path_a.py \
  --limit 5 \
  --out-jsonl out/path_a_smoke.jsonl
```

기대: `llava.load` 성공, 5 샘플 generate, regex 파싱 OK, error 없음. 첫 실행은 weights (~16 GB) 다운로드 포함으로 느릴 수 있음.

### Full test eval (375 샘플)

```bash
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python eval_path_a.py \
  --out-jsonl out/path_a_test.jsonl \
  2>&1 | tee out/path_a_test.log
```

---

## 7. 결과 (test split, 375 `qa_idx` 정렬)

eval 은 test split 377 전수에 대해 실행하지만, Path B/C 와의 공정 비교를 위해 `data/test.jsonl` 의 375 `qa_idx` 교집합에 대해서만 집계한다 (CLAUDE.md §4 각주 및 Baseline §7 과 동일 규칙).

| 지표 | 값 |
| --- | --- |
| **Overall** | **0.5573** (209/375) |
| qa_type=causal | 0.8175 (103/126) |
| qa_type=spatial | 0.3680 (46/125) |
| qa_type=temporal | 0.4839 (60/124) |
| dataset=ego4d | 0.4925 (66/134) |
| dataset=egoexo | 0.6599 (97/147) |
| dataset=egtea | 0.4894 (46/94) |

Baseline (HD-Video + AutoGaze zeroshot) 0.5387 대비 **+0.019** 차이. vanilla NVILA-8B-Video 가 AutoGaze 없이도 baseline 과 거의 동등한 성능 (overall). breakdown 을 보면: causal (-0.024) / ego4d (-0.045) 에서 약간 하락하지만, spatial (+0.008) / temporal (+0.073) 에서 오히려 개선. HD-Video 의 AutoGaze 가 causal 에 도움되지만 temporal 에서는 vanilla 가 유리한 역설적 결과. egoexo 는 두 조건이 정확히 같음 (97/147).

### Baseline 비교 (기록용)

같은 375 `qa_idx` 에서 baseline (`out/baseline_test.jsonl`) 과 per-sample join → Δacc 산출. HD-Video + AutoGaze 대비 vanilla Video 의 상대 성능 변화를 관찰. 377 전수 수치도 `out/path_a_test.jsonl` 에 남으므로 필요 시 별도 집계 가능.

---

## 8. Gotchas

- **HF AutoModel 미지원**: vanilla NVILA-8B-Video 는 `model_type: llava_llama` 로 HF 에 등록되어 있지 않다. `AutoModel.from_pretrained(...)` 호출은 `KeyError: 'llava_llama'` 로 깨짐. 반드시 `llava.load()` 를 사용.
- **`ps3` / `fp8linearqwen2` / `no_init_weights(_enable=...)` 호환 stub**: VILA 는 PS3 vision tower 와 FP8LinearQwen2 를 eager import 한다. 둘 다 vanilla Video 경로에서 실제로 사용되지 않으므로, Path C 와 동일한 방식으로 `sys.modules` 에 더미 모듈을 넣고 `modeling_utils.no_init_weights` 도 `_enable` 인자 허용 wrapper 로 패치 (train_path_c.py 30–86행 참조).
- **HF 모델 id**: `Efficient-Large-Model/NVILA-8B-Video` 가 정식. `nvidia/NVILA-8B-Video` 로 이동했을 수 있으니 redirect 여부 확인.
- **첫 실행은 가중치 다운로드 포함**. HF cache (~/.cache/huggingface/hub) 에 처음엔 `config.json` 만 있을 수 있음. `llava.load` 호출 시 weights 전체 (~16 GB) 가 받아진다.
- **mp4 입력**: `llava.Video(mp4_path)` 가 cv2 로 직접 디코드. `config.num_video_frames` 개수만큼 uniform sample.
- **dataset-root**: `/home/yujin/dataset/EgoGazeVQA` (all_gaze_v1/ 가 아니다 — 클립은 parent 에 있음).

---

## 관련 문서

- [baseline_autogaze_zeroshot.md](baseline_autogaze_zeroshot.md) — HD-Video + AutoGaze native baseline (비교 기준).
- [path_b_autogaze_lora_finetune.md](path_b_autogaze_lora_finetune.md) — HD-Video LoRA SFT.
- [path_c_nvila_lite_8b_lora_finetune.md](path_c_nvila_lite_8b_lora_finetune.md) — Lite-8B LoRA SFT plan.
- 코드: [`vila_hd_work/eval_path_a.py`](../../vila_hd_work/eval_path_a.py) (llava.load API 기반 Path A eval).
