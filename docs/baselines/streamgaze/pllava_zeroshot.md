# PLLaVA-7B × StreamGaze_v2 EGTEA — Vanilla Zero-shot

**구현 완료. Eval 실행 결과는 §10 에 기록.**  PruneVid 의 backbone 인 **PLLaVA-7B** ([ermu2001/pllava-7b](https://huggingface.co/ermu2001/pllava-7b)) 을 token-pruning 을 완전히 비활성화 (`tau=1, temporal_segment_ratio=1, cluster_ratio=1`) 한 상태로 StreamGaze_v2 EGTEA subset 의 8 non-proactive 태스크 (526 QA) 에 zero-shot 평가.

프레임 수 스윕: **16 / 32 / 64** 세 세팅 (모두 `pooling_shape=(16,12,12)` 공통).

```
 StreamGaze_v2/qa/{task}.json                         (8 non-proactive tasks)
 StreamGaze_v2/videos/egtea/original/OP##-R##-*.mp4   (35 EGTEA videos)
     │
     ▼
 StreamGazeEgteaDataset  (subclass of PruneVid EvalDataset)
   - EGTEA video 만 필터 → 526 records
   - task 별 윈도우 정책:
       present_future_action_prediction_egtea  → [0, resp_start]   ← 미래 leak 방지
       나머지 7 tasks                          → [resp_start, resp_end]
   - num_frames ∈ {16, 32, 64} uniform sampling (decord)
     │
     ▼
 PllavaProcessor
   CLIP-ViT-L/14-336 (336×336, mean/std = [0.481, 0.458, 0.408] / [0.269, 0.261, 0.276])
     │   [T, 3, 336, 336]
     ▼
 PllavaForConditionalGeneration (PLLaVA-7B, bf16, use_lora=True, lora_alpha=4)
   - vision_tower:       CLIPVisionModel
   - multi_modal_projector: MLP + pooling  pooling_shape=(16,12,12)
     → T frames → 16 temporal slots × 12×12 spatial tokens  (2304 tokens)
   - language_model:     Llama-2-7b + LoRA (q_proj/v_proj, α=4)
   - PruneVid pruning params: tau=1.0, temp_ratio=1.0, cluster_ratio=1.0  (모두 OFF)
     │
     ▼
 generate(do_sample=False, max_new_tokens=100)
 conv_mode = eval_mvbench,  answer_prompt = "Best option:("
     │
     ▼
 re.split("Best option:(", out)[-1]  →  "(A|B|C|D) ..." 파싱
     │
     ▼
 check_ans  (letter-level)  →  task-wise + overall accuracy
```

---

## 1. Environment

conda env `prunevid` (Python 3.10, CUDA 11.8):

```bash
conda create -n prunevid python=3.10 -y
/opt/conda/envs/prunevid/bin/pip install -r /home/yujin/gaze/PruneVid/requirements.torch.txt
/opt/conda/envs/prunevid/bin/pip install \
  "transformers==4.42.4" "accelerate==0.26.1" "peft==0.10.0" \
  "safetensors" "huggingface_hub" "decord==0.6.0" \
  "opencv-python-headless" "moviepy==1.0.3" "imageio==2.34.0" "imageio-ffmpeg" \
  "numpy<2" "einops" "tqdm" "sentencepiece" "protobuf" \
  "Pillow" "easydict" "matplotlib" "scipy" "scikit-learn"
```

> **주의**: PruneVid `requirements.txt` 는 `transformers==4.37.1` 을 핀하지만 모델 코드가 **4.41+ 에서 추가된 `config.mlp_bias` / `cache_utils.StaticCache`** 를 사용하므로 **4.42.4** 가 실제 최소 요구 버전. `av==10.0.0` (PyAV) 은 `pkg-config` 시스템 패키지가 필요해 빌드 실패하지만 본 eval 은 decord 로만 비디오를 읽으므로 제외 가능. `mmcv-full` 도 optical-flow 모듈용 optional dependency 로 제외.

GPU: **2 × H200 (143GB each)**. 두 GPU 로 데이터 샤딩 (`--multiprocess 1`, default) 시 16f/32f/64f 모두 여유. single-GPU 도 가능 (16f ~14GB, 32f ~25GB, 64f ~45GB bf16 기준).

### 1.1 PruneVid 원본 코드 패치 (transformers 4.42+ 호환)

PruneVid 원본은 구 transformers 에서만 동작하므로 아래 4 군데 패치가 필요:

1. **[tasks/eval/model_utils.py:11](/home/yujin/gaze/PruneVid/tasks/eval/model_utils.py#L11)** — `mmcv.runner.load_checkpoint` import 를 try/except 로 감싸 mmcv 없이도 로드 가능하게:
   ```python
   try:
       from mmcv.runner import load_checkpoint
   except ImportError:
       def load_checkpoint(model, filename, map_location=None, **kwargs):
           raise ImportError("mmcv not installed; optical flow model loading is skipped.")
   ```
   (optical flow 로딩은 `except` 로 fallback 되어 eval 에 영향 없음.)

2. **[models/pllava/llama.py:1480](/home/yujin/gaze/PruneVid/models/pllava/llama.py#L1480)** — `BaseModelOutputWithPast(..., attention_mask=...)` 에서 `attention_mask` 인자 제거 (transformers 의 표준 output class 가 이 필드를 지원하지 않음).

3. **[models/pllava/llama.py:2289](/home/yujin/gaze/PruneVid/models/pllava/llama.py#L2289)** — `CausalLMOutputWithPast(..., attention_mask=outputs.attention_mask)` 에서 `attention_mask` 인자 제거.

4. **[models/pllava/modeling_pllava.py:975](/home/yujin/gaze/PruneVid/models/pllava/modeling_pllava.py#L975)** — `if labels is None: flag = True` 를 `flag = False` 로 변경. Pruning 이후 logits 는 shortened 되는데 attention_mask 로 만든 dummy labels 는 unpruned 크기라 `shift_attention_mask` indexing 에서 IndexError 발생. eval/generate 에서는 loss 계산이 불필요하므로 `labels = None` 을 유지하고 loss 분기 스킵.


---

## 2. 레포 & 모델

**PruneVid 레포** (PLLaVA backbone + token pruning 구현 공유):
```bash
git clone https://github.com/Visual-AI/PruneVid /home/yujin/gaze/PruneVid  # 이미 존재
```

**PLLaVA-7B 가중치** (LoRA adapter + base Llama + CLIP vision tower 포함):
```bash
huggingface-cli download ermu2001/pllava-7b \
  --local-dir /home/yujin/gaze/PruneVid/MODELS/pllava-7b
```

크기: `model-00001-of-00003.safetensors` 등 ~14GB.

---

## 3. Data assets

### 3.1 StreamGaze_v2 QA (read-only)

`/home/yujin/dataset/StreamGaze_v2/qa/*.json` — **8 non-proactive tasks** 만 사용 (proactive 2개 시계열 alert 는 generate 와 호환 안 되어 제외):

| # | Task JSON | EGTEA QA | 비디오 윈도우 |
|---|---|---:|---|
| 1 | `past_gaze_sequence_matching.json` | 64 | `response_time` |
| 2 | `past_non_fixated_object_identification.json` | 68 | `response_time` |
| 3 | `past_object_transition_prediction.json` | 2 | `response_time` |
| 4 | `past_scene_recall.json` | 37 | `response_time` |
| 5 | `present_object_attribute_recognition.json` | 96 | `response_time` |
| 6 | `present_object_identification_easy.json` | 101 | `response_time` |
| 7 | `present_object_identification_hard.json` | 64 | `response_time` |
| 8 | `present_future_action_prediction_egtea.json` | 94 | **`[0, resp_start]`** |
| | **Total** | **526** | |

- Task 8 은 **미래 leak 방지**: `response_time` 시작 시각 이전 프레임만 관찰 (`[0, resp_start]`).
- `present_future_action_prediction.json` (mother file, 921) 은 `_egtea.json` (94) 이 EGTEA-only 분리본과 동일 sample 이므로 후자만 사용.

각 QA entry 구조:
```json
{
  "response_time": "[02:22 - 13:20]",
  "video_path": "OP01-R01-PastaSalad.mp4",
  "questions": [
    {
      "question": "Which object is the user currently gazing at?",
      "time_stamp": "00:08",
      "answer": "C",
      "options": ["A. meat", "B. counter", "C. pot", "D. cabinet"]
    }
  ]
}
```

### 3.2 Videos

`/home/yujin/dataset/StreamGaze_v2/videos/egtea/original/` — 35 MP4 (OP01-R01-PastaSalad.mp4 … OP06-R07-Pizza.mp4).
decord 로 직접 읽음. `frames/egtea/viz/*/*.jpg` 는 사용하지 않음 (viz overlay 버전이라 zero-shot 평가에는 original 사용).

---

## 4. 구현 방식

### 4.1 StreamGazeEgteaDataset — `tasks/eval/streamgaze/__init__.py`

PruneVid 의 [`tasks/eval/egoshcema/__init__.py`](/home/yujin/gaze/PruneVid/tasks/eval/egoshcema/__init__.py) 의 `EgoSchemaDataset` 을 템플릿으로 복제 후 다음을 변경:

- **`TASK_FILES`**: 8 개 JSON 파일명을 리스트로 등록.
- **`__init__`**: 각 JSON 로드 → `video_path ∈ os.listdir(videos/egtea/original)` 필터 → `(sample, q_idx)` 평탄화하여 526 records.
- **Window 결정** (`_parse_response_time` 로 `"[MM:SS - MM:SS]"` → (start_sec, end_sec) 변환):
  ```python
  if task_type == "present_future_action_prediction_egtea":
      bound = (0.0, rt_start)
  else:
      bound = (rt_start, rt_end)
  ```
- **`qa_template`**: StreamGaze 의 `options = ["A. xxx", ...]` 접두사 제거 후 재조합:
  ```python
  m = re.match(r'^\s*([A-D])[\.\):]\s*(.+)$', opt)
  content = m.group(2).strip() if m else opt
  q += f"({chr(ord('A') + idx)}) {content}\n"
  answer = f"({letter}) {content_of_correct}"
  ```
- **`__getitem__`**: base class 의 `read_video(video_path, bound)` 재사용 — bound 는 `(start_sec, end_sec)`, 이 내부에서 decord 로 `num_segments` 개 frame 을 uniform sampling.

### 4.2 Eval 스크립트 — `tasks/eval/streamgaze/pllava_eval_streamgaze.py`

[`tasks/eval/egoshcema/pllava_eval_egoschema.py`](/home/yujin/gaze/PruneVid/tasks/eval/egoshcema/pllava_eval_egoschema.py) 을 템플릿으로 복제 후:

- `EgoSchemaDataset` → `StreamGazeEgteaDataset` 교체.
- `--max_samples` argparse 옵션 추가 (smoke test 용).
- `run()` 시작 시 첫 sample 의 `(task, video, bound)` 를 로깅 (window policy 검증용).
- `infer_mvbench` 는 그대로 재사용:
  - `conv_mode=eval_mvbench` (system: "Carefully watch the video …")
  - `answer_prompt="Best option:("`, `return_prompt='('` → output 을 `"(X)"` prefix 로 정규화.

### 4.3 Prompt 구성

conv template `eval_mvbench` ([eval_utils.py:184](/home/yujin/gaze/PruneVid/tasks/eval/eval_utils.py#L184)):

```
Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.
USER: <image>
Question: {question}
Options:
(A) {opt_A}
(B) {opt_B}
(C) {opt_C}
(D) {opt_D}
Only give the best option.  ASSISTANT: Best option:(
```

`<image>` 는 processor 에서 frame tokens 으로 expand 됨.

### 4.4 Letter 파싱 (`check_ans`)

[tasks/eval/mvbench/__init__.py:11](/home/yujin/gaze/PruneVid/tasks/eval/mvbench/__init__.py#L11) 을 `streamgaze/__init__.py` 에 복제. `pred = "(A) xxx"`, `gt = "(C) xxx"` 형식으로 옴. 첫 공백-split 토큰끼리 `"(a)"` vs `"(c)"` 부분 매칭으로 정오판정 — 모델이 `"(A)"` 만 출력해도 동작.

---

## 5. Eval CLI

### 5.1 3-way launcher — `scripts/eval_streamgaze_vanilla.sh`

16f / 32f / 64f 를 순차 실행 후 `aggregate_results.py` 로 비교 표 출력:

```bash
cd /home/yujin/gaze/PruneVid
bash scripts/eval_streamgaze_vanilla.sh
```

launcher 내부 명령 (한 세팅):
```bash
python -m tasks.eval.streamgaze.pllava_eval_streamgaze \
  --pretrained_model_name_or_path MODELS/pllava-7b \
  --save_path test_results/streamgaze_egtea/vanilla_16f \
  --num_frames 16 \
  --use_lora --lora_alpha 4 --weight_dir MODELS/pllava-7b \
  --pooling_shape 16-12-12 \
  --conv_mode eval_mvbench \
  --tau 1.0 --temporal_segment_ratio 1.0 --cluster_ratio 1.0 \
  --top_p 1.0 --temperature 1.0
```

환경변수로 오버라이드:
- `MODEL_DIR`     (기본 `MODELS/pllava-7b`)
- `LORA_ALPHA`    (기본 4 — PLLaVA 공식; PruneVid eval.sh 는 14)
- `POOLING_SHAPE` (기본 `16-12-12`)
- `FRAMES`        (기본 `"16 32 64"`)
- `SAVE_ROOT`     (기본 `test_results/streamgaze_egtea`)

### 5.2 Smoke test

```bash
python -m tasks.eval.streamgaze.pllava_eval_streamgaze \
  --pretrained_model_name_or_path MODELS/pllava-7b \
  --save_path /tmp/smoke \
  --num_frames 16 \
  --use_lora --lora_alpha 4 --weight_dir MODELS/pllava-7b \
  --pooling_shape 16-12-12 \
  --conv_mode eval_mvbench \
  --tau 1.0 --temporal_segment_ratio 1.0 --cluster_ratio 1.0 \
  --max_samples 8
```

8 sample (task 당 1 개씩) 로 전체 파이프라인 동작 확인. 로그에 `[window-probe] task=... bound=(start, end)` 가 출력됨.

### 5.3 결과 집계

```bash
python -m tasks.eval.streamgaze.aggregate_results \
  --run 16f=test_results/streamgaze_egtea/vanilla_16f \
  --run 32f=test_results/streamgaze_egtea/vanilla_32f \
  --run 64f=test_results/streamgaze_egtea/vanilla_64f
```

markdown 표를 stdout 에 출력.

---

## 6. Token-count probe

| Setting | num_frames | pooling_shape | Vision tower 출력 (per video) | LLM 입력 video tokens |
|---|---|---|---|---|
| Vanilla-16f | 16 | 16-12-12 | [1, 16, 577, 1024]         | 16 × 144 = **2,304** |
| Vanilla-32f | 32 | 16-12-12 | [1, 32, 577, 1024]         | (2:1 avgpool) 16 × 144 = **2,304** |
| Vanilla-64f | 64 | 16-12-12 | [1, 64, 577, 1024]         | (4:1 avgpool) 16 × 144 = **2,304** |

LLM 입력 토큰 수는 `pooling_shape` 에 의해 결정되므로 세 세팅 모두 동일 (2,304 video + prompt ≈ 2.5K). vision tower 연산량만 frame 수에 비례.

---

## 7. 파일 레이아웃

```
PruneVid/
  tasks/eval/streamgaze/
    __init__.py                          ← 신규 (StreamGazeEgteaDataset, check_ans, save/load_results)
    pllava_eval_streamgaze.py            ← 신규 (eval main)
    aggregate_results.py                 ← 신규 (run 간 비교 표)
  scripts/
    eval_streamgaze_vanilla.sh           ← 신규 (16/32/64 일괄)
    eval_streamgaze_prunevid.sh          ← 신규 (pruning ON — §prunevid_zeroshot.md 참고)
  MODELS/pllava-7b/                      ← HF download
    model-0000{1,2,3}-of-00003.safetensors
    config.json / preprocessor_config.json / tokenizer.model / ...
  test_results/streamgaze_egtea/
    vanilla_16f/
      all_results.json                   ← 526 records + per-task accuracy
      upload_leaderboard.json
    vanilla_32f/ ...
    vanilla_64f/ ...
```

---

## 8. Key decisions (why)

- **Vanilla = pruning params OFF, 나머지는 PLLaVA 공식**: `use_lora=True, lora_alpha=4` 는 PLLaVA 원본 release 가 권장하는 inference 세팅. PruneVid `scripts/eval.sh` 는 `lora_alpha=14` 를 쓰지만 이는 PruneVid re-training 파이프라인용이라 PLLaVA zero-shot 비교엔 부적절. 본 eval 에서 pruning 만 off 하면 PLLaVA-7B 원 zero-shot 재현.
- **pooling_shape 고정 `16-12-12`**: PLLaVA-7B pretraining 시 `T=16` 으로 학습된 projector. T 를 frame 수에 맞춰 (32,12,12)/(64,12,12) 로 늘리면 LLM 입력 토큰이 4,608/9,216 까지 증가 → context 초과 + projector 가 학습하지 않은 slot 개수. 따라서 vision tower 만 고해상도 temporal 로 돌리고 LLM 입력은 일정하게 유지 (4:1 까지 평균 pool).
- **frame sweep {16, 32, 64}**: 16f 는 PruneVid official reference (`scripts/eval.sh:2`), 32/64f 는 temporal resolution ablation. 비디오가 1–15 분이라 16 frames/video 는 20–60 초 간격 샘플링으로 fine-grained gaze pattern 을 놓칠 수 있음 — 그 영향 관찰.
- **미래 leak 방지 `[0, resp_start]`**: `present_future_action_prediction` 은 정의상 "resp_start 시점 이후에 일어날 행동" 을 예측. 비디오에서 resp_end 까지 보여주면 모델이 미래 증거를 관찰하게 됨. 다른 7 tasks 는 과거/현재 관찰이라 전구간 OK.
- **Proactive 2 tasks 제외**: 시계열 alert 는 `test_info[]` 에 여러 timestamp 의 binary positive/negative 를 갖는 포맷 — sliding window + yes/no generate 가 필요. 본 eval 의 single-forward MC 인퍼런스 인프라와 호환 안 됨.
- **`present_future_action_prediction_egtea` 만 사용 (mother 제외)**: 두 파일의 EGTEA sample (94 개) 은 같은 content — 중복 방지.
- **`do_sample=False, temperature=1.0, top_p=1.0`**: deterministic decoding. EgoSchema eval 과 동일.

---

## 9. Reproduce quick reference

```bash
cd /home/yujin/gaze/PruneVid

# (0) 모델 다운로드 (한 번만)
huggingface-cli download ermu2001/pllava-7b \
  --local-dir MODELS/pllava-7b

# (1) smoke test
python -m tasks.eval.streamgaze.pllava_eval_streamgaze \
  --pretrained_model_name_or_path MODELS/pllava-7b \
  --save_path /tmp/smoke --num_frames 16 \
  --use_lora --lora_alpha 4 --weight_dir MODELS/pllava-7b \
  --pooling_shape 16-12-12 --conv_mode eval_mvbench \
  --tau 1.0 --temporal_segment_ratio 1.0 --cluster_ratio 1.0 \
  --max_samples 8

# (2) 전체 3-way eval
bash scripts/eval_streamgaze_vanilla.sh

# (3) markdown 표로 집계
python -m tasks.eval.streamgaze.aggregate_results \
  --run 16f=test_results/streamgaze_egtea/vanilla_16f \
  --run 32f=test_results/streamgaze_egtea/vanilla_32f \
  --run 64f=test_results/streamgaze_egtea/vanilla_64f
```

---

## 10. 실행 결과

526 QA 전체 × 3 frame 세팅 완료 (2025-04-24, 2 × H200 샤딩, 각 run ≈ 5-8 분).

### 10.1 Task 별 accuracy (%)

| Task | N | Vanilla-16f | Vanilla-32f | Vanilla-64f |
|---|---:|---:|---:|---:|
| past_gaze_sequence_matching | 64 | 39.06 | 32.81 | 31.25 |
| past_non_fixated_object_identification | 68 | 27.94 | 27.94 | 29.41 |
| past_object_transition_prediction | 2 | 50.00 | 50.00 | 50.00 |
| past_scene_recall | 37 | 8.11 | 8.11 | 8.11 |
| present_object_attribute_recognition | 96 | 59.38 | 55.21 | 56.25 |
| present_object_identification_easy | 101 | 75.25 | 76.24 | 77.23 |
| present_object_identification_hard | 64 | 70.31 | 70.31 | 68.75 |
| present_future_action_prediction_egtea | 94 | 29.79 | 30.85 | 27.66 |
| **Overall (micro)** | **526** | **48.29** | **47.15** | **46.77** |

### 10.2 관측

- **Overall**: frame 수가 늘어도 정확도가 거의 변하지 않거나 미세하게 하락 (48.29 → 47.15 → 46.77). `pooling_shape=(16,12,12)` 고정이라 LLM 입력 토큰 수가 동일하고, 32/64 frame 은 vision tower 에서 4:1-까지 average pool 되어 정보가 평균화됨.
- **past_gaze_sequence_matching**: 16f 가 가장 높음 (39.06). frame 을 더 많이 보여도 EGTEA 비디오의 긴 span(수 분) 내 gaze 전환 패턴을 avgpool 이 흐리는 것으로 보임.
- **past_scene_recall**: 세 세팅 모두 8.11% 로 random (4-choice=25%) 을 크게 하회. PLLaVA 가 "이전에 본 scene recall" 을 잘 못한다는 뜻 — StreamGaze 의 가장 어려운 태스크.
- **present_object_identification_easy**: 세 세팅 모두 70%+ 로 강한 baseline. 현재 gaze 대상 객체는 FOV 중앙이라 프레임 샘플링 분포에 상대적으로 덜 민감.
- **present_future_action_prediction_egtea**: 세 세팅 모두 27-31% (random 대비 소폭 상회). `[0, resp_start]` 클리핑으로 미래 정보 차단된 상태의 baseline 이므로 PruneVid token pruning 이 이 태스크에 영향 주는지 비교 기준이 됨.
- **past_object_transition_prediction**: N=2 라 신뢰도 낮음 (50% 는 우연한 결과).
- **태스크별 frame-민감도**: 가장 민감한 태스크는 `past_gaze_sequence_matching` (Δ -7.81 pp) 와 `present_object_attribute_recognition` (Δ -4.17 pp). 다른 태스크는 ±2 pp 이내.

### 10.3 이슈 및 patch 로그

- 초기 시도에서 `num_frames=32/64` 가 `media_type=None` + `L mismatch` assertion 으로 전부 실패. 두 가지 patch 추가:
  - **[models/pllava/modeling_pllava.py:766](/home/yujin/gaze/PruneVid/models/pllava/modeling_pllava.py#L766)**: `merge_frames_dynamic` 의 `assert L == num_frames × pool_H × pool_W` 를 `pool_T × pool_H × pool_W` 로 수정 (projector 출력 의 temporal dim 은 pool 된 `pooling_shape[0]` 이지 원본 `num_frames` 가 아님).
  - **[models/pllava/modeling_pllava.py:1087](/home/yujin/gaze/PruneVid/models/pllava/modeling_pllava.py#L1087) (`generate`)** 와 **:1097 (`prepare_inputs_for_generation`)**: transformers 4.42 가 generate 후속 step 에서 `media_type` kwargs 를 제거하므로 `self._pending_media_type` 속성으로 stash + fallback, 동시에 2nd step 이후 `pixel_values=None` 으로 강제하여 vision tower 재호출 방지.


---

## 11. References

- PLLaVA: [arXiv:2404.16994](https://arxiv.org/abs/2404.16994) (Xu et al., 2024) — group pooling adapter 로 image LLaVA 를 video 로 확장
- PruneVid: [arXiv:2503.01023](https://arxiv.org/abs/2503.01023) — training-free visual token pruning for video LLMs
- PruneVid 코드: [github.com/Visual-AI/PruneVid](https://github.com/Visual-AI/PruneVid)
- StreamGaze_v2 데이터: `/home/yujin/dataset/StreamGaze_v2/README.md`
- 관련 baseline 문서:
  - [llava_prumerge_lora.md](./llava_prumerge_lora.md) — Video-LLaVA + LLaVA-PruMerge LoRA fine-tune
  - [trajgazemerge_naive_baseline.md](./trajgazemerge_naive_baseline.md) — TrajGazeMerge naive baseline
  - [prunevid_zeroshot.md](./prunevid_zeroshot.md) — pruning ON variant (같은 어댑터 공유)
