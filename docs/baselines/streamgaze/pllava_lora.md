# PLLaVA-7B × StreamGaze_v2 — LoRA Fine-tune (EgoExoLearn+HoloAssist train, EGTEA test)

**구현 완료. Eval 결과는 §10 에 기록 예정.** [pllava_zeroshot.md](./pllava_zeroshot.md) 의 zero-shot baseline 위에 LoRA 어댑터를 학습한 변형. 학습은 StreamGaze_v2 의 EgoExoLearn + HoloAssist 도메인 (5,799 QA) 으로, 평가는 동일 EGTEA 526 QA 로 진행 → zero-shot 과 직접 비교 가능.

```
 StreamGaze_v2/qa/{task}.json         ──filter video_path startswith "OP" (EGTEA 제외)
        │
        ▼
 build_train_json.py                  ──평탄화 → DATAS/streamgaze_v2/train_noegtea.json (5,799 QA)
   per-record: {video, QA:[{q, a}], start, end}
   q  = "Question:...\nOptions:\n(A) ...\n(B) ...\n(C) ...\n(D) ...\nOnly give the best option."
   a  = "Best option:({letter}) {content}"
        │
        ▼
 ITVidTrainDataset (PruneVid 기존 클래스)  ──decord 로 [start_sec, end_sec] uniform sampling
        │
        ▼
 PllavaProcessor + preprocess() (mm_alone=False)
   conv = "{system_mvbench} USER: <Image></Image>\n{q}</s> ASSISTANT: {a}</s>"
   labels: USER 부분은 ignore_index=-100, ASSISTANT 부분만 next-token loss
        │
        ▼
 PllavaForConditionalGeneration  (PLLaVA-7B, bf16, gradient checkpointing)
   - vision_tower: CLIP-ViT-L/14-336    (frozen)
   - multi_modal_projector              (TRAIN, all params unfrozen)
   - language_model: Llama-2-7b + LoRA  (base frozen, LoRA train)
       LoRA r=16, alpha=32, dropout=0.05
       target_modules = q_proj, k_proj, v_proj, o_proj  (LLM attention only)
   - PruneVid pruning: tau=1, cluster_ratio=1, temp_ratio=1  (모두 OFF)
        │
        ▼
 AdamW lr=1e-4, weight_decay=0, cosine 2-epoch, warmup 0.05
 effective batch = 2 (per-GPU) × 8 (grad-accum) = 16
        │
        ▼
 pretrained_epoch01/  (LoRA + projector weights only, ~50MB)
        │
        ▼
 pllava_eval_streamgaze.py + scripts/eval_streamgaze_lora.sh
   load_pllava(... lora_r=16, lora_target_modules=[q,k,v,o]_proj)
   → EGTEA 526 QA × {16f, 64f}
        │
        ▼
 print_csv.py  → 8-task accuracy (지정 순서) + overall_micro
```

---

## 1. Environment

[pllava_zeroshot.md §1](/home/yujin/gaze/trajgaze/docs/baselines/streamgaze/pllava_zeroshot.md) 의 conda env `prunevid` 를 그대로 사용. 추가 의존성 없음 (`accelerate`, `peft==0.10.0` 이미 포함).

zero-shot 의 §1.1 transformers 4.42+ 호환 패치 4 군데도 그대로 유효. 학습 시에는 `labels` 가 not-None 이라 `flag = False` 분기 영향 없고, `tau=cluster_ratio=temp_ratio=1` (no pruning) 이라 logits/labels shape mismatch 없음.

GPU: **2 × H200 (143GB each)**. 64f 학습 효율을 위해 grad checkpointing 필수, batch 2 / accum 8.

---

## 2. 레포 & 모델

zero-shot 과 동일:

```bash
# (이미 존재) PruneVid clone + ermu2001/pllava-7b weight at MODELS/pllava-7b
```

LoRA 어댑터는 학습 결과로 새로 만들어짐 — base model 은 동일 weight 사용.

---

## 3. Data assets

### 3.1 학습 데이터 — 비-EGTEA (5,799 QA)

각 task JSON 의 `video_path` 가 `OP##` 로 시작하는 entry (EGTEA) 는 제외, EgoExoLearn / HoloAssist 만 사용.

| Task | 학습 records |
|---|---:|
| past_gaze_sequence_matching | 122 |
| past_non_fixated_object_identification | 570 |
| past_object_transition_prediction | 492 |
| past_scene_recall | 174 |
| present_object_attribute_recognition | 1,305 |
| present_object_identification_easy | 1,368 |
| present_object_identification_hard | 941 |
| present_future_action_prediction (mother) | 827 |
| **Total** | **5,799** |

source 분포: HoloAssist 1,871 (32%) / EgoExoLearn 3,928 (68%).

> training 에는 mother `present_future_action_prediction.json` (EGTEA 제외) 만 사용. eval 에서 쓰는 `_egtea.json` (94 EGTEA-only) 과 sample 중복 없음 → leakage 차단.

### 3.2 평가 데이터 — EGTEA 526 QA

[pllava_zeroshot.md §3.1](/home/yujin/gaze/trajgaze/docs/baselines/streamgaze/pllava_zeroshot.md) 와 동일. 8 non-proactive task × 35 EGTEA videos. zero-shot 결과와 같은 split 위에서 평가 → 직접 비교.

### 3.3 비디오

```
/home/yujin/dataset/StreamGaze_v2/videos/
    egtea/original/        # 35 mp4 (eval only)
    egoexolearn/original/  # 180 mp4 (train)
    holoassist/original/   # 66 mp4 (train)
```

`build_train_json.py` 는 `video_path` 의 basename 으로 holoassist → egoexolearn 순서로 실제 파일 존재를 검사해 라우팅 (filename prefix 만으로는 일부 HoloAssist 비디오가 z/R 로 시작해 정확하지 않음).

---

## 4. 구현 방식

### 4.1 `tasks/train/streamgaze/build_train_json.py` (신규)

8 task qa JSON → ITVidTrainDataset 포맷 단일 JSON 변환. 핵심:

- `video_path` 가 `OP` prefix → skip (EGTEA).
- `_parse_response_time` 으로 `[MM:SS - MM:SS]` → (start_sec, end_sec).
- `task == 'present_future_action_prediction'` 일 때만 window = `(0.0, rt_start)` (미래 leak 방지), 그 외 `(rt_start, rt_end)`.
- Eval `qa_template` 와 동일 로직으로 question/answer letter+content 재구성.
- `q` 끝에 `\nOnly give the best option.` 추가 (eval `infer_mvbench` 의 `post_query_prompt` 과 정렬).
- `a` = `Best option:({letter}) {content}` (eval 의 `answer_prompt="Best option:("` 와 prefix 일치).

출력: `DATAS/streamgaze_v2/train_noegtea.json`.

### 4.2 `tasks/train/instruction_data.py` (수정)

```python
available_corpus["streamgaze_noegtea"] = [
    [
        "DATAS/streamgaze_v2/train_noegtea.json",
        "/home/yujin/dataset/StreamGaze_v2/videos",
        "video",
    ],
]
```

### 4.3 `tasks/train/config_pllava_streamgaze.py` (신규)

[`config_pllava_nframe.py`](/home/yujin/gaze/PruneVid/tasks/train/config_pllava_nframe.py) 를 참고해 작성한 self-contained config. 핵심 차이:

| 항목 | 값 |
|---|---|
| `train_corpus` | `streamgaze_noegtea` |
| `repo_id` | `MODELS/pllava-7b` |
| `lora_r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 |
| `lora_target_modules` | `[q_proj, k_proj, v_proj, o_proj]` |
| `pooling_shape` | `(16, 12, 12)` (zero-shot 과 동일) |
| `freeze_lm` / `freeze_projector` / `freeze_vision_tower` | True / **False** / True |
| `optimizer.lr` / `weight_decay` / `max_grad_norm` | 1e-4 / 0.0 / 1.0 |
| `scheduler` | cosine, 2 epoch, warmup 0.05 |
| `batch_size` × `grad_accum` | 2 × 8 (eff. 16) |
| `max_txt_l` | 1024 |
| `gradient_checkpointing` | True |
| `preprocess.system` | MVBench system prompt (zero-shot 과 동일) |
| `preprocess.mm_alone` | **False** (USER 한 turn 에 image+question) |
| `preprocess.add_second_msg` | False (auto "Video contains X frames" 비활성) |
| `preprocess.roles` | `['USER: ', 'ASSISTANT:']` (eval role spacing 과 일치) |

PllavaConfig pruning 인자 (`tau=1, cluster_ratio=1, temp_ratio=1, head=8` 등) 도 model dict 에 명시.

### 4.4 `tasks/train/train_pllava_nframe_accel.py` (소수정)

`setup_model()` 의 `PllavaConfig.from_pretrained()` 호출에 `tau/cluster_ratio/temporal_segment_ratio/selected_layer/alpha/head/softmax` 를 config 로부터 명시 전달하도록 1 군데 수정 (원본은 default 의존). `extra_pllava_kwargs` 7 줄 추가.

### 4.5 `tasks/eval/model_utils.py` (수정 — 옵션 A)

`load_pllava()` 시그니처에 `lora_r=128, lora_dropout=0.0, lora_target_modules=("q_proj", "v_proj")` 3 인자 추가, `LoraConfig(...)` 가 위 값 사용. **default 값은 기존 hard-coded 와 동일** → 기존 zero-shot 호출자는 영향 없음.

### 4.6 `tasks/eval/streamgaze/pllava_eval_streamgaze.py` (수정)

argparse 에 `--lora_r`, `--lora_dropout`, `--lora_target_modules` (콤마 구분 string) 추가. `load_pllava(...)` 호출에 위 3 인자 전달.

### 4.7 `tasks/eval/streamgaze/print_csv.py` (신규)

`<save_path>/all_results.json` 을 읽어 사용자 지정 8 task 순서로 콤마 한 줄 + `overall_micro: X.XX` 출력.

순서 고정 (이 문서 §10.2 와 동일):
```
past_gaze_sequence_matching,past_non_fixated_object_identification,past_object_transition_prediction,past_scene_recall,present_object_attribute_recognition,present_object_identification_easy,present_object_identification_hard,present_future_action_prediction
```

(`present_future_action_prediction` 컬럼은 eval dataset 의 `_egtea` 변형 결과를 사용 — 코드는 `present_future_action_prediction_egtea` task_type 에서 읽음.)

---

## 5. Training CLI

### 5.1 Launcher — `scripts/train_streamgaze_lora.sh`

```bash
#!/usr/bin/env bash
# 환경변수:
#   FRAMES   기본 64
#   OUTDIR   기본 test_results/streamgaze_lora/pllava7b_r16_${FRAMES}f
#   NPROC    기본 2 (2×H200)
accelerate launch --config_file ${ACCEL} --num_processes ${NPROC} \
    -m tasks.train.train_pllava_nframe_accel \
    tasks/train/config_pllava_streamgaze.py \
    num_frames ${FRAMES} \
    output_dir ${OUTDIR}
```

### 5.2 명령

```bash
cd /home/yujin/gaze/PruneVid

# (0) 학습 데이터 1회 빌드
python -m tasks.train.streamgaze.build_train_json \
  --qa_dir /home/yujin/dataset/StreamGaze_v2/qa \
  --videos_root /home/yujin/dataset/StreamGaze_v2/videos \
  --out DATAS/streamgaze_v2/train_noegtea.json

# (1) PHASE A: 64-frame 학습
FRAMES=64 OUTDIR=test_results/streamgaze_lora/pllava7b_r16_64f \
    bash scripts/train_streamgaze_lora.sh

# (2) PHASE B: 16-frame 학습 (PHASE A eval 후)
FRAMES=16 OUTDIR=test_results/streamgaze_lora/pllava7b_r16_16f \
    bash scripts/train_streamgaze_lora.sh
```

체크포인트는 `${OUTDIR}/pretrained_epoch{NN}/` 에 매 epoch 저장 (LoRA + projector 만, ~50MB).

### 5.3 Smoke test

config 의 `debug=True` 또는 CLI 로 `max_train_steps 50` 추가 → 50 step 후 자동 break. loss 가 nan/inf 없이 감소하는지만 확인.

---

## 6. Eval CLI

### 6.1 fine-tune adapter 로 EGTEA 526 QA — `scripts/eval_streamgaze_lora.sh`

```bash
python -m tasks.eval.streamgaze.pllava_eval_streamgaze \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --weight_dir ${CKPT} \
    --save_path ${SAVE} \
    --num_frames ${FRAMES} \
    --use_lora \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --pooling_shape 16-12-12 \
    --conv_mode eval_mvbench \
    --tau 1.0 --temporal_segment_ratio 1.0 --cluster_ratio 1.0 \
    --top_p 1.0 --temperature 1.0
```

기존 zero-shot eval pipeline 재사용 — `load_pllava()` 가 `weight_dir` 의 safetensors 에서 LoRA + projector 가중치를 `model.load_state_dict(strict=False)` 로 덮어씀. base PLLaVA-7B + 새 LoRA 어댑터 조합으로 inference.

### 6.2 Smoke

```bash
python -m tasks.eval.streamgaze.pllava_eval_streamgaze \
    ... --max_samples 8
```

8 sample (task 당 1개) 로 letter parsing + accuracy 출력 확인.

### 6.3 결과 콤마 출력

```bash
python -m tasks.eval.streamgaze.print_csv \
    test_results/streamgaze_egtea/lora_r16_64f
```

출력 형식 (zero-shot 16f 예시로 검증 완료):
```
39.06,27.94,50.00,8.11,59.38,75.25,70.31,29.79
overall_micro: 48.29
```

---

## 7. Token-count probe (zero-shot 과 동일)

| Setting | num_frames | pooling_shape | LLM 입력 video tokens |
|---|---:|---|---:|
| LoRA-16f | 16 | 16-12-12 | **2,304** |
| LoRA-64f | 64 | 16-12-12 | **2,304** (4:1 avgpool) |

LLM 입력은 일정 — vision tower 연산량과 frame 다양성만 다름.

---

## 8. 파일 레이아웃

```
PruneVid/
  tasks/train/streamgaze/
    __init__.py                                ← 신규 (빈)
    build_train_json.py                        ← 신규 (qa→ITVid 변환)
  tasks/train/
    instruction_data.py                        ← 수정 (corpus 1줄 추가)
    config_pllava_streamgaze.py                ← 신규 (학습 config)
    train_pllava_nframe_accel.py               ← 수정 (PllavaConfig 에 pruning kwargs 전달)
  tasks/eval/
    model_utils.py                             ← 수정 (load_pllava: LoRA 인자 노출)
  tasks/eval/streamgaze/
    pllava_eval_streamgaze.py                  ← 수정 (LoRA argparse 3 추가)
    print_csv.py                               ← 신규 (8-task 콤마 출력)
  scripts/
    train_streamgaze_lora.sh                   ← 신규
    eval_streamgaze_lora.sh                    ← 신규
  DATAS/streamgaze_v2/
    train_noegtea.json                         ← 생성 (5,799 QA)
  test_results/streamgaze_lora/
    pllava7b_r16_64f/pretrained_epoch{00,01}/  ← 64f 어댑터
    pllava7b_r16_16f/pretrained_epoch{00,01}/  ← 16f 어댑터
  test_results/streamgaze_egtea/
    lora_r16_64f/all_results.json              ← 64f 평가
    lora_r16_16f/all_results.json              ← 16f 평가
```

---

## 9. Key decisions (why)

- **LoRA r=16, α=32 (scale 2.0)**: 사용자 지정 사진 권장값. r=128 (HF release default) 대비 16배 작은 어댑터로도 LoRA-fine-tune 의미가 충분하다는 광범위한 community result. α=2r 이 표준 선택.
- **target = q/k/v/o_proj (LLM attention only)**: q/v 만 attach 하는 경우 대비 K, O 까지 포함하면 attention reorientation 자유도가 커짐. projector/vision tower 는 별도 freeze 정책으로 제어.
- **base LM frozen + projector unfreeze**: 5,799 QA 는 full LoRA + projector unfreeze 에 적합한 규모. base LM 까지 풀면 catastrophic forgetting 위험 + LoRA 의 의미가 약해짐.
- **lr=1e-4, weight_decay=0, cosine 2 epoch, warmup 0.05**: LoRA-only fine-tune 의 community standard. PruneVid 원본 config 의 `lr=2e-5` 는 base LM 까지 푸는 full FT 용이라 부적절.
- **EgoExoLearn + HoloAssist train, EGTEA test**: zero-shot baseline 과 정확히 같은 526 EGTEA QA 위에서 평가 → fine-tune effect 직접 비교. 학습/평가 도메인 분리로 leakage 차단.
- **64f 우선 → 16f**: zero-shot 에서 16f 가 가장 높았지만 64f 가 temporal resolution 에서 더 풍부한 시각 정보를 제공 — projector unfreeze + LoRA 와 결합 시 64f 가 더 큰 수혜를 받을 가능성이 있음. 사용자 요청 우선순위.
- **`mm_alone=False`**: 학습 prompt 가 `USER: <image>\n{q}</s>ASSISTANT: {a}</s>` 형태로 단일 USER turn 에 image + question 결합. `mm_alone=True` 의 double-USER 보다 더 깔끔하고 instruction tuning 패턴에 가까움.
- **`add_second_msg=False`**: ITVidTrainDataset 의 "The video contains X frames sampled at T seconds." auto-msg 는 eval prompt 에 없음. 학습/평가 일관성 위해 비활성.
- **PruneVid pruning OFF (tau=cluster_ratio=temp_ratio=1)**: 본 baseline 의 목적은 vanilla PLLaVA-7B + LoRA. pruning 효과는 후속 실험 (`prunevid_lora.md` — 미작성).

---

## 10. 실행 결과 (학습/평가 후 채움)

### 10.1 Task 별 accuracy (%, EGTEA 526 QA, 4-choice)

| Task | N | ZeroShot-16f | ZeroShot-64f | LoRA-64f | LoRA-16f |
|---|---:|---:|---:|---:|---:|
| past_gaze_sequence_matching | 64 | 39.06 | 31.25 | TBD | TBD |
| past_non_fixated_object_identification | 68 | 27.94 | 29.41 | TBD | TBD |
| past_object_transition_prediction | 2 | 50.00 | 50.00 | TBD | TBD |
| past_scene_recall | 37 | 8.11 | 8.11 | TBD | TBD |
| present_object_attribute_recognition | 96 | 59.38 | 56.25 | TBD | TBD |
| present_object_identification_easy | 101 | 75.25 | 77.23 | TBD | TBD |
| present_object_identification_hard | 64 | 70.31 | 68.75 | TBD | TBD |
| present_future_action_prediction (egtea) | 94 | 29.79 | 27.66 | TBD | TBD |
| **Overall (micro)** | **526** | **48.29** | **46.77** | TBD | TBD |

### 10.2 콤마 출력 — print_csv.py 결과 복붙

순서: past_gaze_sequence_matching, past_non_fixated_object_identification, past_object_transition_prediction, past_scene_recall, present_object_attribute_recognition, present_object_identification_easy, present_object_identification_hard, present_future_action_prediction.

```
ZeroShot-16f: 39.06,27.94,50.00,8.11,59.38,75.25,70.31,29.79  (overall 48.29)
ZeroShot-64f: 31.25,29.41,50.00,8.11,56.25,77.23,68.75,27.66  (overall 46.77)
LoRA-64f:     TBD
LoRA-16f:     TBD
```

### 10.3 관측 (eval 후 작성)

- 도메인 격차 (EgoExoLearn lab/sport, HoloAssist instruction → EGTEA cooking) 의 영향
- task 별 차등 효과 — `past_scene_recall` (zero-shot 8.11%) 의 대폭 향상 여부
- 16f vs 64f 의 fine-tune 후 격차

### 10.4 이슈 / patch 로그 (smoke test 에서 발견·해결)

학습을 처음 돌릴 때 발견한 **4 가지 fork-수준 issue + 1 PEFT/checkpointing issue**, 각 fix 위치:

1. **Right-padded merge 의 zero-embed 충돌** — [models/pllava/modeling_pllava.py:686-693](/home/yujin/gaze/PruneVid/models/pllava/modeling_pllava.py#L686-L693)
   - 원인: `_merge_input_ids_with_image_features` 가 `final_embedding == 0` 으로 image-fill 위치를 찾지만 Llama-2 의 `pad_token_id=0` (`<unk>`) 의 임베딩이 거의 0 이라 right-padding 의 pad slot 들이 image-fill 로 오인식됨. eval 은 padding 없어 영향 무, 학습은 `image_to_overwrite.sum() != image_features.shape[:-1].numel()` 로 즉시 실패.
   - 수정: zero-row heuristic 을 **text 위치 보집합** 계산으로 교체.
     ```python
     image_to_overwrite = torch.ones((batch_size, max_embed_dim), dtype=torch.bool, device=target_device)
     image_to_overwrite[batch_indices, text_to_overwrite] = False
     ```

2. **LLM-VTP elastic_cache 가 right-padding 과 충돌** — config 의 `selected_layer=99` 로 비활성화
   - 원인: [models/pllava/elastic_cache.py:159](/home/yujin/gaze/PruneVid/models/pllava/elastic_cache.py#L159) 의 `obtain_language_attention` 이 image 영역을 `input_ids == pad_token` 로 식별하는데, post-merge input_ids 에는 image-position pad 외에 right-padding pad 도 있어 `img_start..img_end` 가 padding 끝까지 잡혀 text 영역이 비고 `text_to_image_attentions[0].max(dim=0)` 가 zero-size dim 에서 IndexError.
   - 수정: 학습 config 에 `selected_layer=99` 설정 → 32-layer Llama 의 layer index 와 매칭 안 되어 VTP code path 진입 자체 차단. tau/cluster_ratio/temp_ratio=1 과 함께 PruneVid pruning 이 모두 OFF.

3. **PLLaVA-7B safetensors LoRA prefix mismatch** — config 의 `pretrained_path="MODELS/pllava-7b"` + load 시 shape-mismatch 필터
   - 원인: HF release 의 safetensors 가 `language_model.base_model.model.*` 키로 저장됨 (원래 LoRA 학습 후 save 한 상태). `from_pretrained()` 는 깨끗한 모델의 `language_model.model.*` 에 매칭 시도 → 모든 LLM weight 매칭 실패 → 무작위 초기화 → logits 전부 0 → loss=log(V)≈10.376 으로 stuck.
   - 수정 (2 단계):
     - config 에 `pretrained_path="MODELS/pllava-7b"` 추가 — [setup_model](/home/yujin/gaze/PruneVid/tasks/train/train_pllava_nframe_accel.py#L163) 의 LoRA 후 manual safetensors load 가 동작.
     - 위 manual load 가 base safetensors 의 r=128 LoRA 와 우리 r=16 LoRA 사이 shape 충돌로 raise 하던 것을 방어 — [setup_model](/home/yujin/gaze/PruneVid/tasks/train/train_pllava_nframe_accel.py#L186-L200) 에 shape-mismatch key 필터 추가 (기본/projector weight 만 로드, LoRA 는 fresh init).

4. **PEFT + gradient_checkpointing → loss flat at log(V)** — [setup_model](/home/yujin/gaze/PruneVid/tasks/train/train_pllava_nframe_accel.py#L162) 에 `enable_input_require_grads` 추가
   - 원인: LoRA 만 trainable 일 때 input embedding 출력에 `requires_grad=True` 인 입력이 없어 `torch.utils.checkpoint` 가 silent 하게 grad 를 drop. (smoke 에서 이 케이스가 직접 영향 주진 않았지만 — 위 #3 이 dominant — 64f 학습 시 메모리 절감 위해 필요한 경우 대비 안전장치.)
   - 수정: LoRA 어태치 직후 `model.language_model.enable_input_require_grads()`.

5. **PllavaConfig pruning kwargs 명시 전달** — [setup_model](/home/yujin/gaze/PruneVid/tasks/train/train_pllava_nframe_accel.py#L101-L114)
   - PllavaConfig.from_pretrained 호출에 `tau/cluster_ratio/temporal_segment_ratio/selected_layer/alpha/head/softmax` 인자가 빠져 있어 config 의 값이 무시될 가능성. config.model dict 에서 위 키들을 읽어 `extra_pllava_kwargs` 로 전달하도록 7 줄 추가.

**Smoke verify (16f, batch=1, lr=1e-4, 단일 GPU)**: loss progression 10.30 → 10.20 (step 18) → 9.05 (step 36) → 7.59 (step 51) → 3.75 (step 75) → 1.14 (step 105) → 0.65 (step 153). 단일 sample overfit 속도이지만 gradient 가 정상 흐름을 정량 확인.

---

## 11. References

- 같은 폴더 [pllava_zeroshot.md](./pllava_zeroshot.md) — 비교 baseline
- PLLaVA: [arXiv:2404.16994](https://arxiv.org/abs/2404.16994)
- PEFT/LoRA: Hu et al. 2021, target_modules q/k/v/o 는 [QLoRA](https://arxiv.org/abs/2305.14314) §3 권장
- StreamGaze_v2 데이터: `/home/yujin/dataset/StreamGaze_v2/README.md`
