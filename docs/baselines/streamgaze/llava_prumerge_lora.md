# LLaVA-PruMerge × Video-LLaVA — StreamGaze LoRA 파이프라인

**구현 완료 + end-to-end 학습/평가 수행.** LLaVA-PruMerge 논문 (arXiv:2403.15388) 의 **Section 4.1-4.2 메인 파이프라인** (PruMerge+ 가 학습+추론 모두 활성) 을 Video-LLaVA-7B 에 이식하고 StreamGaze_v2 egtea split 에서 LoRA fine-tune 평가.

```
 StreamGaze_v2/videos/{egoexolearn,holoassist}/original/*.mp4   (train)
 StreamGaze_v2/videos/egtea/original/*.mp4                      (test)
     │
     ▼
 decord VideoReader, windowed sampling
   task별 window 정책:
     past_*                        → [resp_start, t_q]
     present_future_action_pred_*  → [0, resp_start]   ← 미래 leak 방지
     기타 present_*                → [resp_start, min(resp_end, t_q+2)]
   32 frames uniform
     │
     ▼
 LanguageBindVideoProcessor.transform  (CenterCrop 224, OpenAI norm)
     │   [C=3, T=32, H=224, W=224]
     ▼
 LanguageBindVideoTowerPruMerge  (subclass of LanguageBindVideoTower)
   ① adapt_num_frames(8→32): temporal_embedding linear-interp [1,8,D]→[1,32,D] on 24 layers
   ② hook layer 23 self_attn.{k,q}_proj  →  per-frame CLS attention → IQR outlier ratio
   ③ top-k patches + spatial supplement (PruMerge+ §3)
   ④ cosine-sim cluster merge on Key vectors
   ⑤ 1 merged token (attention-weighted sum of discarded)
     │   [1, 32, ~149, 1024]   (full 2048 → ~4768 tokens per clip, ratio capped to 1/8)
     ▼
 mm_projector (mlp2x_gelu, frozen except `mm_projector_lr`)
     │   [1, 32, ~149, LLM_hidden]
     ▼
 splice into <image>*32 placeholders in input_ids embedding
     │
     ▼
 Vicuna-7B-v1.5 (LoRA on LLM self_attn.{q,k,v,o}_proj only, r=16 α=32)
   dynamic NTK RoPE scaling factor=2.0 (4K→8K context)
     │
     ▼
 cross-entropy on assistant letter only ("A" / "B" / "C" / "D")
```

---

## 1. Environment

Conda env `videollava` (Python 3.10, CUDA 11.8):

```bash
conda create -n videollava python=3.10 -y
/opt/conda/envs/videollava/bin/pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
cd /home/yujin/gaze/trajgaze/Video-LLaVA
/opt/conda/envs/videollava/bin/pip install -e .
/opt/conda/envs/videollava/bin/pip install decord pytorchvideo \
  "opencv-python-headless==4.9.0.80" "numpy<2"
/opt/conda/envs/videollava/bin/pip install deepspeed==0.9.5
/opt/conda/envs/videollava/bin/pip uninstall -y bitsandbytes   # CUDA libcusparse mismatch 회피
```

주요 핀: `transformers==4.31.0`, `peft==0.4.0`, `torch==2.0.1+cu118`. `opencv-python` 대신 `-headless` 빌드 (libGL 없음).

GPU: 2 × H200 (143GB each). 학습/평가 모두 bf16, DeepSpeed ZeRO-2.

---

## 2. 레포 클론

```bash
git clone https://github.com/PKU-YuanGroup/Video-LLaVA \
  /home/yujin/gaze/trajgaze/Video-LLaVA
```

추가로 **LLaVA-PruMerge** (`/home/yujin/gaze/trajgaze/LLaVA-PruMerge`) 는 `clip_encoder.py` 의 PruMerge+ 알고리즘 **참조용**. 포팅된 코드는 Video-LLaVA 쪽에 저장.

HF 가중치 (다운로드됨, `~/.cache/huggingface/hub`):
- `LanguageBind/Video-LLaVA-7B` — Vicuna-7B + mm_projector + vision tower ref
- `LanguageBind/LanguageBind_Video_merge` — 비디오 tower (LanguageBind CLIP-ViT-L/14 기반)
- `LanguageBind/LanguageBind_Image` — 이미지 tower

---

## 3. Data assets

### 3.1 StreamGaze_v2 QA (read-only)

`/home/yujin/dataset/StreamGaze_v2/qa/*.json` — 9 non-proactive task JSON:
- `past_{gaze_sequence_matching, non_fixated_object_identification, object_transition_prediction, scene_recall}.json`
- `present_future_action_prediction{, _egtea}.json`
- `present_object_{attribute_recognition, identification_easy, identification_hard}.json`

각 QA: `response_time = "[MM:SS - MM:SS]"`, `questions[]` 에 `question / options(4) / answer(letter) / time_stamp` 및 태스크별 필드.

### 3.2 Videos

`/home/yujin/dataset/StreamGaze_v2/videos/{source}/original/*.mp4` — .tar.gz 이미 추출됨
- egtea: `OP##-R##-<dish>.mp4` (35 개)
- egoexolearn: UUID .mp4 (180 개)
- holoassist: `P##-R##-<dish>.mp4` (66 개)

### 3.3 변환된 학습/평가 JSON (자동 생성)

- `data/streamgaze/egtea_eval.json` — 590 records (test)
- `data/streamgaze/train.json` — 5,783 records (egoexolearn 3,927 + holoassist 1,856)

각 record:
```json
{
  "id": "<task>__<video>__<ts>__<qhash>",
  "task": "<source json stem>",
  "video": "egoexolearn/original/<uuid>.mp4",
  "start_sec": 142.0,
  "end_sec": 160.0,
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion: ...\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer with only the letter of the correct option."},
    {"from": "gpt", "value": "C"}
  ]
}
```

**dedup 키**: `(task, video_path, time_stamp, md5(question)[:8])`. easy/hard 가 같은 질문 텍스트를 공유하므로 task 를 포함.

---

## 4. 구현 방식

### 4.1 PruMerge+ 포팅 — `videollava/model/multimodal_encoder/prumerge_video.py`

원본 [LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py](/home/yujin/gaze/trajgaze/LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py) 의 `token_prune_merge_advanced_plus` (L142-281) 를 per-frame 버전으로 재구성.

**핵심 함수** `prumerge_plus_per_frame(patch_features, desired_k, desired_q, ...)`:
1. `cls_attn = softmax(Q @ K^T / √d)[:, 0, 1:]` — CLS row (각 프레임 독립)
2. `reduction_ratio = outlier_IQR(cls_attn)` — 상한 1/8 에 clamp (pathological uniform attention 방어)
3. `topk(cls_attn, k=int(N * ratio))` → `idx`
4. `arange(0, N_patch, max(1, step//3))` 공간 공급 후 dedup → `idx` 에 append
5. 남은 (non-topk) 토큰에 대해 **key cosine similarity** 로 각 kept 토큰에 cluster 32 개 매칭 → weighted sum 후 `x_others[i] += weighted_avg`
6. `extra_one_token = Σ non_topk * non_topk_attn` → 뒤에 concat
7. 반환 `[B*T, N', C]`, 통상 N' ≈ 65–150 per frame

**Tower 래퍼** `LanguageBindVideoTowerPruMerge(LanguageBindVideoTower)`:
- `_forward_one(videos [B,C,T,H,W])`:
  - `register_forward_hook(layer23.self_attn.k_proj)` / `q_proj` — `_HOOK_CACHE` 에 저장
  - 기존 video_tower forward 실행 → `hidden_states[self.select_layer=-2]` shape `[B,T,N,C]`
  - hook 된 K/Q (`[B*T, N, C]`) 를 꺼내 patch_features (CLS 제거) 와 함께 `prumerge_plus_per_frame` 호출
  - reshape back `[B, T, N', C]`
- `@torch.no_grad()` 유지 — vision tower 는 어차피 frozen; mm_projector 는 본인 파라미터로 gradient 받음
- Eval/train 모두 동일 클래스 swap 으로 활성화:
  ```python
  vt = model.get_video_tower()
  vt.__class__ = LanguageBindVideoTowerPruMerge
  vt.prumerge_layer = 23; vt.prumerge_ratio = 1/8; vt.prumerge_adaptive = True
  ```

### 4.2 32프레임 adapt — `videollava/model/multimodal_encoder/temporal_adapt.py`

`adapt_num_frames(model, new_num_frames=32)`:
- `video_tower.config.num_frames = 32`, `video_processor.config.vision_config.num_frames = 32`
- 24 CLIPEncoderLayer 의 `temporal_embedding` (shape `[1, 8, 1024]`) 를 `F.interpolate(mode="linear", size=32)` 로 resize → `[1, 32, 1024]`, `.contiguous()` 강제 (DeepSpeed broadcast 호환)
- 각 layer 의 `self.t = 32` 업데이트

`apply_rope_scaling(model, factor=2.0, type_="dynamic")`:
- `model.config.rope_scaling = {"type": "dynamic", "factor": 2.0}`
- `max_position_embeddings = 4096 → 8192`
- 32 LlamaAttention 의 `rotary_emb` 을 `LlamaDynamicNTKScalingRotaryEmbedding` 으로 in-place 재구성 (head_dim, base=10000, new max_pos)

두 함수는 train.py / eval_streamgaze.py 모델 로드 직후 호출.

### 4.3 LazySupervisedDataset 확장 — `videollava/train/train.py` 패치

- 모듈 최상위 helper `_load_windowed_video(video_path, transform, start_sec, end_sec, num_frames)`:
  decord `VideoReader` + fps 기반 index → `np.linspace(s, e, num_frames)` → processor transform
- `__getitem__` 의 video-only 분기에서 record 에 `start_sec`/`end_sec` 존재 시 helper 사용, 아니면 기존 `video_processor(path)` fallback

### 4.4 LLM-only LoRA target — `train.py`

첨부된 LoRA 세팅 그대로 (r=16, α=32, dropout=0.05, target_modules = LLM qkvo only):

```python
if training_args.lora_llm_only:
    llm_targets = sorted({
        n for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and ".self_attn." in n
        and n.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))
        and all(s not in n for s in ("vision", "video", "image", "mm_projector"))
    })
```

32 layer × 4 = **128 target modules** 선택 확인.

### 4.5 학습 flow (train.py 수정 지점)

```
1. LlavaLlamaForCausalLM.from_pretrained(Video-LLaVA-7B)
2. LoRA wrap (llm_only targets)            ← peft.get_peft_model
3. initialize_vision_modules()             ← vision tower 로드 + processor
4. adapt_num_frames(model, 32)             ← NEW
5. apply_rope_scaling(model, 2.0)          ← NEW
6. swap video_tower.__class__ = LanguageBindVideoTowerPruMerge   ← NEW
7. LLaVATrainer.train() — 5783 samples, 1 epoch, 361 steps
```

---

## 5. Train — CLI 및 스크립트

### 5.1 변환 스크립트

```bash
cd /home/yujin/gaze/trajgaze/Video-LLaVA

# test split (egtea)
/opt/conda/envs/videollava/bin/python scripts/convert_streamgaze_egtea.py
# → data/streamgaze/egtea_eval.json (590 recs)

# train split (egoexolearn + holoassist)
/opt/conda/envs/videollava/bin/python scripts/convert_streamgaze_train.py
# → data/streamgaze/train.json (5783 recs)
```

### 5.2 학습 launcher — `scripts/streamgaze_train_lora.sh`

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

/opt/conda/envs/videollava/bin/deepspeed \
  --include "localhost:${CUDA_VISIBLE_DEVICES}" \
  videollava/train/train.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path LanguageBind/Video-LLaVA-7B \
  --version v1 \
  --data_path ./data/streamgaze/train.json \
  --video_folder /home/yujin/dataset/StreamGaze_v2/videos \
  --video_tower LanguageBind/LanguageBind_Video_merge \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir ./checkpoints/videollava-streamgaze-lora-prumerge-32f \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" --save_steps 250 --save_total_limit 2 \
  --learning_rate 2e-4 --mm_projector_lr 2e-5 \
  --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" \
  --logging_steps 1 --tf32 False \
  --model_max_length 8192 --tokenizer_model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 --lazy_preprocess True \
  --report_to none \
  --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_bias none \
  --lora_llm_only True \
  --streamgaze_num_frames 32 \
  --enable_prumerge True --prumerge_layer 23 --prumerge_ratio 0.125 --prumerge_adaptive True \
  --rope_scaling_factor 2.0 \
  "$@"
```

실행:
```bash
bash scripts/streamgaze_train_lora.sh > /tmp/train.log 2>&1
```
- **step 시간**: ~25s (grad_accum 8 기준 = 16 samples/step)
- **전체**: 361 steps × ~25s ≈ **2시간 30분**
- **로그 체크포인트**: step 250 + 최종 → `checkpoints/videollava-streamgaze-lora-prumerge-32f/{adapter_model.bin, adapter_config.json, non_lora_trainables.bin, config.json}`

### 5.3 학습 관측

초기 셋업 로그 (성공 신호):
```
[LoRA] LLM-only target_modules count=128 (first 3: ['model.layers.0.self_attn.k_proj', ...])
[adapt_num_frames] 8 -> 32; resized 24 temporal_embedding tensors
[apply_rope_scaling] rope_scaling={'type': 'dynamic', 'factor': 2.0}, max_position_embeddings=8192, rebuilt rotary_emb on 32 LlamaAttention modules
[PruMerge+] train-time swap: class=LanguageBindVideoTowerPruMerge, layer=23, ratio=0.125, adaptive=True
```

loss: 5.4 (step 0, warmup) → 0.7 전후 (step 20+), 최종 epoch 평균 **0.74**.

---

## 6. Eval — CLI 및 체크포인트 로딩

### 6.1 Eval launcher — `videollava/eval/streamgaze/eval_streamgaze.py`

LoRA 포함 평가 (PruMerge ON, 메인):
```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/videollava/bin/python \
  -m videollava.eval.streamgaze.eval_streamgaze \
  --model_path LanguageBind/Video-LLaVA-7B \
  --lora_path ./checkpoints/videollava-streamgaze-lora-prumerge-32f \
  --data_path data/streamgaze/egtea_eval.json \
  --output_json results/egtea_lora32_prumerge_on.json \
  --use_prumerge 1 \
  --num_frames 32 \
  --rope_scaling_factor 2.0 \
  --device cuda:0
```

PruMerge OFF 비교 (디버그 — 32f OFF 는 context 초과로 실패, 아래 7-섹션 참조):
```bash
... --use_prumerge 0 --output_json results/egtea_lora32_prumerge_off.json
```

Training-free (LoRA 없음) baseline:
```bash
/opt/conda/envs/videollava/bin/python \
  -m videollava.eval.streamgaze.eval_streamgaze \
  --model_path LanguageBind/Video-LLaVA-7B \
  --data_path data/streamgaze/egtea_eval.json \
  --output_json results/egtea_prumerge_on.json \
  --use_prumerge 1 \
  --num_frames 8
```

### 6.2 LoRA 로딩 경로

`eval_streamgaze.py` 는 `args.lora_path` 지정 시 `load_pretrained_model(lora_path, model_base=model_path, model_name="llava-lora-…")` 로 분기 (videollava/model/builder.py L48-L82):
1. `AutoConfig.from_pretrained(lora_path)` 로 LoRA 설정 로드
2. `LlavaLlamaForCausalLM.from_pretrained(model_path)` — base 체크포인트
3. `non_lora_trainables.bin` 로드 (mm_projector, embed_tokens 등)
4. `PeftModel.from_pretrained(model, lora_path).merge_and_unload()` — LoRA merge

merge 된 모델에 eval 쪽에서 다시 `adapt_num_frames(32)` + `apply_rope_scaling(2.0)` + PruMerge swap 재호출.

### 6.3 Generation

- `conv_templates["llava_v1"]` + `<image>*32` 선행 토큰 → user prompt (question + options + "Answer with only the letter...")
- `model.generate(do_sample=False, temperature=0.0, max_new_tokens=8, stopping_criteria=[KeywordsStoppingCriteria([stop_str], ...)])`
- `re.search(r"[ABCD]", pred_text)` 로 letter 파싱 — fallback 안전망

### 6.4 Token-count probe

eval 첫 샘플에서 video_tower 출력 shape 로깅:
- `--use_prumerge 0 --num_frames 32` → `(1, 32, 257, 1024)` = **8,224 tokens**
- `--use_prumerge 1 --num_frames 32` → `(1, 32, ~150, 1024)` ≈ **~4,800 tokens**

---

## 7. Trainable surface

| 모듈 | Train-time | Note |
| --- | --- | --- |
| `video_tower` (LanguageBind vision) | FROZEN | `requires_grad_(False)` 유지 |
| `video_tower` 내부 `temporal_embedding` | FROZEN (interp 된 값 그대로) | resize 만 하고 학습 안 함 |
| `mm_projector` (mlp2x_gelu) | Full-tune, lr=2e-5 | LoRA 대상 아님 |
| `llm.model.layers[i].self_attn.{q,k,v,o}_proj` | LoRA r=16 α=32 dropout=0.05 | **128개 (32×4)** |
| `llm.model.layers[i].mlp.*` | FROZEN | target modules 에 없음 |
| `llm.embed_tokens`, `lm_head` | FROZEN | non_lora_trainables 에 저장은 안 됨 |

---

## 8. 파일 레이아웃

```
Video-LLaVA/
  videollava/
    model/multimodal_encoder/
      prumerge_video.py          ← 신규 (PruMerge+ per-frame wrapper)
      temporal_adapt.py          ← 신규 (num_frames / rope_scaling helpers)
    train/train.py               ← 수정 (windowed loader, LoRA LLM-only, adapt/PruMerge swap hooks, new TrainingArguments fields)
    eval/streamgaze/
      eval_streamgaze.py         ← 신규 (letter-only MC eval w/ --lora_path --num_frames --rope_scaling_factor)
  scripts/
    convert_streamgaze_egtea.py        ← 신규 (test 변환)
    convert_streamgaze_train.py        ← 신규 (train 변환)
    streamgaze_train_lora.sh           ← 신규 (deepspeed launcher)
    compare_prumerge_results.py        ← 신규 (결과 비교)
  data/streamgaze/
    egtea_eval.json              ← 자동 생성 (590 recs)
    train.json                   ← 자동 생성 (5783 recs)
  checkpoints/
    videollava-streamgaze-lora-prumerge-32f/
      adapter_model.bin          (33 MB, LoRA)
      adapter_config.json
      non_lora_trainables.bin    (42 MB, mm_projector + aux)
      config.json
      checkpoint-250/            (mid-run 저장, step 250)
  results/
    egtea_prumerge_{off,on}.json            ← training-free 8f baseline
    egtea_lora32_prumerge_{off,on}.json     ← LoRA 32f eval
```

---

## 9. Key decisions (why)

- **Video-LLaVA 유지** (Qwen2.5-VL 등 다른 base 전환 안 함): CLIP-based LanguageBind vision tower 가 LLaVA-PruMerge 원본 훅 위치 (`encoder.layers[23].self_attn.{k,q}_proj`) 를 거의 그대로 지원. Qwen2.5-VL 은 CLS 토큰 부재 + fused QKV 라 non-trivial port 필요.
- **Per-frame PruMerge**: `[B·T, 257, C]` 펼쳐 각 프레임 독립 적용 → 논문의 single-image regime 과 동일. Cross-frame pruning 은 CLS proxy 재설계 필요해 보류.
- **학습+추론 PruMerge ON** (Section 4.1-4.2 방식): training-free (§4.3) 가 아닌 메인 파이프라인. Forward `@torch.no_grad()` 로 vision tower 는 어차피 frozen 이라 gradient 흐름 문제 없음 (mm_projector 는 trainable).
- **32 프레임**: temporal resolution 향상. Video-LLaVA 는 8f 학습이라 OOD 이지만 `temporal_embedding` linear interp + RoPE dynamic ×2 로 확장. 4f/16f 대안도 가능.
- **LoRA r=16/α=32/dropout=0.05, LLM qkvo only**: 사용자 지정 세팅. Video-LLaVA 원래 finetune_lora.sh (r=128/α=256, all linears) 보다 훨씬 보수적. Vision tower, mm_projector, LLM FFN 은 LoRA 대상 아님.
- **`--pretrain_mm_mlp_adapter` 생략**: 완제 Video-LLaVA-7B 는 projector 내장 → 추가 pretrain 불필요.
- **future-prediction 윈도우 `[0, resp_start]`**: 질문 시점 이전 비디오만 관찰 → 미래 leak 방지 (past_* 도 동일 원칙).
- **model_max_length=8192**: 32f × ~150 PruMerge tokens ≈ 4,800 + prompt ≈ 5K < 8K. OFF (8224 tokens) 는 초과해 generation 실패 (아래 관찰 참조).
- **bnb 제거**: Video-LLaVA pyproject 에 `bitsandbytes==0.41.0` 이지만 이 환경의 CUDA lib 와 호환 안됨. eval/train 모두 non-quantized 라 bnb 없이 문제 없음.

---

## 10. 실행 결과 (egtea 590 QA)

| 설정 | Overall acc | Notes |
|---|---|---|
| **Training-free 8f OFF** (baseline) | 44.41% | Video-LLaVA 원본, 2,056 tokens/clip |
| **Training-free 8f ON** (§4.3 방식) | 45.25% | PruMerge inference-only, 1,192 tokens/clip |
| **LoRA 32f ON** (§4.1-4.2 메인, train.json) | **37.80%** | letter-shortcut 학습으로 base 성능 하회 |
| LoRA 32f OFF (디버그) | 2.03% | **context 초과로 generation 빈 문자열** |
| **LoRA 32f ON (train_shuffled.json)** | TBD | 재학습 진행 예정, §10.1 bias mitigation |

### Task별 accuracy (%) — 요청 순서 기준

| Task | TF-8f ON | LoRA-32f ON | Δ |
|---|---|---|---|
| past_gaze_sequence_matching | 47.06 | 32.35 | −14.71 |
| past_non_fixated_object_identification | 36.76 | 26.47 | −10.29 |
| past_object_transition_prediction | 0.00 | 0.00 | ±0 |
| past_scene_recall | 27.03 | 32.43 | +5.40 |
| present_object_attribute_recognition | 59.38 | 63.54 | +4.17 |
| present_object_identification_easy | 66.34 | 37.62 | −28.72 |
| present_object_identification_hard | 51.56 | 34.38 | −17.19 |
| present_future_action_prediction | 35.11 | 35.11 | ±0 |
| **Simple mean (8 tasks)** | **40.42** | **32.74** | **−7.67** |

### 10.1 Letter-shortcut bias 진단 + mitigation (option shuffle)

**문제**: 초기 `train.json` 의 gold letter 분포가 task별로 심하게 편향 → LoRA 가 내용 대신 letter shortcut 학습.

| set | A | B | C | D |
|---|---|---|---|---|
| train.json 전체 (n=5,783) | 19.6% | 15.1% | 31.4% | 33.9% |
| train.json `present_object_attribute_recognition` (n=1,305) | 4% | **0%** | 41% | 54% |
| egtea_eval.json gold (reference) | 20% | 15% | 33% | 32% |

LoRA-32f ON 의 eval prediction 분포: **A=2%, B=6%, C=45%, D=47%** (vs TF-8f ON A=28%, B=16%, C=23%, D=33%). 특히 `present_object_attribute_recognition` 에서는 96개 중 A=0, B=0 (D=78, C=18) 로 완전한 C/D-only collapse. `present_object_identification_easy` 가 66→38 로 급락한 것도 같은 이유 (base 모델의 A/B 응답 prior 를 LoRA 가 덮어씀).

**Mitigation**: [scripts/convert_streamgaze_train.py](Video-LLaVA/scripts/convert_streamgaze_train.py) 에 `permute_options` 추가 — md5(`{seed}|{task}|{vp}|{time_stamp}|{qhash}`) 기반 deterministic per-record 셔플로 4개 옵션 순서 + gold letter 를 재배열. 결과:

| set | A | B | C | D |
|---|---|---|---|---|
| train_shuffled.json 전체 | 25.3% | 25.5% | 24.8% | 24.4% |
| train_shuffled `present_object_attribute_recognition` | 25% | 26% | 28% | 22% |

모든 task 가 20–30% 범위로 균일화됨. eval JSON 은 변경 없음 (benchmark 정의 유지).

**재학습 명령** (기존 체크포인트 보존):
```bash
cd /home/yujin/gaze/trajgaze/Video-LLaVA
bash scripts/streamgaze_train_lora.sh > /tmp/train_shuf.log 2>&1
# → checkpoints/videollava-streamgaze-lora-prumerge-32f-shuf/

/opt/conda/envs/videollava/bin/python -m videollava.eval.streamgaze.eval_streamgaze \
    --model_path LanguageBind/Video-LLaVA-7B \
    --lora_path ./checkpoints/videollava-streamgaze-lora-prumerge-32f-shuf \
    --data_path data/streamgaze/egtea_eval.json \
    --output_json results/egtea_lora32_prumerge_on_shuf.json \
    --use_prumerge 1 --num_frames 32 --rope_scaling_factor 2.0
```

---

## 11. Reproduce quick reference

```bash
# 0) env + clone (한번만)
conda create -n videollava python=3.10 -y && ...

# 1) 데이터 변환
cd /home/yujin/gaze/trajgaze/Video-LLaVA
/opt/conda/envs/videollava/bin/python scripts/convert_streamgaze_egtea.py
/opt/conda/envs/videollava/bin/python scripts/convert_streamgaze_train.py

# 2) 학습 (2h30m on 2×H200)
bash scripts/streamgaze_train_lora.sh > /tmp/train.log 2>&1

# 3) 평가 (약 15분)
/opt/conda/envs/videollava/bin/python -m videollava.eval.streamgaze.eval_streamgaze \
  --model_path LanguageBind/Video-LLaVA-7B \
  --lora_path ./checkpoints/videollava-streamgaze-lora-prumerge-32f \
  --data_path data/streamgaze/egtea_eval.json \
  --output_json results/egtea_lora32_prumerge_on.json \
  --use_prumerge 1 --num_frames 32 --rope_scaling_factor 2.0

# 4) 결과 비교
/opt/conda/envs/videollava/bin/python scripts/compare_prumerge_results.py \
  --off results/egtea_prumerge_off.json \
  --on  results/egtea_prumerge_on.json
```

---

## 12. References

- LLaVA-PruMerge 논문: [arXiv:2403.15388](https://arxiv.org/abs/2403.15388) (Shang et al., 2024)
- 원본 image PruMerge 구현: [LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py](/home/yujin/gaze/trajgaze/LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py) — `token_prune_merge_advanced_plus`
- Video-LLaVA: [github.com/PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) (Lin et al., 2023) — LanguageBind video tower + Vicuna-7B-v1.5
- StreamGaze_v2 데이터: `/home/yujin/dataset/StreamGaze_v2/README.md`
