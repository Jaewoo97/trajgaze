# Path C — NVILA-Lite-8B + AutoGaze SigLIP runtime swap + LoRA SFT

**구현 완료** (smoke 검증 진행 중). Path B 의 출발 가중치 (`nvidia/NVILA-8B-HD-Video`, 이미 Stage 5 Patch-Selection-Tuning 반영) 대신 **`Efficient-Large-Model/NVILA-Lite-8B`** (gaze-aware 아님) 에서 시작, 런타임에 inner SigLIP 을 AutoGaze SigLIP 으로 교체 후 LoRA SFT.

```
 JPG frames (trajgaze ordering)         user .pt (gazing_info)
     │                                        │
     ▼                                        ▼
 AutoGazeImageProcessor                slice_gazing_info (K-frame remap, NV=1360)
     │                                        │
     └──────────────┬─────────────────────────┘
                    ▼
   vision_tower (AutoGaze SigLIP runtime swap, scales = "56+112+224+448")
     │   * 상위 N 블록 UNFREEZE (Path B 와 차이)
     │   hidden_states[-2]
     ▼
     per-frame unpad (~if_padded_gazing) → pad to multiple of 9
                    ▼
     mm_projector (mlp_downsample_3x3_fix, full-tune)
         (1, 9M, C) → (M, 9C) 로 직접 grouping 후 DownSample3x3BlockFix 의
         초기 reshape 스킵하고 후속 LayerNorm+MLP 적용
                    ▼
     splice into <vila/video> positions in input_ids embedding
         * tokenizer 에 add_special_tokens + resize_token_embeddings 필요
                    ▼
     Qwen2 LLM
       Phase 1: frozen (vision_tower + projector warm-up)
       Phase 2: LoRA 활성 (r=32, α=64)
                    ▼
     cross-entropy on assistant letter only
```

---

## 1. Environment

- Path B 와 동일 (conda env `trajgaze`, 환경 변수 등).
- 추가 의존성: **`deepspeed`** (VILA `llava_llama.py` → `sequence_parallel.globals` 의 하드 임포트 체인). 설치됨 (`deepspeed-0.18.9`).
- Lite-8B 가중치: 이미 다운로드 완료.
  `/home/irteam/.cache/huggingface/hub/models--Efficient-Large-Model--NVILA-Lite-8B/snapshots/ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d/`
  구조: `config.json`, `llm/`, `mm_projector/`, `vision_tower/`, `trainer_state.json`.

---

## 2. Data assets (read-only) — Path B 와 공용

변경 없음. `vila_hd_work/data/{train,test}.jsonl`, `nvila_perframe/gazing_info/*.pt`, `all_gaze_v1/<dataset>/no_gaze/<video_id>/*.jpg` 그대로 재사용.

---

## 3. Pre-processed artefacts — 재생성 불필요

Path B 와 동일 splits.json / jsonl 공용. scales 및 NV=1360 도 공용.

---

## 4. 구현 방식

### (1) Lite-8B 로딩

`trust_remote_code` 기반 HF 모델이 아니라 **VILA source tree 의 `LlavaLlamaModel`** 을 사용. `train_path_c.py` 가 `sys.path.insert(0, "/home/yujin/gaze/trajgaze/VILA")` 로 VILA 를 import 경로에 추가 후:

```python
from llava.model.language_model.llava_llama import LlavaLlamaModel
model = LlavaLlamaModel.from_pretrained(
    snapshot_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="sdpa",
)
```

### (2) SigLIP → AutoGaze SigLIP 런타임 swap

**옵션: runtime swap**. snapshot 은 불변. `train_path_c.py :: _swap_in_autogaze_siglip(model, scales, dtype)`:

- 기존 `model.vision_tower.vision_tower` (HF `SiglipVisionModel`) 의 config 를 읽어 AutoGaze `SiglipVisionConfig` 로 변환 (`scales="56+112+224+448"`, `attn_type="block_causal"`).
- `AGSiglipVisionModel(ag_cfg)` 새로 생성, dtype/device 정렬 후 `load_state_dict(src, strict=False)` — AutoGaze SigLIP 은 HF SigLIP 의 최소 fork 라 key 호환.
- 로그 `[siglip-swap] N unexpected keys ignored`, `[siglip-swap] M missing keys` 로 mismatch 모니터링 (정상 시 둘 다 0 이 이상적, 소수 (e.g. head) 는 AutoGaze 측 확장분으로 허용).
- `model.vision_tower.vision_tower = new` 로 swap 후 outer wrapper 의 `feature_select` 등은 그대로 둠.

### (3) Tokenizer 에 splice 토큰 등록

Lite-8B 의 저장된 Qwen2 tokenizer 에는 `<vila/video>` 도 `<image>` 도 단일 id 로 존재하지 않음 (sub-piece 로 쪼개짐). `build_model_and_tokenizer` 에서:

```python
if tokenizer.convert_tokens_to_ids(VIDEO_TOKEN) is None:
    tokenizer.add_special_tokens({"additional_special_tokens": [VIDEO_TOKEN]})
    model.llm.resize_token_embeddings(len(tokenizer))
```

새로 추가된 embedding 행은 forward 시 projected feature 로 overwrite 되므로 초기값 무의미. Splice 토큰 이름은 Path A/B 와 동일한 `<vila/video>` 로 통일 (기능은 이름 무관, 일관성 목적).

### (4) Forward / encode — processor bypass

`LlavaLlamaModel.forward` / `_embed` 전체 바이패스. `train_path_c.py :: encode_video_tokens_c(model, pixel_values, gi)`:

1. `ag_siglip(pixel_values, gazing_info=gi, output_hidden_states=True)` → `hidden_states[-2]` 를 per-frame 시퀀스로 사용.
2. `if_padded_gazing` 로 frame 마다 unpad → 9 의 배수가 되도록 last-token pad.
3. `(1, 9M, C)` 를 `(M, 9C)` 로 reshape, `mm_projector.layers` 의 첫 번째 submodule (`DownSample3x3BlockFix`) 를 **스킵**하고 후속 LayerNorm/MLP 만 실행. 수학적으로 TokenShuffle(9) 동일, 단 pretrained weight 는 공간 인접 3×3 가정이라 full-tune 으로 분포 이동 흡수.
4. `(M, llm_hidden)` projected feature 반환 → `inputs_embeds` 의 `<vila/video>` 위치에 splice → Qwen2 LLM forward.

### (5) 학습 스크립트 `train_path_c.py`

주요 함수:
- `build_model_and_tokenizer(snapshot_dir, scales, dtype)` — Lite-8B 로드 + SigLIP swap + tokenizer 등록.
- `_swap_in_autogaze_siglip(model, scales, dtype)` — 실제 swap + state_dict 전이.
- `encode_video_tokens_c(model, pixel_values, gi)` — vision → projector 파이프라인.
- `forward_loss(model, sample, video_token_id, llm)` — splice + CE loss.
- `configure_trainable(model, vision_unfreeze_last_n)` — vision 상위 N + projector 설정, LLM LoRA 부착 후 일단 freeze (Phase 1).
- `activate_phase2(model, optim, lr)` — LoRA param `requires_grad=True` + 신규 optim param group 추가.

CLI: `--phase1-updater-steps` (Phase 2 전환 시점, updater step 기준), `--vision-unfreeze-last-n` (기본 4), 그 외 Path B 와 동일 (`--max-steps`, `--grad-accum`, `--num-frames`, `--logger`).

### (6) Eval 스크립트 `eval_path_c.py`

`build_model_and_tokenizer` + `encode_video_tokens_c` 를 train 에서 import. 체크포인트 로더 `load_checkpoint(model, ckpt_dir)`:

1. `vision_tower.pt` (AutoGaze SigLIP 전체 state_dict) → `model.vision_tower.vision_tower.load_state_dict(strict=True)`.
2. `mm_projector.pt` → `model.mm_projector.load_state_dict(strict=True)`.
3. `llm_lora/` → `PeftModel.from_pretrained(model.llm, ...)` 로 LoRA 부착.

generate 는 Path A 와 동일 (`inputs_embeds` + `attention_mask` + `do_sample=False`, `max_new_tokens=8`).

---

## 5. Trainable surface (Phase 별)

| 모듈 | Phase 1 | Phase 2 |
| --- | --- | --- |
| `vision_tower` (하위 블록) | FROZEN | FROZEN |
| `vision_tower` (상위 N 블록) | requires_grad=True (full-rank) | requires_grad=True (유지) |
| `mm_projector` | full-tune | full-tune |
| `llm` (LoRA) | FROZEN | r=32, α=64, dropout=0.05, target=[q,k,v,o,gate,up,down]_proj |

---

## 6. Key decisions (why)

- **Lite-8B 에서 출발**: Stage 5 미튜닝된 원형. HD-Video 는 이미 gaze-aware 분포로 정렬되어 있지만 Lite-8B 는 plain SigLIP → vision_tower 상위 블록 재학습 필요 (Path B 와 구별되는 핵심 차이).
- **runtime swap (옵션 A 가 아니라)**: snapshot 파일 copy-replace 하면 재현성/revision 관리가 번거로움. VILA import + sys.modules 에 AutoGaze SigLIP 로딩 → `model.vision_tower.vision_tower` 참조만 바꾸면 됨.
- **splice 토큰 `<vila/video>`**: Path A/B 와 동일 명명으로 코드 일관성. Lite-8B tokenizer 에 없어 `add_special_tokens + resize_token_embeddings` 필요 (이름은 기능에 무관).
- **projector 유지 + pad-to-9 trick**: `mlp_downsample_3x3_fix` 의 후속 레이어는 input dim `C*9` 를 기대 — 우리 pad-to-9 flat 입력과 일치. 수학적으로 TokenShuffle(9) 과 동치, pretrained weight 재활용 가능.
- **vision_tower 부분 unfreeze**: AutoGaze SigLIP 자체는 새 학습 파라미터 없지만, sparse·멀티스케일 입력 분포를 Lite-8B 는 본 적 없음 → 상위 블록이라도 재학습.
- **2-phase 스케줄**: Phase 1 에서 vision+projector 먼저 adaptation → LLM 이 noisy feature 보기 전 분포 안정화. Phase 2 에서 LLM LoRA 로 downstream task 적응.
- **같은 scales "56+112+224+448"**: Path B 와 동일 user data 사용. positional embedding 은 `image_size=448` 기준 `interpolate_pos_encoding` 으로 bilinear 보간.

---

## 7. 파일 레이아웃

```
vila_hd_work/
  train_path_c.py              ← 신규 (Path B 기반)
  eval_path_c.py               ← 신규 (Path A 기반)
  runs/<run_name>/
    step_<N>/{vision_tower.pt, mm_projector.pt, llm_lora/}
    final/{vision_tower.pt, mm_projector.pt, llm_lora/}
    smoke.log                  ← smoke test 로그
    train.log                  ← full run tee 로그
    tb/                        ← --logger tensorboard 사용 시
```

---

## 8. 실행 커맨드

### Smoke test (10 samples × 100 micro-step)

```bash
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python train_path_c.py \
  --overfit-n 10 --max-steps 100 --grad-accum 4 \
  --num-frames 32 --vision-unfreeze-last-n 4 \
  --phase1-updater-steps 10 \
  --log-every 2 --save-every 0 \
  --output-dir runs/path_c_smoke \
  2>&1 | tee runs/path_c_smoke/smoke.log
```

기대: `[siglip-swap]` 로그, `[tokenizer]` 등록 로그 → Phase 1 (update 1–10) 에서 loss 감소 → update 10 에서 Phase 2 전환 → loss 추가 감소.

### Full run (2 epoch, wandb)

```bash
mkdir -p /home/yujin/gaze/trajgaze/vila_hd_work/runs/path_c_full && \
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python train_path_c.py \
  --max-steps 2734 --grad-accum 4 --num-frames 32 \
  --vision-unfreeze-last-n 4 \
  --phase1-updater-steps 200 \
  --log-every 5 --save-every 500 \
  --output-dir runs/path_c_full \
  --logger wandb --wandb-project trajgaze-path-c \
  --run-name path_c_full \
  2>&1 | tee runs/path_c_full/train.log
```

- 총 684 updater step ≈ Path B 와 동일 budget.
- Phase 1: 200 step (29%), Phase 2: 484 step (71%).

---

## 9. 로깅 지표 (updater step 기준)

Path B 와 동일 (`train/loss`, `train/lr`, `train/grad_norm`, `train/nv_tok`) + `train/phase` (1 or 2).

---

## 10. 결과 (test split 375 샘플)

| 지표 | 값 |
| --- | --- |
| **Overall** | **0.6267** (235/375) |
| qa_type=causal | 0.8413 (106/126) |
| qa_type=spatial | 0.4160 (52/125) |
| qa_type=temporal | 0.6210 (77/124) |
| dataset=ego4d | 0.6194 (83/134) |
| dataset=egoexo | 0.6939 (102/147) |
| dataset=egtea | 0.5319 (50/94) |

Baseline 0.5387, Path B 0.5893 대비 Path C 최고. 특히 temporal (+0.113 vs Path B) 과 ego4d (+0.082 vs Path B) 에서 큰 개선. egtea 만 Path B 대비 -0.021 로 소폭 후퇴.

---

## 11. Gotchas & 리스크

- **VILA import 체인 dep**: `llava.model.__init__` 이 다수의 optional 모듈을 unconditional import. Path C smoke 에서 필요했던 처리:
  - `pip install deepspeed` (`llava.train.sequence_parallel.globals` → `import deepspeed.comm`).
  - `pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales.git` (`vision_encoder` 쪽).
  - `ps3` 는 **설치 금지** (`ps3-torch` 이 transformers<=4.49 강제 → autogaze(~=4.51) 충돌). `sys.modules["ps3"]` 를 dummy stub 으로 주입.
  - `llava.model.language_model.fp8linearqwen2` 도 stub (transformers 4.51+ 에서 `Qwen2FlashAttention2` 제거됨).
  - `no_init_weights(_enable=True)` → transformers 4.50+ 시그니처 변경 (`_enable` kwarg 삭제) → `no_init_weights` 래퍼 패치.
- **splice 토큰 vocab 누락**: Lite-8B saved tokenizer 에 `<vila/video>` 가 없음 → `add_special_tokens + resize_token_embeddings` 필수. 빠뜨리면 prompt tokenize 시 여러 sub-piece 로 쪼개져 splice assert 실패.
- **projector pad-to-9 reshape**: pretrained weight 는 공간 인접 3×3 그룹 가정. sparse gazed token 은 이 가정을 깨므로 full-tune 으로 분포 이동 흡수. Phase 1 에서 projector loss 기울기 유의 관찰.
- **state_dict swap mismatch**: `[siglip-swap]` 로그의 missing/unexpected 키 개수 모니터링. 많으면 key mapping dict 추가 필요.
- **Lite-8B 의 SigLIP `image_size`**: 448 로 확인됨 (`config.json: vision_tower_cfg.image_size=448`, `patch_size=14`). HD-Video 와 동일.
- **`interpolate_pos_encoding` 동작**: AutoGaze SigLIP 이 학습 시 보지 못한 scale 조합 → 초기 positional embedding 품질 저하. Phase 1 에서 교정 기대.
- **vision_tower 부분 unfreeze 메모리**: full-rank 라 AdamW ≈ 8× param bytes (fp32 state). N=4 권장. 그 이상은 OOM 위험.
- **Phase 2 전환 시점의 lr**: cosine scheduler 가 이미 내려온 상태에서 LoRA param group 추가 → LoRA 가중치는 낮은 lr 로 시작. 필요 시 `optim.param_groups` 별도 lr.
- **체크포인트 크기**: vision_tower 전체 dump ≈ 수백 MB. `--save-every` 조정.
- **1750 QA 소규모**: vision_tower 재학습에 부족. 추가 일반 SFT 데이터 혼합 고려는 본 plan 범위 외.

---

## 12. Verification checklist

1. [x] Lite-8B snapshot 확보 (`ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d`).
2. [ ] Smoke test 통과 — `[siglip-swap]` 로그 + Phase 1→2 전환 + loss 단조 감소.
3. [ ] Full run 완료.
4. [ ] `eval_path_c.py` 로 test 375 샘플 accuracy 측정.
5. [ ] Path A/B 결과와 동일 테이블에 정리 → [CLAUDE.md](../CLAUDE.md) 업데이트.

---

## 관련 문서

- [baseline_autogaze_zeroshot.md](baseline_autogaze_zeroshot.md) — native AutoGaze baseline.
- [path_a_autogaze_precomputed.md](path_a_autogaze_precomputed.md) — HD-Video zero-shot 주입.
- [path_b_autogaze_lora_finetune.md](path_b_autogaze_lora_finetune.md) — HD-Video LoRA SFT.
- [`vila_hd_work/notes/multiscale_wiring.md`](../../vila_hd_work/notes/multiscale_wiring.md) — HD-Video scales override 근거 (Lite-8B 도 동일 원리).
- Reference: [`AutoGaze/INTEGRATION.md`](../../AutoGaze/INTEGRATION.md), [`AutoGaze/autogaze/vision_encoders/siglip/{modeling,configuration}_siglip.py`](../../AutoGaze/autogaze/vision_encoders/siglip/).
