# PLLaVA-7B × StreamGaze_v2 — LoRA Fine-tune (Projector Frozen, 6 epoch)

**구현 + 학습 + 6-epoch eval 완료. 결과 §9.** 자매 baseline [pllava_lora.md](./pllava_lora.md) (projector unfrozen, 2 epoch) 결과가 zero-shot 64f (overall 46.77) 대비 -2.85 회귀 (43.92) 한 데 대한 후속 실험. **projector 를 frozen 으로 두고 LoRA 만 6 epoch 학습** 하여 catastrophic forgetting 줄이고 LoRA 가 충분히 자랄 시간 확보가 목표. EGTEA 526 QA 평가.

**핵심 결과**: PF-LoRA ep00 overall **45.25** (sister 43.92 보다 +1.33, zero-shot 46.77 보다 -1.52). epoch 가 늘어날수록 EGTEA accuracy 단조 감소 (45.25 → 42.59) — **1 epoch 후 즉시 overfit**. Projector freeze 가설은 sister 의 catastrophic forgetting 일부 회복으로 부분 검증되나, EGTEA 도메인엔 여전히 LoRA fine-tune 이 net negative.

```
 [pllava_lora.md base 와 동일 데이터/eval pipeline]
        │
        ▼
 PllavaForConditionalGeneration  (PLLaVA-7B, bf16, gradient checkpointing)
   - vision_tower: CLIP-ViT-L/14-336        (frozen)
   - multi_modal_projector                  (FROZEN ← 변경)
   - language_model: Llama-2-7b + LoRA      (base frozen, LoRA train)
       LoRA r=16, alpha=32, dropout=0.05
       target_modules = q_proj, k_proj, v_proj, o_proj
   - PruneVid pruning: tau=1, cluster_ratio=1, temp_ratio=1  (모두 OFF)
        │
        ▼
 AdamW lr=1e-4, weight_decay=0, cosine 6-epoch, warmup 0.05  ← 변경 (2→6)
 effective batch = 2 (per-GPU) × 8 (grad-accum) = 16
        │
        ▼
 매 epoch 종료 시 pretrained_epoch{NN}/ 자동 저장 (~73MB, LoRA + projector dump)
   → 6 개 어댑터로 per-epoch 성능 곡선 비교
        │
        ▼
 pllava_eval_streamgaze.py (이미 적용된 패치 그대로 사용)
   - load_pllava: base PLLaVA-7B safetensors prefetch + .weight→.base_layer.weight remap
   - conv_mode = eval_mvbench_streamgaze (MM_INTERLEAF, 학습 conv 와 일치)
   - --answer_prompt none --return_prompt none
   → EGTEA 526 QA × 6 epoch
        │
        ▼
 print_csv.py × 6  → epoch 별 8-task accuracy + overall_micro
```

---

## 1. Motivation — projector freeze 가설

자매 실험 [pllava_lora.md §10.1](./pllava_lora.md) 의 결과:

| Task | ZeroShot-64f | LoRA-64f (proj unfrozen, 2 ep) | Δ |
|---|---:|---:|---:|
| past_gaze_sequence_matching | 31.25 | 31.25 | 0 |
| past_non_fixated_object_id | 29.41 | 29.41 | 0 |
| past_object_transition_pred | 50.00 | 50.00 | 0 |
| past_scene_recall | 8.11 | **16.22** | **+8.11** ✅ |
| present_object_attr_recog | 56.25 | 47.92 | **-8.33** ❌ |
| present_object_id_easy | 77.23 | 79.21 | +1.98 ✅ |
| present_object_id_hard | 68.75 | 60.94 | **-7.81** ❌ |
| present_future_action_pred | 27.66 | 20.21 | **-7.45** ❌ |
| **Overall (micro)** | **46.77** | **43.92** | **-2.85** ❌ |

진단 신호:
- `lora_B norm = 0.001740` (학습 후, 32-layer 평균 비슷). LoRA 가 거의 안 움직였다는 뜻 — under-fit.
- `multi_modal_projector` (~21M params) 만 사실상 학습돼, EgoExoLearn/HoloAssist domain image features → frozen LM 분포 사이 mapping 이 source-domain 에 overfit.
- EGTEA 평가 시 projector 출력이 LM 의 학습된 분포에서 벗어나 `present_*` task 들에서 zero-shot 기준 7~8% 회귀 (catastrophic forgetting).

가설: projector 를 frozen 시키면
1. 도메인 격차를 LoRA attention re-orientation 만으로 흡수 → forgetting 감소.
2. 학습 신호가 LoRA 에 집중 → `lora_B` 가 더 자랄 시간 확보 (epoch 6 으로 늘림).

---

## 2. 변경된 hyperparam (vs 자매 baseline)

| 항목 | pllava_lora.md (기존) | **pllava_lora_pf.md (본 실험)** |
|---|---|---|
| `freeze_projector` | False | **True** |
| `scheduler.epochs` | 2 | **6** |
| `output_dir` | `test_results/streamgaze_lora/pllava7b_r16_64f` | **`test_results/streamgaze_lora/pllava7b_r16_pf_64f`** |
| 모든 다른 설정 | — | 동일 (lora_r=16, lr=1e-4, 등) |

---

## 3. Eval pipeline (이전 세션에서 패치된 상태 그대로)

자매 실험 진행 중 발견한 **eval-side 버그 4 가지** 가 이미 fix 된 상태로 본 실험에서 사용:

### Bug 1 — base LM 가중치 미로드 → LM random init
- PLLaVA-7B HF release 의 safetensors 키는 `language_model.base_model.model.*` prefix (release 가 LoRA-wrapped 상태로 저장됨).
- `from_pretrained('MODELS/pllava-7b')` 가 unwrapped 모델에 매칭 시도 → **모든 LM weight 가 random init**. (`Some weights of PllavaForConditionalGeneration were not initialized from the model checkpoint at MODELS/pllava-7b and are newly initialized: [...]` 경고 출력됐었음.)
- 학습 측은 `pretrained_path="MODELS/pllava-7b"` 로 LoRA wrap 후 base 를 다시 로드 → OK.
- Eval 측에는 이 단계 누락 → 우리 73MB 어댑터만 로드하고 base LM 은 random 그대로 → 모든 위치에서 `<unk>` (id 0) emit.

**Fix** ([tasks/eval/model_utils.py:259-310](/home/yujin/gaze/PruneVid/tasks/eval/model_utils.py#L259-L310)):
- `use_lora=True` + `weight_dir != repo_id` 조건에서, LoRA wrap 후 `MODELS/pllava-7b` 의 sharded safetensors 를 prefetch 해 base LM 가중치 채움.

### Bug 2 — `.weight` ↔ `.base_layer.weight` 키 불일치 (k_proj/o_proj)
- PLLaVA-7B base 는 q_proj/v_proj 만 LoRA-wrapped 상태로 release. k_proj/o_proj 는 raw `.weight` 로 저장.
- 우리 모델은 q/k/v/o 모두 LoRA wrap → 모두 `.base_layer.weight` 형태 기대.
- 단순 매칭 시 k_proj/o_proj 만 random.

**Fix** ([tasks/eval/model_utils.py:280-300](/home/yujin/gaze/PruneVid/tasks/eval/model_utils.py#L280-L300)):
- prefetch 하면서 키 미스매치 발생 시 `.weight` → `.base_layer.weight` 자동 remap.

### Bug 3 — train/eval conv template 불일치 (MM_ALONE vs MM_INTERLEAF)
- 학습 config: `mm_alone=False` → image 토큰이 question 과 같은 USER turn 에 결합 (MM_INTERLEAF).
- eval 의 기본 `conv_eval_mvbench`: `mm_style=MM_ALONE` → image 별도 USER turn.
- LoRA-finetuned 모델이 학습 시 본 적 없는 prompt 형식에 OOD → EOS 즉시 emit.

**Fix** ([tasks/eval/eval_utils.py](/home/yujin/gaze/PruneVid/tasks/eval/eval_utils.py)):
- 신규 conv template `conv_eval_mvbench_streamgaze` 추가 (MM_INTERLEAF, 학습 system text 와 일치).
- shell script 가 `--conv_mode eval_mvbench_streamgaze` 사용.

### Bug 4 — answer_prompt prefix 불일치
- 학습 라벨은 `"(A) <content>"` 로 ASSISTANT 응답 시작.
- eval 의 default `answer_prompt="Best option:("` 가 강제되면, 모델은 학습한 패턴이 아닌 prompt 에서 시작 → EOS 즉시 emit.

**Fix** ([tasks/eval/streamgaze/pllava_eval_streamgaze.py](/home/yujin/gaze/PruneVid/tasks/eval/streamgaze/pllava_eval_streamgaze.py)):
- argparse 에 `--answer_prompt` / `--return_prompt` 추가, `'none'` 으로 비활성화 가능.
- LoRA shell script 가 `--answer_prompt none --return_prompt none` 전달.

위 4 가지 fix 모두 본 실험에서 그대로 활용. 자매 doc [pllava_lora.md §10.4](./pllava_lora.md#10.4) 의 issue 목록과 누적된 디버깅 결과.

---

## 4. 구현 방식 (자매 실험 대비 변경 부분만)

### 4.1 `tasks/train/config_pllava_streamgaze_pf.py` (신규)

기존 [config_pllava_streamgaze.py](/home/yujin/gaze/PruneVid/tasks/train/config_pllava_streamgaze.py) 복사 후 3 줄 변경:
- `freeze_projector = True`
- `scheduler.epochs = 6`
- `output_dir = "test_results/streamgaze_lora/pllava7b_r16_pf_64f"`

### 4.2 `scripts/train_streamgaze_lora.sh` (1 라인 수정)

`CONFIG` env var 지원 추가하여 외부에서 config 경로 override 가능. default 는 기존 config (호환성 유지).

```bash
CONFIG=${CONFIG:-tasks/train/config_pllava_streamgaze.py}
${ACCELERATE_BIN} launch ... -m tasks.train.train_pllava_nframe_accel ${CONFIG} ...
```

### 4.3 데이터/eval 코드 변경 없음

- `train_noegtea.json` (5,799 QA) 그대로 사용.
- `tasks/eval/model_utils.py`, `tasks/eval/eval_utils.py`, `scripts/eval_streamgaze_lora.sh`, `tasks/eval/streamgaze/print_csv.py` 모두 이전 세션 패치 상태 그대로.

---

## 5. Training CLI

### 5.1 명령

```bash
cd /home/yujin/gaze/PruneVid

CONFIG=tasks/train/config_pllava_streamgaze_pf.py \
FRAMES=64 \
OUTDIR=test_results/streamgaze_lora/pllava7b_r16_pf_64f \
bash scripts/train_streamgaze_lora.sh > logs/train_phaseA_pf_64f_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

예상 시간: **~5 시간** (per-epoch ~50 분 × 6).

### 5.2 매 epoch 자동 저장

[train_pllava_nframe_accel.py:555-560](/home/yujin/gaze/PruneVid/tasks/train/train_pllava_nframe_accel.py#L555-L560) 가 매 epoch 종료 시:
- `pretrained_epoch{NN}/` (73MB, LoRA + projector dump — projector 는 frozen 이지만 state_dict 에 포함됨, 안전)
- `ckpt_epoch{NN}/` (14GB, accelerator 전체 state for resume)

### 5.3 Smoke 검증 항목

학습 시작 직후 확인:
1. **Trainable params**: PEFT `print_trainable_parameters()` 가 16,777,216 (LoRA only).
2. **Optimizer 로그**: `optimizer -- lr=0.0001 wd=0.0 len(p)=256` (이전 260 에서 -4: projector linear_1/2 weight+bias).
3. **첫 step loss**: ~8-9 정도 시작.
4. **GPU**: 양쪽 H200 60-100% util.
5. **Resume 안 함**: 새 output_dir 이라 `Resumed from checkpoint` 메시지 없어야.

---

## 6. Eval CLI — Per-epoch 6 회

```bash
cd /home/yujin/gaze/PruneVid
for E in 00 01 02 03 04 05; do
  CKPT=test_results/streamgaze_lora/pllava7b_r16_pf_64f/pretrained_epoch${E} \
  SAVE=test_results/streamgaze_egtea/lora_pf_r16_64f_ep${E} \
  bash scripts/eval_streamgaze_lora.sh > logs/eval_pf_64f_ep${E}_$(date +%Y%m%d_%H%M%S).log 2>&1
done
```

총 시간 ~60 분 (6 × ~10 분, 각 526 QA).

### 6.1 결과 추출

```bash
for E in 00 01 02 03 04 05; do
  echo "epoch ${E}:"
  /opt/conda/envs/prunevid/bin/python -m tasks.eval.streamgaze.print_csv \
    test_results/streamgaze_egtea/lora_pf_r16_64f_ep${E}
done
```

---

## 7. 파일 레이아웃

```
PruneVid/
  tasks/train/
    config_pllava_streamgaze_pf.py     ← 신규 (3 줄 변경)
  scripts/
    train_streamgaze_lora.sh           ← 수정 (CONFIG env var)
  test_results/streamgaze_lora/
    pllava7b_r16_64f/                  ← 기존 (보존)
    pllava7b_r16_16f/                  ← 기존 (보존)
    pllava7b_r16_pf_64f/               ← 신규
      pretrained_epoch{00..05}/        ← 6 개, ~73MB each
      ckpt_epoch{00..05}/              ← 학습 끝나면 정리 (~84GB 회수)
  test_results/streamgaze_egtea/
    lora_r16_64f/                      ← 기존 (43.92)
    lora_pf_r16_64f_ep{00..05}/        ← 신규, 6 개 디렉토리
```

---

## 8. Key decisions (why)

- **projector freeze**: §1 가설 — projector unfreeze 가 source-domain overfit + LM 입력 분포 변동의 주범으로 의심. 가장 단순한 격리 실험.
- **6 epoch (vs 2)**: lora_B norm 0.0017 → 학습 더 필요하다는 신호. 2 epoch 의 3배. cosine warmup 도 자동으로 늘어남.
- **Phase A (64f) 만**: 시간/리소스 우선순위. 64f 가 zero-shot 에서 16f 대비 약간 낮았지만 per-frame 정보가 풍부해 LoRA fine-tune 효과 더 클 가능성 (자매 실험의 직관). 16f 는 본 실험 결과 보고 결정.
- **새 output_dir**: 기존 64f LoRA 결과 (43.92) 와 직접 비교 위해. auto_resume 영향 차단.
- **per-epoch 저장 보존**: epoch 별 성능 곡선 → overfitting 시점 / sweet spot 확인.
- **eval 측 default 그대로 유지**: zero-shot baseline (lora_alpha 14, projector 학습된 release) 호환성. LoRA-finetuned 호출만 `--conv_mode eval_mvbench_streamgaze --answer_prompt none --return_prompt none` 전달.

---

## 9. 실행 결과

### 9.1 학습 진행

epoch 별 mini-batch 수 = 1449 (effective batch 16 × 1449 ≈ 23k samples / epoch).

| epoch | video-loss (avg) | wall time | 종료 시각 |
|---:|---:|---:|---|
| 0 | 3.0070 | 0:49:38 | 2026-04-25 17:15 |
| 1 | 1.7938 | 0:48:41 | 2026-04-25 18:04 |
| 2 | 1.6213 | 0:46:13 | 2026-04-26 03:48 |
| 3 | 1.5094 | 0:46:10 | 2026-04-26 04:35 |
| 4 | 1.4435 | 0:45:57 | 2026-04-26 05:21 |
| 5 | 1.4126 | 0:48:15 | 2026-04-26 06:09 |

총 학습 시간: ~4:45 (epoch 0+1 첫 세션 ~1:38 + 재시작 후 epoch 2-5 ~3:08; 자세한 내막 §9.5).

Loss 단조 감소 (3.01 → 1.41). 자매 실험 ep01 avg = 1.79 와 본 실험 ep01 = 1.79 가 거의 일치 — projector unfreeze 의 빠른 loss 감소는 source-domain shortcut 학습이었음을 후행 검증.

### 9.2 lora_B / lora_A norm 진행

각 `pretrained_epoch{NN}/model.safetensors` 에서 다중 레이어 norm 추적. (단일 레이어만 보면 bf16 양자화로 인접 epoch 가 동일 값으로 round 될 수 있어 여러 레이어 함께 확인.)

| epoch | L0 q_proj.lora_B | L15 v_proj.lora_B | L31 k_proj.lora_B | L10 o_proj.lora_A |
|---:|---:|---:|---:|---:|
| 0 | 0.005157 | 0.251953 | 1.273438 | 2.312500 |
| 1 | 0.010132 | 0.318359 | 1.507812 | 2.328125 |
| 2 | 0.010132¹ | 0.349609 | 1.554688 | 2.343750 |
| 3 | 0.010132¹ | 0.373047 | 1.554688² | 2.343750² |
| 4 | 0.010132¹ | 0.382812 | 1.554688² | 2.343750² |
| 5 | 0.010132¹ | 0.382812² | 1.554688² | 2.343750² |

¹ bf16 양자화 우연 일치 (실제 모델 weight 는 변화 — sha256 다름, L15vB 등 다른 레이어가 단조 증가).
² 양자화 후 동일 값으로 round; 인근 정밀도 한계 (~1.5e-2).

비교: 자매 실험 (proj unfrozen, ep01) L0qB = 0.001740 → 본 실험 ep01 = 0.010132 → **약 6배** lora_B 가 자람. projector freeze 가 학습 신호를 LoRA 로 집중시킨 효과 정량 확인.

### 9.3 Per-epoch EGTEA accuracy (8 task, %, micro)

순서: past_gaze_seq, past_nonfix, past_obj_trans, past_scene, present_attr, present_easy, present_hard, present_future, overall.

| 어댑터 | past_gs | past_nfx | past_otp | past_sr | pres_attr | pres_e | pres_h | pres_fut | **overall** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZeroShot-64f | 31.25 | 29.41 | 50.00 | 8.11 | 56.25 | 77.23 | 68.75 | 27.66 | **46.77** |
| LoRA proj-unfrozen ep01 (sister) | 31.25 | 29.41 | 50.00 | 16.22 | 47.92 | 79.21 | 60.94 | 20.21 | **43.92** |
| **PF-LoRA ep00** ★ | 23.44 | 30.88 | 50.00 | 18.92 | 58.33 | 76.24 | 70.31 | 17.02 | **45.25** |
| PF-LoRA ep01 | 12.50 | 30.88 | 50.00 | 21.62 | 57.29 | 75.25 | 70.31 | 15.96 | 43.54 |
| PF-LoRA ep02 | 10.94 | 33.82 | 50.00 | 21.62 | 55.21 | 74.26 | 68.75 | 13.83 | 42.59 |
| PF-LoRA ep03 | 9.38 | 33.82 | 50.00 | 18.92 | 56.25 | 74.26 | 68.75 | 15.96 | 42.78 |
| PF-LoRA ep04 | 9.38 | 33.82 | 50.00 | 18.92 | 56.25 | 74.26 | 68.75 | 15.96 | 42.78 |
| PF-LoRA ep05 | 7.81 | 33.82 | 50.00 | 21.62 | 56.25 | 74.26 | 68.75 | 14.89 | 42.59 |

★ Best PF-LoRA: ep00 (45.25). zero-shot (46.77) 보다 -1.52, sister proj-unfrozen ep01 (43.92) 보다 +1.33.

### 9.4 관측

1. **Overall accuracy 단조 감소** (ep0 45.25 → ep5 42.59): training loss 는 단조 감소 (3.01 → 1.41) 하지만 EGTEA 정확도는 epoch 가 늘어날수록 떨어짐. 전형적인 **transfer-learning overfit** — LoRA 가 source domain (HoloAssist + EgoExoLearn) 분포에 fit 될수록 target (EGTEA) 에서 성능 저하.

2. **`past_gaze_sequence_matching` 이 회귀 주범**: ep00 23.44 → ep05 7.81 (-15.6). 이 task 하나가 overall 을 끌어내림. 학습 데이터의 gaze sequence 분포 (HoloAssist instruction-following + EgoExoLearn lab/sport) 가 EGTEA cooking 의 분포와 매우 다른 듯.

3. **Projector freeze 가설 부분 검증** ✅:
   - sister ep01 의 회귀 task 들 (`present_attr` -8.33, `present_id_hard` -7.81) 가 PF-LoRA ep00 에선 zero-shot 수준 회복 (각각 56.25, 70.31).
   - projector unfreeze 가 LM 입력 분포를 source 쪽으로 끌어당겼다는 가설 일치.

4. **`past_scene_recall` 향상 패턴**: zero-shot 8.11 → sister 16.22 → **PF-LoRA ep01 21.62 (+13.5)**. 두 실험 모두 향상 시켰지만 PF 가 더 크게. 이 task 는 LoRA 학습 신호를 잘 받음.

5. **`past_non_fixated_object_id` 는 늦게 학습**: ep00-ep01 30.88 → ep02-05 33.82. 4 epoch 째에 +3 향상. 일부 task 는 더 길게 학습이 도움.

6. **`present_future_action_prediction` 은 모든 epoch 에서 회귀**: zero-shot 27.66 → ep00 17.02 → ep05 14.89. 학습이 이 task 에 일관되게 해롭. 학습 데이터 측 mother task 분포의 차이.

7. **결론**: PF-LoRA + 짧은 학습 (~1 epoch) 이 최선. 6 epoch 까지 가면 overfit. zero-shot 보다 작은 폭 (-1.5) 으로 회귀하지만 sister 실험 (-2.85) 보다는 개선. **EGTEA 도메인엔 LoRA fine-tune 이 net negative** — domain matching 데이터 추가 또는 더 보수적 학습 필요.

### 9.5 이슈 / patch 로그

#### Issue #1 — silent termination after 1.5 epochs

**증상**: 첫 학습 세션 (16:25 시작) 이 18:18 시점 (epoch 2 step ~350/1449) 에 **에러 없이 silent termination**. dmesg 에 OOM 흔적 없음, 시스템 uptime 50+ 일 (재부팅 아님), background bash task 의 stdout/stderr 모두 비어 있음.

**가설**: 동일 호스트의 다른 사용자/프로세스 경합으로 인한 외부 SIGKILL 추정. 시점 부근 system load avg = 20.5 로 무거웠음.

**복구**: config 의 `auto_resume=True` + `save_steps=500` 으로 18:09 시점 `ckpt_resume_0.0120M/` (14GB, accelerator 전체 state) 가 보존된 상태였음. 다음 날 03:02 동일 명령 재실행 → `Resumed from checkpoint: ckpt_resume_0.0120M`, `Resume from epoch 2, steps 102` 로 자동 이어감 → 별 문제 없이 epoch 2-5 완주.

**교훈**: long-running 학습은 무조건 `auto_resume=True` 와 적당한 `save_steps` (500) 조합 필수. 본 config 의 default 가 그 조합이라 손실 0.

#### Issue #2 — bf16 양자화로 인한 단일-레이어 lora norm 동일값 trap

**증상**: 처음 lora_B norm 진행을 layer 0 q_proj 만 봤을 때 ep01-ep05 가 모두 0.010132 로 동일하게 보여 "학습이 멈췄나" 의심.

**원인**: bf16 정밀도 한계. 실제 다른 레이어 (L15 v_proj 등) 는 단조 증가, 파일 sha256 도 모두 다름. 학습은 정상 진행.

**대응**: §9.2 처럼 다중 레이어 + sha256 hash 함께 추적.

#### Issue #3 — eval 측 이슈 (이미 sister 실험에서 발견·수정)

본 실험에서 새로 발견된 eval 버그는 없음. sister 실험에서 발견·수정된 eval 패치 4 종 (§3) 그대로 적용해 정상 동작.

---

## 10. References

- 자매 baseline: [pllava_lora.md](./pllava_lora.md)
- Zero-shot baseline: [pllava_zeroshot.md](./pllava_zeroshot.md)
- PLLaVA: [arXiv:2404.16994](https://arxiv.org/abs/2404.16994)
- LoRA: Hu et al. 2021. q/k/v/o target = [QLoRA](https://arxiv.org/abs/2305.14314) §3 권장.
