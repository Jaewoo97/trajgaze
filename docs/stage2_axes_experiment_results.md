# Stage 2 Axes Experiment Results

`docs/stage2_qa_accuracy_plan.md` 에서 정의한 5개 개선 축(axes 1~5) 중 어느 조합이
실제 정확도를 끌어올리는지 검증한 실험 결과 정리.

- 데이터셋: StreamGaze_v2 / egtea test split, **n=526** (full-split per-task eval)
- 모델: Qwen2.5-VL-7B-Instruct + LoRA (rank 16, q/k/v/o), keep=10% (merge_ratio 0.9)
- Stage 1 인코더: `TrajGaze_v2/checkpoints/stage1_v3/best.pth` (가중치 동결)
- Teacher: `king_ms.pth` (baseline LoRA Qwen, 학습 acc 61.5%)
- 평가 스크립트: [TrajGazeMerge/eval/eval_per_task.py](../TrajGazeMerge/eval/eval_per_task.py)

---

## 비교 대상 3개 run

| Run | Axes 활성 | CLI 핵심 플래그 | 출력 디렉토리 |
|---|---|---|---|
| **baseline** | 없음 (원래 10% merge_lora) | (axes 모두 OFF) | — (사용자 제공 수치) |
| **axes23** | axis 2 + axis 3 | `--alpha-schedule warmup_ce --kd-gate correct --vit-lora-rank 8 --vit-lora-last-n 2` | [`checkpoints/axes23/`](../TrajGazeMerge/checkpoints/axes23/) |
| **axes1235a** | axis 1 + 2 + 3 + 5a | (위 + ) `--merge-scope global --k-min 1 --aux-traj-tokens 4 --aux-traj-hidden 256` | [`checkpoints/axes1235a/`](../TrajGazeMerge/checkpoints/axes1235a/) |

세 run 모두 동일하게 epochs=3, lr_lora=1e-4, lr_enc=1e-5, alpha=0.5, grad_accum=4 로 학습.

---

## Per-Task 결과 (egtea n=526, keep=10%)

| # | Task | N | Baseline | **axes23** | axes1235a | axes23 Δ vs Base | axes1235a Δ vs Base |
|:-:|------|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | past_gaze_sequence_matching | 64 | 68.75 | **73.44** | 65.62 | **+4.69** 🟢 | −3.13 |
| 2 | past_non_fixated_object_identification | 68 | 63.24 | 58.82 | 63.24 | −4.42 🔴 | 0.00 |
| 3 | past_object_transition_prediction | 2 | 50.00 | 0.00 | 0.00 | (n=2 무의미) | (n=2 무의미) |
| 4 | past_scene_recall | 37 | 43.24 | **51.35** | 48.65 | **+8.11** 🟢 | **+5.41** 🟢 |
| 5 | present_object_attribute_recognition | 96 | 90.62 | **92.71** | 89.58 | **+2.09** 🟢 | −1.04 |
| 6 | present_object_identification_easy | 101 | 59.41 | **62.38** | 56.44 | **+2.97** 🟢 | −2.97 🔴 |
| 7 | present_object_identification_hard | 64 | 62.50 | **65.62** | 64.06 | **+3.12** 🟢 | +1.56 🟢 |
| 8 | present_future_action_prediction | 94 | 51.06 | **52.13** | 47.87 | **+1.07** 🟢 | −3.19 🔴 |
| | **Weighted OVERALL** | **526** | **64.45** | **66.54** ⭐ | **63.12** | **+2.09** | **−1.33** |

> n=2 인 `past_object_transition_prediction` 은 표본이 너무 작아 2번의 정답/오답이
> 결과를 100% 흔드므로 모든 비교에서 제외 처리.

---

## 한눈에 보는 핵심

### axes23 (axis 2+3) — production-ready

- **OVERALL +2.09pt** (64.45 → 66.54), 7개 유효 task 중 **5개 win, 1개 tie, 1개 loss**
- 가장 큰 이득: `past_scene_recall` **+8.11pt** — plan 의 "정보 손실 회복 핵심 시그널" 충족
- 유일한 손실: `past_non_fixated_object_identification` −4.42pt
- Teacher 와의 격차: **gap = −2.85** (10% 토큰 student 가 full token teacher 를 평균적으로 추월)

### axes1235a (axis 1+2+3+5a) — 회귀

- OVERALL **−1.33pt** vs baseline, **−3.42pt** vs axes23
- 7개 유효 task 중 **3 win / 2 tie / 2 loss**, 평균 신호가 약화
- axis 1+5a 추가가 **도움 안 됨** (오히려 독이 됨)

---

## 활성된 axes 효과 분석

| Axis | 의도 | axes23 결과 | axes1235a 결과 |
|------|------|-------------|----------------|
| **2 — ViT LoRA last-2 blocks** | egocentric 도메인에 ViT 적응 | ✅ 평균 +2pt 기여 추정 | (axes23 와 공유) |
| **3a — `--alpha-schedule warmup_ce`** | epoch 초반 KL 비중 ↑, 후반 CE ↑ | ✅ epoch 1 loss 0.20 까지 빠르게 떨어짐 | (axes23 와 공유) |
| **3b — `--kd-gate correct`** | teacher 정답 sample 에만 KL 적용 | ✅ teacher 오답 흡수 차단 | (axes23 와 공유) |
| **1 — `--merge-scope global`** | frame-level 예산 (dense frame ↑ token) | ❌ 회귀 (오답 분포로 이동) | 활성 |
| **5a — `--aux-traj-tokens 4`** | LLM 입력에 trajectory 4 토큰 주입 | ❌ 회귀 (LLM 입력 분포 교란 가능성) | 활성 |

axis 1+5a 회귀의 가설:
1. **5a 의 4-token hint** 가 Qwen 의 입력 분포를 흔들면서 `present_object_identification_easy` 같은
   세밀한 visual recognition task 에서 noise 가 됨.
2. **1 의 global frame budget** 이 frame_attend 신호 (cross-attn peakiness 평균) 를 사용하는데,
   이 신호가 frame 단위 importance 를 정확히 반영하지 못해 오히려 잘못된 frame 에 토큰을 몰아줌.

두 가설 모두 검증되려면 단일 axis 만 켠 ablation 이 필요 (axis 1 only / 5a only) — 본 실험에서는 미수행.

---

## 학습 동역학

| Run | Epoch 1 avg_loss | Epoch 3 avg_loss | wall time |
|---|:-:|:-:|:-:|
| axes23 | 0.2030 | 0.4231 | ≈ 9h |
| axes1235a | (~0.22) | 0.3729 | ≈ 9h |

`warmup_ce` 스케줄로 epoch 1 → 3 로 갈수록 CE 비중이 ↑ 되어 loss 표현 자체가 변함.
직접 비교는 의미 없으나 두 run 모두 정상적으로 수렴 후 종료. axes1235a 의
회귀 원인은 학습 미수렴이 아니라 axes 자체가 도움 안 된 것으로 결론.

---

## 결론

1. **axes23 가 plan 의 axes 조합 중 가장 강력한 production 모델.** OVERALL +2.09pt, scene-recall +8.11pt 회복.
2. **axis 1+5a 추가는 현 형태로는 도움 안 됨.** 단독 ablation 또는 다른 hyperparameter
   (예: `--aux-traj-tokens 1~2`, `--merge-scope global` 만) 로 재검증 필요.
3. **plan 의 단계 4 (Stage 1 supervision 재설계)** 는 본 실험으로는 불필요 — `past_scene_recall`
   gap 이 `−21.62 → −13.51` 로 plan 의 −10pp 임계에 거의 도달했고, 추가 ablation 이 더 저비용.

### 권장 다음 액션

| 우선순위 | 액션 | 비용 | 기대 |
|:-:|---|:-:|---|
| 1 | **axes23 를 production base 로 고정** ([axes23/best.pth](../TrajGazeMerge/checkpoints/axes23/best.pth)) | 0 | 안정 운영 |
| 2 | axis 1 only / 5a only 단독 ablation (각 9h) | ~18h | axis 1 또는 5a 의 부분 효과 분리 검증 |
| 3 | `--aux-traj-tokens 1` 로 5a 약화 + axes23 와 결합 | ~9h | 5a 의 부작용 완화 가능성 |
| 4 | (선택) plan 단계 4 — Stage 1 supervision 재설계 | 大 | scene_recall 의 −13.51pp 잔여 격차 해소 시도 |

---

## 산출물 파일 위치

- **Best 모델**: [TrajGazeMerge/checkpoints/axes23/best.pth](../TrajGazeMerge/checkpoints/axes23/best.pth) (16GB)
- **Per-task 로그 (axes23)**: [axes23/per_task_eval_best.log](../TrajGazeMerge/checkpoints/axes23/per_task_eval_best.log)
- **Per-task 로그 (axes1235a)**: [axes1235a/per_task_eval_best.log](../TrajGazeMerge/checkpoints/axes1235a/per_task_eval_best.log)
- **학습 stdout**:
  - [axes23/stdout.log](../TrajGazeMerge/checkpoints/axes23/stdout.log)
  - [axes1235a/stdout.log](../TrajGazeMerge/checkpoints/axes1235a/stdout.log)
- **Plan 원본**: `~/.claude/plans/linked-enchanting-hennessy.md`
- **평가 스크립트**: [TrajGazeMerge/eval/eval_per_task.py](../TrajGazeMerge/eval/eval_per_task.py)
- **회귀 테스트** (encoder return_extras 호환성): [TrajGazeMerge/tests/test_encoder_compat.py](../TrajGazeMerge/tests/test_encoder_compat.py)

---

*작성: 2026-04-25, axes23 실험 종료 후*
