# TrajGazeMerge E1 (keep 10%) — 진단 리포트

- **Checkpoint**: `TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth`
- **Stage 1 ckpt**: `TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth`
- **Test set**: EGTEA, 526 items, 8 tasks
- **Merge ratio**: 0.9 (상위 10% = 410 / 4096 토큰 유지)
- **실행일**: 2026-05-13

---

## 요약 (TL;DR)

| 질문 (이슈) | 답 | 근거 |
|---|---|---|
| 1. 선택된 토큰이 유용한가? | **부분적. encoder가 random보다 +6pt 좋지만 GT gaze는 무시함.** | gt_gaze_recall 0.074 < random 0.10 ; oracle ablation 54.94% < random 63.50% |
| 2. 토큰이 후반 프레임에 몰리는가? | **그렇다 — 후반 절반에 83% 집중.** | temporal_CoM 0.78, late_half_ratio 0.83 |
| 3. 아키텍처가 의도대로 작동하는가? | **대체로. merge는 균형, score는 방향성 보유.** | cluster_size_mean ≈ 10 = N/keep ; inverted = random ⇒ score 방향성 존재 |
| 4-a. Stage 1 overfitting? | **아니다** — epoch 100까지 val loss 단조 감소 | val loss 0.034 → 0.018 (epoch 10 → 100) |
| 4-b. 객관식 추측 vs 이해? | **혼합 신호.** headline 68.4%는 부풀려진 수치, 실제 ~43.7%. | avg-4-shift 61.7%, agree4 58.4%, C/D 위치 편향, ECE 0.144 |

**핵심 발견**: encoder는 분명히 뭔가를 학습함 (uniform/random/center보다 +5–8pt). 그러나 그것은 **GT gaze 위치에서 벗어나는** 방향으로 학습됨 — 즉 "gaze = 중요" 라는 method의 기본 전제가 EGTEA에서 성립하지 않음.

---

## 1. Per-Example Diagnostic (Module 1+2)

### 전체

| Metric | Value |
|---|---:|
| n_samples | 526 |
| Overall accuracy | **67.68%** (보고된 68.44%와 근접, 샘플링 차이) |
| Keep ratio | 0.100 |
| Mean top1 prob | 0.813 |
| Mean logit margin | 2.764 |
| **ECE (10 bins)** | **0.1437** (과신) |

### 태스크별 정확도

| Task | n | Acc (%) |
|---|---:|---:|
| past_gaze_sequence_matching | 64 | 68.75 |
| past_non_fixated_object_identification | 68 | 63.24 |
| past_object_transition_prediction | 2 | 50.00 |
| past_scene_recall | 37 | 56.76 |
| present_future_action_prediction | 94 | **46.81** ← 최저 |
| present_object_attribute_recognition | 96 | **87.50** ← 최고 |
| present_object_identification_easy | 101 | 70.30 |
| present_object_identification_hard | 64 | 75.00 |

→ Attribute recognition은 쉽고, future-action prediction은 random(25%)에서 그리 멀지 않음 (47%).

### Kept token의 기하학적 특성

| Metric | Value | Uniform 기준 |
|---|---:|---:|
| temporal_center_of_mass | **0.78** | 0.50 |
| late_half_ratio | **0.83** | 0.50 |
| gt_gaze_recall | **0.074** | 0.10 |
| cluster_size_mean | 9.99 | 9.99 |
| max cluster_size | 86 | – |
| src→recv cosine | 0.67 | – |

- **Temporal bias 확정**: 선택된 토큰이 일관되게 후반 프레임에 몰림.
- **Anti-gaze 선택**: encoder가 random보다도 더 GT gaze 위치를 회피함.
- **Merge는 평균적으로 균형** — 단, 한 receiver가 최대 86개 source를 흡수 (전체의 ~21%) 하는 극단 사례 존재.

### Feature effect sizes (|Cohen's d| 기준 정렬)

Cohen's d > 0 ⇒ correct 샘플에서 더 큰 값. AUC = correctness 예측 ROC-AUC.

| Feature | Cohen's d | AUC |
|---|---:|---:|
| logit_margin | **0.926** | 0.750 |
| top1_prob | **0.901** | 0.755 |
| cluster_size_mean | 0.204 | 0.435 |
| temporal_entropy | -0.180 | 0.464 |
| score_entropy | -0.178 | 0.505 |
| cluster_size_std | 0.129 | 0.545 |
| spatial_entropy | -0.125 | 0.525 |
| temporal_center_of_mass | -0.098 | 0.503 |
| late_half_ratio | -0.078 | 0.516 |
| gt_gaze_recall | 0.061 | 0.494 |
| score_mean / std / max | < 0.12 (모두) | ≈ 0.50 |

→ 신뢰도 관련 feature(logit_margin, top1_prob)만 큰 effect를 보임. **선택 품질 관련 지표(gt_gaze_recall, temporal/spatial 구조)는 정답률과 무관**. 모델은 맞고 틀린 샘플에서 거의 동일한 선택 패턴을 보임.

---

## 2. Counterfactual Ablation (Module 3)

동일 모델·동일 입력·동일 r=3686 budget에서 **score source만 교체**하여 정확도 비교.

| Source | Acc% | Δ vs learned | Mean top1 | Logit margin |
|---|---:|---:|---:|---:|
| **learned** | **69.01** | — | 0.814 | 2.767 |
| uniform | 61.41 | −7.60 | 0.789 | 2.472 |
| random | 63.50 | −5.51 | 0.800 | 2.613 |
| inverted | 63.50 | −5.51 | 0.789 | 2.451 |
| center (egocentric prior) | 61.79 | −7.22 | 0.781 | 2.396 |
| **oracle (GT gaze)** | **54.94** | **−14.07** | 0.729 | 1.943 |
| text_only (zero visual) | 53.61 | −15.40 | 0.703 | 1.663 |

### 해석

- **Encoder는 기여한다**: learned가 uniform/center/random을 5–8pt 차이로 이김.
- **Score는 방향성을 가진다**: 뒤집으면 (inverted) random 수준으로 떨어짐 — 즉 score field에 유용한 토큰 쪽으로의 의미 있는 기울기가 존재.
- **Oracle paradox**: GT gaze 위치에 토큰을 두는 것이 learned 대비 14pt 떨어지고, **random보다도 8.5pt 낮음**. `gt_gaze_recall = 0.074` 와 결합해서 보면, **gaze 위치가 EGTEA VQA에 유용한 패치를 가리키지 않는다**는 강력한 증거.
- **시각 정보의 기여 ≈ 15.4pt** (learned 69.0 − text_only 53.6). 그중 절반은 learned scoring에서, 나머지 절반은 단순히 visual token이 존재한다는 사실에서.

---

## 3. Option Permutation (Module 3a)

각 샘플에 대해 A/B/C/D 위치를 4-순환 시프트하여 4회 평가.

| Metric | Value | Reference |
|---|---:|---:|
| 4 shift 평균 정확도 | **61.74%** | random = 25% |
| k0 (원본) | 68.44 | headline과 일치 |
| k1 | 61.22 | |
| k2 | 56.84 | |
| k3 | 60.46 | |
| **Agree4** (4 shift 모두 같은 콘텐츠 선택) | **58.37%** | random ≈ 1.6% |
| Agree2 | 99.62 | |
| **Consistent + correct** | **43.73%** | "진짜 이해" 추정치 |
| Pick frequency A / B / C / D | 20.6 / 18.6 / **31.6** / **29.2** | uniform = 25% |
| Mean top1 prob, agree4 | 0.865 | — |
| Mean top1 prob, disagree | 0.701 | — |

### 해석

- **C/D 위치 편향**: 모델이 확신이 없을 때 후반 글자(C, D)로 쏠림 — 두 글자 합쳐 60.8% 픽 (기대값 50%).
- **Permutation 비용**: k0 (68.4) vs 평균 (61.7) 사이 7pt 갭.
- **"정직한" 정확도 추정**: 4 shift 일관 + 정답 = **43.73%**. 이것이 모델이 진짜로 이해한다고 볼 수 있는 비율의 상한이며, 나머지는 위치/옵션 콘텐츠 단서로 회복된 것.
- 신뢰도는 일관성과 상관됨: 0.865 (agree4) vs 0.701 (disagree).

---

## 4. Stage 1 Held-out Loss Curve (Module 3b)

EgoExoLearn+HoloAssist에서 deterministic 10% held-out, 고정 seed (60 val 샘플, 11 batches @ size ≤2).

| Epoch | Val loss | traj | score_fut | score_past | score_traj |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.0343 | 0.0190 | 0.0015 | 0.0021 | 0.0118 |
| 20 | 0.0270 | 0.0154 | 0.0014 | 0.0017 | 0.0086 |
| 30 | 0.0249 | 0.0144 | 0.0013 | 0.0014 | 0.0078 |
| 40 | 0.0232 | 0.0141 | 0.0011 | 0.0008 | 0.0072 |
| 50 | 0.0212 | 0.0133 | 0.0011 | 0.0006 | 0.0062 |
| 60 | 0.0196 | 0.0123 | 0.0009 | 0.0005 | 0.0059 |
| 70 | 0.0187 | 0.0119 | 0.0008 | 0.0003 | 0.0057 |
| 80 | 0.0183 | 0.0116 | 0.0008 | 0.0003 | 0.0057 |
| 90 | 0.0178 | 0.0114 | 0.0007 | 0.0003 | 0.0055 |
| **100 (best)** | **0.0178** | **0.0113** | **0.0007** | **0.0002** | **0.0055** |

- Val loss가 epoch 100까지 **단조 감소** — overfitting 신호 없음.
- best.pth ≈ epoch 100 (모든 loss 항목에서 일관).
- **시사점**: Stage 1은 over-trained가 아니라 **data-bound**. 더 많은 데이터를 주면 (예: 미사용 `proactive_*` ~2,148건) 추가 이득 가능성 있음.

플롯: [holdout_E1_patch_temporal_loss_curve.png](../TrajGaze_v2/eval_results/holdout_E1_patch_temporal_loss_curve.png)

---

## 5. 종합 권고 (방법론 관점)

영향력 / 방법론적 중요도 순.

### A. Gaze supervision 전제 재검토

가장 중요한 단일 발견은 **oracle (GT gaze) < random**. method 논문의 narrative — "gaze가 attention을 중요 패치로 가이드한다" — 가 EGTEA 데이터에서는 반증됨. 다음 후속 실험들이 필요:
- **Hand-only Stage 1**: gaze 없이 학습해 정확도가 떨어지는지 유지되는지 확인.
- **Anti-gaze supervision**: Stage 1을 gaze에서 *벗어난* 패치를 예측하도록 학습해 성능이 오르는지 확인.
- **태스크별 gaze 유의성**: `gt_gaze_recall`을 태스크로 쪼개기 — `past_gaze_sequence_matching` (gaze 패턴 자체가 질문) 에서는 도움이 되지만 object-recognition 태스크에서는 해가 될 수 있음. M1 parquet에서 재집계 가능.

### B. Temporal bias 정량화 및 완화

- 83% late-half ratio는 Stage 1의 `I_scores_past/future` supervision 자체가 시간적으로 편향되어 있음을 강력히 시사 (예: trajectory-prediction loss가 후반 프레임에 더 큰 가중치를 주는지).
- 빠른 확인: training 데이터에서 `I_scores_past`의 프레임 인덱스별 평균을 plot. 후반 편향이 보이면 encoder가 *잘못된 target*에 정확히 fitting하고 있다는 뜻.
- 가능한 수정: Stage 2에 frame-balanced regularizer; Stage 1 loss schedule 시간 가중 재조정; explicit temporal entropy bonus.

### C. 논문 평가 protocol 변경

**두 개의 숫자**를 함께 보고:
- Headline single-shift 정확도 (현재 68.44%) — 기존 연구와 비교 가능성 위해.
- Permutation-평균 정확도 (61.74%) 와 consistent-correct 비율 (43.73%) — "진짜 이해"의 공정한 측정치.
- **ECE (0.144) 도 테이블에 추가** — calibration이 비교 가능한 의미 있는 축.

### D. 데이터 확장이 아키텍처 변경보다 먼저

- Stage 1이 overfit 아니므로 데이터 추가는 안전한 선택지.
- `proactive_gaze_triggered_alert` (283) + `proactive_object_appearance_alert` (1,865) = **~37% 학습 데이터 증가** 가능. 재아키텍처 전에 가장 저렴한 실험.
- Stage 1 학습 epoch 추가는 효과 미미 (epoch 90 ≈ 100 거의 평탄).

### E. Max-cluster-size outlier 검사

`max_cluster_size = 86` (일부 샘플에서 한 receiver가 전체 source의 21% 흡수) — 이런 샘플에서 score 분포가 퇴화된 모드를 가질 가능성. parquet에서 `cluster_size_max` 상위 10개를 뽑아 `viz_token_selection.py`로 시각화하면 failure mode가 드러날 수 있음.

---

## 6. 산출물 인덱스

진단 산출물은 `TrajGazeMerge/eval_results/` 가 `.gitignore`에 의해 제외되므로 로컬에만 존재.

| Artifact | Path (로컬) |
|---|---|
| Per-sample parquet (M1) | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_diag_per_sample.parquet` |
| M1 summary JSON | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_diag_summary.json` |
| Aggregate plots + summary (M2) | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_diag/` |
| Feature effect sizes CSV | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_diag/feature_effect_size.csv` |
| Ablation per-source parquets | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_abl_ablation_*_per_sample.parquet` |
| Ablation summary JSON | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_abl_ablation_summary.json` |
| Permutation parquet | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_perm_permutation.parquet` |
| Permutation summary JSON | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_perm_permutation_summary.json` |
| Stage 1 holdout JSON | `TrajGaze_v2/eval_results/holdout_E1_patch_temporal_val_loss.json` |
| Stage 1 holdout plot | `TrajGaze_v2/eval_results/holdout_E1_patch_temporal_loss_curve.png` |
| 실행 로그 | `TrajGazeMerge/eval_results/diagnostic/logs/` |

---

## 7. 재현 명령어

```bash
ROOT=/workspace/trajgaze
S1=$ROOT/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth
LORA=$ROOT/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth
PY=/opt/conda/envs/gaze/bin/python
cd $ROOT

# M1 + M2 (~30분)
CUDA_VISIBLE_DEVICES=1 $PY -m TrajGazeMerge.eval.diagnostic_eval \
  --stage1-ckpt $S1 --lora-ckpt $LORA --tag E1_keep10_diag
$PY -m TrajGazeMerge.eval.analyze_diagnostics --tag E1_keep10_diag

# M3 ablation (~3.5h)
CUDA_VISIBLE_DEVICES=0 $PY -m TrajGazeMerge.eval.ablation_score_source \
  --stage1-ckpt $S1 --lora-ckpt $LORA --tag E1_keep10_abl

# M3a permutation (~2h)
CUDA_VISIBLE_DEVICES=1 $PY -m TrajGazeMerge.eval.option_permutation_eval \
  --stage1-ckpt $S1 --lora-ckpt $LORA --tag E1_keep10_perm

# M3b holdout (~30분)
CUDA_VISIBLE_DEVICES=1 $PY -m TrajGaze_v2.training.eval_stage1_holdout \
  --ckpt-dir $ROOT/TrajGaze_v2/checkpoints/E1_patch_temporal \
  --epochs 10 20 30 40 50 60 70 80 90 100 --also-best \
  --tag E1_patch_temporal
```
