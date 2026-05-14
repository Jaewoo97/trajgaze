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
| 1. 선택된 토큰이 유용한가? | **예. method 진짜 작동. encoder는 gaze trajectory를 추상화해 +5pt 추가 기여.** | learned 69.0 > soft_oracle 64.3 > random 63.5 > uniform 61.4 ; counterfactual mask_kept −13pt (receiver 정보 사용 확인) |
| 2. 토큰이 후반 프레임에 몰리는가? | **그렇다 — 후반 절반에 83% 집중.** | temporal_CoM 0.78, late_half_ratio 0.83 |
| 3. 아키텍처가 의도대로 작동하는가? | **대체로. merge는 균형, score는 방향성 보유.** | cluster_size_mean ≈ 10 = N/keep ; inverted = random ⇒ score 방향성 존재 |
| 4-a. Stage 1 overfitting? | **아니다** — epoch 100까지 val loss 단조 감소 | val loss 0.034 → 0.018 (epoch 10 → 100) |
| 4-b. 객관식 추측 vs 이해? | **headline 68.4%의 ~29pt가 MC 구조 보너스.** 진짜 이해 ~34.5%. | open-ended (옵션 미공개) 34.5% (random 25%, +9.5pt) ; agree4 58.4%, ECE 0.144 |
| 4-c. Method가 진짜 작동? | **예 — 그러나 spatial 정렬은 무관, 전반 절반 토큰은 무용.** | mask_kept −13pt ✓ ; shuffle_kept ±0pt (bag-of-tokens) ; mask_early −1pt vs mask_late −11pt |

**Phase 0 verdict (§11)** ⚠️ — paper narrative의 데이터 정합성에 큰 문제:
- **Method가 gaze-intrinsic 태스크에서 GT gaze보다 짐** (learned 70.31 < soft_oracle 71.88, n=64). 헤드라인 +4.75pt 우위는 거의 전부 non-gaze 태스크에서.
- **Method는 LoRA 없이 효과 0** (Δ −0.19pp). 진짜 contribution은 +3.4pp (LoRA의 +16pp 대비 1/5).
- **StreamGaze는 paper 주장 검증에 부적합** — gaze-intrinsic n=64에서 모델이 추측 수준 (consistent_correct 26.6%).

**핵심 발견 (followup 4 실험으로 업데이트)**:
1. **Method 작동 확인** — counterfactual mask_kept −13pt로 receiver 정보 사용 입증.
2. **Encoder의 추가 기여 ≈ +4.75pt** — soft_oracle (GT gaze 위치 단독) 64.26 → learned 69.01. 즉 encoder가 gaze trajectory에서 추출하는 정보는 raw gaze 위치보다 4.75pt 더 가치 있음.
3. **GT gaze 위치도 약하게 유용** — soft_oracle 64.26 > uniform 61.41 (+2.85pt). 이전 hard oracle (54.94%) 의 낮은 점수는 §7 §oracle caveat에서 지적한 argsort tie-breaking 디제너러시 때문이었음 — 본 followup이 이를 해소.
4. **MC 구조가 28.8pt 보너스** — open-ended (옵션 미공개) 34.46% vs MC-logit 63.28%. "진짜 visual+question 이해"의 상한은 ~34.5% (random 25%보다 +9.5pt). headline 68.4%의 절반 이상은 옵션 구조 활용에서 옴.
5. **Spatial 정렬은 무관, bag-of-tokens** — shuffle_kept ±0pt. 모델은 어느 receiver가 어디 있는지 신경 안 씀. 논문의 spatial selection narrative (Table 4) 약화.
6. **전반 프레임 토큰은 무용** — mask_kept_late −11pt vs mask_kept_early −1pt. 학습된 receiver의 ~50%가 정보 기여 거의 없음 → 토큰 budget 낭비.

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

## 5. 논문 기존 ablation과의 관계

`docs/Gaze_hand_Traj.md` Section 5.3에 이미 다음 ablation들이 보고됨:

**Table 2 — Modality ablation (이미 존재)**

| | Avg |
|---|---:|
| OnlyHand | 66.16 |
| OnlyGaze | 64.64 |
| Hand+Gaze | **68.44** |

→ Fusion이 +2.3pp 이득. Gaze는 OI-H (+10.94pp), FAP에서 보완적이라고 논문이 결론.

**Table 3 — Pretraining objectives**: Nopretrain 65.02 / Onlyscoreloss 65.78 / Allloss 68.44.
**Table 4 — Spatial vs temporal pruning**: Nospatial 64.64 / Notemporal 61.43 / Spatio-temporal 67.49.

### 본 진단이 새로 더하는 결과

논문 ablation 어디에서도 다루지 않은 8가지 새 발견:

| 발견 | 본 리포트 위치 | 시사 |
|---|---|---|
| **oracle (GT gaze 위치) = 54.94%** < random 63.5% | §2 | 학습된 score (논문 Table 2 OnlyGaze 64.64) 와 달리, GT gaze 위치 자체는 무용 — encoder가 gaze를 그대로 쓰지 않고 변환한다는 강한 증거 |
| **inverted = random = 63.5%** | §2 | learned score의 *방향성* 정량화 — 논문엔 없음 |
| **uniform = 61.4%** (no-score cosine merge) | §2 | "no-pretrain 65%" (Table 3) 와 달리, "no-score" baseline은 본 진단이 처음 |
| **text_only = 53.6%** | §2 | 시각 정보의 총 기여 = 15.4pp 정량화 |
| **option permutation 평균 61.7%, agree4 58.4%** | §3 | 객관식 평가의 신뢰성 — 정직한 정확도 추정치 43.7% |
| **ECE 0.144 / 과신** | §1 | calibration 축은 논문 어디에도 없음 |
| **gt_gaze_recall 0.074** < random 0.10 | §1 | encoder가 **anti-gaze 방향**으로 학습됨 정량화 |
| **Stage 1 holdout val curve 단조 감소** | §4 | overfitting 부재 — "데이터 확장 가능" 근거 |
| **temporal_CoM 0.78 / late_half 0.83** | §1 | 시간적 편향 정량화 |

### 새 해석: "gaze position ≠ gaze representation"

논문은 OnlyGaze (64.64) < Hand+Gaze (68.44) 로부터 "gaze가 anticipated context를 제공해 fusion에 기여" 라고 해석. 본 진단은 더 강한 형태의 결론을 제시:

- **GT gaze 위치 → score = 무용** (oracle 54.94 < random 63.50)
- **gaze trajectory → encoder → score = 유용** (논문 OnlyGaze 64.64 > 논문 Nopretrain 65.02 와 비슷)
- 즉, encoder가 gaze로부터 추출하는 것은 *"어디를 보았는가" 가 아니라 "어떤 시간적 맥락에서 어떻게 움직였는가"* 라는 추상화된 신호.
- 이는 논문의 "anticipated broader context" 표현과 정합적이지만, 본 진단이 oracle 비교를 통해 **그 차이를 처음으로 정량 분리**했음.

---

## 6. Follow-up 4 실험 결과

§7에서 우선순위 1–5로 제안한 후속 실험 중 4개를 526 EGTEA test 또는 stratified subset에서 실행 완료.

### 6.1 Soft-oracle ablation (526 샘플)

§2 oracle 결과의 argsort tie-breaking 디제너러시를 해소하기 위해 Gaussian falloff (σ = 0.20 × side) 로 GT gaze 위치 점수를 분포시킴.

| Source | Acc% | vs uniform | vs learned |
|---|---:|---:|---:|
| learned | 69.01 | +7.60 | — |
| **soft_oracle** | **64.26** | **+2.85** | **−4.75** |
| random | 63.50 | +2.09 | −5.51 |
| uniform | 61.41 | — | −7.60 |
| ~~hard oracle~~ | ~~54.94~~ | ~~−6.47~~ | ~~−14.07~~ (artifact) |

→ **§2의 결론 정정**: GT gaze 위치는 **약하게 유용** (+2.85pt over uniform), 그러나 encoder는 **추가로 +4.75pt** 를 만들어냄. 두 contribution이 분리됨.

### 6.2 Counterfactual masking (526 샘플 × 5 variants)

| Variant | Acc% | Δ vs baseline | 해석 |
|---|---:|---:|---|
| baseline | 68.25 | — | 생산 설정 |
| **mask_kept** (전체 zero-out) | **55.32** | **−12.93** | Method 진짜 receiver 사용 ✓ |
| mask_kept_late (후반 절반 zero) | 57.60 | −10.65 | 후반 receiver가 정보의 ~85% 보유 |
| mask_kept_early (전반 절반 zero) | 67.30 | −0.95 | **전반 receiver는 사실상 무용** |
| **shuffle_kept** (순서 셔플) | **69.20** | **+0.95** | **Spatial 정렬 무관 — bag-of-tokens** |

→ 강력한 시사:
- **method validity 확인** (mask_kept −13pt)
- **토큰 budget 낭비** — 전반 절반의 receiver (~205개) 가 −1pt 밖에 기여 안 함. 사실상 후반 50% 만으로 충분
- **spatial selection narrative 약화** — 논문 Table 4 ("spatial pruning improves +2.85pp") 가 의문시됨. 모델이 receiver의 spatial 위치를 무시함

### 6.3 Open-ended generation (177 stratified, 목표 200)

옵션을 보여주지 않고 free-text 생성 → TF-IDF cosine으로 어느 옵션과 가장 비슷한지 매칭.

| Mode | Acc% | 비고 |
|---|---:|---|
| MC (logit) | 63.28 | stratified 177 (목표 200, 일부 preprocess fail) |
| MC (generation) | 63.28 | logit/gen match 100% ✓ (sanity) |
| **Open (옵션 미공개)** | **34.46** | random = 25%, **+9.5pt** |
| **Δ (MC − Open)** | **+28.81pt** | **옵션 구조 보너스** |

→ headline 68.4% 중 **약 29pt는 객관식 구조 활용에서 옴**. 진짜 visual+question 이해 상한 ≈ 34.5%. (caveat: TF-IDF 매칭이 의미적 paraphrase 놓칠 수 있어 lower bound 가능성. mean top sim 0.28로 낮음)

### 6.4 Cross-keep-ratio diagnostic (526 × 3 budgets)

| Metric | keep03 (3%) | keep05 (5%) | keep10 (10%) |
|---|---:|---:|---:|
| Accuracy | 65.97 | 64.83 | 67.68 |
| temporal CoM | 0.730 | 0.699 | **0.786** |
| late_half_ratio | 0.759 | 0.739 | **0.834** |
| gt_gaze_recall | 0.026 | 0.041 | 0.077 |
| (random gt_gaze_recall = keep_ratio) | 0.030 | 0.050 | 0.100 |

→ 일관된 패턴:
- **Anti-gaze 패턴 budget과 무관** — 모든 keep ratio에서 gt_gaze_recall ≤ random.
- **Temporal bias는 budget이 커질수록 강해짐** — 토큰이 많아지면 후반에 더 몰림 (0.70 → 0.79). 즉 budget이 작을수록 모델이 시간적으로 더 분산.
- **Accuracy는 budget-monotone 아님** (keep05 64.8 < keep03 66.0) — 작은 keep ratio에 LoRA가 별도 학습되어서 ckpt별 우열 변동. 메서드 trade-off가 단순하지 않음.

---

## 7. 종합 권고 (방법론 관점)

영향력 / 방법론적 중요도 순.

### A. 논문 Table 2 보강 — encoder transformation 시각화

§6.1 soft_oracle 결과로 메시지 명확해짐:
- 논문 Table 2 OnlyGaze (64.64) 와 **GT gaze position 단독 (soft_oracle 64.26)** 가 거의 동일. 즉 *학습된 OnlyGaze score* 는 *raw GT gaze position* 만큼만 함.
- **추가할 행**: "Soft-oracle (GT gaze)" 64.26%, "Hand+Gaze (learned)" 68.44%. 차이 +4.18pp = encoder 의 추상화 기여.
- ~~Anti-gaze supervision~~ → soft_oracle 결과가 narrative 우호적이라 긴급성 낮음. 시간 있으면.
- **태스크별 gaze 유의성**: `gt_gaze_recall`을 태스크로 쪼개기 — M1 parquet에서 재집계 가능 (CPU only, ~30분). 미실행 후속.

### B. Temporal bias 강화 — 토큰 budget 절반은 사실상 무용

§6.2 mask_kept_early −1pt 는 **method의 가장 큰 비효율** 을 노출:
- 학습된 receiver 중 약 50% (전반 프레임) 가 정답률에 −1pt만 기여.
- 즉 keep 10% = 410 토큰 중 ~205개가 "낭비".
- **권장**: encoder/Stage 2에 frame-balanced regularizer를 넣어 전반 frame에서도 informative token을 뽑게 하거나, 혹은 정직하게 "method가 effective budget은 5%" 라고 보고.
- §6.4 cross-keep 결과 (budget↓일수록 temporal bias↓) 와 결합: budget이 충분할 때 모델이 "쉬운 후반 frame"으로 도피하는 경향. 작은 budget이 강제로 분산을 유도.

### C. 논문 평가 protocol 보완 (§6.3 결과 강화)

§6.3 open-ended 결과로 더 강한 권고:
- Headline single-shift 68.44% — 기존 baseline 비교용.
- Permutation 평균 61.74% / consistent-correct 43.73% (§3) — option 위치 노이즈 제거 후 정확도.
- **Open-ended (옵션 미공개) 34.46%** (§6.3) — MC 구조 보너스 제거 후 "진짜 이해" 상한.
- **ECE (0.144)** 도 테이블에 추가 — calibration 축.
- 베이스라인 비교는 동일 MC protocol이라 +4.38pp 가 valid하지만, **절대값은 4가지 평가 모두 보고하는 것이 정직**.

### D. 데이터 확장이 아키텍처 변경보다 먼저

- Stage 1이 overfit 아니므로 데이터 추가는 안전한 선택지.
- `proactive_gaze_triggered_alert` (283) + `proactive_object_appearance_alert` (1,865) = **~37% 학습 데이터 증가** 가능. 재아키텍처 전에 가장 저렴한 실험.
- Stage 1 학습 epoch 추가는 효과 미미 (epoch 90 ≈ 100 거의 평탄).

### E. Spatial selection narrative 점검 (논문 Table 4)

§6.2 shuffle_kept 결과 (+0.95pt vs baseline) 가 의미:
- 모델은 어느 merged token이 어느 receiver position에 있는지 **신경 쓰지 않음** — bag-of-tokens.
- 논문 Table 4 ("Nospatial" 64.64 vs "Spatio-temporal" 67.49 = +2.85pp) 의 spatial pruning 이득이 정말 *spatial 정렬* 덕분인지, 단순히 *어떤* token이 받아들여졌는지의 차이인지 분리 안 됨.
- **권장**: 논문에 "spatial selection isolates salient regions" 같은 강한 주장 자제. 대신 "spatial axis pruning이 token pool 다양성에 기여" 로 톤다운.

### F. Max-cluster-size outlier 검사

`max_cluster_size = 86` (일부 샘플에서 한 receiver가 전체 source의 21% 흡수) — 이런 샘플에서 score 분포가 퇴화된 모드를 가질 가능성. parquet에서 `cluster_size_max` 상위 10개를 뽑아 `viz_token_selection.py`로 시각화하면 failure mode가 드러날 수 있음.

---

## 8. 전략적 우려와 검증 계획 (followup으로 부분 해소)

진단 결과는 두 단계의 더 큰 우려를 제기함. 각각에 대해 분리해서 다룬다.

### 우려 1 — "method가 진짜 작동하는가, 성능만 올린 것인가?"

진단 결과를 두 층위로 분리:

| 층위 | 결과 | 평가 |
|---|---|---|
| **성능 자체** | learned 69.0 vs uniform 61.4 / random 63.5 / center 61.8 / text_only 53.6 | **진짜다.** 모든 reasonable baseline을 5–8pt 차이로 이김. benchmark gaming 아님 |
| **논문이 주장하는 메커니즘** | "gaze가 attention을 중요 패치로 가이드한다" | **이 narrative는 데이터와 충돌.** GT gaze position 자체는 무용 (oracle 54.9 < random) |

→ 즉 *"method가 작동한다"* 는 사실이고, *"왜 작동하는지에 대한 설명"* 이 틀린 상태. 둘은 다른 문제고 다른 처방.

#### Caveat — oracle 구현의 한계

§2의 oracle은 gaze 위치에 1.0, 나머지 0으로 설정. 그러나 keep 410 토큰 중 valid gaze frame은 ~127개뿐이고, 나머지 ~283 자리는 score=0 동률에서 argsort tie-breaking으로 채워짐 (정렬상 앞 인덱스 = 초기 프레임 top-left). **즉 oracle은 "gaze 위치 + 초기 프레임 좌상단" 의 혼합** 이라 순수한 gaze-position 테스트가 아님. 결론 방향은 같지만 (gaze position 단독으로 강한 cue 아님), 결정적 증거가 되려면 **soft oracle (Gaussian-fall around gaze)** 로 재실험 필요.

### 우려 2 — "객관식 (MC) 평가 자체가 문제일 가능성"

진단 결과에 MC 의존성의 흔적이 이미 다수 존재:

| 증거 | 의미 |
|---|---|
| **text_only = 53.6%** (random=25%) | 4지선다라서 언어 prior + 옵션 phrasing 만으로 27pt 회복. 시각 정보 없이 절반은 됨 |
| **C/D 위치 편향 31.6/29.2** vs A/B 20.6/18.6 | 모델이 "찍기" 모드에 들어가면 후반 글자 선호 — 시각 이해와 무관 |
| **agree4 58%, consistent_correct 43.7%** | 헤드라인 68.4% 중 **24.7pt가 옵션 구조에 의존** |
| **logit 기반 채점 (생성 아님)** | A/B/C/D 4개 토큰 logit argmax만 봄 — *왜* 그 답을 골랐는지 검증 안 됨 |
| **ECE 0.144** | 신뢰도와 정확도가 보정 안 됨 — "이해해서" 가 아니라 "logit이 그쪽에 쏠려서" 일 수 있음 |

#### 절대값 vs 상대값 — 결정적 구분

- **좋은 소식 (상대 비교는 공정)**: 비교 baseline들 (attention-pruning, content-merging, frame-selection, full-LoRA) 이 **모두 동일한 MC protocol** 로 평가됨. "방법이 베이스라인보다 +4.38pp 낫다" 는 주장은 MC 한계와 무관하게 성립.
- **나쁜 소식 (절대값은 부풀려짐)**: 68.44% 자체는 부풀려진 수치. "method가 만드는 진짜 marginal gain" 은 4.38pp의 일부일 수 있음.
  - text_only 53.6 → learned 69.0 = +15.4pt = **시각 정보의 총 contribution**
  - 그중 score-driven 부분 ≈ +7.6pt (vs uniform)

### 세 가지 길

| 길 | 비용 | 무엇을 답하나 |
|---|---|---|
| **1. Narrative pivot** | 1–2주 | "gaze attention" → "trajectory-conditioned aggregation" 으로 톤다운. 성능 결과는 그대로 두고 setting만 수정 |
| **2. 메커니즘 규명 후 재프레이밍** | 3–6주 | encoder가 *실제로 무엇을* 학습했는지 정량 규명. narrative를 데이터에 맞춰 재구성 |
| **3. Method 자체 재고** | 6주+ | 길 2의 결과가 "Stage 1 무용"으로 나오면 — 핵심 컴포넌트 재구성 |

### 후속 실험 결과 — 길 결정

§6에서 4개 실험 모두 완료. 결과 기반 길 판정:

| 원래 순위 | 실험 | 결과 | 길 1 (narrative)에 미치는 영향 |
|---:|---|---|---|
| 1 | Distractor 난이도 stratification | **미실행** (CPU only, M1 parquet 활용 후속) | 보류 |
| 2 | Stage 1 ablation (random-init) | **이미 논문 Table 3에 있음** (Nopretrain 65.02) — 재실험 불필요 | Stage 1 +3.4pt 기여 확인됨 |
| 3 | Counterfactual masking ✓ | mask_kept −13pt, shuffle_kept ±0pt, mask_early −1pt | **mixed**: method 작동 (+) , spatial 정렬 무용 (−), 전반 절반 무용 (−) |
| 4 | Open-ended generation ✓ | MC 63.3% → Open 34.5%, Δ +28.8pp | **MC 의존 큼** — 평가 protocol 보완 필요 |
| 5 | Soft-oracle ✓ | 64.26% (vs hard 54.94, vs uniform 61.41) | **narrative 우호적**: GT gaze 약하게 유용 (+2.85pt) |
| (추가) | Cross-keep-ratio diagnostic ✓ | budget 줄여도 anti-gaze 패턴 유지, temporal bias 약화 | 보조 — 패턴의 견고함 입증 |

**길 판정**: 
- **method validity** ✓ — counterfactual mask_kept −13pt가 결정적. 길 3 (재고) 불필요.
- **narrative**: 길 1 (pivot) + 길 2 (selective 메커니즘 규명) 의 **혼합**. 
  - 살아남는 narrative: "gaze trajectory 인코딩이 raw gaze 위치보다 더 유용한 신호를 만든다" (+4.75pt)
  - 톤다운 필요: "spatial selection" (shuffle_kept 결과로 약화), "gaze attention guides patches" (encoder의 추상화 contribution 명시 필요)
  - 보완 필요: 평가 protocol에 open-ended 추가, 또는 적어도 caveat 명시

---

## 9. 산출물 인덱스

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
| **Soft-oracle (§6.1)** | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_soft_oracle_ablation_summary.json`<br>`...E1_keep10_soft_oracle_ablation_soft_oracle_per_sample.parquet` |
| **Counterfactual mask (§6.2)** | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_mask_mask_summary.json`<br>`...E1_keep10_mask_mask_{baseline,mask_kept,mask_kept_late,mask_kept_early,shuffle_kept}_per_sample.parquet` |
| **Open-ended (§6.3)** | `TrajGazeMerge/eval_results/diagnostic/E1_keep10_openend_open_ended_summary.json`<br>`...E1_keep10_openend_open_ended_per_sample.parquet` |
| **Cross-keep diag (§6.4)** | `TrajGazeMerge/eval_results/diagnostic/E1_keep{03,05,10}_diag_summary.json`<br>per-tag analyze 폴더: `E1_keep{03,05,10}_diag/` |
| 실행 로그 | `TrajGazeMerge/eval_results/diagnostic/logs/` |

---

## 10. 재현 명령어

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

# §6.1 Soft-oracle (~3.5h)
CUDA_VISIBLE_DEVICES=0 $PY -m TrajGazeMerge.eval.ablation_score_source \
  --stage1-ckpt $S1 --lora-ckpt $LORA \
  --tag E1_keep10_soft_oracle --sources soft_oracle

# §6.2 Counterfactual masking (~1h)
CUDA_VISIBLE_DEVICES=1 $PY -m TrajGazeMerge.eval.counterfactual_mask_eval \
  --stage1-ckpt $S1 --lora-ckpt $LORA --tag E1_keep10_mask

# §6.3 Open-ended (~2h, 200 stratified)
CUDA_VISIBLE_DEVICES=1 $PY -m TrajGazeMerge.eval.open_ended_eval \
  --stage1-ckpt $S1 --lora-ckpt $LORA \
  --tag E1_keep10_openend --n-samples 200

# §6.4 Cross-keep-ratio (~1.5h, 3 ckpts)
CUDA_VISIBLE_DEVICES=0 bash TrajGazeMerge/eval/run_cross_keep_diagnostic.sh

# §11 Phase 0a-1 (CPU only, ~1분)
$PY -m TrajGazeMerge.eval.analyze_gaze_required_subset

# §11 Phase 0b-1 Frozen-method eval (~30분)
CUDA_VISIBLE_DEVICES=0 $PY -m TrajGazeMerge.eval.frozen_method_eval \
  --stage1-ckpt $S1 --lora-ckpt $LORA --tag frozen_method
```

---

## 11. Phase 0 — 데이터/아키텍처 검증 (가장 중요한 메타-검증)

§6의 결과들이 모두 method 수정에 관한 것이었던 반면, **§11은 method를 검증할 수 있는 setup인지 자체를 점검**.

### 11.1 Gaze-required subset 분석 (Phase 0a-1, CPU only, 기존 parquet 재집계)

본 진단 (§1–§6) 의 모든 결과를 task subset 별로 재집계:
- **gaze_intrinsic**: `past_gaze_sequence_matching` (gaze 패턴 *자체* 가 질문) — n=64
- **conservative**: + `past_non_fixated_object_identification` — n=132
- **liberal**: + `past_scene_recall` + `present_object_identification_hard` — n=233
- **non_gaze**: `present_object_attribute_recognition` + `present_object_identification_easy` + `present_future_action_prediction` — n=291

#### 결과 1: learned 정확도가 gaze-intrinsic에서 우위 없음

| Subset | learned | soft_oracle | learned − soft_oracle |
|---|---:|---:|---:|
| full (526) | 69.01 | 64.26 | **+4.75pt** |
| non_gaze (291) | 69.76 | 62.89 | **+6.87pt** |
| conservative (132) | 66.67 | 67.42 | **−0.76pt** ← soft_oracle 우세 |
| **gaze_intrinsic (64)** | **70.31** | **71.88** | **−1.56pt** ← method가 GT gaze에 짐 |

→ **method의 +4.75pt 우위는 거의 전부 non-gaze 태스크에서 옴.** gaze가 본질인 태스크에서는 raw GT gaze 위치가 learned method를 이김.

#### 결과 2: Gaze-intrinsic에서 모델은 사실상 추측

| Subset | agree4% | consistent_correct% |
|---|---:|---:|
| full | 58.4 | 43.7 |
| non_gaze | 54.6 | 40.2 |
| **gaze_intrinsic** | **32.8** | **26.6** (random=25%) |

→ gaze-intrinsic 64샘플에서 옵션 순서 변경 시 동일 답 비율 32.8%. **랜덤보다 거의 안 나음.** consistent_correct 26.6% = 진짜 이해 비율이 추측 수준.

#### 결과 3: Counterfactual은 gaze-intrinsic에서 *더* 강함 — 그러나 gaze는 아님

| Δ vs baseline | full | gaze_intrinsic | non_gaze |
|---|---:|---:|---:|
| mask_kept | −12.93 | **−15.62** | −10.31 |
| mask_kept_late | −10.65 | −12.50 | −8.25 |
| mask_kept_early | −0.95 | **−4.69** | −0.34 |

→ Receiver의 *정보* 기여는 gaze-intrinsic에서 더 큼 (mask_kept −15.6 vs non_gaze −10.3). 즉 method는 receiver를 *사용* 함. 그러나 §11.1의 결과 1에 따르면 그 정보가 *gaze 위치는 아님*.

#### 11.1 종합 해석 — Method는 gaze가 아닌 다른 시각 신호를 활용

소거법으로 method가 학습한 것:
- ❌ GT gaze 위치 (soft_oracle이 gaze-intrinsic에서 더 잘함)
- ❌ Spatial 정렬 (shuffle_kept ±0pt)
- ❌ Center prior (center 61.79 < random)
- ✓ **Hand trajectory / motion patterns / late-frame action content** 같은 non-gaze 시각 신호를 trajectory encoder의 입력 (gaze+hand) 으로부터 학습한 것으로 보임.

이는 paper의 "gaze guides attention" narrative와 **완전히 다른 메커니즘**.

### 11.2 StreamGaze 데이터셋 검증 verdict

**StreamGaze는 paper의 gaze-attention 주장을 검증할 수 없는 setup**:

1. **gaze-intrinsic subset이 너무 작음** (n=64) — agree4 32.8%로 통계적 신뢰도 한계.
2. **non-gaze subset이 압도적** (291/526 = 55%) — 헤드라인 정확도가 non-gaze 태스크에 dominated.
3. **non-gaze 태스크들은 시각 정보로 풀림** (text_only가 non_gaze에서 56.7%, gaze_intrinsic에서 42.2%) — gaze 없이도 풀리는 태스크가 다수.
4. **Method 한계 vs 데이터셋 한계 분리 불가**: gaze-intrinsic에서 method가 GT gaze보다 진 이유가 (a) method 결함, (b) n=64 노이즈, (c) 학습 데이터 (EgoExoLearn+HoloAssist) 가 EGTEA gaze 분포와 다름 — 셋 다 가능.

### 11.3 Frozen-LLM eval (Phase 0b-1)

526 EGTEA test에서 LoRA 없는 frozen Qwen + method 적용:

| Condition | Acc% | 비고 |
|---|---:|---|
| baseline_frozen (no method, frozen Qwen, 4096 tokens) | **49.24** | LoRA 없음 |
| merge_frozen (method, frozen Qwen, 410 tokens) | **49.05** | LoRA 없음 |
| **Δ (merge − baseline)** | **−0.19pp** | **사실상 동일** |
| *(참고) Nopretrain (Table 3, LoRA only)* | *65.02* | — |
| *(참고) Allloss (Table 2, full method)* | *68.44* | — |

**시사**:
- **Method는 LoRA 없이는 효과 0** — Frozen Qwen + method ≈ Frozen Qwen 단독.
- **4096 → 410 토큰 압축이 frozen LLM 입장에서 무손실** — Qwen이 video token을 bag으로 다룬다는 §6.2 shuffle_kept 결과와 정합.

**기여도 재분해**:

| 추가 요소 | 누적 acc% | 단독 기여 |
|---|---:|---:|
| Frozen Qwen + 4096 토큰 (baseline) | 49.24 | — |
| + LoRA (full 토큰) | ≈ 65.02 | **+16pp** ← LoRA 단독 |
| + Stage 1 + merge (전체 method) | 68.44 | **+3.4pp** ← method 단독 |

→ **LoRA가 method보다 5배 큰 contribution**. Method의 marginal value는 +3.4pp.

추가 관찰: text_only (LoRA + 0 visual) = 53.6% **>** baseline_frozen (no LoRA + full visual) = 49.2% — Qwen2.5-VL의 video token handling 자체가 LoRA fine-tune에 강하게 의존.

### 11.4 Phase 0 종합 verdict

#### Method가 *하는* 일 (확정)
- **LoRA와 co-adapted 된 token compression**: 10× 압축 + 정확도 유지 + 3.4pp 추가 (vs LoRA-full)
- **Non-gaze 시각 신호 활용**: hand trajectory, motion, late-frame action — gaze trajectory **입력**의 non-gaze 성분
- **Receiver 정보 사용 확인**: counterfactual mask_kept −13pt

#### Method가 *안 하는* 일
- ❌ Gaze 위치를 따라가 important 패치 선택 (soft_oracle이 gaze-intrinsic에서 이김, gt_gaze_recall < random)
- ❌ Spatial 정렬 유지 (shuffle_kept ±0pt)
- ❌ Frozen LLM에 의미 있는 신호 전달 (Δ −0.19pp)

#### 데이터셋 검증
- ❌ StreamGaze가 gaze-attention 주장 검증에 부적합 (gaze-intrinsic n=64, agree4 32.8%)

### 11.5 새 narrative 후보

**옵션 A — 보수적 (정직 우선)**
"Behavioral-score-driven token compression for VLMs. Achieves 10× compression while preserving accuracy via LoRA co-adaptation. The behavioral score acts as a learned prior; we observe limitations in raw gaze utilization (oracle / cross-subset analyses)."
- 솔직히 contribution은 작아지지만 reviewer 공격 방어 가능. Phase 0 결과를 limitation으로 명시.

**옵션 B — Pivot (강한 contribution)**
"Bag-of-tokens VLM behavior under aggressive compression: spatial alignment is largely ignored; what matters is *which* tokens are kept, scored by behavioral context. We propose score-driven retention as a simpler alternative to spatial merging."
- shuffle_kept ±0pt + frozen-method 결과를 contribution으로 reframe. **method 자체 단순화** 제안. 차기 venue 새 contribution.

**옵션 C — Cross-dataset 검증 우선**
StreamGaze 외 데이터셋 (EgoSchema, OpenEQA) 추가 평가로 일반화 입증 → 그 결과에 따라 옵션 A 또는 B 결정.

### 11.6 다음 단계 우선순위

| 우선순위 | 실험 | 답하는 질문 | 비용 |
|---:|---|---|---|
| **1** | **다른 egocentric VQA 데이터셋 평가** (EgoSchema, OpenEQA, Ego4D-NLQ) | StreamGaze 외에서 method가 재현되는가? | 2–3주 (data + adapter) |
| **2** | **Score-only baseline (merge 제거)** | shuffle_kept를 contribution으로 검증 — top-k 단순 pruning이 merge와 동등한가? | 1주 (학습 + eval) |
| **3** | **Gaze-intrinsic test 확장** (EGTEA-action 활용 자체 생성) | n=64 → n=500+ 에서 gaze 우위가 보이는가? | 1주 |
| 4 | **Frame-balanced regularizer** | mask_early −1pt 문제 해결 가능한가? | 3–5일 |
| 5 | Cross-VLM (LLaVA-Next, InternVL) | Qwen 의존성 | 3–4주 |

**가장 중요한 메시지**: Phase 0 결과로 paper narrative가 **데이터와 정합적이지 않음** 이 확인됨. 다음 venue submission 전에 narrative 재설계 필수 (옵션 A/B 결정).

---

## 12. Phase M1 — 메커니즘 규명 (encoder가 진짜 학습한 것)

§7–§11이 "method가 의도대로 작동하는가" 를 외부에서 측정했다면, §12는 **encoder가 실제로 무엇을 학습했는지** 직접 진단.

### 12.1 가설 (Phase M1 설계 시점)

| 가설 | 내용 |
|---|---|
| **H1 Hand-tracking** | Encoder가 hand 위치를 따라가 spatial token을 선택 |
| **H2 Temporal late-bias** | Encoder가 단순히 후반 프레임에 토큰을 몰아줌 (spatial은 무관) |
| **H3 Hand-object interaction** | Encoder가 hand-gaze convergence 시점/위치에 집중 |

### 12.2 M1.1 + M1.3 — Hand-position recall (526 EGTEA)

기존 `gt_gaze_recall` 을 hand 위치로 확장:

| Recall (random = keep_ratio 0.10) | Mean | Median |
|---|---:|---:|
| gt_gaze_recall | 0.077 | 0.073 |
| gt_hand_left_recall | **0.100** | 0.047 |
| gt_hand_right_recall | 0.053 | 0.000 |
| gt_hand_mid_recall | 0.084 | 0.000 |
| gt_hand_either_recall | 0.076 | 0.054 |
| frame_center_recall | 0.048 | 0.047 |

→ **H1 기각**. Hand 위치 recall이 random (0.10) 보다 *낮음*. Encoder는 hand도 따라가지 않음. By-correctness 차이도 ±0.005 수준으로 무의미.

### 12.3 M1.2 — Score–trajectory feature correlation (526 EGTEA)

각 샘플별, `kept_per_frame` 과 trajectory-derived per-frame feature의 Pearson r:

| Feature | mean_corr | median | pct strong-positive (r > 0.3) |
|---|---:|---:|---:|
| **frame_index** | **0.498** | **0.509** | **98.1%** |
| gaze_presence | 0.068 | 0.088 | 0.4% |
| hand_presence | 0.055 | 0.003 | 15.5% |
| gaze_speed | −0.040 | −0.080 | 4.6% |
| gaze_to_center | −0.034 | −0.030 | 2.5% |
| hand_velocity | 0.025 | −0.039 | 11.1% |
| right_velocity | 0.021 | −0.038 | 8.7% |
| left_velocity | 0.019 | −0.042 | 8.9% |
| **convergence** | **0.001** | −0.008 | 3.5% |

By-correctness diff는 모두 |Δ| < 0.02 (frame_index 포함). 즉 sample-agnostic.

### 12.4 Verdict — **H2 (temporal late-bias) 확정**

**Encoder는 trajectory dynamics를 frame-level temporal salience curve로 변환할 뿐. Spatial 위치는 사실상 무관.**

이는 §1–§11의 모든 관찰을 단일 메커니즘으로 설명:

| 진단 결과 | H2로 설명 |
|---|---|
| temporal_CoM 0.78 (§1) | 직접 |
| late_half_ratio 0.83 (§1) | 직접 |
| shuffle_kept ±0pt (§6.2) | spatial 무관하므로 셔플 무영향 |
| mask_kept_early −1pt (§6.2) | 전반 frame은 가중치 낮음 |
| mask_kept_late −11pt (§6.2) | 후반 frame이 답 정보 보유 |
| frozen Δ −0.19pp (§11.3) | 프레임 압축은 frozen LLM이 안 씀 |
| learned 69.0 vs uniform 61.4 (§2) | uniform은 argsort tie-break 으로 *early* frame 선호 → 잘못된 시간 분포 |
| soft_oracle 64.26 (§6.1) | gaze 위치 분포는 약한 시간 신호만 줌 |
| gt_gaze/hand_recall < random (§12.2) | spatial은 무관하므로 어디든 회피해도 무영향 |
| gaze-intrinsic 성능 ≈ random (§11.1) | spatial gaze 따라가기는 못 함 |

### 12.5 추가 발견 — Question-conditioning 부재

`codebase_overview.md` 함정 #2에 따르면 `query_emb = torch.zeros(...)`. 즉 **encoder는 질문을 보지 않음**. M1.2의 by-correctness diff가 모두 zero에 가까운 것이 이를 입증 — 같은 trajectory에 대해 encoder 출력은 항상 같음 (question 무관).

이는 method의 근본적 한계: "behavioral score" 가 본질적으로 *질문 conditional* 이 아니라 *trajectory conditional 시간 prior*.

### 12.6 새 정확한 method 설명

> "TrajGazeMerge encoder는 (gaze + hand) trajectory dynamics를 입력받아 frame-level **temporal salience curve** 를 출력. Spatial 차원은 학습 supervision (`I_scores_past`) 과 분리되지 않은 채 학습되지만, 결과적으로 LLM에 의해 활용되지 않음. Score-weighted bipartite merge는 사실상 *late-biased temporal compression* 으로 작동하며, LoRA가 이를 활용해 +3.4pp 의 marginal contribution 을 만든다."

→ 헤드라인 +4.38pp ablation 이득은 전부 **temporal bias** 의 효과로 환원 가능.

---

## 13. 기존 narrative를 실제로 작동시키려면? (Path-Forward)

§12에서 encoder가 spatial-gaze-attention이 아닌 temporal-only 임이 확인됐으므로, paper의 "gaze attention guides important patches + spatial selection isolates salient regions" narrative를 **실제로 작동하게 만들려면** 다음 변경이 필요:

### 13.1 근본 원인 (지금 왜 안 되는지)

| 진단 | 원인 |
|---|---|
| Spatial 선택이 random | Stage 1 supervision (`I_scores_past`) 가 spatial 위치를 강하게 강제하지 않음 |
| 질문 무관 score | `query_emb = zeros` → encoder는 question을 못 봄 |
| Bag-of-tokens 행동 | Qwen2.5-VL이 video token의 spatial 정렬을 약하게 사용 |
| Temporal bias 폭주 | 학습 신호의 강한 prior가 시간축에 집중 (action이 후반에 일어남) |

### 13.2 권장 개입 (영향력/비용 순)

#### A. **Question-conditioning 활성화** ⭐⭐⭐ (가장 큰 ROI)

현재 query encoder가 zero embedding을 받음. 이를 실제 question 임베딩으로 교체:
- Stage 1 학습 시 `query_emb = qwen.text_encoder(question)` 또는 별도 작은 text encoder
- 자연스럽게 sample 간 variation 추가
- M1.2 by-correctness diff가 양수로 갈 것 (질문에 따라 score 다름)
- **비용**: Stage 1 재학습 (~1시간) + Stage 2 재학습 (~수 시간)
- **리스크**: 낮음 (구조 변경 미미)

#### B. **Sharp spatial supervision** ⭐⭐⭐

현재 `I_scores_past` 는 hand+gaze interaction 기반 soft score. 이를 더 강하게 localize:
- GT gaze + hand 위치에 sharp Gaussian (σ ≈ 0.05) 만 1로 두고 나머지 0
- Hard cross-entropy loss로 학습 (KL 대신)
- Stage 1이 끝나면 `score(t,p)` 가 명확히 gaze/hand 패치를 가리키게 됨
- 이후 Stage 2에서 LoRA가 spatial 정보 사용을 학습해야 함
- **비용**: Stage 1 재학습 (~1시간) + 데이터셋 라벨링 작업 없음 (이미 존재하는 gaze/hand 위치 활용)
- **리스크**: Spatial selection이 학습된 후 LoRA가 이를 어떻게 활용할지가 불확실

#### C. **Anti-bag-of-tokens 학습 신호** ⭐⭐

Stage 2 학습 중 random shuffle augmentation 추가:
- 50% 확률로 merged_video를 random permute 후 forward
- Permuted 입력의 정답률이 baseline보다 낮아야 한다는 loss (KL divergence 또는 margin)
- 이는 LLM이 spatial 정렬에 의존하도록 강제
- **비용**: Stage 2 재학습 ~2배 시간 (각 sample에 2 forward)
- **리스크**: LoRA 학습 instability 가능성

#### D. **Temporal-spatial 분리 아키텍처** ⭐⭐

현재 score는 단일 (T, 196) score. 이를:
- `temporal_score[T]` (frame-level weight)
- `spatial_score[T, 196]` (patch-level within frame)
- 최종 score = temporal_score * spatial_score
- Temporal과 spatial supervision 분리

이렇게 하면 spatial 학습이 temporal 학습에 흡수되지 않음. 현재 단일 score path가 spatial을 시각화하긴 어렵게 만듦.
- **비용**: 아키텍처 수정 (1주) + 재학습
- **리스크**: 중간

#### E. **Tighter budget으로 spatial 강제** ⭐

keep ratio을 1–3%로 줄이면 모델이 시간 분산만으론 못 살아남고 spatial 선택을 학습해야 함.
- §6.4 cross-keep 결과: keep03 = 65.97% (10%에서 67.68% 대비 −1.7pt), temporal_CoM 0.73 (10%에서 0.78 대비) — tight budget에서 분산 확대 시작
- 더 tight하게 (e.g., keep ≤ 5%) + 위 A/B 변경 결합
- **비용**: 학습 1–2 runs

#### F. **다른 VLM backbone**

Qwen2.5-VL이 video token bag-of-tokens 경향을 가짐 (§6.2 shuffle_kept). LLaVA-Next 또는 InternVL2처럼 spatial 의존성이 더 강한 VLM에선 spatial selection이 작동할 수도.
- **비용**: 3–4주 (VLM porting + adapter)
- **리스크**: 높음 (다른 quirk 발견 가능)

### 13.3 권장 진행 순서 (paper narrative 복원)

**Sprint 1 (1–2주): A + B 결합**
1. Query embedding 활성화 + sharp spatial supervision
2. Stage 1 재학습 (1 run, ~1시간)
3. Stage 2 재학습 (1 run, ~수 시간)
4. 본 진단 (§1, §12) 재실행 → gt_gaze_recall, shuffle_kept Δ, frame_index corr 측정
5. **성공 기준**:
   - gt_gaze_recall > 0.20 (현재 0.077 → 2.6x)
   - shuffle_kept Δ < −3pt (현재 ±0)
   - frame_index corr < 0.3 (현재 0.50)
   - accuracy ≥ 67% (성능 유지)

**Sprint 2 (1주): 결과 보고 C/E 결정**
- A+B 결과가 좋으면 → Sprint 3로 직행
- spatial selection은 살았지만 정확도 떨어졌으면 → C (shuffle augmentation) 추가
- 여전히 spatial 안 살아나면 → E (tighter budget) 또는 F (다른 VLM)

**Sprint 3 (2–3주): D + 외부 데이터셋 검증**
- Temporal-spatial 분리 아키텍처로 명확한 ablation 가능하게
- EgoMCQ 또는 EgoSchema 등에서 cross-dataset 검증

**Sprint 4 (1–2주): Paper rewrite**
- 새 method section: question-conditional, spatially-localized score
- 새 ablation tables: spatial 진짜 작동 입증 (gt_gaze_recall, shuffle, mask 진단 결과)
- 기존 narrative 살아남음

### 13.4 결정 트리

```
Sprint 1 결과
├── gt_gaze_recall > 0.20 & shuffle Δ < −3 & acc ≥ 67%
│    → 기존 narrative 복원 가능 ✓
│      Sprint 3로 진행 (cross-dataset 검증)
│
├── spatial 살았으나 acc < 67%
│    → trade-off 존재
│      Sprint 2의 C (shuffle aug) 또는 budget 조정으로 회복 시도
│
└── gt_gaze_recall < 0.15 & shuffle Δ ≈ 0
     → Qwen-VL 자체가 spatial 무시
       → F (다른 VLM) 검토, 또는 narrative pivot (옵션 A/B, §11.5)
```

### 13.5 솔직한 위험 평가

이 방향은 "encoder를 의도대로 작동시키는" 데 성공할 가능성과, 그래도 "Qwen-VL의 bag-of-tokens" 한계 때문에 spatial selection이 헤드라인 성능 향상으로 이어지지 않을 가능성이 모두 있음. §6.2 shuffle_kept ±0pp + §11.3 frozen Δ −0.19pp 두 결과는 후자를 강하게 시사. 따라서:

- **낙관적**: Sprint 1 A+B로 spatial selection 활성화 + +1pp 정도 acc 증가. Narrative 살아남음.
- **현실적**: Spatial 활성화는 성공하지만 acc는 유지/약간 감소. "Method works as designed" 정성 분석 추가 가능하지만 헤드라인 숫자 변동 미미.
- **비관적**: Qwen이 spatial을 무시하는 한 spatial selection은 어떤 형태로든 inert. Sprint 2에서 F (다른 VLM) 으로 가야 함.

세 시나리오 모두 paper 방어 가능하지만, 비관 시나리오는 시간이 가장 많이 듦 (3–6주 추가).

**핵심 권고**: **Sprint 1 A+B를 1–2주 안에 실행 후 결정**. 가장 적은 비용으로 narrative 복원 가능성을 답함.
