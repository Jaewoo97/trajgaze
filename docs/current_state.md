# 현재 상황 — 한 페이지 요약

**갱신**: 2026-05-27 (Step 1 cf-mask 진단 + Step 3 paper reframe 완료) | **상태**: Plan `zazzy-sprouting-ladybug` 완전 실행 종료, paper rewrite 단계로 이동.

> 본 문서만 읽어도 전체 상황 파악 가능. 세부는 §끝의 doc navigation 따라가면 됨.

---

## 0. Methodology glossary — 표 읽기 전에 필요한 정의

### 0.1 메소드 (학습/구조)

| 약어 | 풀네임 | 한 줄 정의 | 코드 위치 |
|---|---|---|---|
| **TAS** | Trajectory-Aware Selection | gaze 좌표를 입력받아 **각 패치에 Gaussian-prior score**를 부여하는 학습 가능한 encoder. Stage-1 supervised로 학습 (gaze trajectory → patch importance map), Stage-2 학습 중에는 frozen. 이 score를 `gaze_weighted_merge` 가 받아서 **상위 토큰만 남기고 나머지를 merge** (10× 압축, `merge_ratio=0.9`). | `TrajGaze_v2/models/model_temporal.py` |
| **ATR** | Attention Temporal Regularization | Stage-2 학습에 추가되는 보조 loss. LLM attention이 시간적 순서 정보를 살리도록 압력 (λ=0.5). | `TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py` |
| **CGM** | Counterfactual Gaze Margin | Stage-2 학습에 추가되는 보조 loss. GT gaze 위치를 셔플한 counterfactual 입력의 GT logit이 진짜 입력보다 margin 이상 작도록 강제 (λ=0.3, prob=0.3). visual content에 대한 sensitivity 강제 목적. | 같은 파일 |
| **Stage-1 ckpt** | `E1_combined_AB_TAS/best.pth` | A (visual encoder) + B (query encoder) + TAS 만 학습한 frozen base. cf-mask가 `--stage1-ckpt`로 받음. | — |
| **Stage-2 ckpt** | `E1_combined_*/best.pth` | Stage-1 frozen 위에 **Qwen-VL을 LoRA finetune** (target = q/k/v/o_proj, rank 16) + 위의 보조 loss 조합. 이게 평가 대상 4개 ckpt. | — |
| **merge_ratio = 0.9** | — | 입력 visual token 중 **90%를 merge로 흡수**하고 10%만 LLM에 전달. 본 paper의 "efficient" 정의. | — |

### 0.2 cf-mask variants (진단 도구)

cf-mask = **counterfactual masking eval**. 같은 ckpt·같은 입력에서, **LLM에 들어가는 압축된 visual token을 일부만 zero/shuffle 한 뒤 정확도 변화 (Δ)** 를 측정. `Δ < 0` 이 클수록 그 토큰이 정답에 필수적 = LLM이 visual을 *진짜* 쓴다는 증거.

| Variant | 어떤 토큰을 어떻게 건드리나 | 검증 가설 |
|---|---|---|
| `baseline` | 건드리지 않음 (원본) | 절대 정확도 기준선. |
| **`mask_kept`** | **남긴 토큰 전부 = 0 벡터** (구조·길이는 유지) | **Language-prior floor.** Δ ≈ 0 이면 LLM은 visual content 없이도 같은 답을 냄 = visual 안 봄. |
| `mask_kept_late` | 남긴 토큰 중 **후반 절반 (frame_idx ≥ T/2) 만 zero** | 후반 프레임 의존도. |
| `mask_kept_early` | 남긴 토큰 중 **전반 절반만 zero** | 전반 프레임 의존도. |
| `shuffle_kept` | 남긴 토큰을 **무작위로 섞음** (zero 안 함) | 순서 의존성. Δ > 0 이면 *섞었을 때 더 잘 함* = bag-of-tokens 회귀. |
| `mask_gaze` | 토큰 중 **gaze 좌표 근방 (radius=0.2) 만 zero** | gaze 영역 자체의 정답 기여도. |
| `mask_hand` | 토큰 중 **양손 좌표 근방만 zero** | hand 영역의 정답 기여도. |

**읽는 법 예시**: `TAS-only / StreamGaze: mask_kept Δ = −11.98` →
"남긴 토큰을 전부 0으로 만들면 정확도가 11.98pp 떨어진다 → LLM이 그 토큰의 시각 내용을 진짜 활용한다."
반대로 `TAS-only / EgoGazeVQA: mask_kept Δ = +0.93` →
"0으로 만들었더니 *오히려* 더 잘 한다 → LLM이 시각 내용 없이도 답을 알아내며, 시각 토큰이 noise 역할."

코드: `TrajGazeMerge/eval/counterfactual_mask_eval.py:132-` `apply_mask_variant`.

---

## 1줄 요약 (2026-05-27)

> **TAS-only가 모든 셋업에서 best.** ATR/CGM은 EgoGazeVQA에서 "LLM에 visual을 약하게 주입"하는 효과는 진짜지만, StreamGaze 손실이 더 커서 mean에서 진다. cf-mask 진단으로 **EgoGazeVQA는 language-prior 지배 데이터셋이고 plan 안의 메소드로 그걸 깰 수 없다**는 게 확정됨. paper는 **TAS = headline, ATR/CGM = honest ablation, EgoGazeVQA = §Limitations** 로 reframe.

---

## 한눈에 보는 결과 (best ckpt)

### 셋업 A — 2 dataset (StreamGaze + EgoGazeVQA)

| ckpt | StreamGaze | EgoGazeVQA | **mean** |
|---|---:|---:|---:|
| **TAS-only** ★ | **67.49** | 57.77 | **62.63** |
| TAS+ATR | 64.26 | ~58 | ~61 |
| TAS+ATR+CGM (FULL) | 61.98 | **59.40** | 60.69 |
| CGM-only | 62.74 | 55.92 | 59.33 |
| Sprint-1 baseline | 65.21 | 57.31 | 61.13 |

### 셋업 B — 3 dataset (+HD-EPIC, 2 ckpt만 학습)

| ckpt | StreamGaze | EgoGazeVQA | HD-EPIC | **mean** |
|---|---:|---:|---:|---:|
| **TAS-only-hdepic** ★ | 63.69 | 55.92 | 50.12 | **56.57** |
| TAS+ATR-hdepic | 60.65 | 54.76 | 50.66 | 55.35 |

**두 셋업 모두 TAS-only가 1위.** ATR 추가는 mean 손해 (−1.94 / −1.22 pp).

---

## 핵심 진단 — cf-mask matrix (2026-05-27 finished)

post-LoRA finetuned ckpt 4개에 대해 시각 토큰을 zero/shuffle했을 때 정확도 변화. **mask_kept Δ < 0 = LLM이 visual을 진짜 쓴다는 증거.**

### StreamGaze (visual-heavy)

| ckpt | base | mask_kept | shuffle_kept |
|---|---:|---:|---:|
| TAS-only | 67.49 | **−11.98** ✓ | −0.38 |
| TAS+ATR | 64.26 | −11.22 | +0.38 |
| TAS+ATR+CGM | 61.98 | −7.79 | **+1.33** ⚠ |
| CGM-only | 62.74 | −10.08 | 0.00 |

→ TAS는 visual을 강하게 쓰게 만듦. FULL은 shuffle이 *돕는* (bag-of-tokens 회귀) anomaly.

### EgoGazeVQA (language-prior dominated)

| ckpt | base | mask_kept | language-only가 더 좋은가? |
|---|---:|---:|---|
| **TAS-only** | 56.38 | **57.31** | **YES (+0.93)** ⚠ |
| **CGM-only** | 55.92 | **56.62** | **YES (+0.70)** ⚠ |
| TAS+ATR | 57.08 | 55.45 | NO (visual +1.62) |
| TAS+ATR+CGM | 59.40 | 57.31 | NO (visual +2.09) |

→ **TAS-only/CGM-only는 EgoGazeVQA에서 visual이 *오히려 noise*.** ATR/CGM 결합이 들어가야 LLM이 비로소 visual을 *조금* 보긴 함 (+2.09 vs StreamGaze의 +11.98 = 1/6).

### Decision gate (plan §1)

EgoGazeVQA language-prior floor 4 ckpt spread = **1.86 pp < 2 pp gate**.
→ "메소드가 LLM 행동 못 바꿈" 확정 → **Step 2 (LoRA FFN 확장) 안 가고 Step 3 (paper reframing) 로 직행.**

---

## Plan 진행도 (`zazzy-sprouting-ladybug.md`)

| 단계 | 산출물 | 상태 |
|---|---|---|
| §1 Action 1 — cf-mask 4 ckpt × 2 val | 8 summary json | ✅ 2026-05-27 |
| §1 Action 2 — text-only ablation | `mask_kept` Δ 를 proxy로 (numerically 동등) | ⚠ 엄격 ver 미실행, 필요시 30min/ckpt |
| §1 산출물 | `docs/visual_grounding_diagnosis_v2.md` | ✅ |
| §1 Decision gate | spread 1.86 pp < 2 pp → Step 3 | ✅ |
| §2 (LoRA FFN 확장) | (fallback) | ⏸ gate상 안 돌림 |
| §3 — paper reframe | `docs/paper_narrative_v3.md` | ✅ |

---

## Paper narrative v3 (현재 결정)

**Headline**: "Trajectory-Aware Selection (TAS) lets a 90%-pruned visual stream remain useful for egocentric video QA — when the dataset actually requires visual reasoning."

| 메소드 | Paper에서의 역할 |
|---|---|
| **TAS** | Headline contribution. StreamGaze base 67.49, `past_gaze_sequence_matching` +20 pp, mask_kept Δ −11.98. |
| **ATR** | Ablation. StreamGaze base −3.23 pp 손해, visual grounding 거의 안 바꿈, HD-EPIC mean 손실. |
| **CGM** | Ablation. EgoGazeVQA sign-flip은 진짜이지만 StreamGaze −5.51 / shuffle +1.33 trade-off로 순손익 음수. |
| **EgoGazeVQA** | §Limitations로 강등. "spatial/temporal questions partially derivable from gaze metadata; cf-mask Δ confirms." |

**StreamGaze = 유일한 headline-table benchmark.** EgoGazeVQA는 §Analysis/§Appendix에 cf-mask Δ 컬럼 붙여서 limitation 명시.

---

## 약점 (reviewer가 공격할 가능성)

1. **Single visual-heavy benchmark.** StreamGaze 1개로 main claim 입증. "cherry-pick 아니냐" 공격 가능 → **다음 plan 후보: NExT-QA hard / EgoSchema / EGTEA action recognition 추가**.
2. **shuffle_kept +1.33 anomaly** (FULL/StreamGaze). v3 §Discussion에서 "loss-mixing이 order-agnostic regime 유도" 로 honest framing.
3. **strict text_only 미실행.** mask_kept Δ를 floor proxy로 썼는데, reviewer가 정확한 text_only 숫자 요구할 가능성 — 30min/ckpt 추가.

---

## 남은 옵션 (사용자 판단 영역)

| 옵션 | 비용 | 효과 |
|---|---|---|
| **A. 여기서 stop, paper draft 본격 수정** | 0 | Plan 다 끝남. 가장 합리적 default. |
| **B. NExT-QA hard 또는 EGTEA action 추가 plan** | 코드 추가 + finetune 1 run + cf-mask ~수일 | **본질적 약점 해소.** Reviewer 공격 1번 막음. |
| **C. HD-EPIC ckpt 2개에 cf-mask** | ~1h | §2 ablation row 보강. 거의 sure thing. |
| **D. strict text_only ablation** | 30min × 4 ckpt | §Limitations 정확성. reviewer 대응 보험. |
| **E. LoRA FFN 확장 1 run (plan §2)** | ~27h × 2 GPU | "capacity가 진짜 bottleneck 아니다" 추가 증거. gate 통과 못 한 fallback. |

**추천 순서**: B → C → D → E. A는 ASAP.

---

## Doc navigation (어디부터 읽나)

1. **본 문서** — 1페이지 요약.
2. [`paper_narrative_v3.md`](paper_narrative_v3.md) — paper 어떻게 다시 쓸지 (구조 + 표 초안).
3. [`visual_grounding_diagnosis_v2.md`](visual_grounding_diagnosis_v2.md) — cf-mask 4×2 매트릭스 + decision gate 적용.
4. [`trajectory_grounded_results.md`](trajectory_grounded_results.md) — 직전 narrative (v2, 셋업 A 결과만). v3가 이걸 대체.
5. [`combined_training_results.md`](combined_training_results.md) — Sprint 1 baseline 학습 세부.
6. [`journey_summary.md`](journey_summary.md) — 2026-05-13 ~ 05-20 진단 5 sprint 흐름 (오래된 narrative 포함, 컨텍스트용).
7. [`sprint2_path_forward.md`](sprint2_path_forward.md), [`sprint1_path_forward.md`](sprint1_path_forward.md) — 당시 의사결정 기록.

**Plan 원본**: `/home/irteam/.claude/plans/zazzy-sprouting-ladybug.md`.
**이전 plan**: `/home/irteam/.claude/plans/streamgaze-egogazevqa-swirling-koala.md`.

---

## Background / 자동화 산출물 (참고)

- `TrajGazeMerge/eval/convergence_watcher.sh` — train_log 폴링해서 수렴시 자동 stop (2026-05-26 21:15 UTC 발화).
- `TrajGazeMerge/eval/run_step1_diagnosis.sh` — 학습 종료 대기 후 cf-mask 8슬롯 자동 실행 (2026-05-27 05:47 UTC 완료).
- `TrajGazeMerge/eval_results/diagnostic/E1_combined_*_cfmask_mask_summary.json` — 8개 cf-mask summary.
- `TrajGazeMerge/checkpoints/E1_combined_{TASonly,TAS_ATR}_hdepic_bs8_mb2/best.pth` — 셋업 B best ckpt.
