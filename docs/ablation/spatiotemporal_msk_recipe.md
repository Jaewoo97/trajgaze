# TrajGaze — 단일 Spatio-Temporal 인코더 학습 레시피 (실험 기록)

본 문서는 `TrajGazeV2Temporal` 인코더의 **안정적 단일 인코더 학습**을 만들기 위한
일련의 실험 결과를 정리한다. 출발점은 doc/ablation의 세 보고서가 보여준
"random-init full 모델은 두 단축 ablation보다 6.8× 나쁘게 수렴" 문제였다.

**Branch:** `spatiotemporal_msk` (`main`에서 분기)

**핵심 파일**
- [TrajGaze_v2/models/encoder_temporal.py](../../TrajGaze_v2/models/encoder_temporal.py) — gated residual + parallel frame-score branch + post-fusion iframe
- [TrajGaze_v2/training/loss_schedule.py](../../TrajGaze_v2/training/loss_schedule.py) — step 기반 curriculum 가중치
- [TrajGaze_v2/training/stage1_temporal.py](../../TrajGaze_v2/training/stage1_temporal.py) — 모든 trainer flag, warm-start, gate logging
- [TrajGaze_v2/data/dataset_streamgaze_stage1.py](../../TrajGaze_v2/data/dataset_streamgaze_stage1.py) — 누락 position 로더 복원

---

## 출발점: 본 작업이 다루는 문제

| Model | Stage-1 val loss | Best stage-2 acc |
|---|---|---|
| Full joint (random init) | **0.1274** | 67.49% |
| Spatial-only (`no_temporal`) | 0.0188 | 68.06% |
| Temporal-only (`no_spatial`) | 0.0188 | 68.82% |

Full joint 학습이 6.8× 나쁨 → stage-2 LoRA 신호가 약함. 진단:
**non-stationary query 분포 + 4-loss 간섭**.

---

## 도입한 개선 장치들

본 작업의 변경은 **3개 레이어** (Architecture / Loss / Training procedure) + 데이터 보조에 걸쳐 있음.
어느 레이어가 winner의 핵심 동력이었는지 다음 표에 정리.

### 변경 사항 분류

| # | 변경 | 레이어 | 파일 | 추가 params |
|---|---|---|---|---|
| 1 | Gated residual around InterFrameTransformer (`inter_frame_gate`) | **Encoder** | `encoder_temporal.py` | +1 |
| 2 | Parallel frame-score branch (`frame_attn_pool`, `frame_score_head`) | **Encoder** | `encoder_temporal.py` | +1K |
| 3 | Post-fusion InterFrameTransformer (`inter_frame_post`) | **Encoder** | `encoder_temporal.py` | +4.7M |
| 4 | Loss curriculum (step-aware 4-loss ramping) | **Loss** | `loss_schedule.py` | 0 |
| 5 | Loss subset (`--drop-loss-score-traj`) | **Loss** | `stage1_temporal.py` | 0 |
| 6 | Warm-start (`--init-from`, strict=False) | **Training** | `stage1_temporal.py` | 0 |
| 7 | No-visual mode (`--no-visual`) | **Training** | `stage1_temporal.py` | 0 |
| 8 | Gate config (`--gate-init`, `--freeze-gate`) | **Training** | `stage1_temporal.py` | 0 |
| 9 | Cosine T_max override on resume | **Training** | `stage1_temporal.py` | 0 |
| 10 | Restored `_load_raw_positions` (사라진 모듈 복원) | **Data** | `dataset_streamgaze_stage1.py` (신규) | 0 |

### 각 변경에 대한 요약

**Architecture (인코더 자체 변경)**

1. **Gated residual `InterFrameTransformer`**: `tanh(gate)`로 mixing 정도 조절. `gate=0`에서 시작하면 InterFrameTransformer 완전 bypass → 첫 step에 spatial-only ablation과 동일.
2. **Parallel frame-score branch**: InterFrameTransformer 출력에서 분기, attention pool → frame-level scalar → per-patch score에 broadcast multiply. **(EA1 winner의 핵심)**
3. **Post-fusion iframe**: VisualFusion 다음에 두 번째 InterFrameTransformer. EB2에서만 사용, EA1보다 못함 → 최종 best config에서 미사용.

**Loss (학습 신호 변경)**

4. **Loss curriculum**: 4-loss를 step 기반으로 ramping. 0–10% `l_traj`만, 10–25% `+l_score_past`, 25–50% `+l_score_fut`, 50–100% `+l_score_traj`. **(EA1-C에서 추가 사용)**
5. **Loss subset (`--drop-loss-score-traj`)**: 가장 noisy한 `l_score_traj` 영구 weight=0. R-C에서 score_past 13% 개선했으나 stage-2 inference path가 학습 안 됨 → 최종 best config에서 미사용.

**Training procedure (학습 절차/하이퍼)**

6. **Warm-start (`--init-from`)**: 기존 checkpoint weights만 strict=False로 로드.
7. **No-visual mode (`--no-visual`)**: DINOv2 forward 스킵 — V2 진단 검증용.
8. **Gate config 플래그**: `--gate-init`, `--freeze-gate`로 ablation 컨트롤.
9. **Cosine T_max override on resume**: 학습 연장 시 LR 스케줄 적절히 재계산.

**Data**

10. **`dataset_streamgaze_stage1.py` 복원**: 본 codebase에 빠져있던 `_load_raw_positions` 함수를 TrajGazeMerge의 `_load_traj`를 참조해 30-50줄로 재구성.

---

## Winner config 분해 — 어느 레이어가 효과를 만들었나?

### EA1 (sum_4 최저, stage-2 안전한 winner)

```
변경 사항:
  Architecture:  ✓ gated residual (gate=0 frozen) + ✓ parallel frame-score branch
  Loss:          ✗ (변경 없음 — 기본 4-loss 균등합)
  Training:      ✗ (변경 없음 — random init, no curriculum, no warm-start)
```

→ **EA1의 개선은 100% architecture 변경에서 옴.** Loss는 baseline과 동일 (R-A와 같은 4-loss 균등합).

| 비교 | Architecture diff | sum_4 |
|---|---|---|
| R-A (baseline) | (변경 없음) | 0.03376 |
| V1 (gate=0 frozen만 추가) | gated residual only | 0.03083 (-9%) |
| **EA1 (V1 + parallel branch)** | gated residual + parallel branch | **0.02517 (-25%)** |

각 architecture 변경의 단일 기여:
- **Gated residual로 InterFrameTransformer 비활성** → V1 sum_4 0.03083 (R-A 대비 -9%)
- **Parallel frame-score branch 추가** → EA1 sum_4 0.02517 (V1 대비 -18%, R-A 대비 누적 -25%)

특히 `loss_score_traj`가 R-A의 0.01215 → V1 0.01165 → **EA1 0.00651**로 크게 감소 → parallel branch가 TrajScoreHead의 학습을 안정화하는 효과.

### EA1-C (sum_3 최저, stage-1 표현 품질 winner)

```
변경 사항:
  Architecture:  ✓ gated residual (gate=0 frozen) + ✓ parallel frame-score branch  (= EA1과 동일)
  Loss:          ✓ curriculum 추가
  Training:      ✗ (변경 없음)
```

→ **EA1-C는 EA1 architecture에 + loss curriculum.** sum_3 (인코더 grounding)이 EA1 0.01866 → EA1-C 0.01816으로 개선.

---

## "Encoder만 건드린 거? Loss만 건드린 거?"에 대한 답

| 질문 | 답 |
|---|---|
| 본 작업이 loss만 바꿔서 개선했나? | **아니오** — loss 단독 변경(R-C, EA1-3L)도 의미는 있으나 **EA1의 핵심 25% 개선은 architecture (parallel branch) 추가가 주역** |
| 본 작업이 encoder만 바꿔서 개선했나? | **EA1는 그렇다** (loss/training은 baseline과 동일). EA1-C는 architecture + loss curriculum 결합. |
| 가장 효과 큰 단일 변경은? | **Parallel frame-score branch 추가** (architecture). V1 → EA1로 sum_4 18% 개선. |
| Encoder 변경의 net 효과는? | gated residual → 9% 개선, parallel branch → 추가 18% 개선. **두 변경의 곱 = 25% 개선.** |
| Loss curriculum의 단일 효과는? | EA1 → EA1-C에서 sum_3 2.7% 추가 개선. 하지만 sum_4(stage-2 inference path 포함) 측면에서는 손해 (`l_score_traj` 학습 시간 부족). |

### 한 줄 요약

> **본 작업 winner(EA1)는 Encoder architecture 변경 두 개로 만들어졌다.**
> Loss와 training procedure는 baseline 그대로 두고도 sum_4 0.0338 → 0.0252 (25% 개선) 달성.
> Loss curriculum은 stage-1 인코더 표현(EA1-C, sum_3 0.01816)을 끌어내릴 때만 보조적으로 추가 효과.

---

## 실험 결과 종합 (모두 random init, 30 epoch, batch=2, lr=3e-4)

### Round 1' — 2×2 factorial (loss × recipe)

| ID | Loss | Recipe | `traj` | `past` | `fut` | `st_traj` | sum_3 |
|---|---|---|---|---|---|---|---|
| R-A | 4-loss | legacy (gate=1) | 0.01785 | 0.00211 | 0.00165 | 0.01215 | 0.02161 |
| R-B | 4-loss | recipe (gate=0+curr) | 0.01485 | 0.00206 | 0.00137 | 0.01180 | 0.01828 |
| R-C | 3-loss | legacy | 0.01645 | **0.00184** | 0.00156 | (drop) | 0.01985 |
| R-D | 3-loss | recipe | 0.01502 | 0.00198 | 0.00141 | (drop) | 0.01841 |

**핵심 결과 1**: doc의 0.1274 baseline은 본 환경에서 재현 안 됨 (R-A가 sum_3 0.0216까지 잘 수렴).

**핵심 결과 2**: `l_score_traj` 드롭이 score_past 13% 개선 (R-A → R-C, 0.00211 → 0.00184).

**핵심 결과 3**: Recipe의 gated residual은 **gate가 안 열림** — random init에서도 R-B는 gate≈0, R-D는 gate가 음수(-0.006)까지 내려감. 모델이 InterFrameTransformer를 적극적으로 거부.

### V1/V2 — 진단 검증

| ID | 구조 | sum_3 | 결과 의미 |
|---|---|---|---|
| V1 | 4-loss + iframe **disabled** (gate=0 frozen) | **0.01917** | InterFrameTransformer를 *완전히 끄는 게* 켜는 것보다 11% 좋음 |
| V2 | 3-loss + iframe만, no visual | 0.02474 | InterFrameTransformer 단독으론 score_past 학습 어려움 (visual cross-attn이 진짜 일하는 모듈) |

**V1 vs R-A 비교 (sum_3): 0.01917 vs 0.02161** — InterFrameTransformer가 joint training에서 **redundant 그 이상 — 약한 negative**.

이 발견이 다음 단계(parallel branch)의 동기가 됨:
- Main pipeline에서는 InterFrameTransformer를 빼고 V1과 동등한 안정성 확보
- 시간 정보는 **side branch**로 추출 (frame-level scalar)

### EA1/EB2 — Architecture 변형 검증

| ID | 구조 | `traj` | `past` | `fut` | `st_traj` | sum_4 |
|---|---|---|---|---|---|---|
| **EA1** ★ | iframe disabled in main + parallel branch | 0.01506 | 0.00207 | 0.00152 | **0.00651** | **0.02517** |
| EB2 | iframe disabled pre + post-fusion iframe | 0.01624 | 0.00206 | 0.00156 | 0.00774 | 0.02759 |

**핵심 결과 4**: EA1이 본 실험들 중 **sum_4 최저 (0.02517)** — V1 대비 18% 개선.
특히 `loss_score_traj`가 V1의 0.01165에서 0.00651로 **44% 감소** → parallel branch가 TrajScoreHead 학습을 안정화.

### EA1 변형 sweep — 어느 변형이 추가 이득?

| ID | 변형 | sum_3 | sum_4 |
|---|---|---|---|
| EA1 (baseline) | (없음) | 0.01866 | **0.02517** |
| EA1-3L | + drop l_score_traj | **0.01848** | (st_traj 폭주, stage-2 부적합) |
| EA1-G | + gate trainable | 0.01870 | 0.02780 |
| **EA1-C** ★ | + curriculum | **0.01816** | 0.03774 (st_traj 학습 부족) |
| EA1+EB2 | + post-fusion iframe | 0.01887 | 0.02824 |

**핵심 결과 5**: 두 winner 후보:
- **stage-1 인코더 표현이 가장 좋은 건 EA1-C** (sum_3 0.01816, traj/past/fut 모두 최저)
- **stage-2 inference path까지 안정적인 건 EA1** (TrajScoreHead score_traj 0.00651, EA1-C는 0.0196)

EA1-G(gate trainable), EA1+EB2(post-fusion 추가)는 별 효과 없음 — gate는 안 열리고, post-fusion은 parallel branch와 가산적이지 않음.

---

## 결론적 진단

본 모델 구조에서 시간 정보 활용 양상:

| 출처 | 활용 여부 |
|---|---|
| Per-frame trajectory features (gaze_speed, vel, convergence, lead_lag) | 항상 사용됨 (입력 단계) |
| **InterFrameTransformer (cross-frame mixing)** | **Main pipeline에선 거부됨**. Parallel branch (frame-level scalar 추출용)로만 유의미하게 활용 |
| TrajectoryDecoder cross-attention | 사용됨 (l_traj loss를 통해) |
| TemporalVisualTrajFusion (per-frame visual cross-attn) | 진짜 일하는 핵심 모듈 |

> **결론**: InterFrameTransformer를 main pipeline에서 직렬 연결하면 (random init + warm start 모두에서) 모델이 그걸 차단함. Parallel side branch로 frame-level signal만 추출하는 방식이 가장 효과적. EA1 / EA1-C 두 구조가 본 실험의 best.

---

## 후속 작업 (현재 진행 중)

GPU 0/1 분배:
- **GPU 0**: EA1 (best.pth) → stage-2 (Qwen LoRA + token merge) 학습
- **GPU 1**: EA1-C 60 epoch로 연장 (--resume + cosine T_max override) → 그 다음 stage-2 학습

두 stage-2 결과 비교로 **인코더 표현 품질이 stage-2 정확도에 어떻게 반영되는지** 직접 측정.

---

## Acceptance gate

| Gate | 기준 | 상태 |
|---|---|---|
| G1 (학습 시작 후) | val loss 단조 감소, 발산 없음 | 모든 실험 통과 |
| G2 (gate 거동) | gate ∈ (0.05, 0.5) | 실패 — 모델이 gate를 거부. 대신 **parallel branch로 우회** (EA1) |
| G3 (학습 25%) | val loss ≤ 0.025 | 통과 (최저 EA1-C 0.0192) |
| G4 (stage-1 종료) | val loss ≤ 0.020 (= ablation 수준 회복) | 통과 (EA1-C 0.01816) |
| G5 (stage-2 종료) | overall acc ≥ 68.82% + 0.5pt | **진행 중** |

---

## 본 레시피가 발견한 것

1. **단일 인코더 안정 학습은 가능하다.** Random init에서도 doc의 0.1274 baseline보다 4-6× 좋은 stage-1 (0.018-0.025) 도달.
2. **그러나 InterFrameTransformer는 main pipeline에서 작동하지 않는다.** 다섯 번 검증 (E2/R-B/R-D/EA1-G/EB2) 모두 동일 — gate가 0 또는 음수로 가서 모듈 차단.
3. **Parallel side branch가 시간 정보를 활용하는 효과적 방법이다.** EA1이 V1 대비 sum_4 18% 개선, score_traj 44% 감소.
4. **`l_score_traj`는 noisy하지만 stage-2 inference path 학습에 필요하다.** 드롭 시 sum_3는 미세 개선되지만 TrajScoreHead가 학습 안 돼 stage-2 가져가면 손해.
