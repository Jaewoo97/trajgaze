# Gaze-Guided Token Selection on Egocentric VQA — A Negative Result with Structural Explanation

작성: 2026-06-24. 개정: 2026-06-30 (signrouted·frame-breaking 후속 반영, §1·§7). 대상: egtea
(n=1011, 다지선다 VQA, chance ~25–33%). 베이스라인 **M1 = VisionZip-Complement (7%C ∪ 3%G), 63.01%** (2-GPU).

---

## 0. 한 줄 요약

14개의 gaze 메커니즘 중 어느 것도 VisionZip(M1)을 **통계적으로** 이기지 못했다 (최고 confidence
63.30%, McNemar p=0.735 = 타이). 이는 메커니즘 부족이 아니라 **두 가지 구조적 천장** 때문이며,
세 가지 독립적 증거 라인으로 입증된다: (1) 14개 메커니즘의 체계적 falsification, (2) gaze 노이즈
분석, (3) gaze≈중앙 정량 측정. **gaze-guided token selection on egocentric VQA의 천장은 ~1pp이고,
이는 n=1011에서 통계적으로 검출 불가능하다.** 토큰선택 *바깥*(reasoning·inference-hygiene) 후속도
전부 실패(§1 하단) → 천장은 프레임이 아니라 데이터·모델 구조에 있다.

---

## 1. 메커니즘 falsification 표 (13개)

| # | 메커니즘 | 아이디어 | 결과 | 판정 |
|---|---|---|---|---|
| 1 | VZ coverage | 토큰 시공간 분산 | M1 미달 | FAIL |
| 2 | Scanpath tokens | gaze 경로→토큰 추가 | tie | FAIL |
| 3 | Per-token gaze tags | 토큰별 gaze 태그 | −1.39 (과적합) | FAIL |
| 4 | Budget scaling | 10%→13% 증설 | 62.81 | FAIL |
| 5 | Query complement | 질문관련 토큰 pool | 62.12 | FAIL |
| 6 | Budget curve | 5% 압축 시 gap | flat | FAIL |
| 7 | Foveal ROI | gaze crop 재인코딩 | gaze=attn (p=1.000) | FAIL |
| 8 | Anticipatory | 미래 gaze 예측 | 61.82 (p=0.291) | FAIL |
| 9 | tbudget | temporal 토큰 재배분 | 61.5 (p=0.561) | FAIL |
| 10 | GazeText | fixation 텍스트 prefix | ≈random (p=0.275) | FAIL |
| 11 | GazeSampled | gaze-event 프레임 샘플링 | 62.31≈random 62.02 | FAIL |
| 12 | **Confidence** (noise-aware) | fixation-confidence 가중 | **63.30 (p=0.735 vs M1)** | TIE |
| 13 | Task-adaptive (2-way) | object→conf, else→raw (학습) | 63.20 (3ep 최종) | TIE |
| 14 | **signrouted (3-way)** | object→conf/spat·temp→inverse/else→none | 61.73 (p=0.32) | FAIL |

핵심: 선택(1,4,5,6), 해상도(7), 시간(9,11), 미래(8), 텍스트(10), 노이즈처리(12,13,14) — **모든 차원**에서
gaze 메커니즘이 attention-twin 또는 M1과 동률 이하. confidence/routing 라인(12·13·14)은 닫힘:
object→confidence 신호(+1.85)는 실재하나 inverse 역효과·단일-LoRA dilution·검정력(n=1011)으로 leverage
불가. 상세 = `GAZE_CONFIDENCE_METHOD.md §7`, 메모리 `project_gazeconf_selonly`/`project_signrouted_falsified`.

### Frame-breaking 후속 (2026-06-25~30) — gaze-token-selection 틀 밖, 전부 FAIL
gaze가 죽은 레버(천장 #1)임을 받아들이고 토큰선택 *바깥*을 시도했으나 모두 천장 #2 또는 prior 구조에 막힘:

| 방향 | 결과 | 판정 | 메모리 |
|---|---|---|---|
| A. CoT reasoning (zero-shot) | spatial −3.68 / temporal 0.00 | FAIL (천장 #2: 7B capability wall) | `project_cot_probe_falsified` |
| B. option-marg (test-time) | net −2.67 (attribute −15.62) | FAIL | `project_optmarg_falsified` |
| optaug (train-time 보기셔플) | **60.34 vs 62.92, p=0.040** | FAIL (유의) | `project_optaug_falsified` |

**부차 발견(option-position bias)**: 모델의 A/C/D 선호는 *제거할 오류가 아니라 답분포에 calibrated된
load-bearing prior*. test-time(marg −2.67)·train-time(aug −2.58, p=.040) 둘 다 제거 시 B/E는 회수(+6.6/+18.5)되나
더 흔한 A/C/D 손실(−10.6/−3.8/−5.9)이 압도 → 유의하게 손해. 즉 position-debiasing도 막다른 길.

---

## 2. 천장 #1 — gaze ≈ 중앙 (공짜 prior와 중복)

`eval/derisk_gaze_vs_attention.py`로 측정 (no_gaze 프레임, 정규화 거리):

| metric | SuperMemory P7 | egtea | random(chance) |
|---|---|---|---|
| **center_gaze** (gaze↔중앙) | **0.144** | **0.218** | ~0.41 |
| gaze 중앙편향 (<0.15) | 51% | 35% | — |

Egocentric gaze는 **중앙을 거의 떠나지 않는다** (center_gaze가 random의 1/3~1/2). 머리 장착
카메라에서 조작 대상이 시야 중앙에 오기 때문. 따라서 **"gaze prior" ≈ "center prior"**이고, center는
공짜로 얻는 정보라 gaze가 더하는 독립 정보가 거의 없다. 이것이 "gaze≈attention"보다 정확한 천장의 정체.

(주의: SuperMemory가 egtea보다 **더** 중앙편향 → 데이터셋 전환으로 천장을 못 푼다. Aria 보드게임
세션이 중앙 보드 응시라 더 고정됨.)

---

## 3. 천장 #2 — 오답이 추론-bound (perception/selection으로 못 고침)

M1 오답 374개의 task별 분포 (`dumps/m1.jsonl`):

| task | err% | 모델 정확도 |
|---|---|---|
| spatial | 25.9% | 40.5% (≈chance) |
| temporal | 24.1% | 43.8% (≈chance) |
| future_action | 12.6% | 50.0% |
| object_id / non-fixated | ~21% | 63–70% |
| attribute | 1.9% | 92.7% (해결됨) |

**오답의 50%가 spatial+temporal, 62%가 +future** — 거기서 모델은 chance 수준. 이는 *어떤 토큰을
보냐*의 문제가 아니라 **공간·시간 추론 결함**이다. 토큰 선택으로 추론 능력을 만들 수 없다 (budget을
늘려도 13%→62.81, 줄여도 gap flat → 시각 증거의 양/선택이 병목 아님). 반대로 gaze가 도울 수 있는
object grounding은 이미 70–93%로 headroom이 작다.

---

## 4. 노이즈 비대칭 (메커니즘 12·13의 근거이자 한계)

측정: frame간 gaze jitter median 0.135 (이미지의 13.5%), saccade 28%, binocular conv/lead_lag는
9–11%만 유효(monocular라 죽음). → gaze는 fine 수준에서 noisy하고, 신뢰 신호는 gaze_speed 하나.
**confidence** 메커니즘(saccade 억제)이 object-ID에서 inverse(노이즈 주입) 대비 유의하게 이김
(past_non_fixated p=0.021*, present_object_id_easy p=0.049*) → 노이즈 가설 부분 검증. 그러나
spatial/temporal에선 gaze 동역학을 죽여 손해 → object 이득과 상쇄 → net wash (vs M1 p=0.735).

---

## 5. 통계적 검출 한계

천장 #1·#2 때문에 실제 gaze 효과 ≈ ~1pp. n=1011 다지선다에서 정확도 차이 노이즈 바닥 ~±1.5pp.
→ **효과가 있어도 통계적으로 보이지 않는다.** confidence 63.30 (net +5/1006) = 동전 던지기와 구별 불가.

---

## 6. Contribution 주장 (정직한 형태)

> Egocentric VQA에서 gaze는 (a) 중앙 prior와 중복되고 (b) 오답이 추론-bound라, gaze-guided
> token selection의 성능 천장은 ~1pp이며 표준 벤치마크(n≈1k)에서 통계적으로 검출 불가능하다.
> 13개 메커니즘 · 노이즈 분석 · gaze-중앙 정량화로 삼중 입증.

**부차 발견**: noise-aware gaze(fixation-confidence)는 object grounding에서 saccade-noise 주입 대비
유의한 이득을 보이나(p<0.05), 동적 추론 task의 손해와 상쇄된다 → gaze 노이즈 처리는 task-dependent.

---

## 7. 남은 길 (있다면)

- **②-재시도**: center_gaze 스크린을 통과하는(gaze가 돌아다니는) 데이터셋 — 사회적/탐색/다중에이전트.
  단 egocentric 중앙편향은 보편적일 수 있어 기대 낮음. de-risk 툴로 GPU 전에 스크리닝 필수.
- **추론 공략**: spatial/temporal은 token이 아니라 표현/기하 문제. gaze로 두 번(gazetext·scanpath),
  zero-shot CoT로 한 번(§1 A, spatial −3.68) 실패 → 7B capability wall. 남은 변형은 72B-rationale SFT뿐인데
  zero-shot CoT가 *해치므로* prior 나쁨+고비용 → 권장 안 함.
- **inference-hygiene 공략**: option-position bias는 load-bearing prior라 test/train 제거 둘 다 손해(§1 하단).
- **현실적 결론**: egtea 프레임에서 천장 확인됨 (토큰선택·라우팅·노이즈처리·reasoning·debiasing 전부).
  ③(negative result)가 가장 정직. 진짜 돌파는 데이터셋(②) 또는 더 강한 backbone에서만 가능.

관련 메모리: project_experiments_timeline, project_supermemory_derisk_failed,
project_gaze_confidence_running, project_gazesampled_running, reference_eval_is_multichoice,
project_gazeconf_selonly, project_signrouted_falsified, project_cot_probe_falsified,
project_optmarg_falsified, project_optaug_falsified.
