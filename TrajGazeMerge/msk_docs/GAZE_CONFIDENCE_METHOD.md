# Noise-Aware Gaze: Fixation-Confidence-Weighted Complement (GazeConf)

작성: 2026-06-24. 개정: 2026-06-30 (§7 후속 — selection-only·task_adaptive·signrouted로 계열 종결).
14개 gaze 메커니즘 중 **최고 성능** (best 63.30%, M1 63.01과 McNemar 타이). **최종 판정은 §7**:
object→confidence 신호(+1.85)는 실재하나 leverage 불가 → 라인 종결.
같이 볼 것: `GAZE_NEGATIVE_RESULT_SYNTHESIS.md` (전체 맥락), 메모리 `project_gaze_confidence_running`,
`project_gazeconf_selonly`, `project_signrouted_falsified`.

---

## 1. 동기 — gaze는 noisy하다

이전 메커니즘들은 모두 raw gaze를 **무조건** 주입했다 → tie/loss. 핵심 관찰: gaze는 attention보다
noisy하다 (egtea 측정값):

| 측정 | 값 | 함의 |
|---|---|---|
| frame간 gaze jitter (정규화) | median **0.135** (이미지의 13.5%) | gaze가 fine 수준에서 noisy |
| saccade 비율 (speed>0.25) | **28%** | 프레임의 1/4이 이동중=노이즈 |
| convergence / lead_lag 유효 | **9% / 11%** | binocular 신뢰신호 죽음 (monocular) |

→ 쓸 수 있는 confidence 신호는 **gaze_speed (fixation vs saccade) 하나뿐.**

**설계 원칙**: gaze를 *stable fixation일 때만* 주입, saccade는 억제.

---

## 2. 방법

M1(VisionZip-Complement, 7%C ∪ 3%G)은 **그대로** 두고, 3%G complement의 점수만 재가중한다.
M1의 gaze complement = TAS encoder의 per-patch salience `S(t, patch)`의 top-3% (non-content 중).
여기에 per-frame fixation-confidence `c(t)`를 곱한 뒤 top-k:

```
denoised(t, patch) = S(t, patch) · c(t)
complement = top-3%( denoised , non-content tokens )
```

**c(t) (sustained fixation confidence), 범위 [c_min, 1]:**
1. soft fixation indicator: `fix(t) = sigmoid((SACCADE_SPEED − speed(t)) / TEMP)` — 느릴수록 1
2. sustained: 윈도우 평활 `MA(fix, WINDOW)` — *지속된* 저속만 보상 (순간 dip 억제)
3. invalid gaze frame → 0
4. clip 내 [c_min, 1]로 정규화 (rescale 아닌 relative reweight)

`c(t)`는 raw encoder 출력 `(T_traj, 196)`의 **시간축**에 곱해진 뒤 VisionZip `(N,)` 레이아웃으로 매핑.

**하이퍼파라미터** (`models/gaze_confidence.py`):
`SACCADE_SPEED=0.25, WINDOW=5 (~0.5s@10fps), TEMP=0.10, C_MIN=0.10`.

---

## 3. Arms (통제 설계)

| arm | c(t) | 역할 |
|---|---|---|
| **confidence** | fixation일수록 ↑ | 방법 |
| **inverse** | 1−c (saccade 부스트=노이즈 주입) | **KILL-TEST** |
| **random** | 랜덤 per-frame | placebo |
| **none** | ≡1 (raw M1) | in-protocol baseline |

`confidence > inverse` (유의) ⇒ saccade 토큰이 M1을 깎고 있었고 우리가 고침.
`confidence ≈ none` ⇒ learned encoder가 이미 암묵적 denoise.

---

## 4. 결과 (단일-GPU, grad-accum 8, egtea n=1011)

| arm | ep1 | ep2 | ep3 | best |
|---|---|---|---|---|
| **confidence** | 62.41 | 62.81 | **63.30** | **63.30** (dump 63.60) |
| inverse | 61.82 | 60.93 | — | 61.82 |
| none | 61.62 | 61.92 | 61.52 | 61.92 |
| random | 61.52 | 62.02 | 60.24 | 62.02 |

confidence가 단조 상승하며 단일-GPU arm 중 유일하게 2-GPU M1(63.01)을 숫자상 도달/초과.

**McNemar — confidence vs M1 (2-GPU m1.jsonl):**
```
OVERALL  M1=62.9  conf=63.4  Δ=+0.5  net=+5  b=67 c=72  p=0.735  → TIE
per-task: past_non_fixated_object_id +8.8 (p=0.070), present_object_id_easy +7.9 (p=0.057)
          spatial −4.3, temporal −2.5, past_scene_recall −10.8 (상쇄)
```

**McNemar — confidence vs inverse (kill-test):**
```
OVERALL  inverse=61.5  conf=63.4  Δ=+1.9  net=+19  b=69 c=88  p=0.151
per-task: past_non_fixated_object_id +11.8 (p=0.021*), present_object_id_easy +8.9 (p=0.049*)
```

---

## 5. 정직한 판정

- **vs M1: 통계적 타이** (p=0.735). 숫자상 63.30 > 63.01이나 net +5/1006 = 노이즈. "격파" 아님.
- **노이즈 가설 부분 검증**: object-grounding task에서 confidence가 saccade-noise(inverse) 대비
  **유의하게** 우세 (p=0.021, p=0.049, 두 비교에서 일관 재현). saccade 억제 → 안정 fixation → 물체 식별 ↑.
- **상쇄 구조**: object-ID 이득이 spatial/temporal/scene-recall 손해와 상쇄 → net wash. gaze 동역학을
  죽이면 동적 추론이 손해 보기 때문. → 노이즈 처리는 **task-dependent** (Direction A의 동기).
- **천장**: `GAZE_NEGATIVE_RESULT_SYNTHESIS.md` 참조 — gaze≈중앙 + 오답 추론-bound로 ~1pp 천장.

---

## 6. 재현

```bash
GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=<N> conda run -n trajgaze \
  torchrun --nproc_per_node=1 --master_port=<PORT> \
  -m TrajGazeMerge.training.train_visionzip_gazeconf_lora \
  --arm confidence --output-dir <...>/gazeconf_confidence \
  --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop
# eval dump + McNemar
python -m TrajGazeMerge.eval.eval_dump_gazeconf --ckpt <...>/best.pth --arm confidence --dump <...>.jsonl --gpu <N>
python -m TrajGazeMerge.eval.mcnemar --a dumps/m1.jsonl --label-a M1 --b <...>.jsonl --label-b conf
```

**코드**: `models/gaze_confidence.py` (c(t) + arms + `resolve_arm` for task_adaptive/signrouted),
`training/train_visionzip_gazeconf_lora.py` (`select_complementary_conf`),
`eval/eval_dump_gazeconf.py`. dumps: `gazeconf_{confidence,inverse}.jsonl`.

---

## 7. 후속 (2026-06-25~27) — confidence/routing 계열 종결

§5의 "task-dependent 노이즈 처리(Direction A)" 가설을 끝까지 밀었다. **결론: object→confidence
신호(+1.85)는 실재하나 leverage 불가. 계열 전체가 M1과 타이거나 그 이하.** 모든 후속이 효율
논제(단일 LoRA·1 pass·10% budget)를 지킨다.

### 7.1 selection-only — "이득이 선택규칙인가 LoRA적응인가?" (`project_gazeconf_selonly`)
2-expert 후처리 혼합(M1+confidence LoRA를 task로 라우팅)은 오라클 +1.79(p=0.005)지만 **forward 2회**라
효율 논제 위반. 그래서 **M1 단일 LoRA에 confidence 선택규칙만 per-task 스위칭**(`--arm task_adaptive`,
1 pass)을 평가:
- object task: +1.23 (net +4, p=0.29, **noise 내**) — 오라클 object 이득의 **22%만** 회수.
- 즉 ~78%는 **LoRA 가중치 적응**에 묶여 있어 2-expert(2× 비용) 없이는 못 꺼냄.
- 검증 부산물: dynamic task에서 동일 선택인데도 ~14/682 flip = GPU 비결정성 노이즈 ±1.5pp(프로젝트 SE).
- task_adaptive 단일 LoRA를 *학습*해도 best **63.20**(ep1) = frozen 스위칭과 동일 → 학습이 dilution.

### 7.2 signrouted — 3-way sign-routing (`project_signrouted_falsified`)
confidence net-wash가 **sign-flip**(object는 fixation 선호, spatial/temporal은 saccade 선호 — dual-pool의
test-time 힌트)이라는 가정 하에, 단일 LoRA에 **object→confidence / spatial·temporal→inverse / else→none**
3-way 라우팅을 baking해 학습(`--arm signrouted`, 3ep).

| 그룹(라우팅) | signrouted | M1 | Δ |
|---|---|---|---|
| object → confidence | 76.23 | 74.38 | **+1.85** ✅ |
| spatial/temporal → inverse | 39.94 | 42.11 | **−2.17** ❌ |
| other → none (raw) | 68.25 | 71.31 | **−3.06** (dilution) |
| **overall** | **61.73** | 62.92 | **−1.19** (p=0.32) |

best(ep1)=61.73 < M1 63.01 < task_adaptive 63.20. in-training 곡선 62.41→62.12→60.24(단조하락).
**진단**: object→confidence 절반은 작동(+1.85)하나, dual-pool의 "spatial/temporal→inverse" 힌트는
*test-time 대조*에서 나온 **노이즈라 학습된 단일 LoRA엔 안 살아남고**(−2.17 역효과), 혼합 라우팅이
안 건드린 task까지 dilution(−3.06)으로 깎는다. 코드: `resolve_arm(arm='signrouted')`,
`eval/signrouted_analysis.py`. dump: `signrouted{,_ep1}.jsonl`.

### 7.3 종합 판정
| 변형 | best | vs M1 | 한 줄 |
|---|---|---|---|
| global confidence | 63.30 | tie (p=0.735) | 계열의 시작이자 천장 |
| dual-pool (fix∪sac 분할) | 62.5 | tie/아래 | 상쇄 해소 실패 |
| selection-only (1 pass) | 63.20 | tie | 선택규칙만으론 22%만 |
| task_adaptive 2-way (학습) | 63.20 | tie | dilution |
| **signrouted 3-way (학습)** | **61.73** | **−1.19** | inverse 역효과+dilution |

object grounding에서 confidence가 saccade-noise보다 나은 건 **재현되는 실재 신호**(+1.85)지만,
(a) inverse 라우팅은 역효과, (b) 단일 LoRA는 dilution, (c) n=1011 검정력으론 object 이득도 유의 미달
— 셋 중 어느 것도 우회 못 함. **confidence/routing 라인은 여기서 닫는다.** 더 많은 라우팅 변형 제안 금지.
