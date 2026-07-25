# Gaze-Exclusive Performance Strategy — 세션 발견 정리

> 이 문서는 `CONTRIBUTION_AND_PERFORMANCE_DIRECTIONS.md`의 후속·교정판이다.
> 그 문서의 방향(대부분 token-selection 변형)이 실험으로 막힌 뒤, "그럼 무엇을
> 해야 하는가"를 다시 정의한다. **핵심 결론 한 줄:**
>
> **성능 향상에는 두 종류가 있다 — (A) 모든 방법(=baseline 포함)을 같이 올리는
> orthogonal한 "위생", (B) gaze가 load-bearing이라 baseline이 구조적으로 못
> 따라오는 "기여". 우리가 찾아야 하는 건 (B)다.**

작성 시점 best: **M1 (VisionZip-Complement, 7% content ∪ 3% gaze top-k) = 63.01%**
(egtea 2-way, n=1011).

---

## 1. 지금까지 막힌 것 — "선택은 10%에서 이미 최적"

토큰 **선택**을 바꾸는 모든 시도가 M1 63.01을 넘지 못했다:

| 시도 | 결과 | 분류 |
|---|---|---|
| VZ+traj (attn×traj) | 62.71 | 선택 재가중 |
| coverage de-clustering | < top-k | 선택 분산화 (falsified) |
| soft fusion (norm(attn)+λ·norm(traj)) | null | 선택 융합 (falsified) |
| scanpath side-channel | 63.01 tie (gate→0) | additive 채널 (falsified) |
| gaze-tag per-token | 61.62 (overfit) | additive 채널 (falsified) |
| **query-conditioned complement (7C+1Q+2G)** | **62.12 (−0.89)** | 선택 재배치 (이번 세션 falsified) |
| decoupled budget scaling (M1+13%) | 62.81 < 63.01 | 예산 확대 → "10%가 sweet spot" |

**교훈: 병목은 "어떤 토큰을 고르냐"가 아니다.** 10% 예산에서 M1의 7+3 top-k는
거의 최적이고, 예산 안 재배치/추가/재가중은 전부 실패한다. 자세한 falsification은
메모리 `project_query_complement_falsified`, `project_additive_gaze_channels_falsified`,
`project_vzcomplement_coverage_falsified`, `project_decoupled_gaze_budget_scaling` 참조.

### query_gaze에서 배운 추가 함정 (측정)
- ep1에서 spatial/temporal이 올라(+1.2/+1.25) "query가 추론을 돕는다"처럼 보였으나
  **ep2에서 −3.1/−5.0으로 반전.** → 단일 epoch의 per-task 패턴은 **epoch jitter**이지
  효과가 아니다. M1 자신의 epoch 진폭이 task당 2~16%p.
- 전체 평균도 신뢰 못 함: n=1011·2-way에서 SE≈1.5%p, M1−VZ gap(+0.50)은 노이즈 안,
  epoch jitter(60.83→63.01)가 method gap보다 큼.

---

## 2. 진짜 병목 — per-task가 가리키는 곳

> **[교정 §0 참조]** eval은 2지선다가 아니라 **다지선다(3~5옵션, chance ~20–33%)**.
> 아래 "약한 task"는 chance 위의 *상대적으로 약한* task이지 chance 이하가 아니다.

| task | M1 acc | 해석 |
|---|---|---|
| spatial | ~42% | 가장 약함 (chance ~25% 위지만 헤드룸 큼) |
| temporal | ~42% | 〃 시간추론 |
| present_future_action | ~50% | 약함, 미래예측 (① 타깃) |
| present_object_identification_hard | ~67% | 약함, 객체 식별 (② 타깃) |
| causal | ~84% | 인식·인과는 잘 함 |
| present_object_attribute | ~92% | 〃 |

→ 병목은 **(a) 측정을 막는 분산, (b) 고른 토큰으로 공간/시간/의도를 "추론"하는 능력**
이지, 토큰 선택이 아니다.

---

## 3. 핵심 원리 — Orthogonal(위생) vs Gaze-Exclusive(기여)

우리 기여는 절대 숫자가 아니라 **M1 − VZ gap**("gaze complement가 content-only를
이긴다")이다. 어떤 개선이 **baseline도 같이 올리면 gap은 그대로 → 기여 아님.**

| 레버 | VZ도 오르나 | gap 효과 | 분류 |
|---|---|---|---|
| 옵션 순서 대칭화 | 예(동일) | 없음 | **orthogonal(위생)** |
| seed/frame 앙상블 | 예 | 없음 | orthogonal |
| M1+VZ 앙상블 | — | baseline을 답에 섞음 | **기여 희석(금지)** |
| visual-side LoRA / distillation | 예 | 대체로 없음 | orthogonal |
| LoRA 하이퍼·정규화(α) | 예 | 없음 | orthogonal |
| CoT 추론(β) | 예 | 없음 | orthogonal |
| gaze-overlay 렌더링(δ) | 예(VZ도 같은 프레임) | 대체로 없음 | orthogonal |
| **foveated ROI** | **아니오 (VZ엔 gaze 없음)** | **gap 확대** | **기여(✓)** |
| **budget-curve 진단** | 둘 다 측정 | **gap을 측정** | 기여 진단(✓) |

### 기여를 구조적으로 보호하는 control 원리
VZ만 control로 쓰면 "attention 버전도 되잖아"로 반박당한다. 진짜 보호막:

> **기여 = (gaze로 구동한 메커니즘) − (똑같은 메커니즘의 attention 쌍둥이).**
> control은 VZ가 아니라 **그 메커니즘의 attention 버전.** gaze판이 자기 attention-쌍둥이를
> 이기면 baseline은 그 gap을 구조적으로 못 닫는다.

예: ROI는 "gaze-중심 crop vs **attention-중심 crop**"으로 비교해야 gaze 기여가 증명된다.

### Orthogonal한 것의 올바른 역할 — 버리지 말 것
기여는 아니지만 **위생**으로 가치 있음:
1. **모든 방법에 균일 적용** → gap은 불변이지만 분산↓ → gap의 유의성을 측정 가능하게.
   (예: 대칭화는 M1·VZ 둘 다에 적용 후 "대칭화 후 gap"만 본다.)
2. 표 전체를 경쟁력 있게(절대 숫자). 단 **기여로 주장 금지.**

세션 측정: 옵션 대칭화 100-item 예비 = base 70 → sym 76 (위치편향 −1pp뿐 →
per-item 분산 축소 효과). **위생용으로만 해석**(M1·VZ 동시 적용 필요).

---

## 4. Gaze-Exclusive 기여 아이디어 카탈로그

"content-only가 얼마나 못 하나(exclusivity)" 강한 순.

### ① Gaze 예측성 — 미래/의도 task  *(exclusivity 최강)*
gaze는 행동을 ~0.5–1s 선행. `present_future_action_prediction`(~50%, chance) 등에서
**정답이 아직 프레임에 없음**(미래). content는 구조적으로 답을 못 가짐. gaze의 진행
방향/anticipated fixation만이 예측 정보.
- **메커니즘:** gaze velocity·anticipated-fixation을 future/intent 추론에 주입.
- **control:** 아예 불가능 — attention은 미래를 못 봄. reviewer 반박 불가.
- **타깃:** 약한 task(future ~50%) 직격 → 헤드룸 큼.

### ② Gaze 지시 해소 — 참조 모호성 task
"이 물체는?", "지금 보는 것 기준 다음은?"의 "this/지금"은 **gaze로만 해소.** 다객체
장면에서 content는 referent를 모름 → chance. gaze는 가리킴.
- **메커니즘:** fixation을 referent로 명시(③과 결합).
- **control:** attention-peak를 referent로. 혼잡 장면에서 gaze가 이김 = gap.

### ③ Gaze-grounded 언어측 추론  *(선택 영역 탈출)*
지금까지 gaze로 *토큰만* 골랐고(다 막힘), 대신 gaze를 **언어측 공간 참조**로 변환:
fixation → "사람이 [좌상단/추적된 컵]을 봄"을 구조화 입력으로 LLM에 줌.
- scanpath 실패(raw x,y를 LLM이 해석)와 **다름: semantic grounding**(좌표 아니라 영역/객체명).
- **control:** attention-peak 영역 grounding.
- **타깃:** spatial/temporal(chance) 직격.

### ④ Gaze-guided 해상도 (ROI) — *진행 중, attention-control 필수*
gaze-중심 crop을 고해상도로 ViT 재인코딩 → 기존 토큰에 없던 디테일 주입(=새 픽셀,
선택 아님). 작은/비-salient 타깃에서 attention이 못 잡는 것을 gaze가 잡음.
- **반드시 control = attention-ROI**(attention-peak 중심 crop). gaze-ROI > attn-ROI여야 기여.
- 코드: `models/foveal_roi.py` (완성·CPU 검증). V1=added(10%+K), V2=replace(7%+3%).
- 모든 노브 sweepable: `FovealROIConfig` + `SWEEP_GRID`(crop_frac/foveal_k/roi_max_pixels/n_fix_frames).

### ⑤ gaze≠attention 영역으로 기존 gap 확대 *(새 메커니즘 없이)*
gap이 얇은 건 10%에서 attention≈gaze이기 때문. 둘이 갈라지는 조건으로 밀면 모든 gaze
기여가 증폭:
- **저예산(5% budget-curve, 진행 중)** ← 이 가설의 첫 테스트.
- 혼잡/다객체 장면, 작은 타깃 subset.
- 메커니즘이 아니라 *조건*이라 ①~④ 어디에도 곱해짐.

### (참고) 기여 데모용 — gaze-necessary eval subset
gaze 없으면 못 푸는 item(다객체 중 "보는 것" 지목 등)을 추려 보고하면, 그 subset에서
VZ~chance, gaze법 해결 → gap이 크고 질적으로 명확. (메커니즘 아니라 측정/증명)

---

## 5. 우선순위 로드맵

```
가장 깨끗한 기여:  ① 미래/의도 anticipation (content가 답을 못 가짐)
                  ② 참조 해소 (gaze만 referent 지정)
강한 기여(진행):   ④ ROI  ← attention-ROI control 필수
선택 탈출 신규:    ③ gaze-grounded 언어 추론
증폭 조건:         ⑤ 저예산/혼잡 (budget-curve가 이미 일부 검증)
위생(균일 적용):   대칭화·정규화·앙상블 — gap 측정 정밀화 + 절대표 보강(기여 아님)
폐기:             M1+VZ 앙상블(기여 희석), α 단독(LoRA 하이퍼=VZ도 오름)
```

다음 깨끗한 실험 한 줄:
- **①:** 데이터에서 future/intent task를 식별 → gaze-anticipation이 *거기서만* 이기는지.
- **④:** gaze-ROI vs attention-ROI vs no-ROI(M1) 3자 비교.

---

## 6. 진행 중 / 자산

| 항목 | 상태 |
|---|---|
| **budget-curve Wave 1** (M1@5%, VZ@5%) | 학습 중, epoch-1 eval 임박 → "5%에서 gap 확대?"(⑤ 검증) |
| `models/foveal_roi.py` | 완성·CPU 검증(fixation/crop/주입/sweep). trainer 연결만 남음 |
| `eval/eval_tta.py` | 옵션 대칭화 + 위치편향 측정(위생 도구) |
| `eval/eval_dump.py` + `eval/mcnemar.py` | per-item paired McNemar (gap 유의성 검정) |
| `eval/pertask_compare.py` | 로그→per-task 표 + epoch-jitter 밴드 |
| `--query-mode {cosine,random,shuffle}` | query 컨트롤(현재 저우선) |

**측정 규율(필수):** 어떤 개선도 (1) M1·VZ 둘 다에 적용 후 **gap**으로 판단,
(2) per-task로 *어디서* 이기는지, (3) epoch-jitter/노이즈(±1.5%p) 대비 McNemar로 유의성.
절대 숫자 단독은 신뢰하지 않는다.

---

## 7. 세션 진행 로그 & 결정 (live, 최신이 위)

### [temporal-gaze (tbudget) 발사 — novelty 베팅] 2026-06-20
사용자: ④/⑤ 반증·① 진행 중인 상태에서 free GPU 3개에 **novelty 방법**(하이퍼파라미터 말고)을 더 돌려라.
spatial-token 게임은 다 죽음(selection/coverage/③④⑤) → **다른 축 = temporal**. 기존 `models/temporal_budget.py`
활용: 10% 토큰을 **프레임별로 몇 개씩** 줄지를 gaze/hand 시간 상호작용으로 재배분(within-frame은 VZ 그대로).
- **control 내장:** `w_t=(1−traj_weight)·attn_share + traj_weight·gaze_temporal`. **traj_weight=0=attention-twin,
  =1=gaze-temporal** → 기여 검정. 트레이너 `--no-hdepic` 2-way 패치 추가.
- **de-risk(80 items):** gaze(w=1) vs attn(w=0) 프레임 예산 **L1diff=0.378, cosine=0.890** — 비퇴화(≈19% 예산이
  다른 프레임)이나 상관 높아 효과는 modest 예상. 시간축은 미검증이라 발사 가치 있음 = GREEN.
- **발사:** 2-way 단일-GPU grad-accum 8, epochs 2. w=1.0(GPU0) ∥ w=0.0(GPU2) ∥ w=0.5(GPU3).
  완료 후 McNemar(w0 vs w1) temporal/spatial subset. (launcher 스크립트 self-pkill 버그로 직접 setsid 발사.)
- 다음 novelty 후보(빌드 예정): gaze **궤적 visual overlay**(no_gaze 프레임에 PIL로 scanpath 렌더 = LLM이 읽는 새 신호).

### [① anticipatory ROI 빌드 + de-risk + 학습 발사] 2026-06-20 (④ 반증 후 피벗)
사용자 결정: ④ 반증 후 **① future/intent**로 피벗, none은 GPU0 유지, 새 작업은 다른 GPU.
- **메커니즘:** ① anticipatory ROI = 현재 프레임을 **미래 fixation `gaze[t+Δ]`** 위치에 crop. gaze는
  행동을 선행하므로 다음-행동 타깃을 미리 봄. **control = attn-twin**(frame t에서 gaze[t+Δ] 모름 → 구조적 불가).
- **de-risk(메모리 규율)가 설계 1차안을 반증:** velocity 외삽(`gaze+v·horizon`)은 **퇴화** — egtea gaze
  speed median 0.557/step(거친 샘플링, 큰 점프) → horizon 6에서도 **98% off-frame clamp**. **→ 실제 미래
  gaze 위치 `gaze[t+Δ]`로 전환**(Variant B): Δ=12에서 disp median 0.175, **50% distinct(>crop_half)**,
  항상 on-frame. 비퇴화 확인 후 빌드. `detect_anticipatory_frames`(shift 큰 순간 NMS 선택).
- CPU+GPU 스모크 통과(future_action 아이템 K=32 주입). **GPU1에 단일-GPU grad-accum 8 학습 발사**(04:08,
  ~13h, 다른 arm과 동일 프로토콜). 완료 시 자동 dump+McNemar(`run_antic_mcnemar.sh`): **attn(control) vs
  antic** + gaze vs antic, future_action subset 주목 → `dumps/antic_mcnemar_result.txt`.
- 부수: **none(매칭 no-ROI 단일-GPU) ep1=62.02**(foveal_hit=0) — foveal 없이도 같은 62–63 band, foveal-added
  net 무익 재확인.

### [④ ROI 빌드 완료 + GPU 검증 + 런처 무장] 2026-06-18
- **트레이너 빌드:** `training/train_visionzip_foveal_lora.py` (fork of complement trainer).
  selection은 M1 그대로(7%C∪3%G topk), 그 뒤 foveal K=32 토큰 주입. `--roi-arm {gaze,attn,random,none}`.
- **attention-twin control 추가** (`foveal_roi.py`): `detect_attn_frames` = per-frame attention
  mass로 프레임 선택 + 프레임 내 argmax 패치를 crop 중심으로. **hand=None**(content-only 쌍둥이는
  gaze 유래 hand 안 씀). `detect_random_frames` = placebo. crop 로직은 `crops_from_fixations`로 공유.
  `none` arm = `build_inputs_with_foveal(foveal=None)` ≡ `build_merged_inputs` ≡ **M1 정확히 일치**(검증).
- **버그 발견·수정 (GPU 스모크가 잡음):** `encode_foveal_tokens`가 `processor(text=None, images=...)`로
  호출 → `TypeError(NoneType not iterable)` (Qwen2_5_VLProcessor는 text 필수) → fallback도 실패 →
  **전 arm K=0으로 조용히 no-op**(M1으로 fallback). `processor.image_processor(images=…, max_pixels=…)`로
  교체. **교훈: encode/주입 경로는 반드시 live ViT 단일아이템 스모크로 검증** — CPU 테스트·argparse는 통과했음.
- **GPU 검증(단일 아이템):** gaze/attn/random 각 K=32(d=3584=LLM hidden) 주입, seqlen +K(1508→1540),
  none==M1, ViT 이미지-crop 재인코딩 동작.
- **결정(사용자):** arms = **gaze + attn + random**(placebo 포함). crop **crop_frac 0.35 / margin 0.08**
  (넉넉히 — gaze center 쏠림 대비). no-ROI 기준선 = 기존 M1 63.01(`m1.jsonl`) 재사용(4번째 런 불필요).
- **런처 v2 발사**(`msk_docs/launch_foveal.sh`, setsid-detached): VZ curve 종료로 GPU 2,3 비자
  **gaze(GPU2) ∥ attn(GPU3) 즉시 단일-GPU 학습** 시작(13:47), random은 M1 curve가 GPU0,1 해제 시
  GPU0에. 프로토콜 = **단일-GPU + grad-accum 8**(eff-batch 8 + optimizer-step 수 모두 M1의 2-GPU
  DDP×accum4와 등가; gaze/attn/random 셋 다 동일), 3 epoch, early-stop, GAZE_OVERLAY=1.
  **실전 검증: fov_hit=20/20**(매 아이템 K=32 주입 확인). 하트비트 `checkpoints/foveal_launch.log`.
  ⑤ **확정 null**: M1@5% best=61.13(ep2=60.63 early-stop), VZ@5% best=60.53(ep2=59.55) → 5% gap=+0.60 ≈
  10% gap(+0.50), 압축이 gap 안 벌림. 메모리 `project_budget_curve_falsified`. (3 arm 모두 13:47~13:56 발사: gaze=GPU2, attn=GPU3, random=GPU0)
- **다음:** curve ep2 결과로 ⑤ 최종 판정 → ④ 런 완료 후 object-ID/attribute subset에서
  gaze-ROI **vs** attn-ROI McNemar (gaze>attn = 해상도 배분에서 gaze가 load-bearing = 기여).

### [교정 §0] "2-way" = 2 데이터셋 (다지선다 eval)
데이터 감사 결과 **전 item이 옵션 >2개** → "2-way/3-way"는 답 선택지 수가 아니라
**소스 수**(2-way = StreamGaze+EgoGazeVQA, HD-EPIC 제외). 따라서 **chance ≈ 20–33%**,
앞 절들의 "spatial/temporal이 chance(50%) 이하" 표현은 교정됨(그것들은 chance 위의
가장 약한 task). 노이즈 floor(±1.5%p, n=1011) 논의는 chance와 무관하게 유효.

### [결정] 첫 기여 타깃 = **② referential** (메커니즘은 ③ gaze-grounded language)
데이터 감사(test n=1011, `eval/audit_tasks.py`)가 task↔레버를 거의 1:1로 매핑:

| 레버 | task | n | gaze_val |
|---|---|---|---|
| ① future/intent | present_future_action_prediction (전부 future kw) | 94 | 0.96 |
| **② referential** | object_attribute(96)+id_easy(101)+id_hard(64) = "보는 객체 식별" | **261** | 0.95 |
| gaze-specific | past_gaze_seq(64), non_fixated(68) | 132 | 0.96 |
| 약한 추론 | spatial(163), temporal(160) | 323 | 1.00 |

- **②를 주력으로**: 측정성(261≫94) + 스토리 명확(다객체 중 '보는 것'=gaze만 referent 지정).
  `present_object_identification_hard`(~67%)가 헤드룸. ①(94)은 exclusivity 최강이나
  소표본이라 **보조 증거**로.
- gaze validity 0.95–1.0 전역 → **reliability 레버 불필요**.
- gaze-necessary 후보 **505개**(eval 절반) → `/tmp/audit_gaze_necessary.json` (데모 자산).

### [결과 — 최종] ⑤ budget-curve: **gap이 안 벌어짐 = ⑤ NULL** ✅확정
| budget | M1 best | VZ best | gap |
|---|---|---|---|
| 10% | 63.01 | 62.51 | +0.50 |
| 5%  | 61.13 | 60.53 | **+0.60** |

**둘 다 epoch 1이 best → ep2에서 하락해 early-stop** (M1 61.13→60.63, VZ 60.53→59.55).
5% gap(+0.60) ≈ 10% gap(+0.50) → 압축이 gap을 **안 벌림 = ⑤ 미지지(null)**. selection-gap은
얇고 budget-invariant(노이즈 floor 안). **함의: 얇은 selection-gap을 *증폭*(⑤)하지 말고,
gaze-necessary gap을 *생성*(①②④)하라.** ④ 결정 강화. 메모리: `project_budget_curve_falsified`.

### [de-risk 결과 → ③ 반증, ④로 redirect] ⭐
M1 덤프(63.01 재현) + `eval/derisk_object_id.py` 결과: object-ID 오답 58/256.
- 수치 휴리스틱: 오답 60%가 clear gaze fixation → "STRONG".
- **그러나 정성 검토가 ③(region 언어 grounding)을 반증**:
  1. 질문이 이미 referent 명시("the object **the user is gazing at**") → region 문장 **중복**.
  2. 오답이 **perception/fine-ID**(색/재질/모양, pasta vs spoon)이지 disambiguation 아님.
  3. gaze가 **center 쏠림**(clear 오답 전부 center/lower-center) → 3×3 region **변별력 0**.
  4. hand-relation **0회 발화**(gazing 장면에 손 없음).
- **결론: 헤드룸은 "위치"가 아니라 "해상도" → ④ ROI로 redirect.** clear fixation이라
  crop 위치 신뢰 가능(④ 청신호). ③ 언어 grounding은 이 subset에서 폐기(차후 referent가
  안 주어지는 spatial/temporal엔 재검토 여지).
- 교훈: **수치 verdict 맹신 금지 — 정성 검토가 방향을 뒤집음.** de-risk가 4×15h 절약.

### [결과 — 최종] ④ foveal ROI **FALSIFIED** (gaze = attn-twin, McNemar p=1.000) 🔴 2026-06-19
3 arm 동일 프로토콜 학습(단일-GPU grad-accum 8, 3ep early-stop, crop 0.35/0.08). best-epoch eval:

| arm | train best | re-eval 덤프 |
|---|---|---|
| gaze (방법) | 62.91(ep2) | **63.11** |
| attn (attention-twin) | 62.12(ep2) | **63.11** |
| random (placebo) | 61.42(ep1) | 61.62 |
| M1 (no-ROI 기존) | 63.01 | — |

- **paired McNemar attn vs gaze: OVERALL p=1.000, Δ=0.0, b=43 c=43** — 완전 무승부. per-task도 전부
  n.s.(object_attribute Δ+3.3 p=0.25[discordant 3개]; id_hard/id_easy는 오히려 attn 우세). object-ID/
  attribute subset에서 gaze-ROI가 attn-ROI를 **못 이김.** random vs gaze Δ+1.5 p=0.214(n.s.) — *targeted*
  (gaze든 attn이든) > random 약한 힌트뿐, gaze 특별하지 않음.
- **학습 때 +0.79(62.91 vs 62.12)는 eval 노이즈**: 같은 ckpt 재평가→gaze=attn=63.11. 단일 델타 신뢰 금지,
  McNemar가 p=1.000으로 판정. foveal-added≈M1(63.1≈63.01) → "added" 레짐 자체도 net-neutral.
- **판정: ④는 기여 아님**(attention이 gaze와 같은 crop 위치를 찾음 → content-only가 구조적으로 닫음).
  메모리 `project_foveal_roi_falsified`. 남은 깨끗한 exclusivity 레버 = **① future/intent**(attn이 미래를
  구조적으로 못 봄; n=94 소표본). 도구: `eval/eval_dump_foveal.py`, 덤프 `checkpoints/dumps/foveal_*.jsonl`.
- (none 매칭 no-ROI 기준선은 GPU0에서 학습 중 — foveal-added vs no-ROI 확인용, 부차적.)

### [다음 — ④ ROI 로드맵 (de-risk가 ③→④로 redirect)]
타깃: object-ID/attribute subset(헤드룸=해상도, clear fixation으로 crop 위치 신뢰).
1. **build ④**: `models/foveal_roi.py`를 trainer에 연결. gaze 중심 고해상도 crop을
   ViT 재인코딩 → K 토큰 주입(V1 added). 선택은 M1 그대로.
   - **control = attention-ROI**(attention 최대점 중심 crop, content-only 쌍둥이) + no-ROI(M1).
   - gaze-ROI > attn-ROI → gaze가 해상도 배분에서 load-bearing = 기여.
2. **측정**: object-ID subset에서 M1 vs gaze-ROI vs attn-ROI **McNemar**.
   per-task로 object_id_hard/attribute가 오르는지(해상도 헤드룸 검증).
3. sweet spot: `SWEEP_GRID`(crop_frac/foveal_k/roi_max_pixels/n_fix_frames).
   단 gaze가 center 쏠림 → crop이 너무 작으면 referent 놓침, margin 충분히.

### 신규 도구 (이 세션)
- `eval/audit_tasks.py` — task×(future/deictic/gaze_val) 감사 + gaze-necessary subset.
- `eval/eval_tta.py` — 옵션 대칭화(위생, gap 측정용).
- `eval/eval_dump.py` (보강) — per-item + question/options/pred_text/gt_text.
- `eval/mcnemar.py`, `eval/pertask_compare.py` — 유의성/per-task.
- `models/foveal_roi.py` — ④ ROI 모듈(sweep-ready, ② 다음 후보).
