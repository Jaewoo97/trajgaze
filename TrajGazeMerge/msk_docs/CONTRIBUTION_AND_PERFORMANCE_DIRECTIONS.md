# Research Directions for Stronger Contribution and Performance

## 1. 현재 결과의 핵심 해석

현재 가장 유효한 방법은 **M1: VisionZip-Complement**이다. 

- VisionZip: content attention 기반으로 전체 visual token의 10%를 선택
- M1:
  - VisionZip content token 7%
  - VisionZip이 선택하지 않은 token 중 TAS salience가 높은 complement token 3%
- 성능:
  - VisionZip: 62.51%
  - VZ+traj: 62.71%
  - M1: 63.01%
  - Scanpath: 63.01%이지만 gate collapse로 인해 사실상 M1과 동일
  - Gaze-tag: 61.62%

현재 결과가 보여주는 핵심은 다음과 같다.

> Gaze/trajectory 정보를 별도의 side-channel로 추가하는 것보다, content-based pruning이 놓친 시각적 증거를 복원하는 selection signal로 사용하는 것이 더 효과적이다.

따라서 이후 연구는 raw scanpath token이나 gaze feature를 추가하는 방식보다, **query, temporal context, gaze reliability를 활용하여 visual token selection 자체를 개선하는 방향**이 적절하다.

---

# 2. 논문의 핵심 contribution으로 발전시킬 방향

## 2.1 추천 메인 방향

### Query-Adaptive Foveated Complementary Token Selection

고정된 7% content + 3% trajectory complement를 다음과 같이 확장한다.

```text
10% visual-token budget
├── Global content tokens
├── Query-relevant complement tokens
├── Gaze/hand-relevant complement tokens
└── Temporal-transition or high-resolution foveal tokens
```

핵심 아이디어는 다음과 같다.

1. Content-based selector는 장면의 전역 문맥을 보존한다.
2. Query-conditioned selector는 현재 질문에 필요한 증거를 복원한다.
3. Gaze/hand selector는 content attention이 놓친 interaction-relevant 영역을 복원한다.
4. Temporal selector는 gaze 이동, 손 이동, 객체 상태 변화가 발생하는 시점을 보존한다.
5. 모든 pool은 가능한 한 서로 겹치지 않도록 구성하고, 전체 budget은 고정한다.

### 예상 contribution 문장

> We propose a query-adaptive complementary token selection framework that preserves global visual context while recovering gaze-, hand-, and temporally relevant evidence discarded by content-based video token pruning under a fixed token budget.

### 기존 M1 대비 확장점

| 항목 | M1 | 확장 방법 |
|---|---|---|
| Content 정보 | VisionZip 7% | 유지 또는 개선 |
| Gaze 정보 | TAS complement 3% | reliability-aware adaptive quota |
| Query 정보 | 사용하지 않음 | question-conditioned selection |
| Temporal 정보 | global top-k에 간접 반영 | temporal coverage 및 transition 보장 |
| 해상도 | 기존 token 복원 | gaze ROI high-resolution re-encoding 가능 |
| Budget | 고정 10% | 동일하게 고정 10% |

---

# 3. 우선적으로 시도할 성능 향상 방법

## Priority 1. Query-Adaptive Temporal Complement

### Motivation

현재 M1의 complement는 질문과 무관하게 TAS global top-k로 선택된다.

따라서 다음 문제가 발생할 수 있다.

- 질문과 무관한 gaze 영역 선택
- 특정 프레임으로 token 집중
- gaze가 이동하는 중간 과정 소실
- temporal reasoning 질문에서 필요한 과거 또는 미래 시점 누락

### 제안 구조

```text
Content pool C:
- VisionZip으로 6~7% 선택

Query pool Q:
- C에 포함되지 않은 token 중 question relevance가 높은 token 선택

Gaze pool G:
- C와 Q에 포함되지 않은 token 중 TAS salience가 높은 token 선택

Temporal pool T:
- C, Q, G에 포함되지 않은 token 중 temporal novelty가 높은 token 선택
```

예시 budget:

```text
Content:               7.0%
Query complement:      1.0%
Gaze/hand complement:  1.5%
Temporal complement:   0.5%
Total:                 10.0%
```

### Query relevance 계산 방법

- question stem을 text encoder 또는 Qwen text embedding으로 encoding
- visual token과 cosine similarity 또는 lightweight cross-attention 계산
- 선택지 정보는 사용하지 않거나, 모든 선택지를 대칭적으로 aggregation
- 정답 leakage가 발생하지 않도록 주의

### Temporal novelty 후보

- 연속 프레임 token 간 feature difference
- gaze displacement
- hand velocity
- object-state change
- optical flow magnitude
- TAS salience 변화량

### 추천 이유

- 기존 M1 코드 수정량이 비교적 작음
- 추가 대형 encoder가 필요하지 않음
- 고정 10% budget 유지
- query-aware contribution을 확보할 수 있음
- temporal reasoning 성능 개선 가능성이 높음

---

## Priority 2. Frame-Balanced or Event-Balanced Complement

### Motivation

현재 global top-k 방식은 특정 프레임이나 특정 fixation 주변에 complement token이 집중될 수 있다.

이는 비디오 전체의 temporal coverage를 손상시킨다.

### 방법 A: Per-frame minimum quota

각 선택 프레임에 최소 complement token 수를 보장한다.

```text
각 프레임에 최소 1개 또는 2개 token 할당
남은 budget은 global TAS top-k로 할당
```

### 방법 B: Frame-level importance allocation

프레임 중요도를 다음과 같이 계산한다.

```text
frame_score =
α × question relevance
+ β × gaze confidence
+ γ × hand motion
+ δ × temporal novelty
```

프레임별 budget:

```text
b_t = round(total_complement_budget × softmax(frame_score_t))
```

단, 모든 중요 프레임에 최소 quota를 보장한다.

### 방법 C: Fixation/event-based allocation

고정 프레임 단위가 아니라 다음 event를 중심으로 token을 배정한다.

- fixation onset
- gaze shift
- hand-object contact
- object state transition
- action boundary

### 장점

- 구현 난이도가 낮음
- M1의 global top-k 문제를 직접 검증 가능
- temporal QA에서 개선 가능
- 이후 adaptive routing의 기초가 됨

---

## Priority 3. Gaze-Guided Foveated ROI Re-encoding

### Motivation

M1은 VisionZip이 버린 기존 저해상도 token을 복원한다.

그러나 gaze가 가리키는 물체가 작거나 세부 정보가 중요할 경우, 기존 token 자체에 충분한 시각 정보가 없을 수 있다.

예:

- 작은 버튼
- 칼끝
- 손잡이
- 재료의 상태
- 작은 도구
- object contact point

### 제안 구조

```text
Original video
├── Low-resolution global view
│   └── global content tokens
└── Gaze/hand-centered ROI crops
    └── high-resolution foveal tokens
```

최종 budget 예시:

```text
Global tokens:  6~7%
Foveal tokens:  3~4%
Total:          10%
```

### ROI 생성 방법

1. gaze fixation 추출
2. gaze point 주변 crop 생성
3. hand 위치가 가까우면 ROI 확장
4. fixation 전후 프레임을 temporal tube로 구성
5. 중복 ROI 제거
6. ROI를 고해상도로 vision encoder에 입력
7. ROI token을 global token과 병합

### 중요한 설계 원칙

- gaze point만 매우 작게 crop하지 않음
- object context와 hand-object relation이 포함되도록 margin 사용
- global thumbnail을 반드시 유지
- 모든 프레임을 re-encode하지 않고 fixation/event frame만 사용
- 전체 visual-token budget을 동일하게 유지

### 기대 contribution

> Gaze is used not only to rank existing tokens but also to allocate visual resolution to interaction-relevant regions.

### 단점

- preprocessing 및 vision encoding 비용 증가
- ROI와 global token의 position alignment 필요
- 구현 난이도가 높음

### 평가 가치

성능과 novelty를 동시에 높일 가능성이 가장 큰 방향이다.

---

## Priority 4. Adaptive Content–Gaze–Temporal Budget Routing

### Motivation

모든 질문에 동일한 7:3 비율이 최적이라는 보장은 없다.

질문 유형에 따라 필요한 정보가 다르다.

| 질문 유형 | 주요 정보 |
|---|---|
| Object/spatial | 현재 프레임과 gaze target |
| Temporal | 여러 프레임의 순서 |
| Intent/proactive | gaze 이동, hand motion, interaction target |
| General scene | global content |
| Fine-grained object | high-resolution ROI |

### Rule-based 초기 버전

```text
Spatial question:
- Content 6%
- Query 1%
- Gaze 2.5%
- Temporal 0.5%

Temporal question:
- Content 6%
- Query 1%
- Gaze 1%
- Temporal 2%

Intent question:
- Content 5.5%
- Query 1%
- Gaze 2%
- Temporal 1.5%

General question:
- Content 8%
- Query 1%
- Gaze 0.5%
- Temporal 0.5%
```

### Learned router 버전

입력:

- question embedding
- gaze confidence
- gaze entropy
- hand visibility
- clip motion statistics
- TAS salience distribution

출력:

```text
[r_content, r_query, r_gaze, r_temporal]
```

제약:

```text
r_content + r_query + r_gaze + r_temporal = 0.10
```

### 장점

- 고정 ratio의 한계를 해결
- query-aware contribution 강화
- gaze 품질이 낮은 샘플에서 과도한 gaze 사용 방지

### 주의점

router가 training set에 과적합할 수 있으므로, 먼저 rule-based ablation으로 효과를 확인한 뒤 learned router를 적용한다.

---

## Priority 5. Reliability-Aware Gaze Selection

### Motivation

gaze 정보가 부정확하거나 누락되면 gaze-based selection이 오히려 유해할 수 있다.

현재 M1은 gaze 품질과 무관하게 3% complement를 항상 사용한다.

### Reliability signal 후보

- gaze validity ratio
- fixation duration
- gaze confidence
- gaze trajectory smoothness
- gaze entropy
- frame 밖 gaze 비율
- TAS salience entropy
- gaze–hand consistency
- gaze–object consistency

### 제안

```text
reliability가 높음:
- gaze quota 증가

reliability가 낮음:
- gaze quota 감소
- content 또는 temporal quota 증가
```

예:

```text
r_gaze = 0.5% + 2.5% × reliability
```

### 추가 loss

- reliability calibration loss
- gaze dropout consistency
- shuffled-gaze rejection loss

### 필수 control 실험

- 정상 gaze
- shuffled gaze
- time-reversed gaze
- spatially jittered gaze
- no gaze

정상 gaze만 유의미하게 좋아야 실제 gaze 정보를 사용한다고 주장할 수 있다.

---

## Priority 6. Object-Centric Fixation Tube Selection

### Motivation

raw gaze coordinate와 개별 patch token만으로는 gaze가 어떤 물체를 의미하는지 표현하기 어렵다.

Scanpath side-channel이 실패한 이유 중 하나는 `(x, y)` trajectory를 Qwen이 의미적으로 해석해야 했기 때문일 수 있다.

### 제안 표현

```text
Raw coordinate sequence:
(x1, y1) → (x2, y2) → (x3, y3)

Object-centric sequence:
cup → kettle → cup
```

### Pipeline

```text
gaze fixation
→ fixated region/object proposal
→ adjacent-frame tracking
→ gaze–object–hand tube
→ tube representative token selection
```

각 tube에 포함할 정보:

- object visual embedding
- fixation duration
- gaze transition direction
- hand distance
- hand-object contact
- temporal order
- object state change

### 활용 방법

별도 side-channel로 추가하기보다 다음 용도로 우선 사용한다.

1. fixation tube 내부의 visual token 선택
2. tube별 최소 token quota 보장
3. 동일 object의 중복 token 병합
4. interaction-critical object의 resolution 증가

### 기대 효과

- raw gaze보다 semantic grounding이 강함
- temporal reasoning과 intent reasoning에 유리
- qualitative visualization이 쉬움

---

## Priority 7. Teacher-Distilled QA-Aware Token Selector

### Motivation

TAS salience는 gaze/hand와 관련 있는 patch를 찾지만, 현재 질문의 정답에 필요한 patch와 항상 일치하지 않는다.

### Teacher–student 설정

```text
Teacher:
- full visual tokens 또는 20% token model

Student:
- 10% token model
```

### Lightweight selector 입력

```text
visual token embedding
+ TAS salience
+ question embedding
+ frame position
+ temporal novelty
+ gaze reliability
```

출력:

```text
QA utility score per token
```

### 학습 loss

```text
L =
L_answer
+ λ1 L_logit_distillation
+ λ2 L_token_importance
+ λ3 L_temporal_coverage
+ λ4 L_diversity
```

가능한 teacher signal:

- full-token model의 cross-attention
- leave-one-token-out logit drop
- gradient-based token importance
- masking-based answer confidence drop

### 장점

- heuristic top-k를 넘어 downstream QA objective에 의해 selector 학습
- M1의 TAS prior를 유지하면서 question relevance를 학습
- 강한 methodological contribution 가능

### 단점

- teacher inference 비용
- token importance 계산 비용
- 잘못된 teacher attention을 그대로 학습할 위험

### 효율화

- teacher token importance와 visual embedding을 offline cache
- Qwen backbone 고정
- selector와 LoRA만 학습

---

## Priority 8. Base Content Selector 교체

M1 complement가 충분히 강하다면, 다음 병목은 7% content pool일 수 있다.

### 후보 A: Temporal redundancy-aware content selector

목표:

- 유사한 연속 프레임의 중복 token을 merge
- action transition frame을 더 많이 보존

조합:

```text
Temporal redundancy-aware content 7%
+
TAS complement 3%
```

### 후보 B: Diversity-aware content selector

목표:

- 동일한 배경이나 유사한 객체 token의 중복 선택 방지
- 전역 장면 coverage 향상

조합:

```text
Diverse global content 7%
+
Gaze/hand complement 3%
```

### 후보 C: Query-conditioned content selector

목표:

- VisionZip attention만 사용하지 않고 question relevance 반영

조합:

```text
Query-conditioned content 7%
+
TAS complement 3%
```

### 추천 순서

1. 기존 M1에서 complement 개선
2. 이후 content selector 교체
3. 동일 complement를 여러 base selector에 적용해 일반성 검증

---

# 4. 우선순위가 낮은 접근

현재 결과를 고려하면 다음 방식은 우선순위가 낮다.

## 4.1 Raw scanpath token 재추가

- 기존 Scanpath gate가 거의 0으로 감소
- 추가 token이 실제로 사용되지 않음
- 추가 module과 token budget만 증가

다시 시도하려면 raw `(x, y)`가 아니라 object-grounded fixation representation으로 변경해야 한다.

## 4.2 Gaze-tag feature 추가

- gate가 증가했지만 validation 성능 하락
- gaze feature를 사용하면서 overfitting된 정황
- 현재 데이터와 평가에서는 일반화에 불리할 가능성

## 4.3 Attention과 trajectory score의 단순 합 또는 곱

```text
attention × trajectory
attention + λ × trajectory
```

- VisionZip이 완전히 놓친 token을 복원하기 어려움
- score scale과 distribution에 민감
- 서로 다른 역할의 signal이 동일 ranking 안에서 경쟁

## 4.4 Complement diversity만 강제

기존 top-k complement보다 성능이 낮았다면, gaze target 주변의 유사 token이 실제로 필요한 evidence일 수 있다.

diversity는 complement보다 global content pool에 적용하는 것이 더 자연스럽다.

## 4.5 Epoch 수 증가

현재 epoch 3에서 validation 성능이 하락하므로 단순한 장기 학습은 해결책이 아니다.

필요한 것은:

- regularization
- validation split 개선
- seed 반복
- selector objective 개선
- data augmentation

---

# 5. 추천 실험 로드맵

## Stage 0. 현재 M1 효과 검증

### 필수 실험

- 3~5 random seeds
- mean ± standard deviation
- VisionZip과 M1의 paired prediction 비교
- bootstrap confidence interval 또는 McNemar test
- 정상 gaze / shuffled gaze / no gaze 비교
- token selection visualization

### 목표

M1의 0.50%p 향상이 실제 gaze complement 효과인지 확인한다.

---

## Stage 1. 저비용 성능 개선

### Experiment 1

**Frame-balanced TAS complement**

```text
VisionZip 7%
+
frame-balanced TAS 3%
```

### Experiment 2

**Query + TAS disjoint complement**

```text
VisionZip 7%
+
Query 1%
+
TAS 2%
```

### Experiment 3

**Query + TAS + temporal complement**

```text
VisionZip 7%
+
Query 1%
+
TAS 1.5%
+
Temporal 0.5%
```

### Experiment 4

**Reliability-aware ratio**

```text
gaze reliability에 따라 TAS quota를 0.5~3%로 조절
```

### 이 단계의 목표

추가 대형 모델 없이 10% budget에서 M1을 안정적으로 초과하는 설정을 찾는다.

---

## Stage 2. 논문 contribution 강화

### Experiment 5

**Adaptive query-conditioned router**

```text
Question + gaze reliability + clip statistics
→ token budget allocation
```

### Experiment 6

**Object-centric fixation tube selection**

```text
gaze → object/tube → representative visual tokens
```

### Experiment 7

**High-resolution foveated ROI**

```text
global low-resolution tokens
+
gaze/hand ROI high-resolution tokens
```

### 이 단계의 목표

단순 heuristic 개선을 넘어 독립적인 method contribution을 확보한다.

---

## Stage 3. 최종 메인 방법

### 추천 최종 구조

```text
Query-Adaptive Foveated Complementary Token Selection

1. Global content selection
2. Question-conditioned evidence recovery
3. Reliability-aware gaze/hand complement
4. Temporal transition coverage
5. Optional high-resolution fixation ROI
6. Fixed total token budget
```

### 최종 모델의 핵심 주장

1. Content-based pruning은 interaction-relevant evidence를 제거한다.
2. Gaze는 별도 side-channel보다 discarded evidence recovery에 효과적이다.
3. 질문 종류와 gaze 신뢰도에 따라 필요한 token의 종류가 다르다.
4. 고정 budget 안에서 global, query, gaze, temporal evidence를 동적으로 배분하면 정확도와 효율을 동시에 개선할 수 있다.
5. gaze ROI에 추가 해상도를 배분하면 작은 interaction target에 대한 세부 정보를 보존할 수 있다.

---

# 6. 필수 ablation 구성

## 6.1 Selection component ablation

| Model | Content | Query | Gaze | Temporal | Foveal ROI |
|---|---:|---:|---:|---:|---:|
| VisionZip | ✓ |  |  |  |  |
| M1 | ✓ |  | ✓ |  |  |
| + Query | ✓ | ✓ | ✓ |  |  |
| + Temporal | ✓ | ✓ | ✓ | ✓ |  |
| + Foveal | ✓ | ✓ | ✓ | ✓ | ✓ |

## 6.2 Token ratio ablation

```text
Content:Gaze = 9:1
Content:Gaze = 8:2
Content:Gaze = 7:3
Content:Gaze = 6:4
```

확장 모델:

```text
Content:Query:Gaze:Temporal
7:1:1.5:0.5
6:1:2:1
6:2:1:1
```

## 6.3 Token budget curve

```text
5%
7.5%
10%
15%
20%
```

평가 목적:

- 극단적 compression에서만 유효한지
- 일반적인 token selector인지
- full-token 성능과의 gap을 얼마나 줄이는지

## 6.4 Gaze quality ablation

```text
Ground-truth gaze
Predicted gaze
Jittered gaze
Shuffled gaze
No gaze
```

## 6.5 Temporal ablation

```text
Global top-k
Per-frame quota
Fixation-event quota
Temporal transition quota
```

## 6.6 Query ablation

```text
No query conditioning
Question stem only
Question + all options aggregation
Question-type routing
Learned query router
```

## 6.7 ROI ablation

```text
Gaze point only
Gaze-centered fixed crop
Gaze + hand union crop
Fixation temporal tube
Global + ROI
ROI only
```

---

# 7. 반드시 포함해야 할 분석

## 7.1 질문 유형별 성능

- spatial/object
- temporal ordering
- causal
- intent/proactive
- past
- present
- future
- hand-object interaction

전체 평균만으로는 gaze의 효과가 가려질 수 있다.

## 7.2 Token selection visualization

각 사례에서 다음을 함께 시각화한다.

- VisionZip token
- M1 TAS complement
- query complement
- temporal complement
- gaze point
- hand position
- model prediction
- correct answer

## 7.3 성공 및 실패 사례

### 성공 사례

- VisionZip이 작은 interaction object를 놓침
- complement가 해당 영역을 복원
- 제안 방법만 정답

### 실패 사례

- gaze annotation 오류
- 질문과 무관한 fixation
- object가 ROI 밖에 있음
- temporal context 부족
- duplicate token 과다 선택

## 7.4 Efficiency 분석

- retained token ratio
- FLOPs
- GPU memory
- latency
- trainable parameters
- extra vision encoder cost
- accuracy–efficiency curve

고정 token budget이더라도 foveal ROI re-encoding은 추가 vision cost가 있으므로 반드시 별도로 보고한다.

---

# 8. 실험 선택 기준

| 방법 | 예상 성능 잠재력 | Novelty | 구현 난이도 | 우선순위 |
|---|---:|---:|---:|---:|
| Frame-balanced complement | 중간 | 낮음 | 낮음 | 매우 높음 |
| Query-adaptive complement | 높음 | 중간~높음 | 중간 | 매우 높음 |
| Reliability-aware routing | 중간~높음 | 중간 | 중간 | 높음 |
| Temporal-transition complement | 높음 | 중간 | 중간 | 높음 |
| Foveated ROI re-encoding | 매우 높음 | 높음 | 높음 | 높음 |
| Object-centric fixation tube | 높음 | 높음 | 높음 | 중간~높음 |
| Teacher-distilled selector | 매우 높음 | 매우 높음 | 매우 높음 | 중간 |
| Base selector 교체 | 중간 | 낮음~중간 | 중간~높음 | 중간 |
| Raw scanpath side-channel | 낮음 | 낮음 | 중간 | 낮음 |
| Gaze-tag | 낮음 | 낮음 | 낮음 | 낮음 |

---

# 9. 최종 권장안

## 가장 현실적인 다음 실험

```text
M1
+
Query-conditioned complement
+
Frame-balanced temporal allocation
+
Gaze reliability-aware quota
```

추천 초기 구성:

```text
Content:  7.0%
Query:    1.0%
Gaze:     1.5%
Temporal: 0.5%
Total:   10.0%
```

이 설정은 다음 장점이 있다.

- 기존 M1 코드를 최대한 재사용
- 고정 token budget 유지
- 추가 대형 side-channel 없음
- 질문 및 시간 정보를 명시적으로 활용
- method contribution을 M1보다 강화
- 실패 원인을 component별로 분석 가능

## 가장 강한 최종 방법 후보

```text
Query-Adaptive Foveated Complementary Token Selection
```

구성:

```text
Low-resolution global content
+
Query-relevant discarded evidence
+
Reliability-aware gaze/hand evidence
+
Temporal transition coverage
+
High-resolution fixation ROI
```

이 방향은 성능 향상뿐 아니라 다음과 같은 명확한 논문 메시지를 제공한다.

> Efficient video VLMs should not treat gaze as an auxiliary feature channel. Instead, gaze should guide where limited visual tokens and visual resolution are allocated, while query and temporal context determine which gaze-related evidence is useful for answering the current question.

---

# 10. 단기 결론

다음 순서로 진행하는 것이 가장 효율적이다.

1. M1을 여러 seed와 gaze control로 재검증
2. frame-balanced complement 적용
3. query-conditioned disjoint complement 추가
4. gaze reliability에 따른 quota 조절
5. temporal transition token 추가
6. 성능이 확인되면 foveated ROI re-encoding 도입
7. 최종적으로 adaptive router 또는 teacher-distilled selector로 통합

현재 결과에서는 gaze 표현을 더 추가하는 것보다, **질문에 맞는 gaze-related visual evidence를 고정 budget 안에서 선택하고 필요한 영역에 해상도를 집중하는 방향**이 가장 유망하다.
