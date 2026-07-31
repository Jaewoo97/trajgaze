# Qualitative figure toolkit — 인수인계용 번들

토큰 선택(token selection) 방법 4종을 **같은 10% visual-token 예산**에서 비교하는 정성 figure를
만드는 코드 일습입니다. figure 한 장은 4행이고, 각 행은 같은 프레임 스트립 위에 그 방법이 실제로
남긴 토큰을 박스로 그린 뒤, 그 방법의 MCQ 예측을 오른쪽에 붙입니다.

결과물 예시: `example/expected/fig_qual_sg83.png` (StreamGaze item 83, Ours/PruneVid 정답,
VisionZip/FastVID 오답).

---

## 0. 두 갈래로 나뉩니다 — 먼저 어느 쪽이 필요한지 정하세요

| | A. **렌더링만** (`render/`) | B. **전체 파이프라인** (`pipeline/`) |
|---|---|---|
| 하는 일 | layout JSON 한 개 → PDF / PPTX / PNG | 모델 4종을 돌려서 토큰 선택과 예측을 뽑고 figure + layout JSON 생성 |
| 필요한 것 | python 3.9+, pillow, matplotlib(PDF), python-pptx(PPTX) | 위 + torch, Qwen2.5-VL-7B, LoRA 체크포인트 4개, traj encoder, StreamGaze/EgoGazeVQA 데이터셋, GPU |
| 이식성 | **어디서나 그대로 돎.** 이 번들의 `example/`로 즉시 검증 가능 | 우리 repo(`TrajGazeMerge`) + VisionZip fork에 묶여 있음. 경로 수정 필요 |
| 다른 프로젝트에서 재사용 | **여기가 재사용 지점.** §3의 JSON 스키마만 맞추면 어떤 모델이든 이 figure를 그릴 수 있음 | 우리 세팅 전용 |

즉 **"우리와 다른 모델/데이터로 같은 그림을 그리고 싶다"** 면 A만 가져가고 §3 스키마에 맞춰 JSON을
뱉으면 됩니다. B는 우리 실험을 그대로 재현할 때만 필요합니다.

---

## 1. Quickstart (A: 렌더링) — 이 번들만으로 검증됨

```bash
pip install pillow python-pptx matplotlib      # requirements.txt 참고

# PowerPoint (편집 가능한 도형/텍스트로 재구성)
python render/render_qual_pptx.py example/sg83_layout.json \
  --font render/Inter.ttf --out out/fig.pptx

# 벡터 PDF (텍스트/도형이 벡터, 사진만 raster)
python render/render_qual_vector.py example/sg83_layout.json \
  --font render/Inter.ttf --out out/fig.pdf --png-dpi 300
```

출력이 `example/expected/`의 파일들과 같으면 정상입니다. 두 렌더러 모두 **GPU도, 우리 repo도
필요 없습니다.** 실행 시간은 figure 한 장에 수 초입니다.

주요 옵션

| 옵션 | 대상 | 뜻 |
|---|---|---|
| `--type-scale 1.6` | 둘 다 | 폰트 크기만 배율 적용(레이아웃 재계산). 논문 `\linewidth` 배치용 권장값, §5 참조 |
| `--frames-root DIR` | 둘 다 | JSON에 적힌 프레임 경로가 이 머신에 없을 때 프레임이 있는 위치를 지정 (`render/render_paths.py`가 탐색 순서를 문서화) |
| `--png-dpi N` | vector | PDF와 함께 같은 이름의 PNG도 저장 |
| `--no-group` | pptx | 프레임 단위 그룹 없이 700여 개 낱개 도형으로 (기본은 24개 그룹) |
| `--font-name` | pptx | PowerPoint가 요청할 글꼴 이름 (기본 `Inter`) |

PPTX는 사진 6장만 이미지이고 카드/토큰 박스 609개/gaze ring/칩/모든 문자열이 네이티브 도형이라
PowerPoint에서 바로 문구·색·배치를 고칠 수 있습니다. 도형에 이름이 붙어 있어(`option-B`,
`card-Ours`, `pred-VisionZip`, `token`, `chip`) 선택 창에서 찾기 쉽습니다.

---

## 2. figure가 실제로 보여주는 것

```
 [질문]  Among {spatula, glass, poster, patty}, which did the user never gaze at?
 [보기]  A. spatula   B. glass(정답, 파란 막대)   C. poster   D. patty
 [범례]  ■ content-based 7%   ■ gaze/hand complement 3%   ■ baseline kept 10%   ○ gaze

 ┌ Ours       │ [프레임1][프레임2]…[프레임6] │ PREDICTION: B. glass  (초록=정답)
 ├ VisionZip  │ 같은 프레임, 그 방법이 남긴 토큰 │ PREDICTION: C. poster (빨강=오답)
 ├ PruneVid   │ …                              │ …
 └ FastVID    │ …                              │ …
```

- 각 프레임 위의 박스 = 그 방법이 **실제로 유지한 visual token**을 12x18 토큰 격자 좌표로 그린 것.
  Ours만 2개 레이어(초록 윤곽 = content 7%, 마젠타 채움 = gaze/hand complement 3%)입니다.
- 노란 링 = 그 프레임의 실제 gaze 좌표. 프레임 이미지에 이미 초록 점으로 baked-in 되어 있는
  값과 같아야 정상입니다(오차 중앙값 0.0001로 검증).
- 4행 모두 **같은 frozen backbone**을 쓰지만 **행마다 자기 selector로 학습한 자기 LoRA adapter**를
  로드합니다. footer 문구가 이 사실을 말하도록 되어 있으니, 캡션을 새로 쓸 때도 "선택 규칙만
  다르다"고 쓰면 안 됩니다(그건 사실이 아님).

---

## 3. layout JSON 스키마 — 다른 모델에 이식할 때의 계약

`pipeline/viz_qual_pretty.py --dump-layout DIR`가 뱉는 파일이고, `render/`의 두 렌더러가 먹는
유일한 입력입니다. **이 JSON만 만들어 주면 우리 모델/데이터가 전혀 없어도 같은 figure가 나옵니다.**

```jsonc
{
  "source": "sg",                  // "sg" | "eg". 렌더러는 chip 규칙에만 사용(§4-3)
  "idx": 83, "task": "past_non_fixated_object_identification", "flags": "O1V0P1F0",
  "question": "Among {spatula, glass, poster, patty}, which did the user never gaze at?",
  "options": ["A. spatula", "B. glass", "C. poster", "D. patty"],
  "answer": "B",                   // 정답 letter. 보기 강조에만 사용
  "grid": [12, 18],                // 프레임당 토큰 격자 [행 s_h, 열 s_w]
  "disp_w": 210,                   // 프레임 표시 폭(design unit = pt). 높이는 원본 비율로 계산
  "note": "6 frames selected by the authors",   // footer 꼬리말

  // ↓ 선택. 없으면 우리 프로젝트 기본값이 찍히므로 다른 벤치마크면 반드시 덮어쓸 것
  "title":  "QUALITATIVE  ·  MyBench",          // 기본: source로 StreamGaze/EgoGazeVQA
  "legend": [{"rgb": [52,199,89], "label": "kept 10%"},
             {"rgb": [255,209,26], "label": "gaze", "shape": "ring"}],  // shape 기본 swatch
  "footer": "same budget, same frozen backbone", // note는 이 뒤에 붙음

  "strip": [                       // 표시할 프레임들, 왼→오
    {
      "t": 11,                     // temporal group index (칩 라벨 "t11")
      "half": 1,                   // 0/1: 그룹 안의 첫/둘째 프레임 (§4-2)
      "path": "frames/OP02-R05-Cheeseburger/frame_000556.jpg",
      "gaze": [0.53, 0.83],        // 정규화 좌표 [x,y] (0~1), 없으면 null
      "query_moment": false        // true면 파란 "final fixation" 칩 (§4-3)
    }
  ],

  "rows": [                        // 방법 1개 = 1행, 순서대로 그려짐
    {
      "name": "Ours", "sub": "7% content-based + 3% gaze/hand complement",
      "pred_letter": "B", "correct": true, "is_ours": true,   // is_ours=이름을 파랑으로
      "groups": [                  // strip과 길이가 같아야 함: groups[프레임][레이어]
        [ { "cells": [[0,0],[0,1]],   // [행,열] 토큰 좌표 목록
            "rgb": [52,199,89],       // 박스 색
            "fill_alpha": 0,          // 0=윤곽선만, 1~255=반투명 채움
            "width_u": 2 }            // 선 굵기(design unit)
        ]
      ]
    }
  ]
}
```

지켜야 할 불변식

1. `len(rows[i].groups) == len(strip)` — 행마다 프레임 수만큼의 레이어 묶음이 있어야 합니다.
2. `cells`의 좌표는 `grid` 범위 안이어야 합니다. 격자 크기는 자유(12x18은 Qwen2.5-VL-7B에
   128프레임을 넣었을 때의 값).
3. `path`는 절대/상대 모두 가능합니다. 상대 경로면 **JSON 파일 위치 기준**으로 찾습니다
   (이 번들의 `example/`이 그 방식). 탐색 순서는 `render/render_paths.py` docstring에 있습니다.
4. 행 개수와 프레임 개수는 자유지만, 레이아웃은 4행 x 6프레임에서 튜닝돼 있습니다.
5. `title`/`legend`/`footer`를 안 주면 **우리 프로젝트 문구가 그대로 찍힙니다**("StreamGaze",
   "content-based selection 7%", "frozen Qwen2.5-VL-7B backbone"). 다른 벤치마크에 쓸 때
   가장 먼저 틀리는 지점이니 반드시 덮어쓰세요.

이식 검증은 `example/minimal_layout.json`이 그대로 답입니다. 손으로 쓴 2행 x 2프레임 JSON에
6x8 격자, gaze 없는 프레임 1장, chrome 오버라이드까지 들어 있고 우리 모델은 전혀 관여하지
않습니다. 이걸 렌더링해서 나오면 이식은 끝난 것입니다.

```bash
python render/render_qual_vector.py example/minimal_layout.json \
  --font render/Inter.ttf --out out/minimal.pdf --png-dpi 60
```

---

## 4. 반드시 알아야 할 함정 5가지

우리가 실측으로 잡은 것들입니다. 코드를 고치거나 다른 백본에 이식할 때 그대로 재발합니다.

1. **gaze 마커는 "표시되는 프레임"의 raw gaze여야 합니다.** 예전 코드는 128→64 pooling된
   gaze(`_pool_to_T(gaze,T)[t]`)를 쓰면서 화면에는 raw 프레임을 보여줬고, saccade 구간에서
   링이 두 fixation 사이 허공에 찍혔습니다(표시 프레임의 47%가 프레임 폭 5% 이상 오차, 최악 0.52).
   지금은 `vi = 2t`(+half)의 raw gaze를 씁니다.
2. **temporal group 하나 = 프레임 2장.** Qwen2.5-VL은 `(2t, 2t+1)`을 그룹 t로 합칩니다. 둘 다
   같은 토큰 박스를 쓰므로 어느 쪽을 보여줘도 되지만, gaze는 반드시 표시한 쪽 것을 써야 합니다.
   strip의 `half` 필드가 그 선택입니다(우리 스트립 파일에서는 `"52b"` 표기).
3. **파란 `final fixation` 칩은 조건부입니다.** StreamGaze는 질문 시각에서 프레임 목록을 자르므로
   마지막 프레임이 곧 query moment지만, 그걸 묻는 건 `present_*`/`proactive_*` 과제뿐입니다.
   과거형 질문에 이 칩을 붙이면 독자를 답과 무관한 프레임으로 유도합니다. EgoGazeVQA는 애초에
   cutoff가 없어 어떤 경우에도 붙이면 안 됩니다.
4. **figure로 쓸 아이템은 근거 감사를 통과해야 합니다.** StreamGaze는 답의 근거가 되는 fixation
   구간이 **입력 종료 이후에 시작**하는 경우가 526개 중 401개(76%, 중앙값 2.1초)입니다. 수치를
   무효화하진 않지만(네 방법이 같은 프레임을 봄), figure는 답의 근거 장면을 못 보여줄 수 있습니다.
   `pipeline/audit/audit_evidence.py`가 게이트입니다.
5. **verdict는 재현 확인이 필요합니다.** 옵션 logit이 bf16에서 1/8로 양자화되어 동점이 흔하고,
   Ours 경로는 topk 근처 동점 때문에 실행마다 3% complement가 바뀝니다(같은 아이템 5회에
   margin 0.040~0.174). 새 아이템은 **min margin ≥ 0.05, Ours margin ≥ 0.25, 서로 다른 프로세스
   2회 동일 verdict** 세 조건을 모두 걸어야 합니다.

`docs/HANDOFF_QUAL_FIGURES.md`에 이 다섯 가지의 측정 근거와 아이템별 판정이 다 들어 있습니다.

---

## 5. 논문에 넣을 때: 크기

figure 원본은 28.1 x 15.7 in입니다. `\linewidth`(7 in)에 넣으면 4배 축소되므로 기본 배율에서는
질문이 5.7 pt까지 줄어듭니다. **레이아웃 문제이지 해상도 문제가 아니라서** supersample을 올려도
해결되지 않습니다.

| 요소 | design | ts=1.0 | **ts=1.6(권장)** | ts=2.0 |
|---|---|---|---|---|
| 질문 | 23 | 5.7 pt | **9.2 pt** | 11.5 pt |
| 보기 | 16 | 4.0 pt | **6.4 pt** | 8.0 pt |
| 예측 | 15 | 3.7 pt | **6.0 pt** | 7.5 pt |

본문 10 pt 기준 `--type-scale 1.6`이 적정선이고, 2.0이면 질문이 본문보다 커집니다.

---

## 6. B: 전체 파이프라인을 돌리려면

### 6.1 전제 조건

- **repo**: `TrajGazeMerge` (모델/데이터/학습 코드) + VisionZip fork의 `Qwen2_5_VL`
- **체크포인트 4개 + traj encoder**
  - Ours `visionzip_complement_learned_overlay/best.pth`, VisionZip `visionzip_lora_sgeg_overlay`,
    PruneVid `prunevid_sgeg_overlay`, FastVID `fastvid_sgeg_overlay`
  - traj encoder `stage1_tas_3way_overlay/best.pth` (Ours 전용)
  - 각 adapter는 224 tensor, validation accuracy로 고른 `best.pth`
- **데이터**: StreamGaze_v2 (프레임 + `metadata/egtea.csv`의 fixation episode), EgoGazeVQA
- **환경변수** `GAZE_OVERLAY=1` (gaze 점이 baked-in된 프레임을 쓰겠다는 뜻), GPU 1장

### 6.2 경로부터 고쳐야 합니다

`pipeline/`의 스크립트는 절대 경로가 박혀 있습니다. 그대로는 다른 서버에서 돌지 않습니다.

| 파일 | 고칠 것 |
|---|---|
| `viz_qual_pretty.py`, `viz_qual_compare.py`, `contact_sheet_ours.py`, `scan_candidates.py`, `audit/probe_*.py` | 상단 `sys.path.insert(...)` 2줄 (repo 루트, VisionZip `Qwen2_5_VL`) |
| `viz_qual_compare.py` | `STAGE1_DEFAULT` (traj encoder 경로) |
| `pick_fixation_frames.py` | `META` (`metadata/egtea.csv`) |
| `make_contact_sheet.py`, `audit/audit_*.py`, `audit/fix_gaze_legend.py` | `FONT` / `BASE` 상수 |
| `audit/probe_two_items.py`, `audit/probe_stability.py` | `CK` (체크포인트 디렉터리) |

### 6.3 실행 순서

```bash
# 1) 프레임 스트립 선정 (CPU, 수 초) — fixation episode에 앵커, 정답 정보 미사용
python pipeline/pick_fixation_frames.py --idxs 9,10,11,17 --out frames_sg.json

# 2) figure + layout dump (GPU, 18개 아이템에 ~15분)
GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0 python -m scripts.viz_qual_pretty \
  --source sg --idxs 9,10,11,17 --flags O1V0P0F0,O1V1P0F0 --limit 99 \
  --frames-json frames_sg.json \
  --frames-note "{n} frames anchored on the clip's annotated fixation episodes; the answer was not used" \
  --font Inter.ttf --gpu 0 \
  --ours $CK/visionzip_complement_learned_overlay/best.pth \
  --vz $CK/visionzip_lora_sgeg_overlay/best.pth \
  --pv $CK/prunevid_sgeg_overlay/best.pth \
  --fv $CK/fastvid_sgeg_overlay/best.pth \
  --out-dir out/ --dump-layout out/layout/

# 3) 벡터/PPTX로 재렌더 (CPU) — §1
```

`viz_qual_pretty.py`는 `python -m scripts.viz_qual_pretty`로 repo 루트에서 실행해야 합니다
(패키지 상대 import 때문). 나머지는 단독 실행 가능합니다.

### 6.4 pipeline 스크립트별 역할

| 파일 | 역할 |
|---|---|
| `viz_qual_pretty.py` | **메인 렌더러.** 4개 방법을 돌려 토큰 선택·예측을 얻고 PNG를 조판. `--dump-layout`으로 §3 JSON 저장. 옵션 전부 docstring에 문서화 |
| `viz_qual_compare.py` | 모델 로딩과 방법별 토큰 선택 로직(위가 import함). 프레임 3행 버전의 원본 |
| `pick_fixation_frames.py` | fixation episode에 앵커된 스트립 생성. filler 프레임은 ±2 그룹 내에서 가장 선명한 쪽으로(Laplacian 분산) 이동 |
| `contact_sheet_ours.py` | 128프레임 전부에 Ours 선택을 그린 contact sheet. 스트립을 손으로 고를 때 사용 |
| `make_contact_sheet.py` | 질문·보기만 보여주는(정답 비공개) contact sheet |
| `scan_candidates.py` | 렌더링 없이 verdict + 방법별 top-2 margin만 스캔. 후보 탐색/안정성 게이트 |
| `audit/audit_evidence.py` | **아이템 게이트.** 답의 근거 episode가 입력 안에 있는지, 스트립이 그것을 보여주는지 판정 |
| `audit/check_gaze_align.py`, `audit/check_pool_offset.py` | 함정 1의 회귀 테스트 |
| `audit/probe_two_items.py`, `audit/probe_stability.py` | 아이템 1개의 옵션 분포·margin, 반복 실행 안정성 |

---

## 7. 파일 목록

```
render/                    A. 어디서나 도는 렌더러 (이식 시 여기만 있으면 됨)
  render_qual_pptx.py        layout JSON → 편집 가능한 .pptx
  render_qual_vector.py      layout JSON → 벡터 PDF (+ --png-dpi로 PNG)
  layout_common.py           두 렌더러 공용: 프레임 경로 탐색 + 제목/범례/footer 기본값
  Inter.ttf                  타이포 (variable font)
example/
  sg83_layout.json           §3 스키마 실물(우리 figure), 프레임 경로는 상대 경로
  minimal_layout.json        손으로 쓴 2행 x 2프레임 이식 테스트용 (모델 무관)
  frames/…                   그 6장 (EGTEA Gaze+ 원본, 아래 주의)
  expected/                  기대 출력 (pptx / pdf / png)
  frames_sg.json             우리가 쓰는 18개 아이템의 스트립 정의 (참고용)
pipeline/                  B. 우리 세팅 전용 (GPU + repo + 데이터)
docs/
  HANDOFF_QUAL_FIGURES.md    측정·감사·아이템 판정 전체 기록 (내부 문서, 함정 5가지의 근거)
  design.md                  비주얼 언어(색·타이포·간격 토큰)
  QUAL_FIGURE_sg83.md        예제 아이템의 캡션과 측정치
```

**데이터 주의**: `example/frames/`의 6장은 EGTEA Gaze+ 영상 프레임입니다. 재배포 조건은 EGTEA
라이선스를 따르니, 외부에 넘길 때는 확인하거나 프레임을 빼고 상대에게 직접 받게 하십시오
(프레임이 없으면 렌더러는 경로 에러만 내고, 나머지 코드는 그대로 동작합니다).

## 8. 알려진 제약

- PPTX의 글꼴은 Inter로 지정됩니다. Inter가 설치되지 않은 PC에서 열면 기본 산세리프로 대체되고
  줄바꿈 위치가 조금 달라집니다(`render/Inter.ttf` 설치 또는 `--font-name`).
- PPTX 안에서 예측 문구를 고쳐도 모델을 다시 돌리는 것이 아닙니다. 구조를 바꾸려면 §6의
  `--dump-layout`부터 다시 가야 합니다.
- 벡터 PDF의 텍스트는 matplotlib이 Inter의 default instance로 그리므로 굵기 구분이 PNG만큼
  선명하지 않습니다(PPTX는 bold 속성으로 처리).
