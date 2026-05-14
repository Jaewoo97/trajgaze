# Sprint 1 — Path-Forward Execution Plan

새 세션에서 작업을 이어갈 수 있도록 만든 self-contained 실행 문서. 진단 결과의 *왜* 와 *무엇* 은 `docs/diagnostic_report_E1_keep10.md` §12–§13 에서 다루고, 본 문서는 **어떻게 실행하는가** 만 다룬다.

---

## 0. Context (한 단락 요약)

진단 결과 (`diagnostic_report_E1_keep10.md` §12) 에서 encoder가 spatial-gaze-attention이 아닌 **temporal-only salience curve** 임이 확인됨 (`frame_index` Pearson r = 0.498, 다른 trajectory feature corr ≈ 0). 이는 두 가지 근본 원인 때문:

1. `query_emb = zeros` → encoder가 질문을 못 봄 (sample-agnostic)
2. `I_scores_past` supervision의 SIGMA가 커서 (Gaze=16px, Hand=24px on 224px frame) spatial 위치를 강하게 강제하지 않음

**Sprint 1 목표**: 위 두 가지를 동시에 수정하여 encoder가 paper narrative대로 (질문 conditional + spatially localized) 작동하게 만들고, 그것이 정확도 향상으로 이어지는지 검증.

---

## 1. Success Criteria (Sprint 1 완료 시점 측정)

새 학습 ckpt에 대해 `diagnostic_eval.py` + `ablation_score_source.py` + `counterfactual_mask_eval.py` 를 526 EGTEA test에 재실행. 4개 지표 동시 측정:

| 지표 | 현재 (baseline) | Sprint 1 success | Sprint 1 partial | Sprint 1 fail |
|---|---:|---:|---:|---:|
| **gt_gaze_recall** | 0.077 (< random 0.10) | **> 0.20** | 0.10–0.20 | < 0.10 |
| **shuffle_kept Δ vs baseline** | +0.95pp (no penalty) | **< −3pp** (셔플 시 acc 하락) | −1 ~ −3pp | ≈ 0 |
| **frame_index corr (mean)** | 0.498 | **< 0.30** | 0.30–0.45 | ≥ 0.45 |
| **Overall accuracy** | 68.44% | **≥ 67%** | 65–67% | < 65% |

→ 4개 모두 success 셀이면 narrative 복원, 1–3개면 partial, 0개면 fail.

---

## 2. 개입 A — Question-conditioning 활성화

### 2.1 변경 대상
**파일**: [`TrajGaze_v2/models/model_temporal.py`](../TrajGaze_v2/models/model_temporal.py)

**현재** (line 191):
```python
query_emb = torch.zeros(B, self.query_encoder.d_model, device=device)
```

**변경**:
```python
queries = batch.get("questions", None)  # list[str] of length B
if queries is not None and any(q for q in queries):
    query_emb = self.query_encoder(queries, device)   # (B, d_query)
else:
    # Fallback (e.g., Stage 1 clips without QA)
    query_emb = torch.zeros(B, self.query_encoder.d_model, device=device)
```

### 2.2 데이터 파이프라인 수정
**파일**: [`TrajGaze_v2/data/dataset_temporal.py`](../TrajGaze_v2/data/dataset_temporal.py)

현재 `StreamGazeStage1DatasetTemporal`은 video clip만 반환, QA 안 보임. 두 가지 옵션:

**옵션 A1 (간단)**: Stage 1 데이터셋에 dummy question 추가 — 같은 clip의 첫 번째 question 또는 task name을 넣음. encoder가 *어떤* sample-specific 정보라도 받음.

**옵션 A2 (정확)**: StreamGaze QA를 Stage 1에도 활용 — clip에 연결된 question을 모두 사용해 augment. 더 큰 effective dataset.

→ Sprint 1 첫 시도는 **옵션 A1** (가장 적은 변경으로 검증).

수정 위치 (line ~138):
```python
return {
    "past":            past,
    "future":          future,
    "I_scores_past":   I_scores_t[:T_past],
    "I_scores_future": I_scores_t[T_past:],
    "T_past":          T_past,
    "T_future":        T - T_past,
    "frame_paths":     sampled,
    # NEW: provide a placeholder question for query encoder
    "question":        f"What action is happening involving {stem}?",
}
```

`collate_stage1_temporal` 에도 `questions` 키 추가.

### 2.3 Stage 2 학습은?
[`TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py:125`](../TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py) 의 `get_patch_scores_temporal` 가 이미 `queries = [item["question"]]` 을 호출하지만, Stage 1 인코더는 query_emb=zero로 학습되었으므로 효과 없었음. Stage 1을 재학습하면 자동으로 사용됨.

---

## 3. 개입 B — Sharp spatial supervision

### 3.1 변경 대상
**파일**: [`TrajGaze_v2/data/interaction.py`](../TrajGaze_v2/data/interaction.py)

**현재** (line 21–22):
```python
SIGMA_GAZE = 16.0    # pixels (224-frame, ~1 patch width)
SIGMA_HAND = 24.0    # pixels (~1.5 patch widths)
```

**변경**:
```python
SIGMA_GAZE = 6.0     # ~0.4 patch width (sharp)
SIGMA_HAND = 8.0     # ~0.5 patch width (sharp)
```

이렇게 하면 `I_scores_past[t]` 에서 gaze patch에 ~5x 더 집중된 mass. encoder가 위치 정확도를 학습할 수밖에 없음.

### 3.2 추가 변경 (optional, 강한 강제)
[`TrajGaze_v2/training/stage1_temporal.py`](../TrajGaze_v2/training/stage1_temporal.py) 의 loss schedule (`loss_schedule.py`) 에서 `score_past` weight 를 키움.

현재 schedule을 확인하지 않았지만, 일반적으로 1.0 가중치 → **2.0–3.0** 으로 올리면 encoder가 spatial localization에 더 강하게 fit됨.

### 3.3 빠른 sanity check
변경 후 `I_scores_past` 의 entropy 가 줄었는지 확인:
```bash
python -c "
import sys, numpy as np
sys.path.insert(0, '/workspace/trajgaze')
from TrajGaze_v2.data.dataset_temporal import StreamGazeStage1DatasetTemporal
ds = StreamGazeStage1DatasetTemporal(n_frames=64)
item = ds[0]
s = item['I_scores_past'].numpy()  # (T, 196)
# Per-frame max / mean ratio
ratios = s.max(-1) / (s.mean(-1) + 1e-6)
print('Per-frame max/mean ratio (higher = sharper):', ratios.mean(), ratios.std())
print('Per-frame entropy (lower = sharper):', -(s / (s.sum(-1, keepdims=True) + 1e-6) * np.log(s / (s.sum(-1, keepdims=True) + 1e-6) + 1e-10)).sum(-1).mean())
"
```
변경 전 ratio ~10, 변경 후 ~30 정도 예상.

---

## 4. 학습 + 평가 (실행 순서)

### Step 1: Stage 1 재학습 (~1시간 on 단일 H200)
```bash
# 변경된 supervision + query embedding 활성화 후
cd /workspace/trajgaze
torchrun --nproc_per_node=1 -m TrajGaze_v2.training.stage1_temporal \
  --epochs 100 \
  --use-patch-temporal-branch \
  --freeze-gate \
  --batch-size 2 \
  --workers 4 \
  --output-dir /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_sprint1_AB \
  > /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_sprint1_AB.log 2>&1
```

**중간 검증**: epoch 10, 30, 50 ckpt 만 만들어지면 cheap probe:
```bash
# 위 학습 log에서 정확한 loss 항목 확인
grep "loss" /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_sprint1_AB.log | tail -20
```

**중간 진단 (optional, epoch 50쯤)**: `eval_stage1_holdout.py` 로 새 ckpt val loss 측정 → 기존 supervision 변경이 학습을 망치진 않았는지 확인.

### Step 2: Stage 2 LoRA + merge 학습 (~수 시간)
```bash
cd /workspace/trajgaze
CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
  --model-type full \
  --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_sprint1_AB/best.pth \
  --output-dir  /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_sprint1_AB_keep10 \
  --epochs 3 --merge-ratio 0.9 --grad-accum 4
```

### Step 3: 진단 재실행 (~30분)
```bash
ROOT=/workspace/trajgaze
S1_NEW=$ROOT/TrajGaze_v2/checkpoints/E1_sprint1_AB/best.pth
LORA_NEW=$ROOT/TrajGazeMerge/checkpoints/E1_sprint1_AB_keep10/best.pth
PY=/opt/conda/envs/gaze/bin/python
TAG=E1_sprint1_AB

cd $ROOT
CUDA_VISIBLE_DEVICES=0 $PY -m TrajGazeMerge.eval.diagnostic_eval \
  --stage1-ckpt $S1_NEW --lora-ckpt $LORA_NEW --tag ${TAG}_diag

$PY -m TrajGazeMerge.eval.analyze_diagnostics --tag ${TAG}_diag

CUDA_VISIBLE_DEVICES=0 $PY -m TrajGazeMerge.eval.counterfactual_mask_eval \
  --stage1-ckpt $S1_NEW --lora-ckpt $LORA_NEW --tag ${TAG}_mask

$PY -m TrajGazeMerge.eval.analyze_M1_correlations  # uses E1_keep10_diag_v2 by default;
# may need a `--tag` arg or to point at the new parquet
```

### Step 4: 4-지표 verdict 확인
```bash
$PY -c "
import json
s = json.load(open('/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic/${TAG}_diag_summary.json'))
g = s['global_means']
print(f'acc            = {s[\"overall_accuracy\"]:.2f}%   (target ≥ 67)')
print(f'gt_gaze_recall = {g[\"gt_gaze_recall\"]:.3f}   (target > 0.20)')
print(f'late_half      = {g[\"late_half_ratio\"]:.3f}   (target ≈ 0.5)')
"
$PY -c "
import json
s = json.load(open('/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic/${TAG}_mask_mask_summary.json'))
b = s['variants']['baseline']['overall_accuracy']
sh = s['variants']['shuffle_kept']['overall_accuracy']
print(f'shuffle_kept Δ = {sh-b:+.2f}pp   (target < -3)')
"
```

---

## 5. Decision Tree (Sprint 1 결과 기반)

```
4-지표 status (gt_gaze_recall, shuffle Δ, frame_index corr, acc):
│
├── 4/4 success
│   → narrative 복원 ✓
│   → 다음: cross-dataset (EgoMCQ) 검증 + paper rewrite
│
├── 3/4 success (gt_gaze, shuffle, frame_corr ✓ but acc < 67)
│   → encoder는 의도대로 작동하지만 Qwen-VL 처리 한계
│   → 다음:
│     (a) 개입 C (anti-bag shuffle augmentation) 로 LLM이 spatial 사용하게 강제
│     (b) 그래도 안 되면 개입 F (다른 VLM) — Qwen 한계 확정
│
├── 2/4 success (gt_gaze, shuffle ✓ but corr 여전히 높고 acc 낮음)
│   → 부분 개선. temporal bias도 여전히 strong.
│   → 다음:
│     개입 D (temporal-spatial 분리 아키텍처) 로 더 명확하게 분리
│
├── 1/4 success (gt_gaze 증가만)
│   → spatial은 fit됐지만 method 통합이 깨짐
│   → 다음:
│     loss balance 재조정 후 재학습 또는 narrative 옵션 A (정직 보수)
│
└── 0/4 success
    → encoder 설계로 해결 불가
    → 다음: §11.5 narrative 옵션 B (pivot) 으로 가거나 method 자체 재고
```

---

## 6. EgoMCQ 인프라 준비 (병렬, CPU only)

Sprint 1 학습 진행되는 동안 진행 가능. Sprint 1 verdict 받은 후 평가 launch.

### 6.1 Dataset 확인
EgoMCQ는 Ego4D에 포함 — `https://ego4d-data.org/docs/data/annotations-schemas/` 의 `mcq` 파일.
이미 `/workspace/datasets/Ego4D/` 가 있다면 경로만 추가. 없으면 다운로드 필요 (수십 GB).

### 6.2 Dataset wrapper 작성
`TrajGazeMerge/data/egomcq_dataset.py` (신규) — 인터페이스를 StreamGazeMergeDataset와 동일하게 맞춤:
```python
def __getitem__(self, idx) -> dict:
    return {
        "vlm_frame_paths":  [...],
        "traj_frame_paths": [...],
        "traj":             {"gaze_pos": ..., "gaze_mask": ..., "left_pos": ..., ...},
        "question":         ...,
        "options":          ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer":           "A",
        "task":             "egomcq",
        "dataset":          "ego4d",
    }
```

EgoMCQ에 gaze가 없으므로:
- **옵션 1**: gaze=zero, gaze_mask=False (모두 invalid). 모델은 hand만 사용.
- **옵션 2**: hand-only Stage 1 ckpt 사용 (paper Table 2 OnlyHand variant). 본 repo에 학습된 게 없으면 옵션 1로.

→ **추천: 옵션 1** + `train_merge_lora_temporal_no_kd --model-type full` 그대로 사용 (gaze 입력은 zero라도 정상 forward).

### 6.3 평가
같은 `diagnostic_eval.py` / `ablation_score_source.py` 를 dataset arg만 바꿔서 실행 가능 → 코드 재사용 거의 100%.

---

## 7. 새 세션 체크리스트

다른 세션에서 이어가려면:

- [ ] 본 문서 + `docs/diagnostic_report_E1_keep10.md` §12–§13 읽기
- [ ] `git log --oneline -10` 으로 최신 커밋 확인 (현재 `bd64d55`)
- [ ] 코드 변경 시작 위치:
  - `TrajGaze_v2/models/model_temporal.py:191` (query_emb)
  - `TrajGaze_v2/data/dataset_temporal.py:138-145` (question 추가)
  - `TrajGaze_v2/data/dataset_temporal.py:149-200` (collate_stage1_temporal 에 questions)
  - `TrajGaze_v2/data/interaction.py:21-22` (SIGMA_GAZE, SIGMA_HAND)
- [ ] §3.3의 sanity check 로 supervision 변화 확인
- [ ] Step 1 (Stage 1 학습) 후 epoch 50쯤 중간 holdout eval
- [ ] Step 3 (진단 재실행) → 4-지표 verdict
- [ ] Decision tree에 따라 후속 결정
- [ ] 결과를 `docs/diagnostic_report_E1_keep10.md` §14 (신설) 에 추가

---

## 8. 기존 진단 인프라 (모두 작동 확인됨)

새 ckpt에 모두 그대로 적용 가능 (tag만 변경):

| 진단 | 명령 | 시간 |
|---|---|---:|
| Per-sample diagnostic | `python -m TrajGazeMerge.eval.diagnostic_eval --tag <T>` | ~12분 |
| Aggregate analysis | `python -m TrajGazeMerge.eval.analyze_diagnostics --tag <T>` | ~1분 (CPU) |
| Counterfactual ablation (7 sources) | `python -m TrajGazeMerge.eval.ablation_score_source --tag <T>` | ~3.5h |
| Counterfactual masking | `python -m TrajGazeMerge.eval.counterfactual_mask_eval --tag <T>` | ~1h |
| Option permutation | `python -m TrajGazeMerge.eval.option_permutation_eval --tag <T>` | ~2h |
| Stage 1 holdout val | `python -m TrajGaze_v2.training.eval_stage1_holdout --ckpt-dir <D> --tag <T>` | ~30min |
| Score-trajectory correlation | `python -m TrajGazeMerge.eval.analyze_M1_correlations` | ~5분 (CPU) |
| Gaze-required subset analysis | `python -m TrajGazeMerge.eval.analyze_gaze_required_subset` | ~1분 (CPU) |
| Frozen-method eval | `python -m TrajGazeMerge.eval.frozen_method_eval --tag <T>` | ~25분 |

`<T>` = 새 tag (예: `E1_sprint1_AB`).

---

## 9. 위험 및 대안

- **Sharp supervision 으로 학습이 underfit**: `SIGMA_GAZE` 를 6 → 10 으로 약간 완화, 또는 `score_past` weight 를 2.0 → 1.5 로 낮춰 재학습.
- **Query embedding으로 학습이 느려짐**: QueryEncoder가 무거울 수 있음. 본 sprint 단순화: question을 한 번만 임베딩해서 epoch 내 캐시.
- **Stage 1 retrain이 너무 비싸짐**: epoch 100 대신 50으로 시작, 결과 보고 늘림.
- **새 데이터셋 (EgoMCQ) 준비가 길어짐**: 우선 옵션 1 (gaze=zero) 로 빠르게 시도, 안 되면 hand-only Stage 1 학습.

---

## 10. 빠른 참조 — 핵심 진단 수치 (현재 baseline)

| Metric | 현재 값 |
|---|---:|
| Overall acc (526 EGTEA) | 68.44% (67.68% in our run) |
| gt_gaze_recall | 0.077 |
| late_half_ratio | 0.83 |
| temporal_CoM | 0.78 |
| shuffle_kept Δ | +0.95pp (no penalty) |
| mask_kept Δ | −12.93pp |
| mask_kept_early Δ | −0.95pp |
| frame_index correlation | 0.498 |
| Frozen-method Δ | −0.19pp (LoRA absorbs) |
| Open-ended acc | 34.46% |
| Consistent-correct (permutation) | 43.73% |
| ECE | 0.144 |

Sprint 1 후 모든 지표 재측정하여 비교 가능.
