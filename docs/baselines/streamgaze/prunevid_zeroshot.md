# PruneVid × StreamGaze_v2 EGTEA — Zero-shot (Pruning ON)

**구현 완료. 실행은 [pllava_zeroshot.md](./pllava_zeroshot.md) 의 vanilla baseline 이 안정화된 뒤 진행 예정.** 동일한 `StreamGazeEgteaDataset` + `pllava_eval_streamgaze.py` 를 재사용하고, PruneVid 의 **visual token pruning** 을 활성화한 하이퍼파라미터로 실행하는 variant 다.

```
 ... (vanilla 와 입력 파이프라인 동일)                        ← [pllava_zeroshot.md §1-§4]
     │
     ▼
 PllavaForConditionalGeneration (PLLaVA-7B, use_lora=True, lora_alpha=14)
   - vision_tower + multi_modal_projector (동일)
   - >>>>> PruneVid pruning ACTIVE <<<<<
       selected_layer = 10           (Llama layer 10 의 attention 에서 토큰 중요도 계산)
       alpha          = 0.4          (pruning 강도)
       tau            = 0.8          (dynamic/static 분리 유사도 threshold)
       temporal_segment_ratio = 0.25 (64 시간 토큰 → 16 세그먼트로 DPC-KNN 클러스터)
       cluster_ratio          = 0.5  (공간 토큰 50% 만 유지)
     → ~16-17% 의 원 토큰만 LLM 으로 전달, FLOP 0.20-0.23×
     │
     ▼
 generate → answer 파싱 → check_ans  (동일)
```

---

## 1. Vanilla 대비 차이점 요약

| 항목 | Vanilla (pllava_zeroshot) | PruneVid ON |
|---|---|---|
| 코드 변경 | — | **없음** (hyperparam 만 변경) |
| `tau`                    | 1.0 | **0.8** |
| `temporal_segment_ratio` | 1.0 | **0.25** |
| `cluster_ratio`          | 1.0 | **0.5** |
| `selected_layer`         | 10 (no-op) | 10 (active) |
| `alpha`                  | 0.1 (no-op) | **0.4** |
| `lora_alpha`             | 4 (PLLaVA official) | **14** (PruneVid `scripts/eval.sh` default) |
| LLM 입력 토큰 / video | 2,304 | **~380-400** (16-17%) |
| FLOP                    | 1.0× | **0.20-0.23×** |

하이퍼파라미터는 PruneVid 공식 [scripts/eval.sh](/home/yujin/gaze/PruneVid/scripts/eval.sh) (MVBench/VideoMME/EgoSchema/VCGBench 공통 sweep 의 대표값) 을 그대로 차용.

---

## 2. Pruning 메커니즘 상세

PruneVid 는 PLLaVA-7B 의 frame embedding (vision tower 출력 → projector 후 `[B, T, H*W, D]`) 에 두 단계 pruning 을 적용. 구현은 `models/pllava/modeling_pllava.py` 안에 상주하며, hyperparam 이 기본값이면 no-op.

### 2.1 Temporal segment clustering — DPC-KNN

[models/pllava/modeling_pllava.py:111](/home/yujin/gaze/PruneVid/models/pllava/modeling_pllava.py#L111) `cluster_dpc_knn`:
- **입력**: temporal token sequence `[T, D]` (T frame 의 temporal aggregation)
- **출력**: `K = max(1, round(T * temporal_segment_ratio))` 개 segment
- **방법**: Density-Peak Clustering with KNN (Rodriguez & Laio, 2014)
  - 각 토큰의 local density = K-nearest neighbor 평균 거리의 역수
  - "density peak" 토큰이 cluster center — 다른 토큰은 가장 가까운 density-higher 토큰에 assign
- `temporal_segment_ratio=0.25` → 64 frame 이면 16 segment, 16 frame 이면 4 segment.

### 2.2 Static / Dynamic split + spatial cluster merge — `merge_frames_dynamic`

[models/pllava/modeling_pllava.py:764](/home/yujin/gaze/PruneVid/models/pllava/modeling_pllava.py#L764) `merge_frames_dynamic`:
- 각 segment 내에서 **static vs dynamic 토큰 분리**:
  - `cosine_sim(patch_t, patch_{t-1}) > tau` → static (배경/변화 없는 영역)
  - 그 외 → dynamic (움직이는 영역)
- **Static**: segment 전체에서 한 번만 저장 → 대폭 압축
- **Dynamic**: `spatial_merge_tokens` 로 `N' = round(N * cluster_ratio)` 개 클러스터로 merge (cosine-sim 기준 cluster + weighted sum)
- 출력 token 수 ≈ (dynamic_tokens × cluster_ratio) + static_tokens × 1 segment

### 2.3 Attention-weighted importance — `selected_layer`, `alpha`

Llama `selected_layer` 의 attention score 를 훅으로 추출, `alpha` 로 가중치를 주어 어느 토큰을 유지할지 결정. `alpha=0.4` 는 40% 를 attention-driven, 60% 를 cosine-sim driven 으로 혼합.

### 2.4 Token budget

논문 Table 3 기준 `tau=0.8, temporal_segment_ratio=0.25, cluster_ratio=0.5` 에서 MVBench 토큰 유지율 ≈ 17%, FLOP 0.22×. 구체 수는 비디오 content (static/dynamic 비율) 에 따라 다름 — eval 시 첫 sample 에서 `[token-probe]` 로깅 추가 권장.

---

## 3. Eval CLI

### 3.1 Launcher — `scripts/eval_streamgaze_prunevid.sh`

```bash
cd /home/yujin/gaze/PruneVid
bash scripts/eval_streamgaze_prunevid.sh
```

내부 명령:
```bash
python -m tasks.eval.streamgaze.pllava_eval_streamgaze \
  --pretrained_model_name_or_path MODELS/pllava-7b \
  --save_path test_results/streamgaze_egtea/prunevid_16f_tau0.8_seg0.25_clu0.5 \
  --num_frames 16 \
  --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
  --pooling_shape 16-12-12 \
  --conv_mode eval_mvbench \
  --selected_layer 10 --alpha 0.4 \
  --tau 0.8 --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
  --top_p 1.0 --temperature 1.0
```

환경변수 오버라이드 지원: `MODEL_DIR`, `LORA_ALPHA`, `POOLING_SHAPE`, `NUM_FRAMES`, `SELECTED_LAYER`, `ALPHA`, `TAU`, `TEMPORAL_SEGMENT_RATIO`, `CLUSTER_RATIO`, `SAVE_ROOT`.

### 3.2 Ablation 스윕 (선택)

[scripts/eval.sh](/home/yujin/gaze/PruneVid/scripts/eval.sh) 스타일로 hyperparam sweep:

```bash
for tau in 0.7 0.8 0.9; do
  for seg_ratio in 0.125 0.25 0.5; do
    for clu_ratio in 0.25 0.5; do
      TAU=$tau TEMPORAL_SEGMENT_RATIO=$seg_ratio CLUSTER_RATIO=$clu_ratio \
        bash scripts/eval_streamgaze_prunevid.sh
    done
  done
done
```

각 run 은 `SAVE_DIR` 이름에 hyperparam 이 인코드되어 결과 덮어쓰기 없음.

---

## 4. Vanilla 대비 비교 (예정)

`aggregate_results.py` 확장으로 vanilla + PruneVid 컬럼을 한 번에 비교:

```bash
python -m tasks.eval.streamgaze.aggregate_results \
  --run vanilla_16f=test_results/streamgaze_egtea/vanilla_16f \
  --run prunevid_16f=test_results/streamgaze_egtea/prunevid_16f_tau0.8_seg0.25_clu0.5
```

### 4.1 예상 비교 표 (실행 후 업데이트)

| Task | N | Vanilla-16f | PruneVid-16f | Δ (pp) |
|---|---:|---:|---:|---:|
| past_gaze_sequence_matching | 64 | 39.06 | — | — |
| past_non_fixated_object_identification | 68 | 27.94 | — | — |
| past_object_transition_prediction | 2 | 50.00 | — | — |
| past_scene_recall | 37 | 8.11 | — | — |
| present_object_attribute_recognition | 96 | 59.38 | — | — |
| present_object_identification_easy | 101 | 75.25 | — | — |
| present_object_identification_hard | 64 | 70.31 | — | — |
| present_future_action_prediction_egtea | 94 | 29.79 | — | — |
| **Overall (micro)** | **526** | **48.29** | — | — |

**기대치** (PruneVid 논문 Table 3 기준):
- Overall accuracy: vanilla 대비 ±0.5 pp 이내로 유지되어야 정상 (token pruning 이 기능 보존)
- 만약 Δ 가 크게 음수 (< -3 pp) 면 hyperparam 이 StreamGaze 의 fine-grained gaze/object 태스크에 부적합 — sweep 필요

### 4.2 Token / FLOP 비교 (예정)

| Metric | Vanilla-16f | PruneVid-16f |
|---|---|---|
| video token count per sample | 2,304 | ~400 (17%) |
| vision tower FLOP          | 1.0× | 1.0× (동일) |
| LLM forward FLOP           | 1.0× | ~0.22× |
| end-to-end latency / QA    | TBD | TBD |

---

## 5. 파일 레이아웃

`pllava_zeroshot.md` 와 동일 — 단, 추가로:

```
PruneVid/
  scripts/
    eval_streamgaze_prunevid.sh         ← 본 문서의 launcher
  test_results/streamgaze_egtea/
    prunevid_16f_tau0.8_seg0.25_clu0.5/  ← 본 문서의 결과
      all_results.json
      upload_leaderboard.json
```

---

## 6. Key decisions

- **코드 수정 無**: vanilla 와 완전히 같은 `StreamGazeEgteaDataset` + eval script 사용. Pruning 은 `PllavaConfig` 의 hyperparam 으로 on/off 되므로 CLI 만 다름 → A/B 비교가 깨끗.
- **`lora_alpha=14`** (PruneVid eval.sh default): PLLaVA 공식값 4 보다 LoRA 영향이 크지만, `scripts/eval.sh` 가 MVBench/VideoMME 등 모든 벤치마크에서 14 를 쓰므로 논문 수치 비교 시 동일 baseline 을 유지. Vanilla 문서에서는 PLLaVA 공식 (4) 로 측정하므로 lora_alpha 차이도 평가에 포함.
- **Hyperparam 단일 세팅**: `tau=0.8, seg=0.25, clu=0.5` 는 PruneVid 논문/`scripts/eval.sh` 의 "reference" 값. Frame sweep (16/32/64) 은 vanilla 에서 이미 다루므로 본 run 은 16f 로 고정 — 필요 시 §3.2 의 스윕 스크립트로 확장.
- **미래 leak 방지 규칙 유지**: `present_future_action_prediction_egtea` 에 대해 `[0, resp_start]` window — vanilla 와 동일 정책.

---

## 7. Reproduce quick reference

```bash
cd /home/yujin/gaze/PruneVid

# vanilla baseline 완료 전제 (pllava_zeroshot.md §9 참고)

# PruneVid eval
bash scripts/eval_streamgaze_prunevid.sh

# Vanilla vs PruneVid 비교
python -m tasks.eval.streamgaze.aggregate_results \
  --run vanilla_16f=test_results/streamgaze_egtea/vanilla_16f \
  --run prunevid_16f=test_results/streamgaze_egtea/prunevid_16f_tau0.8_seg0.25_clu0.5
```

---

## 8. References

- PruneVid 논문: [arXiv:2503.01023](https://arxiv.org/abs/2503.01023) (Huang et al., 2025) — training-free visual token pruning
- PLLaVA: [arXiv:2404.16994](https://arxiv.org/abs/2404.16994)
- 관련 문서: [pllava_zeroshot.md](./pllava_zeroshot.md) — vanilla baseline (본 문서의 비교 기준)
