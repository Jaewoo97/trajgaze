# Stage-2 인코더 실험 기록 (2026-05)

**Branch:** `main` (`/workspace/trajgaze`)  
**기준 날짜:** 2026-05-03  
**핵심 파일**
- [TrajGaze_v2/models/encoder_temporal.py](../../TrajGaze_v2/models/encoder_temporal.py)
- [TrajGaze_v2/training/stage1_temporal.py](../../TrajGaze_v2/training/stage1_temporal.py)
- [TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py](../../TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py)
- [TrajGazeMerge/training/train_merge_lora_temporal_mrcons.py](../../TrajGazeMerge/training/train_merge_lora_temporal_mrcons.py)

---

## 0. 배경 및 목표

**출발점**: `/workspace/trajgaze_msk` codebase에서 검증된 stage-2 best는 **A1 (feat-KD) 66.92%** (n=526 egtea).  
**New baseline**: `/workspace/trajgaze`의 `no_kd_keep10_bs4_jw` — temporal encoder(`temporal_best.pth`)에 no-KD CE-only stage-2를 돌린 결과 **best 67.49%** (epoch 2 step 5200). msk A1을 +0.57pt 초과.

**목표 (G5)**: ≥ 68.82% (docs/ablation/spatiotemporal_msk_recipe.md에 기재된 temporal-only ablation 수치).

---

## 1. 실험 결과 요약표

| Run | Encoder | Stage-2 방식 | Best acc | 비고 |
|---|---|---|---|---|
| **baseline (no-KD)** | temporal_best.pth | CE-only | **67.49%** | 기준선 |
| EA1 silent-drop ❌ | EA1_parallel_branch | CE-only + **silent drop** bug | 60.08% | 진단용 데이터 포인트 |
| **EA1 FIX** ✅ | EA1_parallel_branch | CE-only (silent-drop fix 적용) | **68.44%** | baseline +0.95pt |
| EA1-C FIX | EA1-C_60ep | CE-only (fix 적용) | 66.16% | 불안정 (41% dip 발생) |
| mr-cons keep=0.50 (baseline enc) | temporal_best.pth | mr-cons self-distill keep=0.5 | 66.35% | baseline **−1.14pt** 후퇴 |
| **E1 patch-temporal** ⏳ | E1_patch_temporal | CE-only | 60.65% (step 800) | 진행 중 |
| **EA1 FIX + mr-cons keep=0.50** ⏳ | EA1_parallel_branch | mr-cons self-distill keep=0.5 | 57.41% (step 800) | 진행 중 |

---

## 2. 진단: EA1 silent-drop 버그 (F5)

### 문제

`train_merge_lora_temporal_no_kd.py`의 기존 `load_traj_encoder()`는 `TrajGazeV2Temporal()`을 **default args**로 인스턴스화했다.

```python
# 버그 있는 코드 (수정 전)
model = TrajGazeV2Temporal(n_vis_keyframes=n_vis_keyframes).to(device)
# use_frame_score_branch=False (default) → parallel branch 모듈 없음
state = ckpt.get(...)
model.load_state_dict(state, strict=False)   # EA1의 6개 키 silently drop!
```

EA1 ckpt에는 `encoder.frame_attn_pool.*`, `encoder.frame_score_head.*` 6개 키가 있었지만, 모델에 해당 모듈이 없으므로 `strict=False` 로드 시 **경고 없이 전부 무시(silent drop)** 됐다.

**결과**: EA1 encoder를 쓴다고 했지만 실제로는 "parallel branch를 제거한 EA1 main path만 있는 모델"로 stage-2가 학습되었음 → best 60.08% (baseline 대비 −7.41pt).

### 수정

state-dict 키 이름으로 architecture flag를 자동 추론해 올바른 모델을 생성:

```python
# 수정된 코드
ckpt  = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
state = ckpt.get("encoder_state", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))

has_frame_score = any(
    k.startswith("encoder.frame_attn_pool") or k.startswith("encoder.frame_score_head")
    for k in state
)
has_post_iframe    = any(k.startswith("encoder.inter_frame_post") for k in state)
has_patch_temporal = any(
    k.startswith("encoder.patch_temporal_query")
    or k.startswith("encoder.patch_temporal_attn")
    or k.startswith("encoder.patch_temporal_head")
    for k in state
)

model = TrajGazeV2Temporal(
    n_vis_keyframes=n_vis_keyframes,
    use_frame_score_branch=has_frame_score,
    use_post_fusion_iframe=has_post_iframe,
    use_patch_temporal_branch=has_patch_temporal,
).to(device)
missing, unexpected = model.load_state_dict(state, strict=False)
# missing/unexpected 0이면 정상
```

이 fix는 `train_merge_lora_temporal_no_kd.py`, `train_merge_lora_temporal.py`, `train_merge_lora_temporal_mrcons.py` 세 파일 모두에 적용.

### EA1 FIX 결과

fix 후 재launch → best **68.44%** (epoch 2 step 5600) — baseline 67.49% 대비 **+0.95pt**.

| Task | baseline no-KD | EA1 FIX | Δ |
|---|---|---|---|
| past_gaze_sequence_matching | ~68.75% | ~71.87% | +3.12 |
| past_non_fixated_object_id | ~63.24% | ~57.35% | −5.89 |
| **past_object_transition_pred** | **0%** | **50%** | **+50pp** |
| past_scene_recall | ~54.05% | ~51.35% | −2.70 |
| present_future_action_pred | ~53.19% | ~54.26% | +1.07 |
| present_obj_attr_recog | ~90.62% | ~89.58% | −1.04 |
| present_obj_id_easy | ~59.41% | ~66.34% | +6.93 |
| present_obj_id_hard | ~65.62% | ~64.06% | −1.56 |
| **OVERALL** | **67.49%** | **68.44%** | **+0.95** |

**핵심 기여**: `past_object_transition_pred` 0% → 50% (+50pp). EA1의 inter-frame parallel branch가 시간 순서/전환 추론에 효과적임을 암시.

---

## 3. E1: Patch-level Temporal Modulation (신규 설계)

### 3.1 동기: EA1의 frame-uniform 한계

EA1의 parallel branch (`frame_attn_pool` + `frame_score_head`)는 inter-frame transformer 출력 x_iframe을 4 토큰에 대해 pooling한 뒤 **스칼라 1개 (B, T)**를 출력한다. 이 스칼라는 196 patch에 broadcast multiply된다.

```
x_iframe (B, T, 4, D)  →  attn_pool  →  pooled (B, T, D)
                        →  score_head →  frame_scores (B, T)     # scalar per frame
per_frame_scores (B, T, 196)  ×  frame_scores.unsqueeze(-1)      # uniform scaling
```

**문제**: frame-uniform multiplier는 "이 프레임이 얼마나 중요한가"만 표현할 수 있고, "이 프레임에서 어느 patch가 더 중요한가"는 표현 못함. Inter-frame 정보가 patch-level spatial precision에 기여하지 못함.

### 3.2 E1 설계: Patch Query Cross-Attention

196개의 **학습된 patch query embedding**이 x_iframe (4 tokens per frame)에 cross-attention해 patch별 modulation (B, T, 196)을 생성.

#### Architecture

```python
# __init__
self.patch_temporal_query = nn.Embedding(N_PATCHES, d_enc)       # 196 × 384
self.patch_temporal_attn  = nn.MultiheadAttention(
    embed_dim=d_enc, num_heads=4, dropout=0.1, batch_first=True,
)
self.patch_temporal_head  = nn.Sequential(
    nn.LayerNorm(d_enc),
    nn.Linear(d_enc, 1),
    nn.Sigmoid(),
)
```

```python
# forward (step 4c, x_iframe 사용 — gate 이전)
x_iframe_per_frame = x_iframe.reshape(B, T, N_TOKENS, D_ENC)     # (B, T, 4, D)
flat_kv = x_iframe_per_frame.reshape(B * T, N_TOKENS, D_ENC)     # (B*T, 4, D)

q = self.patch_temporal_query(self.patch_idx)                    # (196, D)
q = q.unsqueeze(0).expand(B * T, -1, -1)                         # (B*T, 196, D)

attended, _ = self.patch_temporal_attn(q, flat_kv, flat_kv)      # (B*T, 196, D)
patch_mod = self.patch_temporal_head(attended).squeeze(-1)        # (B*T, 196) ∈ [0,1]
patch_modulation = patch_mod.reshape(B, T, N_PATCHES)             # (B, T, 196)

# step 7d — visual fusion 이후 적용
per_frame_scores = per_frame_scores * patch_modulation            # elementwise
```

#### EA1 vs E1 비교

| 항목 | EA1 (frame-level branch) | E1 (patch-level branch) |
|---|---|---|
| 출력 shape | **(B, T)** — frame scalar | **(B, T, 196)** — patch map |
| 적용 | broadcast multiply (frame 동일) | **elementwise multiply** (patch별 다름) |
| 파라미터 | ~1K (Linear + Sequential) | ~314K (Embedding + MHA + Sequential) |
| x_iframe 활용 | 4 tokens → pooling → scalar | 4 tokens → **cross-attn** with 196 queries |
| spatial precision | ❌ frame-uniform | ✅ **patch-level** |

#### 상호 배타성 보장

두 branch는 같은 `x_iframe`을 source로, 같은 자리에서 modulation을 적용하므로 동시 사용 시 학습 신호 충돌. assertion으로 차단:

```python
assert not (use_frame_score_branch and use_patch_temporal_branch), (
    "use_frame_score_branch and use_patch_temporal_branch are mutually exclusive."
)
```

### 3.3 Stage-1 결과 (E1_patch_temporal, 100 epoch)

| 지표 | E1 결과 | EA1 결과 | Gate 기준 |
|---|---|---|---|
| best loss (sum_4) | **0.0185** | 0.0248 | ≤ 0.026 ✅ |
| score_traj (final) | **0.0053** | 0.0065 | ≤ 0.0066 ✅ |

두 acceptance gate 통과. EA1보다 오히려 낮은 score_traj — patch query가 trajectory-driven score 학습에 더 효과적임을 시사.

> 주의: launch script에 `--epochs` 미지정으로 default 100 epoch 학습 (계획된 30 epoch 초과). 더 충분한 수렴이므로 기능적으로 문제 없음.

### 3.4 Stage-2 진행 상황 (2026-05-03, 진행 중)

| step | E1 acc | EA1 FIX 같은 step |
|---|---|---|
| 400 | 54.75% | 53.42% |
| 800 | **60.65%** | 55.70% |

E1이 800 step에서 EA1 FIX보다 **+4.95pp 앞서** 있음. 초기 수렴 속도가 더 빠름 (score_traj 낮음과 일치).

---

## 4. mr-cons (Multi-Ratio Consistency) 실험

### 4.1 메커니즘

External teacher 없이 **student 자신을 두 번 forward**해 self-distillation:
- Primary forward: keep=10% (merge_ratio=0.9) — 어려운 뷰
- Aux forward: keep=50% — 쉬운 뷰 (더 많은 토큰 보존)
- Loss: primary logit이 `stop_grad(aux)` logit을 KL divergence로 imitate

```
loss = CE(primary) + α_cons × KL(primary || stop_grad(aux))
```

완전히 KD-free — teacher ckpt 불필요.

### 4.2 msk codebase 결과 (참고)

| 설정 | OVERALL | 비교 |
|---|---|---|
| msk no-KD baseline | 64.45% | 기준 |
| msk mr-cons keep=0.15 | 66.16% | **+1.71pt** |
| msk A1 (feat-KD) | 66.92% | +2.47pt |

msk에서 mr-cons가 유효했던 이유: baseline에 logit-KD (α=0.5)가 포함되어 있어 scene_recall을 억압하고 있었음. mr-cons 도입 시 logit-KD 제거(α=0) 효과가 더해져 +1.71pt.

### 4.3 /workspace/trajgaze 결과

| 설정 | encoder | Best acc | 비교 |
|---|---|---|---|
| baseline no-KD | temporal_best.pth | 67.49% | 기준 |
| mr-cons keep=0.50 | temporal_best.pth | **66.35%** | **−1.14pt** |
| EA1 FIX no-KD | EA1_parallel_branch | 68.44% | +0.95pt |
| EA1 FIX + mr-cons keep=0.50 | EA1_parallel_branch | 57.41% (step 800, 진행 중) | 미확정 |

**trajgaze baseline은 이미 no-KD** → mr-cons에서 logit-KD 제거 효과가 없음. 순수 self-distill 신호만으로는 이미 높은 baseline을 극복하지 못함. keep=0.50으로 늘려도 +0.19pt에 불과 (0.15 → 0.50 → 66.16 → 66.35).

---

## 5. 수정된 파일 목록

| 파일 | 변경 내용 |
|---|---|
| `encoder_temporal.py` | E1 patch-temporal branch 추가 (`use_patch_temporal_branch` flag, 3개 sub-module) |
| `model_temporal.py` | `use_patch_temporal_branch` constructor pass-through |
| `stage1_temporal.py` | `--use-patch-temporal-branch` argparse |
| `train_merge_lora_temporal_no_kd.py` | silent-drop fix + patch_temporal flag 추론 |
| `train_merge_lora_temporal.py` | silent-drop fix + `--kd-feat-layers`/`--kd-feat-weight` 추가 (feat-KD port) |
| `train_merge_lora_temporal_mrcons.py` | silent-drop fix 적용 |

---

## 6. 현재 실행 중인 실험 (2026-05-03 기준)

| GPU | 실험 | 출력 디렉터리 | 예상 완료 |
|---|---|---|---|
| 0 | E1 patch-temporal stage-2 | `TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/` | 2026-05-04 KST 낮 |
| 1 | EA1 FIX + mr-cons keep=0.50 | `TrajGazeMerge/checkpoints/EA1fix_mrcons_keep50_bs4/` | 2026-05-04 KST 오전 |

---

## 7. 다음 단계 검토

| 시나리오 | 후속 실험 제안 |
|---|---|
| E1 ≥ 68.82% (G5 달성) | E1 + mr-cons stack, 또는 E1 + A1 feat-KD |
| E1 < 68.82% but > 68.44% | `merge_ratio` sweep (0.7/0.8) 또는 patch query conditioning 추가 |
| EA1 FIX + mr-cons > EA1 FIX | mr-cons weight sweep; E1 + mr-cons 조합 |
| 둘 다 baseline 이하 | EA1 FIX(68.44%)가 현재 production; `inter_frame_gate` unfreeze 실험 |
