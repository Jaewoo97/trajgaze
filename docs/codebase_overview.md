# `/workspace/trajgaze/` 코드 요약

Plan mode에 컨텍스트로 넘기기 좋도록 정리한 문서.

---

## 시스템 개요

**TrajGazeMerge**: Egocentric VQA를 위한 token-reduction 시스템. Gaze + hand trajectory를 활용해 Qwen2.5-VL-7B-Instruct의 visual token을 score-weighted bipartite merging으로 압축.

- **Train**: EgoExoLearn + HoloAssist (5,799 items)
- **Test**: EGTEA (526 items, disjoint)
- **최신 결과**: E1 variant, ρ=0.90 (10% retention), **68.44%** overall

---

## 두 서브프로젝트

### 1. `TrajGaze_v2/` — Stage 1 (trajectory encoder pretraining)

**모델 클래스**: `TrajGazeV2Temporal` ([models/model_temporal.py](../TrajGaze_v2/models/model_temporal.py))

```
입력: gaze/hand trajectory (B, T=128, ...) + 16 keyframes
  ↓
QueryEncoder (Stage 1 미사용, zero vector)
VisualPatchEncoder (frozen DINOv2-S/14, 196×196 → 14×14 patches)
SpatiotemporalEncoder:
  TrajectoryTokenizer → 4 tokens/frame {g, L, R, φ}
  IntraFrameBlock (L1: 2층, 4heads, d_traj=128)
  Proj → d_enc=256 + SinusoidalPE
  InterFrameTransformer (L2: 6층, 8heads) ─┐
  tanh-gated residual (gate=0 frozen)     │
  FiLM (zero query)                        │
  TemporalVisualTrajFusion (per-frame xattn)│
  ─→ per_frame_scores (B, T, 196)          │
                                            │
  E1: PatchTemporalBranch ──────────────────┘
    196 learned queries → xattn(x_iframe)
    → modulation map M (B, T, 196)
    per_frame_scores *= M
  ↓
context (B, T, 4, D)
  ↓
├─ TrajScoreHead → 196 patch scores (inference-time)
├─ TrajectoryDecoder → 6d future positions
└─ ScoreDecoder → 196 future patch scores
```

**Stage 1 4-loss objective** (`stage1_forward`):
- `L_traj`: future trajectory MSE
- `L_score_fut`: future per-frame patch scores MSE
- `L_score_past`: encoder raw attn vs GT past maps
- `L_score_traj`: TrajScoreHead distillation from (dec_attn × enc_attn)

**훈련**: 100 epochs, AdamW lr=3e-4, cosine→1%, batch=2/GPU, ~55분 (단일 H200), gate frozen at 0.

**파라미터**: 36.11M total / 14.05M trainable (DINOv2 22.1M 제외)

---

### 2. `TrajGazeMerge/` — Stage 2 (joint LoRA + token merging)

**핵심 함수**: `gaze_weighted_merge` ([models/merge.py](../TrajGazeMerge/models/merge.py))

```
score-weighted bipartite merge:
  top (N-r) by score → receivers (보호)
  bottom r           → sources (각 source를 cosine-similar receiver로 merge)
  merged_i = (s_i·v_i + Σ s_j·v_j) / (s_i + Σ s_j + ε)
  → scatter_add 구현, top-k/argmax는 non-diff routing
```

**파이프라인** ([training/train_merge_lora_temporal_no_kd.py](../TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py)):

```
1. preprocess_item: Qwen ViT가 128 frames → video_embeds (4096 tokens, frozen)
2. TrajGaze encoder.get_patch_scores → (T=128, 196)
3. score_to_qwen_spatiotemporal:
   14×14 → nearest 16×16 → avgpool 8×8 (=64 spatial)
   then temporal interp T=128 → T_merged=64
   → (4096,) one score per video token
4. gaze_weighted_merge(video_embeds, scores, r=⌊0.9·4096⌋=3686)
   → 410 merged tokens + receiver_idx
5. build_merged_inputs: input_ids/attention/RoPE에서 source 제거,
   receiver 위치에 merged embed 삽입
6. forward → last-position logits over {A,B,C,D} token IDs
7. CE loss only (no KD, no teacher)
```

**Optimizer**: AdamW, LoRA lr=1e-4, encoder lr=1e-5, wd=1e-4. 3 epochs, batch=1, grad_accum=4.

**LoRA**: r=16, α=32, dropout=0.05, targets `{q,k,v,o}_proj` (LLM only, ViT 미적응). 10.09M trainable / 8.30B (0.122%).

**Stage 2 동작 변화**:
- DINOv2: frozen
- Qwen base (ViT + LLM): frozen, LoRA만 학습
- TrajGaze encoder (DINOv2 제외): 전체 fine-tune. gate `g`는 `requires_grad=True`로 release되어 학습됨

---

## 핵심 파일 위치

| 역할 | 파일 |
|---|---|
| Stage 1 entrypoint | [TrajGaze_v2/training/stage1_temporal.py](../TrajGaze_v2/training/stage1_temporal.py) |
| Stage 2 entrypoint | [TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py](../TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py) |
| E1 launcher | [TrajGazeMerge/training/run_e1_patch_temporal.sh](../TrajGazeMerge/training/run_e1_patch_temporal.sh) |
| Main model | [TrajGaze_v2/models/model_temporal.py](../TrajGaze_v2/models/model_temporal.py) |
| Encoder (E1 branch) | [TrajGaze_v2/models/encoder_temporal.py](../TrajGaze_v2/models/encoder_temporal.py) |
| Visual (DINOv2) | [TrajGaze_v2/models/visual_encoder_temporal.py](../TrajGaze_v2/models/visual_encoder_temporal.py) |
| Merge op | [TrajGazeMerge/models/merge.py](../TrajGazeMerge/models/merge.py) |
| Qwen + LoRA + IO | [TrajGazeMerge/models/model.py](../TrajGazeMerge/models/model.py) |
| Dataset (split 정의) | [TrajGazeMerge/data/dataset.py](../TrajGazeMerge/data/dataset.py) |
| Method 논문 | [TrajGazeMerge/paper/method.tex](../TrajGazeMerge/paper/method.tex) |
| 최신 ckpt | [TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth](../TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/) |

---

## Stage 1 아키텍처 variant flag (encoder_temporal.py)

| Flag | 효과 |
|---|---|
| `use_frame_score_branch` | (B,T) frame-level scalar로 per-patch score 곱 (EA1) |
| `use_patch_temporal_branch` | **E1, 현재 사용** — 196 learned queries × x_iframe → (B,T,196) modulation |
| `use_post_fusion_iframe` | enriched_context 위에 second InterFrameTransformer (b2) |
| `use_iframe_query_conditioning` | E1+B, patch_temporal_query를 x_iframe context로 conditioning |

68.44% 결과는 `use_patch_temporal_branch=True`, `--freeze-gate` 조합.

---

## 주의해야 할 함정 (이전 작업 중 발견)

1. **Stage 2 `evaluate()`가 `split="test"` = EGTEA를 직접 사용**. Best checkpoint는 EGTEA test acc 기준으로 저장됨. 단, train(EgoExoLearn+HoloAssist) ≠ test(EGTEA) 이므로 전통적 leakage는 아님 (checkpoint selection bias는 있음).
2. **Stage 1 query는 zero vector** (`torch.zeros(...)`), learned null-query embedding 아님. QueryEncoder는 Stage 1에서 호출 안 됨.
3. **Stage 1 best.pth는 training loss 기준**, validation 아님.
4. **Gate `g`**: Stage 1에서만 frozen. Stage 2에서는 `requires_grad`가 state_dict에 저장 안 되므로 로드 후 자동으로 trainable됨.
5. **eval_every 기본값**: 400 (전체 526건 평가 — 200은 너무 자주).
