그# Session Handoff — 다음 세션 콜드 스타트 가이드

작성: 2026-06-18. 같이 읽을 것: `GAZE_EXCLUSIVE_PERFORMANCE_STRATEGY.md`(why/전략),
메모리 `project_gaze_exclusive_roi_direction`, `reference_eval_is_multichoice`.

---

## 1. 한 줄 현황
M1(VisionZip-Complement, 7%C∪3%G) = **63.01%** best (egtea, n=1011, **다지선다** chance~25%).
토큰 **선택은 maxed**(5+ 변형 falsified, query 포함 −0.89). → **gaze-exclusive 기여**로 전환,
현재 타깃 = **④ foveated ROI** (object-ID/attribute subset, de-risk가 ③→④로 redirect).

**[2026-06-18 업데이트]** ④ trainer **빌드+GPU검증 완료**(`training/train_visionzip_foveal_lora.py`).
attention-twin + random control을 `foveal_roi.py`에 추가, encode 버그(processor text=None→K=0 no-op)
수정. **런처 무장**(`launch_foveal.sh`): curve GPU 해제 시 Wave A(gaze∥attn)→Wave B(random) 자동 발사.
arms=gaze+attn+random, crop_frac 0.35/margin 0.08. 진행은 §7(strategy doc)·아래 §6 참조.

## 2. ⚠️ 재개 즉시 확인할 것 — 돌고 있던 GPU 잡 (세션 clear에도 생존: setsid-detached)
**모니터(run_in_background)는 clear로 죽으므로, run.log를 직접 grep할 것.**

budget-curve Wave 1 (⑤: "5%에서 M1−VZ gap이 벌어지나"):
```bash
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
grep Overall: $R/curve_m1_05pct/run.log   # M1@5%
grep Overall: $R/curve_vz_05pct/run.log   # VZ@5%
pgrep -af curve_.*_05pct                   # 아직 도는지
```
- 기준: gap@10% best = 63.01−62.51 = **+0.50**. gap@5% ep1 = 61.13−60.53 = **+0.60**(안 벌어짐).
- ep2(best) gap이 +0.5 근처면 ⑤(압축=gaze 중요) **최종 null** → ④에 집중. 벌어졌으면 ⑤ 재고.

## 3. 다음 작업 = ④ ROI trainer 빌드 (CPU, 그다음 GPU 런)
**무엇:** `models/foveal_roi.py`(완성·검증)를 trainer에 연결.
- fork `training/train_visionzip_complement_lora.py` → `select_complementary`(M1 topk 그대로) 뒤에
  `extract_roi_crops → encode_foveal_tokens → build_inputs_with_foveal` 삽입.
- **3 arm (control = attention-twin)**:
  - `gaze-ROI`  : gaze 중심 crop (방법)
  - `attn-ROI`  : VisionZip attention 최대점 중심 crop (content-only 쌍둥이; foveal_roi에 point만 교체)
  - `no-ROI`    : M1 그대로 (=63.01 baseline)
  - (옵션) `random-ROI` placebo
- **V1 = added**(10% 토큰 유지 + K foveal 토큰). crop margin **넉넉히**(gaze center 쏠림 → 작은 crop은 referent 놓침).
- **측정**: object-ID subset에서 gaze-ROI vs attn-ROI vs no-ROI → `eval/eval_dump`로 per-item 덤프 → `eval/mcnemar`.
  핵심 = **gaze-ROI > attn-ROI** (gaze가 해상도 배분에서 load-bearing = 기여).
- 학습 프로토콜: M1과 동일(2-GPU DDP grad-accum 4 = eff-batch 8, 3 epoch, early-stop, GAZE_OVERLAY=1).
- sweep(나중): `foveal_roi.SWEEP_GRID` = crop_frac/foveal_k/roi_max_pixels/n_fix_frames.

**미결 결정(빌드 시작 전 사용자 확인):** ① random-ROI placebo 포함 여부, ② crop margin/크기 기본값.

## 4. 이 세션에서 만든 도구 (전부 CPU-검증)
| 파일 | 역할 |
|---|---|
| `models/foveal_roi.py` | ④ ROI: fixation검출/crop/ViT재인코딩/주입, `FovealROIConfig`+`SWEEP_GRID` |
| `models/gaze_grounding.py` | ③ 문장(region+hand OR) — ②subset엔 폐기, spatial/temporal엔 재검토 여지 |
| `eval/audit_tasks.py` | task×(future/deictic/gaze_val) 감사 + gaze-necessary subset |
| `eval/derisk_object_id.py` | ② de-risk: 오답이 해상도/혼동인지 (gaze 특징 조인) |
| `eval/eval_dump.py` | per-item 덤프(+question/options/pred_text/gt_text) — McNemar용 |
| `eval/mcnemar.py` | paired McNemar (exact, scipy 불필요) |
| `eval/pertask_compare.py` | 학습 로그→per-task 표 + epoch-jitter 밴드 |
| `eval/eval_tta.py` | 옵션 대칭화(위생, gap 측정용) |
| `training/train_visionzip_complement_lora.py` | `--query-mode {cosine,random,shuffle}`, `query_gaze` 모드 추가됨 |

자산: M1 per-item 덤프 = `checkpoints/dumps/m1.jsonl`(63.01); gaze-necessary subset = `/tmp/audit_gaze_necessary.json`.
환경: conda env `trajgaze` (`/opt/conda/envs/trajgaze/bin/python`), GPU 4×143GB 공유 박스, `export GAZE_OVERLAY=1`.

## 5. 측정 규율 (반복)
어떤 개선도 (1) **gap**(M1·VZ 또는 gaze-arm vs attn-arm)으로 판단, (2) per-task로 *어디서* 이기는지,
(3) McNemar로 유의성, (4) **수치 verdict 전에 오답 정성 검토**. 절대 숫자 단독 신뢰 금지.

## 6. 재개 체크리스트
- [x] ⑤ budget-curve null 확정 (메모리 `project_budget_curve_falsified`)
- [x] ④ 3 arm 학습 + McNemar → **④ 반증**: gaze-ROI = attn-twin (p=1.000). 메모리 `project_foveal_roi_falsified`
- [x] ① anticipatory 빌드 + de-risk(velocity 퇴화→실제 미래 gaze) + GPU1 학습 발사
- [ ] **① 결과 확인** (~13h, 04:08 발사 06-20): `dumps/antic_mcnemar_result.txt` (자동 dump+McNemar).
      핵심 = **attn vs antic**(미래 gaze가 attention 못 보는 신호 주는지). future_action subset per-task 주목.
- [ ] none(매칭 no-ROI 단일-GPU) best 확인: `foveal_none/run.log` (foveal-added vs no-ROI)
- [ ] ① 결과 → §7 + 메모리 갱신. **①도 tie면 → "gaze≈attention on egtea" 결론, thesis 재구성/세팅변경 결정**
- 하트비트: `checkpoints/foveal_launch.log`
