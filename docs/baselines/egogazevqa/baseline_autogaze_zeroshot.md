# Baseline — NVILA-8B-HD-Video native AutoGaze zero-shot

HF processor 가 사용자 비디오에 대해 **AutoGaze 를 on-the-fly 로 실행** 하고 그 결과 gazing_info 를 그대로 HD-Video 에 전달하는 순정 inference 파이프라인. 학습 없음, 사용자 `.pt` 주입도 없음. Path A/B/C 가 기준선(baseline) 으로 삼는 숫자.

```
 video (mp4)
    ▼
 AutoProcessor.preprocess  (tile 392 + thumbnail split)
    ▼
 AutoGaze.forward          (VideoMAE-based gaze selection, on-the-fly)
    ▼                       → tile/thumbnail gazing_info
 vision_tower              (scales = "56+112+196+392", 1060 tok/frame, default)
    ▼
 mm_projector → Qwen2 LLM.generate → 첫 A-E letter
```

---

## 1. Environment

Path B 와 동일. 세션마다 export:

```bash
export LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8
export LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Conda env `trajgaze`. 하드웨어 2×H200 (bf16, sdpa).
- `autogaze_model_id="nvidia/AutoGaze"` 로 override (`bfshi/AutoGaze` 는 비공개 레포).

---

## 2. Data assets (read-only)

| 용도 | 경로 |
| --- | --- |
| 원본 QA (1750행) | `/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/metadata.csv` |
| 원본 클립 (mp4, processor 가 직접 디코드) | `/home/yujin/dataset/EgoGazeVQA/{dataset}/{video_id}/{start}_{end}.mp4` |
| 8:2 split | `/home/yujin/gaze/trajgaze/vila_hd_work/splits.json` (1373 train / 377 test) |

Baseline 은 test split 377 샘플 전체에 대해 돌리며, 사용자 `.pt` / JPG 시퀀스는 **쓰지 않는다**.

---

## 3. Pipeline internals (`eval_egogaze.py --mode native`)

| 단계 | 구현 |
| --- | --- |
| Processor | `AutoProcessor.from_pretrained("nvidia/NVILA-8B-HD-Video", autogaze_model_id="nvidia/AutoGaze", num_video_frames=32, num_video_frames_thumbnail=16, max_tiles_video=16, …, trust_remote_code=True)` |
| Model | `AutoModel.from_pretrained(..., torch_dtype=bf16, attn_implementation="sdpa", max_batch_size_siglip=16, device_map="auto")` |
| Prompt | `f"{video_token}\n\n{question}\nOptions:\n{A..E}\nAnswer with the letter only."` |
| Generate | `model.generate(**inputs, max_new_tokens=8)` |
| Answer parse | `re.search(r"\b([A-E])\b", raw)` — 첫 매치 letter. |

모델 / processor / AutoGaze 모두 FROZEN. `model.eval()` + `torch.inference_mode()`.

---

## 4. Key decisions (why)

- **`--mode native`**: HD-Video 가 기본으로 상정하는 경로 → precomputed 주입 없이 가장 공정한 "모델만" baseline.
- **`scales` override 없음**: HD-Video 체크포인트 기본 `"56+112+196+392"` 그대로. 분포 이동 0.
- **`num_video_frames=32`**: 2×H200 OOM 없이 돌아가는 최대치 (128 은 OOM). Path A/B/C 와도 같은 K 로 맞춰 비교 공정성 확보.
- **`max_tiles_video=16`**: 기본 48 에서 축소 (HD-Video QuickStart 권장).

---

## 5. 파일 레이아웃 (baseline 관련분만)

```
vila_hd_work/
  eval_egogaze.py              ← native / precomputed eval 엔트리
  splits.json                  ← 8:2 group split
  out/baseline_test.jsonl      ← 377 샘플 per-sample 결과
```

---

## 6. 실행 커맨드

### Full test eval (377 샘플)

```bash
cd /home/yujin/gaze/trajgaze/vila_hd_work && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_PRELOAD=/opt/conda/envs/trajgaze/lib/libjpeg.so.8 \
LD_LIBRARY_PATH=/opt/conda/envs/trajgaze/lib \
/opt/conda/envs/trajgaze/bin/python eval_egogaze.py \
  --mode native --split test \
  --out-jsonl out/baseline_test.jsonl \
  2>&1 | tee out/baseline_test.log
```

---

## 7. 결과 (test split 377 샘플)

| 지표 | 값 |
| --- | --- |
| **Overall** | **204 / 377 = 0.5411** |
| qa_type=causal | 107 / 127 = 0.8425 |
| qa_type=spatial | 45 / 125 = 0.3600 |
| qa_type=temporal | 52 / 125 = 0.4160 |
| dataset=ego4d | 74 / 136 = 0.5441 |
| dataset=egoexo | 84 / 147 = 0.5714 |
| dataset=egtea | 46 / 94 = 0.4894 |

(soft regex로 A-E 추출, 미매치 없음.)

Causal 이 월등히 높고 spatial/temporal 이 낮은 패턴 — Path B LoRA SFT 의 개선 지점 후보.

---

## 8. Gotchas

- **`bfshi/AutoGaze` 401 Unauthorized**: 원 processor 는 이 레포를 찾아감. `autogaze_model_id="nvidia/AutoGaze"` 로 반드시 override.
- **첫 실행은 AutoGaze 가중치 다운로드 포함으로 느림**. 그 다음 실행부터는 캐시 사용.
- **`num_video_frames=128`, `max_tiles_video=48`**: OOM. 16/16/32 구성으로 축소한 것이 현재 baseline 의 전제.
- **dataset-root**: `/home/yujin/dataset/EgoGazeVQA` (all_gaze_v1/ 가 아니다 — 클립은 parent 에 있음).

---

## 관련 문서

- [path_a_autogaze_precomputed.md](path_a_autogaze_precomputed.md) — 사용자 `.pt` 주입 zero-shot.
- [path_b_autogaze_lora_finetune.md](path_b_autogaze_lora_finetune.md) — HD-Video LoRA SFT.
- [path_c_nvila_lite_8b_lora_finetune.md](path_c_nvila_lite_8b_lora_finetune.md) — NVILA-Lite-8B 에 AutoGaze ViT 이식 + LoRA SFT (plan).
- 코드: [`vila_hd_work/eval_egogaze.py`](../../vila_hd_work/eval_egogaze.py).
