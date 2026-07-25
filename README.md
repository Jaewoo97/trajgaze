# In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting

[![Paper](https://img.shields.io/badge/arXiv-2509.07447-b31b1b.svg)](https://arxiv.org/abs/2509.07447)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://taiyi98.github.io/projects/EgoGazeVQA)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/taiyi09/EgoGazeVQA)

> **Taiying Peng<sup>1</sup>, Jiacheng Hua<sup>2</sup>, Miao Liu<sup>2†</sup>, Feng Lu<sup>1†</sup>**  
> <sup>1</sup>State Key Laboratory of VR Technology and Systems, School of CSE, Beihang University  
> <sup>2</sup>College of AI, Tsinghua University  
> **NeurIPS D&B 2025**

---

## TrajGazeMerge — gaze-trajectory token selection for VisionZip (this fork)

This fork extends EgoGazeVQA with **TrajGazeMerge**: a study of gaze/hand-trajectory-guided
visual-token selection for Qwen2.5-VL-7B under a fixed **10% token budget** (VisionZip pruning).
The headline result is **M1 (VZ-complement, 63.01%)** — selecting the gaze/hand-relevant tokens
VisionZip's content attention discarded, added as a disjoint complement to its 7% content set.

- **Best model — M1 (VZ-complement top-k, 63.01%):** [`MODEL_M1_VZ_COMPLEMENT.md`](MODEL_M1_VZ_COMPLEMENT.md)
- **Full experiment grid & conclusions:** [`TRAINING_RUNS.md`](TRAINING_RUNS.md)
- **Scanpath side-channel (63.01 tie, gated off):** [`MODEL_SCANPATH_OURS.md`](MODEL_SCANPATH_OURS.md)
- **Code:** `TrajGazeMerge/` (models, training, data, eval) · `TrajGaze_v2/` (frozen TAS Stage-1 encoder) · `scripts/` (launch scripts)

Protocol: egtea 2-way eval (n=1011), gaze-overlay frames, 3-epoch LoRA (eff-batch 8, early-stop),
`--merge-ratio 0.9`. Model weights and raw datasets are **not** included in this repo.

---

## Overview

This repository provides the official code for **EgoGazeVQA**, a benchmark for evaluating multimodal large language models (MLLMs) on egocentric video understanding tasks with gaze guidance. 

**Code Purpose:**
- Generate gaze-guided QA pairs from egocentric videos (spatial, temporal, causal intent questions)
- Evaluate MLLMs with three gaze-guided prompting strategies (textual, visual marks, salience maps)
- Calculate and analyze model performance on intent understanding tasks

<p align="center">
  <img src="assets/draw_intro.png" alt="EgoGazeVQA Overview" width="100%">
</p>

## Timeline

- **[2025-10]** Code and dataset publicly released
- **[2025-09]** Paper accepted at NeurIPS 2025 D&B Track



## TODO

- [ ] Add fine-tuning scripts for LoRA adaptation



## Repository Structure

```
EgoGazeVQA/
├── generate_tool/
│   ├── auto.sh
│   ├── spatial.py
│   ├── temporal.py
│   ├── causal.py
│   └── create_datasets.py
├── test_tool/
│   ├── qwenvl_test/
│   │   ├── test_wo.py
│   │   ├── test_gaze.py
│   │   ├── test_mark.py
│   │   └── test_saliencemap.py
│   ├── prompt_gazees/
│   ├── multiframes/
│   ├── gaze_trajectory.py
│   └── caculate.py
```



## Installation

```bash
git clone https://github.com/taiyi98/EgoGazeVQA.git
cd EgoGazeVQA

conda create -n egogazevqa python=3.10
conda activate egogazevqa

pip install -r requirements.txt
```



## Dataset

Download from [🤗 Hugging Face](https://huggingface.co/datasets/taiyi09/EgoGazeVQA):

```bash
huggingface-cli download taiyi98/EgoGazeVQA --repo-type dataset --local-dir ./data
```



## Usage

### Generate QA Pairs

```bash
cd generate_tool

# Generate for specific video
python spatial.py --video_id <VIDEO_ID> --target_index <INDEX>
python temporal.py --video_id <VIDEO_ID> --target_index <INDEX>
python causal.py --video_id <VIDEO_ID> --target_index <INDEX>

# Batch processing
bash auto.sh
```

### Evaluate Models

```bash
cd test_tool/qwenvl_test

python test_wo.py              # Baseline (no gaze)
python test_gaze.py            # Textual gaze prompt
python test_mark.py            # Visual gaze marks
python test_saliencemap.py     # Gaze salience maps
```

### Calculate Results

```bash
cd test_tool
python caculate.py --result_file <RESULT_CSV_PATH>
```


## Citation

```bibtex
@misc{peng2025eyemllmbenchmarkingegocentric,
    title={In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting}, 
    author={Taiying Peng and Jiacheng Hua and Miao Liu and Feng Lu},
    year={2025},
    eprint={2509.07447},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.07447}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
