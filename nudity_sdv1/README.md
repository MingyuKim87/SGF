<div align="center">

# Safety-Guided Flow (SGF) : SD-v1 Nudity & CoCo-30k Task (ICLR2026 Oral)

<p align="center">
  [<a href="https://openreview.net/forum?id=EA80Zib9UI"><strong>OpenReview</strong></a>]
</p>

</div>

Official PyTorch implementation of **Safety-Guided Flow (SGF)**, as presented in our paper: \
\
**Safety-Guided Flow (SGF): A Unified Framework for Negative Guidance in Safe Generation (ICLR2026 Oral)** \
Mingyu Kim, Young-Heon Kim, and Mijung Park

---

## Update
- [x] **SGF and Safe Denoiser (Kim et al. 2025) implementation & configs** completed.
- [x] **Generation and Evaluation** code completed (end-to-end runnable pipeline).


## Code Base

This repository is built on top of the code base from:

- **Training-Free Safe Denoisers for Safe Use of Diffusion Models (NeurIPS 2025)**
- Repo: https://github.com/MingyuKim87/Safe_Denoiser

So, you should **first complete the environment / setup required by Safe_Denoiser**, and then apply the additional SGF-specific setup described below.

> **IMPORTANT (Must-do prerequisites)**  
> To run **SGF on SD-v1 nudity task**, the following SafeDenoiser assets are **mandatory**:
> 1) **Checkpoints:** **Nudinet**, **Q16**, **AES**  
> 2) **Negative Datapoints:** **515 nudity images**  
> 3) **Cached repellency projection references (recommended):** `repellency_proj_ref.pt`  
>    - File path example: `datasets/nudity/i2p_sexual/repellency_proj_ref.pt`

---

## Environment Setup

> Follow the original **Safe_Denoiser** repository instructions first (requirements, checkpoints, datasets).
> After that, install any additional dependencies required by SGF.

A typical setup looks like:

```bash
# (example) create env
conda create -n SGF python=3.10 -y
conda activate SGF

# install packages
pip install -r requirements.txt
```

### Nudity (SD-v1.4)

This script evaluates the nudity task on different prompt sets by switching `--data`.

**Supported prompt sets**
- Ring-A-Bell: `datasets/nudity-ring-a-bell.csv`
- UnlearnDiffAtk: `datasets/nudity.csv`
- MMA-Diffusion: `datasets/mma-diffusion-nsfw-adv-prompts.csv`

```bash
# Example: Ring-A-Bell
python generate_unsafe_sgf.py \
  --nudenet-path=pretrained/classifier_model.onnx \
  --nudity_thr=0.6 \
  --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM \
  --data=datasets/nudity-ring-a-bell.csv \
  --category=nudity \
  --task_config=configs/sgf/sgf.yaml \
  --save-dir=results/sgf/sdv1/nudity \
  --erase_id=safree_neg_prompt_rep_time
```

### COCO-30k Prompts (SD-v1.4)

This run generates images from **COCO-30k** prompts by setting `--category=coco`.
It uses the same base setup as the nudity runs (same `--config`, sampling steps, and safety method), but switches the task to COCO prompts.

```bash
python generate_coco30k_sgf.py \
  --nudenet-path=pretrained/classifier_model.onnx \
  --nudity_thr=0.6 \
  --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM \
  --category=coco \
  --task_config=configs/sgf/sgf.yaml \
  --save-dir=results/sgf/sdv1/coco \
  --erase_id=safree_neg_prompt_rep_time \
  --guidance_scale=7.5
```

## Evaluation

### FID and CLIP Score

We provide a simple evaluation script for **COCO-30k** generations that reports **FID** and **CLIP Score**.

Run:
```bash
python evaluate_coco30k_fid_clip.py --target_path <PATH_TO_GENERATED_RESULTS>
```

- `--target_path` should point to the same directory used in `--save-dir` during the generation run.

```bash
# Evaluation (FID, CLIP Score)
python evaluate_coco30k_fid_clip.py \
  --target_path results/safe_denoiser/sdv1/coco
```

## Citation

```bibtex
@inproceedings{
  kim2026safetyguided,
  title={Safety-Guided Flow (SGF): A Unified Framework for Negative Guidance in Safe Generation},
  author={Mingyu Kim and Young-Heon Kim and Mijung Park},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=EA80Zib9UI}
}
```