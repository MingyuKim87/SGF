# SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation

This is the implementation for SAFREE with SD-v3

## Install Dependencies & Download Pre-trained T2I Models
Please refer to StableDiffusion-v3 repo installation/preparation [page](https://huggingface.co/stabilityai/stable-diffusion-3-medium)

## Benchmarks & Datasets Download
Please refer to [P4D](https://joycenerd.github.io/prompting4debugging/), [Ring-A-Bell](https://github.com/chiayi-hsu/Ring-A-Bell), [MMA-Diffusion](https://github.com/cure-lab/MMA-Diffusion), [Unleardiff](https://github.com/OPTML-Group/Diffusion-MU-Attack) and [CoCo](https://huggingface.co/datasets/HuggingFaceM4/COCO)

We also upload dataset/benchmarks [here](../datasets/) for easy setup.

## Running SAFREE with SDv3

```bash
python sdv3 --dataset_name [dataset] --csv [your path dataset] --save_path [your save path]
```


## Evaluation

Todo: upload evaluation scripts
