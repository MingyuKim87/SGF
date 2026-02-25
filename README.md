<div align="center">

# Safety-Guided Flow (SGF) (ICLR2026 Oral)

<p align="center">
  [<a href="https://openreview.net/forum?id=EA80Zib9UI"><strong>OpenReview</strong></a>]
  <!-- [<a href=""><strong>ArXiv</strong></a>] -->
  <!-- [<a href=""><strong>Project</strong></a>] -->
  <!-- [<a href="#citation"><strong>BibTeX</strong></a>] -->
</p>

</div>

Official PyTorch implementation of **Safety-Guided Flow (SGF)**, as presented in our paper: \
\
**Safety-Guided Flow (SGF): A Unified Framework for Negative Guidance in Safe Generation (ICLR2026)** \
Mingyu Kim, Young-Heon Kim and Mijung Park 

<img src="assets/teaser.png" width="100%">  

---

## Update
- [x] **Safety (SD-v1.4):** Nudity + COCO-30k.
- [x] **Memorization (SD-v2.1):** memorization / replication setting.
- [x] **Diversity (SD-v3):** diversity evaluation setting.

---

## How to Run

- **Safety (SD-v1.4)**: see [`nudity_sdv1/README.md`](nudity_sdv1/README.md)
- **Memorization (SD-v2.1)**: see [`memorized_sdv21/README.md`](memorized_sdv21/README.md)
- **Diversity (SD-v3)**: see [`diversity_sdv3/README.md`](diversity_sdv3/README.md)


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
