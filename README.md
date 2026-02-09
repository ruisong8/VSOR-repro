<div align="center">
<h1>VSOR (NeurIPS 2024)</h1>
<h3> An UNOFFICIAL reproducible and extensible implementation version</h3>
</div>

> **Status:** 🚧 Under active development  
> This repository is a **community-maintained reproduction and extension** of the VSOR project.  
> Environment setup, training scripts, and pretrained models will be continuously updated.

---

## 📌 Overview

This project is based on the NeurIPS 2024 paper:

> *A Motion-aware Spatio-temporal Graph for Video Salient Object Ranking* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fc03d122a7e08d21aa92573113790a3-Abstract-Conference.html)

The original released code focuses on the core model and evaluation logic, but lacks environment setup and trained model checkpoint.  
This repository is created to **improve reproducibility, usability, and extensibility** for the research community.

> ⚠️ **Disclaimer**  
> This is **not an official repository** released by the original authors.

Official Implementation: [[Code]](https://github.com/zyf-815/VSOR)


## 🎯 TODO

- Detectron2 installation instructions
- Environment setup steps
- Upload model checkpoint.


## 📊 Datasets

- **RVSOD**  
  https://github.com/Pchank/Ranking-Video-Salient-Object-Detection

- **DAVSOD**  
  https://github.com/DengPingFan/DAVSOD

The official train/test split for DAVSOD can be found in:
`Dataset/DAVSOD/train.txt`
`Dataset/DAVSOD/test.txt`


## 🤝 Acknowledgment

The project is based on VSOR ([Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fc03d122a7e08d21aa92573113790a3-Abstract-Conference.html), [Code](https://github.com/zyf-815/VSOR)), [Detectron2](https://github.com/facebookresearch/detectron2).
