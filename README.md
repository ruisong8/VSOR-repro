<div align="center">
<h1>VSOR (NeurIPS 2024)</h1>
<h3> An UNOFFICIAL reproducible and extensible implementation version</h3>
</div>

> **Status:** 🚧 Under active development  
> Environment setup, training scripts, and pretrained models will be continuously updated.

---

## 📌 Overview

This project is based on the NeurIPS 2024 paper:

> *A Motion-aware Spatio-temporal Graph for Video Salient Object Ranking* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fc03d122a7e08d21aa92573113790a3-Abstract-Conference.html)

The original released code focuses on the core model and evaluation logic, but lacks environment setup and trained model checkpoint.  
This repository is created to share the experience and **improve reproducibility, usability, and extensibility** for the research community.

> ⚠️ **Disclaimer**  
> This is **not an official repository** released by the original authors.

Official Implementation: [[Code]](https://github.com/zyf-815/VSOR)


## 🎯 TODO

- [x] Upload Cleaned detectron2 (Feb 9, 2026)
- [x] Detectron2 installation instructions (Feb 10, 2026)
- [x] Environment setup steps (Feb 10, 2026)
- [ ] Upload model checkpoint
- [ ] ...

## 🛠️ Environment Setup

### Requirements

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision version compatible with the installed PyTorch
- OpenCV (optional, required for visualization)
- GCC & G++ ≥ 5

### ✅ Tested Environment

The following configuration has been tested successfully:

- OS: Ubuntu 22.04
- GPU: NVIDIA RTX 4090 / 4090D (Ada Lovelace)
- Python: 3.10
- PyTorch: 2.1.0
- CUDA: 12.1

### Build from Source

```bash
git clone https://github.com/ruisong8/VSOR-repro.git
cd VSOR-repro
```

> **Important Note**  
> If you clone from the **official VSOR repository** instead of this fork,  
> precompiled Detectron2 binaries may be present and must be removed before building.

To check for existing build artifacts, run:

```bash
find . -name "*.so" -path "*detectron2*" -print
find . -name "build" -type d -prune
```

If any Detectron2 shared objects are found, clean them with:

```bash
rm -f detectron2/_C.cpython-37m-*.so detectron2/_C.cpython-38-*.so
rm -rf detectron2.egg-info build
```

Build Detectron2 from source:

```bash
python -m pip install -e .
```


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
