<div align="center">
<h1>A Motion-aware Spatio-temporal Graph for Video Salient Object Ranking (NeurIPS 2024)</h1>
<h3> An UNOFFICIAL reproducible and extensible implementation version</h3>
</div>


## Overview

This project is based on the NeurIPS 2024 paper:

> *A Motion-aware Spatio-temporal Graph for Video Salient Object Ranking* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fc03d122a7e08d21aa92573113790a3-Abstract-Conference.html)

Thanks to the authors for releasing their code.

The original released code focuses on the core model, but lacks the environment, dataset preparation and trained model checkpoint. And their code cannot run directly.

This repository is created to share the experience and **improve reproducibility, usability, and extensibility** for the research community.

> ⚠️ **Disclaimer**  
> This is **not an official repository** released by the original authors.

Official Implementation: [[Code]](https://github.com/zyf-815/VSOR)


## TODO

- [x] Upload Cleaned detectron2 (Feb 9, 2026)
- [x] Detectron2 installation instructions (Feb 10, 2026)
- [x] Environment setup steps (Feb 10, 2026)
- [x] Revise training code (Feb 13, 2026)
- [x] Upload revised dataset (Feb 13, 2026)
- [x] Revise inference code (Feb 20, 2026)
- [x] Upload model checkpoint (Feb 21, 2026)

## Environment Setup

### Requirements

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision version compatible with the installed PyTorch
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
rm -rf build/ detectron2.egg-info build
```

Build Detectron2 from source:

```bash
pip install -r requirements.txt
python -m pip install -e .
```


## Datasets

### RVSOD

  #### Option 1
  
  Download our revised dataset through [Google Drive](https://drive.google.com/file/d/1eH-wHzw4CPmIg88vTXqP4GuAUNhmtpKU/view?usp=sharing) and extract it to `/Dataset`.
  
  #### Option 2
  1. Download from the [official repository](https://github.com/Pchank/Ranking-Video-Salient-Object-Detection) and extract it to `/Dataset`.
  2. Run `python tools/fix_gt_size_inplace.py` to fix the gt images with the wrong size (you can see them through `python tools/audit_rvsod_rank_masks.py`).
  3. Run `python tools/make_pkl_from_maskpng.py` to generate `train.pkl` and `test.pkl`.

### DAVSOR
  Coming soon.


## Training

1. Download the pretrained model pre_model.pth from [Google Drive](https://drive.google.com/file/d/189II7BcY5Bn6CAxV1AHcjX-5ktRAwSGE/view?usp=sharing) or [Saliency-Ranking](https://github.com/dragonlee258079/Saliency-Ranking) and place it into the ``model/`` directory.

2. Run:
```
python tools/plain_train_net_our.py
```

3. The log and checkpoint will be saved in ``RankSaliency/``.

## Testing

1. Download the trained model from [Google Drive](https://drive.google.com/file/d/1jtL442Dej4gp-7FOMSxlsJgAn5Pcicdu/view?usp=sharing) and place it into the ``RankSaliency/Models/RVSOD(1)/`` directory.

2. Run:
```
python tools/DrawFianlFigure.py
```
to obtain the quantification results and predicted images.
Or run:
```
python tools/plain_test_net.py
```
to obtain the quantification results only.

## Result

| Dataset | SASOR (Original) | Norm. SASOR (Original) | SASOR (All) | Norm. SASOR (All) | SASOR (Reported) | MAE (SOD) | MAE (SOR) | MAE (Reported) |
|:-------:|:----------------:|:----------------------:|:-----------:|:-----------------:|:----------------:|:---------:|:---------:|:--------------:|
| RVSOD | 0.3908 | 0.6954 | 0.2072 | 0.6036 | 0.603 | 0.0668 | 0.0653 | 0.0698 |

You can download the [checkpoint](https://drive.google.com/file/d/1jtL442Dej4gp-7FOMSxlsJgAn5Pcicdu/view?usp=sharing), [config](https://drive.google.com/file/d/1BsOBaihtC7Rv4LoHU4qH3EnCEBmFa-N7/view?usp=sharing), [log
](https://drive.google.com/file/d/1ZH3PZiLFVKyTKCPNb7QMVshS80MzcL8L/view?usp=sharing), [TB log](https://drive.google.com/file/d/1jaGgXU17HLTDDLlL_QiwB7wjybFiSmHt/view?usp=sharing).

For more discussion about the result, please see [this](assets/readme.md).

## Acknowledgment

The project is based on VSOR ([Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fc03d122a7e08d21aa92573113790a3-Abstract-Conference.html), [Code](https://github.com/zyf-815/VSOR)), [Detectron2](https://github.com/facebookresearch/detectron2), Saliency-Ranking ([Paper](https://ieeexplore.ieee.org/iel7/34/4359286/09523772.pdf), [Code](https://github.com/dragonlee258079/Saliency-Ranking)).
