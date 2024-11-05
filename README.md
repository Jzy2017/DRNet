<div align="center">

# 【PR'2025🔥】 DRNet: Learning a dynamic recursion network for chaotic rain streak removal
</div>

Welcome! This is the official implementation of our paper: [**DRNet: Learning a dynamic recursion network for chaotic rain streak removal**](https://www.sciencedirect.com/science/article/pii/S0031320324007556)

Authors: [Zhiying Jiang](https://scholar.google.com/citations?user=uK6WHa0AAAAJ&hl=zh-CN&oi=ao), [Risheng Liu](https://rsliu.tech/), [Shuzhou Yang](https://ysz2022.github.io/), [Zengxi Zhang](https://scholar.google.com/citations?user=lqlA92AAAAAJ&hl=zh-CN&oi=ao), [Xin Fan](https://scholar.google.com/citations?user=vLN1njoAAAAJ&hl=zh-CN)*.

## 🔑 Prerequisites
- Linux or macOS
- Python 3.8
- NVIDIA GPU + CUDA CuDNN

```bash
pip install torch h5py scikit-video
```

Type the command:
```bash
pip install -r requirements.txt
```

## 🤖 Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

## 🚀 Inference
```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

## 📌 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{JIANG2025111004,
title = {DRNet: Learning a dynamic recursion network for chaotic rain streak removal},
journal = {Pattern Recognition},
volume = {158},
pages = {111004},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.111004},
author = {Zhiying Jiang and Risheng Liu and Shuzhou Yang and Zengxi Zhang and Xin Fan}
}
```