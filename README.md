<p align="center">
  <img src="docs/assets/real_mri_slices.png" alt="IXI Brain MRI Slices" width="720"/>
</p>

<h1 align="center">🧠 SMORE — MRI Super-Resolution & Anti-Aliasing</h1>

<p align="center">
  <strong>Digital Image Processing · Advanced Course Project</strong>
</p>

<p align="center">
  <a href="https://github.com/kashif-techdev/smore"><img src="https://img.shields.io/badge/GitHub-kashif--techdev%2Fsmore-181717?logo=github" alt="Repository"/></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Dataset-IXI%20Brain%20MRI-purple" alt="Dataset"/>
  <a href="https://github.com/kashif-techdev/smore/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/kashif-techdev/smore/ci.yml?branch=main&label=CI" alt="CI"/></a>
</p>

<p align="center">
  Muhammad Kashif (231568) · Abdulrehman Afaq (231577) · Umair Habib (231694)
</p>

---

## Overview

This repository implements an **SMORE-inspired** pipeline for **2D brain MRI super-resolution** and **anti-aliasing restoration**, developed as a Digital Image Processing (DIP) capstone project. We train a **U-Net** to recover high-resolution (HR) axial slices from synthetically degraded low-resolution inputs that mimic **nearest-neighbor downsampling artifacts**—a practical analogue to missing anti-aliasing in acquisition or reconstruction chains.

The work is motivated by:

> **SMORE:** *A Self-Supervised Anti-Aliasing and Super-Resolution Algorithm for MRI Using Deep Learning*  
> Zhao et al., IEEE Transactions on Medical Imaging, 2021 · [DOI: 10.1109/TMI.2020.3037187](https://doi.org/10.1109/TMI.2020.3037187)

Our implementation is a **supervised U-Net baseline** on real **IXI** NIfTI volumes (skull-stripped, MNI-registered), with rigorous **PSNR / SSIM** evaluation and ablation against classical bicubic upsampling.

### v1 → v2 improvements

| What | v1 (original) | v2 (current) | Why |
|------|---------------|--------------|-----|
| **Loss** | `nn.MSELoss()` | `combined_loss()` — **80% SSIM + 20% MSE** | MSE alone blurs structure; SSIM optimizes perceptual similarity |
| **Subjects** | 30 | **80** | 2.67× more data → better generalization |
| **Epochs** | 20 | **75** (stopped at **53**) | More time to converge |
| **Early-stop patience** | 5 | **10** | Avoid stopping during loss plateaus |
| **Dependency** | — | `pytorch_msssim` | Differentiable SSIM for training |

---

## Key Results (v2 — test set)

| Method | PSNR ↑ (dB) | SSIM ↑ |
|--------|-------------|--------|
| Aliased LR input | 28.66 | 0.8888 |
| Bicubic interpolation | 29.43 | 0.8806 |
| **U-Net (ours)** | **34.88** | **0.9737** |
| **Gain vs. bicubic** | **+6.22 dB** | **+0.0849** |

> **PSNR +6.22 dB** over bicubic is a large gain for 4× super-resolution. **SSIM 0.9737** confirms the combined loss fixed the structural blurring seen with pure MSE training.

**Training:** 8,552 slices from 80 subjects · converged at **epoch 53/75** · best val **34.87 dB PSNR** · **0.9732 SSIM** (Tesla T4).

---

## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Data["📦 Data"]
        A[Kaggle IXI NIfTI<br/>wm*.nii volumes]
        B[Axial slice extraction<br/>128×128, normalized]
    end

    subgraph Degrade["⚡ Degradation"]
        C[4× nearest-neighbor downsample]
        D[Bicubic upsample → aliased LR]
    end

    subgraph Model["🔬 Model"]
        E[U-Net · 7.76M params]
        F[HR reconstruction]
    end

    subgraph Eval["📊 Evaluation"]
        G[PSNR / SSIM]
        H[Ablation & Sobel edges]
    end

    A --> B --> C --> D --> E --> F --> G --> H
    B -. ground truth .-> E
```

### Degradation comparison

Nearest-neighbor downsampling **without** anti-aliasing introduces ringing and jagged structures; bicubic downsampling is smoother but blurrier. The model is trained to invert the **aliased** path.

<p align="center">
  <img src="docs/assets/degradation_simulation.png" alt="HR vs Bicubic vs Aliased degradation" width="900"/>
  <br/>
  <em>Ground truth vs. bicubic LR vs. aliased LR (4× factor)</em>
</p>

---

## Visual Results

### Super-resolution on test slices

<p align="center">
  <img src="docs/assets/test_comparison.png" alt="LR input vs U-Net output vs ground truth" width="900"/>
  <br/>
  <em>Low-resolution input · U-Net prediction · Ground truth (with per-image PSNR / SSIM)</em>
</p>

### Ablation study

<p align="center">
  <img src="docs/assets/ablation_study.png" alt="Ablation PSNR and SSIM bar charts" width="800"/>
</p>

### Training convergence

<p align="center">
  <img src="docs/assets/training_curves.png" alt="Training and validation loss, PSNR, SSIM" width="900"/>
</p>

### Edge sharpness (Sobel)

SMORE’s clinical motivation includes preserving **anatomical boundaries**. Sobel magnitude maps show sharper structure in model outputs versus the degraded input.

<p align="center">
  <img src="docs/assets/edge_sharpness.png" alt="Sobel edge analysis" width="900"/>
</p>

---

## Project Structure

```
smore/
├── notebooks/
│   └── SMORE_DIP_Project.ipynb   # Full pipeline (run top-to-bottom)
├── src/smore/
│   ├── losses.py                 # CombinedLoss, SSIMLoss
│   └── splits.py                 # subject_level_split()
├── scripts/
│   ├── smoke_test.py             # CI smoke tests
│   └── set_github_about.py       # Update repo description/topics (API)
├── docs/assets/                  # Figures for README & reports
├── .github/workflows/ci.yml
├── CITATION.cff
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset

| Item | Detail |
|------|--------|
| **Source** | [IXI Brain MRI](https://brain-development.org/ixi-dataset/) via [Kaggle (CAT12/SPM preprocessed)](https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi) |
| **Files used** | `mri_IXI_480_2/wm*.nii` — skull-stripped, registered volumes |
| **Subjects in experiment** | **80** (of 397 available volumes) |
| **2D slices** | **8,552** axial slices @ 128×128 |
| **Split** | 70% train · 15% val · 15% test (5,986 / 1,282 / 1,284 slices) |

The notebook downloads ~4.2 GB automatically through `kagglehub` on first run.

---

## Model & Training

| Component | Specification |
|-----------|----------------|
| **Architecture** | 4-level U-Net, skip connections, `base_features=32` |
| **Parameters** | 7,762,465 |
| **Input** | Aliased LR slice (1×128×128) |
| **Target** | HR slice (1×128×128) |
| **Loss** | Combined 80% SSIM + 20% MSE (`pytorch_msssim`) |
| **Epochs** | 75 max · early stop patience 10 · converged at 53 |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-5) |
| **Scheduler** | ReduceLROnPlateau |
| **Batch size** | 16 |
| **Hardware** | NVIDIA Tesla T4 (Google Colab) |

---

## Quick Start

### Option A — Google Colab (recommended)

1. Upload or open `notebooks/SMORE_DIP_Project.ipynb` in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells sequentially. The dataset downloads once and is cached.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kashif-techdev/smore/blob/main/notebooks/SMORE_DIP_Project.ipynb)

### Option B — Local machine

```bash
git clone https://github.com/kashif-techdev/smore.git
cd smore

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
# Install PyTorch with CUDA: https://pytorch.org/get-started/locally/

jupyter notebook notebooks/SMORE_DIP_Project.ipynb
```

> **Note:** Training on CPU is possible but slow. A CUDA GPU is strongly recommended.

---

## Reproducibility Checklist

- [ ] GPU runtime enabled (Colab T4 or local CUDA)
- [ ] Run notebook cells in order (install → download → train → evaluate)
- [ ] Expect ~2–4 min training for 20 epochs on T4 (30 subjects)
- [ ] Outputs: `best_model.pth`, metric tables, and PNG figures

### Suggested improvements

| Change | Expected benefit |
|--------|------------------|
| Subject-level train/val/test split | Stricter generalization evaluation |
| Tune `SSIM_ALPHA` (default `0.8`) | Balance PSNR vs perceptual quality |
| 100+ epochs | Further loss reduction if not converged |
| `base_features` 64 | Greater model capacity |

---

## Relation to the SMORE Paper

| Published SMORE | This repository |
|-----------------|-----------------|
| Self-supervised anti-aliasing + SR framework | Supervised U-Net (HR targets) |
| Full SMORE network & training objective | Standard encoder–decoder U-Net + combined SSIM/MSE loss |
| Paper-specific MRI evaluation protocol | 2D axial slices, synthetic 4× aliasing |

We cite SMORE as **conceptual motivation**; this repo is a **DIP teaching implementation**, not an official reproduction.

---

## References

1. Zhao, Y., et al. (2021). *SMORE: A Self-supervised Anti-aliasing and Super-resolution Algorithm for MRI Using Deep Learning.* IEEE TMI. https://doi.org/10.1109/TMI.2020.3037187  
2. IXI Dataset. https://brain-development.org/ixi-dataset/  
3. Preprocessed IXI (Kaggle). https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi  

---

## Citation

If you use this repository, please cite our software ([`CITATION.cff`](CITATION.cff)) and the original SMORE paper (Zhao et al., IEEE TMI 2021).

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with ❤️ for Digital Image Processing · 2026</sub>
</p>
