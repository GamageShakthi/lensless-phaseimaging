# 🔬 Physics-Informed Deep Inverse Reconstruction for Lensless Phase-Mask Imaging

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Ongoing-yellow" alt="Status">
</p>

> **A student-built project exploring computational lensless imaging** — from the physics of phase-mask cameras to deep unrolled reconstruction algorithms.

---

## 📖 Table of Contents

- [What Is This Project?](#-what-is-this-project)
- [Background Theory](#-background-theory)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Quick Start](#-quick-start)
- [Pipeline Overview](#-pipeline-overview)
- [Usage Guide](#-usage-guide)
- [Tutorial Notebook](#-tutorial-notebook)
- [Results](#-results)
- [References & Learning Resources](#-references--learning-resources)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🧠 What Is This Project?

Traditional cameras use lenses to focus light onto a sensor. **Lensless cameras** replace the lens with a thin optical element — in our case, a **phase diffuser** (a piece of transparent material with a random surface pattern). This creates a seemingly random "caustic" pattern on the sensor instead of a focused image.

The key insight: **if we know exactly how the diffuser scrambles light** (its Point Spread Function / PSF), we can computationally **unscramble** the sensor measurement to recover the original scene.

This project implements the full pipeline:

```
Scene (x) ──► Phase Diffuser ──► Sensor Measurement (y) ──► Reconstruction (x̂)
                  │                        │                        │
           Forward Model (A)        y = Ax + n            Inverse Problem
```

### What I'm Learning

1. **Computational Imaging Fundamentals** — How cameras can work without lenses
2. **Inverse Problems** — Recovering signals from indirect measurements
3. **Classical Algorithms** — Wiener deconvolution, ADMM proximal splitting
4. **Deep Learning for Physics** — Unrolled optimization networks, physics-informed losses
5. **Signal Processing** — Fourier optics, convolution, deconvolution

---

## 📚 Background Theory

### The Forward Model (How a DiffuserCam Works)

A DiffuserCam captures images through a phase diffuser placed directly on the sensor. The measurement process can be modeled as:

$$y = A \cdot x + n$$

Where:
| Symbol | Meaning |
|--------|---------|
| $x$ | The original scene (what we want to recover) |
| $A$ | The system matrix (defined by the PSF of the diffuser) |
| $y$ | The raw sensor measurement (looks like random caustics) |
| $n$ | Sensor noise (Gaussian + Poisson) |

Because the diffuser's effect is **shift-invariant** (approximately), the matrix $A$ represents a **convolution** with the PSF:

$$y = \text{PSF} * x + n$$

This is great news — convolution becomes **element-wise multiplication in Fourier space**, making computation tractable!

### The Inverse Problem (Recovering the Image)

Given measurement $y$ and known PSF, we want to find $x$. This is an **ill-posed inverse problem** because:
- Noise corrupts the measurement
- The PSF may not be invertible at all frequencies
- Small errors in $y$ can cause huge errors in naive inversion

We solve this with **regularized optimization**:

$$\hat{x} = \arg\min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \cdot R(x)$$

Where $R(x)$ is a **regularizer** that encodes prior knowledge about natural images.

### Reconstruction Methods Implemented

| Method | Type | Description |
|--------|------|-------------|
| **Wiener Deconvolution** | Classical | Frequency-domain inverse filtering with noise regularization |
| **ADMM** | Classical | Alternating Direction Method of Multipliers with TV prior |
| **Unrolled ADMM Network** | Deep Learning | ADMM iterations "unrolled" into a neural network with learnable parameters |

### Loss Functions & Regularizers

| Loss | Purpose |
|------|---------|
| **MSE Loss** | Pixel-level reconstruction fidelity |
| **Total Variation (TV)** | Promotes piecewise-smooth images, removes noise |
| **Perceptual Loss (VGG)** | Matches high-level features for visual quality |
| **Combined Physics Loss** | Weighted combination of data fidelity + regularizers |

---

## 📁 Project Structure

```
lensless-phaseimaging/
│
├── README.md                     # You are here!
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── .gitignore                    # Git ignore rules
│
├── src/                          # Core source code
│   ├── __init__.py               # Package init
│   ├── forward_model.py          # PSF simulation & forward model (y = Ax + n)
│   ├── classical_recon.py        # Wiener deconvolution & ADMM
│   ├── deep_recon.py             # Unrolled deep network architecture
│   ├── losses.py                 # TV, perceptual, and combined loss functions
│   ├── utils.py                  # Visualization, I/O, metric helpers
│   └── train.py                  # Training loop for the deep network
│
├── configs/
│   └── default.yaml              # Hyperparameters & configuration
│
├── scripts/                      # Runnable scripts
│   ├── simulate_measurement.py   # Generate synthetic measurements
│   ├── reconstruct.py            # Run reconstruction on a measurement
│   └── train_network.py          # Train the unrolled network
│
├── notebooks/
│   └── tutorial.ipynb            # 📓 Step-by-step guided tutorial
│
├── data/                         # Data directory (PSFs, test images)
│   └── README.md                 # Instructions for getting data
│
├── results/                      # Output directory for reconstructions
│   └── .gitkeep
│
└── pretrained/                   # Pretrained model weights
    └── .gitkeep
```

---

## 🛠 Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, but CPU works for small images)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/lensless-phaseimaging.git
cd lensless-phaseimaging

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .

# 5. Verify installation
python -c "from src import forward_model; print('✅ Installation successful!')"
```

### Quick Dependency Check

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## 🚀 Quick Start

### 1. Generate a Synthetic Measurement

```bash
python scripts/simulate_measurement.py \
    --image data/sample_scene.png \
    --psf_type gaussian_mixture \
    --noise_level 0.01 \
    --output results/measurement.png
```

### 2. Reconstruct with Classical Methods

```bash
# Wiener deconvolution
python scripts/reconstruct.py \
    --measurement results/measurement.png \
    --method wiener \
    --output results/wiener_recon.png

# ADMM with Total Variation
python scripts/reconstruct.py \
    --measurement results/measurement.png \
    --method admm \
    --num_iters 50 \
    --output results/admm_recon.png
```

### 3. Train the Deep Unrolled Network

```bash
python scripts/train_network.py \
    --config configs/default.yaml \
    --epochs 50 \
    --batch_size 4
```

### 4. Reconstruct with the Trained Network

```bash
python scripts/reconstruct.py \
    --measurement results/measurement.png \
    --method learned \
    --checkpoint pretrained/best_model.pth \
    --output results/deep_recon.png
```

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORWARD MODEL                                │
│                                                                     │
│   Scene Image (x)                                                   │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────────┐     ┌──────────┐     ┌───────────────────┐       │
│   │ FFT(x)      │────►│ H·X(f)   │────►│ IFFT + noise      │       │
│   │             │     │          │     │ y = PSF*x + n     │       │
│   └─────────────┘     └──────────┘     └───────────────────┘       │
│                                              │                      │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RECONSTRUCTION                                  │
│                                                                     │
│   Measurement (y)                                                   │
│       │                                                             │
│       ├──► Wiener Filter ──────────────────────────► x̂_wiener      │
│       │                                                             │
│       ├──► ADMM (TV prior) ────────────────────────► x̂_admm       │
│       │     └── x-update (FFT solve)                                │
│       │     └── z-update (TV prox / shrinkage)                      │
│       │     └── u-update (dual variable)                            │
│       │                                                             │
│       └──► Unrolled Deep Network ──────────────────► x̂_deep       │
│             └── K learned ADMM stages                               │
│             └── CNN denoiser replaces TV prox                       │
│             └── Learnable step sizes (ρ, λ)                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📓 Tutorial Notebook

The **[tutorial notebook](notebooks/tutorial.ipynb)** walks you through everything step by step:

1. **Part 1: Understanding the Physics** — What is a PSF? How does convolution model imaging?
2. **Part 2: Building the Forward Model** — Simulate a DiffuserCam measurement
3. **Part 3: Classical Reconstruction** — Implement and compare Wiener & ADMM
4. **Part 4: Going Deep** — Build and train an unrolled reconstruction network
5. **Part 5: Loss Engineering** — Explore TV, perceptual, and combined losses
6. **Part 6: Putting It All Together** — Full pipeline comparison

Open it with:
```bash
jupyter notebook notebooks/tutorial.ipynb
```

---

## 📊 Results

After running the pipeline, you can expect results like:

| Method | PSNR (dB) ↑ | SSIM ↑ | Time (s) ↓ |
|--------|-------------|--------|------------|
| Wiener Deconvolution | ~20-24 | ~0.60-0.70 | <0.1 |
| ADMM (50 iters) | ~25-28 | ~0.75-0.85 | ~2-5 |
| Unrolled Network (10 stages) | ~28-32 | ~0.85-0.92 | ~0.05 |

> **Note:** Results vary significantly based on noise level, PSF quality, and scene complexity. These are approximate ranges for well-conditioned setups.

---

## 📚 References & Learning Resources

### Papers
1. **Antipa, N. et al.** "DiffuserCam: Lensless Single-exposure 3D Imaging." *Optica*, 2018. [Paper](https://doi.org/10.1364/OPTICA.5.000001)
2. **Monakhova, K. et al.** "Learned Reconstructions for Practical Mask-Based Lensless Imaging." *Optics Express*, 2019.
3. **Boyd, S. et al.** "Distributed Optimization and Statistical Learning via ADMM." *Foundations and Trends in ML*, 2011.
4. **Monga, V. et al.** "Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing." *IEEE SPM*, 2021.

### Tutorials & Courses
- [Stanford EE367 - Computational Imaging](https://stanford.edu/class/ee367/)
- [Waller Lab DiffuserCam Tutorial](https://waller-lab.github.io/DiffuserCam/)
- [Deep Learning for Inverse Problems (Ongie et al.)](https://arxiv.org/abs/2003.02693)

### Code References
- [DiffuserCam MATLAB Code](https://github.com/Waller-Lab/DiffuserCam)
- [LearnedSensing PyTorch](https://github.com/Waller-Lab/LearnedSensing)

---

## 🙏 Acknowledgments

This project is built as a **learning exercise** inspired by the incredible work of:
- The [Waller Lab](https://waller-lab.github.io/) at UC Berkeley for the DiffuserCam concept
- The computational imaging community for open-source tools and educational resources
- The PyTorch team for making deep learning accessible

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built with curiosity and PyTorch 🔥</i>
</p>
