# Explainable and Robust CNN-Based Brain Tumour Classification from MRI Scans

**MSc Machine Learning and Artificial Intelligence — Liverpool John Moores University**

**Author:** Eswara Venkatesh Janjanam  
**Supervisors:** Dr. Rituparna Datta and Dr. Dattatraya Parle

---

## Overview

This repository contains the source code for the MSc thesis research on evaluating CNN-based brain tumour classifiers across three pillars of trustworthy AI: **diagnostic accuracy**, **model transparency (explainability)**, and **adversarial robustness**.

Three pre-trained CNN architectures — **ResNet-50**, **EfficientNet-B0**, and **DenseNet201** — are evaluated on the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) across four tumour classes: glioma, meningioma, pituitary tumour, and no tumour.

---

## Repository Structure

```
├── 01_EDA_Brain_Tumor_MRI.ipynb                    # Exploratory Data Analysis notebook
├── 02_CNN_Training_Explainability_Robustness.ipynb  # Main training, evaluation & comparison notebook
├── results/                                         # Saved model trainin history, results, and visualisations
│   ├── all_results.json              # Full results across all architectures and attack conditions
│   ├── gradcam_*.png                 # Grad-CAM visualisations per architecture
│   ├── shap_*.png                    # SHAP visualisations per architecture
│   ├── robustness_*.png              # Robustness bar charts per architecture
│   ├── confusion_*.png               # Confusion matrices per architecture
│   └── curves_*.png                  # Training curves per architecture
└── images/                           # Figures used in the thesis report
```

---

## Notebooks

### 1. `01_EDA_Brain_Tumor_MRI.ipynb` — Exploratory Data Analysis

Performs a comprehensive EDA on the Kaggle Brain Tumor MRI Dataset before any model training. The notebook covers:

| Section | Description |
|---------|-------------|
| 1. Class Distribution | Image counts per class across training and test splits |
| 2. Image Resolution Analysis | Resolution range (150×198 to 1,375×1,446 px), scatter plots |
| 3. Colour Mode Analysis | Grayscale vs RGB vs RGBA distribution per class |
| 4. Sample Images | Four representative images per class displayed in a grid |
| 5. Pixel Intensity Distribution | Per-class intensity histograms |
| 6. Mean Image per Class | Average image per class to reveal spatial patterns |
| 7. Aspect Ratio Distribution | Distribution of width-to-height ratios across the dataset |
| 8. Train/Test Split Summary | Pre-defined split verification (5,600 train / 1,600 test) |
| 9. Label Encoding | PyTorch ImageFolder alphabetical label mapping |
| 10. Augmentation Visualisation | Examples of all augmentation transforms applied during training |
| 11. EDA Summary Statistics | Consolidated dataset statistics |

**Key findings from EDA:**
- 447 unique image resolutions across the dataset
- 42.6% grayscale and 57.3% RGB images — RGB conversion required
- Perfectly balanced classes: 1,400 training and 400 test images per class

---

### 2. `02_CNN_Training_Explainability_Robustness.ipynb` — Full Training, Evaluation and Comparison

The main research notebook. Trains and evaluates all three architectures across two training modes and three evaluation pillars. Designed to run on **Google Colab** with GPU support.

| Section | Description |
|---------|-------------|
| 1. Install & Imports | Package installation and all library imports |
| 2. Configuration | Hyperparameters, paths, class names, attack budgets |
| 3. Data Loaders | Training (with augmentation) and test (clean) data loaders |
| 4. Attack & Metric Helpers | FGSM, PGD implementations; accuracy/precision/recall/F1 computation |
| 5. Model Builders | Architecture definitions with custom 4-class classification heads |
| 6. Model Summary & Hyperparameters | Trainable parameter counts and configuration display |
| 7. Train / Evaluate Functions | Epoch-level training with optional adversarial batch mixing |
| 8. Checkpointing & Training Runner | Progress saving to Google Drive; resume support |
| 9. Perturbation Visualisations | FGSM and PGD perturbation grids per class and epsilon |
| 10. Grad-CAM | Gradient-weighted Class Activation Mapping implementation and visualisation |
| 11. SHAP | GradientExplainer-based SHAP value computation and overlay |
| 12. Plot Helpers | Training curves, confusion matrices, robustness bar charts |
| 13. Run All Architectures | Full training and evaluation loop across all 6 models |
| 14. Cross-Architecture Summary | Comparative results table across all three pillars |
| 15. Cross-Architecture Bar Chart | Visual comparison of accuracy, F1, and robustness |
| 16. Save All Results | Export all metrics to `all_results.json` |

---

## Methodology

### Dataset

- **Source:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Nickparvar, 2024)
- **Size:** 7,200 T1-weighted contrast-enhanced MRI images
- **Classes:** Glioma, Meningioma, Pituitary Tumour, No Tumour
- **Split:** 5,600 training / 1,600 test (400 per class each)

### Training Setup

Two models are trained per architecture:

| Model | Training Strategy |
|-------|------------------|
| **Model A** (Clean) | Standard transfer learning on clean images only |
| **Model B** (Adversarial) | Adversarial batch mixing — 60% clean, 20% FGSM, 20% PGD per batch |

**Common hyperparameters:** AdamW optimiser, weight decay 1×10⁻⁴, label smoothing 0.1, batch size 64, progressive backbone unfreezing at epoch 6.

### Architectures

| Architecture | Parameters | Key Design Feature |
|---|---|---|
| ResNet-50 | 25.6M | Residual skip connections |
| EfficientNet-B0 | 5.3M | Compound scaling |
| DenseNet201 | 20.0M | Dense cross-layer connectivity |

### Explainability

- **Grad-CAM** — Class-discriminative heatmaps from the last convolutional layer
- **SHAP** — Signed pixel-level attributions via GradientExplainer (50-image background set)

### Adversarial Robustness

- **FGSM** — Single-step gradient sign attack
- **PGD** — 5-step iterative attack (step size α = 0.008)
- **Perturbation budgets:** ε = 0.01, 0.03, 0.05

---

## Results Summary

All results are stored in `results/all_results.json`.

| Architecture | Clean Accuracy | Adversarial Accuracy | Adversarial Cost |
|---|---|---|---|
| ResNet-50 | 93.88% | 84.12% | −9.76 pp |
| EfficientNet-B0 | 88.44% | 78.62% | −9.82 pp |
| DenseNet201 | 94.00% | 90.81% | −3.19 pp |

**Key finding:** Gradient masking was detected across all architectures — FGSM robustness was moderate (33–74%) while PGD accuracy dropped severely (1–47%), indicating that adversarial training produced illusory rather than genuine robustness improvements.

---

## Requirements

### Running on Google Colab (Recommended)

The main notebook (`02_CNN_Training_Explainability_Robustness.ipynb`) is designed for **Google Colab** with an NVIDIA T4 GPU. It auto-installs all dependencies in Cell 1:

```python
!pip install torch torchvision scikit-learn matplotlib seaborn Pillow shap torchinfo -q
```

The dataset is downloaded directly from Kaggle in Cell 5 (requires Kaggle API credentials).

### Running Locally

```bash
pip install torch torchvision scikit-learn matplotlib seaborn Pillow shap torchinfo
```

Ensure the dataset is placed at the path configured in Section 2 of the main notebook.

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.0+ | Model training and adversarial attacks |
| torchvision | 0.15+ | Pre-trained architectures and transforms |
| SHAP | 0.42+ | GradientExplainer for pixel-level attributions |
| scikit-learn | Latest | Classification metrics |
| Matplotlib / Seaborn | Latest | Visualisations |

---

## Dataset Access

The dataset used is publicly available on Kaggle:

> Nickparvar, M. (2024). *Brain Tumor MRI Dataset*. Kaggle.  
> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

To download via Kaggle API:
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d BrainTumorDataset
```

---

## Citation

If you use this code or findings in your work, please cite:

```
Janjanam, E.V. (2026). Explainable and Robust CNN-Based Brain Tumour Classification 
from MRI Scans. MSc Thesis, Liverpool John Moores University.
```

---

## Licence

This repository is shared for academic and research purposes in accordance with LJMU thesis submission guidelines. The dataset is subject to its own Kaggle licence terms.
