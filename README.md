# High-Accuracy Hybrid CNN Ensemble for Multi-Class Cancer Classification from Histopathology

This repository contains the official production-grade code implementation for an automated deep learning pipeline designed to classify lung and colon tissue subtypes from histopathology images. By utilizing transfer learning and optimizing a weighted soft-voting ensemble of **EfficientNetB3** and **ResNet50V2**, the framework minimizes inter-class errors across morphologically similar cancer subtypes.

---

## 📊 Performance Summary

The hybrid ensemble leverages complementary spatial and structural feature extractions, achieving a **+6% performance increment** over the baseline model.

| Model Architecture | Test Accuracy | Delta vs. Baseline | Primary Strengths |
| :--- | :---: | :---: | :--- |
| **EfficientNetB3** | 90.00% | *Baseline* | Compound scaling; exceptional multiscale texture feature extraction. |
| **ResNet50V2** | 92.20% | +2.20% | Pre-activation residual blocks; stable gradient flow for deep features. |
| **Hybrid Ensemble (0.6 / 0.4)** | **96.60%** | **+6.60%** | **Minimized cross-class variance; superior boundary discrimination.** |

### Key Classification Diagnostics[cite: 1]
* **Colon Adenocarcinoma:** Achieved a precision of **83.33%** and a recall of **100%**.
* **Malignant / Benign Bounds:** Maintained **97% overall sensitivity** for malignant categories and **96% specificity** for benign tissues.
* **Lung Subtypes:** Significantly mitigated the typical clinical misclassification loop between *Lung Adenocarcinoma* and *Lung Squamous Cell Carcinoma*.

---

## 📁 Dataset & Preprocessing Pipeline

The framework is built and validated using the benchmark **LC25000 histopathology dataset**, which features 25,000 total images evenly distributed across 5 balanced classes (5,000 RGB patches per class)[cite: 1]:
1. **Colon Adenocarcinoma** (`colon_aca`)[cite: 1]
2. **Colon Benign Tissue** (`colon_n`)[cite: 1]
3. **Lung Adenocarcinoma** (`lung_aca`)[cite: 1]
4. **Lung Benign Tissue** (`lung_n`)[cite: 1]
5. **Lung Squamous Cell Carcinoma** (`lung_scc`)[cite: 1]

### Hyperparameter & Regularization Blueprint[cite: 1]
> **Note on Regularization:** To maximize stain-invariant feature robustness and avoid overfitting without relying on pixel-level segmentation masks, the pipeline implements data augmentation directly at the generator level[cite: 1].

* **Input Target Dimensions:** $300 \times 300 \times 3$ pixels[cite: 1]
* **Intensity Scale:** Normalized to $[0, 1]$ intensity range[cite: 1]
* **Augmentations:** Random rotations, spatial flips, zoom, shear distortions, and brightness perturbations[cite: 1]
* **Data Split:** **85% Training** to **15% Validation** split adjusted at runtime[cite: 1]
* **Optimization Head:** Custom Dense configuration (Global Average Pooling $\rightarrow$ Batch Normalization $\rightarrow$ Dense 1024 $\rightarrow$ Dropout 0.4 $\rightarrow$ Dense 512 $\rightarrow$ Softmax)[cite: 1]

---

## 💻 Environment Setup & Repository Structure

```text
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Augmentation & data generation pipelines
│   ├── models.py        # Backbone definitions & unified classifier heads
│   ├── train.py         # 15-epoch Adam optimization sequence
│   └── ensemble.py      # Weighted soft-voting probability matrix evaluation
└── models/
    └── .gitkeep         # Stores trained checkpoints (*_best.h5)
