# Gene Expression-Based Cancer Type Classification Using Deep Learning

**CS 697 / AI 687-001 BK | Spring 2026 | Long Island University, Brooklyn**

**Team:** Nihanth Naidu K, Likitha P, Shreya V


## Overview

This project develops a **1D Convolutional Neural Network (1D-CNN) ensemble** to classify **32 cancer types** from RNA-Seq gene expression data using The Cancer Genome Atlas (TCGA) Pan-Cancer dataset. The model ingests 20,530 gene expression values per patient sample and outputs a cancer-type prediction.

We address three gaps left open by Mostavi et al. (2020):

1. **Class imbalance handling** — inverse-frequency weighted focal loss + SMOTE oversampling for minority classes
2. **Model interpretability** — saliency-based gene importance scores mapped per cancer type, validated against known biomarkers
3. **Full reproducibility** — deterministic seeds, public code, and automated data pipeline

### Key Design Decisions

- **READ/COAD Merge:** Rectum adenocarcinoma (READ) and colon adenocarcinoma (COAD) are merged into a single "Colorectal Cancer (CRC)" class. This is scientifically justified — both originate from the same tissue and share a molecular subtype (Hoadley et al. 2018, Cell). This reduces classes from 33 → 32 and eliminates the most-confused pair.
- **5-Model Ensemble:** Five independently-seeded models are trained and combined via Nelder-Mead optimized weights on the validation set.
- **Test-Time Augmentation (TTA):** Each model generates 10 augmented predictions per sample (Gaussian noise injection), yielding 50 total predictions per patient that are averaged for the final decision.

---

## Final Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.33%** |
| **Macro F1-Score** | **0.9250** |
| **Mean ROC AUC** | **0.9957** |
| Cancer Types Classified | 32 (READ merged into COAD) |
| Total Samples | 10,459 |
| Ensemble Size | 5 models × 10 TTA passes = 50 predictions/sample |
| Total Training Time (GPU) | ~67 minutes |

### Per-Class Performance Highlights

- **7 classes achieve perfect F1 = 1.000:** THCA, PRAD, PCPG, LAML, LUAD, STAD, CHOL
- **25 of 32 classes achieve F1 ≥ 0.85** (excellent classification)
- **Hardest classes:** ACC (F1=0.55), UVM (F1=0.72), ESCA (F1=0.77) — all rare cancer types with <100 samples

---

## Architecture Diagram

![Model Architecture](code%20files/plots/architecture_diagram.png)

The 1D-CNN treats each patient's 20,530 gene expression values as a sequential signal. The architecture uses four convolutional blocks with Squeeze-and-Excitation (SE) attention, followed by dual pooling (average + max) and a three-layer classifier.

```
Input: (batch_size, 20530) → unsqueeze → (batch_size, 1, 20530)

Block 1: Conv1D(1→64,   k=7, pad=3) → BN → ReLU → MaxPool(4) → SE(64)
Block 2: Conv1D(64→128,  k=5, pad=2) → BN → ReLU → MaxPool(4) → SE(128)
Block 3: Conv1D(128→256, k=3, pad=1) → BN → ReLU → MaxPool(2) → SE(256)
Block 4: Conv1D(256→512, k=3, pad=1) → BN → ReLU → SE(512)

Dual Pooling: AdaptiveAvgPool1d(1) ⊕ AdaptiveMaxPool1d(1) → (batch_size, 1024)

Classifier:
  Dense(1024→512) → ReLU → Dropout(0.5)
  Dense(512→256)  → ReLU → Dropout(0.3)
  Dense(256→32)   → Softmax

Total Trainable Parameters: ~480K
```

---

## Pipeline Workflow

![Pipeline Workflow](code%20files/plots/pipeline_workflow.png)

The full pipeline from raw data to final predictions:

1. **Data Download** → UCSC Xena Hub (HiSeqV2 RNA-Seq + phenotype labels)
2. **Preprocessing** → log2(x+1) transform, StandardScaler normalization, stratified split (70/15/15)
3. **Class Balancing** → SMOTE oversampling for classes with <400 samples
4. **Training** → 5 models trained independently with MixUp augmentation + Gaussian noise
5. **Ensemble Fusion** → Nelder-Mead weight optimization on validation set
6. **Calibration** → Temperature scaling for probability calibration
7. **Evaluation** → Confusion matrix, ROC curves, F1 scores, gene importance

---

## Visual Results

### Training Curves

![Training Curves](code%20files/plots/training_curves.png)

Training and validation accuracy/loss averaged across all 5 ensemble models. Individual seed curves shown as thin background lines. The cosine annealing warm restart schedule produces periodic learning rate resets visible as loss spikes.

### Confusion Matrix

![Confusion Matrix](code%20files/plots/confusion_matrix_final_merged.png)

Row-normalized confusion matrix on the held-out test set (15% of data, never seen during training). Most classes cluster strongly on the diagonal (high recall). The primary off-diagonal confusions occur between biologically similar cancer pairs.

### Per-Class F1-Scores

![F1 Scores](code%20files/plots/f1_scores_per_class.png)

Per-class F1-scores sorted by performance. Green bars indicate excellent classification (F1 ≥ 0.85), orange indicates moderate (0.70–0.85), and red indicates classes needing improvement (F1 < 0.70).

### ROC Curves

![ROC Curves](code%20files/plots/roc_curves.png)

One-vs-rest ROC curves for all 32 cancer types. Mean AUC = 0.9957, indicating near-perfect discrimination for the vast majority of classes.

### Biomarker Importance

![Biomarker Importance](code%20files/plots/biomarker_importance.png)

Top gene importance scores computed via saliency-based gradient attribution. Known biomarkers (validated against published literature from Mostavi et al. and Ramirez et al.) are highlighted, confirming the model learns biologically meaningful features.

---

## Repository Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files excluded from git
├── .gitattributes                      # Git LFS tracking for large files
│
├── code files/
│   ├── train.py                        # Main training script (ensemble + evaluation)
│   ├── plot_results.py                 # Generate all visualization plots
│   ├── Data/
│   │   └── data.h5                     # HDF5 dataset (generated via preprocessing)
│   ├── models/
│   │   ├── best_model_seed42.pt        # Ensemble model 1
│   │   ├── best_model_seed123.pt       # Ensemble model 2
│   │   ├── best_model_seed456.pt       # Ensemble model 3
│   │   ├── best_model_seed789.pt       # Ensemble model 4
│   │   └── best_model_seed999.pt       # Ensemble model 5
│   ├── results/
│   │   ├── metrics_final_merged.txt    # Final per-class metrics
│   │   ├── final_summary.txt           # Training run summary
│   │   ├── gene_importance_scores.csv  # Top 50 global gene attributions
│   │   ├── gene_importance_per_cancer.csv  # Top 10 genes per cancer type
│   │   ├── confusion_matrix_final_merged.png
│   │   ├── history_seed{42,123,456,789,999}.csv  # Per-epoch training logs
│   │   └── metrics_seed{42,123,456,789,999}.txt  # Per-model test metrics
│   └── plots/
│       ├── architecture_diagram.png    # CNN architecture visualization
│       ├── pipeline_workflow.png       # End-to-end pipeline diagram
│       ├── training_curves.png         # Accuracy & loss over epochs
│       ├── confusion_matrix_final_merged.png  # Final confusion matrix
│       ├── f1_scores_per_class.png     # Per-class F1 bar chart
│       ├── roc_curves.png              # One-vs-rest ROC curves
│       ├── class_distribution.png      # Sample count per cancer type
│       └── biomarker_importance.png    # Gene importance visualization
│
├── notebooks/
│   └── explore.ipynb                   # EDA: class distribution, t-SNE visualization
│
└── docs/                               # (optional) Project reports and presentations
    ├── TCGA_Full_Project_Report.docx
    ├── TCGA_Final_Presentation.pptx
    └── TCGA_Professor_QA_Guide.docx
```

---

## Dataset

**Source:** The Cancer Genome Atlas (TCGA) via UCSC Xena Hub

| Property | Value |
|----------|-------|
| Total patient samples | 10,459 |
| Features per sample | 20,530 (RNA-Seq gene expression, HiSeqV2 normalized) |
| Cancer types | 32 (after READ/COAD merge) |
| File format | HDF5 (`data.h5`, ~2 GB) |
| Class imbalance ratio | ~27:1 (BRCA: 1,218 samples vs CHOL: 45 samples) |

**The `data.h5` file is NOT committed to the repository due to its size (~2 GB). You must generate it by running the preprocessing notebook as described below.**

### Download Sources (used inside preprocessing notebook)

| File | URL |
|------|-----|
| Gene expression matrix | `https://legacy.xenahubs.net/download/TCGA.PANCAN.sampleMap/HiSeqV2.gz` |
| Phenotype labels | `https://pancanatlas.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz` |

---

## Setup Instructions

### Prerequisites

- **Python 3.9+** (tested with 3.10 and 3.11)
- **pip** (package manager)
- **~4 GB free disk space** (for dataset + model checkpoints)
- **(Optional) NVIDIA GPU** with CUDA support for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/LikithaPulugari09/Gene-expression-cancer-classification-using-deep-learning.git
cd Gene-expression-cancer-classification-using-deep-learning
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**GPU Users:** If you have an NVIDIA GPU, install the CUDA-compatible PyTorch build first:

```bash
# For CUDA 12.8 (check your CUDA version with: nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install the rest
pip install -r requirements.txt
```

**Note on MulticoreTSNE:** This optional package (used only in `notebooks/explore.ipynb` for t-SNE plots) requires a C++ compiler. On Windows, install Visual Studio Build Tools first. If installation fails, skip it — the main training pipeline does not depend on it.

### Step 4: Generate the Dataset

The raw TCGA data must be downloaded and converted to HDF5 format. This is a **one-time** step.

```bash
jupyter notebook notebooks/explore.ipynb
```

Run all cells in order. The notebook will:

1. Download HiSeqV2 RNA-Seq expression data from UCSC Xena (~700 MB compressed)
2. Download TCGA phenotype labels file
3. Parse and align expression data with cancer type labels
4. Map disease names to integer class indices
5. Save the processed dataset to `code files/Data/data.h5`

**Expected output:** `code files/Data/data.h5` (~2 GB), containing 10,459 samples × 20,530 genes.

An internet connection is required. Downloads may take several minutes.

---

## Training

Once `data.h5` exists, run the full training pipeline:

```bash
cd "code files"
python train.py
```

### What `train.py` Does (Automated Pipeline)

1. **Loads and validates** the HDF5 dataset, prints data quality checks (NaN/Inf counts)
2. **Merges READ into COAD** → 32 contiguous class labels
3. **Applies log2(x+1) normalization** to gene expression values
4. **Stratified split** → 70% train / 15% validation / 15% test (all 32 classes represented in each split)
5. **Fits StandardScaler** on training data, transforms all splits
6. **SMOTE oversampling** → minority classes with <400 samples are upsampled to 400
7. **Trains 5 models** independently (seeds: 42, 123, 456, 789, 999) with:
   - MixUp data augmentation (α=0.2)
   - Gaussian noise injection (σ=0.05)
   - Focal loss with label smoothing (0.1)
   - CosineAnnealingWarmRestarts scheduler
   - Early stopping (patience=40 epochs)
8. **Collects predictions** via Test-Time Augmentation (10 noisy passes per model)
9. **Optimizes ensemble weights** using Nelder-Mead on validation cross-entropy
10. **Applies temperature scaling** for probability calibration
11. **Evaluates** on held-out test set → confusion matrix, per-class F1, ROC AUC
12. **Computes gene importance** via saliency-based gradient attribution
13. **Saves everything** to `models/` and `results/`

### Expected Training Time

| Hardware | Approximate Time |
|----------|-----------------|
| NVIDIA GPU (RTX 3060+) | ~67 minutes |
| CPU only | ~4–6 hours |

### Output Files Generated

After training completes, the following files are created:

```
models/best_model_seed{42,123,456,789,999}.pt  — 5 model checkpoints
results/final_summary.txt                       — Full training report
results/metrics_final_merged.txt                — Per-class precision/recall/F1
results/confusion_matrix_final_merged.png       — 32×32 confusion matrix
results/gene_importance_scores.csv              — Top 50 genes globally
results/gene_importance_per_cancer.csv          — Top 10 genes per cancer type
results/history_seed*.csv                       — Epoch-by-epoch training logs
results/final_probabilities.npy                 — Test set probability matrix
results/optimal_weights.npy                     — Ensemble weight vector
```

---

## Generating Plots

After training (or if results already exist), generate publication-quality visualizations:

```bash
cd "code files"

# Generate all three figures:
python plot_results.py

# Or generate individually:
python plot_results.py f1        # Per-class F1-score bar chart
python plot_results.py curves    # Training & validation curves
python plot_results.py dist      # Class distribution chart
```

All plots are saved to `code files/plots/` at 300 DPI.

---

## Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | AdamW | Better generalization than Adam via decoupled weight decay |
| Initial Learning Rate | 0.001 | Standard for CNN training |
| LR Scheduler | CosineAnnealingWarmRestarts (T₀=40, T_mult=2) | Periodic restarts escape local minima |
| Loss Function | Focal Loss + MixUp soft targets | Handles class imbalance + augmentation |
| Label Smoothing | 0.1 | Reduces overconfidence |
| Batch Size | 256 | Maximizes GPU utilization |
| Max Epochs | 200 | Early stopping triggers well before |
| Early Stopping Patience | 40 epochs | Accounts for cosine restart cycles |
| Dropout | 0.5 (layer 1), 0.3 (layer 2) | Progressive regularization |
| MixUp Alpha | 0.2 | Moderate interpolation |
| Gaussian Noise (train) | σ = 0.05 | Input-level regularization |
| TTA Noise (inference) | σ = 0.02 | Gentler noise for test-time diversity |
| SMOTE Target | 400 samples | Brings minority classes to viable count |
| Weight Decay | 1e-4 | L2 regularization |
| Gradient Clipping | max_norm = 1.0 | Stabilizes training |

---

## Gene Importance and Biomarker Discovery

Gene importance scores are computed using **saliency-based gradient attribution**:

1. For each test sample, compute the gradient of the predicted class probability with respect to the input gene expression vector
2. Take the absolute value of each gradient (magnitude = importance)
3. Average across all test samples of the same cancer type
4. Rank genes by their global and per-class importance scores

### Biomarker Validation

The model's top-ranked genes are cross-referenced against known cancer biomarkers from published literature:

| Gene | Cancer Type | Source Paper | Model Detected? |
|------|------------|--------------|-----------------|
| GATA3 | BRCA (Breast) | Mostavi 2020 | ✓ |
| KLK3 / AR | PRAD (Prostate) | Mostavi 2020 | ✓ |
| TTF1 / NKX2-1 | LUAD (Lung Adeno) | Mostavi 2020 | ✓ |
| VHL / PBRM1 | KIRC (Kidney Clear Cell) | Ramirez 2020 | ✓ |
| IDH1 / IDH2 | LGG (Brain Lower Grade Glioma) | Mostavi 2020 | ✓ |
| KRAS / APC | CRC (Colorectal) | Ramirez 2020 | ✓ |
| FLT3 / NPM1 | LAML (Leukemia) | Ramirez 2020 | ✓ |
| BRAF | THCA (Thyroid) | Mostavi 2020 | ✓ |

Results are saved to:
- `results/gene_importance_scores.csv` — Global top 50 genes
- `results/gene_importance_per_cancer.csv` — Top 10 genes per cancer type

---

## Cancer Type Label Mapping (32 Classes)

| Index | TCGA Code | Cancer Type |
|-------|-----------|-------------|
| 0 | SKCM | Skin Cutaneous Melanoma |
| 1 | THCA | Thyroid Carcinoma |
| 2 | SARC | Sarcoma |
| 3 | PRAD | Prostate Adenocarcinoma |
| 4 | PCPG | Pheochromocytoma & Paraganglioma |
| 5 | PAAD | Pancreatic Adenocarcinoma |
| 6 | HNSC | Head & Neck Squamous Cell Carcinoma |
| 7 | ESCA | Esophageal Carcinoma |
| 8 | **CRC** | **Colorectal Cancer (COAD + READ merged)** |
| 9 | CESC | Cervical & Endocervical Cancer |
| 10 | BRCA | Breast Invasive Carcinoma |
| 11 | BLCA | Bladder Urothelial Carcinoma |
| 12 | TGCT | Testicular Germ Cell Tumor |
| 13 | KIRP | Kidney Papillary Cell Carcinoma |
| 14 | KIRC | Kidney Clear Cell Carcinoma |
| 15 | LAML | Acute Myeloid Leukemia |
| 16 | OV | Ovarian Serous Cystadenocarcinoma |
| 17 | LUAD | Lung Adenocarcinoma |
| 18 | LIHC | Liver Hepatocellular Carcinoma |
| 19 | UCEC | Uterine Corpus Endometrioid Carcinoma |
| 20 | GBM | Glioblastoma Multiforme |
| 21 | LGG | Brain Lower Grade Glioma |
| 22 | UCS | Uterine Carcinosarcoma |
| 23 | THYM | Thymoma |
| 24 | STAD | Stomach Adenocarcinoma |
| 25 | DLBC | Diffuse Large B-Cell Lymphoma |
| 26 | LUSC | Lung Squamous Cell Carcinoma |
| 27 | MESO | Mesothelioma |
| 28 | KICH | Kidney Chromophobe |
| 29 | UVM | Uveal Melanoma |
| 30 | CHOL | Cholangiocarcinoma |
| 31 | ACC | Adrenocortical Cancer |

---

## Reproducibility

All random seeds are fixed across:
- Python's built-in `random` module
- NumPy
- PyTorch (CPU + CUDA)
- cuDNN (deterministic mode enabled, benchmark disabled)

Running `python train.py` on the same dataset with the same hardware produces identical results across runs.

**Seeds used:** 42, 123, 456, 789, 999 (one per ensemble member)

---

## Requirements

```
numpy>=1.21.0
torch>=2.0.0
h5py>=3.8.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
scipy>=1.10.0
pandas>=1.5.0
seaborn>=0.12.0
progressbar2>=4.2.0
plotly>=5.0.0
imbalanced-learn>=0.10.1
```

**Optional:** `MulticoreTSNE>=0.1` (only for t-SNE visualization in notebooks, requires C++ build tools)

**Python version:** 3.9 or higher recommended (tested on 3.10, 3.11)

---

## Quick Start (TL;DR)

```bash
# 1. Clone
git clone https://github.com/LikithaPulugari09/Gene-expression-cancer-classification-using-deep-learning.git
cd Gene-expression-cancer-classification-using-deep-learning

# 2. Setup environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Generate dataset (one-time, requires internet)
jupyter notebook notebooks/explore.ipynb
# Run all cells → produces code files/Data/data.h5

# 4. Train the model
cd "code files"
python train.py

# 5. Generate plots
python plot_results.py
```

---

## References

1. Mostavi, M., Chung, Y., Khalili, M., et al. (2020). *Ensembles of Deep LSTM and Convolutional Neural Networks for classifying TCGA cancer types.* Computational Biology and Chemistry, Elsevier.
2. The Cancer Genome Atlas Research Network. (2013). *The Cancer Genome Atlas Pan-Cancer Analysis Project.* Nature Genetics, 45, 1113–1120.
3. Hoadley, K.A., et al. (2018). *Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer.* Cell, 173(2), 291–304.
4. Ramirez, R., et al. (2020). *Classification of Cancer Types Using Graph Convolutional Neural Networks.* Frontiers in Physics, 8:203.
5. Lopez-Garcia, G., et al. (2020). *Application of Machine Learning Methods for Cancer Classification Using Gene Expression Data.* IEEE Access.
6. Sun, Y., et al. (2019). *A deep learning-based method for cancer classification using gene expression data.* Briefings in Bioinformatics.
7. Simonyan, K., Vedaldi, A., and Zisserman, A. (2014). *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.* ICLR Workshop.
8. Chawla, N.V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.

---

## Course Information

| | |
|--|--|
| **Course** | CS 697 / AI 687-001 BK — Bioinformatics and Artificial Intelligence |
| **Semester** | Spring 2026 |
| **Institution** | Long Island University, Brooklyn |
| **Team** | Nihanth Naidu K, Likitha P, Shreya V |

---

## License

This project is for academic purposes as part of the CS 697 / AI 687-001 BK coursework at Long Island University. The TCGA dataset is publicly available through the Genomic Data Commons and UCSC Xena Hub.
