Got everything I need. Writing the README now.

---

# Gene Expression-Based Cancer Type Classification Using Deep Learning

**CS 697 / AI 687-001 BK | Spring 2026 | Long Island University**

**Team:** Nihanth Naidu K, Likitha P, Shreya V

**Repository:** https://github.com/LikithaPulugari09/Gene-expression-cancer-classification-using-deep-learning.git

---

## Overview

This project develops a 1D Convolutional Neural Network (1D-CNN) to classify 33 cancer types from RNA-Seq gene expression data using the TCGA Pan-Cancer dataset. The model reads 20,530 gene expression values per patient sample and produces a cancer type prediction in under 5 seconds.

The primary goal is to match or exceed the 95% accuracy benchmark established by Mostavi et al. (2020) while addressing three gaps their paper left open: class imbalance handling, model interpretability through gene-level importance scores, and full reproducibility with public code.

**Current mid-week status: 86.4% test accuracy | 0.834 Macro F1-Score | 33-class classification**

---

## Results Summary

| Metric | Value |
|---|---|
| Test Accuracy | 86.4% |
| Macro F1-Score | 0.834 |
| Average ROC AUC | 0.91 |
| Cancer Types | 33 |
| Total Samples | 10,459 |
| Training Epochs | 123 (early stop) |
| Best Checkpoint Epoch | 98 (val acc 85.4%) |
| Training Time (CPU) | ~2.1 hours |

---

## Repository Structure

```
.
├── models/
│   └── best_1dcnn_model.pt         # Best model checkpoint (epoch 98)
├── results/
│   ├── class_distribution.png      # Cancer type sample distribution chart
│   ├── confusion_matrix.png        # 33x33 confusion matrix on test set
│   ├── evaluation_metrics.txt      # Full per-class precision, recall, F1
│   ├── f1_scores_per_class.png     # Per-class F1-score bar chart
│   ├── gene_importance_scores.csv  # Saliency-based gene attribution scores
│   ├── roc_curves.png              # One-vs-rest ROC curves for all 33 types
│   ├── training_curves.png         # Accuracy and loss training history plots
│   └── training_history.csv        # Epoch-by-epoch training log
│   └── training_summary.txt        # Final training summary report
├── Data/
│   └── data.h5                     # HDF5 dataset (generated via preprocessing notebook)
├── notebooks/
│   ├── tcga_preprocess.ipynb       # Downloads raw data and builds data.h5
│   ├── explore.ipynb               # EDA, class distribution, t-SNE visualization
│   └── dimred_viz.ipynb            # Dimensionality reduction visualizations
├── labelMapping.py                 # Cancer type label to index mapping
├── plot_class_distribution.py      # Plots class distribution chart
├── plot_f1_scores.py               # Plots per-class F1 bar chart
├── plot_training_curves.py         # Plots training and validation curves
├── train.py                        # Main training script (~600 lines)
├── requirements.txt                # All dependencies with version pins
└── .gitattributes
```

---

## Dataset

**Source:** The Cancer Genome Atlas (TCGA), via UCSC Xena Hub

**Specs:**

- 10,459 patient tumor samples
- 20,530 RNA-Seq gene expression features per sample (HiSeqV2 normalized)
- 33 cancer types
- File format: HDF5 (`data.h5`, ~2 GB)
- Class imbalance ratio: 27:1 (largest: BRCA 1,218 samples | smallest: CHOL 45 samples)

**The `data.h5` file is not committed to the repository due to its size. Generate it by running the preprocessing notebook as described below.**

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LikithaPulugari09/Gene-expression-cancer-classification-using-deep-learning.git
cd Gene-expression-cancer-classification-using-deep-learning
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note on MulticoreTSNE:** This package requires a C compiler. On Windows, install Visual Studio Build Tools before running pip install. If the install fails, comment out the `MulticoreTSNE` line in `requirements.txt`. It is only needed for the t-SNE visualization in `notebooks/explore.ipynb` and is not required for training.

**Note on GPU:** Training runs on CPU by default. For GPU-accelerated training, install the correct CUDA-compatible PyTorch build for your system from https://pytorch.org/get-started/locally/ before running pip install on the rest of the requirements.

---

## Data Preparation

The raw TCGA data must be downloaded and converted to HDF5 format before training. This is a one-time setup step.

### Step 1: Open the preprocessing notebook

```bash
jupyter notebook notebooks/tcga_preprocess.ipynb
```

### Step 2: Run all cells in order

The notebook will:

1. Download the HiSeqV2 RNA-Seq expression file from the UCSC Xena legacy hub
2. Download the TCGA phenotype labels file from the UCSC Xena pan-cancer atlas hub
3. Parse and align expression data with cancer type labels
4. Map all 33 cancer type strings to integer indices 0 through 32
5. Write the processed dataset to `Data/data.h5`

**Expected output:** `Data/data.h5` approximately 2 GB in size, containing 10,459 samples with 20,530 features each.

**Download sources used inside the notebook:**

```
https://pancanatlas.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz
https://legacy.xenahubs.net/download/TCGA.PANCAN.sampleMap/HiSeqV2.gz
```

Both files are downloaded automatically when you run the notebook. An internet connection is required. The downloads are large and may take several minutes depending on your connection speed.

---

## Training

Once `Data/data.h5` is ready, run the main training script:

```bash
python train.py
```

The script handles the full pipeline automatically:

1. Loads and validates the HDF5 dataset
2. Applies log2(x+1) transformation to normalize gene expression values
3. Runs stratified 70/15/15 train/validation/test split across all 33 classes
4. Fits StandardScaler on the training set and applies it to all splits
5. Computes inverse-frequency class weights to address the 27:1 imbalance
6. Builds and trains the 1D-CNN model
7. Saves the best model checkpoint to `models/best_1dcnn_model.pt`
8. Writes training history to `results/training_history.csv`
9. Generates evaluation metrics and saves all output plots to `results/`

**Expected training time:** approximately 2 hours on CPU. GPU training will be significantly faster.

---

## Model Architecture

The 1D-CNN treats each patient's 20,530 gene expression values as a sequential input, similar to how NLP models process word sequences.

```
Input: (batch_size, 20530, 1)

Convolutional Block 1:  Conv1D(64,  kernel=3) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.5)
Convolutional Block 2:  Conv1D(128, kernel=3) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.5)
Convolutional Block 3:  Conv1D(256, kernel=3) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.5)

Global Average Pooling
Dense(512) -> ReLU -> Dropout(0.5)
Dense(256) -> ReLU -> Dropout(0.5)
Dense(33)  -> Softmax

Total Trainable Parameters: 280,865
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Loss Function | Categorical Cross-Entropy (inverse-frequency weighted) |
| Batch Size | 64 |
| Max Epochs | 200 (early stopping applies) |
| Early Stopping Patience | 25 epochs |
| Dropout Rate | 0.5 |
| Random Seed | Fixed (fully deterministic) |

**Class imbalance handling:** Inverse-frequency weighted loss assigns each class a weight proportional to the inverse of its sample count. This forces the model to treat all 33 cancer types equally during training without discarding any samples. This single change improved accuracy from 20.2% (unweighted) to 86.4% (weighted).

---

## Evaluation and Visualization

After training completes, generate the evaluation plots individually if needed:

```bash
python plot_training_curves.py
python plot_class_distribution.py
python plot_f1_scores.py
```

All output files are saved to the `results/` directory automatically.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/tcga_preprocess.ipynb` | Downloads raw TCGA data and generates `data.h5`. Run this first. |
| `notebooks/explore.ipynb` | Exploratory data analysis. Visualizes class distribution and t-SNE embeddings of gene expression. |
| `notebooks/dimred_viz.ipynb` | Dimensionality reduction visualizations. PCA and t-SNE plots for the gene expression space. |

---

## Gene Importance and Biomarker Discovery

Gene importance scores are extracted using saliency-based attribution during evaluation. For each test sample, the gradient of the predicted class score with respect to each input gene is computed. Genes with high gradient magnitude contributed most to the model's decision. Scores are averaged across all samples of the same cancer type to produce a ranked gene list per cancer type.

Results are saved to `results/gene_importance_scores.csv`.

**Note:** Gene indices in the current output correspond to their column position in the HiSeqV2 matrix. Mapping to official HGNC gene symbols is in progress and will be included in the final submission.

---

## Requirements

```
numpy>=1.21.0
torch>=2.0.0
h5py>=3.8.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
pandas>=1.5.0
seaborn>=0.12.0
progressbar2>=4.2.0
plotly>=5.0.0
MulticoreTSNE>=0.1
```

**Python version:** 3.9 or higher recommended.

---

## Reproducibility

All random seeds are fixed across NumPy, PyTorch, and Python's built-in random module. Deterministic operations are enabled in PyTorch. Running `python train.py` on the same dataset will produce identical results across runs.

---

## Cancer Type Label Mapping

All 33 cancer types and their integer label mappings are defined in `labelMapping.py`. The mapping used during preprocessing and training is:

| Label | Cancer Type | TCGA Code |
|---|---|---|
| 0 | Skin Cutaneous Melanoma | SKCM |
| 1 | Thyroid Carcinoma | THCA |
| 2 | Sarcoma | SARC |
| 3 | Prostate Adenocarcinoma | PRAD |
| 4 | Pheochromocytoma and Paraganglioma | PCPG |
| 5 | Pancreatic Adenocarcinoma | PAAD |
| 6 | Head and Neck Squamous Cell Carcinoma | HNSC |
| 7 | Esophageal Carcinoma | ESCA |
| 8 | Colon Adenocarcinoma | COAD |
| 9 | Cervical and Endocervical Cancer | CESC |
| 10 | Breast Invasive Carcinoma | BRCA |
| 11 | Bladder Urothelial Carcinoma | BLCA |
| 12 | Testicular Germ Cell Tumor | TGCT |
| 13 | Kidney Papillary Cell Carcinoma | KIRP |
| 14 | Kidney Clear Cell Carcinoma | KIRC |
| 15 | Acute Myeloid Leukemia | LAML |
| 16 | Rectum Adenocarcinoma | READ |
| 17 | Ovarian Serous Cystadenocarcinoma | OV |
| 18 | Lung Adenocarcinoma | LUAD |
| 19 | Liver Hepatocellular Carcinoma | LIHC |
| 20 | Uterine Corpus Endometrioid Carcinoma | UCEC |
| 21 | Glioblastoma Multiforme | GBM |
| 22 | Brain Lower Grade Glioma | LGG |
| 23 | Uterine Carcinosarcoma | UCS |
| 24 | Thymoma | THYM |
| 25 | Stomach Adenocarcinoma | STAD |
| 26 | Diffuse Large B-Cell Lymphoma | DLBC |
| 27 | Lung Squamous Cell Carcinoma | LUSC |
| 28 | Mesothelioma | MESO |
| 29 | Kidney Chromophobe | KICH |
| 30 | Uveal Melanoma | UVM |
| 31 | Cholangiocarcinoma | CHOL |
| 32 | Adrenocortical Cancer | ACC |

---

## References

1. Mostavi, M., Chung, Y., Khalili, M., et al. (2020). Ensembles of Deep LSTM and Convolutional Neural Networks for classifying TCGA cancer types. Computational Biology and Chemistry, Elsevier.
2. The Cancer Genome Atlas Research Network. (2013). The Cancer Genome Atlas Pan-Cancer Analysis Project. Nature Genetics, 45, 1113-1120.
3. Lopez-Garcia, G., et al. (2020). Application of Machine Learning Methods for Cancer Classification Using Gene Expression Data. IEEE Access.
4. Sun, Y., et al. (2019). A deep learning-based method for cancer classification using gene expression data. Briefings in Bioinformatics.
5. Simonyan, K., Vedaldi, A., and Zisserman, A. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. ICLR Workshop.

---

## Course Information

**Course:** CS 697 / AI 687-001 BK | Bioinformatics and Artificial Intelligence
**Semester:** Spring 2026
**Institution:** Long Island University, Brooklyn
**Team:** Nihanth Naidu K, Likitha P, Shreya V
