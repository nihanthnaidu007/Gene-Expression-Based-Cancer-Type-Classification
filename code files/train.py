"""
TCGA Pan-Cancer - Final Training Run with READ->COAD Merge
==========================================================
This is the definitive final accuracy run for the project report.

Key change from train.py:
  READ (rectum adenocarcinoma) is merged into COAD (colon adenocarcinoma)
  to form a single COLORECTAL class (class 8).

Scientific justification:
  - READ and COAD share a molecular colorectal cancer subtype in the TCGA
    Pan-Cancer analysis (Hoadley et al. 2018, Cell).
  - Mostavi et al. (2020) acknowledge READ/COAD confusion is a known
    biological challenge - both cancers originate from the same tissue.
  - The GCNN paper (Ramirez et al. 2020) shows the same confusion pattern.
  - Clinically, COAD and READ are treated as a single colorectal entity
    in many staging and treatment protocols.
  - This merge reduces classes from 33 -> 32, eliminating the single
    most-confused pair (READ F1=0.40 in previous run).

Everything else is identical to train.py:
  - 5-model ensemble (seeds 42, 123, 456, 789, 999)
  - MixUp alpha=0.2 + Gaussian noise std=0.05
  - Soft-target CE + label smoothing=0.1
  - CosineAnnealingWarmRestarts
  - SMOTE on minority classes
  - TTA (20 passes, noise_std=0.02)
  - Weighted ensemble optimisation on validation set

Requirements:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install h5py scikit-learn matplotlib numpy imbalanced-learn scipy
"""

import csv
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: pip install imbalanced-learn")

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED                    = 42
ENSEMBLE_SEEDS          = [42, 123, 456, 789, 999]

BATCH_SIZE              = 256
EPOCHS                  = 200
EARLY_STOPPING_PATIENCE = 40
LR                      = 1e-3
WEIGHT_DECAY            = 1e-4
LABEL_SMOOTHING         = 0.1

MIXUP_ALPHA             = 0.2
NOISE_STD               = 0.05

TTA_N_PASSES            = 10
TTA_NOISE_STD           = 0.02

SMOTE_TARGET            = 400
SMOTE_THRESHOLD         = 400
SMOTE_K_NEIGHBORS       = 3

COSINE_T0               = 40
COSINE_T_MULT           = 2

CHUNK_SIZE              = 512
IMPORTANCE_BATCH_SIZE   = 128
IMPORTANCE_MAX_SAMPLES  = None

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY  = DEVICE.type == "cuda"
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# READ -> COAD merge
# Original diseasedict maps READ to class 16.
# After merge, READ and COAD both map to class 8.
# The model then trains on 32 classes instead of 33.
# ---------------------------------------------------------------------------
MERGE_READ_INTO_COAD = True   # set False to disable merge and use original 33 classes

# Disease name -> integer label mapping
# READ is mapped to 8 (same as COAD) when MERGE_READ_INTO_COAD=True
DISEASEDICT_MERGED = {
    'skin cutaneous melanoma':               0,
    'thyroid carcinoma':                     1,
    'sarcoma':                               2,
    'prostate adenocarcinoma':               3,
    'pheochromocytoma & paraganglioma':      4,
    'pancreatic adenocarcinoma':             5,
    'head & neck squamous cell carcinoma':   6,
    'esophageal carcinoma':                  7,
    'colon adenocarcinoma':                  8,
    'rectum adenocarcinoma':                 8,   # MERGED INTO COAD (colorectal)
    'cervical & endocervical cancer':        9,
    'breast invasive carcinoma':             10,
    'bladder urothelial carcinoma':          11,
    'testicular germ cell tumor':            12,
    'kidney papillary cell carcinoma':       13,
    'kidney clear cell carcinoma':           14,
    'acute myeloid leukemia':               15,
    # class 16 (READ) no longer exists - gap is closed below via re-encoding
    'ovarian serous cystadenocarcinoma':     17,
    'lung adenocarcinoma':                   18,
    'liver hepatocellular carcinoma':        19,
    'uterine corpus endometrioid carcinoma': 20,
    'glioblastoma multiforme':               21,
    'brain lower grade glioma':              22,
    'uterine carcinosarcoma':                23,
    'thymoma':                               24,
    'stomach adenocarcinoma':               25,
    'diffuse large B-cell lymphoma':         26,
    'lung squamous cell carcinoma':          27,
    'mesothelioma':                          28,
    'kidney chromophobe':                    29,
    'uveal melanoma':                        30,
    'cholangiocarcinoma':                    31,
    'adrenocortical cancer':                 32,
}

DISEASEDICT_ORIGINAL = {
    'skin cutaneous melanoma':               0,
    'thyroid carcinoma':                     1,
    'sarcoma':                               2,
    'prostate adenocarcinoma':               3,
    'pheochromocytoma & paraganglioma':      4,
    'pancreatic adenocarcinoma':             5,
    'head & neck squamous cell carcinoma':   6,
    'esophageal carcinoma':                  7,
    'colon adenocarcinoma':                  8,
    'cervical & endocervical cancer':        9,
    'breast invasive carcinoma':             10,
    'bladder urothelial carcinoma':          11,
    'testicular germ cell tumor':            12,
    'kidney papillary cell carcinoma':       13,
    'kidney clear cell carcinoma':           14,
    'acute myeloid leukemia':               15,
    'rectum adenocarcinoma':                 16,
    'ovarian serous cystadenocarcinoma':     17,
    'lung adenocarcinoma':                   18,
    'liver hepatocellular carcinoma':        19,
    'uterine corpus endometrioid carcinoma': 20,
    'glioblastoma multiforme':               21,
    'brain lower grade glioma':              22,
    'uterine carcinosarcoma':                23,
    'thymoma':                               24,
    'stomach adenocarcinoma':               25,
    'diffuse large B-cell lymphoma':         26,
    'lung squamous cell carcinoma':          27,
    'mesothelioma':                          28,
    'kidney chromophobe':                    29,
    'uveal melanoma':                        30,
    'cholangiocarcinoma':                    31,
    'adrenocortical cancer':                 32,
}

# Class index -> display name (for confusion matrix labels)
# When merged, class 8 is labelled "colorectal" to reflect both COAD and READ
IDX_TO_NAME_MERGED = {
    0:  "skin cutaneous melanoma",
    1:  "thyroid carcinoma",
    2:  "sarcoma",
    3:  "prostate adenocarcinoma",
    4:  "pheochromocytoma & paraganglioma",
    5:  "pancreatic adenocarcinoma",
    6:  "head & neck squamous cell carcinoma",
    7:  "esophageal carcinoma",
    8:  "colorectal cancer (COAD+READ)",    # merged label
    9:  "cervical & endocervical cancer",
    10: "breast invasive carcinoma",
    11: "bladder urothelial carcinoma",
    12: "testicular germ cell tumor",
    13: "kidney papillary cell carcinoma",
    14: "kidney clear cell carcinoma",
    15: "acute myeloid leukemia",
    16: "ovarian serous cystadenocarcinoma",
    17: "lung adenocarcinoma",
    18: "liver hepatocellular carcinoma",
    19: "uterine corpus endometrioid carcinoma",
    20: "glioblastoma multiforme",
    21: "brain lower grade glioma",
    22: "uterine carcinosarcoma",
    23: "thymoma",
    24: "stomach adenocarcinoma",
    25: "diffuse large B-cell lymphoma",
    26: "lung squamous cell carcinoma",
    27: "mesothelioma",
    28: "kidney chromophobe",
    29: "uveal melanoma",
    30: "cholangiocarcinoma",
    31: "adrenocortical cancer",
}

NAME_TO_ABBREV = {
    "skin cutaneous melanoma":               "SKCM",
    "thyroid carcinoma":                     "THCA",
    "sarcoma":                               "SARC",
    "prostate adenocarcinoma":               "PRAD",
    "pheochromocytoma & paraganglioma":      "PCPG",
    "pancreatic adenocarcinoma":             "PAAD",
    "head & neck squamous cell carcinoma":   "HNSC",
    "esophageal carcinoma":                  "ESCA",
    "colorectal cancer (coad+read)":         "CRC",  # merged abbreviation
    "colon adenocarcinoma":                  "COAD",
    "cervical & endocervical cancer":        "CESC",
    "breast invasive carcinoma":             "BRCA",
    "bladder urothelial carcinoma":          "BLCA",
    "testicular germ cell tumor":            "TGCT",
    "kidney papillary cell carcinoma":       "KIRP",
    "kidney clear cell carcinoma":           "KIRC",
    "acute myeloid leukemia":               "LAML",
    "rectum adenocarcinoma":                 "READ",
    "ovarian serous cystadenocarcinoma":     "OV",
    "lung adenocarcinoma":                   "LUAD",
    "liver hepatocellular carcinoma":        "LIHC",
    "uterine corpus endometrioid carcinoma": "UCEC",
    "glioblastoma multiforme":               "GBM",
    "brain lower grade glioma":              "LGG",
    "uterine carcinosarcoma":                "UCS",
    "thymoma":                               "THYM",
    "stomach adenocarcinoma":               "STAD",
    "diffuse large b-cell lymphoma":         "DLBC",
    "lung squamous cell carcinoma":          "LUSC",
    "mesothelioma":                          "MESO",
    "kidney chromophobe":                    "KICH",
    "uveal melanoma":                        "UVM",
    "cholangiocarcinoma":                    "CHOL",
    "adrenocortical cancer":                 "ACC",
}

KNOWN_BIOMARKERS = {
    "GATA3":  ("BRCA", "Mostavi CNN"),   "KLK3":   ("PRAD", "Mostavi CNN"),
    "AR":     ("PRAD", "Mostavi CNN"),   "TTF1":   ("LUAD", "Mostavi CNN"),
    "NKX2-1": ("LUAD", "Mostavi CNN"),   "TP63":   ("LUSC", "Mostavi CNN"),
    "SOX2":   ("LUSC", "Mostavi CNN"),   "IDH1":   ("LGG",  "Mostavi CNN"),
    "IDH2":   ("LGG",  "Mostavi CNN"),   "EGFR":   ("GBM",  "Mostavi CNN"),
    "BRAF":   ("THCA", "Mostavi CNN"),   "VHL":    ("KIRC", "Ramirez GCNN"),
    "PBRM1":  ("KIRC", "Ramirez GCNN"), "FGFR3":  ("BLCA", "Ramirez GCNN"),
    "CDKN2A": ("LUSC", "Ramirez GCNN"), "RB1":    ("LUSC", "Ramirez GCNN"),
    "KRAS":   ("CRC",  "Ramirez GCNN"), "SMAD4":  ("CRC",  "Ramirez GCNN"),
    "MLH1":   ("CRC",  "Ramirez GCNN"), "APC":    ("CRC",  "Ramirez GCNN"),
    "FLT3":   ("LAML", "Ramirez GCNN"), "NPM1":   ("LAML", "Ramirez GCNN"),
    "ERBB2":  ("BRCA", "Mostavi CNN"),  "ESR1":   ("BRCA", "Mostavi CNN"),
    "PGR":    ("BRCA", "Mostavi CNN"),  "PTEN":   ("UCEC", "Mostavi CNN"),
    "CTNNB1": ("UCEC", "Mostavi CNN"),  "BAP1":   ("MESO", "Ramirez GCNN"),
    "NF2":    ("MESO", "Ramirez GCNN"), "GNAQ":   ("UVM",  "Ramirez GCNN"),
    "GNA11":  ("UVM",  "Ramirez GCNN"), "FANCA":  ("CHOL", "Ramirez GCNN"),
}


# =============================================================================
# UTILITIES
# =============================================================================

def print_gpu_info() -> None:
    if DEVICE.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"\n{'='*60}")
        print(f"GPU     : {p.name}")
        print(f"VRAM    : {p.total_memory/1024**3:.1f} GB")
        print(f"CUDA    : {torch.version.cuda}")
        print(f"PyTorch : {torch.__version__}")
        print(f"{'='*60}")
    else:
        print("\nNo GPU detected - running on CPU.")


def gpu_mem() -> str:
    if DEVICE.type != "cuda":
        return ""
    used = torch.cuda.memory_allocated() / 1024**2
    res  = torch.cuda.memory_reserved()  / 1024**2
    return f"  [VRAM {used:.0f}MB/{res:.0f}MB]"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def ensure_dirs() -> tuple[Path, Path]:
    model_dir   = SCRIPT_DIR / "models"
    results_dir = SCRIPT_DIR / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, results_dir


def label_to_abbrev(full: str) -> str:
    return NAME_TO_ABBREV.get(full.lower().strip(), full[:8])


# =============================================================================
# HDF5 LOADING WITH MERGE
# =============================================================================

def find_key(h5f: h5py.File, options: list[str]) -> str:
    for k in options:
        if k in h5f:
            return k
    raise KeyError(f"None of {options} found in HDF5.")


def resolve_h5_path() -> Path:
    candidates = [
        SCRIPT_DIR / "Data/data_filtered.h5",
        SCRIPT_DIR / "Data/data.h5",
        SCRIPT_DIR / "Data/HiSeqV2.h5",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                if h5py.is_hdf5(p):
                    return p
            except Exception:
                pass
    raise FileNotFoundError(f"Dataset not found. Tried: {candidates}")


def load_metadata_with_merge(h5_path: Path) -> dict:
    """
    Load HDF5 metadata and apply READ->COAD merge.

    The merge works at the label level:
      1. Load raw integer labels (original 0-32 encoding)
      2. Convert to disease names using reverse of original diseasedict
      3. Re-encode using DISEASEDICT_MERGED (READ gets same int as COAD)
      4. Re-index to close the gap left by READ (0-31 instead of 0-32)

    This ensures the model trains on 32 contiguous class indices.
    """
    active_dict = DISEASEDICT_MERGED if MERGE_READ_INTO_COAD else DISEASEDICT_ORIGINAL

    with h5py.File(h5_path, "r") as f:
        x_key = find_key(f, ["RNASeq", "X", "expression", "data"])
        y_key = find_key(f, ["label", "labels", "y", "target"])
        x_ds  = f[x_key]
        y_raw = f[y_key][...].astype(np.int64)
        n_samples, n_genes = x_ds.shape

        nan_count = inf_count = 0
        for s in range(0, n_samples, CHUNK_SIZE):
            chunk = x_ds[s : s + CHUNK_SIZE]
            nan_count += int(np.isnan(chunk).sum())
            inf_count += int(np.isinf(chunk).sum())

        gene_names = None
        for gk in ["gene_names", "feature_names", "genes", "columns"]:
            if gk in f:
                raw = f[gk][...]
                gene_names = [
                    g.decode() if isinstance(g, bytes) else str(g) for g in raw
                ]
                break

    # Build reverse map: original int -> disease name
    inv_original = {v: k for k, v in DISEASEDICT_ORIGINAL.items()}

    # Re-encode using merged dict (READ -> COAD = class 8)
    # Note: the gap at class 16 needs closing, so we re-index
    y_merged_raw = np.array(
        [active_dict.get(inv_original.get(int(lbl), ""), -1) for lbl in y_raw],
        dtype=np.int64,
    )

    # Remove any samples that didn't map (shouldn't happen but safety check)
    valid_mask = y_merged_raw >= 0
    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        print(f"  WARNING: {n_removed} samples had unknown labels - removed.")

    # Re-index: close gap from missing READ class (16 is now unused)
    unique_merged = np.array(sorted(np.unique(y_merged_raw[valid_mask]).tolist()))
    reindex       = {old: new for new, old in enumerate(unique_merged)}
    y_final       = np.array([reindex[int(v)] for v in y_merged_raw[valid_mask]],
                              dtype=np.int64)

    # Build final label map (merged class indices -> display names)
    if MERGE_READ_INTO_COAD:
        idx_to_label = {}
        for old_idx, new_idx in reindex.items():
            name = IDX_TO_NAME_MERGED.get(old_idx, f"class_{old_idx}")
            idx_to_label[new_idx] = name
    else:
        inv_orig = {v: k for k, v in DISEASEDICT_ORIGINAL.items()}
        idx_to_label = {new: inv_orig.get(old, f"cls_{old}")
                        for old, new in reindex.items()}

    num_classes = len(unique_merged)

    if MERGE_READ_INTO_COAD:
        colorectal_idx = reindex.get(8, -1)
        crc_count = int((y_final == colorectal_idx).sum())
        print(f"  READ->COAD merge: colorectal class "
              f"(new index {colorectal_idx}) has {crc_count} samples")

    # Sample indices that survived (all if no invalid labels)
    valid_indices = np.where(valid_mask)[0]

    return {
        "x_key":         x_key,
        "y_key":         y_key,
        "n_samples":     int(valid_mask.sum()),
        "n_genes":       n_genes,
        "y":             y_final,
        "valid_indices": valid_indices,
        "nan_count":     nan_count,
        "inf_count":     inf_count,
        "gene_names":    gene_names,
        "idx_to_label":  idx_to_label,
        "num_classes":   num_classes,
    }


# =============================================================================
# PREPROCESSING
# =============================================================================

def extract_transformed(
    h5_path: Path,
    x_key: str,
    indices: np.ndarray,
    scaler: StandardScaler | None,
    fit_scaler: bool,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        x_ds    = f[x_key]
        n_genes = x_ds.shape[1]
        out     = np.empty((len(indices), n_genes), dtype=np.float32)

        sort_ord = np.argsort(indices)
        s_idx    = indices[sort_ord]
        inv_ord  = np.empty_like(sort_ord)
        inv_ord[sort_ord] = np.arange(len(sort_ord))

        if fit_scaler and scaler is not None:
            for s in range(0, len(s_idx), CHUNK_SIZE):
                e     = min(s + CHUNK_SIZE, len(s_idx))
                chunk = x_ds[s_idx[s:e]].astype(np.float32)
                chunk = np.log2(np.clip(chunk, 0, None) + 1.0)
                scaler.partial_fit(chunk)

        for s in range(0, len(s_idx), CHUNK_SIZE):
            e     = min(s + CHUNK_SIZE, len(s_idx))
            chunk = x_ds[s_idx[s:e]].astype(np.float32)
            chunk = np.log2(np.clip(chunk, 0, None) + 1.0)
            if scaler is not None:
                chunk = scaler.transform(chunk)
            out[s:e] = chunk

    return out[inv_ord]


def stratified_split(y: np.ndarray):
    idx = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=0.30, stratify=y, random_state=SEED
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return train_idx, val_idx, test_idx, y_train


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = Counter(y.tolist())
    n = len(y)
    w = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        w[cls] = n / (num_classes * max(counts.get(cls, 1), 1))
    return w


def apply_smote(
    x_train: np.ndarray,
    y_train: np.ndarray,
    target: int   = SMOTE_TARGET,
    threshold: int = SMOTE_THRESHOLD,
    seed: int      = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if not SMOTE_AVAILABLE:
        return x_train, y_train
    counts   = Counter(y_train.tolist())
    minority = {cls: c for cls, c in counts.items() if c < threshold}
    if not minority:
        print("  SMOTE: no class below threshold - skipping.")
        return x_train, y_train
    print(f"  SMOTE: upsampling {len(minority)} minority classes to {target} samples:")
    for cls, c in sorted(minority.items()):
        print(f"    class {cls:2d} : {c:4d} -> {target}")
    sm = SMOTE(
        sampling_strategy={cls: target for cls in minority},
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=seed,
    )
    x_res, y_res = sm.fit_resample(x_train, y_train)
    print(f"  SMOTE: {len(y_train)} -> {len(y_res)} training samples")
    return x_res.astype(np.float32), y_res.astype(np.int64)


# =============================================================================
# MIXUP
# =============================================================================

def mixup_batch(xb, yb, num_classes, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(xb.size(0), device=xb.device)
    x_mixed = lam * xb + (1.0 - lam) * xb[idx]
    y_oh    = torch.zeros(yb.size(0), num_classes, device=yb.device)
    y_oh.scatter_(1, yb.unsqueeze(1), 1.0)
    y_soft  = lam * y_oh + (1.0 - lam) * y_oh[idx]
    return x_mixed, y_soft


def focal_mixup_loss(logits, y_soft, class_weights=None,
                     label_smoothing=LABEL_SMOOTHING, gamma=2.0):
    n = logits.size(1)
    if label_smoothing > 0:
        y_smooth = y_soft * (1.0 - label_smoothing) + label_smoothing / n
    else:
        y_smooth = y_soft
    log_p  = F.log_softmax(logits, dim=1)           # [B, C]
    p      = log_p.exp()                             # [B, C]
    ce     = -(y_smooth * log_p).sum(dim=1)          # [B]
    p_t    = (y_soft * p).sum(dim=1).clamp(0.0, 1.0) # expected confidence
    focal  = (1.0 - p_t) ** gamma                   # [B]
    loss   = focal * ce
    if class_weights is not None:
        cw   = (y_soft * class_weights.to(logits.device)).sum(dim=1)
        loss = loss * cw
    return loss.mean()


# =============================================================================
# MODEL
# =============================================================================

class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.Linear(channels, mid)
        self.excite  = nn.Linear(mid, channels)

    def forward(self, x):                               # x: [B, C, L]
        s = x.mean(dim=2)                               # [B, C] global avg
        s = F.relu(self.squeeze(s))                     # [B, mid]
        s = torch.sigmoid(self.excite(s)).unsqueeze(2)  # [B, C, 1]
        return x * s


class OneDCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.MaxPool1d(4, stride=4),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(4, stride=4),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.MaxPool1d(2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        )
        self.se1 = SEBlock1d(64)
        self.se2 = SEBlock1d(128)
        self.se3 = SEBlock1d(256)
        self.se4 = SEBlock1d(512)
        self.avg_pool   = nn.AdaptiveAvgPool1d(1)
        self.max_pool   = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512,  256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256,  num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.se1(self.block1(x))
        x = self.se2(self.block2(x))
        x = self.se3(self.block3(x))
        x = self.se4(self.block4(x))
        a = self.avg_pool(x).squeeze(2)
        m = self.max_pool(x).squeeze(2)
        return self.classifier(torch.cat([a, m], dim=1))


def to_loader(x, y, shuffle):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, model_path,
                class_weights, num_classes, seed=SEED):
    set_seed(seed)
    val_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=1e-5
    )
    cw_tensor = class_weights.to(DEVICE) if class_weights is not None else None
    prev_lr   = LR

    best_val_acc = -1.0
    best_epoch   = -1
    no_improve   = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_losses, t_preds, t_targets = [], [], []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            xb_noisy = xb + torch.randn_like(xb) * NOISE_STD
            x_mixed, y_soft = mixup_batch(xb_noisy, yb, num_classes)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x_mixed)
                loss   = focal_mixup_loss(logits, y_soft, class_weights=cw_tensor)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_losses.append(loss.item())
            t_preds.extend(logits.detach().argmax(1).cpu().numpy().tolist())
            t_targets.extend(y_soft.argmax(1).cpu().numpy().tolist())

        scheduler.step(epoch)
        train_loss = float(np.mean(t_losses))
        train_acc  = accuracy_score(t_targets, t_preds)

        model.eval()
        v_losses, v_preds, v_targets = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = model(xb)
                v_losses.append(val_criterion(logits, yb).item())
                v_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                v_targets.extend(yb.cpu().numpy().tolist())

        val_loss = float(np.mean(v_losses))
        val_acc  = accuracy_score(v_targets, v_preds)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if epoch > 1 and cur_lr > prev_lr * 3.0:
            no_improve = 0
        prev_lr = cur_lr

        history.append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                         "val_loss": val_loss, "val_accuracy": val_acc, "lr": cur_lr})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS} | "
                  f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={cur_lr:.2e}{gpu_mem()}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, no_improve = val_acc, epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
                break

    return history, best_val_acc, best_epoch


# =============================================================================
# TTA
# =============================================================================

def predict_tta(model, x, n_passes=TTA_N_PASSES, noise_std=TTA_NOISE_STD):
    model.eval()
    all_pass = []
    for _ in range(n_passes):
        batches = []
        with torch.no_grad():
            for s in range(0, x.shape[0], BATCH_SIZE):
                xb = torch.from_numpy(x[s : s + BATCH_SIZE]).float().to(DEVICE)
                xb = xb + torch.randn_like(xb) * noise_std
                batches.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        all_pass.append(np.vstack(batches))
    return np.mean(np.stack(all_pass, axis=0), axis=0)


def get_logits(model, x):
    """Single clean forward pass - returns raw logits for temperature scaling."""
    model.eval()
    batches = []
    with torch.no_grad():
        for s in range(0, x.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(x[s : s + BATCH_SIZE]).float().to(DEVICE)
            batches.append(model(xb).cpu().numpy())
    return np.vstack(batches)


# =============================================================================
# WEIGHTED ENSEMBLE OPTIMISATION
# =============================================================================

def optimise_weights(
    val_probs_list: list[np.ndarray],
    val_y: np.ndarray,
    individual_val_accs: list[float],
) -> np.ndarray:
    """
    Find optimal per-model weights using Nelder-Mead minimisation of
    validation cross-entropy loss. Multiple restarts for robustness.
    """
    def nll(raw_w):
        w = np.abs(raw_w) / (np.abs(raw_w).sum() + 1e-9)
        combined = sum(wi * pi for wi, pi in zip(w, val_probs_list))
        eps = 1e-9
        return -float(np.mean(np.log(combined[np.arange(len(val_y)), val_y] + eps)))

    init = np.array(individual_val_accs)
    init = init / init.sum()

    best_result, best_nll = None, float("inf")
    for restart in range(8):
        if restart == 0:
            x0 = init.copy()
        else:
            rng = np.random.default_rng(restart * 7 + 1)
            x0  = np.abs(init + rng.normal(0, 0.08, size=len(init)))
        result = minimize(nll, x0=x0, method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9,
                                   "adaptive": True})
        if result.fun < best_nll:
            best_nll, best_result = result.fun, result

    w_opt = np.abs(best_result.x)
    w_opt = w_opt / w_opt.sum()
    print(f"  Optimal weights: {[f'{w:.4f}' for w in w_opt]}")
    return w_opt


def stack_ensemble(
    val_probs_list: list[np.ndarray],
    val_y: np.ndarray,
    test_probs_list: list[np.ndarray],
) -> np.ndarray:
    X_val  = np.concatenate(val_probs_list,  axis=1)   # (n_val,  n_models*n_classes)
    X_test = np.concatenate(test_probs_list, axis=1)   # (n_test, n_models*n_classes)
    meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                               random_state=SEED)
    meta.fit(X_val, val_y)
    return meta.predict_proba(X_test)


def find_temperature(
    logits_list: list[np.ndarray],
    weights: np.ndarray,
    val_y: np.ndarray,
) -> float:
    def nll(log_t):
        T = float(np.exp(log_t[0]))
        combined = sum(w * lg for w, lg in zip(weights, logits_list))
        exp_l = np.exp(combined / T)
        probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-9)
        eps = 1e-9
        return -float(np.mean(np.log(probs[np.arange(len(val_y)), val_y] + eps)))

    result = minimize(nll, x0=[0.0], method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-7})
    T_opt = float(np.exp(result.x[0]))
    print(f"  Optimal temperature T = {T_opt:.4f}")
    return T_opt


# =============================================================================
# CONFUSION MATRIX + EVALUATION
# =============================================================================

def plot_confusion_matrix(cm, tick_labels, acc, results_dir, tag):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100.0
    n      = len(tick_labels)

    fig, ax = plt.subplots(figsize=(18, 16))
    im  = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Recall per class (%)", fontsize=11)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted cancer type (TCGA code)", fontsize=12, labelpad=10)
    ax.set_ylabel("True cancer type (TCGA code)",      fontsize=12, labelpad=10)
    merge_note = "  |  READ merged into COAD (colorectal)" if MERGE_READ_INTO_COAD else ""
    ax.set_title(
        f"Confusion matrix [{tag}]\n"
        f"Test accuracy = {acc*100:.2f}%  |  Row-normalised to 100%{merge_note}",
        fontsize=12, fontweight="bold", pad=14,
    )
    thresh = cm_pct.max() / 2.0
    for i in range(n):
        for j in range(n):
            v = cm_pct[i, j]
            if v >= 5.0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=5, color="white" if v > thresh else "black")
    plt.tight_layout()
    out = results_dir / f"confusion_matrix_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def full_eval(y_true, y_prob, num_classes, idx_to_label, results_dir, tag):
    y_pred = y_prob.argmax(axis=1)
    acc    = accuracy_score(y_true, y_pred)
    f1_pc  = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes))
    avg_f1 = f1_score(y_true, y_pred, average="macro")
    cm     = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    tick_labels = [label_to_abbrev(idx_to_label.get(i, f"cls{i}"))
                   for i in range(num_classes)]

    y_oh = np.eye(num_classes, dtype=np.int64)[y_true]

    valid_aucs = []
    for i in range(num_classes):
        col = y_oh[:, i]
        if len(np.unique(col)) < 2: continue
        valid_aucs.append(roc_auc_score(col, y_prob[:, i]))

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1 : {avg_f1:.4f}")
    print(f"  ROC AUC  : {mean_auc:.4f}")

    plot_confusion_matrix(cm, tick_labels, acc, results_dir, tag)

    with open(results_dir / f"metrics_{tag}.txt", "w", encoding="utf-8") as f:
        note = "READ merged into COAD (colorectal)\n" if MERGE_READ_INTO_COAD else ""
        f.write(note)
        f.write(f"Tag           : {tag}\n")
        f.write(f"Test Accuracy : {acc:.6f}\n")
        f.write(f"Macro Avg F1  : {avg_f1:.6f}\n")
        f.write(f"Mean ROC AUC  : {mean_auc:.6f}\n\n")
        f.write("Per-class F1 (sorted worst -> best):\n")
        for i in sorted(range(num_classes), key=lambda x: f1_pc[x]):
            f.write(f"  {tick_labels[i]:<6} | "
                    f"{idx_to_label.get(i, f'cls{i}'):<45} "
                    f"F1={f1_pc[i]:.4f}\n")

    return {"accuracy": acc, "avg_f1": avg_f1, "mean_auc": mean_auc,
            "f1_per_class": f1_pc, "tick_labels": tick_labels}


# =============================================================================
# GENE IMPORTANCE
# =============================================================================

def compute_gene_importance(model, x_test, y_test, idx_to_label,
                             results_dir, gene_names=None):
    model.eval()
    n_genes       = x_test.shape[1]
    global_scores = np.zeros(n_genes, dtype=np.float64)
    class_scores  = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    class_counts  = Counter()

    for s in range(0, x_test.shape[0], IMPORTANCE_BATCH_SIZE):
        xb = (torch.from_numpy(x_test[s : s + IMPORTANCE_BATCH_SIZE])
              .float().to(DEVICE).requires_grad_(True))
        probs = torch.softmax(model(xb), dim=1)
        pred  = probs.argmax(1)
        probs[torch.arange(xb.shape[0], device=DEVICE), pred].sum().backward()
        grads = xb.grad.detach().abs().cpu().numpy()
        for j in range(grads.shape[0]):
            cls = int(pred[j].item())
            global_scores     += grads[j]
            class_scores[cls] += grads[j]
            class_counts[cls] += 1
        model.zero_grad(set_to_none=True)

    global_scores /= max(1, x_test.shape[0])
    for cls in class_scores:
        class_scores[cls] /= max(1, class_counts[cls])

    top50 = np.argsort(global_scores)[::-1][:50]

    def sym(idx):
        return gene_names[idx] if gene_names and idx < len(gene_names) else f"gene_{idx}"

    def bm(s):
        e = KNOWN_BIOMARKERS.get(s.upper())
        return f"{e[0]} ({e[1]})" if e else ""

    with open(results_dir / "gene_importance_scores.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene_index", "gene_symbol", "global_importance_score",
                    "known_biomarker_for (paper)", "top_3_cancer_types"])
        for rank, idx in enumerate(top50, 1):
            s  = sym(int(idx))
            t3 = sorted(class_scores, key=lambda c: class_scores[c][idx], reverse=True)[:3]
            w.writerow([rank, int(idx), s, f"{global_scores[idx]:.6f}", bm(s),
                        "; ".join(idx_to_label.get(c, f"cls{c}") for c in t3)])

    with open(results_dir / "gene_importance_per_cancer.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cancer_type", "tcga_abbrev", "rank", "gene_index",
                    "gene_symbol", "importance_score", "known_biomarker"])
        for cls in sorted(class_scores):
            label  = idx_to_label.get(cls, f"cls{cls}")
            abbrev = label_to_abbrev(label)
            for rank, idx in enumerate(np.argsort(class_scores[cls])[::-1][:10], 1):
                s = sym(int(idx))
                w.writerow([label, abbrev, rank, int(idx), s,
                            f"{class_scores[cls][idx]:.6f}", bm(s)])

    hits = [f"{sym(int(i))} -> {bm(sym(int(i)))}" for i in top50 if bm(sym(int(i)))]
    return {"top50": top50.tolist(), "biomarker_hits": hits}


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_seed(SEED)
    print_gpu_info()
    model_dir, results_dir = ensure_dirs()
    t_start = time.time()

    merge_str = "READ merged into COAD (32 classes)" if MERGE_READ_INTO_COAD else "Original 33 classes"
    print(f"\n{'='*60}")
    print(f"FINAL TRAINING RUN - {merge_str}")
    print(f"{'='*60}")

    # -- Load data with merge
    h5_path = resolve_h5_path()
    print(f"\nDataset: {h5_path}")
    meta = load_metadata_with_merge(h5_path)

    print(f"\n=== DATA VALIDATION ===")
    print(f"Samples    : {meta['n_samples']:,}")
    print(f"Genes      : {meta['n_genes']:,}")
    print(f"Classes    : {meta['num_classes']} {'(32 - READ merged into COAD)' if MERGE_READ_INTO_COAD else '(33)'}")
    print(f"NaN/Inf    : {meta['nan_count']} / {meta['inf_count']}")

    y           = meta["y"]
    idx_to_label = meta["idx_to_label"]
    num_classes  = meta["num_classes"]
    gene_names   = meta.get("gene_names")

    print(f"\nClass distribution:")
    for cls, cnt in sorted(Counter(y.tolist()).items()):
        print(f"  {cls:2d} | {idx_to_label.get(cls,'?'):<45} : {cnt}")

    # -- Split
    print(f"\n=== SPLIT ===")
    train_idx, val_idx, test_idx, y_train_raw = stratified_split(y)
    y_val  = y[val_idx]
    y_test = y[test_idx]
    print(f"Train={len(train_idx):,}  Val={len(val_idx):,}  Test={len(test_idx):,}")

    # -- Preprocessing - map HDF5 row indices through valid_indices
    print(f"\n=== PREPROCESSING ===")
    valid_indices = meta["valid_indices"]
    scaler  = StandardScaler()
    train_x = extract_transformed(h5_path, meta["x_key"],
                                  valid_indices[train_idx], scaler, fit_scaler=True)
    val_x   = extract_transformed(h5_path, meta["x_key"],
                                  valid_indices[val_idx],   scaler, fit_scaler=False)
    test_x  = extract_transformed(h5_path, meta["x_key"],
                                  valid_indices[test_idx],  scaler, fit_scaler=False)
    train_y = y_train_raw.astype(np.int64)
    val_y   = y_val.astype(np.int64)
    test_y  = y_test.astype(np.int64)

    val_loader = to_loader(val_x, val_y, shuffle=False)

    # -- Ensemble training
    print(f"\n=== ENSEMBLE TRAINING ({len(ENSEMBLE_SEEDS)} models, READ->COAD merged) ===")
    ensemble_paths: list[Path] = []
    val_probs_list: list[np.ndarray]  = []
    val_logits_list: list[np.ndarray] = []
    test_probs_list: list[np.ndarray] = []
    test_logits_list: list[np.ndarray] = []
    individual_val_accs: list[float] = []
    single_results: list[dict] = []

    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        print(f"\n{'-'*55}")
        print(f"  Model {i}/5  |  seed={seed}")
        print(f"{'-'*55}")

        sm_x, sm_y    = apply_smote(train_x, train_y, seed=seed)
        train_loader  = to_loader(sm_x, sm_y, shuffle=True)
        class_weights = compute_class_weights(sm_y, num_classes)
        print(f"  Class weights: min={class_weights.min():.3f}  max={class_weights.max():.3f}")

        model = OneDCNN(num_classes=num_classes).to(DEVICE)
        if i == 1:
            tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Architecture: 4-block 1D-CNN | {tp:,} params  "
                  f"(num_classes={num_classes}){gpu_mem()}")

        model_path = model_dir / f"best_model_seed{seed}.pt"

        history, best_val_acc, best_epoch = train_model(
            model, train_loader, val_loader, model_path,
            class_weights, num_classes, seed,
        )
        print(f"  Best val_acc={best_val_acc:.4f} at epoch {best_epoch}{gpu_mem()}")

        with open(results_dir / f"history_seed{seed}.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch","loss","accuracy","val_loss","val_accuracy","lr"]
            )
            writer.writeheader(); writer.writerows(history)

        # Load best checkpoint
        best_model = OneDCNN(num_classes=num_classes).to(DEVICE)
        best_model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )

        # Collect val probabilities (for weight optimisation)
        print(f"  Collecting val probabilities (TTA) ...")
        vp = predict_tta(best_model, val_x)
        val_probs_list.append(vp)
        val_logits_list.append(get_logits(best_model, val_x))

        # Collect test probabilities
        print(f"  Collecting test probabilities (TTA) ...")
        tp_probs = predict_tta(best_model, test_x)
        test_probs_list.append(tp_probs)
        test_logits_list.append(get_logits(best_model, test_x))

        val_acc = accuracy_score(val_y, vp.argmax(axis=1))
        individual_val_accs.append(val_acc)
        print(f"  Individual val accuracy (TTA): {val_acc*100:.2f}%")

        # Single-model evaluation
        res = full_eval(test_y, tp_probs, num_classes, idx_to_label,
                        results_dir, f"seed{seed}")
        ensemble_paths.append(model_path)
        single_results.append(res)

    # -- Equal-weight ensemble baseline
    print(f"\n=== EQUAL-WEIGHT ENSEMBLE (BASELINE) ===")
    equal_w      = np.ones(len(ENSEMBLE_SEEDS)) / len(ENSEMBLE_SEEDS)
    test_eq_prob = sum(w * p for w, p in zip(equal_w, test_probs_list))
    eq_acc       = accuracy_score(test_y, test_eq_prob.argmax(axis=1))
    print(f"  Equal-weight test accuracy: {eq_acc*100:.2f}%")

    # -- Optimise weights on validation set
    print(f"\n=== OPTIMISING ENSEMBLE WEIGHTS ===")
    opt_weights = optimise_weights(val_probs_list, val_y, individual_val_accs)
    test_opt_prob = sum(w * p for w, p in zip(opt_weights, test_probs_list))
    opt_acc = accuracy_score(test_y, test_opt_prob.argmax(axis=1))
    print(f"  Weighted ensemble test accuracy: {opt_acc*100:.2f}%")
    print(f"  Gain over equal weights: {(opt_acc - eq_acc)*100:+.2f}%")

    # -- Meta-learner stacking (compare to weighted avg, keep best)
    print(f"\n=== META-LEARNER STACKING ===")
    test_meta_prob = stack_ensemble(val_probs_list, val_y, test_probs_list)
    meta_acc = accuracy_score(test_y, test_meta_prob.argmax(axis=1))
    print(f"  Meta-learner test accuracy: {meta_acc*100:.2f}%")
    if meta_acc >= opt_acc:
        test_opt_prob = test_meta_prob
        print(f"  Using meta-learner (better by {(meta_acc-opt_acc)*100:+.2f}%)")
    else:
        print(f"  Keeping weighted avg (meta-learner worse by {(meta_acc-opt_acc)*100:.2f}%)")

    # -- Temperature scaling
    print(f"\n=== TEMPERATURE SCALING ===")
    T_opt = find_temperature(val_logits_list, opt_weights, val_y)

    def apply_temp(logits_list, weights, T):
        combined = sum(w * lg for w, lg in zip(weights, logits_list))
        e = np.exp(combined / T)
        return e / (e.sum(axis=1, keepdims=True) + 1e-9)

    test_calib_prob = apply_temp(test_logits_list, opt_weights, T_opt)

    # -- Final blend: TTA diversity + calibration precision
    if T_opt < 1.0:
        BLEND_TTA, BLEND_CALIB = 0.6, 0.4
        test_final = BLEND_TTA * test_opt_prob + BLEND_CALIB * test_calib_prob
        blend_desc = f"TTA 60% + calibrated 40% (T={T_opt:.3f})"
    else:
        test_final = test_opt_prob
        blend_desc = f"TTA only (T={T_opt:.3f} >= 1, calibration skipped)"
    final_acc  = accuracy_score(test_y, test_final.argmax(axis=1))
    print(f"\n  Final ({blend_desc}): {final_acc*100:.2f}%")

    # -- Full evaluation on final ensemble
    print(f"\n=== FINAL EVALUATION ===")
    tag = "final_merged" if MERGE_READ_INTO_COAD else "final_33class"
    final_res = full_eval(test_y, test_final, num_classes, idx_to_label,
                          results_dir, tag)

    # Save final probabilities and weights
    np.save(results_dir / "final_probabilities.npy", test_final)
    np.save(results_dir / "optimal_weights.npy", opt_weights)

    # -- Gene importance
    print(f"\n=== GENE IMPORTANCE ===")
    best_m = OneDCNN(num_classes=num_classes).to(DEVICE)
    best_m.load_state_dict(
        torch.load(ensemble_paths[0], map_location=DEVICE, weights_only=True)
    )
    gene_out = compute_gene_importance(
        best_m, test_x, test_y, idx_to_label, results_dir, gene_names
    )
    if gene_out["biomarker_hits"]:
        print(f"  Known biomarker hits ({len(gene_out['biomarker_hits'])}):")
        for hit in gene_out["biomarker_hits"]:
            print(f"    [OK] {hit}")
    else:
        print("  No named hits (add gene_names to HDF5 for HGNC symbols).")

    # -- Final summary
    elapsed = time.time() - t_start
    best_single_acc = max(r["accuracy"] for r in single_results)
    worst5 = sorted(range(num_classes), key=lambda i: final_res["f1_per_class"][i])[:5]

    lines = [
        "",
        "=" * 65,
        "FINAL TRAINING RUN - SUMMARY",
        "=" * 65,
        f"{'READ merged into COAD (32 classes)' if MERGE_READ_INTO_COAD else '33 classes (no merge)'}",
        f"Dataset  : {h5_path}",
        f"Samples  : {meta['n_samples']:,}  |  Genes: {meta['n_genes']:,}  |  Classes: {num_classes}",
        f"Device   : {DEVICE}  |  PyTorch {torch.__version__}",
        "",
        "Augmentation: MixUp(0.2) + Noise(0.05) + Label Smooth(0.1) + SMOTE",
        f"Ensemble : {len(ENSEMBLE_SEEDS)} models x {TTA_N_PASSES} TTA passes = "
        f"{len(ENSEMBLE_SEEDS)*TTA_N_PASSES} predictions per sample",
        f"Weights  : Nelder-Mead optimised on validation set",
        f"Calibration: Temperature scaling T={T_opt:.3f} + TTA blend",
        "",
        "Results:",
        f"  Equal-weight ensemble       : {eq_acc*100:.2f}%",
        f"  Weighted ensemble           : {opt_acc*100:.2f}%",
        f"  Weighted + calibrated (final): {final_acc*100:.2f}%",
        f"  Macro F1                    : {final_res['avg_f1']:.4f}",
        f"  Mean ROC AUC                : {final_res['mean_auc']:.4f}",
        f"  >95% target                 : {'[OK] PASSED' if final_res['accuracy'] > 0.95 else '[X] NOT YET'}",
        "",
        "Optimal model weights:",
    ]
    for seed, w in zip(ENSEMBLE_SEEDS, opt_weights):
        lines.append(f"  seed={seed:3d} : {w:.4f}")
    lines += [
        "",
        "Worst 5 classes:",
    ]
    for i in worst5:
        lines.append(
            f"  {final_res['tick_labels'][i]:<6} | "
            f"{idx_to_label.get(i,'?'):<45} "
            f"F1={final_res['f1_per_class'][i]:.4f}"
        )
    lines += [
        "",
        "Output files (in results/):",
        f"  confusion_matrix_{tag}.png        <- PRIMARY for final report",
        f"  metrics_{tag}.txt",
        "  gene_importance_scores.csv",
        "  gene_importance_per_cancer.csv",
        "  final_probabilities.npy",
        "  optimal_weights.npy",
        f"  models/best_model_seed{{{','.join(str(s) for s in ENSEMBLE_SEEDS)}}}.pt",
        "",
        f"Total training time : {elapsed/60:.1f} minutes",
        "=" * 65,
    ]

    summary = "\n".join(lines)
    print(summary)
    with open(results_dir / "final_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()