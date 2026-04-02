import csv
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from labelMapping import diseasedict
except Exception:
    diseasedict = {}


SEED = 42
BATCH_SIZE = 64
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
CHUNK_SIZE = 256
IMPORTANCE_BATCH_SIZE = 64
IMPORTANCE_MAX_SAMPLES = None
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> tuple[Path, Path]:
    model_dir = Path("models")
    results_dir = Path("results")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, results_dir


def resolve_h5_path() -> Path:
    candidates = [
        Path("Data/data.h5"),
        Path("Data/HiSeqV2.h5"),
        Path("data/HiSeqV2.h5"),
        Path("../data/HiSeqV2.h5"),
    ]
    invalid_candidates = []
    for path in candidates:
        if not path.exists():
            continue
        if not path.is_file():
            invalid_candidates.append(f"{path} (exists but is not a file)")
            continue
        try:
            if not h5py.is_hdf5(path):
                invalid_candidates.append(f"{path} (exists but is not a valid HDF5 file)")
                continue
        except Exception as exc:
            invalid_candidates.append(f"{path} (HDF5 check failed: {exc})")
            continue
        return path
    raise FileNotFoundError(
        "Could not find dataset. Expected one of: "
        + ", ".join(str(p) for p in candidates)
        + ". Invalid candidates: "
        + ("; ".join(invalid_candidates) if invalid_candidates else "none")
    )


def find_dataset_key(h5f: h5py.File, options: list[str]) -> str:
    for key in options:
        if key in h5f:
            return key
    raise KeyError(f"None of the expected keys found: {options}")


def load_metadata(h5_path: Path):
    with h5py.File(h5_path, "r") as h5f:
        x_key = find_dataset_key(h5f, ["RNASeq", "X", "expression", "data"])
        y_key = find_dataset_key(h5f, ["label", "labels", "y", "target"])
        name_key = "name" if "name" in h5f else None

        x_ds = h5f[x_key]
        y = h5f[y_key][...].astype(np.int64)
        names = h5f[name_key][...] if name_key else None

        n_samples, n_genes = x_ds.shape
        nan_count = 0
        inf_count = 0
        for start in range(0, n_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_samples)
            chunk = x_ds[start:end]
            nan_count += int(np.isnan(chunk).sum())
            inf_count += int(np.isinf(chunk).sum())

    return {
        "x_key": x_key,
        "y_key": y_key,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "y": y,
        "sample_names": names,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def make_label_map(y: np.ndarray):
    unique_labels = sorted(np.unique(y).tolist())
    idx_to_label = {}
    if diseasedict:
        inv = {v: k for k, v in diseasedict.items()}
        for i in unique_labels:
            idx_to_label[int(i)] = inv.get(int(i), f"class_{i}")
    else:
        for i in unique_labels:
            idx_to_label[int(i)] = f"class_{i}"
    return idx_to_label


def encode_labels(y: np.ndarray):
    unique = sorted(np.unique(y).tolist())
    to_encoded = {orig: i for i, orig in enumerate(unique)}
    encoded = np.array([to_encoded[int(v)] for v in y], dtype=np.int64)
    return encoded, to_encoded


def stratified_split(y: np.ndarray):
    idx = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=0.30, stratify=y, random_state=SEED
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return train_idx, val_idx, test_idx, y_train


def print_distribution(title: str, labels: np.ndarray, idx_to_label: dict[int, str]):
    print(f"\n{title}")
    counts = Counter(labels.tolist())
    for k in sorted(counts):
        print(f"  {k:2d} | {idx_to_label.get(k, f'class_{k}'):<45} : {counts[k]}")


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = Counter(y.tolist())
    n = len(y)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        c = counts.get(cls, 1)
        weights[cls] = n / (num_classes * c)
    return weights


def extract_transformed(
    h5_path: Path,
    x_key: str,
    indices: np.ndarray,
    scaler: StandardScaler | None,
    fit_scaler: bool,
):
    with h5py.File(h5_path, "r") as h5f:
        x_ds = h5f[x_key]
        n_genes = x_ds.shape[1]
        out = np.empty((len(indices), n_genes), dtype=np.float32)

        sorted_order = np.argsort(indices)
        sorted_idx = indices[sorted_order]
        inv_order = np.empty_like(sorted_order)
        inv_order[sorted_order] = np.arange(len(sorted_order))

        if fit_scaler and scaler is not None:
            for start in range(0, len(sorted_idx), CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, len(sorted_idx))
                rows = sorted_idx[start:end]
                chunk = x_ds[rows].astype(np.float32)
                chunk = np.log2(np.clip(chunk, 0, None) + 1.0)
                scaler.partial_fit(chunk)

        for start in range(0, len(sorted_idx), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(sorted_idx))
            rows = sorted_idx[start:end]
            chunk = x_ds[rows].astype(np.float32)
            chunk = np.log2(np.clip(chunk, 0, None) + 1.0)
            if scaler is not None:
                chunk = scaler.transform(chunk)
            out[start:end] = chunk

    return out[inv_order]


# ---------------------------------------------------------------------------
# Model — deeper 1D-CNN with larger kernels, progressive pooling, and
# combined avg+max global pooling for richer feature extraction.
# ---------------------------------------------------------------------------
class OneDCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1 — broad 7-wide kernel captures multi-gene motifs
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)                         # (B, 1, n_genes)
        x = self.conv_blocks(x)                     # (B, 256, L)
        a = self.avg_pool(x).squeeze(2)             # (B, 256)
        m = self.max_pool(x).squeeze(2)             # (B, 256)
        x = torch.cat([a, m], dim=1)                # (B, 512)
        return self.classifier(x)


def to_loader(x: np.ndarray, y: np.ndarray, shuffle: bool):
    tx = torch.from_numpy(x).float()
    ty = torch.from_numpy(y).long()
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_path: Path,
    class_weights: torch.Tensor | None = None,
):
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            train_targets.extend(yb.cpu().numpy().tolist())

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_targets, train_preds)

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                val_targets.extend(yb.cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_targets, val_preds)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": current_lr,
            }
        )

        if (epoch % 10 == 0) or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.6f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_acc={best_val_acc:.4f} (epoch {best_epoch})"
                )
                break

    return history, best_val_acc, best_epoch


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    idx_to_label: dict[int, str],
    results_dir: Path,
):
    model.eval()
    all_logits = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_logits.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.numpy().tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.vstack(all_logits)

    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes))
    avg_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    lb = LabelBinarizer()
    lb.fit(np.arange(num_classes))
    y_true_onehot = lb.transform(y_true)
    if y_true_onehot.shape[1] != num_classes:
        full = np.zeros((len(y_true), num_classes), dtype=np.int64)
        for i, cls in enumerate(y_true):
            full[i, cls] = 1
        y_true_onehot = full

    roc_info = {}
    for i in range(num_classes):
        y_col = y_true_onehot[:, i]
        if len(np.unique(y_col)) < 2:
            roc_info[i] = {"auc": np.nan, "fpr": np.array([0, 1]), "tpr": np.array([0, 1])}
            continue
        fpr, tpr, _ = roc_curve(y_col, y_prob[:, i])
        auc = roc_auc_score(y_col, y_prob[:, i])
        roc_info[i] = {"auc": auc, "fpr": fpr, "tpr": tpr}

    plt.figure(figsize=(14, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 10))
    for i in range(num_classes):
        info = roc_info[i]
        if np.isnan(info["auc"]):
            continue
        plt.plot(
            info["fpr"],
            info["tpr"],
            lw=1,
            label=f"{idx_to_label.get(i, f'class_{i}')[:22]} AUC={info['auc']:.3f}",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curves.png", dpi=300)
    plt.close()

    metrics_path = results_dir / "evaluation_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc:.6f}\n")
        f.write(f"Macro Avg F1: {avg_f1:.6f}\n")
        f.write("Per-class F1:\n")
        for i in range(num_classes):
            f.write(
                f"  {i:2d} | {idx_to_label.get(i, f'class_{i}'):<45} : "
                f"{f1_per_class[i]:.6f}\n"
            )

    return {
        "accuracy": acc,
        "avg_f1": avg_f1,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "roc_info": roc_info,
    }


def compute_gene_importance(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    idx_to_label: dict[int, str],
    results_dir: Path,
):
    model.eval()
    n_genes = x_test.shape[1]
    x_use = x_test
    if IMPORTANCE_MAX_SAMPLES is not None and x_test.shape[0] > IMPORTANCE_MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(x_test.shape[0], size=IMPORTANCE_MAX_SAMPLES, replace=False)
        x_use = x_test[pick]
        print(
            f"Gene importance: using {IMPORTANCE_MAX_SAMPLES} / {x_test.shape[0]} "
            f"test samples."
        )
    else:
        print(
            f"Gene importance: batched saliency on {x_use.shape[0]} test samples "
            f"(batch_size={IMPORTANCE_BATCH_SIZE})."
        )

    global_scores = np.zeros(n_genes, dtype=np.float64)
    class_scores = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    class_counts = Counter()

    for start in range(0, x_use.shape[0], IMPORTANCE_BATCH_SIZE):
        end = min(start + IMPORTANCE_BATCH_SIZE, x_use.shape[0])
        xb = torch.from_numpy(x_use[start:end]).float().to(DEVICE).requires_grad_(True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        batch_idx = torch.arange(xb.shape[0], device=DEVICE)
        selected = probs[batch_idx, pred]
        loss = selected.sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = xb.grad.detach().abs().cpu().numpy()
        for j in range(grads.shape[0]):
            cls = int(pred[j].item())
            global_scores += grads[j]
            class_scores[cls] += grads[j]
            class_counts[cls] += 1

    global_scores /= max(1, x_use.shape[0])
    for cls in class_scores:
        class_scores[cls] /= max(1, class_counts[cls])

    top_global = np.argsort(global_scores)[::-1][:50]
    out_csv = results_dir / "gene_importance_scores.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gene_rank",
                "gene_index",
                "global_importance_score",
                "top_class_labels_where_gene_is_important",
            ]
        )
        for rank, gene_idx in enumerate(top_global, start=1):
            ranked_classes = sorted(
                class_scores.keys(),
                key=lambda c: class_scores[c][gene_idx],
                reverse=True,
            )[:3]
            top_classes = "; ".join(idx_to_label.get(c, f"class_{c}") for c in ranked_classes)
            writer.writerow([rank, int(gene_idx), float(global_scores[gene_idx]), top_classes])

    per_class_notes = []
    for cls in sorted(class_scores.keys()):
        top_class_genes = np.argsort(class_scores[cls])[::-1][:10].tolist()
        per_class_notes.append(
            f"{idx_to_label.get(cls, f'class_{cls}')}: top_gene_indices={top_class_genes}"
        )

    significance_text = (
        "Gene-effect scores are estimated using absolute input gradients of the predicted "
        "class probabilities (saliency-based attribution). Higher score means a gene has "
        "greater influence on model predictions in the standardized log2-expression space. "
        "This is a biologically guided proxy for candidate biomarker relevance, and should "
        "be interpreted with pathway/literature validation before clinical use."
    )

    return {
        "top_global": top_global.tolist(),
        "global_scores": global_scores,
        "per_class_notes": per_class_notes,
        "significance_text": significance_text,
    }


def main():
    set_seed(SEED)
    model_dir, results_dir = ensure_dirs()
    best_model_path = model_dir / "best_1dcnn_model.pt"

    start_time = time.time()
    h5_path = resolve_h5_path()
    print(f"Using dataset: {h5_path}")

    meta = load_metadata(h5_path)
    y_raw = meta["y"]
    idx_to_label_raw = make_label_map(y_raw)
    y, orig_to_enc = encode_labels(y_raw)
    enc_to_orig = {v: k for k, v in orig_to_enc.items()}
    idx_to_label = {enc: idx_to_label_raw.get(orig, f"class_{orig}") for enc, orig in enc_to_orig.items()}
    num_classes = len(np.unique(y))
    print("\n=== DATA VALIDATION PHASE ===")
    print(f"Dataset shape: samples={meta['n_samples']}, genes={meta['n_genes']}")
    print(f"NaN values: {meta['nan_count']}")
    print(f"Infinite values: {meta['inf_count']}")
    if meta["nan_count"] > 0 or meta["inf_count"] > 0:
        print("Data quality issue detected: NaN/Inf present.")
    else:
        print("Data quality check passed: no NaN/Inf values.")

    print_distribution("Cancer type frequencies (full dataset):", y, idx_to_label)

    train_idx, val_idx, test_idx, y_train_full = stratified_split(y)
    y_val = y[val_idx]
    y_test = y[test_idx]
    print(
        f"\nSplit sizes: train={len(train_idx)} ({len(train_idx)/len(y):.1%}), "
        f"val={len(val_idx)} ({len(val_idx)/len(y):.1%}), "
        f"test={len(test_idx)} ({len(test_idx)/len(y):.1%})"
    )

    # ── Class imbalance handling: inverse-frequency class weights ──────────
    print("\n=== DATA PREPROCESSING PHASE ===")
    train_counts = Counter(y_train_full.tolist())
    class_weights = compute_class_weights(y_train_full, num_classes)
    print_distribution("Training set class distribution:", y_train_full, idx_to_label)
    print("\nClass imbalance strategy: inverse-frequency weighted CrossEntropyLoss")
    print("  (All training samples are kept — no under/over-sampling.)")
    print(f"  Computed class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f}):")
    for cls in sorted(train_counts):
        print(
            f"    {cls:2d} | {idx_to_label.get(cls, f'class_{cls}'):<45} "
            f"n={train_counts[cls]:>4d}  w={class_weights[cls]:.3f}"
        )

    scaler = StandardScaler()
    train_x = extract_transformed(
        h5_path, meta["x_key"], train_idx, scaler=scaler, fit_scaler=True
    )
    val_x = extract_transformed(h5_path, meta["x_key"], val_idx, scaler=scaler, fit_scaler=False)
    test_x = extract_transformed(h5_path, meta["x_key"], test_idx, scaler=scaler, fit_scaler=False)

    train_y = y_train_full.astype(np.int64)
    val_y = y_val.astype(np.int64)
    test_y = y_test.astype(np.int64)

    np.save("train_X.npy", train_x)
    np.save("train_y.npy", train_y)
    np.save("val_X.npy", val_x)
    np.save("val_y.npy", val_y)
    np.save("test_X.npy", test_x)
    np.save("test_y.npy", test_y)
    print("Saved preprocessed arrays: train_X/train_y/val_X/val_y/test_X/test_y .npy")

    # ── Model + training ──────────────────────────────────────────────────
    print("\n=== 1D-CNN MODEL + TRAINING PHASE ===")
    model = OneDCNN(num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model device: {DEVICE}")
    print(f"Trainable parameters: {total_params:,}")

    train_loader = to_loader(train_x, train_y, shuffle=True)
    val_loader = to_loader(val_x, val_y, shuffle=False)
    test_loader = to_loader(test_x, test_y, shuffle=False)

    history, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=best_model_path,
        class_weights=class_weights,
    )
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    history_path = results_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "lr"]
        )
        writer.writeheader()
        writer.writerows(history)

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n=== EVALUATION PHASE ===")
    best_model = OneDCNN(num_classes=num_classes).to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    eval_out = evaluate_model(
        model=best_model,
        test_loader=test_loader,
        num_classes=num_classes,
        idx_to_label=idx_to_label,
        results_dir=results_dir,
    )

    # ── Gene importance ────────────────────────────────────────────────────
    print("\n=== FEATURE IMPORTANCE PHASE ===")
    gene_out = compute_gene_importance(
        model=best_model,
        x_test=test_x,
        y_test=test_y,
        idx_to_label=idx_to_label,
        results_dir=results_dir,
    )

    # ── Summary report ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    target = 0.95
    hit_target = eval_out["accuracy"] > target

    summary_lines = [
        "TCGA Pan-Cancer 1D-CNN Training Summary",
        "=" * 60,
        f"Dataset path: {h5_path}",
        f"Dataset composition: samples={meta['n_samples']}, genes={meta['n_genes']}, classes={num_classes}",
        "Cancer type composition (full dataset):",
    ]
    for cls, c in sorted(Counter(y.tolist()).items()):
        summary_lines.append(f"  {cls:2d} | {idx_to_label.get(cls, f'class_{cls}'):<45} : {c}")
    summary_lines += [
        "",
        "Class imbalance handling strategy:",
        "  - Inverse-frequency weighted CrossEntropyLoss (all training samples kept)",
        f"  - Training set min/max class counts: {min(train_counts.values())}/{max(train_counts.values())}",
        f"  - Weight range: {class_weights.min():.3f} – {class_weights.max():.3f}",
        "",
        "Model architecture details:",
        "  - Conv1d(1->64, k=7, p=3) + BN + ReLU + MaxPool(4)",
        "  - Conv1d(64->128, k=5, p=2) + BN + ReLU + MaxPool(4)",
        "  - Conv1d(128->256, k=3, p=1) + BN + ReLU",
        "  - AdaptiveAvgPool1d(1) || AdaptiveMaxPool1d(1) → concat (512)",
        "  - Dense(512->256) + ReLU + Dropout(0.5)",
        "  - Dense(256->num_classes)",
        f"  - Trainable parameters: {total_params:,}",
        "",
        "Training hyperparameters:",
        f"  - epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}",
        f"  - early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        f"  - lr_scheduler=ReduceLROnPlateau(factor=0.5, patience=7)",
        f"  - optimizer=AdamW, gradient_clipping=1.0",
        f"  - best_model_path={best_model_path}",
        "",
        f"Final test accuracy: {eval_out['accuracy']:.6f}",
        f"Final macro F1-score: {eval_out['avg_f1']:.6f}",
        f"Goal comparison (>95% accuracy): {'PASSED' if hit_target else 'NOT MET'}",
        "",
        "Gene importance findings:",
        f"  - Top 50 genes saved to {results_dir / 'gene_importance_scores.csv'}",
        f"  - Top 10 global gene indices: {gene_out['top_global'][:10]}",
        f"  - Interpretation: {gene_out['significance_text']}",
        "  - Per-cancer-type critical gene notes:",
    ]
    summary_lines.extend([f"    * {line}" for line in gene_out["per_class_notes"]])
    summary_lines += [
        "",
        f"Time taken (seconds): {elapsed:.2f}",
        f"Device used: {DEVICE}",
        "",
        "Artifacts:",
        f"  - {results_dir / 'evaluation_metrics.txt'}",
        f"  - {results_dir / 'confusion_matrix.png'}",
        f"  - {results_dir / 'roc_curves.png'}",
        f"  - {results_dir / 'training_summary.txt'}",
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(results_dir / "training_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
