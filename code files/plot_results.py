"""
plot_results.py — All visualisation figures for the final report
=================================================================
Generates three figures from the results/ outputs of train.py.

  Figure 1 — Per-class F1-score bar chart
  Figure 2 — Training & validation curves (per seed or averaged)
  Figure 3 — Class distribution bar chart

Usage:
    python "code files/plot_results.py"            # all three figures
    python "code files/plot_results.py" f1         # F1 chart only
    python "code files/plot_results.py" curves      # training curves only
    python "code files/plot_results.py" dist        # class distribution only

All outputs saved to plots/ at 300 DPI.
"""

import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR  = SCRIPT_DIR / "results"
PLOTS_DIR    = SCRIPT_DIR / "plots"
METRICS_FILE = RESULTS_DIR / "metrics_final_merged.txt"
SUMMARY_FILE = RESULTS_DIR / "final_summary.txt"

HISTORY_FILES = [
    RESULTS_DIR / f"history_seed{s}.csv"
    for s in [42, 123, 456, 789, 999]
]

# ---------------------------------------------------------------------------
# Cancer type name → TCGA abbreviation (32-class merged set)
# ---------------------------------------------------------------------------
NAME_TO_ABBREV = {
    "skin cutaneous melanoma":               "SKCM",
    "thyroid carcinoma":                     "THCA",
    "sarcoma":                               "SARC",
    "prostate adenocarcinoma":               "PRAD",
    "pheochromocytoma & paraganglioma":      "PCPG",
    "pancreatic adenocarcinoma":             "PAAD",
    "head & neck squamous cell carcinoma":   "HNSC",
    "esophageal carcinoma":                  "ESCA",
    "colorectal cancer (coad+read)":         "CRC",
    "colon adenocarcinoma":                  "COAD",
    "cervical & endocervical cancer":        "CESC",
    "breast invasive carcinoma":             "BRCA",
    "bladder urothelial carcinoma":          "BLCA",
    "testicular germ cell tumor":            "TGCT",
    "kidney papillary cell carcinoma":       "KIRP",
    "kidney clear cell carcinoma":           "KIRC",
    "acute myeloid leukemia":               "LAML",
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


def abbrev(name: str) -> str:
    return NAME_TO_ABBREV.get(name.lower().strip(), name.upper()[:4])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family":          "DejaVu Sans",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


# ---------------------------------------------------------------------------
# Figure 1 — Per-class F1-score bar chart
# ---------------------------------------------------------------------------
def plot_f1() -> None:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Not found: {METRICS_FILE}\nRun train_merged.py first.")

    test_acc = macro_f1 = 0.0
    per_class: list[tuple[str, float]] = []
    line_pat = re.compile(r"^\s*(\S+)\s*\|\s*(.+?)\s+F1=([\d.]+)\s*$")

    with open(METRICS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Test Accuracy"):
                test_acc = float(line.split(":")[1].strip())
            elif line.startswith("Macro Avg F1"):
                macro_f1 = float(line.split(":")[1].strip())
            else:
                m = line_pat.match(line)
                if m:
                    per_class.append((m.group(1).strip(), float(m.group(3))))

    if not per_class:
        raise ValueError(f"Could not parse per-class F1 from {METRICS_FILE}")

    per_class.sort(key=lambda x: x[1], reverse=True)
    labels  = [r[0] for r in per_class]
    values  = [r[1] for r in per_class]
    colors  = ["#4CAF50" if v >= 0.85 else "#FF9800" if v >= 0.70 else "#F44336"
               for v in values]
    n_ex    = sum(1 for v in values if v >= 0.85)
    n_mod   = sum(1 for v in values if 0.70 <= v < 0.85)
    n_poor  = sum(1 for v in values if v < 0.70)

    apply_style()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0.80,     color="#2196F3", linestyle="--", linewidth=1.8)
    ax.axhline(macro_f1, color="#607D8B", linestyle="-",  linewidth=1.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_xlabel("Cancer type (TCGA code, sorted by F1)", fontsize=12)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title(
        f"Per-class F1-scores  |  Macro avg = {macro_f1:.3f}  |  "
        f"Test accuracy = {test_acc*100:.1f}%",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)

    ax.legend(handles=[
        mpatches.Patch(facecolor="#4CAF50", label=f"F1 ≥ 0.85 — excellent ({n_ex} classes)"),
        mpatches.Patch(facecolor="#FF9800", label=f"0.70–0.85 — moderate ({n_mod} classes)"),
        mpatches.Patch(facecolor="#F44336", label=f"F1 < 0.70 — needs improvement ({n_poor} classes)"),
        plt.Line2D([0],[0], color="#2196F3", linestyle="--", linewidth=1.8, label="F1 = 0.80 threshold"),
        plt.Line2D([0],[0], color="#607D8B", linestyle="-",  linewidth=1.5,
                   label=f"Macro avg F1 = {macro_f1:.3f}"),
    ], fontsize=9, loc="lower left")

    out = PLOTS_DIR / "f1_scores_per_class.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"  Accuracy={test_acc*100:.2f}%  MacroF1={macro_f1:.4f}  "
          f"Excellent={n_ex}  Moderate={n_mod}  Poor={n_poor}")


# ---------------------------------------------------------------------------
# Figure 2 — Training curves (averaged across all seeds that have a CSV)
# ---------------------------------------------------------------------------
def plot_curves() -> None:
    available = [p for p in HISTORY_FILES if p.exists()]
    if not available:
        raise FileNotFoundError(
            f"No history CSV files found in {RESULTS_DIR}.\n"
            "Run train_merged.py first."
        )

    dfs = [pd.read_csv(p) for p in available]
    seed_labels = [p.stem.replace("history_seed", "seed ") for p in available]

    # Average across seeds for the main lines; show individual as thin background lines
    min_len  = min(len(d) for d in dfs)
    df_avg   = pd.DataFrame({
        "epoch":        dfs[0]["epoch"].iloc[:min_len].values,
        "accuracy":     sum(d["accuracy"].iloc[:min_len].values     for d in dfs) / len(dfs),
        "val_accuracy": sum(d["val_accuracy"].iloc[:min_len].values for d in dfs) / len(dfs),
        "loss":         sum(d["loss"].iloc[:min_len].values          for d in dfs) / len(dfs),
        "val_loss":     sum(d["val_loss"].iloc[:min_len].values      for d in dfs) / len(dfs),
    })

    best_idx     = df_avg["val_accuracy"].idxmax()
    best_epoch   = int(df_avg.loc[best_idx, "epoch"])
    best_val_acc = float(df_avg.loc[best_idx, "val_accuracy"])
    last_epoch   = int(df_avg["epoch"].iloc[-1])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Training history — 1D-CNN TCGA pan-cancer "
        f"({len(available)} models, averaged, {last_epoch} epochs shown)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, y_train, y_val, ylabel, title in [
        (axes[0], "accuracy",  "val_accuracy", "Accuracy",  "Accuracy over epochs"),
        (axes[1], "loss",      "val_loss",     "Loss",      "Loss over epochs"),
    ]:
        # Thin per-seed lines for context
        for df_s, lbl in zip(dfs, seed_labels):
            ep = df_s["epoch"].values[:min_len]
            ax.plot(ep, df_s[y_train].values[:min_len],
                    color="#90CAF9", linewidth=0.6, alpha=0.5)
            ax.plot(ep, df_s[y_val].values[:min_len],
                    color="#FFAB91", linewidth=0.6, alpha=0.5)

        # Bold averaged lines
        ax.plot(df_avg["epoch"], df_avg[y_train], color="#2196F3",
                linewidth=2.0, label="Train (avg)")
        ax.plot(df_avg["epoch"], df_avg[y_val],   color="#FF5722",
                linewidth=2.0, label="Val (avg)")

        ax.axvline(best_epoch, color="#4CAF50", linestyle="--", linewidth=1.6,
                   label=f"Best epoch {best_epoch}  (val={best_val_acc:.3f})")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    axes[0].set_ylim(0, 1.05)

    out = PLOTS_DIR / "training_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"  Models averaged : {len(available)}  ({', '.join(seed_labels)})")
    print(f"  Best avg epoch  : {best_epoch}  val_acc={best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Figure 3 — Class distribution bar chart
# ---------------------------------------------------------------------------
def plot_distribution() -> None:
    # Try metrics file first (has class counts in distribution block)
    # Fall back to summary file
    source = METRICS_FILE if METRICS_FILE.exists() else SUMMARY_FILE
    if not source.exists():
        raise FileNotFoundError(
            f"Neither {METRICS_FILE} nor {SUMMARY_FILE} found.\n"
            "Run train_merged.py first."
        )

    counts: dict[str, int] = {}
    pattern = re.compile(r"^\s*\d+\s*\|\s*(.+?)\s*:\s*(\d+)\s*$")
    in_block = False

    with open(source, encoding="utf-8") as f:
        for line in f:
            if "Class distribution" in line or "Cancer type" in line.lower():
                in_block = True
                continue
            if in_block:
                m = pattern.match(line)
                if m:
                    counts[m.group(1).strip().lower()] = int(m.group(2))
                elif line.strip() == "" and counts:
                    break

    if not counts:
        raise ValueError(f"Could not parse class distribution from {source}")

    labels = [abbrev(n) for n in counts]
    values = list(counts.values())
    total  = sum(values)
    mean_v = total // len(values)
    max_i  = values.index(max(values))
    min_i  = values.index(min(values))
    ratio  = max(values) / max(min(values), 1)

    colors            = ["#90CAF9"] * len(labels)
    colors[max_i]     = "#F44336"
    colors[min_i]     = "#FF9800"

    apply_style()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(mean_v, color="#607D8B", linestyle="--", linewidth=1.2,
               label=f"Mean ({mean_v} samples)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_xlabel("Cancer type (TCGA abbreviation)", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_title(
        f"Class distribution — {len(labels)} cancer types  |  "
        f"Total = {total:,} samples  |  Imbalance ratio = {ratio:.0f}:1",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    for i, (idx, col) in enumerate([(max_i, "#C62828"), (min_i, "#E65100")]):
        ax.text(idx, values[idx] + total * 0.002, str(values[idx]),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)

    ax.legend(handles=[
        mpatches.Patch(facecolor="#F44336", label=f"{labels[max_i]} — largest ({values[max_i]} samples)"),
        mpatches.Patch(facecolor="#FF9800", label=f"{labels[min_i]} — smallest ({values[min_i]} samples)"),
        mpatches.Patch(facecolor="#90CAF9", label="Other cancer types"),
        plt.Line2D([0],[0], color="#607D8B", linestyle="--", linewidth=1.2,
                   label=f"Mean ({mean_v} samples)"),
    ], fontsize=9, loc="upper right")

    out = PLOTS_DIR / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"  Classes={len(labels)}  Total={total:,}  "
          f"Largest={labels[max_i]}({values[max_i]})  "
          f"Smallest={labels[min_i]}({values[min_i]})  "
          f"Ratio={ratio:.1f}:1")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
FIGURES = {"f1": plot_f1, "curves": plot_curves, "dist": plot_distribution}

if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    to_run = [FIGURES[a] for a in args if a in FIGURES] or list(FIGURES.values())

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for fn in to_run:
        try:
            fn()
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {fn.__name__}: {e}")