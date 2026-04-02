"""
Figure 5 – Per-Class F1-Score Bar Chart
=========================================
Reads  : results/evaluation_metrics.txt   (written by train.py)
Writes : results/f1_scores_per_class.png

Run after train.py has completed at least one full training run.
Re-run any time to regenerate the figure with the latest metrics.
"""

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
METRICS_TXT = Path("results/evaluation_metrics.txt")
OUTPUT_PNG  = Path("results/f1_scores_per_class.png")

# Full disease name → TCGA abbreviation
NAME_TO_ABBREV = {
    "skin cutaneous melanoma":              "SKCM",
    "thyroid carcinoma":                    "THCA",
    "sarcoma":                              "SARC",
    "prostate adenocarcinoma":              "PRAD",
    "pheochromocytoma & paraganglioma":     "PCPG",
    "pancreatic adenocarcinoma":            "PAAD",
    "head & neck squamous cell carcinoma":  "HNSC",
    "esophageal carcinoma":                 "ESCA",
    "colon adenocarcinoma":                 "COAD",
    "cervical & endocervical cancer":       "CESC",
    "breast invasive carcinoma":            "BRCA",
    "bladder urothelial carcinoma":         "BLCA",
    "testicular germ cell tumor":           "TGCT",
    "kidney papillary cell carcinoma":      "KIRP",
    "kidney clear cell carcinoma":          "KIRC",
    "acute myeloid leukemia":               "LAML",
    "rectum adenocarcinoma":                "READ",
    "ovarian serous cystadenocarcinoma":    "OV",
    "lung adenocarcinoma":                  "LUAD",
    "liver hepatocellular carcinoma":       "LIHC",
    "uterine corpus endometrioid carcinoma":"UCEC",
    "glioblastoma multiforme":              "GBM",
    "brain lower grade glioma":             "LGG",
    "uterine carcinosarcoma":               "UCS",
    "thymoma":                              "THYM",
    "stomach adenocarcinoma":               "STAD",
    "diffuse large B-cell lymphoma":        "DLBC",
    "lung squamous cell carcinoma":         "LUSC",
    "mesothelioma":                         "MESO",
    "kidney chromophobe":                   "KICH",
    "uveal melanoma":                       "UVM",
    "cholangiocarcinoma":                   "CHOL",
    "adrenocortical cancer":                "ACC",
}


def load_metrics(path: Path) -> tuple[float, float, list[tuple[str, float]]]:
    """
    Parse evaluation_metrics.txt.
    Returns (test_accuracy, macro_f1, [(abbrev, f1_score), ...])
    Lines look like:
        Test Accuracy: 0.864882
        Macro Avg F1: 0.833891
           0 | skin cutaneous melanoma       : 0.853503
    """
    test_acc  = 0.0
    macro_f1  = 0.0
    per_class: list[tuple[str, float]] = []

    line_pat = re.compile(r"^\s*\d+\s*\|\s*(.+?)\s*:\s*([\d.]+)\s*$")

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Test Accuracy:"):
                test_acc = float(line.split(":")[1].strip())
            elif line.startswith("Macro Avg F1:"):
                macro_f1 = float(line.split(":")[1].strip())
            else:
                m = line_pat.match(line)
                if m:
                    name  = m.group(1).strip().lower()
                    score = float(m.group(2))
                    abbrev = NAME_TO_ABBREV.get(name, name.upper()[:4])
                    per_class.append((abbrev, score))

    if not per_class:
        raise ValueError("Could not parse per-class F1 scores from evaluation_metrics.txt.")
    return test_acc, macro_f1, per_class


def main() -> None:
    if not METRICS_TXT.exists():
        raise FileNotFoundError(
            f"Cannot find {METRICS_TXT}. Run train.py first."
        )

    test_acc, macro_f1, per_class = load_metrics(METRICS_TXT)

    # Sort by F1 descending
    per_class.sort(key=lambda x: x[1], reverse=True)
    abbrevs = [r[0] for r in per_class]
    f1_vals  = [r[1] for r in per_class]

    # Color tiers
    colors = [
        "#4CAF50" if v >= 0.85 else "#FF9800" if v >= 0.70 else "#F44336"
        for v in f1_vals
    ]

    n_excellent = sum(1 for v in f1_vals if v >= 0.85)
    n_moderate  = sum(1 for v in f1_vals if 0.70 <= v < 0.85)
    n_poor      = sum(1 for v in f1_vals if v < 0.70)

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(abbrevs)), f1_vals, color=colors, edgecolor="white", linewidth=0.6)

    ax.axhline(0.80,     color="#2196F3", linestyle="--", linewidth=1.8, label="F1 = 0.80 threshold")
    ax.axhline(macro_f1, color="#607D8B", linestyle="-",  linewidth=1.5,
               label=f"Macro Avg F1 = {macro_f1:.3f}")

    ax.set_xticks(range(len(abbrevs)))
    ax.set_xticklabels(abbrevs, rotation=90, fontsize=9)
    ax.set_xlabel("Cancer Type (sorted by F1-Score)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title(
        f"Per-Class F1-Scores on Test Set  |  "
        f"Macro Avg F1 = {macro_f1:.3f}  |  Test Accuracy = {test_acc * 100:.1f}%",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)

    legend_els = [
        mpatches.Patch(facecolor="#4CAF50", label=f"F1 ≥ 0.85 – Excellent  ({n_excellent} classes)"),
        mpatches.Patch(facecolor="#FF9800", label=f"0.70 ≤ F1 < 0.85 – Moderate  ({n_moderate} classes)"),
        mpatches.Patch(facecolor="#F44336", label=f"F1 < 0.70 – Needs Improvement  ({n_poor} classes)"),
        plt.Line2D([0], [0], color="#2196F3", linestyle="--", linewidth=1.8, label="F1 = 0.80 threshold"),
        plt.Line2D([0], [0], color="#607D8B", linestyle="-",  linewidth=1.5,
                   label=f"Macro Avg F1 = {macro_f1:.3f}"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower left")

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_PNG}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Macro Avg F1  : {macro_f1:.4f}")
    print(f"  Excellent (>=0.85)    : {n_excellent} classes")
    print(f"  Moderate  (0.70-0.85) : {n_moderate} classes")
    print(f"  Poor      (<0.70) : {n_poor} classes")
    print(f"  Top 3  : {abbrevs[:3]}")
    print(f"  Bottom 3: {abbrevs[-3:]}")


if __name__ == "__main__":
    main()
