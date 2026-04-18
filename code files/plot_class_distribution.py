"""
Figure 4 – Class Distribution Bar Chart
=========================================
Reads  : results/training_summary.txt   (written by train.py)
         Falls back to labelMapping.py + Data/data.h5 if the summary is absent.
Writes : results/class_distribution.png

Run after train.py has completed at least one full training run.
Re-run any time to regenerate the figure.
"""

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
SUMMARY_TXT = Path("results/training_summary.txt")
OUTPUT_PNG  = Path("results/class_distribution.png")

# Mapping from full disease name (as stored in training_summary.txt) to
# standard TCGA abbreviation shown on the plot axis.
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


def load_counts_from_summary(path: Path) -> dict[str, int]:
    """
    Parse lines like:
       0 | skin cutaneous melanoma                       : 474
    from the 'Cancer type composition' block in training_summary.txt.
    """
    counts: dict[str, int] = {}
    in_block = False
    pattern = re.compile(r"^\s*\d+\s*\|\s*(.+?)\s*:\s*(\d+)\s*$")

    with open(path, encoding="utf-8") as f:
        for line in f:
            if "Cancer type composition" in line:
                in_block = True
                continue
            if in_block:
                m = pattern.match(line)
                if m:
                    name  = m.group(1).strip().lower()
                    count = int(m.group(2))
                    counts[name] = count
                elif line.strip() == "":
                    break  # blank line ends the block

    if not counts:
        raise ValueError("Could not parse class counts from training_summary.txt.")
    return counts


def main() -> None:
    if not SUMMARY_TXT.exists():
        raise FileNotFoundError(
            f"Cannot find {SUMMARY_TXT}. Run train.py first."
        )

    raw_counts = load_counts_from_summary(SUMMARY_TXT)

    # Build ordered lists: class index order preserved from the summary
    abbrevs: list[str] = []
    counts:  list[int] = []
    for name, cnt in raw_counts.items():
        abbrevs.append(NAME_TO_ABBREV.get(name, name.upper()[:4]))
        counts.append(cnt)

    total       = sum(counts)
    max_count   = max(counts)
    min_count   = min(counts)
    max_idx     = counts.index(max_count)
    min_idx     = counts.index(min_count)
    imbalance   = max_count / min_count
    mean_count  = total // len(counts)

    colors = ["#90CAF9"] * len(abbrevs)
    colors[max_idx] = "#F44336"   # largest class – red
    colors[min_idx] = "#FF9800"   # smallest class – orange

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(abbrevs)), counts, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(mean_count, color="#607D8B", linestyle="--", linewidth=1.2,
               label=f"Mean ({mean_count} samples)")

    ax.set_xticks(range(len(abbrevs)))
    ax.set_xticklabels(abbrevs, rotation=90, fontsize=9)
    ax.set_xlabel("Cancer Type (TCGA Abbreviation)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(
        f"Class Distribution Across {len(abbrevs)} Cancer Types  |  "
        f"Total = {total:,} samples  |  Imbalance Ratio = {imbalance:.0f}:1",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    # Annotate max and min bars
    ax.text(max_idx, max_count + total * 0.002, str(max_count),
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#C62828")
    ax.text(min_idx, min_count + total * 0.002, str(min_count),
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E65100")

    legend_els = [
        mpatches.Patch(facecolor="#F44336", label=f"{abbrevs[max_idx]} – Largest ({max_count} samples)"),
        mpatches.Patch(facecolor="#FF9800", label=f"{abbrevs[min_idx]} – Smallest ({min_count} samples)"),
        mpatches.Patch(facecolor="#90CAF9", label="Other Cancer Types"),
        plt.Line2D([0], [0], color="#607D8B", linestyle="--", linewidth=1.2,
                   label=f"Mean ({mean_count} samples)"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="upper right")

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_PNG}")
    print(f"  Classes : {len(abbrevs)}")
    print(f"  Total   : {total:,} samples")
    print(f"  Largest : {abbrevs[max_idx]} = {max_count}")
    print(f"  Smallest: {abbrevs[min_idx]} = {min_count}")
    print(f"  Imbalance ratio: {imbalance:.1f}:1")


if __name__ == "__main__":
    main()
