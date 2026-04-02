"""
Figure 3 – Training & Validation Curves
========================================
Reads  : results/training_history.csv   (written by train.py)
Writes : results/training_curves.png

Run after train.py has completed at least one full training run.
Re-run any time to regenerate the figure from the saved CSV.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
HISTORY_CSV = Path("results/training_history.csv")
OUTPUT_PNG  = Path("results/training_curves.png")


def main() -> None:
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {HISTORY_CSV}. Run train.py first."
        )

    df = pd.read_csv(HISTORY_CSV)

    # ── derive key events from the data itself ────────────────────────────────
    best_idx     = df["val_accuracy"].idxmax()
    best_epoch   = int(df.loc[best_idx, "epoch"])
    best_val_acc = float(df.loc[best_idx, "val_accuracy"])
    last_epoch   = int(df["epoch"].iloc[-1])

    # LR reduction points: any epoch where lr dropped vs the previous epoch
    lr_drops = df[df["lr"] < df["lr"].shift(1)]["epoch"].tolist()

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Training History – 1D-CNN on TCGA Pan-Cancer ({last_epoch} Epochs)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Accuracy ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(df["epoch"], df["accuracy"],     color="#2196F3", linewidth=1.8, label="Train Accuracy")
    ax.plot(df["epoch"], df["val_accuracy"], color="#FF5722", linewidth=1.8, label="Val Accuracy")

    for i, ep in enumerate(lr_drops):
        ax.axvline(ep, color="#9C27B0", linestyle=":", linewidth=1.0,
                   label="LR Reduction" if i == 0 else "_nolegend_")

    ax.axvline(best_epoch, color="#4CAF50", linestyle="--", linewidth=1.8,
               label=f"Best Epoch {best_epoch}  (val={best_val_acc:.3f})")
    ax.plot(best_epoch, best_val_acc, "o", color="#4CAF50", markersize=8, zorder=5)
    ax.annotate(
        f"  Epoch {best_epoch}\nVal={best_val_acc:.3f}",
        xy=(best_epoch, best_val_acc),
        xytext=(best_epoch + 3, best_val_acc - 0.06),
        fontsize=8, color="#4CAF50",
        arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.2),
    )
    ax.axvline(last_epoch, color="#607D8B", linestyle="-.", linewidth=1.4,
               label=f"Early Stop (Epoch {last_epoch})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy over Epochs", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(df["epoch"], df["loss"],     color="#2196F3", linewidth=1.8, label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], color="#FF5722", linewidth=1.8, label="Val Loss")

    for i, ep in enumerate(lr_drops):
        ax.axvline(ep, color="#9C27B0", linestyle=":", linewidth=1.0,
                   label="LR Reduction" if i == 0 else "_nolegend_")

    ax.axvline(best_epoch, color="#4CAF50", linestyle="--", linewidth=1.8,
               label=f"Best Epoch {best_epoch}")
    ax.axvline(last_epoch, color="#607D8B", linestyle="-.", linewidth=1.4,
               label=f"Early Stop (Epoch {last_epoch})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss over Epochs", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_PNG}")
    print(f"  Best epoch : {best_epoch}  (val_accuracy={best_val_acc:.4f})")
    print(f"  LR drops   : {lr_drops}")
    print(f"  Total epochs trained: {last_epoch}")


if __name__ == "__main__":
    main()
