"""
Genera gráficos PNG de las curvas de entrenamiento para la presentación:
  - SegNT: F1 val y loss por época
  - Inpactor 2 Detect: loss y F1 por época
  - Inpactor 2 Class: loss y F1 por época (últimas 5 épocas visibles)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "slides_png"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Datos reales ----------

# SegNT (Colab, 13 épocas antes de que muriera la sesión)
segnt = {
    "epoch":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "loss":   [1.1129, 1.0380, 1.0174, 0.9943, 0.9991, 0.9753, 0.9719,
               0.9583, 0.9508, 0.9498, 0.9481, 0.9376, 0.9325],
    "f1":     [0.870, 0.877, 0.883, 0.881, 0.878, 0.884, 0.885,
               0.884, 0.877, 0.885, 0.886, 0.886, 0.885],
    "prec":   [0.77, 0.78, 0.79, 0.79, 0.78, 0.79, 0.79,
               0.79, 0.78, 0.79, 0.79, 0.80, 0.79],
    "rec":    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
               1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "best":   12,
}

# Inpactor 2 Detect (5 épocas)
detect = {
    "epoch": [1, 2, 3, 4, 5],
    "loss":  [0.8127, 0.6626, 4.8071, 0.6535, 0.6179],
    "f1":    [0.4595, 0.3451, 0.3828, 0.3083, 0.3920],
    "best":  5,
}

# Inpactor 2 Class (últimas 5 épocas visibles)
class_i2 = {
    "epoch": [18, 19, 20, 21, 22],
    "loss":  [1.9551, 1.6033, 1.7395, 1.7890, 1.5926],
    "f1":    [0.7500, 0.7656, 0.7969, 0.8100, 0.7974],
    "best":  21,
}


def curve_segnt():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # F1 y Precision
    ax1.plot(segnt["epoch"], segnt["f1"], "o-", label="F1 val",
             color="#4C1D95", linewidth=2, markersize=7)
    ax1.plot(segnt["epoch"], segnt["prec"], "s--", label="Precision",
             color="#059669", linewidth=2, markersize=6)
    ax1.plot(segnt["epoch"], segnt["rec"], "^:", label="Recall",
             color="#DC2626", linewidth=2, markersize=6)
    ax1.axvline(segnt["best"], color="#B41E1E", linestyle=":", alpha=0.5)
    ax1.annotate(f"best F1=0.886\n(ep 12)",
                 xy=(segnt["best"], 0.886), xytext=(9, 0.93),
                 fontsize=11, color="#B41E1E", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#B41E1E"))
    ax1.set_xlabel("Época", fontsize=12)
    ax1.set_ylabel("Métrica val", fontsize=12)
    ax1.set_title("SegNT · Métricas por época", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.7, 1.05)

    # Loss
    ax2.plot(segnt["epoch"], segnt["loss"], "o-", color="#4C1D95",
             linewidth=2, markersize=7)
    ax2.set_xlabel("Época", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("SegNT · Loss de entrenamiento", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.annotate(f"1.11 → 0.93", xy=(7, 1.0), fontsize=12, color="#4C1D95")

    fig.suptitle("Inpactor 3-SegNT · 13 épocas sobre PanTEon cross-kingdom "
                 "(52k train + 4k val)", fontsize=15, y=1.02)
    fig.tight_layout()
    out = OUT / "curve_segnt.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[ok] {out}")
    plt.close(fig)


def curve_inpactor2():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Detect · F1
    ax = axes[0, 0]
    ax.plot(detect["epoch"], detect["f1"], "o-", color="#1D4ED8",
            linewidth=2, markersize=8)
    ax.set_xlabel("Época")
    ax.set_ylabel("F1 val")
    ax.set_title("Inpactor 2 · Detect · F1 val (5 épocas)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.6)
    ax.axhline(detect["f1"][-1], color="#B41E1E", linestyle=":", alpha=0.5)
    ax.annotate(f"F1 final = {detect['f1'][-1]:.2f}",
                xy=(5, detect["f1"][-1]), xytext=(3.5, 0.5),
                fontsize=11, color="#B41E1E", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B41E1E"))

    # Detect · loss
    ax = axes[0, 1]
    ax.plot(detect["epoch"], detect["loss"], "s-", color="#1D4ED8",
            linewidth=2, markersize=8)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss val")
    ax.set_title("Inpactor 2 · Detect · Loss val", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.annotate("outlier ep 3\n(inestabilidad)", xy=(3, 4.81),
                xytext=(3.5, 3.5), fontsize=10, color="#B41E1E",
                arrowprops=dict(arrowstyle="->", color="#B41E1E"))

    # Class · F1
    ax = axes[1, 0]
    ax.plot(class_i2["epoch"], class_i2["f1"], "o-", color="#059669",
            linewidth=2, markersize=8)
    ax.set_xlabel("Época")
    ax.set_ylabel("F1 val")
    ax.set_title("Inpactor 2 · Class · F1 val (épocas 18-22)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.7, 0.85)
    ax.axvline(class_i2["best"], color="#B41E1E", linestyle=":", alpha=0.5)
    ax.annotate(f"best F1=0.81\n(ep 21)",
                xy=(21, 0.81), xytext=(18.5, 0.83),
                fontsize=11, color="#B41E1E", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B41E1E"))

    # Class · loss
    ax = axes[1, 1]
    ax.plot(class_i2["epoch"], class_i2["loss"], "s-", color="#059669",
            linewidth=2, markersize=8)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss val")
    ax.set_title("Inpactor 2 · Class · Loss val", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.suptitle("Inpactor 2 re-entrenado con PanTEon · métricas val",
                 fontsize=15, y=1.00, fontweight="bold")
    fig.tight_layout()
    out = OUT / "curve_inpactor2.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[ok] {out}")
    plt.close(fig)


def curve_comparativa():
    """Comparación lado a lado del F1 val."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # SegNT
    ax.plot(segnt["epoch"], segnt["f1"], "o-", color="#4C1D95",
            linewidth=2.5, markersize=8, label="SegNT (Nucleotide Transformer)")

    # Inpactor 2 Class re-mapeado sobre eje homogeneizado
    # Class va de época 18-22, la ploteamos como si empezara en 1
    class_ep_norm = list(range(1, len(class_i2["epoch"]) + 1))
    ax.plot(class_ep_norm, class_i2["f1"], "s-", color="#059669",
            linewidth=2.5, markersize=8,
            label="Inpactor 2 Class (últimas 5 épocas)")

    # Detect
    ax.plot(detect["epoch"], detect["f1"], "^-", color="#1D4ED8",
            linewidth=2.5, markersize=8,
            label="Inpactor 2 Detect")

    # Baseline SegNT anterior
    ax.axhline(0.14, color="#94A3B8", linestyle="--", linewidth=1.5)
    ax.text(9, 0.16, "SegNT anterior (solo InpactorDB plantas): F1=0.14",
            fontsize=10, color="#64748B", style="italic")

    ax.set_xlabel("Época", fontsize=13)
    ax.set_ylabel("F1 val", fontsize=13)
    ax.set_title("Comparación · F1 val por época de los 3 modelos entrenados",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="center right", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Anotaciones de mejores
    ax.annotate("0.886", xy=(12, 0.886), xytext=(12, 0.95),
                fontsize=12, fontweight="bold", color="#4C1D95",
                ha="center", arrowprops=dict(arrowstyle="->", color="#4C1D95"))
    ax.annotate("0.81", xy=(4, 0.81), xytext=(4, 0.88),
                fontsize=12, fontweight="bold", color="#059669",
                ha="center", arrowprops=dict(arrowstyle="->", color="#059669"))
    ax.annotate("0.39", xy=(5, 0.39), xytext=(6.5, 0.30),
                fontsize=12, fontweight="bold", color="#1D4ED8",
                arrowprops=dict(arrowstyle="->", color="#1D4ED8"))

    fig.tight_layout()
    out = OUT / "curve_comparativa.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[ok] {out}")
    plt.close(fig)


if __name__ == "__main__":
    curve_segnt()
    curve_inpactor2()
    curve_comparativa()
