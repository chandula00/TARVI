"""Generate cross-dataset mean performance radar chart (5 datasets, no dentate gyrus)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_PATH = "outputs/cross_dataset_radar.png"
os.makedirs("outputs", exist_ok=True)

# ---- aggregate means (5 datasets: bonemarrow, chromaffin, forebrain, pancreas, sceu_organoid) ----
# NaN entries for scVelo dyn (ICCoH/CBDir/VelConf) are set to 0
METRICS = ["ICCoH", "CBDir", "Vel Conf", "PT-Spear", "PT-DistCorr", "PT-Cons", "Root Acc"]

METHODS = {
    "TARVI (full)":      [0.776, 0.933, 0.927, 0.747, 0.771, 0.986, 0.671],
    "VeloVI":           [0.787, 0.931, 0.920, 0.331, 0.341, 0.904, 0.723],
    "TFvelo":           [0.933, 0.569, 0.915, 0.397, 0.412, 0.794, 0.655],
    "scVelo dyn":       [0.000, 0.000, 0.000, 0.604, 0.628, 0.852, 0.814],  # 3 NaN → 0
    "Velocyto":         [0.813, 0.505, 0.755, 0.700, 0.684, 0.873, 0.745],
}

COLORS = {
    "TARVI (full)":  "#d62728",
    "VeloVI":       "#2ca02c",
    "TFvelo":       "#9467bd",
    "scVelo dyn":   "#1f77b4",
    "Velocyto":     "#ff7f0e",
}

N = len(METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for name, vals in METHODS.items():
    v = vals + vals[:1]
    lw = 2.5 if name == "TARVI (full)" else 1.6
    ls = "-"
    alpha_fill = 0.12 if name == "TARVI (full)" else 0.06
    ax.plot(angles, v, "o-", linewidth=lw, linestyle=ls,
            label=name, color=COLORS[name])
    ax.fill(angles, v, alpha=alpha_fill, color=COLORS[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(METRICS, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

ax.set_title("Cross-dataset mean performance (5 datasets)", fontsize=12,
             fontweight="bold", pad=22)

ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
          fontsize=9, framealpha=0.85)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
