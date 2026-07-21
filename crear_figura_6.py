from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "figuras_articulo"
OUTPUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Nominal": (
        BASE
        / "resultados_comparacion_5_metodos_nominal_170000_170199"
        / "metricas_predictivas_por_episodio.csv"
    ),
    "Moderate": (
        BASE
        / "resultados_comparacion_5_metodos_incertidumbre_moderada_170000_170199"
        / "metricas_predictivas_por_episodio.csv"
    ),
    "Severe": (
        BASE
        / "resultados_comparacion_5_metodos_incertidumbre_severa_170000_170199"
        / "metricas_predictivas_por_episodio.csv"
    ),
}

METRICS = [
    "Collision Brier",
    "Collision ECE",
    "Near-risk Brier",
    "Near-risk ECE",
]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

rows = []

for condition, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    data = pd.read_csv(path)
    upo = data[data["metodo"] == "A* + SAC-UPO-TTC"].copy()

    if upo.empty:
        raise ValueError(f"No SAC-UPO-TTC rows were found in {path}")

    rows.append(
        {
            "Condition": condition,
            "Collision Brier": upo["brier_probabilidad_colision"].mean(),
            "Collision ECE": upo["ece_probabilidad_colision"].mean(),
            "Near-risk Brier": upo[
                "brier_probabilidad_casi_colision"
            ].mean(),
            "Near-risk ECE": upo[
                "ece_probabilidad_casi_colision"
            ].mean(),
            "Mean collision probability": upo[
                "probabilidad_colision_media"
            ].mean(),
            "Mean near-risk probability": upo[
                "probabilidad_casi_colision_media"
            ].mean(),
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT / "resumen_calibracion_figura6.csv", index=False)

x = np.arange(len(METRICS))
width = 0.24
fig, ax = plt.subplots(figsize=(7.2, 3.8))

for index, condition in enumerate(summary["Condition"]):
    values = summary.loc[
        summary["Condition"] == condition, METRICS
    ].iloc[0].to_numpy(float)

    offset = (index - 1) * width
    bars = ax.bar(x + offset, values, width, label=condition)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.00035,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

ax.set_xticks(x, METRICS)
ax.set_ylabel("Score")
ax.set_title("Aggregate probability quality of SAC-UPO-TTC")
ax.set_ylim(0, max(summary[METRICS].to_numpy().ravel()) * 1.35)
ax.grid(axis="y", alpha=0.25)
ax.legend(
    title="Perception condition",
    frameon=False,
    ncol=3,
    loc="upper left",
)

fig.tight_layout()
fig.savefig(OUTPUT / "Fig06_probability_calibration.pdf", bbox_inches="tight")
fig.savefig(
    OUTPUT / "Fig06_probability_calibration.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(summary.to_string(index=False))
