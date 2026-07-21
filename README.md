# Predictive SAC Navigation under Noisy and Delayed Perception

This repository contains the exact audited controller implementations,
selected frozen actor checkpoints, training curves, paired evaluation data,
statistical outputs, and article-figure scripts associated with the study:

**Deterministic and Probabilistic Predictive Soft Actor-Critic Control for
Mobile Robot Navigation under Noisy and Delayed Perception**

## Controllers

- A* + Dynamic Window Approach (DWA)
- A* + Twin Delayed Deep Deterministic Policy Gradient (TD3)
- A* + reactive Soft Actor-Critic (SAC-R)
- A* + deterministic predictive SAC (SAC-PO-TTC)
- A* + uncertainty-aware probabilistic occupancy SAC (SAC-UPO-TTC)

## Why the Python files remain at the repository root

Several audited programs import one another by exact filename using
`Path(__file__).resolve().parent` and `importlib`. Their original filenames and
co-location are therefore preserved to retain executable compatibility.

## Main programs

- `SAC_predictivo_entrenamiento_completo.py`
- `SAC_reactivo_mejorado_entrenamiento_tercera_corrida.py`
- `SAC_probabilistico_ocupacion_TTC_entrenamiento.py`
- `TD3_tercera_corrida_desde_cero.py`
- `Comparador_5_metodos_nominal_metricas_ampliadas.py`
- `Comparador_5_metodos_fase2_incertidumbre_comun.py`
- `crear_figuras_articulo.py`
- `crear_figura_6.py`

## Data structure

- `resultados_sac/`: selected SAC checkpoints, periodic validation CSV files,
  training curves, and independent 100-seed validation outputs.
- `resultados_td3/`: selected TD3 checkpoint, training curve, periodic
  validation CSV, and independent 100-seed validation outputs.
- `resultados_comparacion_5_metodos_nominal_170000_170199/`: nominal paired
  evaluation results for 200 scenarios.
- `resultados_comparacion_5_metodos_incertidumbre_moderada_170000_170199/`:
  moderate-noise/delay paired evaluation results.
- `resultados_comparacion_5_metodos_incertidumbre_severa_170000_170199/`:
  severe-noise/delay paired evaluation results.
- `resultados_robustez_tres_condiciones_170000_170199/`: cross-condition
  summaries and degradation data.
- `figuras_articulo/`: final article figures and the Figure 6 summary CSV.
- `figuras_fuente/`: editable source of the framework diagram.
- `metadata/`: file manifest, SHA-256 checksums, and packaging notes.

## Reproducing the paired evaluations

Nominal condition:

```bash
python Comparador_5_metodos_nominal_metricas_ampliadas.py \
  --semilla-base 170000 --numero-semillas 200
```

Moderate and severe conditions:

```bash
python Comparador_5_metodos_fase2_incertidumbre_comun.py \
  --condicion ambas --semilla-base 170000 --numero-semillas 200
```

Article Figures 2--5:

```bash
python crear_figuras_articulo.py
```

Article Figure 6:

```bash
python crear_figura_6.py
```

## Checkpoint policy

The selected frozen actor checkpoints used in the article are included. The
three full SAC optimizer checkpoints were not included because each exceeds
250 MB and is not required to reproduce the reported frozen-controller
evaluations.
