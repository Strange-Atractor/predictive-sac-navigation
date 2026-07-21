from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

BASE = Path(__file__).resolve().parent

ARCHIVOS = {
    "Nominal": (
        BASE
        / "resultados_comparacion_5_metodos_nominal_170000_170199"
        / "resumen_por_metodo.csv"
    ),
    "Moderate": (
        BASE
        / "resultados_comparacion_5_metodos_incertidumbre_moderada_170000_170199"
        / "resumen_por_metodo.csv"
    ),
    "Severe": (
        BASE
        / "resultados_comparacion_5_metodos_incertidumbre_severa_170000_170199"
        / "resumen_por_metodo.csv"
    ),
}

DIRECTORIO_SALIDA = BASE / "figuras_articulo"
DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# ORDEN FIJO DE MÉTODOS
# ============================================================

METODOS = [
    "A* + DWA",
    "A* + TD3",
    "A* + SAC-R",
    "A* + SAC-PO-TTC",
    "A* + SAC-UPO-TTC",
]

CONDICIONES = ["Nominal", "Moderate", "Severe"]

MARCADORES = ["o", "s", "^", "D", "P"]
ESTILOS = ["-", "--", "-.", ":", "-"]


# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# LECTURA Y VALIDACIÓN
# ============================================================

def cargar_resultados() -> pd.DataFrame:
    """Carga los tres archivos resumen_por_metodo.csv."""

    tablas = []

    for condicion, ruta in ARCHIVOS.items():
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo:\n{ruta}\n\n"
                "Revisa el nombre de la carpeta y del archivo CSV."
            )

        tabla = pd.read_csv(ruta)
        tabla["condicion_articulo"] = condicion
        tablas.append(tabla)

    datos = pd.concat(tablas, ignore_index=True)

    columnas_necesarias = {
        "metodo",
        "tasa_exito",
        "ic_wilson_95_inferior",
        "ic_wilson_95_superior",
        "tasa_colision_dinamica",
        "tasa_episodios_con_casi_colision",
        "spl_medio",
        "clearance_dinamico_minimo_media_m",
        "ttc_real_minimo_media_s",
        "tiempo_ciclo_medio_ms",
    }

    faltantes = columnas_necesarias - set(datos.columns)

    if faltantes:
        raise KeyError(
            "Faltan estas columnas en resumen_por_metodo.csv:\n"
            + "\n".join(sorted(faltantes))
        )

    return datos


def obtener_serie(
    datos: pd.DataFrame,
    metodo: str,
    columna: str,
    factor: float = 1.0,
) -> np.ndarray:
    """Extrae una métrica respetando el orden de las condiciones."""

    valores = []

    for condicion in CONDICIONES:
        seleccion = datos[
            (datos["metodo"] == metodo)
            & (datos["condicion_articulo"] == condicion)
        ]

        if len(seleccion) != 1:
            raise ValueError(
                f"Se esperaba una fila para {metodo} - {condicion}, "
                f"pero se encontraron {len(seleccion)}."
            )

        valor = float(seleccion.iloc[0][columna])
        valores.append(factor * valor)

    return np.asarray(valores, dtype=float)


def guardar_figura(figura: plt.Figure, nombre: str) -> None:
    """Guarda una figura en PDF y PNG."""

    ruta_pdf = DIRECTORIO_SALIDA / f"{nombre}.pdf"
    ruta_png = DIRECTORIO_SALIDA / f"{nombre}.png"

    figura.savefig(ruta_pdf, bbox_inches="tight")
    figura.savefig(ruta_png, bbox_inches="tight", dpi=300)
    plt.close(figura)

    print(f"Creada: {ruta_pdf}")
    print(f"Creada: {ruta_png}")


# ============================================================
# FIGURA 2: ÉXITO Y SPL
# ============================================================

def crear_figura_2(datos: pd.DataFrame) -> None:
    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    x = np.arange(len(CONDICIONES))

    # Panel (a): success rate con intervalos Wilson
    for indice, metodo in enumerate(METODOS):
        exito = obtener_serie(
            datos,
            metodo,
            "tasa_exito",
            factor=100.0,
        )

        inferior = obtener_serie(
            datos,
            metodo,
            "ic_wilson_95_inferior",
            factor=100.0,
        )

        superior = obtener_serie(
            datos,
            metodo,
            "ic_wilson_95_superior",
            factor=100.0,
        )

        errores = np.vstack(
            [
                exito - inferior,
                superior - exito,
            ]
        )

        ejes[0].errorbar(
            x,
            exito,
            yerr=errores,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.4,
            markersize=5,
            capsize=2.5,
            label=metodo,
        )

    ejes[0].set_xticks(x, CONDICIONES)
    ejes[0].set_ylabel("Success rate (%)")
    ejes[0].set_xlabel("Perception condition")
    ejes[0].set_title("(a) Navigation success")
    ejes[0].set_ylim(55, 103)
    ejes[0].grid(True, alpha=0.25)

    # Panel (b): SPL
    for indice, metodo in enumerate(METODOS):
        spl = obtener_serie(datos, metodo, "spl_medio")

        ejes[1].plot(
            x,
            spl,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.4,
            markersize=5,
            label=metodo,
        )

    ejes[1].set_xticks(x, CONDICIONES)
    ejes[1].set_ylabel("SPL")
    ejes[1].set_xlabel("Perception condition")
    ejes[1].set_title("(b) Path efficiency")
    ejes[1].set_ylim(0.60, 1.01)
    ejes[1].grid(True, alpha=0.25)

    handles, labels = ejes[0].get_legend_handles_labels()

    figura.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    figura.tight_layout(rect=(0, 0.12, 1, 1))
    guardar_figura(figura, "Fig02_navigation_performance")


# ============================================================
# FIGURA 3: COLISIONES Y CASI COLISIONES
# ============================================================

def crear_figura_3(datos: pd.DataFrame) -> None:
    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    x = np.arange(len(CONDICIONES))

    for indice, metodo in enumerate(METODOS):
        colisiones = obtener_serie(
            datos,
            metodo,
            "tasa_colision_dinamica",
            factor=100.0,
        )

        casi_colisiones = obtener_serie(
            datos,
            metodo,
            "tasa_episodios_con_casi_colision",
            factor=100.0,
        )

        ejes[0].plot(
            x,
            colisiones,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.4,
            markersize=5,
            label=metodo,
        )

        ejes[1].plot(
            x,
            casi_colisiones,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.4,
            markersize=5,
            label=metodo,
        )

    ejes[0].set_xticks(x, CONDICIONES)
    ejes[0].set_ylabel("Dynamic-collision rate (%)")
    ejes[0].set_xlabel("Perception condition")
    ejes[0].set_title("(a) Dynamic collisions")
    ejes[0].set_ylim(bottom=0)
    ejes[0].grid(True, alpha=0.25)

    ejes[1].set_xticks(x, CONDICIONES)
    ejes[1].set_ylabel("Episodes with near collision (%)")
    ejes[1].set_xlabel("Perception condition")
    ejes[1].set_title("(b) Real near-collision exposure")
    ejes[1].set_ylim(bottom=0)
    ejes[1].grid(True, alpha=0.25)

    handles, labels = ejes[0].get_legend_handles_labels()

    figura.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    figura.tight_layout(rect=(0, 0.12, 1, 1))
    guardar_figura(figura, "Fig03_collision_risk")


# ============================================================
# FIGURA 4: CLEARANCE Y TTC
# ============================================================

def crear_figura_4(datos: pd.DataFrame) -> None:
    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    x = np.arange(len(CONDICIONES))

    for indice, metodo in enumerate(METODOS):
        clearance = obtener_serie(
            datos,
            metodo,
            "clearance_dinamico_minimo_media_m",
        )

        ttc = obtener_serie(
            datos,
            metodo,
            "ttc_real_minimo_media_s",
        )

        ejes[0].plot(
            x,
            clearance,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.4,
            markersize=5,
            label=metodo,
        )

        ejes[1].plot(
            x,
            ttc,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.4,
            markersize=5,
            label=metodo,
        )

    ejes[0].set_xticks(x, CONDICIONES)
    ejes[0].set_ylabel("Mean minimum dynamic clearance (m)")
    ejes[0].set_xlabel("Perception condition")
    ejes[0].set_title("(a) Geometric safety margin")
    ejes[0].set_ylim(bottom=0)
    ejes[0].grid(True, alpha=0.25)

    ejes[1].set_xticks(x, CONDICIONES)
    ejes[1].set_ylabel("Mean minimum real TTC (s)")
    ejes[1].set_xlabel("Perception condition")
    ejes[1].set_title("(b) Temporal safety margin")
    ejes[1].set_ylim(bottom=0)
    ejes[1].grid(True, alpha=0.25)

    handles, labels = ejes[0].get_legend_handles_labels()

    figura.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    figura.tight_layout(rect=(0, 0.12, 1, 1))
    guardar_figura(figura, "Fig04_safety_margins")


# ============================================================
# FIGURA 5: TIEMPO DE CÓMPUTO
# ============================================================

def crear_figura_5(datos: pd.DataFrame) -> None:
    figura, eje = plt.subplots(figsize=(6.2, 3.7))

    x = np.arange(len(CONDICIONES))

    for indice, metodo in enumerate(METODOS):
        tiempo = obtener_serie(
            datos,
            metodo,
            "tiempo_ciclo_medio_ms",
        )

        eje.plot(
            x,
            tiempo,
            marker=MARCADORES[indice],
            linestyle=ESTILOS[indice],
            linewidth=1.5,
            markersize=5,
            label=metodo,
        )

    eje.set_xticks(x, CONDICIONES)
    eje.set_xlabel("Perception condition")
    eje.set_ylabel("Mean controller-cycle time (ms)")
    eje.set_title("Observed controller-cycle computation time")
    eje.set_yscale("log")
    eje.grid(True, which="both", alpha=0.25)

    eje.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    figura.tight_layout()
    guardar_figura(figura, "Fig05_computation_time")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    try:
        datos = cargar_resultados()

        print("Archivos cargados correctamente.")
        print(datos[["condicion_articulo", "metodo"]])

        crear_figura_2(datos)
        crear_figura_3(datos)
        crear_figura_4(datos)
        crear_figura_5(datos)

        print("\nTodas las figuras fueron creadas correctamente.")
        print(f"Carpeta de salida:\n{DIRECTORIO_SALIDA}")

    except Exception as error:
        print("\nERROR AL CREAR LAS FIGURAS")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()