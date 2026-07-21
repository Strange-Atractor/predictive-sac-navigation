"""
FASE 2 - VERIFICACIÓN DE DISCREPANCIA ENTRE EL MODELO PREDICTIVO
Y EL MOVIMIENTO REAL DE LOS OBSTÁCULOS DINÁMICOS.

Este programa NO entrena, NO modifica checkpoints y NO ejecuta todavía
la comparación completa de controladores. Su propósito es validar que:

1. El caso "modelo_conocido" reproduce exactamente el modelo original.
2. Los perfiles no vistos generan movimiento real distinto del supuesto
   por el predictor determinista.
3. El predictor continúa usando el modelo original de velocidad actual
   constante con rebotes.
4. Los perfiles son deterministas para una misma semilla.
5. Todos los estados permanecen finitos y dentro del mapa.

Coloque este archivo en la misma carpeta que:

    Comparacion_final_DWA_SAC_reactivo_SAC_predictivo.py

Ejecución:

    python Fase2_verificacion_discrepancia_modelo.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import os

import matplotlib
import numpy as np

# En la computadora del usuario se conserva TkAgg. Esta compatibilidad
# sólo permite verificar el archivo en entornos sin interfaz gráfica.
_USAR_AGG = os.environ.get("MPLBACKEND", "").strip().lower() == "agg"
_USAR_MATPLOTLIB_ORIGINAL = matplotlib.use

if _USAR_AGG:
    def _usar_backend_compatible(backend, *args, **kwargs):
        if str(backend).strip().lower() == "tkagg":
            return _USAR_MATPLOTLIB_ORIGINAL("Agg", force=True)
        return _USAR_MATPLOTLIB_ORIGINAL(backend, *args, **kwargs)

    matplotlib.use = _usar_backend_compatible

import Comparacion_final_DWA_SAC_reactivo_SAC_predictivo as base

matplotlib.use = _USAR_MATPLOTLIB_ORIGINAL
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURACIÓN DE LA FASE 2
# ==========================================================

PERFILES_DINAMICA_FASE2: Tuple[str, ...] = (
    "modelo_conocido",
    "aceleracion_suave",
    "stop_and_go",
    "cambio_direccion",
    "curva_sinusoidal",
)

SEMILLA_VERIFICACION_FASE2 = 91000
DT_VERIFICACION_FASE2 = float(base.DT)
HORIZONTE_VERIFICACION_FASE2 = 3.0

HORIZONTES_COMPARACION_FASE2: Tuple[float, ...] = (
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
)

DIRECTORIO_FASE2 = Path(
    "resultados_fase2_robustez/verificacion_discrepancia_modelo"
)

RUTA_CSV_FASE2 = DIRECTORIO_FASE2 / "errores_prediccion_por_perfil.csv"
RUTA_GRAFICA_FASE2 = DIRECTORIO_FASE2 / "trayectorias_reales_vs_predichas.png"
RUTA_RESUMEN_FASE2 = DIRECTORIO_FASE2 / "resumen_verificacion.txt"

# Umbral deliberadamente pequeño: sólo se exige demostrar que el perfil
# deja de coincidir con la predicción nominal.
ERROR_MINIMO_DISCREPANCIA_M = 0.03
TOLERANCIA_MODELO_CONOCIDO_M = 1e-9


# ==========================================================
# UTILIDADES NUMÉRICAS
# ==========================================================

def normalizar_angulo_local(angulo: float) -> float:
    """Normaliza un ángulo al intervalo [-pi, pi)."""

    return float((angulo + math.pi) % (2.0 * math.pi) - math.pi)


def copiar_obstaculos(
    obstaculos: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Crea copias independientes de los diccionarios de obstáculos."""

    return [obstaculo.copy() for obstaculo in obstaculos]


def validar_perfil(perfil: str) -> str:
    """Valida y normaliza el nombre de un perfil dinámico."""

    perfil = str(perfil).strip().lower()

    if perfil not in PERFILES_DINAMICA_FASE2:
        raise ValueError(
            f"Perfil no reconocido: {perfil!r}. "
            f"Perfiles válidos: {PERFILES_DINAMICA_FASE2}."
        )

    return perfil


# ==========================================================
# INICIALIZACIÓN DE METADATOS DE ROBUSTEZ
# ==========================================================

def inicializar_obstaculo_fase2(
    obstaculo: Dict[str, object],
    perfil: str,
    semilla: int,
    indice_obstaculo: int,
) -> Dict[str, object]:
    """
    Añade parámetros deterministas del perfil sin alterar la geometría
    ni el estado cinemático inicial del obstáculo.
    """

    perfil = validar_perfil(perfil)
    actualizado = obstaculo.copy()

    vx = float(actualizado.get("vx", 0.0))
    vy = float(actualizado.get("vy", 0.0))

    velocidad_base = float(math.hypot(vx, vy))
    angulo_base = float(math.atan2(vy, vx)) if velocidad_base > 1e-12 else 0.0

    # Cada obstáculo obtiene parámetros reproducibles, pero diferentes.
    codigo_perfil = PERFILES_DINAMICA_FASE2.index(perfil)
    semilla_local = int(semilla) + 1009 * int(indice_obstaculo) + 7919 * codigo_perfil
    generador = np.random.default_rng(semilla_local)

    actualizado.update(
        {
            "perfil_fase2": perfil,
            "indice_fase2": int(indice_obstaculo),
            "semilla_fase2": int(semilla_local),
            "velocidad_base_fase2": velocidad_base,
            "angulo_base_fase2": angulo_base,
            "fase_fase2": float(generador.uniform(0.0, 2.0 * math.pi)),
            "periodo_fase2": float(generador.uniform(3.5, 5.5)),
            "amplitud_angular_fase2": float(
                generador.uniform(math.radians(22.0), math.radians(35.0))
            ),
            "tiempo_cambio_fase2": float(generador.uniform(0.7, 1.2)),
            "signo_cambio_fase2": float(generador.choice([-1.0, 1.0])),
            "direccion_cambiada_fase2": False,
        }
    )

    return actualizado


def inicializar_obstaculos_fase2(
    obstaculos: Sequence[Dict[str, object]],
    perfil: str,
    semilla: int,
) -> List[Dict[str, object]]:
    """Inicializa una lista completa de obstáculos para un perfil."""

    return [
        inicializar_obstaculo_fase2(
            obstaculo=obstaculo,
            perfil=perfil,
            semilla=semilla,
            indice_obstaculo=indice,
        )
        for indice, obstaculo in enumerate(obstaculos)
    ]


# ==========================================================
# LEY DE MOVIMIENTO REAL DE LA FASE 2
# ==========================================================

def calcular_velocidad_real_fase2(
    obstaculo: Dict[str, object],
    tiempo_actual: float,
) -> Tuple[float, float, Dict[str, object]]:
    """
    Calcula la velocidad real que seguirá el obstáculo en la Fase 2.

    El predictor no conoce esta ley. El predictor sólo observa la
    velocidad instantánea actual y la propaga mediante el modelo original.
    """

    actualizado = obstaculo.copy()
    perfil = validar_perfil(str(actualizado.get("perfil_fase2", "modelo_conocido")))

    tiempo_inicio = float(actualizado.get("tiempo_inicio", 0.0))
    tiempo_activo = max(0.0, float(tiempo_actual) - tiempo_inicio)

    velocidad_base = float(actualizado.get("velocidad_base_fase2", 0.0))
    angulo_base = float(actualizado.get("angulo_base_fase2", 0.0))
    fase = float(actualizado.get("fase_fase2", 0.0))
    periodo = max(float(actualizado.get("periodo_fase2", 4.0)), 1e-6)

    # Antes del instante de aparición se conserva el estado original.
    if float(tiempo_actual) < tiempo_inicio:
        return (
            float(actualizado.get("vx", 0.0)),
            float(actualizado.get("vy", 0.0)),
            actualizado,
        )

    velocidad = velocidad_base
    angulo = angulo_base

    if perfil == "modelo_conocido":
        # No se fuerza una ley adicional; se conserva exactamente la
        # velocidad actual para reproducir el modelo original.
        return (
            float(actualizado.get("vx", 0.0)),
            float(actualizado.get("vy", 0.0)),
            actualizado,
        )

    if perfil == "aceleracion_suave":
        factor = 1.0 + 0.45 * math.sin(
            2.0 * math.pi * tiempo_activo / periodo + fase
        )
        velocidad = velocidad_base * factor

    elif perfil == "stop_and_go":
        # Ciclo de cuatro segundos: 1.2 s detenido y 2.8 s en movimiento.
        posicion_ciclo = (tiempo_activo + 0.35) % 4.0
        velocidad = 0.0 if posicion_ciclo < 1.2 else velocidad_base

    elif perfil == "cambio_direccion":
        tiempo_cambio = float(actualizado.get("tiempo_cambio_fase2", 1.0))
        signo = float(actualizado.get("signo_cambio_fase2", 1.0))

        if tiempo_activo >= tiempo_cambio:
            actualizado["direccion_cambiada_fase2"] = True
            angulo = angulo_base + signo * math.pi / 2.0

    elif perfil == "curva_sinusoidal":
        amplitud = float(
            actualizado.get("amplitud_angular_fase2", math.radians(30.0))
        )
        angulo = angulo_base + amplitud * math.sin(
            2.0 * math.pi * tiempo_activo / periodo + fase
        )

    velocidad = max(0.0, float(velocidad))
    angulo = normalizar_angulo_local(float(angulo))

    vx = velocidad * math.cos(angulo)
    vy = velocidad * math.sin(angulo)

    return float(vx), float(vy), actualizado


def actualizar_obstaculos_dinamicos_reales_fase2(
    obstaculos_dinamicos: Sequence[Dict[str, object]],
    obstaculos_estaticos: Sequence[Dict[str, object]],
    tiempo_actual: float,
    dt: float = DT_VERIFICACION_FASE2,
) -> List[Dict[str, object]]:
    """
    Actualiza el movimiento REAL de la Fase 2.

    Primero aplica el perfil no visto y después reutiliza la resolución
    original de paredes, obstáculos estáticos y colisiones dinámicas.
    """

    preparados: List[Dict[str, object]] = []

    for obstaculo in obstaculos_dinamicos:
        vx, vy, preparado = calcular_velocidad_real_fase2(
            obstaculo=obstaculo,
            tiempo_actual=tiempo_actual,
        )
        preparado["vx"] = float(vx)
        preparado["vy"] = float(vy)
        preparados.append(preparado)

    # Ésta es la función original. El predictor seguirá usando esta misma
    # función, pero sin conocer los cambios futuros impuestos arriba.
    actualizados = base.actualizar_obstaculos_dinamicos(
        obstaculos_dinamicos=preparados,
        obstaculos_estaticos=obstaculos_estaticos,
        tiempo_actual=float(tiempo_actual),
        dt=float(dt),
    )

    # Cuando la resolución geométrica invierte o desvía una velocidad,
    # se actualiza la dirección de referencia para evitar insistir contra
    # la misma pared en los pasos posteriores.
    for preparado, actualizado in zip(preparados, actualizados):
        velocidad_deseada = math.hypot(
            float(preparado.get("vx", 0.0)),
            float(preparado.get("vy", 0.0)),
        )
        velocidad_resultante = math.hypot(
            float(actualizado.get("vx", 0.0)),
            float(actualizado.get("vy", 0.0)),
        )

        if velocidad_deseada > 1e-9 and velocidad_resultante > 1e-9:
            ux_deseado = float(preparado["vx"]) / velocidad_deseada
            uy_deseado = float(preparado["vy"]) / velocidad_deseada
            ux_real = float(actualizado["vx"]) / velocidad_resultante
            uy_real = float(actualizado["vy"]) / velocidad_resultante
            producto = ux_deseado * ux_real + uy_deseado * uy_real

            if producto < 0.50:
                actualizado["angulo_base_fase2"] = float(
                    math.atan2(uy_real, ux_real)
                )

    return actualizados


# ==========================================================
# SIMULACIÓN NOMINAL DEL PREDICTOR Y SIMULACIÓN REAL
# ==========================================================

def simular_modelo_nominal(
    obstaculos_iniciales: Sequence[Dict[str, object]],
    obstaculos_estaticos: Sequence[Dict[str, object]],
    horizonte: float,
    dt: float,
) -> Tuple[np.ndarray, List[List[Dict[str, object]]]]:
    """Propaga los obstáculos con el modelo original del predictor."""

    estados = copiar_obstaculos(obstaculos_iniciales)
    historial = [copiar_obstaculos(estados)]
    tiempos = [0.0]

    numero_pasos = int(round(float(horizonte) / float(dt)))

    for paso in range(numero_pasos):
        tiempo = (paso + 1) * float(dt)
        estados = base.actualizar_obstaculos_dinamicos(
            obstaculos_dinamicos=estados,
            obstaculos_estaticos=obstaculos_estaticos,
            tiempo_actual=tiempo,
            dt=float(dt),
        )
        tiempos.append(tiempo)
        historial.append(copiar_obstaculos(estados))

    return np.asarray(tiempos, dtype=float), historial


def simular_movimiento_real_fase2(
    obstaculos_iniciales: Sequence[Dict[str, object]],
    obstaculos_estaticos: Sequence[Dict[str, object]],
    horizonte: float,
    dt: float,
) -> Tuple[np.ndarray, List[List[Dict[str, object]]]]:
    """Propaga los obstáculos con la dinámica real de la Fase 2."""

    estados = copiar_obstaculos(obstaculos_iniciales)
    historial = [copiar_obstaculos(estados)]
    tiempos = [0.0]

    numero_pasos = int(round(float(horizonte) / float(dt)))

    for paso in range(numero_pasos):
        tiempo = (paso + 1) * float(dt)
        estados = actualizar_obstaculos_dinamicos_reales_fase2(
            obstaculos_dinamicos=estados,
            obstaculos_estaticos=obstaculos_estaticos,
            tiempo_actual=tiempo,
            dt=float(dt),
        )
        tiempos.append(tiempo)
        historial.append(copiar_obstaculos(estados))

    return np.asarray(tiempos, dtype=float), historial


def extraer_trayectoria_primer_obstaculo(
    historial: Sequence[Sequence[Dict[str, object]]],
) -> np.ndarray:
    """Convierte el historial del primer obstáculo en una matriz Nx2."""

    if len(historial) == 0 or len(historial[0]) == 0:
        raise ValueError("El historial de obstáculos está vacío.")

    return np.asarray(
        [
            [
                float(estado[0]["x"]),
                float(estado[0]["y"]),
            ]
            for estado in historial
        ],
        dtype=float,
    )


def indice_tiempo(tiempos: np.ndarray, tiempo_objetivo: float) -> int:
    """Obtiene el índice más cercano a un tiempo objetivo."""

    return int(np.argmin(np.abs(tiempos - float(tiempo_objetivo))))


# ==========================================================
# VERIFICACIÓN POR PERFIL
# ==========================================================

def verificar_perfil_fase2(
    perfil: str,
    semilla: int,
) -> Dict[str, object]:
    """Ejecuta y resume una verificación determinista de un perfil."""

    perfil = validar_perfil(perfil)

    obstaculo_base: Dict[str, object] = {
        "x": 5.0,
        "y": 5.0,
        "vx": 0.80,
        "vy": 0.0,
        "radio": 0.30,
        "tipo": "verificacion_fase2",
        "indice_cruce": None,
        "punto_cruce": None,
        "tiempo_inicio": 0.0,
        "tiempo_estimado_cruce": None,
        "tiempo_estimado_robot": None,
    }

    obstaculos_iniciales = inicializar_obstaculos_fase2(
        obstaculos=[obstaculo_base],
        perfil=perfil,
        semilla=semilla,
    )

    obstaculos_estaticos: List[Dict[str, object]] = []

    tiempos_nominales, historial_nominal = simular_modelo_nominal(
        obstaculos_iniciales=obstaculos_iniciales,
        obstaculos_estaticos=obstaculos_estaticos,
        horizonte=HORIZONTE_VERIFICACION_FASE2,
        dt=DT_VERIFICACION_FASE2,
    )

    tiempos_reales, historial_real = simular_movimiento_real_fase2(
        obstaculos_iniciales=obstaculos_iniciales,
        obstaculos_estaticos=obstaculos_estaticos,
        horizonte=HORIZONTE_VERIFICACION_FASE2,
        dt=DT_VERIFICACION_FASE2,
    )

    # Segunda ejecución para comprobar determinismo.
    _, historial_real_repetido = simular_movimiento_real_fase2(
        obstaculos_iniciales=obstaculos_iniciales,
        obstaculos_estaticos=obstaculos_estaticos,
        horizonte=HORIZONTE_VERIFICACION_FASE2,
        dt=DT_VERIFICACION_FASE2,
    )

    trayectoria_nominal = extraer_trayectoria_primer_obstaculo(historial_nominal)
    trayectoria_real = extraer_trayectoria_primer_obstaculo(historial_real)
    trayectoria_real_repetida = extraer_trayectoria_primer_obstaculo(
        historial_real_repetido
    )

    if not np.array_equal(tiempos_nominales, tiempos_reales):
        raise RuntimeError("Las bases temporales nominal y real no coinciden.")

    errores = np.linalg.norm(trayectoria_real - trayectoria_nominal, axis=1)

    errores_horizontes: Dict[str, float] = {}
    for horizonte in HORIZONTES_COMPARACION_FASE2:
        indice = indice_tiempo(tiempos_reales, horizonte)
        errores_horizontes[f"error_{horizonte:.1f}s_m"] = float(errores[indice])

    todos_finitos = bool(
        np.all(np.isfinite(trayectoria_nominal))
        and np.all(np.isfinite(trayectoria_real))
        and np.all(np.isfinite(errores))
    )

    dentro_mapa = bool(
        np.all(trayectoria_real[:, 0] >= 0.0)
        and np.all(trayectoria_real[:, 0] <= float(base.ANCHO_MAPA))
        and np.all(trayectoria_real[:, 1] >= 0.0)
        and np.all(trayectoria_real[:, 1] <= float(base.ALTO_MAPA))
    )

    determinista = bool(np.array_equal(trayectoria_real, trayectoria_real_repetida))
    error_maximo = float(np.max(errores))
    error_final = float(errores[-1])

    if perfil == "modelo_conocido":
        discrepancia_correcta = error_maximo <= TOLERANCIA_MODELO_CONOCIDO_M
    else:
        discrepancia_correcta = error_maximo >= ERROR_MINIMO_DISCREPANCIA_M

    return {
        "perfil": perfil,
        "semilla": int(semilla),
        "tiempos": tiempos_reales,
        "trayectoria_nominal": trayectoria_nominal,
        "trayectoria_real": trayectoria_real,
        "error_maximo_m": error_maximo,
        "error_final_m": error_final,
        **errores_horizontes,
        "estados_finitos": todos_finitos,
        "dentro_mapa": dentro_mapa,
        "determinista": determinista,
        "discrepancia_correcta": bool(discrepancia_correcta),
    }


# ==========================================================
# GUARDADO DE RESULTADOS
# ==========================================================

def guardar_csv_resultados(resultados: Sequence[Dict[str, object]]) -> str:
    """Guarda únicamente las métricas escalares de cada perfil."""

    DIRECTORIO_FASE2.mkdir(parents=True, exist_ok=True)

    campos = [
        "perfil",
        "semilla",
        "error_0.5s_m",
        "error_1.0s_m",
        "error_1.5s_m",
        "error_2.0s_m",
        "error_3.0s_m",
        "error_maximo_m",
        "error_final_m",
        "estados_finitos",
        "dentro_mapa",
        "determinista",
        "discrepancia_correcta",
    ]

    with RUTA_CSV_FASE2.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()

        for resultado in resultados:
            escritor.writerow({campo: resultado[campo] for campo in campos})

    return str(RUTA_CSV_FASE2)


def crear_grafica_resultados(resultados: Sequence[Dict[str, object]]) -> str:
    """Compara las trayectorias reales y nominales de los cinco perfiles."""

    DIRECTORIO_FASE2.mkdir(parents=True, exist_ok=True)

    figura, eje = plt.subplots(figsize=(11, 8))

    for resultado in resultados:
        perfil = str(resultado["perfil"])
        trayectoria_nominal = np.asarray(resultado["trayectoria_nominal"], dtype=float)
        trayectoria_real = np.asarray(resultado["trayectoria_real"], dtype=float)

        if perfil == "modelo_conocido":
            eje.plot(
                trayectoria_nominal[:, 0],
                trayectoria_nominal[:, 1],
                linestyle="--",
                linewidth=2.0,
                label="Predicción nominal",
            )

        eje.plot(
            trayectoria_real[:, 0],
            trayectoria_real[:, 1],
            linewidth=2.0,
            label=f"Real: {perfil}",
        )

        eje.scatter(
            trayectoria_real[-1, 0],
            trayectoria_real[-1, 1],
            s=30,
        )

    eje.set_title(
        "Fase 2: discrepancia entre la trayectoria predicha y la real"
    )
    eje.set_xlabel("X [m]")
    eje.set_ylabel("Y [m]")
    eje.axis("equal")
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(RUTA_GRAFICA_FASE2, dpi=300, bbox_inches="tight")
    plt.close(figura)

    return str(RUTA_GRAFICA_FASE2)


def construir_resumen_texto(resultados: Sequence[Dict[str, object]]) -> str:
    """Construye el informe textual de la verificación."""

    lineas = [
        "FASE 2 - VERIFICACIÓN DE DISCREPANCIA DEL MODELO",
        "=" * 80,
        "",
        "Entrenamiento: False",
        "Checkpoints modificados: False",
        "Semillas finales 70000-70099 reutilizadas: False",
        "Predictor nominal: función original actualizar_obstaculos_dinamicos",
        "Movimiento real: perfiles separados de Fase 2",
        "",
    ]

    for resultado in resultados:
        lineas.extend(
            [
                str(resultado["perfil"]),
                "-" * 80,
                f"Error a 0.5 s: {float(resultado['error_0.5s_m']):.6f} m",
                f"Error a 1.0 s: {float(resultado['error_1.0s_m']):.6f} m",
                f"Error a 1.5 s: {float(resultado['error_1.5s_m']):.6f} m",
                f"Error máximo: {float(resultado['error_maximo_m']):.6f} m",
                f"Estados finitos: {bool(resultado['estados_finitos'])}",
                f"Dentro del mapa: {bool(resultado['dentro_mapa'])}",
                f"Determinista: {bool(resultado['determinista'])}",
                f"Discrepancia correcta: {bool(resultado['discrepancia_correcta'])}",
                "",
            ]
        )

    return "\n".join(lineas) + "\n"


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("FASE 2: VERIFICACIÓN DE DISCREPANCIA DEL MODELO")
    print("=" * 80)

    print("\nEsta ejecución NO entrena y NO modifica los actores.")
    print("Las semillas finales 70000-70099 no se utilizan.")

    resultados: List[Dict[str, object]] = []

    for indice, perfil in enumerate(PERFILES_DINAMICA_FASE2):
        resultado = verificar_perfil_fase2(
            perfil=perfil,
            semilla=SEMILLA_VERIFICACION_FASE2 + indice,
        )
        resultados.append(resultado)

        print("\n" + perfil)
        print(
            "Error 0.5/1.0/1.5 s:",
            f"{resultado['error_0.5s_m']:.4f} /",
            f"{resultado['error_1.0s_m']:.4f} /",
            f"{resultado['error_1.5s_m']:.4f} m",
        )
        print("Error máximo:", f"{resultado['error_maximo_m']:.4f} m")
        print("Determinista:", resultado["determinista"])
        print("Estados válidos:", resultado["estados_finitos"] and resultado["dentro_mapa"])
        print("Discrepancia correcta:", resultado["discrepancia_correcta"])

    DIRECTORIO_FASE2.mkdir(parents=True, exist_ok=True)

    ruta_csv = guardar_csv_resultados(resultados)
    ruta_grafica = crear_grafica_resultados(resultados)

    texto_resumen = construir_resumen_texto(resultados)
    RUTA_RESUMEN_FASE2.write_text(texto_resumen, encoding="utf-8")

    verificaciones = {
        "modelo_conocido_coincide": bool(
            resultados[0]["error_maximo_m"] <= TOLERANCIA_MODELO_CONOCIDO_M
        ),
        "perfiles_no_vistos_difieren": bool(
            all(
                resultado["error_maximo_m"] >= ERROR_MINIMO_DISCREPANCIA_M
                for resultado in resultados[1:]
            )
        ),
        "todos_deterministas": bool(
            all(resultado["determinista"] for resultado in resultados)
        ),
        "todos_finitos": bool(
            all(resultado["estados_finitos"] for resultado in resultados)
        ),
        "todos_dentro_mapa": bool(
            all(resultado["dentro_mapa"] for resultado in resultados)
        ),
        "archivos_guardados": bool(
            Path(ruta_csv).is_file()
            and Path(ruta_grafica).is_file()
            and RUTA_RESUMEN_FASE2.is_file()
        ),
    }

    todo_correcto = all(verificaciones.values())

    print("\n" + "=" * 80)
    print("VERIFICACIONES TÉCNICAS")
    print("=" * 80)

    for nombre, valor in verificaciones.items():
        print(f"{nombre}: {valor}")

    print("\nARCHIVOS")
    print("CSV:", ruta_csv)
    print("Gráfica:", ruta_grafica)
    print("Resumen:", RUTA_RESUMEN_FASE2)

    if todo_correcto:
        print("\nRESULTADO DEL MAIN: TODO CORRECTO")
        print("FASE 2 LISTA PARA LA COMPARACIÓN DE ROBUSTEZ")
    else:
        print("\nRESULTADO DEL MAIN: HAY VERIFICACIONES POR CORREGIR")
        raise RuntimeError(
            "La discrepancia del modelo todavía no quedó validada."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
