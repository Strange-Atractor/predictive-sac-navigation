"""
SAC-UPO-TTC: Soft Actor-Critic con ocupación futura probabilística,
riesgo TTC y entrenamiento con incertidumbre perceptual.

Este archivo extiende el programa:
    SAC_predictivo_entrenamiento_completo.py
    SAC_predictivo_entrenamiento_completo(3).py

y reutiliza sin modificar:
    - la generación del mapa y la ruta global A*;
    - la dinámica real nominal del robot y de los obstáculos;
    - la arquitectura CNN + MLP del actor;
    - los dos críticos, temperatura automática y replay buffer;
    - las métricas y la selección periódica del mejor actor.

La diferencia metodológica es exclusivamente la rama predictiva:
    1. Observación retrasada y ruidosa de obstáculos dinámicos.
    2. Predicción Monte Carlo con velocidad, dirección, aceleración,
       giro y detenciones inciertas.
    3. Canal dinámico continuo P(ocupación futura) en [0, 1].
    4. Riesgo mediante probabilidad de colisión, casi colisión,
       clearance-CVaR y TTC-CVaR.

Uso recomendado:
    python SAC_probabilistico_ocupacion_TTC_entrenamiento.py --solo-verificacion
    python SAC_probabilistico_ocupacion_TTC_entrenamiento.py --episodios 400

El archivo base debe estar en la misma carpeta. El entrenamiento completo
requiere el checkpoint reactivo mejorado que ya utiliza el programa base.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib
import numpy as np
import torch


# ==========================================================
# IMPORTACIÓN SEGURA DEL PROGRAMA BASE
# ==========================================================

_DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
_CANDIDATOS_BASE = (
    "SAC_predictivo_entrenamiento_completo.py",
    "SAC_predictivo_entrenamiento_completo(3).py",
    "SAC_predictivo_entrenamiento_completo(1).py",
)


def _localizar_programa_base() -> Path:
    for nombre in _CANDIDATOS_BASE:
        ruta = _DIRECTORIO_ACTUAL / nombre
        if ruta.is_file():
            return ruta

    nombres = "\n".join(f"  - {nombre}" for nombre in _CANDIDATOS_BASE)
    raise FileNotFoundError(
        "No se encontró el programa base. Coloque este archivo en la misma "
        "carpeta que uno de los siguientes archivos:\n" + nombres
    )


_RUTA_BASE = _localizar_programa_base()

# Permite ejecutar la verificación en servidores sin interfaz gráfica.
_USAR_AGG = os.environ.get("MPLBACKEND", "").strip().lower() == "agg"
_USAR_MATPLOTLIB_ORIGINAL = matplotlib.use

if _USAR_AGG:
    def _usar_backend_compatible(backend, *args, **kwargs):
        if str(backend).strip().lower() == "tkagg":
            return _USAR_MATPLOTLIB_ORIGINAL("Agg", force=True)
        return _USAR_MATPLOTLIB_ORIGINAL(backend, *args, **kwargs)

    matplotlib.use = _usar_backend_compatible

_ESPECIFICACION = importlib.util.spec_from_file_location(
    "sac_predictivo_base_para_probabilistico",
    _RUTA_BASE,
)

if _ESPECIFICACION is None or _ESPECIFICACION.loader is None:
    raise ImportError(f"No fue posible preparar la importación de {_RUTA_BASE}")

base = importlib.util.module_from_spec(_ESPECIFICACION)
sys.modules[_ESPECIFICACION.name] = base
_ESPECIFICACION.loader.exec_module(base)
matplotlib.use = _USAR_MATPLOTLIB_ORIGINAL


# ==========================================================
# CONFIGURACIÓN DEL MÉTODO SAC-UPO-TTC
# ==========================================================

NOMBRE_METODO = "SAC-UPO-TTC"
VARIANTE_CHECKPOINT = "probabilistico_ocupacion_futura_ttc"

HORIZONTES_PROBABILISTICOS: Tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
)

PESOS_HORIZONTES_PROBABILISTICOS: Tuple[float, ...] = (
    1.00,
    0.75,
    0.50,
    0.25,
)

# Se puede modificar por argumento de línea de comandos.
NUMERO_MUESTRAS_MONTE_CARLO = 12

# Incertidumbre de percepción muestreada por episodio.
LATENCIA_MAXIMA_PASOS = 3             # 0.0 a 0.3 s con DT=0.1
SIGMA_POSICION_MIN_M = 0.00
SIGMA_POSICION_MAX_M = 0.08
SIGMA_VELOCIDAD_MIN_M_S = 0.00
SIGMA_VELOCIDAD_MAX_M_S = 0.12

# Incertidumbre del modelo futuro Monte Carlo.
SIGMA_POSICION_MODELO_M = 0.025
SIGMA_VELOCIDAD_RELATIVA = 0.18
SIGMA_DIRECCION_RAD = math.radians(15.0)
SIGMA_ACELERACION_M_S2 = 0.30
SIGMA_GIRO_RAD_S = math.radians(25.0)
PROBABILIDAD_DETENCION = 0.12
TIEMPO_DETENCION_MIN_S = 0.25
TIEMPO_DETENCION_MAX_S = 1.50

# Riesgo probabilístico.
HORIZONTE_RIESGO_PROBABILISTICO_S = 2.0
PASO_RIESGO_PROBABILISTICO_S = float(base.DT)
ALFA_CVAR = 0.25
UMBRAL_CASI_COLISION_M = 0.20
CLEARANCE_CVAR_OBJETIVO_M = 0.60
TTC_CVAR_OBJETIVO_S = 1.50

# Pesos de la recompensa de seguridad.
PESO_CLEARANCE_CVAR = 1.50
PESO_TTC_CVAR = 2.50
PESO_PROBABILIDAD_COLISION = 4.00
PESO_PROBABILIDAD_CASI_COLISION = 1.50
PESO_AVANCE_RIESGOSO = 2.00
PESO_DETENCION_EN_RIESGO = 1.00

# Separación de semillas del resto de los experimentos previos.
SEMILLA_BASE_ENTRENAMIENTO_PROBABILISTICO = 120000
SEMILLA_BASE_VALIDACION_PROBABILISTICO = 130000
SEMILLA_DIAGNOSTICA_PROBABILISTICO = 131000

# Claves auxiliares del entorno.
CLAVE_GENERADOR_SENSOR = "generador_sensor_sac_probabilistico"
CLAVE_GENERADOR_MODELO = "generador_modelo_sac_probabilistico"
CLAVE_HISTORIAL = "historial_obstaculos_verdaderos_probabilistico"
CLAVE_CONFIGURACION = "configuracion_incertidumbre_probabilistica"
CLAVE_OBSERVADOS = "obstaculos_observados_probabilistico"
CLAVE_MODELO_MC = "modelo_monte_carlo_probabilistico"


# ==========================================================
# UTILIDADES NUMÉRICAS
# ==========================================================

def _copiar_obstaculos(
    obstaculos: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    return [dict(obstaculo) for obstaculo in obstaculos]


def _validar_generador(
    generador: np.random.Generator,
) -> None:
    if not isinstance(generador, np.random.Generator):
        raise TypeError("Se requiere un generador np.random.Generator.")


def _reflejar_intervalo(
    valores: np.ndarray,
    limite_inferior: float,
    limite_superior: float,
) -> np.ndarray:
    """Refleja posiciones en un intervalo, incluso con varios rebotes."""

    inferior = float(limite_inferior)
    superior = float(limite_superior)
    amplitud = superior - inferior

    if not np.isfinite(amplitud) or amplitud <= 0.0:
        return np.full_like(valores, 0.5 * (inferior + superior), dtype=float)

    fase = np.mod(np.asarray(valores, dtype=float) - inferior, 2.0 * amplitud)
    reflejados = np.where(
        fase <= amplitud,
        inferior + fase,
        superior - (fase - amplitud),
    )
    return reflejados.astype(float, copy=False)


def _cvar_inferior(
    valores: Sequence[float],
    alfa: float = ALFA_CVAR,
) -> float:
    """Media del peor alfa de valores cuando valores pequeños son peligrosos."""

    arreglo = np.asarray(valores, dtype=float)
    arreglo = arreglo[np.isfinite(arreglo)]

    if arreglo.size == 0:
        return float("inf")

    alfa = float(np.clip(alfa, 1e-6, 1.0))
    cantidad = max(1, int(math.ceil(alfa * arreglo.size)))
    ordenados = np.sort(arreglo)
    return float(np.mean(ordenados[:cantidad]))


def _trayectoria_robot_constante(
    estado_robot: Sequence[float],
    velocidad_lineal: float,
    velocidad_angular: float,
    tiempos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagación exacta del modelo cinemático unicycle con control constante."""

    x0, y0, theta0 = (float(valor) for valor in estado_robot)
    v = float(velocidad_lineal)
    omega = float(velocidad_angular)
    tiempos = np.asarray(tiempos, dtype=float)

    if abs(omega) < 1e-8:
        x = x0 + v * np.cos(theta0) * tiempos
        y = y0 + v * np.sin(theta0) * tiempos
    else:
        x = x0 + (v / omega) * (
            np.sin(theta0 + omega * tiempos) - math.sin(theta0)
        )
        y = y0 - (v / omega) * (
            np.cos(theta0 + omega * tiempos) - math.cos(theta0)
        )

    return x.astype(float), y.astype(float)


# ==========================================================
# INCERTIDUMBRE DE PERCEPCIÓN POR EPISODIO
# ==========================================================

def crear_configuracion_incertidumbre_episodio(
    semilla: int,
) -> Dict[str, float]:
    generador = np.random.default_rng(int(semilla) + 701001)

    return {
        "latencia_pasos": int(
            generador.integers(0, LATENCIA_MAXIMA_PASOS + 1)
        ),
        "sigma_posicion_m": float(
            generador.uniform(SIGMA_POSICION_MIN_M, SIGMA_POSICION_MAX_M)
        ),
        "sigma_velocidad_m_s": float(
            generador.uniform(
                SIGMA_VELOCIDAD_MIN_M_S,
                SIGMA_VELOCIDAD_MAX_M_S,
            )
        ),
    }


def obtener_obstaculos_observados_probabilisticos(
    entorno: MutableMapping[str, object],
) -> List[Dict[str, object]]:
    """Devuelve una observación retardada y ruidosa sin alterar el mundo real."""

    generador = entorno[CLAVE_GENERADOR_SENSOR]
    _validar_generador(generador)

    historial = entorno[CLAVE_HISTORIAL]
    configuracion = entorno[CLAVE_CONFIGURACION]

    if not isinstance(historial, list) or len(historial) == 0:
        raise RuntimeError("El historial verdadero de obstáculos está vacío.")

    latencia = int(configuracion["latencia_pasos"])
    indice = max(0, len(historial) - 1 - latencia)
    observados = _copiar_obstaculos(historial[indice])

    sigma_posicion = float(configuracion["sigma_posicion_m"])
    sigma_velocidad = float(configuracion["sigma_velocidad_m_s"])

    for obstaculo in observados:
        radio = float(obstaculo["radio"])

        x = float(obstaculo["x"]) + float(generador.normal(0.0, sigma_posicion))
        y = float(obstaculo["y"]) + float(generador.normal(0.0, sigma_posicion))
        vx = float(obstaculo["vx"]) + float(generador.normal(0.0, sigma_velocidad))
        vy = float(obstaculo["vy"]) + float(generador.normal(0.0, sigma_velocidad))

        obstaculo["x"] = float(np.clip(x, radio, base.ANCHO_MAPA - radio))
        obstaculo["y"] = float(np.clip(y, radio, base.ALTO_MAPA - radio))
        obstaculo["vx"] = vx
        obstaculo["vy"] = vy
        obstaculo["observacion_latencia_pasos"] = latencia
        obstaculo["observacion_sigma_posicion_m"] = sigma_posicion
        obstaculo["observacion_sigma_velocidad_m_s"] = sigma_velocidad

    entorno[CLAVE_OBSERVADOS] = _copiar_obstaculos(observados)
    return observados


# ==========================================================
# MODELO MONTE CARLO DE MOVIMIENTO FUTURO
# ==========================================================

def crear_modelo_monte_carlo_obstaculos(
    obstaculos_observados: Sequence[Mapping[str, object]],
    generador: np.random.Generator,
    numero_muestras: int = NUMERO_MUESTRAS_MONTE_CARLO,
) -> List[Dict[str, np.ndarray | float]]:
    """Crea futuros posibles sin consultar la dinámica real del simulador."""

    _validar_generador(generador)
    numero_muestras = int(numero_muestras)

    if numero_muestras < 2:
        raise ValueError("Se requieren al menos dos muestras Monte Carlo.")

    modelos: List[Dict[str, np.ndarray | float]] = []

    for obstaculo in obstaculos_observados:
        x_observado = float(obstaculo["x"])
        y_observado = float(obstaculo["y"])
        vx_observado = float(obstaculo["vx"])
        vy_observado = float(obstaculo["vy"])
        radio = float(obstaculo["radio"])

        velocidad_base = math.hypot(vx_observado, vy_observado)
        direccion_base = math.atan2(vy_observado, vx_observado)

        x0 = x_observado + generador.normal(
            0.0,
            SIGMA_POSICION_MODELO_M,
            size=numero_muestras,
        )
        y0 = y_observado + generador.normal(
            0.0,
            SIGMA_POSICION_MODELO_M,
            size=numero_muestras,
        )

        escala_velocidad = 1.0 + generador.normal(
            0.0,
            SIGMA_VELOCIDAD_RELATIVA,
            size=numero_muestras,
        )
        velocidad_inicial = np.maximum(0.0, velocidad_base * escala_velocidad)

        direccion_inicial = direccion_base + generador.normal(
            0.0,
            SIGMA_DIRECCION_RAD,
            size=numero_muestras,
        )
        aceleracion = generador.normal(
            0.0,
            SIGMA_ACELERACION_M_S2,
            size=numero_muestras,
        )
        velocidad_giro = generador.normal(
            0.0,
            SIGMA_GIRO_RAD_S,
            size=numero_muestras,
        )

        detener = generador.random(numero_muestras) < PROBABILIDAD_DETENCION
        tiempo_detencion = np.full(numero_muestras, np.inf, dtype=float)
        tiempo_detencion[detener] = generador.uniform(
            TIEMPO_DETENCION_MIN_S,
            TIEMPO_DETENCION_MAX_S,
            size=int(np.sum(detener)),
        )

        x0 = np.clip(x0, radio, base.ANCHO_MAPA - radio)
        y0 = np.clip(y0, radio, base.ALTO_MAPA - radio)

        modelos.append(
            {
                "x0": x0.astype(float),
                "y0": y0.astype(float),
                "velocidad_inicial": velocidad_inicial.astype(float),
                "direccion_inicial": direccion_inicial.astype(float),
                "aceleracion": aceleracion.astype(float),
                "velocidad_giro": velocidad_giro.astype(float),
                "tiempo_detencion": tiempo_detencion.astype(float),
                "radio": radio,
            }
        )

    return modelos


def posiciones_modelo_monte_carlo(
    modelo: Mapping[str, object],
    horizonte: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evalúa todas las muestras de un obstáculo en un horizonte."""

    horizonte = max(0.0, float(horizonte))
    x0 = np.asarray(modelo["x0"], dtype=float)
    y0 = np.asarray(modelo["y0"], dtype=float)
    velocidad0 = np.asarray(modelo["velocidad_inicial"], dtype=float)
    direccion0 = np.asarray(modelo["direccion_inicial"], dtype=float)
    aceleracion = np.asarray(modelo["aceleracion"], dtype=float)
    giro = np.asarray(modelo["velocidad_giro"], dtype=float)
    tiempo_detencion = np.asarray(modelo["tiempo_detencion"], dtype=float)
    radio = float(modelo["radio"])

    tiempo_movimiento = np.minimum(horizonte, tiempo_detencion)
    velocidad_final = np.maximum(0.0, velocidad0 + aceleracion * tiempo_movimiento)
    distancia = 0.5 * (velocidad0 + velocidad_final) * tiempo_movimiento
    direccion_media = direccion0 + 0.5 * giro * tiempo_movimiento

    x = x0 + distancia * np.cos(direccion_media)
    y = y0 + distancia * np.sin(direccion_media)

    x = _reflejar_intervalo(x, radio, base.ANCHO_MAPA - radio)
    y = _reflejar_intervalo(y, radio, base.ALTO_MAPA - radio)
    return x, y


def predecir_obstaculos_dinamicos_sac_probabilistico(
    obstaculos_dinamicos,
    obstaculos_estaticos,
    tiempo_actual,
    horizontes=HORIZONTES_PROBABILISTICOS,
    paso_prediccion=PASO_RIESGO_PROBABILISTICO_S,
    generador=None,
    modelo_monte_carlo=None,
):
    """Interfaz compatible con el predictor previo, ahora probabilística."""

    del obstaculos_estaticos, tiempo_actual, paso_prediccion

    horizontes = tuple(float(valor) for valor in horizontes)
    if len(horizontes) == 0 or any(valor < 0.0 for valor in horizontes):
        raise ValueError("Los horizontes probabilísticos no son válidos.")

    if modelo_monte_carlo is None:
        if generador is None:
            generador = np.random.default_rng(123456)
        modelo_monte_carlo = crear_modelo_monte_carlo_obstaculos(
            obstaculos_dinamicos,
            generador,
            NUMERO_MUESTRAS_MONTE_CARLO,
        )

    predicciones = []
    for horizonte in horizontes:
        muestras_obstaculos = []
        for modelo in modelo_monte_carlo:
            x, y = posiciones_modelo_monte_carlo(modelo, horizonte)
            muestras_obstaculos.append(
                {
                    "x": x,
                    "y": y,
                    "radio": float(modelo["radio"]),
                }
            )

        predicciones.append(
            {
                "horizonte": horizonte,
                "muestras_obstaculos": muestras_obstaculos,
                "numero_muestras": NUMERO_MUESTRAS_MONTE_CARLO,
            }
        )

    return predicciones


# ==========================================================
# MAPA EGOCÉNTRICO DE PROBABILIDAD DE OCUPACIÓN FUTURA
# ==========================================================

def construir_parche_egocentrico_sac_probabilistico(
    estado_robot,
    obstaculos_estaticos,
    obstaculos_dinamicos,
    tiempo_actual=0.0,
    horizontes_prediccion=HORIZONTES_PROBABILISTICOS,
    intensidades_prediccion=PESOS_HORIZONTES_PROBABILISTICOS,
    generador=None,
    modelo_monte_carlo=None,
):
    horizontes = tuple(float(valor) for valor in horizontes_prediccion)
    pesos = tuple(float(valor) for valor in intensidades_prediccion)

    if len(horizontes) != len(pesos):
        raise ValueError("Cada horizonte debe tener un peso asociado.")
    if any(peso < 0.0 or peso > 1.0 for peso in pesos):
        raise ValueError("Los pesos deben pertenecer a [0, 1].")

    # Se conserva exactamente el canal estático del agente previo.
    parche_base = base.construir_parche_egocentrico_sac(
        estado_robot=estado_robot,
        obstaculos_estaticos=obstaculos_estaticos,
        obstaculos_dinamicos=[],
    )
    canal_estatico = parche_base[0].copy()
    canal_dinamico = np.zeros(
        (base.RESOLUCION_PARCHE_SAC, base.RESOLUCION_PARCHE_SAC),
        dtype=np.float32,
    )

    x_robot, y_robot, theta_robot = (float(valor) for valor in estado_robot)
    mitad = base.TAMANO_PARCHE_SAC / 2.0
    tamano_celda = base.TAMANO_PARCHE_SAC / base.RESOLUCION_PARCHE_SAC

    coordenadas = mitad - (
        np.arange(base.RESOLUCION_PARCHE_SAC, dtype=float) + 0.5
    ) * tamano_celda
    x_local, y_local = np.meshgrid(coordenadas, coordenadas, indexing="ij")

    coseno = math.cos(theta_robot)
    seno = math.sin(theta_robot)
    x_mundo = x_robot + coseno * x_local - seno * y_local
    y_mundo = y_robot + seno * x_local + coseno * y_local

    predicciones = predecir_obstaculos_dinamicos_sac_probabilistico(
        obstaculos_dinamicos=obstaculos_dinamicos,
        obstaculos_estaticos=obstaculos_estaticos,
        tiempo_actual=tiempo_actual,
        horizontes=horizontes,
        generador=generador,
        modelo_monte_carlo=modelo_monte_carlo,
    )

    for prediccion, peso_horizonte in zip(predicciones, pesos):
        for muestras in prediccion["muestras_obstaculos"]:
            x_muestras = np.asarray(muestras["x"], dtype=float)
            y_muestras = np.asarray(muestras["y"], dtype=float)
            radio_ocupado = float(muestras["radio"]) + float(base.RADIO_ROBOT)

            distancia_cuadrada = (
                x_mundo[None, :, :] - x_muestras[:, None, None]
            ) ** 2 + (
                y_mundo[None, :, :] - y_muestras[:, None, None]
            ) ** 2

            probabilidad = np.mean(
                distancia_cuadrada <= radio_ocupado ** 2,
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

            canal_dinamico = np.maximum(
                canal_dinamico,
                np.float32(peso_horizonte) * probabilidad,
            )

    canal_dinamico = np.clip(canal_dinamico, 0.0, 1.0).astype(np.float32)
    parche = np.stack([canal_estatico, canal_dinamico], axis=0).astype(np.float32)

    forma = (
        base.CANALES_PARCHE_SAC,
        base.RESOLUCION_PARCHE_SAC,
        base.RESOLUCION_PARCHE_SAC,
    )
    if parche.shape != forma:
        raise RuntimeError(f"Forma probabilística incorrecta: {parche.shape}")
    if not np.all(np.isfinite(parche)):
        raise RuntimeError("El mapa probabilístico contiene valores no finitos.")
    if np.min(parche) < 0.0 or np.max(parche) > 1.0:
        raise RuntimeError("El mapa probabilístico salió del intervalo [0, 1].")

    return parche


def construir_observacion_sac_probabilistico(
    estado_robot,
    submeta,
    meta,
    obstaculos_estaticos,
    obstaculos_dinamicos,
    velocidad_lineal_actual,
    velocidad_angular_actual,
    indice_progreso,
    numero_puntos_camino,
    tiempo_actual=0.0,
    generador=None,
    modelo_monte_carlo=None,
):
    parche = construir_parche_egocentrico_sac_probabilistico(
        estado_robot=estado_robot,
        obstaculos_estaticos=obstaculos_estaticos,
        obstaculos_dinamicos=obstaculos_dinamicos,
        tiempo_actual=tiempo_actual,
        generador=generador,
        modelo_monte_carlo=modelo_monte_carlo,
    )

    escalares = base.construir_escalares_sac(
        estado_robot=estado_robot,
        submeta=submeta,
        meta=meta,
        velocidad_lineal_actual=velocidad_lineal_actual,
        velocidad_angular_actual=velocidad_angular_actual,
        indice_progreso=indice_progreso,
        numero_puntos_camino=numero_puntos_camino,
    )

    return {
        "parche": parche.copy(),
        "escalares": escalares.copy(),
    }


# ==========================================================
# RIESGO TTC Y CLEARANCE SOBRE LA DISTRIBUCIÓN DE FUTUROS
# ==========================================================

def calcular_riesgo_dinamico_sac_probabilistico(
    estado_robot,
    velocidad_lineal,
    velocidad_angular,
    obstaculos_dinamicos,
    obstaculos_estaticos,
    tiempo_actual,
    horizonte=HORIZONTE_RIESGO_PROBABILISTICO_S,
    paso_prediccion=PASO_RIESGO_PROBABILISTICO_S,
    generador=None,
    modelo_monte_carlo=None,
):
    del obstaculos_estaticos, tiempo_actual

    horizonte = float(horizonte)
    paso = float(paso_prediccion)
    if horizonte <= 0.0 or paso <= 0.0:
        raise ValueError("El horizonte y el paso de riesgo deben ser positivos.")

    if modelo_monte_carlo is None:
        if generador is None:
            generador = np.random.default_rng(654321)
        modelo_monte_carlo = crear_modelo_monte_carlo_obstaculos(
            obstaculos_dinamicos,
            generador,
            NUMERO_MUESTRAS_MONTE_CARLO,
        )

    tiempos = np.arange(paso, horizonte + 0.5 * paso, paso, dtype=float)
    x_robot, y_robot = _trayectoria_robot_constante(
        estado_robot,
        velocidad_lineal,
        velocidad_angular,
        tiempos,
    )

    numero_muestras = NUMERO_MUESTRAS_MONTE_CARLO
    clearances_por_tiempo = np.full(
        (numero_muestras, tiempos.size),
        np.inf,
        dtype=float,
    )

    for modelo in modelo_monte_carlo:
        radio_total = float(modelo["radio"]) + float(base.RADIO_ROBOT)
        x_obstaculo = np.empty((numero_muestras, tiempos.size), dtype=float)
        y_obstaculo = np.empty((numero_muestras, tiempos.size), dtype=float)

        for indice_tiempo, tiempo in enumerate(tiempos):
            x, y = posiciones_modelo_monte_carlo(modelo, float(tiempo))
            x_obstaculo[:, indice_tiempo] = x
            y_obstaculo[:, indice_tiempo] = y

        distancia = np.sqrt(
            (x_obstaculo - x_robot[None, :]) ** 2
            + (y_obstaculo - y_robot[None, :]) ** 2
        ) - radio_total

        clearances_por_tiempo = np.minimum(clearances_por_tiempo, distancia)

    if len(modelo_monte_carlo) == 0:
        clearances_por_tiempo.fill(np.inf)

    clearance_minimo_muestra = np.min(clearances_por_tiempo, axis=1)
    colision_muestra = np.any(clearances_por_tiempo <= 0.0, axis=1)
    casi_colision_muestra = np.any(
        clearances_por_tiempo <= UMBRAL_CASI_COLISION_M,
        axis=1,
    )

    ttc_muestra = np.full(numero_muestras, np.inf, dtype=float)
    for indice_muestra in range(numero_muestras):
        indices_colision = np.flatnonzero(clearances_por_tiempo[indice_muestra] <= 0.0)
        if indices_colision.size > 0:
            ttc_muestra[indice_muestra] = float(tiempos[indices_colision[0]])

    clearance_cvar = _cvar_inferior(clearance_minimo_muestra, ALFA_CVAR)
    clearance_percentil_10 = float(np.percentile(clearance_minimo_muestra, 10.0))
    ttc_cvar = _cvar_inferior(ttc_muestra, ALFA_CVAR)

    diferencias = -np.diff(clearances_por_tiempo, axis=1) / max(paso, 1e-8)
    velocidad_cierre_maxima = (
        float(np.max(np.maximum(diferencias, 0.0)))
        if diferencias.size > 0 and np.any(np.isfinite(diferencias))
        else 0.0
    )

    probabilidad_colision = float(np.mean(colision_muestra))
    probabilidad_casi_colision = float(np.mean(casi_colision_muestra))

    clearance_actual = float(
        base.calcular_clearance_dinamico(estado_robot, obstaculos_dinamicos)
    )

    return {
        # Claves compatibles con el SAC predictivo previo.
        "horizonte": horizonte,
        "paso_prediccion": paso,
        "clearance_dinamico_actual": clearance_actual,
        "clearance_dinamico_predicho_minimo": float(np.min(clearance_minimo_muestra)),
        "tiempo_clearance_minimo": float(
            tiempos[int(np.argmin(np.min(clearances_por_tiempo, axis=0)))]
            if tiempos.size > 0 and np.any(np.isfinite(clearances_por_tiempo))
            else 0.0
        ),
        "ttc_predicho": float(np.min(ttc_muestra)),
        "velocidad_cierre_maxima": velocidad_cierre_maxima,
        "colision_predicha": bool(probabilidad_colision > 0.0),
        "tiempos_prediccion": tiempos.tolist(),
        "clearances_predichos": np.min(clearances_por_tiempo, axis=0).tolist(),
        # Nuevas métricas probabilísticas.
        "numero_muestras_monte_carlo": numero_muestras,
        "probabilidad_colision_predicha": probabilidad_colision,
        "probabilidad_casi_colision_predicha": probabilidad_casi_colision,
        "clearance_cvar_inferior": clearance_cvar,
        "clearance_percentil_10": clearance_percentil_10,
        "ttc_cvar_inferior": ttc_cvar,
        "clearances_minimos_por_muestra": clearance_minimo_muestra.tolist(),
        "ttc_por_muestra": ttc_muestra.tolist(),
    }


def calcular_penalizacion_riesgo_sac_probabilistico(
    riesgo_predictivo,
    velocidad_lineal,
):
    clearance_cvar = float(riesgo_predictivo["clearance_cvar_inferior"])
    ttc_cvar = float(riesgo_predictivo["ttc_cvar_inferior"])
    prob_colision = float(riesgo_predictivo["probabilidad_colision_predicha"])
    prob_casi_colision = float(
        riesgo_predictivo["probabilidad_casi_colision_predicha"]
    )
    velocidad_lineal = float(velocidad_lineal)

    riesgo_clearance = (
        float(
            np.clip(
                (CLEARANCE_CVAR_OBJETIVO_M - clearance_cvar)
                / max(CLEARANCE_CVAR_OBJETIVO_M, 1e-8),
                0.0,
                1.0,
            )
        )
        if np.isfinite(clearance_cvar)
        else 0.0
    )

    riesgo_ttc = (
        float(
            np.clip(
                (TTC_CVAR_OBJETIVO_S - ttc_cvar)
                / max(TTC_CVAR_OBJETIVO_S, 1e-8),
                0.0,
                1.0,
            )
        )
        if np.isfinite(ttc_cvar)
        else 0.0
    )

    velocidad_normalizada = float(
        np.clip(velocidad_lineal / max(base.VELOCIDAD_MAXIMA, 1e-8), 0.0, 1.0)
    )

    penalizacion_clearance = PESO_CLEARANCE_CVAR * riesgo_clearance
    penalizacion_ttc = PESO_TTC_CVAR * riesgo_ttc
    penalizacion_colision = PESO_PROBABILIDAD_COLISION * prob_colision
    penalizacion_casi_colision = (
        PESO_PROBABILIDAD_CASI_COLISION * prob_casi_colision
    )
    riesgo_avance = max(riesgo_ttc, prob_casi_colision, prob_colision)
    penalizacion_avance = (
        PESO_AVANCE_RIESGOSO * riesgo_avance * velocidad_normalizada ** 2
    )

    detenido = velocidad_lineal <= base.VELOCIDAD_BAJA_RIESGO_PREDICTIVO_SAC
    penalizacion_detencion = (
        PESO_DETENCION_EN_RIESGO * prob_colision if detenido else 0.0
    )

    penalizacion_total = (
        penalizacion_clearance
        + penalizacion_ttc
        + penalizacion_colision
        + penalizacion_casi_colision
        + penalizacion_avance
        + penalizacion_detencion
    )

    return {
        # Claves compatibles.
        "riesgo_clearance_dinamico_predicho": riesgo_clearance,
        "riesgo_ttc_dinamico_predicho": riesgo_ttc,
        "velocidad_lineal_normalizada": velocidad_normalizada,
        "robot_detenido_o_muy_lento": detenido,
        "penalizacion_clearance_dinamico_predicho": penalizacion_clearance,
        "penalizacion_ttc_dinamico_predicho": penalizacion_ttc,
        "penalizacion_avance_riesgoso_predictivo": penalizacion_avance,
        "penalizacion_detencion_trayectoria_predictiva": penalizacion_detencion,
        "penalizacion_riesgo_dinamico_predictivo": float(penalizacion_total),
        # Componentes nuevos.
        "penalizacion_probabilidad_colision": penalizacion_colision,
        "penalizacion_probabilidad_casi_colision": penalizacion_casi_colision,
        "riesgo_probabilidad_colision": prob_colision,
        "riesgo_probabilidad_casi_colision": prob_casi_colision,
        "penalizacion_riesgo_probabilistico_total": float(penalizacion_total),
    }


# ==========================================================
# ENTORNO SAC PROBABILÍSTICO
# ==========================================================

def reiniciar_entorno_sac_probabilistico(semilla):
    semilla = int(semilla)
    entorno, observacion_reactiva = base.reiniciar_entorno_sac(semilla=semilla)

    entorno[CLAVE_GENERADOR_SENSOR] = np.random.default_rng(semilla + 702001)
    entorno[CLAVE_GENERADOR_MODELO] = np.random.default_rng(semilla + 703001)
    entorno[CLAVE_CONFIGURACION] = crear_configuracion_incertidumbre_episodio(
        semilla
    )
    entorno[CLAVE_HISTORIAL] = [
        _copiar_obstaculos(entorno["obstaculos_dinamicos"])
    ]

    observados = obtener_obstaculos_observados_probabilisticos(entorno)
    modelo = crear_modelo_monte_carlo_obstaculos(
        observados,
        entorno[CLAVE_GENERADOR_MODELO],
        NUMERO_MUESTRAS_MONTE_CARLO,
    )
    entorno[CLAVE_MODELO_MC] = modelo

    observacion = construir_observacion_sac_probabilistico(
        estado_robot=entorno["estado_robot"],
        submeta=entorno["submeta"],
        meta=entorno["meta"],
        obstaculos_estaticos=entorno["obstaculos_estaticos"],
        obstaculos_dinamicos=observados,
        velocidad_lineal_actual=entorno["velocidad_lineal_actual"],
        velocidad_angular_actual=entorno["velocidad_angular_actual"],
        indice_progreso=entorno["indice_progreso"],
        numero_puntos_camino=len(entorno["camino_mundo"]),
        tiempo_actual=entorno["tiempo_actual"],
        modelo_monte_carlo=modelo,
    )

    riesgo = calcular_riesgo_dinamico_sac_probabilistico(
        estado_robot=entorno["estado_robot"],
        velocidad_lineal=0.0,
        velocidad_angular=0.0,
        obstaculos_dinamicos=observados,
        obstaculos_estaticos=entorno["obstaculos_estaticos"],
        tiempo_actual=entorno["tiempo_actual"],
        modelo_monte_carlo=modelo,
    )

    # Se mantiene el valor interno "predictivo" para reutilizar sin cambios
    # las funciones de entrenamiento y evaluación del programa base.
    entorno["modo_sac"] = "predictivo"
    entorno["subvariante_sac"] = VARIANTE_CHECKPOINT
    entorno["observacion_reactiva_inicial"] = {
        "parche": observacion_reactiva["parche"].copy(),
        "escalares": observacion_reactiva["escalares"].copy(),
    }
    entorno["observacion"] = {
        "parche": observacion["parche"].copy(),
        "escalares": observacion["escalares"].copy(),
    }
    entorno["riesgo_dinamico_predictivo"] = riesgo

    registro = entorno["registro"]
    for clave in (
        "clearances_dinamicos_predichos",
        "ttc_dinamicos_predichos",
        "velocidades_cierre_dinamicas",
        "colisiones_dinamicas_predichas",
        "penalizaciones_riesgo_predictivo",
        "probabilidades_colision_predichas",
        "probabilidades_casi_colision_predichas",
        "clearances_cvar_predichos",
        "ttc_cvar_predichos",
    ):
        registro[clave] = []
    entorno["registro"] = registro

    return entorno, {
        "parche": observacion["parche"].copy(),
        "escalares": observacion["escalares"].copy(),
    }


def ejecutar_paso_entorno_sac_probabilistico(
    entorno,
    accion,
    pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
    dt=base.DT,
):
    if entorno.get("subvariante_sac") != VARIANTE_CHECKPOINT:
        raise ValueError("El entorno no fue reiniciado como SAC probabilístico.")

    (
        observacion_reactiva,
        recompensa_reactiva,
        terminado,
        truncado,
        informacion,
    ) = base.ejecutar_paso_entorno_sac(
        entorno=entorno,
        accion=accion,
        pasos_maximos=pasos_maximos,
        dt=dt,
    )

    historial = entorno[CLAVE_HISTORIAL]
    historial.append(_copiar_obstaculos(entorno["obstaculos_dinamicos"]))
    maximo_historial = LATENCIA_MAXIMA_PASOS + 2
    if len(historial) > maximo_historial:
        del historial[:-maximo_historial]
    entorno[CLAVE_HISTORIAL] = historial

    observados = obtener_obstaculos_observados_probabilisticos(entorno)
    modelo = crear_modelo_monte_carlo_obstaculos(
        observados,
        entorno[CLAVE_GENERADOR_MODELO],
        NUMERO_MUESTRAS_MONTE_CARLO,
    )
    entorno[CLAVE_MODELO_MC] = modelo

    riesgo = calcular_riesgo_dinamico_sac_probabilistico(
        estado_robot=entorno["estado_robot"],
        velocidad_lineal=informacion["velocidad_lineal"],
        velocidad_angular=informacion["velocidad_angular"],
        obstaculos_dinamicos=observados,
        obstaculos_estaticos=entorno["obstaculos_estaticos"],
        tiempo_actual=entorno["tiempo_actual"],
        modelo_monte_carlo=modelo,
    )
    componentes_probabilisticos = calcular_penalizacion_riesgo_sac_probabilistico(
        riesgo,
        informacion["velocidad_lineal"],
    )

    penalizacion = float(
        componentes_probabilisticos["penalizacion_riesgo_dinamico_predictivo"]
    )
    recompensa_total = float(recompensa_reactiva - penalizacion)

    observacion = construir_observacion_sac_probabilistico(
        estado_robot=entorno["estado_robot"],
        submeta=entorno["submeta"],
        meta=entorno["meta"],
        obstaculos_estaticos=entorno["obstaculos_estaticos"],
        obstaculos_dinamicos=observados,
        velocidad_lineal_actual=entorno["velocidad_lineal_actual"],
        velocidad_angular_actual=entorno["velocidad_angular_actual"],
        indice_progreso=entorno["indice_progreso"],
        numero_puntos_camino=len(entorno["camino_mundo"]),
        tiempo_actual=entorno["tiempo_actual"],
        modelo_monte_carlo=modelo,
    )

    entorno["observacion_reactiva_ultima"] = {
        "parche": observacion_reactiva["parche"].copy(),
        "escalares": observacion_reactiva["escalares"].copy(),
    }
    entorno["observacion"] = {
        "parche": observacion["parche"].copy(),
        "escalares": observacion["escalares"].copy(),
    }
    entorno["riesgo_dinamico_predictivo"] = riesgo

    registro = entorno["registro"]
    registro["clearances_dinamicos_predichos"].append(
        float(riesgo["clearance_dinamico_predicho_minimo"])
    )
    registro["ttc_dinamicos_predichos"].append(float(riesgo["ttc_predicho"]))
    registro["velocidades_cierre_dinamicas"].append(
        float(riesgo["velocidad_cierre_maxima"])
    )
    registro["colisiones_dinamicas_predichas"].append(
        int(riesgo["colision_predicha"])
    )
    registro["penalizaciones_riesgo_predictivo"].append(penalizacion)
    registro["probabilidades_colision_predichas"].append(
        float(riesgo["probabilidad_colision_predicha"])
    )
    registro["probabilidades_casi_colision_predichas"].append(
        float(riesgo["probabilidad_casi_colision_predicha"])
    )
    registro["clearances_cvar_predichos"].append(
        float(riesgo["clearance_cvar_inferior"])
    )
    registro["ttc_cvar_predichos"].append(float(riesgo["ttc_cvar_inferior"]))
    entorno["registro"] = registro

    componentes_recompensa = informacion["componentes_recompensa"].copy()
    componentes_recompensa.update(riesgo)
    componentes_recompensa.update(componentes_probabilisticos)
    componentes_recompensa["recompensa_reactiva"] = float(recompensa_reactiva)
    componentes_recompensa["recompensa_total"] = recompensa_total

    configuracion = entorno[CLAVE_CONFIGURACION]
    informacion.update(
        {
            "modo_sac": "predictivo",
            "subvariante_sac": VARIANTE_CHECKPOINT,
            "recompensa_reactiva": float(recompensa_reactiva),
            "recompensa_predictiva": recompensa_total,
            "recompensa_probabilistica": recompensa_total,
            "clearance_dinamico_predicho_minimo": float(
                riesgo["clearance_dinamico_predicho_minimo"]
            ),
            "tiempo_clearance_dinamico_minimo": float(
                riesgo["tiempo_clearance_minimo"]
            ),
            "ttc_dinamico_predicho": float(riesgo["ttc_predicho"]),
            "velocidad_cierre_dinamica_maxima": float(
                riesgo["velocidad_cierre_maxima"]
            ),
            "colision_dinamica_predicha": bool(riesgo["colision_predicha"]),
            "probabilidad_colision_predicha": float(
                riesgo["probabilidad_colision_predicha"]
            ),
            "probabilidad_casi_colision_predicha": float(
                riesgo["probabilidad_casi_colision_predicha"]
            ),
            "clearance_cvar_inferior": float(riesgo["clearance_cvar_inferior"]),
            "ttc_cvar_inferior": float(riesgo["ttc_cvar_inferior"]),
            "latencia_observacion_pasos": int(configuracion["latencia_pasos"]),
            "sigma_posicion_observacion_m": float(
                configuracion["sigma_posicion_m"]
            ),
            "sigma_velocidad_observacion_m_s": float(
                configuracion["sigma_velocidad_m_s"]
            ),
            "penalizacion_riesgo_dinamico_predictivo": penalizacion,
            "penalizacion_riesgo_probabilistico": penalizacion,
            "componentes_recompensa": componentes_recompensa,
            "recompensa": recompensa_total,
        }
    )

    return (
        {
            "parche": observacion["parche"].copy(),
            "escalares": observacion["escalares"].copy(),
        },
        recompensa_total,
        terminado,
        truncado,
        informacion,
    )


# ==========================================================
# PARCHEO CONTROLADO DEL PROGRAMA BASE
# ==========================================================

def configurar_programa_base(
    numero_episodios: int,
    mostrar_graficas: bool,
) -> None:
    """Sustituye únicamente la rama predictiva y las rutas de salida."""

    # Funciones que las rutinas de entrenamiento resuelven en tiempo de ejecución.
    base.predecir_obstaculos_dinamicos_sac_predictivo = (
        predecir_obstaculos_dinamicos_sac_probabilistico
    )
    base.construir_parche_egocentrico_sac_predictivo = (
        construir_parche_egocentrico_sac_probabilistico
    )
    base.construir_observacion_sac_predictivo = (
        construir_observacion_sac_probabilistico
    )
    base.calcular_riesgo_dinamico_predictivo_sac = (
        calcular_riesgo_dinamico_sac_probabilistico
    )
    base.calcular_penalizacion_riesgo_dinamico_predictivo_sac = (
        calcular_penalizacion_riesgo_sac_probabilistico
    )
    base.reiniciar_entorno_sac_predictivo = reiniciar_entorno_sac_probabilistico
    base.ejecutar_paso_entorno_sac_predictivo = (
        ejecutar_paso_entorno_sac_probabilistico
    )

    # Configuración que queda registrada por el checkpoint compatible.
    base.HORIZONTES_PREDICCION_PARCHE_SAC = HORIZONTES_PROBABILISTICOS
    base.INTENSIDADES_PREDICCION_PARCHE_SAC = PESOS_HORIZONTES_PROBABILISTICOS
    base.HORIZONTE_RIESGO_DINAMICO_SAC = HORIZONTE_RIESGO_PROBABILISTICO_S
    base.PASO_PREDICCION_DINAMICA_SAC = PASO_RIESGO_PROBABILISTICO_S
    base.CLEARANCE_DINAMICO_PREDICHO_OBJETIVO_SAC = CLEARANCE_CVAR_OBJETIVO_M
    base.TTC_OBJETIVO_SAC = TTC_CVAR_OBJETIVO_S

    base.NUMERO_EPISODIOS_ENTRENAMIENTO_SAC_PREDICTIVO = int(numero_episodios)
    base.SEMILLA_BASE_ENTRENAMIENTO_SAC_PREDICTIVO = (
        SEMILLA_BASE_ENTRENAMIENTO_PROBABILISTICO
    )
    base.SEMILLA_BASE_VALIDACION_SAC_PREDICTIVO = (
        SEMILLA_BASE_VALIDACION_PROBABILISTICO
    )
    base.SEMILLA_DIAGNOSTICA_SAC_PREDICTIVO = (
        SEMILLA_DIAGNOSTICA_PROBABILISTICO
    )

    directorio = Path("resultados_sac/entrenamiento_sac_probabilistico_upo_ttc")
    base.DIRECTORIO_ENTRENAMIENTO_SAC_PREDICTIVO = directorio
    base.RUTA_MEJOR_ACTOR_SAC_PREDICTIVO = (
        directorio / "checkpoint_mejor_actor_sac_probabilistico.pt"
    )
    base.RUTA_CHECKPOINT_FINAL_SAC_PREDICTIVO = (
        directorio / "checkpoint_final_completo_sac_probabilistico.pt"
    )
    base.DIRECTORIO_GRAFICAS_ENTRENAMIENTO_SAC_PREDICTIVO = (
        directorio / "graficas_entrenamiento"
    )
    base.RUTA_CSV_VALIDACIONES_SAC_PREDICTIVO = (
        directorio / "validaciones_periodicas_50_semillas.csv"
    )
    base.RUTA_GRAFICA_VALIDACIONES_SAC_PREDICTIVO = (
        directorio / "validaciones_periodicas_50_semillas.png"
    )
    base.RUTA_GRAFICA_DIAGNOSTICA_SAC_PREDICTIVO = (
        directorio / "trayectoria_diagnostica_mejor_actor_probabilistico.png"
    )

    # La verificación original exige valores discretos {0, intensidades}; el
    # mapa probabilístico es continuo, por eso se usa la verificación propia.
    base.EJECUTAR_VERIFICACION_PREVIA_SAC_PREDICTIVO = False
    base.MOSTRAR_RESULTADOS_ENTRENAMIENTO_SAC_PREDICTIVO = bool(mostrar_graficas)


# ==========================================================
# VERIFICACIÓN PREVIA ESPECÍFICA
# ==========================================================

def verificar_sac_probabilistico() -> bool:
    entorno_reactivo, observacion_reactiva = base.reiniciar_entorno_sac(
        semilla=base.SEMILLA
    )
    entorno_1, observacion_1 = reiniciar_entorno_sac_probabilistico(base.SEMILLA)
    entorno_2, observacion_2 = reiniciar_entorno_sac_probabilistico(base.SEMILLA)

    mismo_escenario = (
        entorno_reactivo["estado_inicial"] == entorno_1["estado_inicial"]
        and entorno_reactivo["meta"] == entorno_1["meta"]
        and entorno_reactivo["camino_mundo"] == entorno_1["camino_mundo"]
    )
    formas_correctas = (
        observacion_1["parche"].shape
        == (
            base.CANALES_PARCHE_SAC,
            base.RESOLUCION_PARCHE_SAC,
            base.RESOLUCION_PARCHE_SAC,
        )
        and observacion_1["escalares"].shape == (base.DIMENSION_ESCALARES_SAC,)
    )
    canal_estatico_igual = np.array_equal(
        observacion_reactiva["parche"][0],
        observacion_1["parche"][0],
    )
    canal_probabilistico = observacion_1["parche"][1]
    canal_en_rango = (
        np.all(np.isfinite(canal_probabilistico))
        and float(np.min(canal_probabilistico)) >= 0.0
        and float(np.max(canal_probabilistico)) <= 1.0
    )
    reproducible = (
        entorno_1[CLAVE_CONFIGURACION] == entorno_2[CLAVE_CONFIGURACION]
        and np.array_equal(observacion_1["parche"], observacion_2["parche"])
        and np.array_equal(observacion_1["escalares"], observacion_2["escalares"])
    )

    accion = np.zeros(base.DIMENSION_ACCION_SAC, dtype=np.float32)
    observacion_nueva, recompensa, terminado, truncado, informacion = (
        ejecutar_paso_entorno_sac_probabilistico(
            entorno_1,
            accion,
            pasos_maximos=10,
            dt=base.DT,
        )
    )

    claves = {
        "probabilidad_colision_predicha",
        "probabilidad_casi_colision_predicha",
        "clearance_cvar_inferior",
        "ttc_cvar_inferior",
        "latencia_observacion_pasos",
        "penalizacion_riesgo_probabilistico",
    }
    paso_integrado = (
        claves.issubset(informacion)
        and np.isfinite(recompensa)
        and observacion_nueva["parche"].shape == observacion_1["parche"].shape
        and isinstance(terminado, bool)
        and isinstance(truncado, bool)
    )
    probabilidades_validas = (
        0.0 <= informacion["probabilidad_colision_predicha"] <= 1.0
        and 0.0 <= informacion["probabilidad_casi_colision_predicha"] <= 1.0
    )
    registro_actualizado = all(
        len(entorno_1["registro"][clave]) == 1
        for clave in (
            "probabilidades_colision_predichas",
            "probabilidades_casi_colision_predichas",
            "clearances_cvar_predichos",
            "ttc_cvar_predichos",
        )
    )

    resultados = {
        "mismo_escenario": mismo_escenario,
        "formas_correctas": formas_correctas,
        "canal_estatico_igual": canal_estatico_igual,
        "canal_probabilistico_en_rango": canal_en_rango,
        "reproducibilidad": reproducible,
        "paso_integrado": paso_integrado,
        "probabilidades_validas": probabilidades_validas,
        "registro_actualizado": registro_actualizado,
    }

    print("\n" + "=" * 80)
    print("VERIFICACIÓN PREVIA DEL SAC-UPO-TTC")
    print("=" * 80)
    print(f"Programa base: {_RUTA_BASE.name}")
    print(f"Muestras Monte Carlo: {NUMERO_MUESTRAS_MONTE_CARLO}")
    for clave, valor in resultados.items():
        print(f"{clave}: {valor}")
    print("Configuración de incertidumbre:", entorno_1[CLAVE_CONFIGURACION])
    print(
        "Rango del canal dinámico:",
        float(np.min(canal_probabilistico)),
        "a",
        float(np.max(canal_probabilistico)),
    )

    correcto = all(resultados.values())
    if not correcto:
        raise RuntimeError("La verificación del SAC probabilístico falló.")

    print("RESULTADO DE VERIFICACIÓN: TODO CORRECTO")
    print("=" * 80)
    return True


# ==========================================================
# METADATOS CIENTÍFICOS DEL CHECKPOINT
# ==========================================================

def _configuracion_probabilistica_checkpoint() -> Dict[str, object]:
    return {
        "metodo": NOMBRE_METODO,
        "subvariante": VARIANTE_CHECKPOINT,
        "numero_muestras_monte_carlo": int(NUMERO_MUESTRAS_MONTE_CARLO),
        "horizontes_s": tuple(HORIZONTES_PROBABILISTICOS),
        "pesos_horizontes": tuple(PESOS_HORIZONTES_PROBABILISTICOS),
        "latencia_maxima_pasos": int(LATENCIA_MAXIMA_PASOS),
        "sigma_posicion_m": (SIGMA_POSICION_MIN_M, SIGMA_POSICION_MAX_M),
        "sigma_velocidad_m_s": (
            SIGMA_VELOCIDAD_MIN_M_S,
            SIGMA_VELOCIDAD_MAX_M_S,
        ),
        "sigma_velocidad_relativa_modelo": SIGMA_VELOCIDAD_RELATIVA,
        "sigma_direccion_rad": SIGMA_DIRECCION_RAD,
        "sigma_aceleracion_m_s2": SIGMA_ACELERACION_M_S2,
        "sigma_giro_rad_s": SIGMA_GIRO_RAD_S,
        "probabilidad_detencion": PROBABILIDAD_DETENCION,
        "alfa_cvar": ALFA_CVAR,
        "umbral_casi_colision_m": UMBRAL_CASI_COLISION_M,
        "clearance_cvar_objetivo_m": CLEARANCE_CVAR_OBJETIVO_M,
        "ttc_cvar_objetivo_s": TTC_CVAR_OBJETIVO_S,
        "entrenamiento_con_ruido_y_latencia": True,
        "predictor_comparte_dinamica_real": False,
    }


def etiquetar_checkpoint_probabilistico(
    ruta: Path,
    tipo_probabilistico: str,
) -> None:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint {ruta}")

    checkpoint = torch.load(ruta, map_location="cpu", weights_only=False)
    checkpoint["tipo_compatible_previo"] = checkpoint.get("tipo")
    checkpoint["variante_compatible_previa"] = checkpoint.get("variante_sac")
    checkpoint["tipo"] = tipo_probabilistico
    checkpoint["variante_sac"] = VARIANTE_CHECKPOINT
    checkpoint["observacion_predictiva"] = True
    checkpoint["observacion_probabilistica"] = True
    checkpoint["usa_ttc"] = True
    checkpoint["usa_ttc_probabilistico"] = True
    checkpoint["usa_cvar"] = True
    checkpoint["prediccion_monte_carlo"] = True
    checkpoint["configuracion_probabilistica"] = (
        _configuracion_probabilistica_checkpoint()
    )

    temporal = ruta.with_suffix(ruta.suffix + ".temporal_probabilistico")
    torch.save(checkpoint, temporal)
    temporal.replace(ruta)


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def analizar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena SAC-UPO-TTC con ocupación futura probabilística."
    )
    parser.add_argument(
        "--solo-verificacion",
        action="store_true",
        help="Verifica la integración sin requerir checkpoints ni entrenar.",
    )
    parser.add_argument(
        "--episodios",
        type=int,
        default=400,
        help="Número de episodios de entrenamiento. Predeterminado: 400.",
    )
    parser.add_argument(
        "--muestras",
        type=int,
        default=12,
        help="Muestras Monte Carlo por obstáculo. Predeterminado: 12.",
    )
    parser.add_argument(
        "--sin-graficas",
        action="store_true",
        help="No mostrar ventanas al terminar; los archivos sí se guardan.",
    )
    return parser.parse_args()


def main() -> None:
    global NUMERO_MUESTRAS_MONTE_CARLO

    argumentos = analizar_argumentos()
    if argumentos.episodios <= 0:
        raise ValueError("El número de episodios debe ser positivo.")
    if argumentos.muestras < 2:
        raise ValueError("El número de muestras Monte Carlo debe ser al menos 2.")

    NUMERO_MUESTRAS_MONTE_CARLO = int(argumentos.muestras)
    configurar_programa_base(
        numero_episodios=int(argumentos.episodios),
        mostrar_graficas=not argumentos.sin_graficas,
    )

    verificar_sac_probabilistico()
    if argumentos.solo_verificacion:
        return

    print("\n" + "=" * 80)
    print("COMIENZA EL ENTRENAMIENTO SAC-UPO-TTC")
    print("=" * 80)
    print("Actor inicial: SAC reactivo mejorado")
    print("Observación: ocupación futura probabilística")
    print("Riesgo: P(colisión), P(casi colisión), clearance-CVaR y TTC-CVaR")
    print("Ruido y latencia durante entrenamiento: activados")
    print("Dinámica real nominal compartida con experimentos anteriores: sí")
    print("Modelo predictor idéntico a dinámica real: no")

    # Reutiliza el entrenamiento validado del programa base.
    base.main()

    etiquetar_checkpoint_probabilistico(
        base.RUTA_MEJOR_ACTOR_SAC_PREDICTIVO,
        "mejor_actor_sac_probabilistico_upo_ttc",
    )
    etiquetar_checkpoint_probabilistico(
        base.RUTA_CHECKPOINT_FINAL_SAC_PREDICTIVO,
        "sac_probabilistico_upo_ttc_entrenamiento_completo",
    )

    print("\n" + "=" * 80)
    print("CHECKPOINTS ETIQUETADOS COMO SAC-UPO-TTC")
    print("Mejor actor:", base.RUTA_MEJOR_ACTOR_SAC_PREDICTIVO)
    print("Checkpoint final:", base.RUTA_CHECKPOINT_FINAL_SAC_PREDICTIVO)
    print("=" * 80)


if __name__ == "__main__":
    main()
