from __future__ import annotations

"""
FASE 2 — Comparación pareada de robustez perceptual de cinco métodos.

Métodos:
    1. A* + DWA
    2. A* + TD3
    3. A* + SAC-R
    4. A* + SAC-PO-TTC
    5. A* + SAC-UPO-TTC

Este programa reutiliza el comparador nominal y los mismos checkpoints congelados,
pero añade una corrupción sensorial EXTERNA, determinista y común para los cinco
métodos. La dinámica física, las colisiones y las métricas se calculan siempre con
el estado verdadero del simulador.

Condiciones fijadas antes de observar la fase 2:
    moderada: latencia=1 paso, sigma_pos=0.04 m, sigma_vel=0.06 m/s
    severa:   latencia=3 pasos, sigma_pos=0.08 m, sigma_vel=0.12 m/s

La misma perturbación se obtiene para una combinación dada de:
    condición + semilla + paso + índice de obstáculo.
Por ello, el orden de ejecución de los controladores no modifica la percepción.

Archivos requeridos en la misma carpeta:
    Comparador_5_metodos_nominal_metricas_ampliadas.py
    SAC_predictivo_entrenamiento_completo.py
    TD3_tercera_corrida_desde_cero.py
    SAC_probabilistico_ocupacion_TTC_entrenamiento.py

y los cuatro checkpoints utilizados en la fase 1.

Verificación técnica:
    python Comparador_5_metodos_fase2_incertidumbre_comun.py \
        --condicion moderada --solo-verificacion --sin-graficas

Condición moderada:
    python Comparador_5_metodos_fase2_incertidumbre_comun.py \
        --condicion moderada --semilla-base 170000 --numero-semillas 200

Condición severa:
    python Comparador_5_metodos_fase2_incertidumbre_comun.py \
        --condicion severa --semilla-base 170000 --numero-semillas 200

Ambas condiciones, una después de la otra:
    python Comparador_5_metodos_fase2_incertidumbre_comun.py \
        --condicion ambas --semilla-base 170000 --numero-semillas 200
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# IMPORTAR EL COMPARADOR NOMINAL COMO BIBLIOTECA
# =============================================================================

DIRECTORIO_PROGRAMA = Path(__file__).resolve().parent
NOMBRE_COMPARADOR_NOMINAL = "Comparador_5_metodos_nominal_metricas_ampliadas.py"
RUTA_COMPARADOR_NOMINAL = DIRECTORIO_PROGRAMA / NOMBRE_COMPARADOR_NOMINAL


def importar_comparador_nominal():
    if not RUTA_COMPARADOR_NOMINAL.is_file():
        raise FileNotFoundError(
            f"No se encontró {NOMBRE_COMPARADOR_NOMINAL} junto a este programa."
        )
    spec = importlib.util.spec_from_file_location(
        "comparador_nominal_biblioteca_fase2", RUTA_COMPARADOR_NOMINAL
    )
    if spec is None or spec.loader is None:
        raise ImportError("No fue posible preparar la importación del comparador nominal")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


nominal = importar_comparador_nominal()


# =============================================================================
# CONFIGURACIÓN EXPERIMENTAL PREESPECIFICADA
# =============================================================================

VERSION_GENERADOR_RUIDO = "fase2_percepcion_comun_v1"
SEMILLA_VERIFICACION_FASE2 = 169901
PASOS_HUELLA_PERCEPCION = 3


@dataclass(frozen=True)
class CondicionIncertidumbre:
    nombre: str
    etiqueta: str
    latencia_pasos: int
    sigma_posicion_m: float
    sigma_velocidad_m_s: float


CONDICIONES: Dict[str, CondicionIncertidumbre] = {
    "moderada": CondicionIncertidumbre(
        nombre="incertidumbre_moderada",
        etiqueta="Moderada",
        latencia_pasos=1,
        sigma_posicion_m=0.04,
        sigma_velocidad_m_s=0.06,
    ),
    "severa": CondicionIncertidumbre(
        nombre="incertidumbre_severa",
        etiqueta="Severa",
        latencia_pasos=3,
        sigma_posicion_m=0.08,
        sigma_velocidad_m_s=0.12,
    ),
}

CAMPOS_PERCEPCION = [
    "condicion",
    "semilla",
    "metodo",
    "numero_pasos_percepcion",
    "numero_pasos_huella",
    "huella_percepcion_primeros_pasos",
    "latencia_pasos",
    "sigma_posicion_m",
    "sigma_velocidad_m_s",
    "version_generador_ruido",
]


# =============================================================================
# PERCEPCIÓN COMÚN, DETERMINISTA Y SEPARADA DEL MUNDO REAL
# =============================================================================


def copiar_obstaculos(obstaculos: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [copy.deepcopy(dict(obstaculo)) for obstaculo in obstaculos]


def semilla_determinista_ruido(
    condicion: CondicionIncertidumbre,
    semilla_escenario: int,
    paso_actual: int,
    indice_obstaculo: int,
) -> int:
    texto = (
        f"{VERSION_GENERADOR_RUIDO}|{condicion.nombre}|{int(semilla_escenario)}|"
        f"{int(paso_actual)}|{int(indice_obstaculo)}"
    )
    digest = hashlib.sha256(texto.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


class SensorDinamicoComun:
    """Produce la misma observación retardada y ruidosa para todos los métodos."""

    def __init__(
        self,
        condicion: CondicionIncertidumbre,
        semilla_escenario: int,
        ancho_mapa: float,
        alto_mapa: float,
    ) -> None:
        self.condicion = condicion
        self.semilla_escenario = int(semilla_escenario)
        self.ancho_mapa = float(ancho_mapa)
        self.alto_mapa = float(alto_mapa)
        self.historial_verdadero: Dict[int, List[Dict[str, Any]]] = {}
        self.observaciones_cache: Dict[int, List[Dict[str, Any]]] = {}
        self.huellas_por_paso: Dict[int, str] = {}

    def _registrar_estado_verdadero(
        self,
        paso_actual: int,
        obstaculos_verdaderos: Sequence[Mapping[str, Any]],
    ) -> None:
        paso_actual = int(paso_actual)
        copia = copiar_obstaculos(obstaculos_verdaderos)
        if paso_actual in self.historial_verdadero:
            anterior = nominal.huella_json(self.historial_verdadero[paso_actual])
            nuevo = nominal.huella_json(copia)
            if anterior != nuevo:
                raise RuntimeError(
                    "El estado verdadero de obstáculos cambió al repetir el mismo paso."
                )
        else:
            self.historial_verdadero[paso_actual] = copia

    def observar(
        self,
        paso_actual: int,
        obstaculos_verdaderos: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        paso_actual = int(paso_actual)
        self._registrar_estado_verdadero(paso_actual, obstaculos_verdaderos)

        if paso_actual in self.observaciones_cache:
            return copiar_obstaculos(self.observaciones_cache[paso_actual])

        paso_retrasado = max(0, paso_actual - int(self.condicion.latencia_pasos))
        if paso_retrasado not in self.historial_verdadero:
            pasos_disponibles = [p for p in self.historial_verdadero if p <= paso_retrasado]
            paso_retrasado = max(pasos_disponibles) if pasos_disponibles else min(self.historial_verdadero)

        observados = copiar_obstaculos(self.historial_verdadero[paso_retrasado])

        for indice, obstaculo in enumerate(observados):
            generador = np.random.default_rng(
                semilla_determinista_ruido(
                    self.condicion,
                    self.semilla_escenario,
                    paso_actual,
                    indice,
                )
            )
            ruido = generador.standard_normal(4)
            radio = float(obstaculo.get("radio", 0.0))

            x = float(obstaculo["x"]) + self.condicion.sigma_posicion_m * float(ruido[0])
            y = float(obstaculo["y"]) + self.condicion.sigma_posicion_m * float(ruido[1])
            vx = float(obstaculo.get("vx", 0.0)) + self.condicion.sigma_velocidad_m_s * float(ruido[2])
            vy = float(obstaculo.get("vy", 0.0)) + self.condicion.sigma_velocidad_m_s * float(ruido[3])

            obstaculo["x"] = float(np.clip(x, radio, self.ancho_mapa - radio))
            obstaculo["y"] = float(np.clip(y, radio, self.alto_mapa - radio))
            obstaculo["vx"] = float(vx)
            obstaculo["vy"] = float(vy)
            obstaculo["observacion_latencia_pasos"] = int(self.condicion.latencia_pasos)
            obstaculo["observacion_sigma_posicion_m"] = float(
                self.condicion.sigma_posicion_m
            )
            obstaculo["observacion_sigma_velocidad_m_s"] = float(
                self.condicion.sigma_velocidad_m_s
            )
            obstaculo["observacion_paso_actual"] = paso_actual
            obstaculo["observacion_paso_retrasado"] = paso_retrasado

        self.observaciones_cache[paso_actual] = copiar_obstaculos(observados)
        self.huellas_por_paso[paso_actual] = nominal.huella_json(observados)
        return copiar_obstaculos(observados)

    def huella_primeros_pasos(self, cantidad: int = PASOS_HUELLA_PERCEPCION) -> str:
        pasos = sorted(self.huellas_por_paso)[: int(cantidad)]
        contenido = [(paso, self.huellas_por_paso[paso]) for paso in pasos]
        return nominal.huella_json(contenido)

    def numero_pasos(self) -> int:
        return len(self.huellas_por_paso)


# =============================================================================
# CONSTRUCCIÓN DE OBSERVACIONES BAJO INCERTIDUMBRE
# =============================================================================


def construir_observacion_reactiva(base, entorno, obstaculos_observados):
    return base.construir_observacion_sac(
        estado_robot=entorno["estado_robot"],
        submeta=entorno["submeta"],
        meta=entorno["meta"],
        obstaculos_estaticos=entorno["obstaculos_estaticos"],
        obstaculos_dinamicos=obstaculos_observados,
        velocidad_lineal_actual=entorno["velocidad_lineal_actual"],
        velocidad_angular_actual=entorno["velocidad_angular_actual"],
        indice_progreso=entorno["indice_progreso"],
        numero_puntos_camino=len(entorno["camino_mundo"]),
    )


def inicializar_registro_predictivo(registro: MutableMapping[str, Any], probabilistico: bool) -> None:
    claves = [
        "clearances_dinamicos_predichos",
        "ttc_dinamicos_predichos",
        "velocidades_cierre_dinamicas",
        "colisiones_dinamicas_predichas",
        "penalizaciones_riesgo_predictivo",
    ]
    if probabilistico:
        claves.extend(
            [
                "probabilidades_colision_predichas",
                "probabilidades_casi_colision_predichas",
                "clearances_cvar_predichos",
                "ttc_cvar_predichos",
            ]
        )
    for clave in claves:
        registro[clave] = []


def registrar_riesgo_predictivo(
    registro: MutableMapping[str, Any],
    riesgo: Mapping[str, Any],
    penalizacion: float,
    probabilistico: bool,
) -> None:
    registro["clearances_dinamicos_predichos"].append(
        float(riesgo["clearance_dinamico_predicho_minimo"])
    )
    registro["ttc_dinamicos_predichos"].append(float(riesgo["ttc_predicho"]))
    registro["velocidades_cierre_dinamicas"].append(
        float(riesgo["velocidad_cierre_maxima"])
    )
    registro["colisiones_dinamicas_predichas"].append(
        int(bool(riesgo["colision_predicha"]))
    )
    registro["penalizaciones_riesgo_predictivo"].append(float(penalizacion))

    if probabilistico:
        registro["probabilidades_colision_predichas"].append(
            float(riesgo["probabilidad_colision_predicha"])
        )
        registro["probabilidades_casi_colision_predichas"].append(
            float(riesgo["probabilidad_casi_colision_predicha"])
        )
        registro["clearances_cvar_predichos"].append(
            float(riesgo["clearance_cvar_inferior"])
        )
        registro["ttc_cvar_predichos"].append(
            float(riesgo["ttc_cvar_inferior"])
        )


def adjuntar_informacion_sensor(resultado: Dict[str, Any], sensor: SensorDinamicoComun) -> Dict[str, Any]:
    resultado["huella_percepcion_primeros_pasos"] = sensor.huella_primeros_pasos()
    resultado["numero_pasos_percepcion"] = sensor.numero_pasos()
    return resultado


# =============================================================================
# EJECUCIÓN DE LOS MÉTODOS
# =============================================================================


def ejecutar_reactivo_fase2(
    base,
    actor: torch.nn.Module,
    semilla: int,
    condicion: CondicionIncertidumbre,
    metodo: str,
    td3=None,
) -> Dict[str, Any]:
    inicio_preparacion = time.perf_counter()
    entorno, _ = base.reiniciar_entorno_sac(int(semilla))
    sensor = SensorDinamicoComun(
        condicion, semilla, base.ANCHO_MAPA, base.ALTO_MAPA
    )
    observados = sensor.observar(0, entorno["obstaculos_dinamicos"])
    observacion = construir_observacion_reactiva(base, entorno, observados)
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    terminado = False
    truncado = False
    recompensa_acumulada = 0.0
    tiempos_inferencia: List[float] = []
    tiempos_paso: List[float] = []
    actor.eval()

    while not (terminado or truncado):
        inicio_inferencia = time.perf_counter()
        if td3 is None:
            accion, info_accion = base.seleccionar_accion_actor_sac(
                observacion=observacion,
                actor=actor,
                dispositivo=base.DISPOSITIVO_SAC,
                determinista=True,
            )
            if info_accion.get("modo") != "determinista":
                raise RuntimeError(f"{metodo} no produjo una acción determinista")
        else:
            accion = td3.accion_td3(
                actor=actor,
                observacion=observacion,
                dispositivo=base.DISPOSITIVO_SAC,
                ruido=0.0,
            )
        tiempos_inferencia.append(time.perf_counter() - inicio_inferencia)

        inicio_paso = time.perf_counter()
        _, recompensa, terminado, truncado, _ = base.ejecutar_paso_entorno_sac(
            entorno=entorno,
            accion=accion,
            pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
            dt=base.DT,
        )
        paso = int(entorno["paso_actual"])
        observados = sensor.observar(paso, entorno["obstaculos_dinamicos"])
        observacion = construir_observacion_reactiva(base, entorno, observados)
        entorno["observacion"] = {
            "parche": observacion["parche"].copy(),
            "escalares": observacion["escalares"].copy(),
        }
        tiempos_paso.append(time.perf_counter() - inicio_paso)
        recompensa_acumulada += float(recompensa)

    resultado = nominal.crear_resultado_desde_entorno(
        base=base,
        entorno=entorno,
        recompensa_acumulada=recompensa_acumulada,
        tiempos_inferencia=tiempos_inferencia,
        tiempos_paso=tiempos_paso,
        tiempo_preparacion=tiempo_preparacion,
        metodo=metodo,
    )
    return adjuntar_informacion_sensor(resultado, sensor)


def ejecutar_po_ttc_fase2(
    base,
    actor: torch.nn.Module,
    semilla: int,
    condicion: CondicionIncertidumbre,
) -> Dict[str, Any]:
    inicio_preparacion = time.perf_counter()
    entorno, _ = base.reiniciar_entorno_sac(int(semilla))
    inicializar_registro_predictivo(entorno["registro"], probabilistico=False)
    entorno["modo_sac"] = "predictivo"

    sensor = SensorDinamicoComun(
        condicion, semilla, base.ANCHO_MAPA, base.ALTO_MAPA
    )
    observados = sensor.observar(0, entorno["obstaculos_dinamicos"])
    observacion = base.construir_observacion_sac_predictivo(
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
    )
    entorno["observacion"] = {
        "parche": observacion["parche"].copy(),
        "escalares": observacion["escalares"].copy(),
    }
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    terminado = False
    truncado = False
    recompensa_acumulada = 0.0
    tiempos_inferencia: List[float] = []
    tiempos_paso: List[float] = []
    actor.eval()

    while not (terminado or truncado):
        inicio_inferencia = time.perf_counter()
        accion, info_accion = base.seleccionar_accion_actor_sac(
            observacion=observacion,
            actor=actor,
            dispositivo=base.DISPOSITIVO_SAC,
            determinista=True,
        )
        tiempos_inferencia.append(time.perf_counter() - inicio_inferencia)
        if info_accion.get("modo") != "determinista":
            raise RuntimeError("SAC-PO-TTC no produjo una acción determinista")

        inicio_paso = time.perf_counter()
        _, recompensa_reactiva, terminado, truncado, informacion = (
            base.ejecutar_paso_entorno_sac(
                entorno=entorno,
                accion=accion,
                pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                dt=base.DT,
            )
        )
        paso = int(entorno["paso_actual"])
        observados = sensor.observar(paso, entorno["obstaculos_dinamicos"])

        riesgo = base.calcular_riesgo_dinamico_predictivo_sac(
            estado_robot=entorno["estado_robot"],
            velocidad_lineal=informacion["velocidad_lineal"],
            velocidad_angular=informacion["velocidad_angular"],
            obstaculos_dinamicos=observados,
            obstaculos_estaticos=entorno["obstaculos_estaticos"],
            tiempo_actual=entorno["tiempo_actual"],
        )
        componentes = base.calcular_penalizacion_riesgo_dinamico_predictivo_sac(
            riesgo_predictivo=riesgo,
            velocidad_lineal=informacion["velocidad_lineal"],
        )
        penalizacion = float(componentes["penalizacion_riesgo_dinamico_predictivo"])
        recompensa_total = float(recompensa_reactiva - penalizacion)
        registrar_riesgo_predictivo(
            entorno["registro"], riesgo, penalizacion, probabilistico=False
        )

        observacion = base.construir_observacion_sac_predictivo(
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
        )
        entorno["observacion"] = {
            "parche": observacion["parche"].copy(),
            "escalares": observacion["escalares"].copy(),
        }
        entorno["riesgo_dinamico_predictivo"] = dict(riesgo)
        tiempos_paso.append(time.perf_counter() - inicio_paso)
        recompensa_acumulada += recompensa_total

    resultado = nominal.crear_resultado_desde_entorno(
        base=base,
        entorno=entorno,
        recompensa_acumulada=recompensa_acumulada,
        tiempos_inferencia=tiempos_inferencia,
        tiempos_paso=tiempos_paso,
        tiempo_preparacion=tiempo_preparacion,
        metodo="A* + SAC-PO-TTC",
    )
    return adjuntar_informacion_sensor(resultado, sensor)


def ejecutar_upo_ttc_fase2(
    upo,
    actor: torch.nn.Module,
    semilla: int,
    condicion: CondicionIncertidumbre,
) -> Dict[str, Any]:
    base = upo.base
    inicio_preparacion = time.perf_counter()
    entorno, _ = base.reiniciar_entorno_sac(int(semilla))
    inicializar_registro_predictivo(entorno["registro"], probabilistico=True)
    entorno["modo_sac"] = "predictivo"
    entorno["subvariante_sac"] = upo.VARIANTE_CHECKPOINT
    entorno[upo.CLAVE_CONFIGURACION] = {
        "latencia_pasos": int(condicion.latencia_pasos),
        "sigma_posicion_m": float(condicion.sigma_posicion_m),
        "sigma_velocidad_m_s": float(condicion.sigma_velocidad_m_s),
    }

    sensor = SensorDinamicoComun(
        condicion, semilla, base.ANCHO_MAPA, base.ALTO_MAPA
    )
    generador_modelo = np.random.default_rng(int(semilla) + 703001)
    observados = sensor.observar(0, entorno["obstaculos_dinamicos"])
    modelo = upo.crear_modelo_monte_carlo_obstaculos(
        observados,
        generador_modelo,
        upo.NUMERO_MUESTRAS_MONTE_CARLO,
    )
    entorno[upo.CLAVE_MODELO_MC] = modelo
    entorno[upo.CLAVE_OBSERVADOS] = copiar_obstaculos(observados)

    observacion = upo.construir_observacion_sac_probabilistico(
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
    entorno["observacion"] = {
        "parche": observacion["parche"].copy(),
        "escalares": observacion["escalares"].copy(),
    }
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    terminado = False
    truncado = False
    recompensa_acumulada = 0.0
    tiempos_inferencia: List[float] = []
    tiempos_paso: List[float] = []
    actor.eval()

    while not (terminado or truncado):
        inicio_inferencia = time.perf_counter()
        accion, info_accion = base.seleccionar_accion_actor_sac(
            observacion=observacion,
            actor=actor,
            dispositivo=base.DISPOSITIVO_SAC,
            determinista=True,
        )
        tiempos_inferencia.append(time.perf_counter() - inicio_inferencia)
        if info_accion.get("modo") != "determinista":
            raise RuntimeError("SAC-UPO-TTC no produjo una acción determinista")

        inicio_paso = time.perf_counter()
        _, recompensa_reactiva, terminado, truncado, informacion = (
            base.ejecutar_paso_entorno_sac(
                entorno=entorno,
                accion=accion,
                pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                dt=base.DT,
            )
        )
        paso = int(entorno["paso_actual"])
        observados = sensor.observar(paso, entorno["obstaculos_dinamicos"])
        modelo = upo.crear_modelo_monte_carlo_obstaculos(
            observados,
            generador_modelo,
            upo.NUMERO_MUESTRAS_MONTE_CARLO,
        )
        entorno[upo.CLAVE_MODELO_MC] = modelo
        entorno[upo.CLAVE_OBSERVADOS] = copiar_obstaculos(observados)

        riesgo = upo.calcular_riesgo_dinamico_sac_probabilistico(
            estado_robot=entorno["estado_robot"],
            velocidad_lineal=informacion["velocidad_lineal"],
            velocidad_angular=informacion["velocidad_angular"],
            obstaculos_dinamicos=observados,
            obstaculos_estaticos=entorno["obstaculos_estaticos"],
            tiempo_actual=entorno["tiempo_actual"],
            modelo_monte_carlo=modelo,
        )
        componentes = upo.calcular_penalizacion_riesgo_sac_probabilistico(
            riesgo, informacion["velocidad_lineal"]
        )
        penalizacion = float(componentes["penalizacion_riesgo_dinamico_predictivo"])
        recompensa_total = float(recompensa_reactiva - penalizacion)
        registrar_riesgo_predictivo(
            entorno["registro"], riesgo, penalizacion, probabilistico=True
        )

        observacion = upo.construir_observacion_sac_probabilistico(
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
        entorno["observacion"] = {
            "parche": observacion["parche"].copy(),
            "escalares": observacion["escalares"].copy(),
        }
        entorno["riesgo_dinamico_predictivo"] = dict(riesgo)
        tiempos_paso.append(time.perf_counter() - inicio_paso)
        recompensa_acumulada += recompensa_total

    resultado = nominal.crear_resultado_desde_entorno(
        base=base,
        entorno=entorno,
        recompensa_acumulada=recompensa_acumulada,
        tiempos_inferencia=tiempos_inferencia,
        tiempos_paso=tiempos_paso,
        tiempo_preparacion=tiempo_preparacion,
        metodo="A* + SAC-UPO-TTC",
    )
    return adjuntar_informacion_sensor(resultado, sensor)


def ejecutar_dwa_fase2(
    base,
    semilla: int,
    condicion: CondicionIncertidumbre,
) -> Dict[str, Any]:
    sensor = SensorDinamicoComun(
        condicion, semilla, base.ANCHO_MAPA, base.ALTO_MAPA
    )
    controlador_original = base.controlador_dwa
    firma = inspect.signature(controlador_original)

    def controlador_con_percepcion_comun(*args, **kwargs):
        ligados = firma.bind_partial(*args, **kwargs)
        ligados.apply_defaults()
        tiempo_actual = float(ligados.arguments["tiempo_actual"])
        dt = float(ligados.arguments.get("dt", base.DT))
        paso = int(round(tiempo_actual / max(dt, 1e-12)))
        verdaderos = ligados.arguments["obstaculos_dinamicos"]
        ligados.arguments["obstaculos_dinamicos"] = sensor.observar(paso, verdaderos)
        return controlador_original(*ligados.args, **ligados.kwargs)

    base.controlador_dwa = controlador_con_percepcion_comun
    try:
        resultado = nominal.ejecutar_dwa_instrumentado(base, int(semilla))
    finally:
        base.controlador_dwa = controlador_original
    return adjuntar_informacion_sensor(resultado, sensor)


# =============================================================================
# ARCHIVOS, VERIFICACIONES Y ESTADÍSTICA POR CONDICIÓN
# =============================================================================


def fila_percepcion(
    resultado: Mapping[str, Any],
    condicion: CondicionIncertidumbre,
    metodo: str,
) -> Dict[str, Any]:
    numero_pasos = int(resultado.get("numero_pasos_percepcion", 0))
    return {
        "condicion": condicion.nombre,
        "semilla": int(resultado["semilla"]),
        "metodo": metodo,
        "numero_pasos_percepcion": numero_pasos,
        "numero_pasos_huella": min(PASOS_HUELLA_PERCEPCION, numero_pasos),
        "huella_percepcion_primeros_pasos": str(
            resultado.get("huella_percepcion_primeros_pasos", "")
        ),
        "latencia_pasos": int(condicion.latencia_pasos),
        "sigma_posicion_m": float(condicion.sigma_posicion_m),
        "sigma_velocidad_m_s": float(condicion.sigma_velocidad_m_s),
        "version_generador_ruido": VERSION_GENERADOR_RUIDO,
    }


def leer_csv_generico(ruta: Path) -> List[Dict[str, str]]:
    if not ruta.is_file():
        return []
    with ruta.open("r", newline="", encoding="utf-8") as archivo:
        return [dict(fila) for fila in csv.DictReader(archivo)]


def verificar_percepcion_comun(
    filas_percepcion: Sequence[Mapping[str, Any]],
    semillas: Sequence[int],
) -> Tuple[bool, List[Dict[str, Any]]]:
    salida: List[Dict[str, Any]] = []
    correcto_global = True
    for semilla in semillas:
        grupo = [f for f in filas_percepcion if int(f["semilla"]) == int(semilla)]
        huellas = {
            str(f.get("huella_percepcion_primeros_pasos", ""))
            for f in grupo
            if str(f.get("huella_percepcion_primeros_pasos", ""))
        }
        metodos = {str(f.get("metodo", "")) for f in grupo}
        pasos_huella = [int(float(f.get("numero_pasos_huella", 0))) for f in grupo]
        suficiente = len(pasos_huella) == len(nominal.METODOS) and min(pasos_huella, default=0) >= 1
        correcto = (
            len(grupo) == len(nominal.METODOS)
            and metodos == set(nominal.METODOS)
            and len(huellas) == 1
            and suficiente
        )
        correcto_global = correcto_global and correcto
        salida.append(
            {
                "semilla": int(semilla),
                "metodos_presentes": len(metodos),
                "numero_huellas_distintas": len(huellas),
                "pasos_huella_minimo": min(pasos_huella, default=0),
                "percepcion_comun_correcta": int(correcto),
            }
        )
    return correcto_global, salida


def construir_resumen_condicion(
    condicion: CondicionIncertidumbre,
    resumen: Sequence[Mapping[str, Any]],
    cochran: Mapping[str, Any],
    directorio: Path,
) -> str:
    lineas = [
        "FASE 2 — COMPARACIÓN PAREADA DE CINCO MÉTODOS",
        "=" * 80,
        f"Condición: {condicion.nombre}",
        f"Latencia: {condicion.latencia_pasos} pasos",
        f"Sigma posición: {condicion.sigma_posicion_m:.4f} m",
        f"Sigma velocidad: {condicion.sigma_velocidad_m_s:.4f} m/s",
        "Corrupción externa común para los cinco métodos: SÍ",
        "Mundo real y métricas de colisión sin corromper: SÍ",
        "Entrenamiento durante la evaluación: NO",
        "Actores congelados: SÍ",
        "",
        f"Q de Cochran: {cochran.get('estadistico_q', float('nan'))}",
        f"p global: {cochran.get('p_valor', float('nan'))}",
        "",
        "RESULTADOS DESCRIPTIVOS",
        "-" * 80,
    ]
    for fila in resumen:
        lineas.extend(
            [
                str(fila["metodo"]),
                f"  Éxito: {100.0 * float(fila['tasa_exito']):.2f}%",
                f"  Colisión dinámica: {100.0 * float(fila['tasa_colision_dinamica']):.2f}%",
                f"  Episodios con casi colisión: {100.0 * float(fila['tasa_episodios_con_casi_colision']):.2f}%",
                f"  SPL: {float(fila['spl_medio']):.4f}",
            ]
        )
    lineas.extend(["", f"Directorio: {directorio}", ""])
    return "\n".join(lineas)


def ejecutar_condicion(
    condicion: CondicionIncertidumbre,
    argumentos: argparse.Namespace,
    base,
    td3,
    upo,
    actores: Mapping[str, torch.nn.Module],
    info_actores: Mapping[str, Mapping[str, Any]],
    rutas_programas: Mapping[str, str],
) -> Path:
    final_semilla = argumentos.semilla_base + argumentos.numero_semillas - 1
    if argumentos.directorio_salida_base:
        directorio = Path(argumentos.directorio_salida_base) / condicion.nombre
    else:
        directorio = Path(
            f"resultados_comparacion_5_metodos_{condicion.nombre}_"
            f"{argumentos.semilla_base}_{final_semilla}"
        )
    if argumentos.solo_verificacion:
        directorio = Path(f"resultados_verificacion_fase2_{condicion.nombre}")

    if argumentos.reiniciar_resultados and directorio.exists():
        shutil.rmtree(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_resultados = directorio / "resultados_por_semilla_y_metodo.csv"
    ruta_percepcion = directorio / "huellas_percepcion_comun.csv"
    ruta_configuracion = directorio / "configuracion_experimento.json"

    configuracion = {
        "fase": 2,
        "condicion": condicion.nombre,
        "semilla_base": int(argumentos.semilla_base),
        "numero_semillas": int(argumentos.numero_semillas),
        "metodos": list(nominal.METODOS),
        "dt": float(base.DT),
        "pasos_maximos": int(base.PASOS_MAXIMOS_SEGUIMIENTO),
        "percepcion_externa_comun": {
            "latencia_pasos": int(condicion.latencia_pasos),
            "sigma_posicion_m": float(condicion.sigma_posicion_m),
            "sigma_velocidad_m_s": float(condicion.sigma_velocidad_m_s),
            "version_generador_ruido": VERSION_GENERADOR_RUIDO,
            "aplicada_a": "obstaculos_dinamicos_x_y_vx_vy",
            "mapa_estatico": "sin_corrupcion",
            "estado_robot": "sin_corrupcion",
        },
        "upo_modelo_monte_carlo": {
            "numero_muestras": int(upo.NUMERO_MUESTRAS_MONTE_CARLO),
            "sigma_posicion_modelo_m": float(upo.SIGMA_POSICION_MODELO_M),
            "sigma_velocidad_relativa": float(upo.SIGMA_VELOCIDAD_RELATIVA),
            "sigma_direccion_rad": float(upo.SIGMA_DIRECCION_RAD),
            "sigma_aceleracion_m_s2": float(upo.SIGMA_ACELERACION_M_S2),
            "sigma_giro_rad_s": float(upo.SIGMA_GIRO_RAD_S),
        },
        "programas": dict(rutas_programas),
        "checkpoints": {
            metodo: {k: v for k, v in info.items() if k != "estado_inicial"}
            for metodo, info in info_actores.items()
        },
    }

    if ruta_configuracion.is_file():
        anterior = json.loads(ruta_configuracion.read_text(encoding="utf-8"))
        if anterior != configuracion:
            raise RuntimeError(
                f"La carpeta {directorio} contiene una configuración diferente. "
                "Use otra carpeta o reinicie solo si la corrida era técnicamente inválida."
            )
    else:
        ruta_configuracion.write_text(
            json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    filas_existentes = nominal.convertir_tipos_filas(
        nominal.leer_csv_resultados(ruta_resultados)
    )
    claves_completadas = {
        (int(f["semilla"]), str(f["metodo"])) for f in filas_existentes
    }
    percepcion_existente = leer_csv_generico(ruta_percepcion)
    claves_percepcion = {
        (int(float(f["semilla"])), str(f["metodo"])) for f in percepcion_existente
    }

    total = argumentos.numero_semillas * len(nominal.METODOS)
    print("\n" + "=" * 92)
    print(f"FASE 2 — CONDICIÓN {condicion.etiqueta.upper()}")
    print("=" * 92)
    print(
        f"Latencia={condicion.latencia_pasos} pasos | "
        f"sigma_pos={condicion.sigma_posicion_m:.3f} m | "
        f"sigma_vel={condicion.sigma_velocidad_m_s:.3f} m/s"
    )
    print(f"Episodios ya guardados: {len(claves_completadas)}/{total}")
    print("No se entrena ni se modifica ningún actor.")

    contador_nuevo = 0
    for indice in range(argumentos.numero_semillas):
        semilla = argumentos.semilla_base + indice
        resultados_semilla: Dict[str, Dict[str, Any]] = {}

        if (semilla, "A* + DWA") not in claves_completadas:
            resultados_semilla["A* + DWA"] = ejecutar_dwa_fase2(
                base, semilla, condicion
            )
        if (semilla, "A* + TD3") not in claves_completadas:
            resultados_semilla["A* + TD3"] = ejecutar_reactivo_fase2(
                base,
                actores["A* + TD3"],
                semilla,
                condicion,
                "A* + TD3",
                td3=td3,
            )
        if (semilla, "A* + SAC-R") not in claves_completadas:
            resultados_semilla["A* + SAC-R"] = ejecutar_reactivo_fase2(
                base,
                actores["A* + SAC-R"],
                semilla,
                condicion,
                "A* + SAC-R",
            )
        if (semilla, "A* + SAC-PO-TTC") not in claves_completadas:
            resultados_semilla["A* + SAC-PO-TTC"] = ejecutar_po_ttc_fase2(
                base,
                actores["A* + SAC-PO-TTC"],
                semilla,
                condicion,
            )
        if (semilla, "A* + SAC-UPO-TTC") not in claves_completadas:
            resultados_semilla["A* + SAC-UPO-TTC"] = ejecutar_upo_ttc_fase2(
                upo,
                actores["A* + SAC-UPO-TTC"],
                semilla,
                condicion,
            )

        for metodo in nominal.METODOS:
            if metodo not in resultados_semilla:
                continue
            resultado = resultados_semilla[metodo]
            numero_parametros = (
                0 if metodo == "A* + DWA" else int(info_actores[metodo]["numero_parametros"])
            )
            fila = nominal.convertir_resultado_a_fila(
                resultado=resultado,
                metodo=metodo,
                numero_parametros=numero_parametros,
                base=upo.base if metodo == "A* + SAC-UPO-TTC" else base,
                upo=upo if metodo == "A* + SAC-UPO-TTC" else None,
            )
            fila["condicion"] = condicion.nombre
            nominal.anexar_fila_csv(
                ruta_resultados, fila, nominal.CAMPOS_RESULTADOS
            )
            filas_existentes.append(fila)
            claves_completadas.add((semilla, metodo))

            fp = fila_percepcion(resultado, condicion, metodo)
            if (semilla, metodo) not in claves_percepcion:
                nominal.anexar_fila_csv(ruta_percepcion, fp, CAMPOS_PERCEPCION)
                percepcion_existente.append(fp)
                claves_percepcion.add((semilla, metodo))
            contador_nuevo += 1

        numero = indice + 1
        imprimir = (
            numero == 1
            or numero == argumentos.numero_semillas
            or (
                argumentos.frecuencia_impresion > 0
                and numero % argumentos.frecuencia_impresion == 0
            )
        )
        if imprimir:
            grupo = [f for f in filas_existentes if int(f["semilla"]) == semilla]
            texto = " | ".join(
                f"{m.replace('A* + ', '')}="
                f"{next((str(f['resultado']) for f in grupo if f['metodo'] == m), 'pendiente')}"
                for m in nominal.METODOS
            )
            print(
                f"Escenario {numero:3d}/{argumentos.numero_semillas:3d} "
                f"| semilla={semilla} | {texto}"
            )

    filas = nominal.convertir_tipos_filas(
        nominal.leer_csv_resultados(ruta_resultados)
    )
    esperado = argumentos.numero_semillas * len(nominal.METODOS)
    if len(filas) != esperado:
        raise RuntimeError(
            f"Ejecución incompleta: {len(filas)}/{esperado}. "
            "Ejecute nuevamente el mismo comando para reanudar."
        )

    huellas_correctas, verificacion_huellas = nominal.verificar_huellas_por_semilla(filas)
    nominal.guardar_csv(
        directorio / "verificacion_escenarios_identicos.csv", verificacion_huellas
    )
    if not huellas_correctas:
        raise RuntimeError("Los escenarios físicos no fueron idénticos entre métodos")

    percepcion_correcta, verificacion_percepcion = verificar_percepcion_comun(
        leer_csv_generico(ruta_percepcion),
        [argumentos.semilla_base + i for i in range(argumentos.numero_semillas)],
    )
    nominal.guardar_csv(
        directorio / "verificacion_percepcion_comun.csv", verificacion_percepcion
    )
    if not percepcion_correcta:
        raise RuntimeError(
            "La verificación detectó diferencias en la percepción común entre métodos"
        )

    verificacion_actores = nominal.verificar_actores_finales(actores, info_actores)
    nominal.guardar_csv(
        directorio / "verificacion_actores_congelados.csv", verificacion_actores
    )
    actores_congelados = all(
        int(f["verificacion_correcta"]) == 1 for f in verificacion_actores
    )
    if not actores_congelados:
        raise RuntimeError("Uno o más actores cambiaron durante la fase 2")

    resumen = nominal.resumen_por_metodo(filas)
    comparaciones_exito, cochran, desacuerdos = nominal.comparaciones_binarias(
        filas, argumentos.muestras_bootstrap
    )
    globales_continuas, pares_continuos = nominal.comparaciones_continuas(filas)

    nominal.guardar_csv(directorio / "resumen_por_metodo.csv", resumen)
    nominal.guardar_csv(
        directorio / "comparaciones_pareadas_exito.csv", comparaciones_exito
    )
    nominal.guardar_csv(directorio / "prueba_global_cochran.csv", [cochran])
    nominal.guardar_csv(
        directorio / "semillas_con_desacuerdo_de_exito.csv", desacuerdos
    )
    nominal.guardar_csv(
        directorio / "pruebas_globales_metricas_continuas.csv", globales_continuas
    )
    nominal.guardar_csv(
        directorio / "comparaciones_pareadas_metricas_continuas.csv",
        pares_continuos,
    )
    predictivas = [
        f
        for f in filas
        if f["metodo"] in ("A* + SAC-PO-TTC", "A* + SAC-UPO-TTC")
    ]
    nominal.guardar_csv(
        directorio / "metricas_predictivas_por_episodio.csv",
        predictivas,
        nominal.CAMPOS_RESULTADOS,
    )

    if not argumentos.sin_graficas:
        nominal.crear_graficas(directorio, resumen)

    texto = construir_resumen_condicion(condicion, resumen, cochran, directorio)
    (directorio / "resumen_comparacion.txt").write_text(texto, encoding="utf-8")

    print("\nRESULTADO TÉCNICO")
    for fila in resumen:
        print(
            f"{fila['metodo']:<20s} | éxito={100.0 * float(fila['tasa_exito']):6.2f}% "
            f"| col. din={100.0 * float(fila['tasa_colision_dinamica']):6.2f}% "
            f"| casi col.={100.0 * float(fila['tasa_episodios_con_casi_colision']):6.2f}% "
            f"| SPL={float(fila['spl_medio']):.3f}"
        )
    print("Escenarios idénticos:", huellas_correctas)
    print("Percepción común:", percepcion_correcta)
    print("Actores congelados:", actores_congelados)
    print("Resultados:", directorio)
    print("Episodios nuevos ejecutados:", contador_nuevo)
    return directorio


# =============================================================================
# INTEGRACIÓN NOMINAL + MODERADA + SEVERA
# =============================================================================


def leer_resultados_condicion(directorio: Path) -> Optional[List[Dict[str, Any]]]:
    ruta = directorio / "resultados_por_semilla_y_metodo.csv"
    if not ruta.is_file():
        return None
    return nominal.convertir_tipos_filas(nominal.leer_csv_resultados(ruta))


def grafica_robustez(
    directorio: Path,
    resumen_integrado: Sequence[Mapping[str, Any]],
) -> None:
    orden_condiciones = ["nominal_limpia", "incertidumbre_moderada", "incertidumbre_severa"]
    etiquetas = ["Nominal", "Moderate", "Severe"]

    figura, eje = plt.subplots(figsize=(11, 7))
    for metodo in nominal.METODOS:
        valores = []
        for condicion in orden_condiciones:
            fila = next(
                (f for f in resumen_integrado if f["metodo"] == metodo and f["condicion"] == condicion),
                None,
            )
            valores.append(100.0 * float(fila["tasa_exito"]) if fila else np.nan)
        eje.plot(etiquetas, valores, marker="o", linewidth=2, label=metodo)
    eje.set_ylabel("Success rate (%)")
    eje.set_xlabel("Perceptual uncertainty condition")
    eje.set_title("Navigation robustness under common perception uncertainty")
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(directorio / "robustness_success_rate_en.png", dpi=300, bbox_inches="tight")
    plt.close(figura)


def integrar_tres_condiciones(
    argumentos: argparse.Namespace,
    directorios_fase2: Mapping[str, Path],
) -> Optional[Path]:
    nominal_dir = Path(argumentos.directorio_nominal)
    moderada_dir = directorios_fase2.get("moderada")
    severa_dir = directorios_fase2.get("severa")

    if moderada_dir is None:
        if argumentos.directorio_salida_base:
            moderada_dir = Path(argumentos.directorio_salida_base) / "incertidumbre_moderada"
        else:
            final_semilla = argumentos.semilla_base + argumentos.numero_semillas - 1
            moderada_dir = Path(
                f"resultados_comparacion_5_metodos_incertidumbre_moderada_"
                f"{argumentos.semilla_base}_{final_semilla}"
            )
    if severa_dir is None:
        if argumentos.directorio_salida_base:
            severa_dir = Path(argumentos.directorio_salida_base) / "incertidumbre_severa"
        else:
            final_semilla = argumentos.semilla_base + argumentos.numero_semillas - 1
            severa_dir = Path(
                f"resultados_comparacion_5_metodos_incertidumbre_severa_"
                f"{argumentos.semilla_base}_{final_semilla}"
            )

    conjuntos = {
        "nominal_limpia": leer_resultados_condicion(nominal_dir),
        "incertidumbre_moderada": leer_resultados_condicion(moderada_dir),
        "incertidumbre_severa": leer_resultados_condicion(severa_dir),
    }
    if any(valor is None for valor in conjuntos.values()):
        return None

    esperado = argumentos.numero_semillas * len(nominal.METODOS)
    if any(len(valor or []) != esperado for valor in conjuntos.values()):
        return None

    directorio = Path(
        f"resultados_robustez_tres_condiciones_{argumentos.semilla_base}_"
        f"{argumentos.semilla_base + argumentos.numero_semillas - 1}"
    )
    directorio.mkdir(parents=True, exist_ok=True)

    todas: List[Dict[str, Any]] = []
    resumen_integrado: List[Dict[str, Any]] = []
    for nombre, filas in conjuntos.items():
        assert filas is not None
        for fila in filas:
            copia = dict(fila)
            copia["condicion"] = nombre
            todas.append(copia)
        resumen = nominal.resumen_por_metodo(filas)
        for fila in resumen:
            copia = dict(fila)
            copia["condicion"] = nombre
            resumen_integrado.append(copia)

    nominal.guardar_csv(
        directorio / "resultados_tres_condiciones.csv",
        todas,
        nominal.CAMPOS_RESULTADOS,
    )
    nominal.guardar_csv(
        directorio / "resumen_tres_condiciones.csv", resumen_integrado
    )

    degradacion: List[Dict[str, Any]] = []
    for metodo in nominal.METODOS:
        base_fila = next(
            f for f in resumen_integrado
            if f["metodo"] == metodo and f["condicion"] == "nominal_limpia"
        )
        for condicion in ("incertidumbre_moderada", "incertidumbre_severa"):
            actual = next(
                f for f in resumen_integrado
                if f["metodo"] == metodo and f["condicion"] == condicion
            )
            degradacion.append(
                {
                    "metodo": metodo,
                    "condicion": condicion,
                    "tasa_exito_nominal": float(base_fila["tasa_exito"]),
                    "tasa_exito_condicion": float(actual["tasa_exito"]),
                    "cambio_exito_puntos_porcentuales": 100.0 * (
                        float(actual["tasa_exito"]) - float(base_fila["tasa_exito"])
                    ),
                    "cambio_colision_dinamica_puntos_porcentuales": 100.0 * (
                        float(actual["tasa_colision_dinamica"])
                        - float(base_fila["tasa_colision_dinamica"])
                    ),
                    "cambio_casi_colision_puntos_porcentuales": 100.0 * (
                        float(actual["tasa_episodios_con_casi_colision"])
                        - float(base_fila["tasa_episodios_con_casi_colision"])
                    ),
                    "cambio_spl": float(actual["spl_medio"])
                    - float(base_fila["spl_medio"]),
                    "cambio_clearance_dinamico_m": float(
                        actual["clearance_dinamico_minimo_media_m"]
                    )
                    - float(base_fila["clearance_dinamico_minimo_media_m"]),
                }
            )
    nominal.guardar_csv(directorio / "degradacion_respecto_nominal.csv", degradacion)
    grafica_robustez(directorio, resumen_integrado)
    return directorio


# =============================================================================
# CLI Y FUNCIÓN PRINCIPAL
# =============================================================================


def analizar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 2 con ruido y latencia comunes para cinco métodos."
    )
    parser.add_argument(
        "--condicion",
        choices=("moderada", "severa", "ambas"),
        default="moderada",
    )
    parser.add_argument("--semilla-base", type=int, default=170000)
    parser.add_argument("--numero-semillas", type=int, default=200)
    parser.add_argument("--frecuencia-impresion", type=int, default=5)
    parser.add_argument("--muestras-bootstrap", type=int, default=5000)
    parser.add_argument("--directorio-salida-base", type=str, default="")
    parser.add_argument(
        "--directorio-nominal",
        type=str,
        default="resultados_comparacion_5_metodos_nominal_170000_170199",
    )
    parser.add_argument("--solo-verificacion", action="store_true")
    parser.add_argument("--solo-integrar", action="store_true")
    parser.add_argument("--sin-graficas", action="store_true")
    parser.add_argument(
        "--reiniciar-resultados",
        action="store_true",
        help=(
            "Elimina la carpeta de la condición antes de ejecutar. No utilizar "
            "después de observar resultados válidos."
        ),
    )
    return parser.parse_args()


def main() -> None:
    argumentos = analizar_argumentos()
    if argumentos.numero_semillas <= 0:
        raise ValueError("--numero-semillas debe ser positivo")
    if argumentos.frecuencia_impresion < 0:
        raise ValueError("--frecuencia-impresion no puede ser negativa")
    if argumentos.muestras_bootstrap <= 0:
        raise ValueError("--muestras-bootstrap debe ser positivo")

    if argumentos.solo_verificacion:
        argumentos.semilla_base = SEMILLA_VERIFICACION_FASE2
        argumentos.numero_semillas = 1
        argumentos.frecuencia_impresion = 1

    condiciones_clave = (
        ["moderada", "severa"]
        if argumentos.condicion == "ambas"
        else [argumentos.condicion]
    )

    if argumentos.solo_integrar:
        integrado = integrar_tres_condiciones(argumentos, {})
        if integrado is None:
            raise RuntimeError(
                "No se encontraron completas las tres condiciones para integrarlas."
            )
        print("Análisis integrado guardado en:", integrado)
        return

    base, td3, upo, rutas_programas = nominal.preparar_modulos()
    if argumentos.solo_verificacion:
        base.PASOS_MAXIMOS_SEGUIMIENTO = min(int(base.PASOS_MAXIMOS_SEGUIMIENTO), 3)
        upo.base.PASOS_MAXIMOS_SEGUIMIENTO = min(
            int(upo.base.PASOS_MAXIMOS_SEGUIMIENTO), 3
        )
        upo.NUMERO_MUESTRAS_MONTE_CARLO = min(
            int(upo.NUMERO_MUESTRAS_MONTE_CARLO), 2
        )
        if not torch.cuda.is_available():
            torch.set_num_threads(1)

    actores, info_actores = nominal.cargar_todos_los_actores(base, td3, upo)

    print("\nACTORES CARGADOS Y CONGELADOS")
    for metodo, info in info_actores.items():
        print(
            f"{metodo}: episodio={info['episodio']} | "
            f"parámetros={info['numero_parametros']} | checkpoint={info['ruta']}"
        )
    print("DWA: sin checkpoint")
    print("Dispositivo:", base.DISPOSITIVO_SAC)
    print("UPO: incertidumbre Monte Carlo interna activa")

    directorios: Dict[str, Path] = {}
    for clave in condiciones_clave:
        directorios[clave] = ejecutar_condicion(
            CONDICIONES[clave],
            argumentos,
            base,
            td3,
            upo,
            actores,
            info_actores,
            rutas_programas,
        )

    if not argumentos.solo_verificacion:
        integrado = integrar_tres_condiciones(argumentos, directorios)
        if integrado is not None:
            print("\nAnálisis nominal + moderado + severo guardado en:", integrado)
        else:
            print(
                "\nEl análisis de tres condiciones se generará cuando estén completas "
                "la fase nominal, la moderada y la severa."
            )
        print(
            "ADVERTENCIA: estas semillas quedan observadas bajo la condición ejecutada; "
            "no deben utilizarse para ajustar actores o hiperparámetros."
        )
    else:
        print("Verificación técnica completada con una semilla ajena al conjunto final.")


if __name__ == "__main__":
    main()
