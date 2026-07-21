from __future__ import annotations

"""
Comparación confirmatoria pareada de cinco métodos de navegación:

    1. A* + DWA
    2. A* + TD3 reactivo
    3. A* + SAC reactivo mejorado
    4. A* + SAC-PO-TTC determinista
    5. A* + SAC-UPO-TTC probabilístico

Características metodológicas:
    - No entrena ni modifica ningún actor.
    - Carga los checkpoints congelados.
    - Ejecuta todos los métodos sobre las mismas semillas.
    - Usa una condición NOMINAL LIMPIA para todos los métodos.
    - En UPO-TTC se anulan ruido y latencia sensoriales externos, pero se
      conserva la incertidumbre Monte Carlo interna del método.
    - Guarda métricas comunes de éxito, eficiencia, seguridad, suavidad y
      costo computacional.
    - Guarda métricas predictivas adicionales para PO-TTC y UPO-TTC.
    - Calcula comparaciones pareadas e intervalos de confianza.
    - Guarda el progreso después de cada episodio y permite reanudar.

Archivos requeridos en la misma carpeta:
    SAC_predictivo_entrenamiento_completo.py
    TD3_tercera_corrida_desde_cero.py
    SAC_probabilistico_ocupacion_TTC_entrenamiento.py

Checkpoints requeridos:
    resultados_td3/tercera_corrida_td3_desde_cero/
        checkpoint_mejor_actor_td3.pt
    resultados_sac/tercera_corrida_reactivo_mejorado/
        checkpoint_mejor_actor_reactivo_mejorado.pt
    resultados_sac/entrenamiento_sac_predictivo/
        checkpoint_mejor_actor_sac_predictivo.pt
    resultados_sac/entrenamiento_sac_probabilistico_upo_ttc/
        checkpoint_mejor_actor_sac_probabilistico.pt

Prueba técnica, sin consumir las semillas finales:
    python Comparador_5_metodos_nominal_metricas_ampliadas.py --solo-verificacion

Comparación confirmatoria completa:
    python Comparador_5_metodos_nominal_metricas_ampliadas.py \
        --semilla-base 170000 --numero-semillas 200
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Evita que TkAgg bloquee el programa al finalizar.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

DIRECTORIO_PROGRAMA = Path(__file__).resolve().parent

CANDIDATOS_BASE = (
    "SAC_predictivo_entrenamiento_completo.py",
    "SAC_predictivo_entrenamiento_completo(3).py",
    "Comparacion_final_DWA_SAC_reactivo_SAC_predictivo.py",
)

CANDIDATOS_TD3 = (
    "TD3_tercera_corrida_desde_cero.py",
    "TD3_tercera_corrida_desde_cero(1).py",
    "TD3_entrenamiento_desde_cero.py",
)

CANDIDATOS_UPO = (
    "SAC_probabilistico_ocupacion_TTC_entrenamiento.py",
    "SAC_probabilistico_ocupacion_TTC_entrenamiento(1).py",
    "SAC_probabilistico_ocupacion_TTC_entrenamiento(2).py",
    "SAC_probabilistico_ocupacion_TTC_entrenamiento(3).py",
)

RUTA_CHECKPOINT_TD3 = Path(
    "resultados_td3/tercera_corrida_td3_desde_cero/"
    "checkpoint_mejor_actor_td3.pt"
)
RUTA_CHECKPOINT_SAC_REACTIVO = Path(
    "resultados_sac/tercera_corrida_reactivo_mejorado/"
    "checkpoint_mejor_actor_reactivo_mejorado.pt"
)
RUTA_CHECKPOINT_SAC_PREDICTIVO = Path(
    "resultados_sac/entrenamiento_sac_predictivo/"
    "checkpoint_mejor_actor_sac_predictivo.pt"
)
RUTA_CHECKPOINT_SAC_UPO = Path(
    "resultados_sac/entrenamiento_sac_probabilistico_upo_ttc/"
    "checkpoint_mejor_actor_sac_probabilistico.pt"
)

SEMILLA_BASE_PREDETERMINADA = 170000
NUMERO_SEMILLAS_PREDETERMINADO = 200
SEMILLA_VERIFICACION = 169900
FRECUENCIA_IMPRESION_PREDETERMINADA = 5
MUESTRAS_BOOTSTRAP_PREDETERMINADAS = 5000

UMBRAL_CASI_COLISION_M = 0.20
UMBRAL_TTC_RIESGOSO_S = 1.00
UMBRAL_PARADA_M_S = 0.05
UMBRAL_OMEGA_SIGNO_RAD_S = 0.02
HORIZONTE_CALIBRACION_S = 2.0

METODOS = (
    "A* + DWA",
    "A* + TD3",
    "A* + SAC-R",
    "A* + SAC-PO-TTC",
    "A* + SAC-UPO-TTC",
)

METRICAS_CONTINUAS_PAREADAS = (
    "spl",
    "tiempo_simulado_s",
    "longitud_trayectoria_m",
    "distancia_final_meta_m",
    "clearance_dinamico_minimo_m",
    "clearance_total_percentil_5_m",
    "eventos_casi_colision_real",
    "tiempo_en_casi_colision_s",
    "ttc_real_minimo_s",
    "aceleracion_angular_rms_rad_s2",
    "jerk_angular_rms_rad_s3",
    "esfuerzo_control_medio",
    "tiempo_ciclo_medio_ms",
)

# Orden fijo y común del CSV. Mantenerlo estable permite reanudar la ejecución.
CAMPOS_RESULTADOS = [
    "condicion",
    "semilla",
    "metodo",
    "resultado",
    "exito",
    "colision_estatica",
    "colision_dinamica",
    "fuera_mapa",
    "timeout",
    "pasos",
    "tiempo_simulado_s",
    "recompensa_acumulada",
    "distancia_inicial_meta_m",
    "distancia_final_meta_m",
    "progreso_normalizado",
    "longitud_astar_m",
    "longitud_trayectoria_m",
    "exceso_longitud_m",
    "eficiencia_geometrica",
    "spl",
    "error_medio_ruta_m",
    "error_rmse_ruta_m",
    "error_maximo_ruta_m",
    "clearance_estatico_minimo_m",
    "clearance_dinamico_minimo_m",
    "clearance_total_minimo_m",
    "clearance_total_percentil_5_m",
    "clearance_total_promedio_m",
    "hubo_casi_colision_real",
    "eventos_casi_colision_real",
    "pasos_en_casi_colision",
    "tiempo_en_casi_colision_s",
    "fraccion_tiempo_casi_colision",
    "pasos_en_riesgo_total",
    "tiempo_en_riesgo_total_s",
    "ttc_real_minimo_s",
    "pasos_ttc_menor_1s",
    "tiempo_ttc_menor_1s_s",
    "eventos_ttc_menor_1s",
    "velocidad_lineal_media_m_s",
    "velocidad_angular_abs_media_rad_s",
    "numero_paradas",
    "variacion_total_v",
    "variacion_total_omega",
    "aceleracion_lineal_rms_m_s2",
    "aceleracion_angular_rms_rad_s2",
    "jerk_lineal_rms_m_s3",
    "jerk_angular_rms_rad_s3",
    "cambios_signo_omega",
    "esfuerzo_control_medio",
    "tiempo_preparacion_ms",
    "tiempo_control_total_ms",
    "tiempo_ciclo_medio_ms",
    "inferencia_actor_media_ms",
    "inferencia_actor_p95_ms",
    "inferencia_actor_p99_ms",
    "procesamiento_paso_medio_ms",
    "numero_parametros",
    "huella_escenario",
    "huella_dinamicos_iniciales",
    # Métricas predictivas; NaN en los métodos que no predicen.
    "clearance_predicho_minimo_m",
    "clearance_predicho_promedio_m",
    "clearance_futuro_real_minimo_m",
    "error_mae_clearance_predicho_m",
    "ttc_predicho_minimo_s",
    "ttc_predicho_promedio_s",
    "error_mae_ttc_predicho_condicional_s",
    "fraccion_colision_predicha",
    "brier_colision_predicha",
    "precision_colision_predicha",
    "recall_colision_predicha",
    "f1_colision_predicha",
    "probabilidad_colision_media",
    "probabilidad_colision_maxima",
    "brier_probabilidad_colision",
    "ece_probabilidad_colision",
    "probabilidad_casi_colision_media",
    "probabilidad_casi_colision_maxima",
    "brier_probabilidad_casi_colision",
    "ece_probabilidad_casi_colision",
    "clearance_cvar_minimo_m",
    "clearance_cvar_promedio_m",
    "ttc_cvar_minimo_s",
    "ttc_cvar_promedio_s",
    "penalizacion_predictiva_media",
    "velocidad_cierre_predicha_maxima_m_s",
    "latencia_observacion_pasos",
    "sigma_posicion_observacion_m",
    "sigma_velocidad_observacion_m_s",
    "numero_muestras_monte_carlo",
]


# =============================================================================
# UTILIDADES DE IMPORTACIÓN
# =============================================================================


def localizar_archivo(candidatos: Sequence[str], descripcion: str) -> Path:
    for nombre in candidatos:
        ruta = DIRECTORIO_PROGRAMA / nombre
        if ruta.is_file():
            return ruta
    lista = "\n".join(f"  - {nombre}" for nombre in candidatos)
    raise FileNotFoundError(
        f"No se encontró {descripcion}. Coloque este comparador junto a:\n{lista}"
    )


def importar_modulo_desde_ruta(nombre_modulo: str, ruta: Path):
    # Los programas base fuerzan TkAgg. Durante la importación se sustituye
    # temporalmente por Agg para evitar ventanas y bloqueos.
    usar_original = matplotlib.use

    def usar_compatible(backend, *args, **kwargs):
        if str(backend).strip().lower() == "tkagg":
            return usar_original("Agg", force=True)
        return usar_original(backend, *args, **kwargs)

    matplotlib.use = usar_compatible
    try:
        especificacion = importlib.util.spec_from_file_location(nombre_modulo, ruta)
        if especificacion is None or especificacion.loader is None:
            raise ImportError(f"No fue posible preparar la importación de {ruta}")
        modulo = importlib.util.module_from_spec(especificacion)
        sys.modules[nombre_modulo] = modulo
        especificacion.loader.exec_module(modulo)
        return modulo
    finally:
        matplotlib.use = usar_original


def resolver_ruta_proyecto(ruta: Path) -> Path:
    ruta = Path(ruta)
    if ruta.is_absolute():
        return ruta
    desde_cwd = Path.cwd() / ruta
    if desde_cwd.is_file():
        return desde_cwd
    desde_programa = DIRECTORIO_PROGRAMA / ruta
    return desde_programa


def preparar_modulos():
    ruta_base = localizar_archivo(CANDIDATOS_BASE, "el programa base")
    ruta_td3 = localizar_archivo(CANDIDATOS_TD3, "el programa TD3")
    ruta_upo = localizar_archivo(CANDIDATOS_UPO, "el programa SAC-UPO-TTC")

    base = importar_modulo_desde_ruta("base_comparador_cinco_metodos", ruta_base)
    td3 = importar_modulo_desde_ruta("td3_comparador_cinco_metodos", ruta_td3)
    # Obliga a TD3 a usar exactamente el mismo entorno base del comparador.
    td3.base = base

    upo = importar_modulo_desde_ruta("upo_comparador_cinco_metodos", ruta_upo)

    # Condición nominal limpia. Se conserva la incertidumbre del MODELO MC,
    # pero se elimina la corrupción SENSORIAL externa para que todos reciban
    # observaciones actuales y sin ruido.
    upo.LATENCIA_MAXIMA_PASOS = 0
    upo.SIGMA_POSICION_MIN_M = 0.0
    upo.SIGMA_POSICION_MAX_M = 0.0
    upo.SIGMA_VELOCIDAD_MIN_M_S = 0.0
    upo.SIGMA_VELOCIDAD_MAX_M_S = 0.0
    upo.configurar_programa_base(numero_episodios=1, mostrar_graficas=False)

    return base, td3, upo, {
        "programa_base": str(ruta_base),
        "programa_td3": str(ruta_td3),
        "programa_upo": str(ruta_upo),
    }


# =============================================================================
# CHECKPOINTS Y ACTORES CONGELADOS
# =============================================================================


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        while True:
            bloque = archivo.read(1024 * 1024)
            if not bloque:
                break
            digest.update(bloque)
    return digest.hexdigest()


def clonar_estado_actor(actor: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        nombre: tensor.detach().cpu().clone()
        for nombre, tensor in actor.state_dict().items()
    }


def actor_sin_cambios(actor: torch.nn.Module, estado_inicial: Mapping[str, torch.Tensor]) -> bool:
    estado_final = actor.state_dict()
    return all(
        nombre in estado_final
        and torch.equal(estado_inicial[nombre], estado_final[nombre].detach().cpu())
        for nombre in estado_inicial
    )


def cargar_actor_sac(base, ruta_relativa: Path, nombre: str):
    ruta = resolver_ruta_proyecto(ruta_relativa)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint de {nombre}:\n{ruta}")
    checkpoint = torch.load(ruta, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise KeyError(f"El checkpoint de {nombre} no contiene actor_state_dict")

    actor = base.crear_actor_sac(dispositivo=base.DISPOSITIVO_SAC)
    resultado = actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    if resultado.missing_keys or resultado.unexpected_keys:
        raise RuntimeError(f"La carga estricta de {nombre} falló")
    actor.eval()
    for parametro in actor.parameters():
        parametro.requires_grad_(False)

    info = {
        "nombre": nombre,
        "ruta": str(ruta),
        "sha256": sha256_archivo(ruta),
        "episodio": int(checkpoint.get("episodio", checkpoint.get("mejor_episodio", -1))),
        "tipo": str(checkpoint.get("tipo", "")),
        "variante": str(checkpoint.get("variante_sac", checkpoint.get("variante", ""))),
        "numero_parametros": int(sum(p.numel() for p in actor.parameters())),
        "estado_inicial": clonar_estado_actor(actor),
    }
    return actor, info


def cargar_actor_td3(td3, base, ruta_relativa: Path):
    ruta = resolver_ruta_proyecto(ruta_relativa)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint TD3:\n{ruta}")
    checkpoint = torch.load(ruta, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise KeyError("El checkpoint TD3 no contiene actor_state_dict")

    actor = td3.crear_actor_td3(base.DISPOSITIVO_SAC)
    resultado = actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    if resultado.missing_keys or resultado.unexpected_keys:
        raise RuntimeError("La carga estricta del TD3 falló")
    actor.eval()
    for parametro in actor.parameters():
        parametro.requires_grad_(False)

    info = {
        "nombre": "A* + TD3",
        "ruta": str(ruta),
        "sha256": sha256_archivo(ruta),
        "episodio": int(checkpoint.get("episodio", -1)),
        "tipo": str(checkpoint.get("tipo", "")),
        "variante": str(checkpoint.get("variante", "")),
        "numero_parametros": int(sum(p.numel() for p in actor.parameters())),
        "estado_inicial": clonar_estado_actor(actor),
    }
    return actor, info


def cargar_todos_los_actores(base, td3, upo):
    actor_td3, info_td3 = cargar_actor_td3(td3, base, RUTA_CHECKPOINT_TD3)
    actor_reactivo, info_reactivo = cargar_actor_sac(
        base, RUTA_CHECKPOINT_SAC_REACTIVO, "A* + SAC-R"
    )
    actor_predictivo, info_predictivo = cargar_actor_sac(
        base, RUTA_CHECKPOINT_SAC_PREDICTIVO, "A* + SAC-PO-TTC"
    )
    actor_upo, info_upo = cargar_actor_sac(
        upo.base, RUTA_CHECKPOINT_SAC_UPO, "A* + SAC-UPO-TTC"
    )

    actores = {
        "A* + TD3": actor_td3,
        "A* + SAC-R": actor_reactivo,
        "A* + SAC-PO-TTC": actor_predictivo,
        "A* + SAC-UPO-TTC": actor_upo,
    }
    informacion = {
        "A* + TD3": info_td3,
        "A* + SAC-R": info_reactivo,
        "A* + SAC-PO-TTC": info_predictivo,
        "A* + SAC-UPO-TTC": info_upo,
    }
    return actores, informacion


# =============================================================================
# UTILIDADES NUMÉRICAS
# =============================================================================


def valor_finito_o_nan(valor: Any) -> float:
    try:
        numero = float(valor)
    except Exception:
        return float("nan")
    return numero if np.isfinite(numero) else float("nan")


def arreglo_float(valores: Iterable[Any]) -> np.ndarray:
    try:
        return np.asarray(list(valores), dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def valores_finitos(valores: Iterable[Any]) -> np.ndarray:
    arreglo = arreglo_float(valores)
    return arreglo[np.isfinite(arreglo)]


def media_finita(valores: Iterable[Any]) -> float:
    arreglo = valores_finitos(valores)
    return float(np.mean(arreglo)) if arreglo.size else float("nan")


def minimo_finito(valores: Iterable[Any]) -> float:
    arreglo = valores_finitos(valores)
    return float(np.min(arreglo)) if arreglo.size else float("nan")


def maximo_finito(valores: Iterable[Any]) -> float:
    arreglo = valores_finitos(valores)
    return float(np.max(arreglo)) if arreglo.size else float("nan")


def percentil_finito(valores: Iterable[Any], percentil: float) -> float:
    arreglo = valores_finitos(valores)
    return float(np.percentile(arreglo, percentil)) if arreglo.size else float("nan")


def rms(valores: Iterable[Any]) -> float:
    arreglo = valores_finitos(valores)
    return float(np.sqrt(np.mean(arreglo ** 2))) if arreglo.size else 0.0


def contar_eventos_mascara(mascara: Sequence[bool]) -> int:
    anterior = False
    eventos = 0
    for actual in mascara:
        actual_bool = bool(actual)
        if actual_bool and not anterior:
            eventos += 1
        anterior = actual_bool
    return eventos


def huella_json(objeto: Any) -> str:
    def convertir(valor: Any):
        if isinstance(valor, Mapping):
            return {str(k): convertir(v) for k, v in sorted(valor.items(), key=lambda x: str(x[0]))}
        if isinstance(valor, (list, tuple)):
            return [convertir(v) for v in valor]
        if isinstance(valor, np.ndarray):
            return convertir(valor.tolist())
        if isinstance(valor, (np.floating, float)):
            return round(float(valor), 10)
        if isinstance(valor, (np.integer, int)):
            return int(valor)
        if isinstance(valor, (np.bool_, bool)):
            return bool(valor)
        if valor is None or isinstance(valor, str):
            return valor
        return str(valor)

    contenido = json.dumps(convertir(objeto), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def calcular_huellas(resultado: Mapping[str, Any]) -> Tuple[str, str]:
    escenario = resultado["escenario"]
    escenario_reducido = {
        "inicio": escenario["inicio"],
        "meta": escenario["meta"],
        "obstaculos": escenario["obstaculos"],
        "camino_mundo": escenario["camino_mundo"],
        "longitud_camino": escenario["longitud_camino"],
    }
    dinamicos = resultado["obstaculos_dinamicos_iniciales"]
    return huella_json(escenario_reducido), huella_json(dinamicos)


def intervalo_wilson(exitos: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = exitos / total
    denominador = 1.0 + z * z / total
    centro = (p + z * z / (2.0 * total)) / denominador
    margen = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominador
    )
    return max(0.0, centro - margen), min(1.0, centro + margen)


# =============================================================================
# EJECUCIÓN UNIFORME DE LOS CINCO MÉTODOS
# =============================================================================


def crear_resultado_desde_entorno(
    base,
    entorno: Mapping[str, Any],
    recompensa_acumulada: float,
    tiempos_inferencia: Sequence[float],
    tiempos_paso: Sequence[float],
    tiempo_preparacion: float,
    metodo: str,
) -> Dict[str, Any]:
    registro = entorno["registro"]
    escenario = entorno["escenario"]
    metricas = base.calcular_metricas_episodio_estandar(
        registro=registro,
        camino_mundo=entorno["camino_mundo"],
        meta=entorno["meta"],
        longitud_camino_astar=escenario["longitud_camino"],
        dt=base.DT,
    )
    return {
        "metodo": metodo,
        "semilla": int(escenario["semilla"]),
        "entorno": entorno,
        "escenario": escenario,
        "camino_mundo": entorno["camino_mundo"],
        "inicio": escenario["inicio"],
        "meta": entorno["meta"],
        "obstaculos_estaticos": entorno["obstaculos_estaticos"],
        "obstaculos_dinamicos_iniciales": entorno["obstaculos_dinamicos_iniciales"],
        "registro": registro,
        "resultado": str(entorno["resultado"]),
        "recompensa_acumulada": float(recompensa_acumulada),
        "numero_pasos": int(registro["pasos_ejecutados"]),
        "metricas": metricas,
        "tiempos_inferencia_s": list(tiempos_inferencia),
        "tiempos_paso_s": list(tiempos_paso),
        "tiempo_preparacion_s": float(tiempo_preparacion),
    }


def ejecutar_sac_manual(
    base,
    actor: torch.nn.Module,
    semilla: int,
    predictivo: bool,
    metodo: str,
) -> Dict[str, Any]:
    inicio_preparacion = time.perf_counter()
    if predictivo:
        entorno, observacion = base.reiniciar_entorno_sac_predictivo(int(semilla))
    else:
        entorno, observacion = base.reiniciar_entorno_sac(int(semilla))
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    terminado = False
    truncado = False
    recompensa_acumulada = 0.0
    tiempos_inferencia: List[float] = []
    tiempos_paso: List[float] = []

    actor.eval()
    while not (terminado or truncado):
        inicio_inferencia = time.perf_counter()
        accion, informacion_accion = base.seleccionar_accion_actor_sac(
            observacion=observacion,
            actor=actor,
            dispositivo=base.DISPOSITIVO_SAC,
            determinista=True,
        )
        tiempos_inferencia.append(time.perf_counter() - inicio_inferencia)

        if informacion_accion.get("modo") != "determinista":
            raise RuntimeError(f"{metodo} no produjo una acción determinista")

        inicio_paso = time.perf_counter()
        if predictivo:
            observacion, recompensa, terminado, truncado, _ = (
                base.ejecutar_paso_entorno_sac_predictivo(
                    entorno=entorno,
                    accion=accion,
                    pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                    dt=base.DT,
                )
            )
        else:
            observacion, recompensa, terminado, truncado, _ = base.ejecutar_paso_entorno_sac(
                entorno=entorno,
                accion=accion,
                pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                dt=base.DT,
            )
        tiempos_paso.append(time.perf_counter() - inicio_paso)
        recompensa_acumulada += float(recompensa)

    return crear_resultado_desde_entorno(
        base=base,
        entorno=entorno,
        recompensa_acumulada=recompensa_acumulada,
        tiempos_inferencia=tiempos_inferencia,
        tiempos_paso=tiempos_paso,
        tiempo_preparacion=tiempo_preparacion,
        metodo=metodo,
    )


def ejecutar_td3_manual(base, td3, actor: torch.nn.Module, semilla: int) -> Dict[str, Any]:
    inicio_preparacion = time.perf_counter()
    entorno, observacion = base.reiniciar_entorno_sac(int(semilla))
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    terminado = False
    truncado = False
    recompensa_acumulada = 0.0
    tiempos_inferencia: List[float] = []
    tiempos_paso: List[float] = []

    actor.eval()
    while not (terminado or truncado):
        inicio_inferencia = time.perf_counter()
        accion = td3.accion_td3(
            actor=actor,
            observacion=observacion,
            dispositivo=base.DISPOSITIVO_SAC,
            ruido=0.0,
        )
        tiempos_inferencia.append(time.perf_counter() - inicio_inferencia)

        inicio_paso = time.perf_counter()
        observacion, recompensa, terminado, truncado, _ = base.ejecutar_paso_entorno_sac(
            entorno=entorno,
            accion=accion,
            pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
            dt=base.DT,
        )
        tiempos_paso.append(time.perf_counter() - inicio_paso)
        recompensa_acumulada += float(recompensa)

    return crear_resultado_desde_entorno(
        base=base,
        entorno=entorno,
        recompensa_acumulada=recompensa_acumulada,
        tiempos_inferencia=tiempos_inferencia,
        tiempos_paso=tiempos_paso,
        tiempo_preparacion=tiempo_preparacion,
        metodo="A* + TD3",
    )


def ejecutar_dwa_instrumentado(base, semilla: int) -> Dict[str, Any]:
    inicio_preparacion = time.perf_counter()
    escenario = base.generar_escenario_valido(int(semilla))
    if escenario is None:
        raise RuntimeError(f"No se generó un escenario válido para la semilla {semilla}")
    estado_inicial = base.crear_estado_inicial_escenario(escenario)
    semilla_dinamica = int(semilla) + int(base.DESFASE_SEMILLA_DINAMICA)
    obstaculos_dinamicos = base.generar_obstaculos_dinamicos(
        camino_mundo=escenario["camino_mundo"],
        obstaculos_estaticos=escenario["obstaculos"],
        inicio=escenario["inicio"],
        meta=escenario["meta"],
        numero_obstaculos=base.NUMERO_OBSTACULOS_DINAMICOS,
        semilla=semilla_dinamica,
    )
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    tiempos_controlador: List[float] = []
    controlador_original = base.controlador_dwa

    def controlador_cronometrado(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = controlador_original(*args, **kwargs)
        tiempos_controlador.append(time.perf_counter() - inicio)
        return resultado

    base.controlador_dwa = controlador_cronometrado
    inicio_simulacion = time.perf_counter()
    try:
        simulacion = base.simular_seguimiento_dinamico_dwa(
            estado_inicial=estado_inicial,
            camino_mundo=escenario["camino_mundo"],
            meta=escenario["meta"],
            obstaculos_estaticos=escenario["obstaculos"],
            obstaculos_dinamicos_iniciales=obstaculos_dinamicos,
            pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
            dt=base.DT,
        )
    finally:
        base.controlador_dwa = controlador_original
    tiempo_simulacion = time.perf_counter() - inicio_simulacion

    registro = base.convertir_simulacion_dwa_a_registro(
        simulacion_dwa=simulacion,
        obstaculos_estaticos=escenario["obstaculos"],
    )
    metricas = base.calcular_metricas_episodio_estandar(
        registro=registro,
        camino_mundo=escenario["camino_mundo"],
        meta=escenario["meta"],
        longitud_camino_astar=escenario["longitud_camino"],
        dt=base.DT,
    )

    pasos = max(1, int(registro["pasos_ejecutados"]))
    # El tiempo no atribuido al controlador incluye actualización dinámica,
    # colisiones, observación y registro. Se reparte por paso como tiempo de paso.
    tiempo_controladores = float(sum(tiempos_controlador))
    tiempo_resto = max(0.0, tiempo_simulacion - tiempo_controladores)
    tiempos_paso = [tiempo_resto / pasos] * pasos

    return {
        "metodo": "A* + DWA",
        "semilla": int(semilla),
        "escenario": escenario,
        "camino_mundo": escenario["camino_mundo"],
        "inicio": escenario["inicio"],
        "meta": escenario["meta"],
        "obstaculos_estaticos": escenario["obstaculos"],
        "obstaculos_dinamicos_iniciales": [dict(o) for o in obstaculos_dinamicos],
        "registro": registro,
        "resultado": str(metricas["resultado"]),
        "recompensa_acumulada": float("nan"),
        "numero_pasos": int(registro["pasos_ejecutados"]),
        "metricas": metricas,
        "tiempos_inferencia_s": tiempos_controlador,
        "tiempos_paso_s": tiempos_paso,
        "tiempo_preparacion_s": tiempo_preparacion,
    }


# =============================================================================
# MÉTRICAS COMUNES Y PREDICTIVAS
# =============================================================================

def ttc_circulos_movimiento_constante(
    estado_robot: Sequence[float],
    velocidad_lineal: float,
    obstaculo: Mapping[str, Any],
    radio_robot: float,
) -> float:
    x_r, y_r, theta = float(estado_robot[0]), float(estado_robot[1]), float(estado_robot[2])
    v_rx = float(velocidad_lineal) * math.cos(theta)
    v_ry = float(velocidad_lineal) * math.sin(theta)

    px = float(obstaculo["x"]) - x_r
    py = float(obstaculo["y"]) - y_r
    vx = float(obstaculo["vx"]) - v_rx
    vy = float(obstaculo["vy"]) - v_ry
    radio_total = float(radio_robot) + float(obstaculo["radio"])

    a = vx * vx + vy * vy
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - radio_total * radio_total

    if c <= 0.0:
        return 0.0
    if a <= 1e-12:
        return float("inf")

    discriminante = b * b - 4.0 * a * c
    if discriminante < 0.0:
        return float("inf")
    raiz = math.sqrt(max(0.0, discriminante))
    t1 = (-b - raiz) / (2.0 * a)
    t2 = (-b + raiz) / (2.0 * a)
    candidatos = [t for t in (t1, t2) if t >= 0.0]
    return min(candidatos) if candidatos else float("inf")


def calcular_serie_ttc_real(registro: Mapping[str, Any], radio_robot: float) -> np.ndarray:
    estados = list(registro.get("estados", []))
    controles = list(registro.get("controles", []))
    historial = list(registro.get("obstaculos_dinamicos", []))
    n = min(len(controles), len(historial), max(0, len(estados) - 1))
    serie = []
    for indice in range(n):
        estado = estados[indice + 1]
        velocidad_lineal = float(controles[indice][0])
        obstaculos = historial[indice]
        valores = [
            ttc_circulos_movimiento_constante(
                estado_robot=estado,
                velocidad_lineal=velocidad_lineal,
                obstaculo=obstaculo,
                radio_robot=radio_robot,
            )
            for obstaculo in obstaculos
        ]
        serie.append(min(valores) if valores else float("inf"))
    return np.asarray(serie, dtype=float)


def metricas_control(controles: Sequence[Sequence[float]], dt: float) -> Dict[str, float]:
    if not controles:
        return {
            "velocidad_lineal_media_m_s": 0.0,
            "velocidad_angular_abs_media_rad_s": 0.0,
            "numero_paradas": 0,
            "variacion_total_v": 0.0,
            "variacion_total_omega": 0.0,
            "aceleracion_lineal_rms_m_s2": 0.0,
            "aceleracion_angular_rms_rad_s2": 0.0,
            "jerk_lineal_rms_m_s3": 0.0,
            "jerk_angular_rms_rad_s3": 0.0,
            "cambios_signo_omega": 0,
            "esfuerzo_control_medio": 0.0,
        }

    arreglo = np.asarray(controles, dtype=float)
    v = arreglo[:, 0]
    omega = arreglo[:, 1]
    dv = np.diff(v)
    domega = np.diff(omega)
    aceleracion_v = dv / dt if dv.size else np.asarray([], dtype=float)
    aceleracion_omega = domega / dt if domega.size else np.asarray([], dtype=float)
    jerk_v = np.diff(aceleracion_v) / dt if aceleracion_v.size >= 2 else np.asarray([], dtype=float)
    jerk_omega = (
        np.diff(aceleracion_omega) / dt
        if aceleracion_omega.size >= 2
        else np.asarray([], dtype=float)
    )

    mascara_parada = v <= UMBRAL_PARADA_M_S
    numero_paradas = contar_eventos_mascara(mascara_parada)

    omega_filtrada = omega.copy()
    omega_filtrada[np.abs(omega_filtrada) < UMBRAL_OMEGA_SIGNO_RAD_S] = 0.0
    signos_no_cero = np.sign(omega_filtrada)
    cambios_signo = 0
    ultimo = 0.0
    for signo in signos_no_cero:
        if signo == 0.0:
            continue
        if ultimo != 0.0 and signo != ultimo:
            cambios_signo += 1
        ultimo = signo

    esfuerzo = np.mean(
        (v / max(1e-8, 1.0)) ** 2
        + (omega / max(1e-8, 1.2)) ** 2
    )

    return {
        "velocidad_lineal_media_m_s": float(np.mean(v)),
        "velocidad_angular_abs_media_rad_s": float(np.mean(np.abs(omega))),
        "numero_paradas": int(numero_paradas),
        "variacion_total_v": float(np.sum(np.abs(dv))) if dv.size else 0.0,
        "variacion_total_omega": float(np.sum(np.abs(domega))) if domega.size else 0.0,
        "aceleracion_lineal_rms_m_s2": rms(aceleracion_v),
        "aceleracion_angular_rms_rad_s2": rms(aceleracion_omega),
        "jerk_lineal_rms_m_s3": rms(jerk_v),
        "jerk_angular_rms_rad_s3": rms(jerk_omega),
        "cambios_signo_omega": int(cambios_signo),
        "esfuerzo_control_medio": float(esfuerzo),
    }


def etiquetas_futuras_desde_clearance(
    clearances_dinamicos: np.ndarray,
    horizonte_pasos: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = clearances_dinamicos.size
    minimos = np.full(n, np.nan, dtype=float)
    colision = np.zeros(n, dtype=float)
    casi_colision = np.zeros(n, dtype=float)
    ttc = np.full(n, np.inf, dtype=float)

    for indice in range(n):
        ventana = clearances_dinamicos[indice : min(n, indice + horizonte_pasos + 1)]
        if ventana.size == 0:
            continue
        finitos = ventana[np.isfinite(ventana)]
        if finitos.size:
            minimos[indice] = float(np.min(finitos))
        indices_colision = np.flatnonzero(ventana <= 0.0)
        indices_casi = np.flatnonzero(ventana <= UMBRAL_CASI_COLISION_M)
        colision[indice] = float(indices_colision.size > 0)
        casi_colision[indice] = float(indices_casi.size > 0)
        if indices_colision.size:
            ttc[indice] = float(indices_colision[0] * dt)

    return minimos, colision, casi_colision, ttc


def brier(predicciones: Sequence[float], objetivos: Sequence[float]) -> float:
    p = np.asarray(predicciones, dtype=float)
    y = np.asarray(objetivos, dtype=float)
    n = min(p.size, y.size)
    if n == 0:
        return float("nan")
    p, y = p[:n], y[:n]
    mascara = np.isfinite(p) & np.isfinite(y)
    if not np.any(mascara):
        return float("nan")
    p = np.clip(p[mascara], 0.0, 1.0)
    y = y[mascara]
    return float(np.mean((p - y) ** 2))


def ece(predicciones: Sequence[float], objetivos: Sequence[float], numero_bins: int = 10) -> float:
    p = np.asarray(predicciones, dtype=float)
    y = np.asarray(objetivos, dtype=float)
    n = min(p.size, y.size)
    if n == 0:
        return float("nan")
    p, y = p[:n], y[:n]
    mascara = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[mascara], 0.0, 1.0), y[mascara]
    if p.size == 0:
        return float("nan")
    bordes = np.linspace(0.0, 1.0, numero_bins + 1)
    error = 0.0
    for indice in range(numero_bins):
        if indice == numero_bins - 1:
            mascara_bin = (p >= bordes[indice]) & (p <= bordes[indice + 1])
        else:
            mascara_bin = (p >= bordes[indice]) & (p < bordes[indice + 1])
        cantidad = int(np.sum(mascara_bin))
        if cantidad == 0:
            continue
        confianza = float(np.mean(p[mascara_bin]))
        frecuencia = float(np.mean(y[mascara_bin]))
        error += (cantidad / p.size) * abs(confianza - frecuencia)
    return float(error)
def precision_recall_f1(predicciones_binarias: Sequence[float], objetivos: Sequence[float]):
    p = np.asarray(predicciones_binarias, dtype=float)
    y = np.asarray(objetivos, dtype=float)
    n = min(p.size, y.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = p[:n] >= 0.5
    y = y[:n] >= 0.5
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0
        else float("nan")
    )
    return float(precision), float(recall), float(f1)
def metricas_predictivas(
    registro: Mapping[str, Any],
    clearances_dinamicos: np.ndarray,
    dt: float,
    entorno: Optional[Mapping[str, Any]],
    upo=None,
) -> Dict[str, float]:
    resultado = {
        clave: float("nan")
        for clave in (
            "clearance_predicho_minimo_m",
            "clearance_predicho_promedio_m",
            "clearance_futuro_real_minimo_m",
            "error_mae_clearance_predicho_m",
            "ttc_predicho_minimo_s",
            "ttc_predicho_promedio_s",
            "error_mae_ttc_predicho_condicional_s",
            "fraccion_colision_predicha",
            "brier_colision_predicha",
            "precision_colision_predicha",
            "recall_colision_predicha",
            "f1_colision_predicha",
            "probabilidad_colision_media",
            "probabilidad_colision_maxima",
            "brier_probabilidad_colision",
            "ece_probabilidad_colision",
            "probabilidad_casi_colision_media",
            "probabilidad_casi_colision_maxima",
            "brier_probabilidad_casi_colision",
            "ece_probabilidad_casi_colision",
            "clearance_cvar_minimo_m",
            "clearance_cvar_promedio_m",
            "ttc_cvar_minimo_s",
            "ttc_cvar_promedio_s",
            "penalizacion_predictiva_media",
            "velocidad_cierre_predicha_maxima_m_s",
            "latencia_observacion_pasos",
            "sigma_posicion_observacion_m",
            "sigma_velocidad_observacion_m_s",
            "numero_muestras_monte_carlo",
        )}
    pred_clearance = arreglo_float(registro.get("clearances_dinamicos_predichos", []))
    pred_ttc = arreglo_float(registro.get("ttc_dinamicos_predichos", []))
    pred_colision = arreglo_float(registro.get("colisiones_dinamicas_predichas", []))
    pred_cierre = arreglo_float(registro.get("velocidades_cierre_dinamicas", []))
    penalizaciones = arreglo_float(registro.get("penalizaciones_riesgo_predictivo", []))
    if pred_clearance.size == 0 and pred_ttc.size == 0:
        return resultado
    horizonte_pasos = max(1, int(round(HORIZONTE_CALIBRACION_S / dt)))
    real_minimo, real_colision, real_casi, real_ttc = etiquetas_futuras_desde_clearance(
        clearances_dinamicos, horizonte_pasos, dt
    )
    resultado["clearance_predicho_minimo_m"] = minimo_finito(pred_clearance)
    resultado["clearance_predicho_promedio_m"] = media_finita(pred_clearance)
    resultado["clearance_futuro_real_minimo_m"] = minimo_finito(real_minimo)
    n_clearance = min(pred_clearance.size, real_minimo.size)
    if n_clearance:
        mascara = np.isfinite(pred_clearance[:n_clearance]) & np.isfinite(real_minimo[:n_clearance])
        if np.any(mascara):
            resultado["error_mae_clearance_predicho_m"] = float(
                np.mean(np.abs(pred_clearance[:n_clearance][mascara] - real_minimo[:n_clearance][mascara]))
            )
    resultado["ttc_predicho_minimo_s"] = minimo_finito(pred_ttc)
    resultado["ttc_predicho_promedio_s"] = media_finita(pred_ttc)
    n_ttc = min(pred_ttc.size, real_ttc.size)
    if n_ttc:
        mascara = np.isfinite(pred_ttc[:n_ttc]) & np.isfinite(real_ttc[:n_ttc])
        if np.any(mascara):
            resultado["error_mae_ttc_predicho_condicional_s"] = float(
                np.mean(np.abs(pred_ttc[:n_ttc][mascara] - real_ttc[:n_ttc][mascara]))
            )
    if pred_colision.size:
        resultado["fraccion_colision_predicha"] = media_finita(pred_colision)
        resultado["brier_colision_predicha"] = brier(pred_colision, real_colision)
        precision, recall, f1 = precision_recall_f1(pred_colision, real_colision)
        resultado["precision_colision_predicha"] = precision
        resultado["recall_colision_predicha"] = recall
        resultado["f1_colision_predicha"] = f1
    resultado["penalizacion_predictiva_media"] = media_finita(penalizaciones)
    resultado["velocidad_cierre_predicha_maxima_m_s"] = maximo_finito(pred_cierre)
    probabilidades_colision = arreglo_float(
        registro.get("probabilidades_colision_predichas", [])
    )
    probabilidades_casi = arreglo_float(
        registro.get("probabilidades_casi_colision_predichas", [])
    )
    clearances_cvar = arreglo_float(registro.get("clearances_cvar_predichos", []))
    ttc_cvar = arreglo_float(registro.get("ttc_cvar_predichos", []))
    if probabilidades_colision.size:
        resultado["probabilidad_colision_media"] = media_finita(probabilidades_colision)
        resultado["probabilidad_colision_maxima"] = maximo_finito(probabilidades_colision)
        resultado["brier_probabilidad_colision"] = brier(
            probabilidades_colision, real_colision
        )
        resultado["ece_probabilidad_colision"] = ece(
            probabilidades_colision, real_colision
        )
        precision, recall, f1 = precision_recall_f1(
            probabilidades_colision, real_colision
        )
        resultado["precision_colision_predicha"] = precision
        resultado["recall_colision_predicha"] = recall
        resultado["f1_colision_predicha"] = f1

    if probabilidades_casi.size:
        resultado["probabilidad_casi_colision_media"] = media_finita(probabilidades_casi)
        resultado["probabilidad_casi_colision_maxima"] = maximo_finito(probabilidades_casi)
        resultado["brier_probabilidad_casi_colision"] = brier(
            probabilidades_casi, real_casi
        )
        resultado["ece_probabilidad_casi_colision"] = ece(
            probabilidades_casi, real_casi
        )
    resultado["clearance_cvar_minimo_m"] = minimo_finito(clearances_cvar)
    resultado["clearance_cvar_promedio_m"] = media_finita(clearances_cvar)
    resultado["ttc_cvar_minimo_s"] = minimo_finito(ttc_cvar)
    resultado["ttc_cvar_promedio_s"] = media_finita(ttc_cvar)
    if upo is not None:
        resultado["numero_muestras_monte_carlo"] = float(
            getattr(upo, "NUMERO_MUESTRAS_MONTE_CARLO", float("nan"))
        )
        if entorno is not None and hasattr(upo, "CLAVE_CONFIGURACION"):
            configuracion = entorno.get(upo.CLAVE_CONFIGURACION, {})
            resultado["latencia_observacion_pasos"] = valor_finito_o_nan(
                configuracion.get("latencia_pasos", float("nan"))
            )
            resultado["sigma_posicion_observacion_m"] = valor_finito_o_nan(
                configuracion.get("sigma_posicion_m", float("nan"))
            )
            resultado["sigma_velocidad_observacion_m_s"] = valor_finito_o_nan(
                configuracion.get("sigma_velocidad_m_s", float("nan"))
            )
    return resultado

def convertir_resultado_a_fila(
    resultado: Mapping[str, Any],
    metodo: str,
    numero_parametros: int,
    base,
    upo=None,
) -> Dict[str, Any]:
    registro = resultado["registro"]
    escenario = resultado["escenario"]
    estados = list(registro.get("estados", []))
    controles = list(registro.get("controles", []))
    clearances_estaticos = arreglo_float(registro.get("clearances_estaticos", []))
    clearances_dinamicos = arreglo_float(registro.get("clearances_dinamicos", []))
    n_clearance = min(clearances_estaticos.size, clearances_dinamicos.size)
    clearances_totales = (
        np.minimum(clearances_estaticos[:n_clearance], clearances_dinamicos[:n_clearance])
        if n_clearance
        else np.asarray([], dtype=float)
    )

    posiciones = np.asarray([[float(e[0]), float(e[1])] for e in estados], dtype=float)
    longitud_trayectoria = (
        float(np.sum(np.linalg.norm(np.diff(posiciones, axis=0), axis=1)))
        if posiciones.shape[0] >= 2
        else 0.0
    )
    longitud_astar = float(escenario["longitud_camino"])
    distancia_inicial = float(
        base.distancia_entre_puntos(escenario["inicio"], escenario["meta"])
    )
    estado_final = estados[-1]
    distancia_final = float(
        base.distancia_entre_puntos((float(estado_final[0]), float(estado_final[1])), escenario["meta"])
    )
    progreso = (
        float(np.clip((distancia_inicial - distancia_final) / max(distancia_inicial, 1e-8), -1.0, 1.0))
    )

    errores = np.asarray(
        [
            base.distancia_punto_camino((float(e[0]), float(e[1])), escenario["camino_mundo"])
            for e in estados
        ],
        dtype=float,
    )

    resultado_texto = str(resultado["resultado"])
    exito = int(resultado_texto == "meta")
    pasos = int(registro.get("pasos_ejecutados", len(controles)))
    dt = float(base.DT)
    spl = float(exito * longitud_astar / max(longitud_astar, longitud_trayectoria, 1e-8))
    eficiencia_geometrica = float(longitud_astar / max(longitud_astar, longitud_trayectoria, 1e-8))

    mascara_casi = (
        np.isfinite(clearances_dinamicos)
        & (clearances_dinamicos > 0.0)
        & (clearances_dinamicos <= UMBRAL_CASI_COLISION_M)
    )
    mascara_riesgo_total = (
        np.isfinite(clearances_dinamicos)
        & (clearances_dinamicos <= UMBRAL_CASI_COLISION_M)
    )

    serie_ttc = calcular_serie_ttc_real(registro, float(base.RADIO_ROBOT))
    mascara_ttc = np.isfinite(serie_ttc) & (serie_ttc < UMBRAL_TTC_RIESGOSO_S)

    control = metricas_control(controles, dt)

    tiempos_inferencia = arreglo_float(resultado.get("tiempos_inferencia_s", []))
    tiempos_paso = arreglo_float(resultado.get("tiempos_paso_s", []))
    tiempo_control_total = float(np.sum(tiempos_inferencia) + np.sum(tiempos_paso))
    tiempo_ciclo = tiempo_control_total / max(1, pasos)

    huella_escenario, huella_dinamicos = calcular_huellas(resultado)

    fila: Dict[str, Any] = {
        "condicion": "nominal_limpia",
        "semilla": int(resultado["semilla"]),
        "metodo": metodo,
        "resultado": resultado_texto,
        "exito": exito,
        "colision_estatica": int(resultado_texto == "colision_estatica"),
        "colision_dinamica": int(resultado_texto == "colision_dinamica"),
        "fuera_mapa": int(resultado_texto == "fuera_mapa"),
        "timeout": int(resultado_texto == "timeout"),
        "pasos": pasos,
        "tiempo_simulado_s": float(pasos * dt),
        "recompensa_acumulada": valor_finito_o_nan(resultado.get("recompensa_acumulada")),
        "distancia_inicial_meta_m": distancia_inicial,
        "distancia_final_meta_m": distancia_final,
        "progreso_normalizado": progreso,
        "longitud_astar_m": longitud_astar,
        "longitud_trayectoria_m": longitud_trayectoria,
        "exceso_longitud_m": float(longitud_trayectoria - longitud_astar),
        "eficiencia_geometrica": eficiencia_geometrica,
        "spl": spl,
        "error_medio_ruta_m": media_finita(errores),
        "error_rmse_ruta_m": rms(errores),
        "error_maximo_ruta_m": maximo_finito(errores),
        "clearance_estatico_minimo_m": minimo_finito(clearances_estaticos),
        "clearance_dinamico_minimo_m": minimo_finito(clearances_dinamicos),
        "clearance_total_minimo_m": minimo_finito(clearances_totales),
        "clearance_total_percentil_5_m": percentil_finito(clearances_totales, 5.0),
        "clearance_total_promedio_m": media_finita(clearances_totales),
        "hubo_casi_colision_real": int(np.any(mascara_casi)),
        "eventos_casi_colision_real": int(contar_eventos_mascara(mascara_casi)),
        "pasos_en_casi_colision": int(np.sum(mascara_casi)),
        "tiempo_en_casi_colision_s": float(np.sum(mascara_casi) * dt),
        "fraccion_tiempo_casi_colision": float(np.mean(mascara_casi)) if mascara_casi.size else 0.0,
        "pasos_en_riesgo_total": int(np.sum(mascara_riesgo_total)),
        "tiempo_en_riesgo_total_s": float(np.sum(mascara_riesgo_total) * dt),
        "ttc_real_minimo_s": minimo_finito(serie_ttc),
        "pasos_ttc_menor_1s": int(np.sum(mascara_ttc)),
        "tiempo_ttc_menor_1s_s": float(np.sum(mascara_ttc) * dt),
        "eventos_ttc_menor_1s": int(contar_eventos_mascara(mascara_ttc)),
        **control,
        "tiempo_preparacion_ms": 1000.0 * float(resultado.get("tiempo_preparacion_s", 0.0)),
        "tiempo_control_total_ms": 1000.0 * tiempo_control_total,
        "tiempo_ciclo_medio_ms": 1000.0 * tiempo_ciclo,
        "inferencia_actor_media_ms": 1000.0 * media_finita(tiempos_inferencia),
        "inferencia_actor_p95_ms": 1000.0 * percentil_finito(tiempos_inferencia, 95.0),
        "inferencia_actor_p99_ms": 1000.0 * percentil_finito(tiempos_inferencia, 99.0),
        "procesamiento_paso_medio_ms": 1000.0 * media_finita(tiempos_paso),
        "numero_parametros": int(numero_parametros),
        "huella_escenario": huella_escenario,
        "huella_dinamicos_iniciales": huella_dinamicos,
    }

    entorno = resultado.get("entorno")
    fila.update(
        metricas_predictivas(
            registro=registro,
            clearances_dinamicos=clearances_dinamicos,
            dt=dt,
            entorno=entorno,
            upo=upo if metodo == "A* + SAC-UPO-TTC" else None,
        )
    )
    # DWA no tiene actor neuronal. Su tiempo de "inferencia" corresponde al
    # controlador DWA y se conserva, pero el número de parámetros es cero.
    for campo in CAMPOS_RESULTADOS:
        fila.setdefault(campo, float("nan"))
    return {campo: fila[campo] for campo in CAMPOS_RESULTADOS}
# =============================================================================
# GUARDADO Y REANUDACIÓN
# =============================================================================
def guardar_csv(ruta: Path, filas: Sequence[Mapping[str, Any]], campos: Optional[Sequence[str]] = None):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if campos is None:
        campos = list(filas[0].keys()) if filas else ["sin_datos"]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(campos), extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)
def anexar_fila_csv(ruta: Path, fila: Mapping[str, Any], campos: Sequence[str]):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.is_file() and ruta.stat().st_size > 0
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(campos), extrasaction="ignore")
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)
        archivo.flush()
        os.fsync(archivo.fileno())
def leer_csv_resultados(ruta: Path) -> List[Dict[str, Any]]:
    if not ruta.is_file():
        return []
    with ruta.open("r", newline="", encoding="utf-8") as archivo:
        return [dict(fila) for fila in csv.DictReader(archivo)]
def convertir_tipos_filas(filas: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    campos_texto = {"condicion", "metodo", "resultado", "huella_escenario", "huella_dinamicos_iniciales"}
    campos_enteros = {
        "semilla", "exito", "colision_estatica", "colision_dinamica", "fuera_mapa", "timeout",
        "pasos", "hubo_casi_colision_real", "eventos_casi_colision_real",
        "pasos_en_casi_colision", "pasos_en_riesgo_total", "pasos_ttc_menor_1s",
        "eventos_ttc_menor_1s", "numero_paradas", "cambios_signo_omega", "numero_parametros",
    }
    salida = []
    for fila_original in filas:
        fila = dict(fila_original)
        for campo in CAMPOS_RESULTADOS:
            valor = fila.get(campo, "")
            if campo in campos_texto:
                fila[campo] = str(valor)
            elif campo in campos_enteros:
                try:
                    fila[campo] = int(float(valor))
                except Exception:
                    fila[campo] = 0
            else:
                try:
                    fila[campo] = float(valor)
                except Exception:
                    fila[campo] = float("nan")
        salida.append(fila)
    return salida


# =============================================================================
# RESÚMENES Y ESTADÍSTICA PAREADA
# =============================================================================

def resumen_por_metodo(filas: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    resumen = []
    for metodo in METODOS:
        grupo = [fila for fila in filas if fila["metodo"] == metodo]
        n = len(grupo)
        if n == 0:
            continue
        exitos = sum(int(fila["exito"]) for fila in grupo)
        ic_inf, ic_sup = intervalo_wilson(exitos, n)
        exitos_grupo = [fila for fila in grupo if int(fila["exito"]) == 1]
        resumen.append({
            "metodo": metodo,
            "numero_episodios": n,
            "exitos": exitos,
            "tasa_exito": exitos / n,
            "ic_wilson_95_inferior": ic_inf,
            "ic_wilson_95_superior": ic_sup,
            "colision_estatica": sum(int(f["colision_estatica"]) for f in grupo),
            "tasa_colision_estatica": np.mean([int(f["colision_estatica"]) for f in grupo]),
            "colision_dinamica": sum(int(f["colision_dinamica"]) for f in grupo),
            "tasa_colision_dinamica": np.mean([int(f["colision_dinamica"]) for f in grupo]),
            "fuera_mapa": sum(int(f["fuera_mapa"]) for f in grupo),
            "tasa_fuera_mapa": np.mean([int(f["fuera_mapa"]) for f in grupo]),
            "timeout": sum(int(f["timeout"]) for f in grupo),
            "tasa_timeout": np.mean([int(f["timeout"]) for f in grupo]),
            "spl_medio": media_finita(f["spl"] for f in grupo),
            "tiempo_exitos_media_s": media_finita(f["tiempo_simulado_s"] for f in exitos_grupo),
            "longitud_exitos_media_m": media_finita(f["longitud_trayectoria_m"] for f in exitos_grupo),
            "distancia_final_media_m": media_finita(f["distancia_final_meta_m"] for f in grupo),
            "clearance_dinamico_minimo_media_m": media_finita(f["clearance_dinamico_minimo_m"] for f in grupo),
            "clearance_total_p5_media_m": media_finita(f["clearance_total_percentil_5_m"] for f in grupo),
            "episodios_con_casi_colision": sum(int(f["hubo_casi_colision_real"]) for f in grupo),
            "tasa_episodios_con_casi_colision": np.mean([int(f["hubo_casi_colision_real"]) for f in grupo]),
            "eventos_casi_colision_media": media_finita(f["eventos_casi_colision_real"] for f in grupo),
            "tiempo_casi_colision_media_s": media_finita(f["tiempo_en_casi_colision_s"] for f in grupo),
            "ttc_real_minimo_media_s": media_finita(f["ttc_real_minimo_s"] for f in grupo),
            "jerk_angular_rms_media": media_finita(f["jerk_angular_rms_rad_s3"] for f in grupo),
            "esfuerzo_control_medio": media_finita(f["esfuerzo_control_medio"] for f in grupo),
            "tiempo_ciclo_medio_ms": media_finita(f["tiempo_ciclo_medio_ms"] for f in grupo),
            "inferencia_actor_media_ms": media_finita(f["inferencia_actor_media_ms"] for f in grupo),
            "brier_colision_predicha_media": media_finita(f["brier_colision_predicha"] for f in grupo),
            "brier_probabilidad_colision_media": media_finita(f["brier_probabilidad_colision"] for f in grupo),
            "ece_probabilidad_colision_media": media_finita(f["ece_probabilidad_colision"] for f in grupo),
        })
    return resumen

def p_mcnemar_exacto(b: int, c: int) -> float:
    n = int(b + c)
    if n == 0:
        return 1.0
    menor = min(int(b), int(c))
    probabilidad = sum(math.comb(n, k) for k in range(menor + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * probabilidad))

def bootstrap_diferencia_pareada(
    a: np.ndarray,
    b: np.ndarray,
    muestras: int,
    semilla: int,
) -> Tuple[float, float]:
    n = min(a.size, b.size)
    if n == 0:
        return float("nan"), float("nan")
    diferencias = b[:n] - a[:n]
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, n, size=(muestras, n))
    medias = np.mean(diferencias[indices], axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))

def ajustar_holm(p_valores: Sequence[float]) -> List[float]:
    p = np.asarray(p_valores, dtype=float)
    salida = np.full(p.shape, np.nan, dtype=float)
    validos = np.flatnonzero(np.isfinite(p))
    if validos.size == 0:
        return salida.tolist()
    orden_local = validos[np.argsort(p[validos])]
    m = orden_local.size
    maximo_previo = 0.0
    for rango, indice in enumerate(orden_local):
        ajustado = min(1.0, (m - rango) * p[indice])
        maximo_previo = max(maximo_previo, ajustado)
        salida[indice] = maximo_previo
    return salida.tolist()


def matriz_pareada(filas: Sequence[Mapping[str, Any]], metrica: str):
    por_clave = {(int(f["semilla"]), str(f["metodo"])): f for f in filas}
    semillas = sorted({int(f["semilla"]) for f in filas})
    matriz = []
    semillas_validas = []
    for semilla in semillas:
        valores = []
        completo = True
        for metodo in METODOS:
            fila = por_clave.get((semilla, metodo))
            if fila is None:
                completo = False
                break
            valor = valor_finito_o_nan(fila[metrica])
            if not np.isfinite(valor):
                completo = False
                break
            valores.append(valor)
        if completo:
            matriz.append(valores)
            semillas_validas.append(semilla)
    return np.asarray(matriz, dtype=float), semillas_validas

def comparaciones_binarias(
    filas: Sequence[Mapping[str, Any]],
    muestras_bootstrap: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    por_clave = {(int(f["semilla"]), str(f["metodo"])): f for f in filas}
    semillas = sorted({int(f["semilla"]) for f in filas})
    pares = []
    desacuerdos = []
    p_sin_ajustar = []

    for indice_par, (metodo_a, metodo_b) in enumerate(itertools.combinations(METODOS, 2)):
        a, b, semillas_par = [], [], []
        for semilla in semillas:
            fila_a = por_clave.get((semilla, metodo_a))
            fila_b = por_clave.get((semilla, metodo_b))
            if fila_a is None or fila_b is None:
                continue
            valor_a, valor_b = int(fila_a["exito"]), int(fila_b["exito"])
            a.append(valor_a)
            b.append(valor_b)
            semillas_par.append(semilla)
            if valor_a != valor_b:
                desacuerdos.append({
                    "semilla": semilla,
                    "metodo_a": metodo_a,
                    "resultado_a": fila_a["resultado"],
                    "metodo_b": metodo_b,
                    "resultado_b": fila_b["resultado"],
                })
        a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        a_exito_b_fallo = int(np.sum((a_arr == 1) & (b_arr == 0)))
        a_fallo_b_exito = int(np.sum((a_arr == 0) & (b_arr == 1)))
        p = p_mcnemar_exacto(a_exito_b_fallo, a_fallo_b_exito)
        ic_inf, ic_sup = bootstrap_diferencia_pareada(
            a_arr, b_arr, muestras_bootstrap, 900000 + indice_par
        )
        fila = {
            "metodo_a": metodo_a,
            "metodo_b": metodo_b,
            "numero_pares": int(a_arr.size),
            "tasa_exito_a": float(np.mean(a_arr)) if a_arr.size else float("nan"),
            "tasa_exito_b": float(np.mean(b_arr)) if b_arr.size else float("nan"),
            "diferencia_exito_b_menos_a": float(np.mean(b_arr - a_arr)) if a_arr.size else float("nan"),
            "ic_bootstrap_95_inferior": ic_inf,
            "ic_bootstrap_95_superior": ic_sup,
            "a_exito_b_fallo": a_exito_b_fallo,
            "a_fallo_b_exito": a_fallo_b_exito,
            "p_mcnemar_exacto": p,
        }
        pares.append(fila)
        p_sin_ajustar.append(p)

    p_holm = ajustar_holm(p_sin_ajustar)
    for fila, p_ajustado in zip(pares, p_holm):
        fila["p_mcnemar_holm"] = p_ajustado
        fila["significativo_0_05_holm"] = int(np.isfinite(p_ajustado) and p_ajustado < 0.05)

    matriz, semillas_completas = matriz_pareada(filas, "exito")
    cochran = {
        "numero_semillas_completas": len(semillas_completas),
        "numero_metodos": len(METODOS),
        "q_cochran": float("nan"),
        "grados_libertad": len(METODOS) - 1,
        "p_global": float("nan"),
    }
    if matriz.size:
        k = matriz.shape[1]
        sumas_columnas = np.sum(matriz, axis=0)
        sumas_filas = np.sum(matriz, axis=1)
        total = float(np.sum(sumas_columnas))
        denominador = k * total - float(np.sum(sumas_filas ** 2))
        if denominador > 0.0:
            q = (k - 1.0) * (
                k * float(np.sum(sumas_columnas ** 2)) - total ** 2
            ) / denominador
            cochran["q_cochran"] = float(q)
            if scipy_stats is not None:
                cochran["p_global"] = float(scipy_stats.chi2.sf(q, k - 1))

    return pares, cochran, desacuerdos


def comparaciones_continuas(filas: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    globales = []
    pares_todos = []
    for metrica in METRICAS_CONTINUAS_PAREADAS:
        matriz, semillas = matriz_pareada(filas, metrica)
        global_fila = {
            "metrica": metrica,
            "numero_semillas_completas": int(matriz.shape[0]) if matriz.ndim == 2 else 0,
            "estadistico_friedman": float("nan"),
            "p_friedman": float("nan"),
        }
        if scipy_stats is not None and matriz.ndim == 2 and matriz.shape[0] >= 3:
            try:
                estadistico, p = scipy_stats.friedmanchisquare(
                    *[matriz[:, indice] for indice in range(matriz.shape[1])]
                )
                global_fila["estadistico_friedman"] = float(estadistico)
                global_fila["p_friedman"] = float(p)
            except Exception:
                pass
        globales.append(global_fila)

        filas_metrica = []
        p_valores = []
        if matriz.ndim != 2 or matriz.shape[0] == 0:
            continue
        for metodo_a, metodo_b in itertools.combinations(METODOS, 2):
            ia, ib = METODOS.index(metodo_a), METODOS.index(metodo_b)
            a, b = matriz[:, ia], matriz[:, ib]
            diferencias = b - a
            p = float("nan")
            estadistico = float("nan")
            if scipy_stats is not None and diferencias.size > 0 and not np.allclose(diferencias, 0.0):
                try:
                    prueba = scipy_stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    estadistico, p = float(prueba.statistic), float(prueba.pvalue)
                except Exception:
                    pass
            positivos = int(np.sum(diferencias > 0.0))
            negativos = int(np.sum(diferencias < 0.0))
            denominador = positivos + negativos
            rank_biserial_aprox = (
                (positivos - negativos) / denominador if denominador else 0.0
            )
            fila = {
                "metrica": metrica,
                "metodo_a": metodo_a,
                "metodo_b": metodo_b,
                "numero_pares": int(diferencias.size),
                "media_a": float(np.mean(a)),
                "media_b": float(np.mean(b)),
                "mediana_a": float(np.median(a)),
                "mediana_b": float(np.median(b)),
                "diferencia_media_b_menos_a": float(np.mean(diferencias)),
                "diferencia_mediana_b_menos_a": float(np.median(diferencias)),
                "estadistico_wilcoxon": estadistico,
                "p_wilcoxon": p,
                "rank_biserial_signos_aprox": float(rank_biserial_aprox),
            }
            filas_metrica.append(fila)
            p_valores.append(p)

        p_holm = ajustar_holm(p_valores)
        for fila, p_ajustado in zip(filas_metrica, p_holm):
            fila["p_wilcoxon_holm"] = p_ajustado
            fila["significativo_0_05_holm"] = int(np.isfinite(p_ajustado) and p_ajustado < 0.05)
        pares_todos.extend(filas_metrica)

    return globales, pares_todos

# =============================================================================
# GRÁFICAS Y RESUMEN
# =============================================================================

def crear_graficas(directorio: Path, resumen: Sequence[Mapping[str, Any]]):
    etiquetas = [fila["metodo"] for fila in resumen]

    figura, eje = plt.subplots(figsize=(12, 7))
    tasas = np.asarray([100.0 * float(f["tasa_exito"]) for f in resumen])
    inferiores = np.asarray([100.0 * float(f["ic_wilson_95_inferior"]) for f in resumen])
    superiores = np.asarray([100.0 * float(f["ic_wilson_95_superior"]) for f in resumen])
    errores = np.vstack([tasas - inferiores, superiores - tasas])
    posiciones = np.arange(len(etiquetas))
    barras = eje.bar(posiciones, tasas, yerr=errores, capsize=5)
    eje.set_xticks(posiciones, etiquetas, rotation=20, ha="right")
    eje.set_ylabel("Tasa de éxito (%)")
    eje.set_ylim(0, 105)
    eje.set_title("Comparación pareada de éxito — condición nominal limpia")
    eje.grid(axis="y", alpha=0.25)
    for barra, valor in zip(barras, tasas):
        eje.text(barra.get_x() + barra.get_width() / 2.0, valor + 1.0, f"{valor:.1f}%", ha="center")
    figura.tight_layout()
    figura.savefig(directorio / "01_tasa_exito_ic_wilson.png", dpi=300, bbox_inches="tight")
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(12, 7))
    fondo = np.zeros(len(etiquetas))
    for clave, titulo in (
        ("tasa_colision_dinamica", "Colisión dinámica"),
        ("tasa_colision_estatica", "Colisión estática"),
        ("tasa_timeout", "Timeout"),
        ("tasa_fuera_mapa", "Fuera del mapa"),
    ):
        valores = 100.0 * np.asarray([float(f[clave]) for f in resumen])
        eje.bar(posiciones, valores, bottom=fondo, label=titulo)
        fondo += valores
    eje.set_xticks(posiciones, etiquetas, rotation=20, ha="right")
    eje.set_ylabel("Episodios fallidos (%)")
    eje.set_title("Distribución de fallos")
    eje.grid(axis="y", alpha=0.25)
    eje.legend()
    figura.tight_layout()
    figura.savefig(directorio / "02_distribucion_fallos.png", dpi=300, bbox_inches="tight")
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(12, 7))
    valores = [float(f["eventos_casi_colision_media"]) for f in resumen]
    eje.bar(posiciones, valores)
    eje.set_xticks(posiciones, etiquetas, rotation=20, ha="right")
    eje.set_ylabel("Eventos por episodio")
    eje.set_title(f"Casi colisiones reales (0 < clearance dinámico ≤ {UMBRAL_CASI_COLISION_M:.2f} m)")
    eje.grid(axis="y", alpha=0.25)
    figura.tight_layout()
    figura.savefig(directorio / "03_eventos_casi_colision.png", dpi=300, bbox_inches="tight")
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(12, 7))
    valores = [float(f["tiempo_ciclo_medio_ms"]) for f in resumen]
    eje.bar(posiciones, valores)
    eje.set_xticks(posiciones, etiquetas, rotation=20, ha="right")
    eje.set_ylabel("Tiempo por ciclo (ms)")
    eje.set_title("Costo computacional observado")
    eje.grid(axis="y", alpha=0.25)
    figura.tight_layout()
    figura.savefig(directorio / "04_tiempo_ciclo.png", dpi=300, bbox_inches="tight")
    plt.close(figura)


def construir_resumen_texto(
    resumen: Sequence[Mapping[str, Any]],
    cochran: Mapping[str, Any],
    comparaciones_exito: Sequence[Mapping[str, Any]],
    info_actores: Mapping[str, Mapping[str, Any]],
    argumentos: argparse.Namespace,
) -> str:
    lineas = [
        "COMPARACIÓN PAREADA DE CINCO MÉTODOS",
        "=" * 80,
        "Condición: nominal limpia",
        f"Semillas: {argumentos.semilla_base} a {argumentos.semilla_base + argumentos.numero_semillas - 1}",
        f"Número de escenarios: {argumentos.numero_semillas}",
        "Entrenamiento durante la comparación: NO",
        "UPO-TTC: ruido sensorial = 0 y latencia = 0; incertidumbre Monte Carlo interna activa.",
        "",
        "CHECKPOINTS CONGELADOS",
        "-" * 80,
    ]
    for metodo, info in info_actores.items():
        lineas.append(
            f"{metodo}: episodio={info['episodio']} | parámetros={info['numero_parametros']} | "
            f"sha256={info['sha256']} | {info['ruta']}"
        )

    lineas.extend(["", "RESULTADOS POR MÉTODO", "-" * 80])
    for fila in resumen:
        lineas.extend([
            str(fila["metodo"]),
            f"  Éxito: {100.0 * float(fila['tasa_exito']):.2f}% "
            f"[IC Wilson: {100.0 * float(fila['ic_wilson_95_inferior']):.2f}%, "
            f"{100.0 * float(fila['ic_wilson_95_superior']):.2f}%]",
            f"  Colisión dinámica: {100.0 * float(fila['tasa_colision_dinamica']):.2f}%",
            f"  Colisión estática: {100.0 * float(fila['tasa_colision_estatica']):.2f}%",
            f"  Timeout: {100.0 * float(fila['tasa_timeout']):.2f}%",
            f"  SPL medio: {float(fila['spl_medio']):.4f}",
            f"  Tasa de episodios con casi colisión: "
            f"{100.0 * float(fila['tasa_episodios_con_casi_colision']):.2f}%",
            f"  Eventos de casi colisión/episodio: {float(fila['eventos_casi_colision_media']):.4f}",
            f"  Tiempo de ciclo medio: {float(fila['tiempo_ciclo_medio_ms']):.4f} ms",
            "",
        ])
    lineas.extend([
        "PRUEBA GLOBAL DE ÉXITO",
        "-" * 80,
        f"Q de Cochran: {cochran.get('q_cochran')}",
        f"gl: {cochran.get('grados_libertad')}",
        f"p global: {cochran.get('p_global')}",
        "",
        "COMPARACIONES PAREADAS DE ÉXITO",
        "-" * 80,
    ])
    for fila in comparaciones_exito:
        lineas.append(
            f"{fila['metodo_a']} vs {fila['metodo_b']}: "
            f"diferencia B-A={100.0 * float(fila['diferencia_exito_b_menos_a']):.2f} puntos; "
            f"IC95=[{100.0 * float(fila['ic_bootstrap_95_inferior']):.2f}, "
            f"{100.0 * float(fila['ic_bootstrap_95_superior']):.2f}]; "
            f"McNemar p={float(fila['p_mcnemar_exacto']):.6g}; "
            f"Holm p={float(fila['p_mcnemar_holm']):.6g}"
        )
    lineas.extend([
        "",
        "NOTA METODOLÓGICA",
        "-" * 80,
        "Estas semillas quedan consumidas después de observar los resultados. No deben usarse para ajustar los controladores y volver a presentarse como evaluación confirmatoria.",
    ])
    return "\n".join(lineas) + "\n"
# =============================================================================
# VERIFICACIONES
# =============================================================================
def verificar_huellas_por_semilla(filas: Sequence[Mapping[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    detalles = []
    todo_correcto = True
    for semilla in sorted({int(f["semilla"]) for f in filas}):
        grupo = [f for f in filas if int(f["semilla"]) == semilla]
        huellas_escenario = {str(f["huella_escenario"]) for f in grupo}
        huellas_dinamicos = {str(f["huella_dinamicos_iniciales"]) for f in grupo}
        correcto = len(grupo) == len(METODOS) and len(huellas_escenario) == 1 and len(huellas_dinamicos) == 1
        todo_correcto = todo_correcto and correcto
        detalles.append({
            "semilla": semilla,
            "numero_metodos": len(grupo),
            "escenarios_identicos": int(len(huellas_escenario) == 1),
            "dinamicos_iniciales_identicos": int(len(huellas_dinamicos) == 1),
            "verificacion_correcta": int(correcto),
        })
    return todo_correcto, detalles


def verificar_actores_finales(actores, informacion) -> List[Dict[str, Any]]:
    filas = []
    for metodo, actor in actores.items():
        sin_cambios = actor_sin_cambios(actor, informacion[metodo]["estado_inicial"])
        sin_gradientes = all(parametro.grad is None for parametro in actor.parameters())
        finitos = all(bool(torch.isfinite(parametro).all()) for parametro in actor.parameters())
        filas.append({
            "metodo": metodo,
            "actor_sin_cambios": int(sin_cambios),
            "actor_sin_gradientes": int(sin_gradientes),
            "parametros_finitos": int(finitos),
            "verificacion_correcta": int(sin_cambios and sin_gradientes and finitos),
        })
    return filas


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================


def analizar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparador nominal pareado de DWA, TD3, SAC-R, SAC-PO-TTC y SAC-UPO-TTC."
    )
    parser.add_argument("--semilla-base", type=int, default=SEMILLA_BASE_PREDETERMINADA)
    parser.add_argument("--numero-semillas", type=int, default=NUMERO_SEMILLAS_PREDETERMINADO)
    parser.add_argument("--frecuencia-impresion", type=int, default=FRECUENCIA_IMPRESION_PREDETERMINADA)
    parser.add_argument("--muestras-bootstrap", type=int, default=MUESTRAS_BOOTSTRAP_PREDETERMINADAS)
    parser.add_argument("--directorio-salida", type=str, default="")
    parser.add_argument("--solo-verificacion", action="store_true")
    parser.add_argument("--sin-graficas", action="store_true")
    parser.add_argument(
        "--reiniciar-resultados",
        action="store_true",
        help="Elimina la carpeta de salida antes de ejecutar. No usar después de observar resultados finales salvo que la corrida haya sido técnicamente inválida.",
    )
    return parser.parse_args()


def main() -> None:
    argumentos = analizar_argumentos()
    if argumentos.numero_semillas <= 0:
        raise ValueError("El número de semillas debe ser positivo")
    if argumentos.frecuencia_impresion < 0:
        raise ValueError("La frecuencia de impresión no puede ser negativa")
    if argumentos.muestras_bootstrap <= 0:
        raise ValueError("El número de muestras bootstrap debe ser positivo")

    if argumentos.solo_verificacion:
        argumentos.semilla_base = SEMILLA_VERIFICACION
        argumentos.numero_semillas = 1
        argumentos.frecuencia_impresion = 1
        if not argumentos.directorio_salida:
            argumentos.directorio_salida = "resultados_verificacion_comparador_5_metodos"

    if argumentos.directorio_salida:
        directorio_salida = Path(argumentos.directorio_salida)
    else:
        final_semilla = argumentos.semilla_base + argumentos.numero_semillas - 1
        directorio_salida = Path(
            f"resultados_comparacion_5_metodos_nominal_{argumentos.semilla_base}_{final_semilla}"
        )

    ruta_resultados = directorio_salida / "resultados_por_semilla_y_metodo.csv"
    ruta_configuracion = directorio_salida / "configuracion_experimento.json"

    if argumentos.reiniciar_resultados and directorio_salida.exists():
        shutil.rmtree(directorio_salida)

    directorio_salida.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print("COMPARACIÓN PAREADA DE CINCO MÉTODOS — CONDICIÓN NOMINAL LIMPIA")
    print("=" * 88)
    print("Métodos:")
    for metodo in METODOS:
        print("  -", metodo)
    print(
        f"Semillas: {argumentos.semilla_base} a "
        f"{argumentos.semilla_base + argumentos.numero_semillas - 1}"
    )
    print("Los actores permanecerán congelados; no se ejecutará entrenamiento.")

    base, td3, upo, rutas_programas = preparar_modulos()
    # La verificación técnica usa pocos pasos para comprobar integración,
    # checkpoints, igualdad de escenarios y guardado sin ejecutar episodios
    # completos de 400 pasos. La comparación final conserva los 400 pasos.
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
    actores, info_actores = cargar_todos_los_actores(base, td3, upo)

    print("\nACTORES CARGADOS")
    for metodo, info in info_actores.items():
        print(
            f"{metodo}: episodio={info['episodio']} | parámetros={info['numero_parametros']} | "
            f"checkpoint={info['ruta']}"
        )
    print("DWA: sin checkpoint y sin parámetros neuronales")
    print("Dispositivo:", base.DISPOSITIVO_SAC)
    print("UPO sensor nominal: latencia=0, sigma_posición=0, sigma_velocidad=0")
    print("UPO incertidumbre Monte Carlo interna: activa")

    configuracion_actual = {
        "condicion": "nominal_limpia",
        "semilla_base": int(argumentos.semilla_base),
        "numero_semillas": int(argumentos.numero_semillas),
        "metodos": list(METODOS),
        "umbral_casi_colision_m": UMBRAL_CASI_COLISION_M,
        "umbral_ttc_riesgoso_s": UMBRAL_TTC_RIESGOSO_S,
        "horizonte_calibracion_s": HORIZONTE_CALIBRACION_S,
        "dt": float(base.DT),
        "pasos_maximos": int(base.PASOS_MAXIMOS_SEGUIMIENTO),
        "programas": rutas_programas,
        "checkpoints": {
            metodo: {k: v for k, v in info.items() if k != "estado_inicial"}
            for metodo, info in info_actores.items()
        },
        "upo_sensor": {
            "latencia_maxima_pasos": int(upo.LATENCIA_MAXIMA_PASOS),
            "sigma_posicion_min_m": float(upo.SIGMA_POSICION_MIN_M),
            "sigma_posicion_max_m": float(upo.SIGMA_POSICION_MAX_M),
            "sigma_velocidad_min_m_s": float(upo.SIGMA_VELOCIDAD_MIN_M_S),
            "sigma_velocidad_max_m_s": float(upo.SIGMA_VELOCIDAD_MAX_M_S),
        },
        "upo_modelo_monte_carlo": {
            "numero_muestras": int(upo.NUMERO_MUESTRAS_MONTE_CARLO),
            "sigma_posicion_modelo_m": float(upo.SIGMA_POSICION_MODELO_M),
            "sigma_velocidad_relativa": float(upo.SIGMA_VELOCIDAD_RELATIVA),
            "sigma_direccion_rad": float(upo.SIGMA_DIRECCION_RAD),
            "sigma_aceleracion_m_s2": float(upo.SIGMA_ACELERACION_M_S2),
            "sigma_giro_rad_s": float(upo.SIGMA_GIRO_RAD_S),
        },
    }

    if ruta_configuracion.is_file():
        configuracion_anterior = json.loads(ruta_configuracion.read_text(encoding="utf-8"))
        claves_criticas = (
            "condicion",
            "semilla_base",
            "numero_semillas",
            "metodos",
            "dt",
            "pasos_maximos",
            "checkpoints",
            "upo_sensor",
            "upo_modelo_monte_carlo",
        )
        if any(configuracion_anterior.get(k) != configuracion_actual.get(k) for k in claves_criticas):
            raise RuntimeError(
                "La carpeta de salida contiene una configuración distinta. Use otra carpeta o "
                "--reiniciar-resultados solamente si la ejecución anterior fue técnicamente inválida."
            )
    else:
        ruta_configuracion.write_text(
            json.dumps(configuracion_actual, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    filas_existentes = convertir_tipos_filas(leer_csv_resultados(ruta_resultados))
    claves_completadas = {
        (int(fila["semilla"]), str(fila["metodo"]))
        for fila in filas_existentes
    }
    total_episodios = argumentos.numero_semillas * len(METODOS)
    print(f"Episodios ya guardados: {len(claves_completadas)}/{total_episodios}")

    contador_nuevo = 0
    for indice in range(argumentos.numero_semillas):
        semilla = argumentos.semilla_base + indice
        resultados_semilla: Dict[str, Mapping[str, Any]] = {}

        # Solo se ejecutan los métodos que aún no estén guardados.
        if (semilla, "A* + DWA") not in claves_completadas:
            if argumentos.solo_verificacion:
                print("  Verificando DWA...", flush=True)
            resultados_semilla["A* + DWA"] = ejecutar_dwa_instrumentado(base, semilla)
        if (semilla, "A* + TD3") not in claves_completadas:
            if argumentos.solo_verificacion:
                print("  Verificando TD3...", flush=True)
            resultados_semilla["A* + TD3"] = ejecutar_td3_manual(
                base, td3, actores["A* + TD3"], semilla
            )
        if (semilla, "A* + SAC-R") not in claves_completadas:
            if argumentos.solo_verificacion:
                print("  Verificando SAC-R...", flush=True)
            resultados_semilla["A* + SAC-R"] = ejecutar_sac_manual(
                base, actores["A* + SAC-R"], semilla, False, "A* + SAC-R"
            )
        if (semilla, "A* + SAC-PO-TTC") not in claves_completadas:
            if argumentos.solo_verificacion:
                print("  Verificando SAC-PO-TTC...", flush=True)
            resultados_semilla["A* + SAC-PO-TTC"] = ejecutar_sac_manual(
                base, actores["A* + SAC-PO-TTC"], semilla, True, "A* + SAC-PO-TTC"
            )
        if (semilla, "A* + SAC-UPO-TTC") not in claves_completadas:
            if argumentos.solo_verificacion:
                print("  Verificando SAC-UPO-TTC...", flush=True)
            resultados_semilla["A* + SAC-UPO-TTC"] = ejecutar_sac_manual(
                upo.base,
                actores["A* + SAC-UPO-TTC"],
                semilla,
                True,
                "A* + SAC-UPO-TTC",
            )

        for metodo in METODOS:
            if metodo not in resultados_semilla:
                continue
            numero_parametros = 0 if metodo == "A* + DWA" else info_actores[metodo]["numero_parametros"]
            fila = convertir_resultado_a_fila(
                resultado=resultados_semilla[metodo],
                metodo=metodo,
                numero_parametros=numero_parametros,
                base=upo.base if metodo == "A* + SAC-UPO-TTC" else base,
                upo=upo,
            )
            anexar_fila_csv(ruta_resultados, fila, CAMPOS_RESULTADOS)
            filas_existentes.append(fila)
            claves_completadas.add((semilla, metodo))
            contador_nuevo += 1

        numero_semilla = indice + 1
        imprimir = (
            numero_semilla == 1
            or numero_semilla == argumentos.numero_semillas
            or (
                argumentos.frecuencia_impresion > 0
                and numero_semilla % argumentos.frecuencia_impresion == 0
            )
        )
        if imprimir:
            # Lee de memoria tanto resultados nuevos como reanudados.
            grupo = [f for f in filas_existentes if int(f["semilla"]) == semilla]
            texto_metodos = " | ".join(
                f"{metodo.replace('A* + ', '')}="
                f"{next((str(f['resultado']) for f in grupo if f['metodo'] == metodo), 'pendiente')}"
                for metodo in METODOS
            )
            print(
                f"Escenario {numero_semilla:3d}/{argumentos.numero_semillas:3d} "
                f"| semilla={semilla} | {texto_metodos}"
            )

    filas = convertir_tipos_filas(leer_csv_resultados(ruta_resultados))
    esperado = argumentos.numero_semillas * len(METODOS)
    if len(filas) != esperado:
        raise RuntimeError(
            f"La ejecución quedó incompleta: {len(filas)}/{esperado} episodios guardados. "
            "Vuelva a ejecutar el mismo comando para reanudar."
        )

    huellas_correctas, verificacion_huellas = verificar_huellas_por_semilla(filas)
    guardar_csv(
        directorio_salida / "verificacion_escenarios_identicos.csv",
        verificacion_huellas,
    )
    if not huellas_correctas:
        raise RuntimeError(
            "La verificación detectó escenarios o dinámicos iniciales distintos entre métodos. "
            "Los resultados se conservaron para diagnóstico, pero no deben interpretarse."
        )

    verificaciones_actores = verificar_actores_finales(actores, info_actores)
    guardar_csv(
        directorio_salida / "verificacion_actores_congelados.csv",
        verificaciones_actores,
    )
    if not all(int(f["verificacion_correcta"]) == 1 for f in verificaciones_actores):
        raise RuntimeError("Uno o más actores cambiaron durante la evaluación")

    resumen = resumen_por_metodo(filas)
    comparaciones_exito, cochran, desacuerdos = comparaciones_binarias(
        filas, argumentos.muestras_bootstrap
    )
    globales_continuas, pares_continuos = comparaciones_continuas(filas)

    guardar_csv(directorio_salida / "resumen_por_metodo.csv", resumen)
    guardar_csv(
        directorio_salida / "comparaciones_pareadas_exito.csv",
        comparaciones_exito,
    )
    guardar_csv(directorio_salida / "prueba_global_cochran.csv", [cochran])
    guardar_csv(
        directorio_salida / "semillas_con_desacuerdo_de_exito.csv",
        desacuerdos,
    )
    guardar_csv(
        directorio_salida / "pruebas_globales_metricas_continuas.csv",
        globales_continuas,
    )
    guardar_csv(
        directorio_salida / "comparaciones_pareadas_metricas_continuas.csv",
        pares_continuos,
    )

    metricas_predictivas_filas = [
        fila
        for fila in filas
        if fila["metodo"] in ("A* + SAC-PO-TTC", "A* + SAC-UPO-TTC")
    ]
    guardar_csv(
        directorio_salida / "metricas_predictivas_por_episodio.csv",
        metricas_predictivas_filas,
        CAMPOS_RESULTADOS,
    )

    if not argumentos.sin_graficas:
        crear_graficas(directorio_salida, resumen)

    texto = construir_resumen_texto(
        resumen=resumen,
        cochran=cochran,
        comparaciones_exito=comparaciones_exito,
        info_actores=info_actores,
        argumentos=argumentos,
    )
    (directorio_salida / "resumen_comparacion.txt").write_text(texto, encoding="utf-8")

    print("\n" + "=" * 88)
    print("RESULTADO TÉCNICO DE LA COMPARACIÓN")
    print("=" * 88)
    for fila in resumen:
        print(
            f"{fila['metodo']:<20s} | éxito={100.0 * float(fila['tasa_exito']):6.2f}% "
            f"| col. din={100.0 * float(fila['tasa_colision_dinamica']):6.2f}% "
            f"| casi col.={100.0 * float(fila['tasa_episodios_con_casi_colision']):6.2f}% "
            f"| SPL={float(fila['spl_medio']):.3f}"
        )
    print("Escenarios idénticos:", huellas_correctas)
    print("Actores congelados:", all(int(f["verificacion_correcta"]) == 1 for f in verificaciones_actores))
    print("Resultados:", directorio_salida)
    print("Episodios nuevos ejecutados en esta llamada:", contador_nuevo)
    print("RESULTADO TÉCNICO: COMPARACIÓN DE CINCO MÉTODOS COMPLETADA")
    if argumentos.solo_verificacion:
        print("Esta fue una prueba técnica con una semilla ajena al conjunto final.")
    else:
        print("ADVERTENCIA: las semillas de esta corrida quedan consumidas como evaluación confirmatoria.")

if __name__ == "__main__":
    main()
