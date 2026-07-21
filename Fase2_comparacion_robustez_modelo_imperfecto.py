"""
FASE 2 - COMPARACIÓN PAREADA DE ROBUSTEZ ANTE DISCREPANCIA DEL MODELO.

Evalúa, sin entrenamiento y con controladores congelados:

    1. A* + DWA
    2. A* + SAC reactivo mejorado
    3. A* + SAC predictivo determinista

bajo cinco perfiles reales de movimiento de obstáculos:

    - modelo_conocido
    - aceleracion_suave
    - stop_and_go
    - cambio_direccion
    - curva_sinusoidal

Los controladores predictivos continúan utilizando el modelo nominal original.
Únicamente la actualización REAL del entorno usa los perfiles de Fase 2.

Coloque este archivo en la misma carpeta que:

    Comparacion_final_DWA_SAC_reactivo_SAC_predictivo.py

Ejecución:

    python Fase2_comparacion_robustez_modelo_imperfecto.py

La ejecución es reanudable. Si se interrumpe, vuelva a ejecutar el mismo archivo;
los episodios ya guardados en el CSV parcial se omitirán.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib
import numpy as np

# Compatibilidad con entornos sin interfaz gráfica durante verificación.
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
# CONFIGURACIÓN EXPERIMENTAL
# ==========================================================

PERFILES_DINAMICA_FASE2: Tuple[str, ...] = (
    "modelo_conocido",
    "aceleracion_suave",
    "stop_and_go",
    "cambio_direccion",
    "curva_sinusoidal",
)

ETIQUETAS_PERFILES_FASE2: Dict[str, str] = {
    "modelo_conocido": "Modelo conocido",
    "aceleracion_suave": "Aceleración suave",
    "stop_and_go": "Stop-and-go",
    "cambio_direccion": "Cambio de dirección",
    "curva_sinusoidal": "Curva sinusoidal",
}

SEMILLA_BASE_FASE2 = 92000
NUMERO_SEMILLAS_POR_PERFIL_FASE2 = 100
FRECUENCIA_IMPRESION_FASE2 = 10

# Los mismos escenarios base se repiten entre perfiles para aislar el efecto
# de la ley de movimiento. Dentro de cada perfil, los tres métodos reciben
# exactamente el mismo escenario y los mismos obstáculos iniciales.
REUTILIZAR_MISMAS_SEMILLAS_ENTRE_PERFILES = True

DIRECTORIO_FASE2 = Path(
    "resultados_fase2_robustez/comparacion_modelo_imperfecto"
)

RUTA_CSV_PARCIAL_FASE2 = (
    DIRECTORIO_FASE2 / "resultados_parciales_reanudables.csv"
)
RUTA_CSV_RESULTADOS_FASE2 = (
    DIRECTORIO_FASE2 / "resultados_por_semilla_perfil_metodo.csv"
)
RUTA_CSV_RESUMEN_FASE2 = (
    DIRECTORIO_FASE2 / "resumen_por_perfil_y_metodo.csv"
)
RUTA_CSV_PAREADAS_FASE2 = (
    DIRECTORIO_FASE2 / "comparaciones_pareadas_por_perfil.csv"
)
RUTA_CSV_DESACUERDOS_FASE2 = (
    DIRECTORIO_FASE2 / "semillas_con_desacuerdo_por_perfil.csv"
)
RUTA_CSV_DEGRADACION_FASE2 = (
    DIRECTORIO_FASE2 / "degradacion_respecto_modelo_conocido.csv"
)
RUTA_RESUMEN_TEXTO_FASE2 = (
    DIRECTORIO_FASE2 / "resumen_comparacion_robustez.txt"
)
RUTA_GRAFICA_EXITO_FASE2 = (
    DIRECTORIO_FASE2 / "01_tasa_exito_por_perfil.png"
)
RUTA_GRAFICA_COLISION_FASE2 = (
    DIRECTORIO_FASE2 / "02_colision_dinamica_por_perfil.png"
)
RUTA_GRAFICA_DEGRADACION_FASE2 = (
    DIRECTORIO_FASE2 / "03_degradacion_exito_respecto_modelo_conocido.png"
)
RUTA_GRAFICA_ERROR_MODELO_FASE2 = (
    DIRECTORIO_FASE2 / "04_error_modelo_un_paso.png"
)

MOSTRAR_GRAFICAS_FASE2 = True

METODOS_FASE2: Tuple[str, ...] = tuple(base.METODOS_COMPARACION_FINAL)

RUTA_ACTOR_REACTIVO_FASE2 = Path(
    "resultados_sac/tercera_corrida_reactivo_mejorado/"
    "checkpoint_mejor_actor_reactivo_mejorado.pt"
)
RUTA_ACTOR_PREDICTIVO_FASE2 = Path(
    "resultados_sac/entrenamiento_sac_predictivo/"
    "checkpoint_mejor_actor_sac_predictivo.pt"
)


# ==========================================================
# ESTADO GLOBAL CONTROLADO DE LA FASE 2
# ==========================================================

_ACTUALIZAR_OBSTACULOS_NOMINAL = base.actualizar_obstaculos_dinamicos
_GENERAR_OBSTACULOS_NOMINAL = base.generar_obstaculos_dinamicos

_CONTEXTO_FASE2: Dict[str, object] = {
    "activo": False,
    "perfil": "modelo_conocido",
    "semilla": None,
    "metodo": None,
    "errores_modelo_un_paso": [],
    "llamadas_reales": 0,
    "llamadas_nominales": 0,
    "llamadores_reales": set(),
    "llamadores_nominales": set(),
}

# En estos dos lugares ocurre la transición REAL del entorno. Las llamadas
# realizadas desde los predictores DWA/SAC no aparecen en este conjunto.
_LLAMADORES_ACTUALIZACION_REAL = {
    "simular_seguimiento_dinamico_dwa",
    "ejecutar_paso_entorno_sac",
}


# ==========================================================
# UTILIDADES GENERALES
# ==========================================================

def normalizar_angulo_fase2(angulo: float) -> float:
    return float((float(angulo) + math.pi) % (2.0 * math.pi) - math.pi)


def validar_perfil_fase2(perfil: str) -> str:
    perfil = str(perfil).strip().lower()
    if perfil not in PERFILES_DINAMICA_FASE2:
        raise ValueError(
            f"Perfil no reconocido: {perfil!r}. "
            f"Perfiles válidos: {PERFILES_DINAMICA_FASE2}."
        )
    return perfil


def convertir_json_seguro(valor):
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, tuple):
        return list(valor)
    if isinstance(valor, set):
        return sorted(valor)
    raise TypeError(f"Tipo no serializable: {type(valor).__name__}")


def calcular_hash_objeto(objeto) -> str:
    texto = json.dumps(
        objeto,
        sort_keys=True,
        ensure_ascii=False,
        default=convertir_json_seguro,
        separators=(",", ":"),
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def activar_contexto_fase2(perfil: str, semilla: int, metodo: str) -> None:
    _CONTEXTO_FASE2["activo"] = True
    _CONTEXTO_FASE2["perfil"] = validar_perfil_fase2(perfil)
    _CONTEXTO_FASE2["semilla"] = int(semilla)
    _CONTEXTO_FASE2["metodo"] = str(metodo)
    _CONTEXTO_FASE2["errores_modelo_un_paso"] = []
    _CONTEXTO_FASE2["llamadas_reales"] = 0
    _CONTEXTO_FASE2["llamadas_nominales"] = 0
    _CONTEXTO_FASE2["llamadores_reales"] = set()
    _CONTEXTO_FASE2["llamadores_nominales"] = set()


def desactivar_contexto_fase2() -> None:
    _CONTEXTO_FASE2["activo"] = False
    _CONTEXTO_FASE2["metodo"] = None


def obtener_diagnostico_contexto_fase2() -> Dict[str, object]:
    errores = np.asarray(
        _CONTEXTO_FASE2.get("errores_modelo_un_paso", []),
        dtype=float,
    )
    errores = errores[np.isfinite(errores)]

    return {
        "error_modelo_un_paso_medio_m": (
            float(np.mean(errores)) if len(errores) > 0 else 0.0
        ),
        "error_modelo_un_paso_maximo_m": (
            float(np.max(errores)) if len(errores) > 0 else 0.0
        ),
        "numero_comparaciones_modelo": int(len(errores)),
        "llamadas_actualizacion_real": int(
            _CONTEXTO_FASE2.get("llamadas_reales", 0)
        ),
        "llamadas_actualizacion_nominal": int(
            _CONTEXTO_FASE2.get("llamadas_nominales", 0)
        ),
        "llamadores_reales": "|".join(
            sorted(_CONTEXTO_FASE2.get("llamadores_reales", set()))
        ),
        "llamadores_nominales": "|".join(
            sorted(_CONTEXTO_FASE2.get("llamadores_nominales", set()))
        ),
    }


# ==========================================================
# INICIALIZACIÓN DETERMINISTA DE LOS PERFILES
# ==========================================================

def inicializar_obstaculo_fase2(
    obstaculo: Mapping[str, object],
    perfil: str,
    semilla: int,
    indice_obstaculo: int,
) -> Dict[str, object]:
    perfil = validar_perfil_fase2(perfil)
    actualizado = dict(obstaculo)

    vx = float(actualizado.get("vx", 0.0))
    vy = float(actualizado.get("vy", 0.0))
    velocidad_base = float(math.hypot(vx, vy))
    angulo_base = float(math.atan2(vy, vx)) if velocidad_base > 1e-12 else 0.0

    codigo_perfil = PERFILES_DINAMICA_FASE2.index(perfil)
    semilla_local = (
        int(semilla)
        + 1009 * int(indice_obstaculo)
        + 7919 * int(codigo_perfil)
    )
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
    obstaculos: Sequence[Mapping[str, object]],
    perfil: str,
    semilla: int,
) -> List[Dict[str, object]]:
    return [
        inicializar_obstaculo_fase2(
            obstaculo=obstaculo,
            perfil=perfil,
            semilla=semilla,
            indice_obstaculo=indice,
        )
        for indice, obstaculo in enumerate(obstaculos)
    ]


def generar_obstaculos_dinamicos_despachador_fase2(*args, **kwargs):
    obstaculos = _GENERAR_OBSTACULOS_NOMINAL(*args, **kwargs)

    if not bool(_CONTEXTO_FASE2.get("activo", False)):
        return obstaculos

    if "semilla" in kwargs:
        semilla_dinamica = int(kwargs["semilla"])
    else:
        # La función original recibe la semilla como sexto argumento.
        semilla_dinamica = int(args[5])

    return inicializar_obstaculos_fase2(
        obstaculos=obstaculos,
        perfil=str(_CONTEXTO_FASE2["perfil"]),
        semilla=semilla_dinamica,
    )


# ==========================================================
# MOVIMIENTO REAL NO CONOCIDO POR LOS PREDICTORES
# ==========================================================

def calcular_velocidad_real_fase2(
    obstaculo: Mapping[str, object],
    tiempo_actual: float,
) -> Tuple[float, float, Dict[str, object]]:
    actualizado = dict(obstaculo)
    perfil = validar_perfil_fase2(
        str(actualizado.get("perfil_fase2", "modelo_conocido"))
    )

    tiempo_inicio = float(actualizado.get("tiempo_inicio", 0.0))
    tiempo_activo = max(0.0, float(tiempo_actual) - tiempo_inicio)

    velocidad_base = float(actualizado.get("velocidad_base_fase2", 0.0))
    angulo_base = float(actualizado.get("angulo_base_fase2", 0.0))
    fase = float(actualizado.get("fase_fase2", 0.0))
    periodo = max(float(actualizado.get("periodo_fase2", 4.0)), 1e-6)

    if float(tiempo_actual) < tiempo_inicio:
        return (
            float(actualizado.get("vx", 0.0)),
            float(actualizado.get("vy", 0.0)),
            actualizado,
        )

    if perfil == "modelo_conocido":
        return (
            float(actualizado.get("vx", 0.0)),
            float(actualizado.get("vy", 0.0)),
            actualizado,
        )

    velocidad = velocidad_base
    angulo = angulo_base

    if perfil == "aceleracion_suave":
        factor = 1.0 + 0.45 * math.sin(
            2.0 * math.pi * tiempo_activo / periodo + fase
        )
        velocidad = velocidad_base * factor

    elif perfil == "stop_and_go":
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
    angulo = normalizar_angulo_fase2(angulo)

    return (
        float(velocidad * math.cos(angulo)),
        float(velocidad * math.sin(angulo)),
        actualizado,
    )


def actualizar_obstaculos_dinamicos_reales_fase2(
    obstaculos_dinamicos: Sequence[Mapping[str, object]],
    obstaculos_estaticos: Sequence[Mapping[str, object]],
    tiempo_actual: float,
    dt: float = base.DT,
) -> List[Dict[str, object]]:
    preparados: List[Dict[str, object]] = []

    for obstaculo in obstaculos_dinamicos:
        vx, vy, preparado = calcular_velocidad_real_fase2(
            obstaculo=obstaculo,
            tiempo_actual=tiempo_actual,
        )
        preparado["vx"] = float(vx)
        preparado["vy"] = float(vy)
        preparados.append(preparado)

    actualizados = _ACTUALIZAR_OBSTACULOS_NOMINAL(
        obstaculos_dinamicos=preparados,
        obstaculos_estaticos=obstaculos_estaticos,
        tiempo_actual=float(tiempo_actual),
        dt=float(dt),
    )

    # Adaptar la dirección base después de un rebote geométrico.
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


def calcular_errores_posicion_obstaculos(
    nominales: Sequence[Mapping[str, object]],
    reales: Sequence[Mapping[str, object]],
) -> List[float]:
    errores: List[float] = []
    for nominal, real in zip(nominales, reales):
        error = math.hypot(
            float(real["x"]) - float(nominal["x"]),
            float(real["y"]) - float(nominal["y"]),
        )
        if np.isfinite(error):
            errores.append(float(error))
    return errores


def actualizar_obstaculos_dinamicos_despachador_fase2(
    obstaculos_dinamicos,
    obstaculos_estaticos,
    tiempo_actual,
    dt=base.DT,
):
    marco = inspect.currentframe()
    llamador = "desconocido"
    if marco is not None and marco.f_back is not None:
        llamador = str(marco.f_back.f_code.co_name)

    activo = bool(_CONTEXTO_FASE2.get("activo", False))
    es_actualizacion_real = activo and llamador in _LLAMADORES_ACTUALIZACION_REAL

    if not es_actualizacion_real:
        if activo:
            _CONTEXTO_FASE2["llamadas_nominales"] = int(
                _CONTEXTO_FASE2.get("llamadas_nominales", 0)
            ) + 1
            _CONTEXTO_FASE2["llamadores_nominales"].add(llamador)

        return _ACTUALIZAR_OBSTACULOS_NOMINAL(
            obstaculos_dinamicos=obstaculos_dinamicos,
            obstaculos_estaticos=obstaculos_estaticos,
            tiempo_actual=tiempo_actual,
            dt=dt,
        )

    _CONTEXTO_FASE2["llamadas_reales"] = int(
        _CONTEXTO_FASE2.get("llamadas_reales", 0)
    ) + 1
    _CONTEXTO_FASE2["llamadores_reales"].add(llamador)

    nominales = _ACTUALIZAR_OBSTACULOS_NOMINAL(
        obstaculos_dinamicos=obstaculos_dinamicos,
        obstaculos_estaticos=obstaculos_estaticos,
        tiempo_actual=tiempo_actual,
        dt=dt,
    )

    reales = actualizar_obstaculos_dinamicos_reales_fase2(
        obstaculos_dinamicos=obstaculos_dinamicos,
        obstaculos_estaticos=obstaculos_estaticos,
        tiempo_actual=tiempo_actual,
        dt=dt,
    )

    _CONTEXTO_FASE2["errores_modelo_un_paso"].extend(
        calcular_errores_posicion_obstaculos(nominales, reales)
    )

    return reales


# ==========================================================
# INSTALACIÓN Y RESTAURACIÓN DE LOS DESPACHADORES
# ==========================================================

def instalar_despachadores_fase2() -> None:
    base.generar_obstaculos_dinamicos = (
        generar_obstaculos_dinamicos_despachador_fase2
    )
    base.actualizar_obstaculos_dinamicos = (
        actualizar_obstaculos_dinamicos_despachador_fase2
    )


def restaurar_funciones_nominales() -> None:
    base.generar_obstaculos_dinamicos = _GENERAR_OBSTACULOS_NOMINAL
    base.actualizar_obstaculos_dinamicos = _ACTUALIZAR_OBSTACULOS_NOMINAL


# ==========================================================
# GUARDADO REANUDABLE
# ==========================================================

CAMPOS_ENTEROS_RESULTADOS = {
    "semilla",
    "exito",
    "colision_estatica",
    "colision_dinamica",
    "fuera_mapa",
    "timeout",
    "pasos_ejecutados",
    "registro_consistente",
    "numero_comparaciones_modelo",
    "llamadas_actualizacion_real",
    "llamadas_actualizacion_nominal",
}

CAMPOS_FLOTANTES_RESULTADOS = {
    "tiempo_navegacion_s",
    "longitud_recorrida_m",
    "longitud_astar_m",
    "distancia_final_meta_m",
    "eficiencia_navegacion",
    "exceso_longitud_porcentual",
    "error_medio_ruta_m",
    "error_rmse_ruta_m",
    "error_maximo_ruta_m",
    "clearance_estatico_minimo_m",
    "clearance_estatico_promedio_m",
    "clearance_dinamico_minimo_m",
    "clearance_dinamico_promedio_m",
    "clearance_total_minimo_m",
    "clearance_total_promedio_m",
    "variacion_total_v",
    "variacion_total_omega",
    "aceleracion_lineal_rms_m_s2",
    "aceleracion_angular_rms_rad_s2",
    "esfuerzo_control",
    "error_modelo_un_paso_medio_m",
    "error_modelo_un_paso_maximo_m",
}


def ordenar_resultados_fase2(resultados: Iterable[Mapping[str, object]]):
    orden_perfil = {perfil: indice for indice, perfil in enumerate(PERFILES_DINAMICA_FASE2)}
    orden_metodo = {metodo: indice for indice, metodo in enumerate(METODOS_FASE2)}
    return sorted(
        [dict(fila) for fila in resultados],
        key=lambda fila: (
            orden_perfil[str(fila["perfil_dinamico"])],
            int(fila["semilla"]),
            orden_metodo[str(fila["metodo"])],
        ),
    )


def guardar_csv_diccionarios_atomico(
    ruta: Path,
    filas: Sequence[Mapping[str, object]],
) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    filas = [dict(fila) for fila in filas]
    if len(filas) == 0:
        ruta.write_text("", encoding="utf-8")
        return

    campos: List[str] = []
    for fila in filas:
        for clave in fila.keys():
            if clave not in campos:
                campos.append(clave)

    temporal = ruta.with_suffix(ruta.suffix + ".temporal")
    with temporal.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({campo: fila.get(campo, "") for campo in campos})

    temporal.replace(ruta)


def cargar_resultados_parciales_fase2() -> List[Dict[str, object]]:
    if not RUTA_CSV_PARCIAL_FASE2.is_file():
        return []

    resultados: List[Dict[str, object]] = []
    with RUTA_CSV_PARCIAL_FASE2.open("r", newline="", encoding="utf-8") as archivo:
        for fila_csv in csv.DictReader(archivo):
            fila: Dict[str, object] = dict(fila_csv)
            for campo in CAMPOS_ENTEROS_RESULTADOS:
                if campo in fila and str(fila[campo]).strip() != "":
                    fila[campo] = int(float(str(fila[campo])))
            for campo in CAMPOS_FLOTANTES_RESULTADOS:
                if campo in fila and str(fila[campo]).strip() != "":
                    fila[campo] = float(str(fila[campo]))
            resultados.append(fila)

    return ordenar_resultados_fase2(resultados)


def clave_resultado_fase2(fila: Mapping[str, object]) -> Tuple[str, int, str]:
    return (
        str(fila["perfil_dinamico"]),
        int(fila["semilla"]),
        str(fila["metodo"]),
    )


# ==========================================================
# EJECUCIÓN DE UN MÉTODO
# ==========================================================

def construir_hashes_episodio(resultado: Mapping[str, object]) -> Tuple[str, str]:
    escenario = resultado["escenario"]
    obstaculos_dinamicos = resultado["obstaculos_dinamicos_iniciales"]

    objeto_escenario = {
        "inicio": escenario["inicio"],
        "meta": escenario["meta"],
        "obstaculos": escenario["obstaculos"],
        "camino_nodos": escenario["camino_nodos"],
    }

    return (
        calcular_hash_objeto(objeto_escenario),
        calcular_hash_objeto(obstaculos_dinamicos),
    )


def ejecutar_metodo_fase2(
    perfil: str,
    semilla: int,
    metodo: str,
    actor_reactivo,
    actor_predictivo,
) -> Dict[str, object]:
    activar_contexto_fase2(perfil=perfil, semilla=semilla, metodo=metodo)

    try:
        if metodo == "A* + DWA":
            resultado = base.ejecutar_episodio_dwa_estandar(
                semilla=semilla,
                pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                dt=base.DT,
            )

        elif metodo == "A* + SAC reactivo mejorado":
            resultado = base.evaluar_episodio_sac(
                semilla=semilla,
                actor=actor_reactivo,
                pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                dt=base.DT,
                dispositivo=base.DISPOSITIVO_SAC,
            )

        elif metodo == "A* + SAC predictivo":
            resultado = base.evaluar_episodio_sac_predictivo(
                semilla=semilla,
                actor=actor_predictivo,
                pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
                dt=base.DT,
                dispositivo=base.DISPOSITIVO_SAC,
            )

        else:
            raise ValueError(f"Método no reconocido: {metodo!r}.")

        fila = base.convertir_metricas_a_fila_comparacion_final(
            semilla=semilla,
            metodo=metodo,
            metricas=resultado["metricas"],
        )

        hash_escenario, hash_dinamicos = construir_hashes_episodio(resultado)
        diagnostico = obtener_diagnostico_contexto_fase2()

        fila.update(
            {
                "perfil_dinamico": perfil,
                "hash_escenario": hash_escenario,
                "hash_obstaculos_dinamicos_iniciales": hash_dinamicos,
                **diagnostico,
            }
        )
        return fila

    finally:
        desactivar_contexto_fase2()


# ==========================================================
# RESÚMENES Y COMPARACIONES
# ==========================================================

def calcular_resumen_por_perfil_fase2(
    resultados: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    resumen_total: List[Dict[str, object]] = []

    for perfil in PERFILES_DINAMICA_FASE2:
        filas_perfil = [
            dict(fila)
            for fila in resultados
            if fila["perfil_dinamico"] == perfil
        ]
        resumen_perfil = base.calcular_resumen_metodos_comparacion_final(
            filas_perfil
        )

        for fila in resumen_perfil:
            filas_metodo = [
                registro
                for registro in filas_perfil
                if registro["metodo"] == fila["metodo"]
            ]
            fila["perfil_dinamico"] = perfil
            fila["error_modelo_un_paso_medio_m"] = float(
                np.mean(
                    [
                        float(registro["error_modelo_un_paso_medio_m"])
                        for registro in filas_metodo
                    ]
                )
            )
            fila["error_modelo_un_paso_maximo_m"] = float(
                np.max(
                    [
                        float(registro["error_modelo_un_paso_maximo_m"])
                        for registro in filas_metodo
                    ]
                )
            )
            resumen_total.append(fila)

    return resumen_total


def calcular_pareadas_por_perfil_fase2(
    resultados: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    comparaciones_totales: List[Dict[str, object]] = []
    desacuerdos_totales: List[Dict[str, object]] = []

    for perfil in PERFILES_DINAMICA_FASE2:
        filas_perfil = [
            dict(fila)
            for fila in resultados
            if fila["perfil_dinamico"] == perfil
        ]
        comparaciones, desacuerdos = (
            base.calcular_comparaciones_pareadas_finales(filas_perfil)
        )

        for fila in comparaciones:
            fila["perfil_dinamico"] = perfil
            comparaciones_totales.append(fila)

        for fila in desacuerdos:
            fila["perfil_dinamico"] = perfil
            desacuerdos_totales.append(fila)

    return comparaciones_totales, desacuerdos_totales


def calcular_degradacion_respecto_modelo_conocido(
    resultados: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    degradaciones: List[Dict[str, object]] = []

    for metodo in METODOS_FASE2:
        filas_base = {
            int(fila["semilla"]): fila
            for fila in resultados
            if fila["perfil_dinamico"] == "modelo_conocido"
            and fila["metodo"] == metodo
        }

        for indice_perfil, perfil in enumerate(PERFILES_DINAMICA_FASE2[1:], start=1):
            filas_perfil = {
                int(fila["semilla"]): fila
                for fila in resultados
                if fila["perfil_dinamico"] == perfil
                and fila["metodo"] == metodo
            }

            semillas = sorted(set(filas_base) & set(filas_perfil))
            if len(semillas) == 0:
                continue

            exito_base = np.asarray(
                [int(filas_base[s]["exito"]) for s in semillas],
                dtype=float,
            )
            exito_perfil = np.asarray(
                [int(filas_perfil[s]["exito"]) for s in semillas],
                dtype=float,
            )
            diferencias = exito_perfil - exito_base

            ic_inf, ic_sup = base.bootstrap_diferencia_pareada_comparacion_final(
                diferencias,
                numero_remuestreos=10000,
                semilla=654321 + indice_perfil,
            )

            base_exito_perfil_fallo = int(
                np.sum((exito_base == 1.0) & (exito_perfil == 0.0))
            )
            base_fallo_perfil_exito = int(
                np.sum((exito_base == 0.0) & (exito_perfil == 1.0))
            )
            p_mcnemar = base.probabilidad_binomial_exacta_dos_colas_comparacion_final(
                base_exito_perfil_fallo,
                base_fallo_perfil_exito,
            )

            def tasa(campo: str, tabla: Mapping[int, Mapping[str, object]]) -> float:
                return float(np.mean([int(tabla[s][campo]) for s in semillas]))

            degradaciones.append(
                {
                    "metodo": metodo,
                    "perfil_referencia": "modelo_conocido",
                    "perfil_perturbado": perfil,
                    "numero_semillas_pareadas": len(semillas),
                    "tasa_exito_referencia": float(np.mean(exito_base)),
                    "tasa_exito_perturbado": float(np.mean(exito_perfil)),
                    "diferencia_exito_perturbado_menos_referencia": float(
                        np.mean(diferencias)
                    ),
                    "ic_bootstrap_95_inferior": ic_inf,
                    "ic_bootstrap_95_superior": ic_sup,
                    "referencia_exito_perturbado_fallo": base_exito_perfil_fallo,
                    "referencia_fallo_perturbado_exito": base_fallo_perfil_exito,
                    "p_mcnemar_exacto": p_mcnemar,
                    "diferencia_colision_dinamica": (
                        tasa("colision_dinamica", filas_perfil)
                        - tasa("colision_dinamica", filas_base)
                    ),
                    "diferencia_colision_estatica": (
                        tasa("colision_estatica", filas_perfil)
                        - tasa("colision_estatica", filas_base)
                    ),
                    "diferencia_timeout": (
                        tasa("timeout", filas_perfil)
                        - tasa("timeout", filas_base)
                    ),
                }
            )

    return degradaciones


# ==========================================================
# GRÁFICAS
# ==========================================================

def obtener_fila_resumen(
    resumen: Sequence[Mapping[str, object]],
    perfil: str,
    metodo: str,
) -> Mapping[str, object]:
    coincidencias = [
        fila
        for fila in resumen
        if fila["perfil_dinamico"] == perfil and fila["metodo"] == metodo
    ]
    if len(coincidencias) != 1:
        raise RuntimeError(
            f"Se esperaba una fila para {perfil!r}, {metodo!r}; "
            f"se encontraron {len(coincidencias)}."
        )
    return coincidencias[0]


def crear_grafica_tasas_fase2(
    resumen: Sequence[Mapping[str, object]],
    clave: str,
    ylabel: str,
    titulo: str,
    ruta: Path,
) -> None:
    posiciones = np.arange(len(PERFILES_DINAMICA_FASE2), dtype=float)
    ancho = 0.25

    figura, eje = plt.subplots(figsize=(13, 6))

    for indice_metodo, metodo in enumerate(METODOS_FASE2):
        valores = [
            100.0 * float(obtener_fila_resumen(resumen, perfil, metodo)[clave])
            for perfil in PERFILES_DINAMICA_FASE2
        ]
        desplazamiento = (indice_metodo - 1) * ancho
        barras = eje.bar(
            posiciones + desplazamiento,
            valores,
            width=ancho,
            label=metodo,
        )
        for barra, valor in zip(barras, valores):
            eje.text(
                barra.get_x() + barra.get_width() / 2.0,
                barra.get_height() + 0.5,
                f"{valor:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    eje.set_xticks(posiciones)
    eje.set_xticklabels(
        [ETIQUETAS_PERFILES_FASE2[p] for p in PERFILES_DINAMICA_FASE2],
        rotation=15,
        ha="right",
    )
    eje.set_ylabel(ylabel)
    eje.set_title(titulo)
    eje.grid(True, axis="y", alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(figura)


def crear_grafica_degradacion_fase2(
    degradaciones: Sequence[Mapping[str, object]],
) -> None:
    perfiles = list(PERFILES_DINAMICA_FASE2[1:])
    posiciones = np.arange(len(perfiles), dtype=float)
    ancho = 0.25

    figura, eje = plt.subplots(figsize=(12, 6))

    for indice_metodo, metodo in enumerate(METODOS_FASE2):
        valores = []
        for perfil in perfiles:
            fila = next(
                registro
                for registro in degradaciones
                if registro["metodo"] == metodo
                and registro["perfil_perturbado"] == perfil
            )
            valores.append(
                100.0 * float(
                    fila["diferencia_exito_perturbado_menos_referencia"]
                )
            )

        desplazamiento = (indice_metodo - 1) * ancho
        eje.bar(
            posiciones + desplazamiento,
            valores,
            width=ancho,
            label=metodo,
        )

    eje.axhline(0.0, linestyle="--", linewidth=1.2)
    eje.set_xticks(posiciones)
    eje.set_xticklabels(
        [ETIQUETAS_PERFILES_FASE2[p] for p in perfiles],
        rotation=15,
        ha="right",
    )
    eje.set_ylabel("Cambio de éxito [puntos porcentuales]")
    eje.set_title("Degradación pareada respecto al modelo conocido")
    eje.grid(True, axis="y", alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(
        RUTA_GRAFICA_DEGRADACION_FASE2,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)


def crear_grafica_error_modelo_fase2(
    resumen: Sequence[Mapping[str, object]],
) -> None:
    # El error del movimiento real depende del perfil, no del controlador;
    # se promedian las tres filas de método para mostrar una sola barra.
    valores = []
    for perfil in PERFILES_DINAMICA_FASE2:
        valores_perfil = [
            float(fila["error_modelo_un_paso_medio_m"])
            for fila in resumen
            if fila["perfil_dinamico"] == perfil
        ]
        valores.append(float(np.mean(valores_perfil)))

    posiciones = np.arange(len(PERFILES_DINAMICA_FASE2))
    figura, eje = plt.subplots(figsize=(11, 5.5))
    barras = eje.bar(posiciones, valores)

    for barra, valor in zip(barras, valores):
        eje.text(
            barra.get_x() + barra.get_width() / 2.0,
            barra.get_height(),
            f"{valor:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    eje.set_xticks(posiciones)
    eje.set_xticklabels(
        [ETIQUETAS_PERFILES_FASE2[p] for p in PERFILES_DINAMICA_FASE2],
        rotation=15,
        ha="right",
    )
    eje.set_ylabel("Error nominal-real de un paso [m]")
    eje.set_title("Discrepancia efectiva del modelo durante la evaluación")
    eje.grid(True, axis="y", alpha=0.3)
    figura.tight_layout()
    figura.savefig(
        RUTA_GRAFICA_ERROR_MODELO_FASE2,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)


def crear_graficas_fase2(
    resumen: Sequence[Mapping[str, object]],
    degradaciones: Sequence[Mapping[str, object]],
) -> None:
    DIRECTORIO_FASE2.mkdir(parents=True, exist_ok=True)

    crear_grafica_tasas_fase2(
        resumen=resumen,
        clave="tasa_exito",
        ylabel="Tasa de éxito [%]",
        titulo="Fase 2: éxito bajo discrepancia del modelo",
        ruta=RUTA_GRAFICA_EXITO_FASE2,
    )
    crear_grafica_tasas_fase2(
        resumen=resumen,
        clave="tasa_colision_dinamica",
        ylabel="Colisión dinámica [%]",
        titulo="Fase 2: colisiones dinámicas bajo discrepancia del modelo",
        ruta=RUTA_GRAFICA_COLISION_FASE2,
    )
    crear_grafica_degradacion_fase2(degradaciones)
    crear_grafica_error_modelo_fase2(resumen)

    if MOSTRAR_GRAFICAS_FASE2:
        # Se muestran después de haber guardado y cerrado las figuras. Esta
        # llamada se conserva para mantener el flujo de trabajo del proyecto.
        plt.show()


# ==========================================================
# INFORME TEXTUAL
# ==========================================================

def construir_resumen_texto_fase2(
    resumen: Sequence[Mapping[str, object]],
    comparaciones: Sequence[Mapping[str, object]],
    degradaciones: Sequence[Mapping[str, object]],
    informacion_reactivo: Mapping[str, object],
    informacion_predictivo: Mapping[str, object],
) -> str:
    lineas = [
        "FASE 2 - COMPARACIÓN DE ROBUSTEZ ANTE MODELO IMPERFECTO",
        "=" * 80,
        "",
        "Entrenamiento durante la evaluación: False",
        "Actores modificados: False",
        f"Semillas por perfil: {NUMERO_SEMILLAS_POR_PERFIL_FASE2}",
        f"Rango de semillas: {SEMILLA_BASE_FASE2}-"
        f"{SEMILLA_BASE_FASE2 + NUMERO_SEMILLAS_POR_PERFIL_FASE2 - 1}",
        "Semillas finales 70000-70099 reutilizadas: False",
        f"Actor reactivo: {informacion_reactivo['ruta_checkpoint']}",
        f"Actor predictivo: {informacion_predictivo['ruta_checkpoint']}",
        "",
    ]

    for perfil in PERFILES_DINAMICA_FASE2:
        lineas.extend(
            [
                ETIQUETAS_PERFILES_FASE2[perfil],
                "-" * 80,
            ]
        )
        for metodo in METODOS_FASE2:
            fila = obtener_fila_resumen(resumen, perfil, metodo)
            lineas.extend(
                [
                    metodo,
                    f"  Éxito: {int(fila['exitos'])}/{int(fila['episodios'])} "
                    f"({100.0 * float(fila['tasa_exito']):.2f}%)",
                    f"  Colisión dinámica: "
                    f"{100.0 * float(fila['tasa_colision_dinamica']):.2f}%",
                    f"  Colisión estática: "
                    f"{100.0 * float(fila['tasa_colision_estatica']):.2f}%",
                    f"  Timeout: {100.0 * float(fila['tasa_timeout']):.2f}%",
                    f"  Error nominal-real medio por paso: "
                    f"{float(fila['error_modelo_un_paso_medio_m']):.6f} m",
                ]
            )

        lineas.append("")
        lineas.append("Comparaciones pareadas de éxito")
        for fila in comparaciones:
            if fila["perfil_dinamico"] != perfil:
                continue
            lineas.extend(
                [
                    f"  {fila['metodo_a']} vs {fila['metodo_b']}",
                    f"    Diferencia B-A: "
                    f"{100.0 * float(fila['diferencia_exito_b_menos_a']):.2f} puntos",
                    f"    McNemar p: {float(fila['p_mcnemar_exacto']):.6f}",
                ]
            )
        lineas.append("")

    lineas.extend(
        [
            "DEGRADACIÓN RESPECTO AL MODELO CONOCIDO",
            "=" * 80,
            "",
        ]
    )

    for fila in degradaciones:
        lineas.extend(
            [
                f"{fila['metodo']} | {fila['perfil_perturbado']}",
                f"  Cambio de éxito: "
                f"{100.0 * float(fila['diferencia_exito_perturbado_menos_referencia']):.2f} puntos",
                f"  IC bootstrap 95%: "
                f"[{100.0 * float(fila['ic_bootstrap_95_inferior']):.2f}, "
                f"{100.0 * float(fila['ic_bootstrap_95_superior']):.2f}]",
                f"  McNemar p: {float(fila['p_mcnemar_exacto']):.6f}",
                f"  Cambio de colisión dinámica: "
                f"{100.0 * float(fila['diferencia_colision_dinamica']):.2f} puntos",
                "",
            ]
        )

    return "\n".join(lineas) + "\n"


# ==========================================================
# VERIFICACIONES TÉCNICAS
# ==========================================================

def verificar_igualdad_escenarios_resultados(
    resultados: Sequence[Mapping[str, object]],
) -> Tuple[bool, bool]:
    escenarios_correctos = True
    dinamicos_correctos = True

    por_perfil_semilla: MutableMapping[Tuple[str, int], List[Mapping[str, object]]] = defaultdict(list)
    for fila in resultados:
        por_perfil_semilla[
            (str(fila["perfil_dinamico"]), int(fila["semilla"]))
        ].append(fila)

    for filas in por_perfil_semilla.values():
        escenarios_correctos = escenarios_correctos and (
            len({fila["hash_escenario"] for fila in filas}) == 1
        )
        dinamicos_correctos = dinamicos_correctos and (
            len({fila["hash_obstaculos_dinamicos_iniciales"] for fila in filas}) == 1
        )

    return bool(escenarios_correctos), bool(dinamicos_correctos)


def verificar_separacion_modelo_real(
    resultados: Sequence[Mapping[str, object]],
) -> Tuple[bool, bool, bool]:
    modelo_conocido_cero = all(
        float(fila["error_modelo_un_paso_maximo_m"]) <= 1e-9
        for fila in resultados
        if fila["perfil_dinamico"] == "modelo_conocido"
    )

    perfiles_discrepantes = all(
        any(
            float(fila["error_modelo_un_paso_maximo_m"]) > 1e-6
            for fila in resultados
            if fila["perfil_dinamico"] == perfil
        )
        for perfil in PERFILES_DINAMICA_FASE2[1:]
    )

    llamadores_reales_correctos = all(
        set(str(fila["llamadores_reales"]).split("|"))
        <= _LLAMADORES_ACTUALIZACION_REAL
        for fila in resultados
        if str(fila["llamadores_reales"]).strip() != ""
    )

    return (
        bool(modelo_conocido_cero),
        bool(perfiles_discrepantes),
        bool(llamadores_reales_correctos),
    )


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("FASE 2: COMPARACIÓN DE ROBUSTEZ ANTE MODELO IMPERFECTO")
    print("=" * 80)

    print("\nMETODOLOGÍA")
    print("Controladores congelados: DWA, SAC reactivo y SAC predictivo")
    print("Entrenamiento: False")
    print(
        "Semillas:",
        f"{SEMILLA_BASE_FASE2}-"
        f"{SEMILLA_BASE_FASE2 + NUMERO_SEMILLAS_POR_PERFIL_FASE2 - 1}",
    )
    print("Perfiles:", ", ".join(PERFILES_DINAMICA_FASE2))
    print("Semillas 70000-70099 reutilizadas: False")
    print("Ejecución reanudable: True")

    DIRECTORIO_FASE2.mkdir(parents=True, exist_ok=True)

    actor_reactivo, informacion_reactivo = base.cargar_actor_comparacion_final(
        ruta_checkpoint=RUTA_ACTOR_REACTIVO_FASE2,
        variante_esperada="reactivo",
        dispositivo=base.DISPOSITIVO_SAC,
    )
    actor_predictivo, informacion_predictivo = base.cargar_actor_comparacion_final(
        ruta_checkpoint=RUTA_ACTOR_PREDICTIVO_FASE2,
        variante_esperada="predictivo",
        dispositivo=base.DISPOSITIVO_SAC,
    )

    print("\nACTORES")
    print(
        "Reactivo:", informacion_reactivo["ruta_checkpoint"],
        "| episodio:", informacion_reactivo["episodio"],
    )
    print(
        "Predictivo:", informacion_predictivo["ruta_checkpoint"],
        "| episodio:", informacion_predictivo["episodio"],
    )
    print("Dispositivo:", base.DISPOSITIVO_SAC)

    resultados = cargar_resultados_parciales_fase2()
    resultados_por_clave = {
        clave_resultado_fase2(fila): dict(fila)
        for fila in resultados
    }

    print("\nResultados recuperados:", len(resultados_por_clave))
    total_esperado = (
        len(PERFILES_DINAMICA_FASE2)
        * NUMERO_SEMILLAS_POR_PERFIL_FASE2
        * len(METODOS_FASE2)
    )
    print("Total esperado:", total_esperado)

    instalar_despachadores_fase2()

    try:
        for indice_perfil, perfil in enumerate(PERFILES_DINAMICA_FASE2):
            print("\n" + "-" * 80)
            print("PERFIL:", ETIQUETAS_PERFILES_FASE2[perfil])
            print("-" * 80)

            for indice_semilla in range(NUMERO_SEMILLAS_POR_PERFIL_FASE2):
                if REUTILIZAR_MISMAS_SEMILLAS_ENTRE_PERFILES:
                    semilla = SEMILLA_BASE_FASE2 + indice_semilla
                else:
                    semilla = (
                        SEMILLA_BASE_FASE2
                        + indice_perfil * NUMERO_SEMILLAS_POR_PERFIL_FASE2
                        + indice_semilla
                    )

                resultados_semilla: Dict[str, str] = {}

                for metodo in METODOS_FASE2:
                    clave = (perfil, semilla, metodo)

                    if clave in resultados_por_clave:
                        fila = resultados_por_clave[clave]
                        resultados_semilla[metodo] = str(fila["resultado"])
                        continue

                    fila = ejecutar_metodo_fase2(
                        perfil=perfil,
                        semilla=semilla,
                        metodo=metodo,
                        actor_reactivo=actor_reactivo,
                        actor_predictivo=actor_predictivo,
                    )
                    resultados_por_clave[clave] = fila
                    resultados_semilla[metodo] = str(fila["resultado"])

                    guardar_csv_diccionarios_atomico(
                        RUTA_CSV_PARCIAL_FASE2,
                        ordenar_resultados_fase2(resultados_por_clave.values()),
                    )

                if (
                    indice_semilla == 0
                    or (indice_semilla + 1) % FRECUENCIA_IMPRESION_FASE2 == 0
                    or indice_semilla + 1 == NUMERO_SEMILLAS_POR_PERFIL_FASE2
                ):
                    print(
                        f"Escenario {indice_semilla + 1:3d}/"
                        f"{NUMERO_SEMILLAS_POR_PERFIL_FASE2} "
                        f"({semilla}) | "
                        f"DWA={resultados_semilla.get('A* + DWA', 'guardado'):<18} | "
                        f"Reactivo={resultados_semilla.get('A* + SAC reactivo mejorado', 'guardado'):<18} | "
                        f"Predictivo={resultados_semilla.get('A* + SAC predictivo', 'guardado'):<18}"
                    )

    finally:
        desactivar_contexto_fase2()
        restaurar_funciones_nominales()

    resultados = ordenar_resultados_fase2(resultados_por_clave.values())

    if len(resultados) != total_esperado:
        raise RuntimeError(
            f"La evaluación quedó incompleta: {len(resultados)}/{total_esperado} filas. "
            "Vuelva a ejecutar el archivo para reanudar."
        )

    resumen = calcular_resumen_por_perfil_fase2(resultados)
    comparaciones, desacuerdos = calcular_pareadas_por_perfil_fase2(resultados)
    degradaciones = calcular_degradacion_respecto_modelo_conocido(resultados)

    guardar_csv_diccionarios_atomico(RUTA_CSV_RESULTADOS_FASE2, resultados)
    guardar_csv_diccionarios_atomico(RUTA_CSV_RESUMEN_FASE2, resumen)
    guardar_csv_diccionarios_atomico(RUTA_CSV_PAREADAS_FASE2, comparaciones)
    guardar_csv_diccionarios_atomico(RUTA_CSV_DESACUERDOS_FASE2, desacuerdos)
    guardar_csv_diccionarios_atomico(RUTA_CSV_DEGRADACION_FASE2, degradaciones)

    texto_resumen = construir_resumen_texto_fase2(
        resumen=resumen,
        comparaciones=comparaciones,
        degradaciones=degradaciones,
        informacion_reactivo=informacion_reactivo,
        informacion_predictivo=informacion_predictivo,
    )
    RUTA_RESUMEN_TEXTO_FASE2.write_text(texto_resumen, encoding="utf-8")

    crear_graficas_fase2(resumen=resumen, degradaciones=degradaciones)

    reactivo_sin_cambios, reactivo_sin_gradientes = (
        base.verificar_actor_sin_cambios_comparacion_final(
            actor_reactivo,
            informacion_reactivo["estado_inicial"],
        )
    )
    predictivo_sin_cambios, predictivo_sin_gradientes = (
        base.verificar_actor_sin_cambios_comparacion_final(
            actor_predictivo,
            informacion_predictivo["estado_inicial"],
        )
    )

    escenarios_identicos, dinamicos_identicos = (
        verificar_igualdad_escenarios_resultados(resultados)
    )
    (
        modelo_conocido_coincide,
        perfiles_discrepantes,
        llamadores_reales_correctos,
    ) = verificar_separacion_modelo_real(resultados)

    filas_correctas = len(resultados) == total_esperado
    combinaciones_unicas = len(
        {clave_resultado_fase2(fila) for fila in resultados}
    ) == total_esperado
    registros_consistentes = all(
        int(fila["registro_consistente"]) == 1
        for fila in resultados
    )

    archivos_correctos = all(
        ruta.is_file()
        for ruta in [
            RUTA_CSV_PARCIAL_FASE2,
            RUTA_CSV_RESULTADOS_FASE2,
            RUTA_CSV_RESUMEN_FASE2,
            RUTA_CSV_PAREADAS_FASE2,
            RUTA_CSV_DESACUERDOS_FASE2,
            RUTA_CSV_DEGRADACION_FASE2,
            RUTA_RESUMEN_TEXTO_FASE2,
            RUTA_GRAFICA_EXITO_FASE2,
            RUTA_GRAFICA_COLISION_FASE2,
            RUTA_GRAFICA_DEGRADACION_FASE2,
            RUTA_GRAFICA_ERROR_MODELO_FASE2,
        ]
    )

    verificaciones = {
        "carga_reactiva_estricta": bool(informacion_reactivo["carga_estricta"]),
        "carga_predictiva_estricta": bool(informacion_predictivo["carga_estricta"]),
        "escenarios_estaticos_identicos": escenarios_identicos,
        "obstaculos_iniciales_identicos": dinamicos_identicos,
        "modelo_conocido_coincide": modelo_conocido_coincide,
        "perfiles_no_vistos_difieren": perfiles_discrepantes,
        "actualizacion_real_separada": llamadores_reales_correctos,
        "numero_filas_correcto": filas_correctas,
        "combinaciones_unicas": combinaciones_unicas,
        "registros_consistentes": registros_consistentes,
        "actor_reactivo_sin_cambios": reactivo_sin_cambios,
        "actor_predictivo_sin_cambios": predictivo_sin_cambios,
        "actor_reactivo_sin_gradientes": reactivo_sin_gradientes,
        "actor_predictivo_sin_gradientes": predictivo_sin_gradientes,
        "archivos_correctos": archivos_correctos,
    }

    todo_correcto = all(verificaciones.values())

    print("\n" + "=" * 80)
    print("RESULTADO DE LA FASE 2: ROBUSTEZ ANTE MODELO IMPERFECTO")
    print("=" * 80)

    for perfil in PERFILES_DINAMICA_FASE2:
        print("\n" + ETIQUETAS_PERFILES_FASE2[perfil])
        for metodo in METODOS_FASE2:
            fila = obtener_fila_resumen(resumen, perfil, metodo)
            print(
                f"{metodo}: "
                f"éxito={100.0 * float(fila['tasa_exito']):6.2f}% | "
                f"col. dinámica={100.0 * float(fila['tasa_colision_dinamica']):6.2f}% | "
                f"col. estática={100.0 * float(fila['tasa_colision_estatica']):6.2f}% | "
                f"timeout={100.0 * float(fila['tasa_timeout']):6.2f}%"
            )

    print("\nDEGRADACIÓN DE ÉXITO RESPECTO AL MODELO CONOCIDO")
    for fila in degradaciones:
        print(
            f"{fila['metodo']} | {fila['perfil_perturbado']}: "
            f"{100.0 * float(fila['diferencia_exito_perturbado_menos_referencia']):+.2f} puntos | "
            f"McNemar p={float(fila['p_mcnemar_exacto']):.6f}"
        )

    print("\nARCHIVOS")
    print("Resultados:", RUTA_CSV_RESULTADOS_FASE2)
    print("Resumen:", RUTA_CSV_RESUMEN_FASE2)
    print("Pareadas:", RUTA_CSV_PAREADAS_FASE2)
    print("Degradación:", RUTA_CSV_DEGRADACION_FASE2)
    print("Informe:", RUTA_RESUMEN_TEXTO_FASE2)
    print("Gráfica de éxito:", RUTA_GRAFICA_EXITO_FASE2)
    print("Gráfica de colisión:", RUTA_GRAFICA_COLISION_FASE2)
    print("Gráfica de degradación:", RUTA_GRAFICA_DEGRADACION_FASE2)
    print("Gráfica de error del modelo:", RUTA_GRAFICA_ERROR_MODELO_FASE2)

    print("\nVERIFICACIONES TÉCNICAS")
    for nombre, valor in verificaciones.items():
        print(f"{nombre}: {valor}")

    if todo_correcto:
        print("\nRESULTADO TÉCNICO: FASE 2 COMPLETADA")
    else:
        print("\nRESULTADO TÉCNICO: HAY VERIFICACIONES POR CORREGIR")
        raise RuntimeError("La Fase 2 terminó con verificaciones fallidas.")

    print("=" * 80)


if __name__ == "__main__":
    main()
