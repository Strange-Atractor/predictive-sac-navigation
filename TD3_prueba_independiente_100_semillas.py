from __future__ import annotations

"""
Prueba independiente del TD3 estabilizado entrenado desde cero.

Evalúa sin entrenamiento el checkpoint del episodio seleccionado por validación.
Requiere, en la misma carpeta:
    TD3_tercera_corrida_desde_cero.py
    SAC_predictivo_entrenamiento_completo.py

y el checkpoint:
    resultados_td3/tercera_corrida_td3_desde_cero/
    checkpoint_mejor_actor_td3.pt
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
CANDIDATOS_TD3 = (
    "TD3_tercera_corrida_desde_cero.py",
    "TD3_tercera_corrida_desde_cero(1).py",
    "TD3_entrenamiento_desde_cero.py",
)

RUTA_CHECKPOINT = Path(
    "resultados_td3/tercera_corrida_td3_desde_cero/"
    "checkpoint_mejor_actor_td3.pt"
)

SEMILLA_BASE_PREDETERMINADA = 168000
NUMERO_SEMILLAS_PREDETERMINADO = 100
FRECUENCIA_IMPRESION = 10

DIRECTORIO_RESULTADOS = Path("resultados_td3/prueba_independiente_td3_desde_cero")
RUTA_CSV = DIRECTORIO_RESULTADOS / "resultados_100_semillas.csv"
RUTA_CSV_FALLOS = DIRECTORIO_RESULTADOS / "semillas_fallidas.csv"
RUTA_RESUMEN = DIRECTORIO_RESULTADOS / "resumen_prueba_independiente.txt"
RUTA_GRAFICA = DIRECTORIO_RESULTADOS / "distribucion_resultados_100_semillas.png"

# Criterio general, idéntico al aplicado a los SAC.
CRITERIO_GENERAL_EXITO_MINIMO = 0.85
CRITERIO_GENERAL_COLISION_DINAMICA_MAXIMA = 0.10
CRITERIO_GENERAL_COLISION_ESTATICA_MAXIMA = 0.10
CRITERIO_GENERAL_FUERA_MAPA_MAXIMA = 0.05
CRITERIO_GENERAL_TIMEOUT_MAXIMO = 0.05

# Umbral mínimo previamente fijado para considerar TD3 un baseline utilizable.
CRITERIO_BASELINE_EXITO_MINIMO = 0.70
CRITERIO_BASELINE_COLISION_DINAMICA_MAXIMA = 0.20
CRITERIO_BASELINE_COLISION_ESTATICA_MAXIMA = 0.10
CRITERIO_BASELINE_FUERA_MAPA_MAXIMA = 0.05
CRITERIO_BASELINE_TIMEOUT_MAXIMO = 0.05


# ==========================================================
# IMPORTACIÓN SEGURA
# ==========================================================

def localizar_programa_td3() -> Path:
    for nombre in CANDIDATOS_TD3:
        ruta = DIRECTORIO_ACTUAL / nombre
        if ruta.is_file():
            return ruta
    candidatos = "\n".join(f"  - {nombre}" for nombre in CANDIDATOS_TD3)
    raise FileNotFoundError(
        "No se encontró la tercera corrida TD3 desde cero. Coloque este archivo "
        "junto a uno de estos programas:\n" + candidatos
    )


def importar_programa_td3():
    ruta = localizar_programa_td3()
    especificacion = importlib.util.spec_from_file_location(
        "td3_desde_cero_para_prueba_independiente",
        ruta,
    )
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No fue posible importar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo, ruta


# ==========================================================
# UTILIDADES
# ==========================================================

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


def promedio_finito(valores: Sequence[float]) -> float:
    arreglo = np.asarray(valores, dtype=float)
    arreglo = arreglo[np.isfinite(arreglo)]
    return float(np.mean(arreglo)) if arreglo.size else float("nan")


def guardar_csv(ruta: Path, filas: Sequence[Mapping[str, object]]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        ruta.write_text("semilla,resultado\n", encoding="utf-8")
        return
    campos = list(filas[0].keys())
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def crear_grafica(conteos: Mapping[str, int], tasa_exito: float) -> None:
    etiquetas = ["Meta", "Colisión\nestática", "Colisión\ndinámica", "Fuera del\nmapa", "Timeout"]
    claves = ["meta", "colision_estatica", "colision_dinamica", "fuera_mapa", "timeout"]
    valores = [int(conteos.get(clave, 0)) for clave in claves]
    figura, eje = plt.subplots(figsize=(10, 6))
    barras = eje.bar(etiquetas, valores)
    eje.set_title(f"Prueba independiente TD3 desde cero\nÉxito: {100.0 * tasa_exito:.2f}%")
    eje.set_ylabel("Número de episodios")
    eje.set_ylim(0, max(valores + [1]) + 10)
    eje.grid(axis="y", alpha=0.25)
    for barra, valor in zip(barras, valores):
        eje.text(
            barra.get_x() + barra.get_width() / 2.0,
            barra.get_height() + 0.8,
            str(valor),
            ha="center",
            va="bottom",
        )
    figura.tight_layout()
    RUTA_GRAFICA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(RUTA_GRAFICA, dpi=300, bbox_inches="tight")
    plt.close(figura)


def analizar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba independiente del TD3 entrenado desde cero."
    )
    parser.add_argument("--semilla-base", type=int, default=SEMILLA_BASE_PREDETERMINADA)
    parser.add_argument("--numero-semillas", type=int, default=NUMERO_SEMILLAS_PREDETERMINADO)
    parser.add_argument("--frecuencia-impresion", type=int, default=FRECUENCIA_IMPRESION)
    return parser.parse_args()


# ==========================================================
# EVALUACIÓN DETALLADA
# ==========================================================

def ejecutar_episodio_detallado(
    td3,
    actor,
    semilla: int,
    dispositivo: torch.device,
) -> Dict[str, object]:
    base = td3.base
    entorno, observacion = base.reiniciar_entorno_sac(int(semilla))

    estado_inicial = tuple(float(v) for v in entorno["estado_robot"])
    estados = [estado_inicial]
    acciones: List[np.ndarray] = []
    velocidades_lineales: List[float] = []
    velocidades_angulares: List[float] = []
    clearances_estaticos: List[float] = []
    clearances_dinamicos: List[float] = []
    clearances_totales: List[float] = []
    tiempos_inferencia: List[float] = []

    terminado = False
    truncado = False
    recompensa_total = 0.0

    while not (terminado or truncado):
        inicio_inferencia = time.perf_counter()
        accion = td3.accion_td3(
            actor=actor,
            observacion=observacion,
            dispositivo=dispositivo,
            ruido=0.0,
        )
        tiempos_inferencia.append(time.perf_counter() - inicio_inferencia)

        observacion, recompensa, terminado, truncado, informacion = (
            base.ejecutar_paso_entorno_sac(entorno, accion)
        )
        recompensa_total += float(recompensa)
        acciones.append(accion.copy())
        estados.append(tuple(float(v) for v in entorno["estado_robot"]))

        velocidades_lineales.append(float(informacion.get("velocidad_lineal", 0.0)))
        velocidades_angulares.append(float(informacion.get("velocidad_angular", 0.0)))
        clearances_estaticos.append(float(informacion.get("clearance_estatico", float("inf"))))
        clearances_dinamicos.append(float(informacion.get("clearance_dinamico", float("inf"))))
        clearances_totales.append(float(informacion.get("clearance_total", float("inf"))))

    posiciones = np.asarray([[e[0], e[1]] for e in estados], dtype=float)
    diferencias = np.diff(posiciones, axis=0)
    longitud = float(np.sum(np.linalg.norm(diferencias, axis=1))) if diferencias.size else 0.0

    camino = entorno["camino_mundo"]
    errores_ruta = [
        float(base.distancia_punto_camino((estado[0], estado[1]), camino))
        for estado in estados
    ]
    error_medio_ruta = promedio_finito(errores_ruta)

    estado_final = entorno["estado_robot"]
    distancia_final = float(
        base.distancia_entre_puntos(
            (float(estado_final[0]), float(estado_final[1])),
            entorno["meta"],
        )
    )

    acciones_arreglo = np.asarray(acciones, dtype=float)
    esfuerzo_control = (
        float(np.mean(np.sum(acciones_arreglo ** 2, axis=1)))
        if acciones_arreglo.size
        else 0.0
    )
    cambios_signo_omega = 0
    if len(velocidades_angulares) >= 2:
        omega = np.asarray(velocidades_angulares, dtype=float)
        signos = np.sign(omega)
        cambios_signo_omega = int(np.sum(signos[1:] * signos[:-1] < 0.0))

    return {
        "semilla": int(semilla),
        "resultado": str(entorno["resultado"]),
        "exito": int(entorno["resultado"] == "meta"),
        "colision_estatica": int(entorno["resultado"] == "colision_estatica"),
        "colision_dinamica": int(entorno["resultado"] == "colision_dinamica"),
        "fuera_mapa": int(entorno["resultado"] == "fuera_mapa"),
        "timeout": int(entorno["resultado"] == "timeout"),
        "pasos": int(entorno["paso_actual"]),
        "tiempo_simulado_s": float(entorno["paso_actual"] * base.DT),
        "recompensa": float(recompensa_total),
        "distancia_final_m": distancia_final,
        "longitud_trayectoria_m": longitud,
        "error_medio_ruta_m": error_medio_ruta,
        "clearance_minimo_m": promedio_finito([np.min(clearances_totales)]) if clearances_totales else float("inf"),
        "clearance_estatico_minimo_m": promedio_finito([np.min(clearances_estaticos)]) if clearances_estaticos else float("inf"),
        "clearance_dinamico_minimo_m": promedio_finito([np.min(clearances_dinamicos)]) if clearances_dinamicos else float("inf"),
        "velocidad_lineal_media_m_s": promedio_finito(velocidades_lineales),
        "velocidad_angular_abs_media_rad_s": promedio_finito(np.abs(velocidades_angulares)),
        "esfuerzo_control_medio": esfuerzo_control,
        "cambios_signo_omega": cambios_signo_omega,
        "inferencia_media_ms": 1000.0 * promedio_finito(tiempos_inferencia),
        "inferencia_p95_ms": 1000.0 * float(np.percentile(tiempos_inferencia, 95.0)) if tiempos_inferencia else float("nan"),
        "inferencia_p99_ms": 1000.0 * float(np.percentile(tiempos_inferencia, 99.0)) if tiempos_inferencia else float("nan"),
    }


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def main() -> None:
    argumentos = analizar_argumentos()
    if argumentos.numero_semillas <= 0:
        raise ValueError("El número de semillas debe ser positivo.")
    if argumentos.frecuencia_impresion < 0:
        raise ValueError("La frecuencia de impresión no puede ser negativa.")

    td3, ruta_programa = importar_programa_td3()
    base = td3.base
    dispositivo = base.DISPOSITIVO_SAC

    print("\n" + "=" * 80)
    print("PRUEBA INDEPENDIENTE DEL TD3 ENTRENADO DESDE CERO")
    print("=" * 80)
    print("Programa TD3:", ruta_programa.name)

    if not RUTA_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "No se encontró el mejor checkpoint TD3 desde cero:\n"
            f"{RUTA_CHECKPOINT}"
        )

    checkpoint = torch.load(RUTA_CHECKPOINT, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint or "episodio" not in checkpoint:
        raise KeyError("El checkpoint no contiene actor_state_dict y episodio.")

    variante_correcta = checkpoint.get("variante") == "td3_reactivo_tercera_corrida_desde_cero"
    actor_desde_cero = (
        checkpoint.get("configuracion", {}).get("actor_inicial")
        == "inicializacion_aleatoria_desde_cero"
    )
    checkpoint_correcto = bool(
        checkpoint.get("tipo") == "mejor_actor_td3"
        and variante_correcta
        and actor_desde_cero
    )
    if not checkpoint_correcto:
        raise ValueError("El checkpoint no corresponde al TD3 oficial entrenado desde cero.")

    actor = td3.crear_actor_td3(dispositivo)
    resultado_carga = actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    carga_estricta = (
        len(resultado_carga.missing_keys) == 0
        and len(resultado_carga.unexpected_keys) == 0
    )
    if not carga_estricta:
        raise RuntimeError("No fue posible cargar estrictamente el actor TD3.")

    actor.eval()
    episodio_actor = int(checkpoint["episodio"])
    parametros_antes = {
        nombre: valor.detach().cpu().clone()
        for nombre, valor in actor.state_dict().items()
    }
    parametros_finitos = all(
        bool(torch.isfinite(parametro).all()) for parametro in actor.parameters()
    )

    semilla_base = int(argumentos.semilla_base)
    numero_semillas = int(argumentos.numero_semillas)

    print("\n1. Actor evaluado")
    print("Checkpoint:", RUTA_CHECKPOINT)
    print("Episodio:", episodio_actor)
    print("Dispositivo:", dispositivo)
    print("Número de parámetros:", sum(p.numel() for p in actor.parameters()))
    print("Política determinista: True")
    print("Entrenamiento durante la prueba: False")
    print("Actor entrenado desde cero: True")

    print("\n2. Semillas independientes")
    print("Número:", numero_semillas)
    print("Rango:", semilla_base, "a", semilla_base + numero_semillas - 1)

    print("\n3. Comienza la evaluación")
    resultados: List[Dict[str, object]] = []
    exitos_acumulados = 0
    for indice in range(numero_semillas):
        semilla = semilla_base + indice
        fila = ejecutar_episodio_detallado(td3, actor, semilla, dispositivo)
        resultados.append(fila)
        exitos_acumulados += int(fila["exito"])
        numero = indice + 1
        if (
            numero == 1
            or numero == numero_semillas
            or (argumentos.frecuencia_impresion > 0 and numero % argumentos.frecuencia_impresion == 0)
        ):
            print(
                f"Validación {numero:3d}/{numero_semillas:3d} | "
                f"semilla={semilla:7d} | "
                f"resultado={str(fila['resultado']):18s} | "
                f"éxito acumulado={100.0 * exitos_acumulados / numero:6.2f}% | "
                f"pasos={int(fila['pasos']):4d}"
            )

    conteos = {
        clave: sum(int(fila[clave]) for fila in resultados)
        for clave in ("exito", "colision_estatica", "colision_dinamica", "fuera_mapa", "timeout")
    }
    conteos_resultados = {
        "meta": conteos["exito"],
        "colision_estatica": conteos["colision_estatica"],
        "colision_dinamica": conteos["colision_dinamica"],
        "fuera_mapa": conteos["fuera_mapa"],
        "timeout": conteos["timeout"],
    }

    tasa_exito = conteos["exito"] / numero_semillas
    tasa_colision_estatica = conteos["colision_estatica"] / numero_semillas
    tasa_colision_dinamica = conteos["colision_dinamica"] / numero_semillas
    tasa_fuera_mapa = conteos["fuera_mapa"] / numero_semillas
    tasa_timeout = conteos["timeout"] / numero_semillas
    ic_inf, ic_sup = intervalo_wilson(conteos["exito"], numero_semillas)

    exitos = [fila for fila in resultados if int(fila["exito"]) == 1]
    fallos = [fila for fila in resultados if int(fila["exito"]) == 0]
    semillas_fallidas = [int(fila["semilla"]) for fila in fallos]

    criterios_generales = {
        "exito": tasa_exito >= CRITERIO_GENERAL_EXITO_MINIMO,
        "colision_dinamica": tasa_colision_dinamica <= CRITERIO_GENERAL_COLISION_DINAMICA_MAXIMA,
        "colision_estatica": tasa_colision_estatica <= CRITERIO_GENERAL_COLISION_ESTATICA_MAXIMA,
        "fuera_mapa": tasa_fuera_mapa <= CRITERIO_GENERAL_FUERA_MAPA_MAXIMA,
        "timeout": tasa_timeout <= CRITERIO_GENERAL_TIMEOUT_MAXIMO,
    }
    criterios_generales["todos"] = all(criterios_generales.values())

    criterios_baseline = {
        "exito": tasa_exito >= CRITERIO_BASELINE_EXITO_MINIMO,
        "colision_dinamica": tasa_colision_dinamica <= CRITERIO_BASELINE_COLISION_DINAMICA_MAXIMA,
        "colision_estatica": tasa_colision_estatica <= CRITERIO_BASELINE_COLISION_ESTATICA_MAXIMA,
        "fuera_mapa": tasa_fuera_mapa <= CRITERIO_BASELINE_FUERA_MAPA_MAXIMA,
        "timeout": tasa_timeout <= CRITERIO_BASELINE_TIMEOUT_MAXIMO,
    }
    criterios_baseline["todos"] = all(criterios_baseline.values())

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    guardar_csv(RUTA_CSV, resultados)
    guardar_csv(RUTA_CSV_FALLOS, fallos)
    crear_grafica(conteos_resultados, tasa_exito)

    resumen = {
        "recompensa_media": promedio_finito([float(f["recompensa"]) for f in resultados]),
        "pasos_promedio": promedio_finito([float(f["pasos"]) for f in resultados]),
        "distancia_final_media": promedio_finito([float(f["distancia_final_m"]) for f in resultados]),
        "error_ruta_medio": promedio_finito([float(f["error_medio_ruta_m"]) for f in resultados]),
        "clearance_minimo_promedio": promedio_finito([float(f["clearance_minimo_m"]) for f in resultados]),
        "clearance_minimo_absoluto": float(np.min([float(f["clearance_minimo_m"]) for f in resultados])),
        "tiempo_promedio_exitos": promedio_finito([float(f["tiempo_simulado_s"]) for f in exitos]),
        "longitud_promedio_exitos": promedio_finito([float(f["longitud_trayectoria_m"]) for f in exitos]),
        "clearance_promedio_exitos": promedio_finito([float(f["clearance_minimo_m"]) for f in exitos]),
        "inferencia_media_ms": promedio_finito([float(f["inferencia_media_ms"]) for f in resultados]),
        "inferencia_p95_episodios_ms": promedio_finito([float(f["inferencia_p95_ms"]) for f in resultados]),
    }

    texto = "\n".join([
        "PRUEBA INDEPENDIENTE TD3 DESDE CERO",
        "=" * 72,
        f"Checkpoint: {RUTA_CHECKPOINT}",
        f"Episodio: {episodio_actor}",
        f"Semillas: {semilla_base} a {semilla_base + numero_semillas - 1}",
        f"Éxitos: {conteos['exito']}/{numero_semillas}",
        f"Tasa de éxito: {100.0 * tasa_exito:.2f}%",
        f"IC Wilson 95%: [{100.0 * ic_inf:.2f}%, {100.0 * ic_sup:.2f}%]",
        f"Conteos: {conteos_resultados}",
        f"Colisión dinámica: {100.0 * tasa_colision_dinamica:.2f}%",
        f"Colisión estática: {100.0 * tasa_colision_estatica:.2f}%",
        f"Fuera del mapa: {100.0 * tasa_fuera_mapa:.2f}%",
        f"Timeout: {100.0 * tasa_timeout:.2f}%",
        f"Métricas: {resumen}",
        f"Criterios generales: {criterios_generales}",
        f"Criterios mínimos de baseline: {criterios_baseline}",
        f"Semillas fallidas: {semillas_fallidas}",
    ]) + "\n"
    RUTA_RESUMEN.write_text(texto, encoding="utf-8")

    parametros_despues = {
        nombre: valor.detach().cpu().clone()
        for nombre, valor in actor.state_dict().items()
    }
    actor_sin_cambios = all(
        torch.equal(parametros_antes[nombre], parametros_despues[nombre])
        for nombre in parametros_antes
    )
    actor_sin_gradientes = all(parametro.grad is None for parametro in actor.parameters())
    numero_resultados_correcto = len(resultados) == numero_semillas
    conteos_completos = sum(conteos_resultados.values()) == numero_semillas
    semillas_correctas = bool(
        resultados
        and int(resultados[0]["semilla"]) == semilla_base
        and int(resultados[-1]["semilla"]) == semilla_base + numero_semillas - 1
    )
    archivos_correctos = all(
        ruta.is_file()
        for ruta in (RUTA_CSV, RUTA_CSV_FALLOS, RUTA_RESUMEN, RUTA_GRAFICA)
    )

    print("\n" + "=" * 80)
    print("RESULTADO DE LA PRUEBA INDEPENDIENTE TD3")
    print("=" * 80)
    print("\nRESULTADOS GENERALES")
    print("Éxitos:", f"{conteos['exito']}/{numero_semillas}")
    print("Tasa de éxito:", f"{100.0 * tasa_exito:.2f}%")
    print("IC Wilson 95%:", f"[{100.0 * ic_inf:.2f}%, {100.0 * ic_sup:.2f}%]")
    print("Conteos:", conteos_resultados)
    print("Colisión estática:", f"{100.0 * tasa_colision_estatica:.2f}%")
    print("Colisión dinámica:", f"{100.0 * tasa_colision_dinamica:.2f}%")
    print("Fuera del mapa:", f"{100.0 * tasa_fuera_mapa:.2f}%")
    print("Timeout:", f"{100.0 * tasa_timeout:.2f}%")

    print("\nMÉTRICAS")
    for clave, valor in resumen.items():
        print(f"{clave}: {valor}")

    print("\nCRITERIOS GENERALES (IGUALES A LOS SAC)")
    print("Éxito >= 85%:", criterios_generales["exito"])
    print("Colisión dinámica <= 10%:", criterios_generales["colision_dinamica"])
    print("Colisión estática <= 10%:", criterios_generales["colision_estatica"])
    print("Fuera del mapa <= 5%:", criterios_generales["fuera_mapa"])
    print("Timeout <= 5%:", criterios_generales["timeout"])
    print("TODOS LOS CRITERIOS GENERALES:", criterios_generales["todos"])

    print("\nCRITERIOS MÍNIMOS PARA BASELINE TD3")
    print("Éxito >= 70%:", criterios_baseline["exito"])
    print("Colisión dinámica <= 20%:", criterios_baseline["colision_dinamica"])
    print("Colisión estática <= 10%:", criterios_baseline["colision_estatica"])
    print("Fuera del mapa <= 5%:", criterios_baseline["fuera_mapa"])
    print("Timeout <= 5%:", criterios_baseline["timeout"])
    print("TODOS LOS CRITERIOS DE BASELINE:", criterios_baseline["todos"])

    print("\nSEMILLAS FALLIDAS")
    print(semillas_fallidas)

    print("\nARCHIVOS")
    print("CSV completo:", RUTA_CSV)
    print("CSV de fallos:", RUTA_CSV_FALLOS)
    print("Resumen:", RUTA_RESUMEN)
    print("Gráfica:", RUTA_GRAFICA)

    print("\nVERIFICACIONES TÉCNICAS")
    print("Checkpoint correcto:", checkpoint_correcto)
    print("Carga estricta:", carga_estricta)
    print("Parámetros finitos:", parametros_finitos)
    print("Número de resultados correcto:", numero_resultados_correcto)
    print("Conteos completos:", conteos_completos)
    print("Semillas correctas:", semillas_correctas)
    print("Archivos correctos:", archivos_correctos)
    print("Actor sin cambios:", actor_sin_cambios)
    print("Actor sin gradientes:", actor_sin_gradientes)

    tecnicamente_correcto = all([
        checkpoint_correcto,
        carga_estricta,
        parametros_finitos,
        numero_resultados_correcto,
        conteos_completos,
        semillas_correctas,
        archivos_correctos,
        actor_sin_cambios,
        actor_sin_gradientes,
    ])
    print(
        "\nRESULTADO TÉCNICO:",
        "PRUEBA INDEPENDIENTE COMPLETADA"
        if tecnicamente_correcto
        else "HAY ALGO POR CORREGIR",
    )
    print(
        "RESULTADO CIENTÍFICO:",
        "TD3 APTO COMO BASELINE"
        if criterios_baseline["todos"]
        else "TD3 AÚN NO APTO COMO BASELINE",
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
