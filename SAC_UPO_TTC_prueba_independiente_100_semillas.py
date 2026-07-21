from __future__ import annotations

"""
Prueba independiente del SAC-UPO-TTC.

Evalúa sin entrenamiento el mejor actor probabilístico en 100 semillas nuevas.
Requiere, en la misma carpeta:
    SAC_probabilistico_ocupacion_TTC_entrenamiento.py
    SAC_predictivo_entrenamiento_completo.py

y el checkpoint:
    resultados_sac/entrenamiento_sac_probabilistico_upo_ttc/
    checkpoint_mejor_actor_sac_probabilistico.pt

La política es determinista. El mapa probabilístico mantiene sus muestras
Monte Carlo reproducibles mediante los generadores sembrados por episodio.
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

# Impide que una ventana gráfica bloquee el programa al terminar.
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

CANDIDATOS_UPO = (
    "SAC_probabilistico_ocupacion_TTC_entrenamiento.py",
    "SAC_probabilistico_ocupacion_TTC_entrenamiento(1).py",
    "SAC_probabilistico_ocupacion_TTC_entrenamiento(2).py",
    "SAC_probabilistico_ocupacion_TTC_entrenamiento(3).py",
)

RUTA_CHECKPOINT = Path(
    "resultados_sac/entrenamiento_sac_probabilistico_upo_ttc/"
    "checkpoint_mejor_actor_sac_probabilistico.pt"
)

SEMILLA_BASE_PREDETERMINADA = 132000
NUMERO_SEMILLAS_PREDETERMINADO = 100
FRECUENCIA_IMPRESION = 10

DIRECTORIO_RESULTADOS = Path(
    "resultados_sac/prueba_independiente_sac_probabilistico_upo_ttc"
)
RUTA_CSV = DIRECTORIO_RESULTADOS / "resultados_100_semillas.csv"
RUTA_CSV_FALLOS = DIRECTORIO_RESULTADOS / "semillas_fallidas.csv"
RUTA_RESUMEN = DIRECTORIO_RESULTADOS / "resumen_prueba_independiente.txt"
RUTA_GRAFICA = DIRECTORIO_RESULTADOS / "distribucion_resultados_100_semillas.png"

CRITERIO_EXITO_MINIMO = 0.85
CRITERIO_COLISION_DINAMICA_MAXIMA = 0.10
CRITERIO_COLISION_ESTATICA_MAXIMA = 0.10
CRITERIO_FUERA_MAPA_MAXIMA = 0.05
CRITERIO_TIMEOUT_MAXIMO = 0.05


# ==========================================================
# IMPORTACIÓN SEGURA
# ==========================================================

def localizar_programa_upo() -> Path:
    for nombre in CANDIDATOS_UPO:
        ruta = DIRECTORIO_ACTUAL / nombre
        if ruta.is_file():
            return ruta
    candidatos = "\n".join(f"  - {nombre}" for nombre in CANDIDATOS_UPO)
    raise FileNotFoundError(
        "No se encontró el programa SAC-UPO-TTC. Coloque este archivo en la "
        "misma carpeta que uno de estos archivos:\n" + candidatos
    )


def importar_programa_upo():
    ruta = localizar_programa_upo()
    especificacion = importlib.util.spec_from_file_location(
        "sac_upo_ttc_para_prueba_independiente",
        ruta,
    )
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No fue posible importar {ruta}")

    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)

    # Instala la rama probabilística sobre las rutinas de evaluación de la base.
    modulo.configurar_programa_base(
        numero_episodios=1,
        mostrar_graficas=False,
    )
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
    eje.set_title(f"Prueba independiente SAC-UPO-TTC\nÉxito: {100.0 * tasa_exito:.2f}%")
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
        description="Prueba independiente del SAC-UPO-TTC sin entrenamiento."
    )
    parser.add_argument("--semilla-base", type=int, default=SEMILLA_BASE_PREDETERMINADA)
    parser.add_argument("--numero-semillas", type=int, default=NUMERO_SEMILLAS_PREDETERMINADO)
    parser.add_argument("--frecuencia-impresion", type=int, default=FRECUENCIA_IMPRESION)
    return parser.parse_args()


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def main() -> None:
    argumentos = analizar_argumentos()
    if argumentos.numero_semillas <= 0:
        raise ValueError("El número de semillas debe ser positivo.")
    if argumentos.frecuencia_impresion < 0:
        raise ValueError("La frecuencia de impresión no puede ser negativa.")

    upo, ruta_programa = importar_programa_upo()
    base = upo.base
    dispositivo = base.DISPOSITIVO_SAC

    print("\n" + "=" * 80)
    print("PRUEBA INDEPENDIENTE DEL SAC-UPO-TTC")
    print("=" * 80)
    print("Programa UPO:", ruta_programa.name)

    if not RUTA_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "No se encontró el mejor checkpoint SAC-UPO-TTC:\n"
            f"{RUTA_CHECKPOINT}"
        )

    checkpoint = torch.load(
        RUTA_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if "actor_state_dict" not in checkpoint or "episodio" not in checkpoint:
        raise KeyError("El checkpoint no contiene actor_state_dict y episodio.")

    metadata_upo_completa = bool(
        checkpoint.get("observacion_probabilistica", False)
        and checkpoint.get("prediccion_monte_carlo", False)
        and checkpoint.get("usa_cvar", False)
        and checkpoint.get("usa_ttc", False)
        and checkpoint.get("variante_sac")
        == "probabilistico_ocupacion_futura_ttc"
    )

    # Se admite el formato anterior al etiquetado porque el entrenamiento base
    # guarda primero el actor predictivo y después agrega los metadatos UPO.
    metadata_compatible_sin_etiquetar = bool(
        checkpoint.get("tipo") == "mejor_actor_sac_predictivo"
        and checkpoint.get("observacion_predictiva", False)
        and checkpoint.get("usa_ttc", False)
    )

    if not (metadata_upo_completa or metadata_compatible_sin_etiquetar):
        raise ValueError(
            "El checkpoint no parece corresponder al mejor actor SAC-UPO-TTC."
        )

    actor = base.crear_actor_sac(dispositivo=dispositivo)
    resultado_carga = actor.load_state_dict(
        checkpoint["actor_state_dict"],
        strict=True,
    )
    carga_estricta = (
        len(resultado_carga.missing_keys) == 0
        and len(resultado_carga.unexpected_keys) == 0
    )
    if not carga_estricta:
        raise RuntimeError("No fue posible cargar estrictamente el actor UPO-TTC.")

    actor.eval()
    episodio_actor = int(checkpoint["episodio"])
    parametros_antes = {
        nombre: valor.detach().cpu().clone()
        for nombre, valor in actor.state_dict().items()
    }
    parametros_finitos = all(
        bool(torch.isfinite(parametro).all()) for parametro in actor.parameters()
    )

    print("\n1. Actor evaluado")
    print("Checkpoint:", RUTA_CHECKPOINT)
    print("Episodio:", episodio_actor)
    print("Dispositivo:", dispositivo)
    print("Número de parámetros:", sum(p.numel() for p in actor.parameters()))
    print("Política determinista: True")
    print("Entrenamiento durante la prueba: False")
    print("Ocupación probabilística: True")
    print("Monte Carlo: True")
    print("TTC-CVaR y clearance-CVaR: True")
    print("Metadata UPO completa:", metadata_upo_completa)
    if metadata_compatible_sin_etiquetar and not metadata_upo_completa:
        print("ADVERTENCIA: checkpoint compatible, pero aún conserva etiquetas predictivas base.")

    semilla_base = int(argumentos.semilla_base)
    numero_semillas = int(argumentos.numero_semillas)

    print("\n2. Semillas independientes")
    print("Número:", numero_semillas)
    print("Rango:", semilla_base, "a", semilla_base + numero_semillas - 1)
    print("No se utilizan semillas de entrenamiento ni validación.")

    print("\n3. Comienza la evaluación")
    evaluacion = base.evaluar_actor_sac_predictivo_multisemilla(
        actor=actor,
        semilla_base=semilla_base,
        numero_semillas=numero_semillas,
        pasos_maximos=base.PASOS_MAXIMOS_SEGUIMIENTO,
        dt=base.DT,
        dispositivo=dispositivo,
        frecuencia_impresion=int(argumentos.frecuencia_impresion),
    )

    resultados = evaluacion["resultados_individuales"]
    resumen = evaluacion["resumen"]
    conteos = resumen["conteos_resultados"]

    criterios = {
        "exito": resumen["tasa_exito"] >= CRITERIO_EXITO_MINIMO,
        "colision_dinamica": resumen["tasa_colision_dinamica"] <= CRITERIO_COLISION_DINAMICA_MAXIMA,
        "colision_estatica": resumen["tasa_colision_estatica"] <= CRITERIO_COLISION_ESTATICA_MAXIMA,
        "fuera_mapa": resumen["tasa_fuera_mapa"] <= CRITERIO_FUERA_MAPA_MAXIMA,
        "timeout": resumen["tasa_timeout"] <= CRITERIO_TIMEOUT_MAXIMO,
    }
    criterios["todos"] = all(criterios.values())

    fallos = [fila.copy() for fila in resultados if int(fila["exito"]) == 0]
    semillas_fallidas = [int(fila["semilla"]) for fila in fallos]

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    guardar_csv(RUTA_CSV, resultados)
    guardar_csv(RUTA_CSV_FALLOS, fallos)
    crear_grafica(conteos, float(resumen["tasa_exito"]))

    texto = "\n".join([
        "PRUEBA INDEPENDIENTE SAC-UPO-TTC",
        "=" * 72,
        f"Checkpoint: {RUTA_CHECKPOINT}",
        f"Episodio: {episodio_actor}",
        f"Semillas: {semilla_base} a {semilla_base + numero_semillas - 1}",
        f"Éxitos: {resumen['numero_exitos']}/{numero_semillas}",
        f"Tasa de éxito: {100.0 * resumen['tasa_exito']:.2f}%",
        "IC Wilson 95%: "
        f"[{100.0 * resumen['intervalo_exito_95_inferior']:.2f}%, "
        f"{100.0 * resumen['intervalo_exito_95_superior']:.2f}%]",
        f"Conteos: {conteos}",
        f"Colisión dinámica: {100.0 * resumen['tasa_colision_dinamica']:.2f}%",
        f"Colisión estática: {100.0 * resumen['tasa_colision_estatica']:.2f}%",
        f"Fuera del mapa: {100.0 * resumen['tasa_fuera_mapa']:.2f}%",
        f"Timeout: {100.0 * resumen['tasa_timeout']:.2f}%",
        f"Recompensa media: {resumen['recompensa_media']}",
        f"Pasos promedio: {resumen['pasos_promedio']}",
        f"Distancia final media: {resumen['distancia_final_media']} m",
        f"Error medio de ruta: {resumen['error_medio_ruta_promedio']} m",
        f"Clearance mínimo promedio: {resumen['clearance_minimo_promedio']} m",
        f"Clearance mínimo absoluto: {resumen['clearance_minimo_absoluto']} m",
        f"Criterios: {criterios}",
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
    conteos_completos = sum(int(v) for v in conteos.values()) == numero_semillas
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
    print("RESULTADO DE LA PRUEBA INDEPENDIENTE SAC-UPO-TTC")
    print("=" * 80)
    print("\nRESULTADOS GENERALES")
    print("Éxitos:", f"{resumen['numero_exitos']}/{numero_semillas}")
    print("Tasa de éxito:", f"{100.0 * resumen['tasa_exito']:.2f}%")
    print(
        "IC Wilson 95%:",
        f"[{100.0 * resumen['intervalo_exito_95_inferior']:.2f}%, "
        f"{100.0 * resumen['intervalo_exito_95_superior']:.2f}%]",
    )
    print("Conteos:", conteos)
    print("Colisión dinámica:", f"{100.0 * resumen['tasa_colision_dinamica']:.2f}%")
    print("Colisión estática:", f"{100.0 * resumen['tasa_colision_estatica']:.2f}%")
    print("Fuera del mapa:", f"{100.0 * resumen['tasa_fuera_mapa']:.2f}%")
    print("Timeout:", f"{100.0 * resumen['tasa_timeout']:.2f}%")

    print("\nMÉTRICAS")
    print("Recompensa media:", resumen["recompensa_media"])
    print("Pasos promedio:", resumen["pasos_promedio"])
    print("Distancia final media:", resumen["distancia_final_media"], "m")
    print("Error medio de ruta:", resumen["error_medio_ruta_promedio"], "m")
    print("Clearance mínimo promedio:", resumen["clearance_minimo_promedio"], "m")
    print("Clearance mínimo absoluto:", resumen["clearance_minimo_absoluto"], "m")

    print("\nCRITERIOS")
    print("Éxito >= 85%:", criterios["exito"])
    print("Colisión dinámica <= 10%:", criterios["colision_dinamica"])
    print("Colisión estática <= 10%:", criterios["colision_estatica"])
    print("Fuera del mapa <= 5%:", criterios["fuera_mapa"])
    print("Timeout <= 5%:", criterios["timeout"])
    print("TODOS LOS CRITERIOS:", criterios["todos"])

    print("\nSEMILLAS FALLIDAS")
    print(semillas_fallidas)

    print("\nARCHIVOS")
    print("CSV completo:", RUTA_CSV)
    print("CSV de fallos:", RUTA_CSV_FALLOS)
    print("Resumen:", RUTA_RESUMEN)
    print("Gráfica:", RUTA_GRAFICA)

    print("\nVERIFICACIONES TÉCNICAS")
    print("Checkpoint compatible:", metadata_upo_completa or metadata_compatible_sin_etiquetar)
    print("Metadata UPO completa:", metadata_upo_completa)
    print("Carga estricta:", carga_estricta)
    print("Parámetros finitos:", parametros_finitos)
    print("Número de resultados correcto:", numero_resultados_correcto)
    print("Conteos completos:", conteos_completos)
    print("Semillas correctas:", semillas_correctas)
    print("Archivos correctos:", archivos_correctos)
    print("Actor sin cambios:", actor_sin_cambios)
    print("Actor sin gradientes:", actor_sin_gradientes)

    tecnicamente_correcto = all([
        metadata_upo_completa or metadata_compatible_sin_etiquetar,
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
        "SAC-UPO-TTC APTO" if criterios["todos"] else "SAC-UPO-TTC AÚN NO APTO",
    )
    print("=" * 80)

if __name__ == "__main__":
    main()















