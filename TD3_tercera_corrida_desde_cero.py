from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# IMPORTACIÓN DEL ENTORNO BASE
# ==========================================================

DIRECTORIO = Path(__file__).resolve().parent
CANDIDATOS_BASE = (
    "SAC_predictivo_entrenamiento_completo.py",
    "SAC_predictivo_entrenamiento_completo(3).py",
    "Comparacion_final_DWA_SAC_reactivo_SAC_predictivo.py",
)


def localizar_base() -> Path:
    for nombre in CANDIDATOS_BASE:
        ruta = DIRECTORIO / nombre
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "Coloque este archivo junto a SAC_predictivo_entrenamiento_completo.py"
    )


RUTA_BASE = localizar_base()
_USAR_AGG = os.environ.get("MPLBACKEND", "").lower() == "agg"
_USAR_ORIGINAL = matplotlib.use
if _USAR_AGG:
    def _usar_backend(backend, *args, **kwargs):
        if str(backend).lower() == "tkagg":
            return _USAR_ORIGINAL("Agg", force=True)
        return _USAR_ORIGINAL(backend, *args, **kwargs)
    matplotlib.use = _usar_backend

_spec = importlib.util.spec_from_file_location("base_td3_navegacion", RUTA_BASE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"No se pudo importar {RUTA_BASE}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
matplotlib.use = _USAR_ORIGINAL
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURACIÓN TD3
# ==========================================================

NOMBRE_METODO = "A* + TD3 estabilizado desde cero"
VARIANTE = "td3_reactivo_tercera_corrida_desde_cero"
EPISODIOS_PREDETERMINADOS = 400
SEMILLA_ENTRENAMIENTO = 166000
SEMILLA_VALIDACION = 167000
NUMERO_SEMILLAS_VALIDACION = 50
INTERVALO_VALIDACION = 20
SEMILLA_DIAGNOSTICA = 167100

CAPACIDAD_BUFFER = 300_000
TRANSICIONES_ALEATORIAS = 10_000
TAMANO_LOTE = 256
GAMMA = 0.99
TAU = 0.0025
LR_ACTOR = 5e-5
LR_CRITICOS = 1e-4
RUIDO_EXPLORACION_INICIAL = 0.12
RUIDO_EXPLORACION_FINAL = 0.03
TRANSICIONES_DECAIMIENTO_RUIDO = 60_000
RUIDO_POLITICA_OBJETIVO = 0.10
LIMITE_RUIDO_OBJETIVO = 0.25
RETARDO_ACTOR = 2
RECORTE_GRADIENTE = 5.0

DIRECTORIO_RESULTADOS = Path("resultados_td3/tercera_corrida_td3_desde_cero")
RUTA_MEJOR = DIRECTORIO_RESULTADOS / "checkpoint_mejor_actor_td3.pt"
RUTA_FINAL = DIRECTORIO_RESULTADOS / "checkpoint_final_td3.pt"
RUTA_CSV = DIRECTORIO_RESULTADOS / "validaciones_periodicas_50_semillas.csv"
RUTA_GRAFICA = DIRECTORIO_RESULTADOS / "entrenamiento_td3.png"

# ==========================================================
# REDES
# ==========================================================


def fijar_semillas(semilla: int) -> None:
    random.seed(int(semilla))
    np.random.seed(int(semilla))
    torch.manual_seed(int(semilla))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(semilla))


def crear_actor_td3(dispositivo: torch.device) -> nn.ModuleDict:
    actor = nn.ModuleDict({
        "rama_cnn": base.crear_rama_cnn_sac(dispositivo),
        "rama_mlp": base.crear_rama_mlp_escalares_sac(dispositivo),
        "red_compartida": base.crear_red_compartida_actor_sac(dispositivo),
        "cabeza_accion": nn.Linear(
            base.NEURONAS_CAPA_2_ACTOR_SAC,
            base.DIMENSION_ACCION_SAC,
        ).to(dispositivo),
    }).to(dispositivo)
    return actor


def ejecutar_actor_td3(
    actor: nn.ModuleDict,
    parche: torch.Tensor,
    escalares: torch.Tensor,
) -> torch.Tensor:
    cnn = actor["rama_cnn"](parche)
    mlp = actor["rama_mlp"](escalares)
    fusion = base.fusionar_caracteristicas_sac(cnn, mlp)
    oculto = actor["red_compartida"](fusion)
    return torch.tanh(actor["cabeza_accion"](oculto))


def tensores_observacion(
    observacion: Mapping[str, np.ndarray],
    dispositivo: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    parche = torch.as_tensor(
        observacion["parche"], dtype=torch.float32, device=dispositivo
    ).unsqueeze(0)
    escalares = torch.as_tensor(
        observacion["escalares"], dtype=torch.float32, device=dispositivo
    ).unsqueeze(0)
    return parche, escalares


def accion_td3(
    actor: nn.ModuleDict,
    observacion: Mapping[str, np.ndarray],
    dispositivo: torch.device,
    ruido: float = 0.0,
    generador: np.random.Generator | None = None,
) -> np.ndarray:
    modo = actor.training
    actor.eval()
    with torch.no_grad():
        parche, escalares = tensores_observacion(observacion, dispositivo)
        accion = ejecutar_actor_td3(actor, parche, escalares)[0].cpu().numpy()
    actor.train(modo)
    if ruido > 0.0:
        if generador is None:
            generador = np.random.default_rng()
        accion = accion + generador.normal(0.0, ruido, size=accion.shape)
    return np.clip(accion, -1.0, 1.0).astype(np.float32)


def actualizar_objetivo(red: nn.Module, objetivo: nn.Module, tau: float = TAU) -> None:
    with torch.no_grad():
        for p, pt in zip(red.parameters(), objetivo.parameters()):
            pt.mul_(1.0 - tau).add_(p, alpha=tau)

# ==========================================================
# ACTUALIZACIÓN TD3
# ==========================================================


def actualizar_td3(
    actor: nn.ModuleDict,
    actor_objetivo: nn.ModuleDict,
    q1: nn.ModuleDict,
    q2: nn.ModuleDict,
    q1_objetivo: nn.ModuleDict,
    q2_objetivo: nn.ModuleDict,
    optim_actor: torch.optim.Optimizer,
    optim_q1: torch.optim.Optimizer,
    optim_q2: torch.optim.Optimizer,
    buffer: Dict[str, object],
    dispositivo: torch.device,
    numero_actualizacion: int,
) -> Dict[str, float]:
    lote = base.muestrear_lote_buffer_sac(
        buffer=buffer,
        tamano_lote=TAMANO_LOTE,
        dispositivo=dispositivo,
    )
    p = lote["parches"]
    e = lote["escalares"]
    a = lote["acciones"]
    r = lote["recompensas"]
    sp = lote["siguientes_parches"]
    se = lote["siguientes_escalares"]
    d = lote["terminados"]

    with torch.no_grad():
        siguiente_accion = ejecutar_actor_td3(actor_objetivo, sp, se)
        ruido = torch.randn_like(siguiente_accion) * RUIDO_POLITICA_OBJETIVO
        ruido = ruido.clamp(-LIMITE_RUIDO_OBJETIVO, LIMITE_RUIDO_OBJETIVO)
        siguiente_accion = (siguiente_accion + ruido).clamp(-1.0, 1.0)
        q1t = base.ejecutar_critico_sac(sp, se, siguiente_accion, q1_objetivo)["valor_q"]
        q2t = base.ejecutar_critico_sac(sp, se, siguiente_accion, q2_objetivo)["valor_q"]
        objetivo = r + GAMMA * (1.0 - d) * torch.minimum(q1t, q2t)

    q1_pred = base.ejecutar_critico_sac(p, e, a, q1)["valor_q"]
    q2_pred = base.ejecutar_critico_sac(p, e, a, q2)["valor_q"]
    perdida_q1 = F.smooth_l1_loss(q1_pred, objetivo)
    perdida_q2 = F.smooth_l1_loss(q2_pred, objetivo)

    optim_q1.zero_grad(set_to_none=True)
    perdida_q1.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), RECORTE_GRADIENTE)
    optim_q1.step()

    optim_q2.zero_grad(set_to_none=True)
    perdida_q2.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), RECORTE_GRADIENTE)
    optim_q2.step()

    perdida_actor = float("nan")
    actor_actualizado = numero_actualizacion % RETARDO_ACTOR == 0
    if actor_actualizado:
        acciones_actor = ejecutar_actor_td3(actor, p, e)
        perdida = -base.ejecutar_critico_sac(p, e, acciones_actor, q1)["valor_q"].mean()
        optim_actor.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), RECORTE_GRADIENTE)
        optim_actor.step()
        perdida_actor = float(perdida.detach().cpu())
        actualizar_objetivo(actor, actor_objetivo)
        actualizar_objetivo(q1, q1_objetivo)
        actualizar_objetivo(q2, q2_objetivo)

    return {
        "perdida_q1": float(perdida_q1.detach().cpu()),
        "perdida_q2": float(perdida_q2.detach().cpu()),
        "perdida_actor": perdida_actor,
        "actor_actualizado": float(actor_actualizado),
    }

# ==========================================================
# EVALUACIÓN
# ==========================================================


def ejecutar_episodio(
    actor: nn.ModuleDict,
    semilla: int,
    dispositivo: torch.device,
) -> Dict[str, float | int | str]:
    entorno, observacion = base.reiniciar_entorno_sac(semilla)
    terminado = truncado = False
    recompensa_total = 0.0
    ultima_info: Dict[str, object] = {}
    while not (terminado or truncado):
        accion = accion_td3(actor, observacion, dispositivo)
        observacion, recompensa, terminado, truncado, ultima_info = (
            base.ejecutar_paso_entorno_sac(entorno, accion)
        )
        recompensa_total += float(recompensa)
    registro = entorno["registro"]
    clearances = registro.get("clearances", [])
    clearance_min = float(np.min(clearances)) if len(clearances) else float("inf")
    estado = entorno["estado_robot"]
    distancia_final = float(base.distancia_entre_puntos((estado[0], estado[1]), entorno["meta"]))
    return {
        "semilla": int(semilla),
        "resultado": str(entorno["resultado"]),
        "pasos": int(entorno["paso_actual"]),
        "recompensa": recompensa_total,
        "distancia_final": distancia_final,
        "clearance_minimo": clearance_min,
    }


def validar_actor(
    actor: nn.ModuleDict,
    semillas: Sequence[int],
    dispositivo: torch.device,
) -> Dict[str, object]:
    resultados = [ejecutar_episodio(actor, s, dispositivo) for s in semillas]
    conteos = {k: 0 for k in ("meta", "colision_estatica", "colision_dinamica", "fuera_mapa", "timeout")}
    for fila in resultados:
        conteos[str(fila["resultado"])] = conteos.get(str(fila["resultado"]), 0) + 1
    exitos = [f for f in resultados if f["resultado"] == "meta"]
    n = len(resultados)
    return {
        "resultados": resultados,
        "conteos": conteos,
        "tasa_exito": len(exitos) / n,
        "tasa_timeout": conteos.get("timeout", 0) / n,
        "tasa_colision_dinamica": conteos.get("colision_dinamica", 0) / n,
        "tasa_colision_estatica": conteos.get("colision_estatica", 0) / n,
        "tasa_fuera_mapa": conteos.get("fuera_mapa", 0) / n,
        "distancia_final_media": float(np.mean([f["distancia_final"] for f in resultados])),
        "clearance_exitos": float(np.mean([f["clearance_minimo"] for f in exitos])) if exitos else -float("inf"),
    }


def criterio_validacion(v: Mapping[str, object]) -> Tuple[float, ...]:
    return (
        float(v["tasa_exito"]),
        -float(v["tasa_timeout"]),
        -float(v["tasa_colision_dinamica"]),
        -float(v["tasa_colision_estatica"]),
        -float(v["tasa_fuera_mapa"]),
        -float(v["distancia_final_media"]),
        float(v["clearance_exitos"]),
    )


def guardar_checkpoint(
    ruta: Path,
    actor: nn.ModuleDict,
    episodio: int,
    validacion: Mapping[str, object],
    extras: Mapping[str, object],
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tipo": "mejor_actor_td3" if "mejor" in ruta.name else "td3_entrenamiento_completo",
        "variante": VARIANTE,
        "metodo": NOMBRE_METODO,
        "episodio": int(episodio),
        "actor_state_dict": actor.state_dict(),
        "validacion": {k: v for k, v in validacion.items() if k != "resultados"},
        "configuracion": {
            "gamma": GAMMA, "tau": TAU, "lr_actor": LR_ACTOR,
            "lr_criticos": LR_CRITICOS, "policy_delay": RETARDO_ACTOR,
            "policy_noise": RUIDO_POLITICA_OBJETIVO,
            "noise_clip": LIMITE_RUIDO_OBJETIVO,
            "observacion": "reactiva_actual",
            "perdida_criticos": "smooth_l1_huber",
            "transiciones_aleatorias": TRANSICIONES_ALEATORIAS,
            "transiciones_decaimiento_ruido": TRANSICIONES_DECAIMIENTO_RUIDO,
            "presupuesto_episodios": int(EPISODIOS_PREDETERMINADOS),
            "actor_inicial": "inicializacion_aleatoria_desde_cero",
            "semilla_inicializacion": int(SEMILLA_ENTRENAMIENTO),
        },
        **dict(extras),
    }, ruta)

# ==========================================================
# VERIFICACIÓN Y ENTRENAMIENTO
# ==========================================================


def verificar_integracion() -> None:
    dispositivo = base.DISPOSITIVO_SAC
    actor = crear_actor_td3(dispositivo)
    q1, q2 = base.crear_dos_criticos_sac(dispositivo)
    entorno, obs = base.reiniciar_entorno_sac(base.SEMILLA)
    a = accion_td3(actor, obs, dispositivo)
    siguiente, r, terminado, truncado, info = base.ejecutar_paso_entorno_sac(entorno, a, pasos_maximos=5)
    parche, escalares = tensores_observacion(obs, dispositivo)
    at = ejecutar_actor_td3(actor, parche, escalares)
    qt = base.ejecutar_critico_sac(parche, escalares, at, q1)["valor_q"]
    comprobaciones = {
        "accion_forma": a.shape == (base.DIMENSION_ACCION_SAC,),
        "accion_rango": bool(np.all(a >= -1.0) and np.all(a <= 1.0)),
        "observacion_compatible": siguiente["parche"].shape == obs["parche"].shape,
        "recompensa_finita": np.isfinite(r),
        "q_finito": bool(torch.isfinite(qt).all()),
        "entorno_reactivo": info.get("modo_sac", "reactivo") != "predictivo",
    }
    print("\n" + "=" * 80)
    print("VERIFICACIÓN TD3")
    print("=" * 80)
    for k, v in comprobaciones.items():
        print(f"{k}: {v}")
    if not all(comprobaciones.values()):
        raise RuntimeError("Falló la verificación TD3")
    print("RESULTADO DE VERIFICACIÓN: TODO CORRECTO")


def entrenar(episodios: int, mostrar: bool) -> None:
    fijar_semillas(SEMILLA_ENTRENAMIENTO)
    dispositivo = base.DISPOSITIVO_SAC
    # Inicialización completamente aleatoria y reproducible.
    # No se carga ningún actor de corridas TD3 anteriores.
    actor = crear_actor_td3(dispositivo)
    actor_obj = copy.deepcopy(actor).to(dispositivo).eval()
    q1, q2 = base.crear_dos_criticos_sac(dispositivo)
    q1_obj, q2_obj = base.crear_dos_criticos_objetivo_sac(q1, q2, dispositivo)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optim_q1 = torch.optim.Adam(q1.parameters(), lr=LR_CRITICOS)
    optim_q2 = torch.optim.Adam(q2.parameters(), lr=LR_CRITICOS)
    buffer = base.crear_buffer_repeticion_sac(CAPACIDAD_BUFFER, SEMILLA_ENTRENAMIENTO + 99)
    rng = np.random.default_rng(SEMILLA_ENTRENAMIENTO + 123)
    semillas_val = list(range(SEMILLA_VALIDACION, SEMILLA_VALIDACION + NUMERO_SEMILLAS_VALIDACION))

    historial: List[Dict[str, object]] = []
    recompensas: List[float] = []
    exitos: List[int] = []
    numero_actualizacion = 0
    mejor_criterio: Tuple[float, ...] | None = None
    mejor_validacion: Dict[str, object] | None = None
    mejor_episodio = 0
    ultima_validacion: Dict[str, object] | None = None

    print("\n" + "=" * 80)
    print("TERCERA CORRIDA TD3 ESTABILIZADA DESDE CERO")
    print("=" * 80)
    print(f"Dispositivo: {dispositivo}")
    print("Actor inicial: inicialización aleatoria desde cero")
    print(f"Semilla de inicialización: {SEMILLA_ENTRENAMIENTO}")
    print("Actor, críticos, objetivos, optimizadores y buffer: creados desde cero")
    print(f"Episodios: {episodios}")
    print(f"Semillas de entrenamiento: {SEMILLA_ENTRENAMIENTO} a {SEMILLA_ENTRENAMIENTO + episodios - 1}")
    print(f"Semillas nuevas de validación fija: {SEMILLA_VALIDACION} a {SEMILLA_VALIDACION + NUMERO_SEMILLAS_VALIDACION - 1}")

    for episodio in range(1, episodios + 1):
        entorno, observacion = base.reiniciar_entorno_sac(SEMILLA_ENTRENAMIENTO + episodio - 1)
        terminado = truncado = False
        recompensa_ep = 0.0
        while not (terminado or truncado):
            if int(buffer["total_transiciones_agregadas"]) < TRANSICIONES_ALEATORIAS:
                accion = rng.uniform(-1.0, 1.0, size=base.DIMENSION_ACCION_SAC).astype(np.float32)
            else:
                transiciones_politica = max(
                    0,
                    int(buffer["total_transiciones_agregadas"])
                    - TRANSICIONES_ALEATORIAS,
                )
                fraccion = min(
                    1.0,
                    transiciones_politica
                    / max(1, TRANSICIONES_DECAIMIENTO_RUIDO),
                )
                sigma = (
                    RUIDO_EXPLORACION_INICIAL
                    + fraccion
                    * (RUIDO_EXPLORACION_FINAL - RUIDO_EXPLORACION_INICIAL)
                )
                accion = accion_td3(actor, observacion, dispositivo, sigma, rng)
            siguiente, recompensa, terminado, truncado, info = base.ejecutar_paso_entorno_sac(entorno, accion)
            base.agregar_transicion_buffer_sac(
                buffer, observacion, accion, recompensa, siguiente, bool(terminado or truncado)
            )
            observacion = siguiente
            recompensa_ep += float(recompensa)
            if int(buffer["tamano"]) >= TAMANO_LOTE and int(buffer["total_transiciones_agregadas"]) >= TRANSICIONES_ALEATORIAS:
                numero_actualizacion += 1
                actualizar_td3(
                    actor, actor_obj, q1, q2, q1_obj, q2_obj,
                    optim_actor, optim_q1, optim_q2, buffer,
                    dispositivo, numero_actualizacion,
                )
        recompensas.append(recompensa_ep)
        exitos.append(int(entorno["resultado"] == "meta"))

        if episodio % INTERVALO_VALIDACION == 0 or episodio == episodios:
            val = validar_actor(actor, semillas_val, dispositivo)
            ultima_validacion = val
            criterio = criterio_validacion(val)
            nuevo = mejor_criterio is None or criterio > mejor_criterio
            if nuevo:
                mejor_criterio = criterio
                mejor_validacion = val
                mejor_episodio = episodio
                guardar_checkpoint(RUTA_MEJOR, actor, episodio, val, {"actualizaciones": numero_actualizacion})
            fila = {
                "episodio": episodio,
                "tasa_exito": val["tasa_exito"],
                "timeout": val["tasa_timeout"],
                "colision_dinamica": val["tasa_colision_dinamica"],
                "colision_estatica": val["tasa_colision_estatica"],
                "fuera_mapa": val["tasa_fuera_mapa"],
                "distancia_final_media": val["distancia_final_media"],
                "clearance_exitos": val["clearance_exitos"],
            }
            historial.append(fila)
            print(
                f"Episodios {episodio:4d}/{episodios:4d} | "
                f"R móvil={np.mean(recompensas[-25:]):9.2f} | "
                f"éxito train={100*np.mean(exitos[-25:]):6.2f}% | "
                f"validación={100*float(val['tasa_exito']):6.2f}% | "
                f"col. din={100*float(val['tasa_colision_dinamica']):5.2f}% | "
                f"{'NUEVO MEJOR' if nuevo else ''}"
            )

    if mejor_validacion is None:
        raise RuntimeError("No se realizó validación TD3")
    if ultima_validacion is None:
        raise RuntimeError("No existe validación final TD3")
    guardar_checkpoint(
        RUTA_FINAL, actor, episodios, ultima_validacion,
        {
            "actualizaciones": numero_actualizacion,
            "actor_objetivo_state_dict": actor_obj.state_dict(),
            "q1_state_dict": q1.state_dict(), "q2_state_dict": q2.state_dict(),
            "q1_objetivo_state_dict": q1_obj.state_dict(),
            "q2_objetivo_state_dict": q2_obj.state_dict(),
            "optimizador_actor": optim_actor.state_dict(),
            "optimizador_q1": optim_q1.state_dict(),
            "optimizador_q2": optim_q2.state_dict(),
        },
    )

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with RUTA_CSV.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        escritor.writeheader(); escritor.writerows(historial)

    figura = plt.figure(figsize=(12, 7))
    ax1 = figura.add_subplot(211)
    ax1.plot(range(1, len(recompensas)+1), recompensas)
    ax1.set_ylabel("Recompensa"); ax1.grid(True, alpha=.3)
    ax2 = figura.add_subplot(212)
    ax2.plot([h["episodio"] for h in historial], [100*h["tasa_exito"] for h in historial], marker="o")
    ax2.set_xlabel("Episodio"); ax2.set_ylabel("Éxito validación (%)"); ax2.grid(True, alpha=.3)
    figura.tight_layout(); figura.savefig(RUTA_GRAFICA, dpi=160)
    # La figura ya fue guardada. Se cierra para evitar que plt.show()
    # bloquee el programa después de terminar los 400 episodios.
    plt.close(figura)

    mejor = torch.load(RUTA_MEJOR, map_location=dispositivo, weights_only=False)
    actor.load_state_dict(mejor["actor_state_dict"], strict=True)
    diagnostico = ejecutar_episodio(actor, SEMILLA_DIAGNOSTICA, dispositivo)
    print("\n" + "=" * 80)
    print("RESULTADO DEL ENTRENAMIENTO TD3")
    print("=" * 80)
    print(f"Mejor episodio: {mejor_episodio}")
    print(f"Éxito validación: {100*float(mejor_validacion['tasa_exito']):.2f}%")
    print(f"Colisión dinámica: {100*float(mejor_validacion['tasa_colision_dinamica']):.2f}%")
    print(f"Colisión estática: {100*float(mejor_validacion['tasa_colision_estatica']):.2f}%")
    print(f"Timeout: {100*float(mejor_validacion['tasa_timeout']):.2f}%")
    print(f"Diagnóstico {SEMILLA_DIAGNOSTICA}: {diagnostico['resultado']}")
    print(f"Mejor actor: {RUTA_MEJOR}")
    print("RESULTADO TÉCNICO: ENTRENAMIENTO TD3 COMPLETADO")

def analizar_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline TD3 para el mismo entorno de navegación")
    p.add_argument("--solo-verificacion", action="store_true")
    p.add_argument("--episodios", type=int, default=EPISODIOS_PREDETERMINADOS)
    p.add_argument("--sin-graficas", action="store_true")
    return p.parse_args()

def main() -> None:
    args = analizar_argumentos()
    verificar_integracion()
    if not args.solo_verificacion:
        entrenar(args.episodios, not args.sin_graficas)

if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# IMPORTACIÓN DEL ENTORNO BASE
# ==========================================================

DIRECTORIO = Path(__file__).resolve().parent
CANDIDATOS_BASE = (
    "SAC_predictivo_entrenamiento_completo.py",
    "SAC_predictivo_entrenamiento_completo(3).py",
    "Comparacion_final_DWA_SAC_reactivo_SAC_predictivo.py",
)


def localizar_base() -> Path:
    for nombre in CANDIDATOS_BASE:
        ruta = DIRECTORIO / nombre
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "Coloque este archivo junto a SAC_predictivo_entrenamiento_completo.py"
    )


RUTA_BASE = localizar_base()
_USAR_AGG = os.environ.get("MPLBACKEND", "").lower() == "agg"
_USAR_ORIGINAL = matplotlib.use
if _USAR_AGG:
    def _usar_backend(backend, *args, **kwargs):
        if str(backend).lower() == "tkagg":
            return _USAR_ORIGINAL("Agg", force=True)
        return _USAR_ORIGINAL(backend, *args, **kwargs)
    matplotlib.use = _usar_backend

_spec = importlib.util.spec_from_file_location("base_td3_navegacion", RUTA_BASE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"No se pudo importar {RUTA_BASE}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
matplotlib.use = _USAR_ORIGINAL
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURACIÓN TD3
# ==========================================================

NOMBRE_METODO = "A* + TD3 estabilizado desde cero"
VARIANTE = "td3_reactivo_tercera_corrida_desde_cero"
EPISODIOS_PREDETERMINADOS = 400
SEMILLA_ENTRENAMIENTO = 166000
SEMILLA_VALIDACION = 167000
NUMERO_SEMILLAS_VALIDACION = 50
INTERVALO_VALIDACION = 20
SEMILLA_DIAGNOSTICA = 167100

CAPACIDAD_BUFFER = 300_000
TRANSICIONES_ALEATORIAS = 10_000
TAMANO_LOTE = 256
GAMMA = 0.99
TAU = 0.0025
LR_ACTOR = 5e-5
LR_CRITICOS = 1e-4
RUIDO_EXPLORACION_INICIAL = 0.12
RUIDO_EXPLORACION_FINAL = 0.03
TRANSICIONES_DECAIMIENTO_RUIDO = 60_000
RUIDO_POLITICA_OBJETIVO = 0.10
LIMITE_RUIDO_OBJETIVO = 0.25
RETARDO_ACTOR = 2
RECORTE_GRADIENTE = 5.0

DIRECTORIO_RESULTADOS = Path("resultados_td3/tercera_corrida_td3_desde_cero")
RUTA_MEJOR = DIRECTORIO_RESULTADOS / "checkpoint_mejor_actor_td3.pt"
RUTA_FINAL = DIRECTORIO_RESULTADOS / "checkpoint_final_td3.pt"
RUTA_CSV = DIRECTORIO_RESULTADOS / "validaciones_periodicas_50_semillas.csv"
RUTA_GRAFICA = DIRECTORIO_RESULTADOS / "entrenamiento_td3.png"

# ==========================================================
# REDES
# ==========================================================


def fijar_semillas(semilla: int) -> None:
    random.seed(int(semilla))
    np.random.seed(int(semilla))
    torch.manual_seed(int(semilla))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(semilla))


def crear_actor_td3(dispositivo: torch.device) -> nn.ModuleDict:
    actor = nn.ModuleDict({
        "rama_cnn": base.crear_rama_cnn_sac(dispositivo),
        "rama_mlp": base.crear_rama_mlp_escalares_sac(dispositivo),
        "red_compartida": base.crear_red_compartida_actor_sac(dispositivo),
        "cabeza_accion": nn.Linear(
            base.NEURONAS_CAPA_2_ACTOR_SAC,
            base.DIMENSION_ACCION_SAC,
        ).to(dispositivo),
    }).to(dispositivo)
    return actor


def ejecutar_actor_td3(
    actor: nn.ModuleDict,
    parche: torch.Tensor,
    escalares: torch.Tensor,
) -> torch.Tensor:
    cnn = actor["rama_cnn"](parche)
    mlp = actor["rama_mlp"](escalares)
    fusion = base.fusionar_caracteristicas_sac(cnn, mlp)
    oculto = actor["red_compartida"](fusion)
    return torch.tanh(actor["cabeza_accion"](oculto))


def tensores_observacion(
    observacion: Mapping[str, np.ndarray],
    dispositivo: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    parche = torch.as_tensor(
        observacion["parche"], dtype=torch.float32, device=dispositivo
    ).unsqueeze(0)
    escalares = torch.as_tensor(
        observacion["escalares"], dtype=torch.float32, device=dispositivo
    ).unsqueeze(0)
    return parche, escalares


def accion_td3(
    actor: nn.ModuleDict,
    observacion: Mapping[str, np.ndarray],
    dispositivo: torch.device,
    ruido: float = 0.0,
    generador: np.random.Generator | None = None,
) -> np.ndarray:
    modo = actor.training
    actor.eval()
    with torch.no_grad():
        parche, escalares = tensores_observacion(observacion, dispositivo)
        accion = ejecutar_actor_td3(actor, parche, escalares)[0].cpu().numpy()
    actor.train(modo)
    if ruido > 0.0:
        if generador is None:
            generador = np.random.default_rng()
        accion = accion + generador.normal(0.0, ruido, size=accion.shape)
    return np.clip(accion, -1.0, 1.0).astype(np.float32)


def actualizar_objetivo(red: nn.Module, objetivo: nn.Module, tau: float = TAU) -> None:
    with torch.no_grad():
        for p, pt in zip(red.parameters(), objetivo.parameters()):
            pt.mul_(1.0 - tau).add_(p, alpha=tau)

# ==========================================================
# ACTUALIZACIÓN TD3
# ==========================================================


def actualizar_td3(
    actor: nn.ModuleDict,
    actor_objetivo: nn.ModuleDict,
    q1: nn.ModuleDict,
    q2: nn.ModuleDict,
    q1_objetivo: nn.ModuleDict,
    q2_objetivo: nn.ModuleDict,
    optim_actor: torch.optim.Optimizer,
    optim_q1: torch.optim.Optimizer,
    optim_q2: torch.optim.Optimizer,
    buffer: Dict[str, object],
    dispositivo: torch.device,
    numero_actualizacion: int,
) -> Dict[str, float]:
    lote = base.muestrear_lote_buffer_sac(
        buffer=buffer,
        tamano_lote=TAMANO_LOTE,
        dispositivo=dispositivo,
    )
    p = lote["parches"]
    e = lote["escalares"]
    a = lote["acciones"]
    r = lote["recompensas"]
    sp = lote["siguientes_parches"]
    se = lote["siguientes_escalares"]
    d = lote["terminados"]

    with torch.no_grad():
        siguiente_accion = ejecutar_actor_td3(actor_objetivo, sp, se)
        ruido = torch.randn_like(siguiente_accion) * RUIDO_POLITICA_OBJETIVO
        ruido = ruido.clamp(-LIMITE_RUIDO_OBJETIVO, LIMITE_RUIDO_OBJETIVO)
        siguiente_accion = (siguiente_accion + ruido).clamp(-1.0, 1.0)
        q1t = base.ejecutar_critico_sac(sp, se, siguiente_accion, q1_objetivo)["valor_q"]
        q2t = base.ejecutar_critico_sac(sp, se, siguiente_accion, q2_objetivo)["valor_q"]
        objetivo = r + GAMMA * (1.0 - d) * torch.minimum(q1t, q2t)

    q1_pred = base.ejecutar_critico_sac(p, e, a, q1)["valor_q"]
    q2_pred = base.ejecutar_critico_sac(p, e, a, q2)["valor_q"]
    perdida_q1 = F.smooth_l1_loss(q1_pred, objetivo)
    perdida_q2 = F.smooth_l1_loss(q2_pred, objetivo)

    optim_q1.zero_grad(set_to_none=True)
    perdida_q1.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), RECORTE_GRADIENTE)
    optim_q1.step()

    optim_q2.zero_grad(set_to_none=True)
    perdida_q2.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), RECORTE_GRADIENTE)
    optim_q2.step()

    perdida_actor = float("nan")
    actor_actualizado = numero_actualizacion % RETARDO_ACTOR == 0
    if actor_actualizado:
        acciones_actor = ejecutar_actor_td3(actor, p, e)
        perdida = -base.ejecutar_critico_sac(p, e, acciones_actor, q1)["valor_q"].mean()
        optim_actor.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), RECORTE_GRADIENTE)
        optim_actor.step()
        perdida_actor = float(perdida.detach().cpu())
        actualizar_objetivo(actor, actor_objetivo)
        actualizar_objetivo(q1, q1_objetivo)
        actualizar_objetivo(q2, q2_objetivo)

    return {
        "perdida_q1": float(perdida_q1.detach().cpu()),
        "perdida_q2": float(perdida_q2.detach().cpu()),
        "perdida_actor": perdida_actor,
        "actor_actualizado": float(actor_actualizado),
    }

# ==========================================================
# EVALUACIÓN
# ==========================================================


def ejecutar_episodio(
    actor: nn.ModuleDict,
    semilla: int,
    dispositivo: torch.device,
) -> Dict[str, float | int | str]:
    entorno, observacion = base.reiniciar_entorno_sac(semilla)
    terminado = truncado = False
    recompensa_total = 0.0
    ultima_info: Dict[str, object] = {}
    while not (terminado or truncado):
        accion = accion_td3(actor, observacion, dispositivo)
        observacion, recompensa, terminado, truncado, ultima_info = (
            base.ejecutar_paso_entorno_sac(entorno, accion)
        )
        recompensa_total += float(recompensa)
    registro = entorno["registro"]
    clearances = registro.get("clearances", [])
    clearance_min = float(np.min(clearances)) if len(clearances) else float("inf")
    estado = entorno["estado_robot"]
    distancia_final = float(base.distancia_entre_puntos((estado[0], estado[1]), entorno["meta"]))
    return {
        "semilla": int(semilla),
        "resultado": str(entorno["resultado"]),
        "pasos": int(entorno["paso_actual"]),
        "recompensa": recompensa_total,
        "distancia_final": distancia_final,
        "clearance_minimo": clearance_min,
    }


def validar_actor(
    actor: nn.ModuleDict,
    semillas: Sequence[int],
    dispositivo: torch.device,
) -> Dict[str, object]:
    resultados = [ejecutar_episodio(actor, s, dispositivo) for s in semillas]
    conteos = {k: 0 for k in ("meta", "colision_estatica", "colision_dinamica", "fuera_mapa", "timeout")}
    for fila in resultados:
        conteos[str(fila["resultado"])] = conteos.get(str(fila["resultado"]), 0) + 1
    exitos = [f for f in resultados if f["resultado"] == "meta"]
    n = len(resultados)
    return {
        "resultados": resultados,
        "conteos": conteos,
        "tasa_exito": len(exitos) / n,
        "tasa_timeout": conteos.get("timeout", 0) / n,
        "tasa_colision_dinamica": conteos.get("colision_dinamica", 0) / n,
        "tasa_colision_estatica": conteos.get("colision_estatica", 0) / n,
        "tasa_fuera_mapa": conteos.get("fuera_mapa", 0) / n,
        "distancia_final_media": float(np.mean([f["distancia_final"] for f in resultados])),
        "clearance_exitos": float(np.mean([f["clearance_minimo"] for f in exitos])) if exitos else -float("inf"),
    }


def criterio_validacion(v: Mapping[str, object]) -> Tuple[float, ...]:
    return (
        float(v["tasa_exito"]),
        -float(v["tasa_timeout"]),
        -float(v["tasa_colision_dinamica"]),
        -float(v["tasa_colision_estatica"]),
        -float(v["tasa_fuera_mapa"]),
        -float(v["distancia_final_media"]),
        float(v["clearance_exitos"]),
    )


def guardar_checkpoint(
    ruta: Path,
    actor: nn.ModuleDict,
    episodio: int,
    validacion: Mapping[str, object],
    extras: Mapping[str, object],
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tipo": "mejor_actor_td3" if "mejor" in ruta.name else "td3_entrenamiento_completo",
        "variante": VARIANTE,
        "metodo": NOMBRE_METODO,
        "episodio": int(episodio),
        "actor_state_dict": actor.state_dict(),
        "validacion": {k: v for k, v in validacion.items() if k != "resultados"},
        "configuracion": {
            "gamma": GAMMA, "tau": TAU, "lr_actor": LR_ACTOR,
            "lr_criticos": LR_CRITICOS, "policy_delay": RETARDO_ACTOR,
            "policy_noise": RUIDO_POLITICA_OBJETIVO,
            "noise_clip": LIMITE_RUIDO_OBJETIVO,
            "observacion": "reactiva_actual",
            "perdida_criticos": "smooth_l1_huber",
            "transiciones_aleatorias": TRANSICIONES_ALEATORIAS,
            "transiciones_decaimiento_ruido": TRANSICIONES_DECAIMIENTO_RUIDO,
            "presupuesto_episodios": int(EPISODIOS_PREDETERMINADOS),
            "actor_inicial": "inicializacion_aleatoria_desde_cero",
            "semilla_inicializacion": int(SEMILLA_ENTRENAMIENTO),
        },
        **dict(extras),
    }, ruta)

# ==========================================================
# VERIFICACIÓN Y ENTRENAMIENTO
# ==========================================================


def verificar_integracion() -> None:
    dispositivo = base.DISPOSITIVO_SAC
    actor = crear_actor_td3(dispositivo)
    q1, q2 = base.crear_dos_criticos_sac(dispositivo)
    entorno, obs = base.reiniciar_entorno_sac(base.SEMILLA)
    a = accion_td3(actor, obs, dispositivo)
    siguiente, r, terminado, truncado, info = base.ejecutar_paso_entorno_sac(entorno, a, pasos_maximos=5)
    parche, escalares = tensores_observacion(obs, dispositivo)
    at = ejecutar_actor_td3(actor, parche, escalares)
    qt = base.ejecutar_critico_sac(parche, escalares, at, q1)["valor_q"]
    comprobaciones = {
        "accion_forma": a.shape == (base.DIMENSION_ACCION_SAC,),
        "accion_rango": bool(np.all(a >= -1.0) and np.all(a <= 1.0)),
        "observacion_compatible": siguiente["parche"].shape == obs["parche"].shape,
        "recompensa_finita": np.isfinite(r),
        "q_finito": bool(torch.isfinite(qt).all()),
        "entorno_reactivo": info.get("modo_sac", "reactivo") != "predictivo",
    }
    print("\n" + "=" * 80)
    print("VERIFICACIÓN TD3")
    print("=" * 80)
    for k, v in comprobaciones.items():
        print(f"{k}: {v}")
    if not all(comprobaciones.values()):
        raise RuntimeError("Falló la verificación TD3")
    print("RESULTADO DE VERIFICACIÓN: TODO CORRECTO")


def entrenar(episodios: int, mostrar: bool) -> None:
    fijar_semillas(SEMILLA_ENTRENAMIENTO)
    dispositivo = base.DISPOSITIVO_SAC
    # Inicialización completamente aleatoria y reproducible.
    # No se carga ningún actor de corridas TD3 anteriores.
    actor = crear_actor_td3(dispositivo)
    actor_obj = copy.deepcopy(actor).to(dispositivo).eval()
    q1, q2 = base.crear_dos_criticos_sac(dispositivo)
    q1_obj, q2_obj = base.crear_dos_criticos_objetivo_sac(q1, q2, dispositivo)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optim_q1 = torch.optim.Adam(q1.parameters(), lr=LR_CRITICOS)
    optim_q2 = torch.optim.Adam(q2.parameters(), lr=LR_CRITICOS)
    buffer = base.crear_buffer_repeticion_sac(CAPACIDAD_BUFFER, SEMILLA_ENTRENAMIENTO + 99)
    rng = np.random.default_rng(SEMILLA_ENTRENAMIENTO + 123)
    semillas_val = list(range(SEMILLA_VALIDACION, SEMILLA_VALIDACION + NUMERO_SEMILLAS_VALIDACION))

    historial: List[Dict[str, object]] = []
    recompensas: List[float] = []
    exitos: List[int] = []
    numero_actualizacion = 0
    mejor_criterio: Tuple[float, ...] | None = None
    mejor_validacion: Dict[str, object] | None = None
    mejor_episodio = 0
    ultima_validacion: Dict[str, object] | None = None

    print("\n" + "=" * 80)
    print("TERCERA CORRIDA TD3 ESTABILIZADA DESDE CERO")
    print("=" * 80)
    print(f"Dispositivo: {dispositivo}")
    print("Actor inicial: inicialización aleatoria desde cero")
    print(f"Semilla de inicialización: {SEMILLA_ENTRENAMIENTO}")
    print("Actor, críticos, objetivos, optimizadores y buffer: creados desde cero")
    print(f"Episodios: {episodios}")
    print(f"Semillas de entrenamiento: {SEMILLA_ENTRENAMIENTO} a {SEMILLA_ENTRENAMIENTO + episodios - 1}")
    print(f"Semillas nuevas de validación fija: {SEMILLA_VALIDACION} a {SEMILLA_VALIDACION + NUMERO_SEMILLAS_VALIDACION - 1}")

    for episodio in range(1, episodios + 1):
        entorno, observacion = base.reiniciar_entorno_sac(SEMILLA_ENTRENAMIENTO + episodio - 1)
        terminado = truncado = False
        recompensa_ep = 0.0
        while not (terminado or truncado):
            if int(buffer["total_transiciones_agregadas"]) < TRANSICIONES_ALEATORIAS:
                accion = rng.uniform(-1.0, 1.0, size=base.DIMENSION_ACCION_SAC).astype(np.float32)
            else:
                transiciones_politica = max(
                    0,
                    int(buffer["total_transiciones_agregadas"])
                    - TRANSICIONES_ALEATORIAS,
                )
                fraccion = min(
                    1.0,
                    transiciones_politica
                    / max(1, TRANSICIONES_DECAIMIENTO_RUIDO),
                )
                sigma = (
                    RUIDO_EXPLORACION_INICIAL
                    + fraccion
                    * (RUIDO_EXPLORACION_FINAL - RUIDO_EXPLORACION_INICIAL)
                )
                accion = accion_td3(actor, observacion, dispositivo, sigma, rng)
            siguiente, recompensa, terminado, truncado, info = base.ejecutar_paso_entorno_sac(entorno, accion)
            base.agregar_transicion_buffer_sac(
                buffer, observacion, accion, recompensa, siguiente, bool(terminado or truncado)
            )
            observacion = siguiente
            recompensa_ep += float(recompensa)
            if int(buffer["tamano"]) >= TAMANO_LOTE and int(buffer["total_transiciones_agregadas"]) >= TRANSICIONES_ALEATORIAS:
                numero_actualizacion += 1
                actualizar_td3(
                    actor, actor_obj, q1, q2, q1_obj, q2_obj,
                    optim_actor, optim_q1, optim_q2, buffer,
                    dispositivo, numero_actualizacion,
                )
        recompensas.append(recompensa_ep)
        exitos.append(int(entorno["resultado"] == "meta"))

        if episodio % INTERVALO_VALIDACION == 0 or episodio == episodios:
            val = validar_actor(actor, semillas_val, dispositivo)
            ultima_validacion = val
            criterio = criterio_validacion(val)
            nuevo = mejor_criterio is None or criterio > mejor_criterio
            if nuevo:
                mejor_criterio = criterio
                mejor_validacion = val
                mejor_episodio = episodio
                guardar_checkpoint(RUTA_MEJOR, actor, episodio, val, {"actualizaciones": numero_actualizacion})
            fila = {
                "episodio": episodio,
                "tasa_exito": val["tasa_exito"],
                "timeout": val["tasa_timeout"],
                "colision_dinamica": val["tasa_colision_dinamica"],
                "colision_estatica": val["tasa_colision_estatica"],
                "fuera_mapa": val["tasa_fuera_mapa"],
                "distancia_final_media": val["distancia_final_media"],
                "clearance_exitos": val["clearance_exitos"],
            }
            historial.append(fila)
            print(
                f"Episodios {episodio:4d}/{episodios:4d} | "
                f"R móvil={np.mean(recompensas[-25:]):9.2f} | "
                f"éxito train={100*np.mean(exitos[-25:]):6.2f}% | "
                f"validación={100*float(val['tasa_exito']):6.2f}% | "
                f"col. din={100*float(val['tasa_colision_dinamica']):5.2f}% | "
                f"{'NUEVO MEJOR' if nuevo else ''}"
            )

    if mejor_validacion is None:
        raise RuntimeError("No se realizó validación TD3")
    if ultima_validacion is None:
        raise RuntimeError("No existe validación final TD3")
    guardar_checkpoint(
        RUTA_FINAL, actor, episodios, ultima_validacion,
        {
            "actualizaciones": numero_actualizacion,
            "actor_objetivo_state_dict": actor_obj.state_dict(),
            "q1_state_dict": q1.state_dict(), "q2_state_dict": q2.state_dict(),
            "q1_objetivo_state_dict": q1_obj.state_dict(),
            "q2_objetivo_state_dict": q2_obj.state_dict(),
            "optimizador_actor": optim_actor.state_dict(),
            "optimizador_q1": optim_q1.state_dict(),
            "optimizador_q2": optim_q2.state_dict(),
        },
    )

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with RUTA_CSV.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        escritor.writeheader(); escritor.writerows(historial)

    figura = plt.figure(figsize=(12, 7))
    ax1 = figura.add_subplot(211)
    ax1.plot(range(1, len(recompensas)+1), recompensas)
    ax1.set_ylabel("Recompensa"); ax1.grid(True, alpha=.3)
    ax2 = figura.add_subplot(212)
    ax2.plot([h["episodio"] for h in historial], [100*h["tasa_exito"] for h in historial], marker="o")
    ax2.set_xlabel("Episodio"); ax2.set_ylabel("Éxito validación (%)"); ax2.grid(True, alpha=.3)
    figura.tight_layout(); figura.savefig(RUTA_GRAFICA, dpi=160)
    # La figura ya fue guardada. Se cierra para evitar que plt.show()
    # bloquee el programa después de terminar los 400 episodios.
    plt.close(figura)

    mejor = torch.load(RUTA_MEJOR, map_location=dispositivo, weights_only=False)
    actor.load_state_dict(mejor["actor_state_dict"], strict=True)
    diagnostico = ejecutar_episodio(actor, SEMILLA_DIAGNOSTICA, dispositivo)
    print("\n" + "=" * 80)
    print("RESULTADO DEL ENTRENAMIENTO TD3")
    print("=" * 80)
    print(f"Mejor episodio: {mejor_episodio}")
    print(f"Éxito validación: {100*float(mejor_validacion['tasa_exito']):.2f}%")
    print(f"Colisión dinámica: {100*float(mejor_validacion['tasa_colision_dinamica']):.2f}%")
    print(f"Colisión estática: {100*float(mejor_validacion['tasa_colision_estatica']):.2f}%")
    print(f"Timeout: {100*float(mejor_validacion['tasa_timeout']):.2f}%")
    print(f"Diagnóstico {SEMILLA_DIAGNOSTICA}: {diagnostico['resultado']}")
    print(f"Mejor actor: {RUTA_MEJOR}")
    print("RESULTADO TÉCNICO: ENTRENAMIENTO TD3 COMPLETADO")

def analizar_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline TD3 para el mismo entorno de navegación")
    p.add_argument("--solo-verificacion", action="store_true")
    p.add_argument("--episodios", type=int, default=EPISODIOS_PREDETERMINADOS)
    p.add_argument("--sin-graficas", action="store_true")
    return p.parse_args()

def main() -> None:
    args = analizar_argumentos()
    verificar_integracion()
    if not args.solo_verificacion:
        entrenar(args.episodios, not args.sin_graficas)

if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# IMPORTACIÓN DEL ENTORNO BASE
# ==========================================================

DIRECTORIO = Path(__file__).resolve().parent
CANDIDATOS_BASE = (
    "SAC_predictivo_entrenamiento_completo.py",
    "SAC_predictivo_entrenamiento_completo(3).py",
    "Comparacion_final_DWA_SAC_reactivo_SAC_predictivo.py",
)


def localizar_base() -> Path:
    for nombre in CANDIDATOS_BASE:
        ruta = DIRECTORIO / nombre
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "Coloque este archivo junto a SAC_predictivo_entrenamiento_completo.py"
    )


RUTA_BASE = localizar_base()
_USAR_AGG = os.environ.get("MPLBACKEND", "").lower() == "agg"
_USAR_ORIGINAL = matplotlib.use
if _USAR_AGG:
    def _usar_backend(backend, *args, **kwargs):
        if str(backend).lower() == "tkagg":
            return _USAR_ORIGINAL("Agg", force=True)
        return _USAR_ORIGINAL(backend, *args, **kwargs)
    matplotlib.use = _usar_backend

_spec = importlib.util.spec_from_file_location("base_td3_navegacion", RUTA_BASE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"No se pudo importar {RUTA_BASE}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
matplotlib.use = _USAR_ORIGINAL
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURACIÓN TD3
# ==========================================================

NOMBRE_METODO = "A* + TD3 estabilizado desde cero"
VARIANTE = "td3_reactivo_tercera_corrida_desde_cero"
EPISODIOS_PREDETERMINADOS = 400
SEMILLA_ENTRENAMIENTO = 166000
SEMILLA_VALIDACION = 167000
NUMERO_SEMILLAS_VALIDACION = 50
INTERVALO_VALIDACION = 20
SEMILLA_DIAGNOSTICA = 167100

CAPACIDAD_BUFFER = 300_000
TRANSICIONES_ALEATORIAS = 10_000
TAMANO_LOTE = 256
GAMMA = 0.99
TAU = 0.0025
LR_ACTOR = 5e-5
LR_CRITICOS = 1e-4
RUIDO_EXPLORACION_INICIAL = 0.12
RUIDO_EXPLORACION_FINAL = 0.03
TRANSICIONES_DECAIMIENTO_RUIDO = 60_000
RUIDO_POLITICA_OBJETIVO = 0.10
LIMITE_RUIDO_OBJETIVO = 0.25
RETARDO_ACTOR = 2
RECORTE_GRADIENTE = 5.0

DIRECTORIO_RESULTADOS = Path("resultados_td3/tercera_corrida_td3_desde_cero")
RUTA_MEJOR = DIRECTORIO_RESULTADOS / "checkpoint_mejor_actor_td3.pt"
RUTA_FINAL = DIRECTORIO_RESULTADOS / "checkpoint_final_td3.pt"
RUTA_CSV = DIRECTORIO_RESULTADOS / "validaciones_periodicas_50_semillas.csv"
RUTA_GRAFICA = DIRECTORIO_RESULTADOS / "entrenamiento_td3.png"

# ==========================================================
# REDES
# ==========================================================


def fijar_semillas(semilla: int) -> None:
    random.seed(int(semilla))
    np.random.seed(int(semilla))
    torch.manual_seed(int(semilla))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(semilla))


def crear_actor_td3(dispositivo: torch.device) -> nn.ModuleDict:
    actor = nn.ModuleDict({
        "rama_cnn": base.crear_rama_cnn_sac(dispositivo),
        "rama_mlp": base.crear_rama_mlp_escalares_sac(dispositivo),
        "red_compartida": base.crear_red_compartida_actor_sac(dispositivo),
        "cabeza_accion": nn.Linear(
            base.NEURONAS_CAPA_2_ACTOR_SAC,
            base.DIMENSION_ACCION_SAC,
        ).to(dispositivo),
    }).to(dispositivo)
    return actor


def ejecutar_actor_td3(
    actor: nn.ModuleDict,
    parche: torch.Tensor,
    escalares: torch.Tensor,
) -> torch.Tensor:
    cnn = actor["rama_cnn"](parche)
    mlp = actor["rama_mlp"](escalares)
    fusion = base.fusionar_caracteristicas_sac(cnn, mlp)
    oculto = actor["red_compartida"](fusion)
    return torch.tanh(actor["cabeza_accion"](oculto))


def tensores_observacion(
    observacion: Mapping[str, np.ndarray],
    dispositivo: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    parche = torch.as_tensor(
        observacion["parche"], dtype=torch.float32, device=dispositivo
    ).unsqueeze(0)
    escalares = torch.as_tensor(
        observacion["escalares"], dtype=torch.float32, device=dispositivo
    ).unsqueeze(0)
    return parche, escalares


def accion_td3(
    actor: nn.ModuleDict,
    observacion: Mapping[str, np.ndarray],
    dispositivo: torch.device,
    ruido: float = 0.0,
    generador: np.random.Generator | None = None,
) -> np.ndarray:
    modo = actor.training
    actor.eval()
    with torch.no_grad():
        parche, escalares = tensores_observacion(observacion, dispositivo)
        accion = ejecutar_actor_td3(actor, parche, escalares)[0].cpu().numpy()
    actor.train(modo)
    if ruido > 0.0:
        if generador is None:
            generador = np.random.default_rng()
        accion = accion + generador.normal(0.0, ruido, size=accion.shape)
    return np.clip(accion, -1.0, 1.0).astype(np.float32)


def actualizar_objetivo(red: nn.Module, objetivo: nn.Module, tau: float = TAU) -> None:
    with torch.no_grad():
        for p, pt in zip(red.parameters(), objetivo.parameters()):
            pt.mul_(1.0 - tau).add_(p, alpha=tau)

# ==========================================================
# ACTUALIZACIÓN TD3
# ==========================================================


def actualizar_td3(
    actor: nn.ModuleDict,
    actor_objetivo: nn.ModuleDict,
    q1: nn.ModuleDict,
    q2: nn.ModuleDict,
    q1_objetivo: nn.ModuleDict,
    q2_objetivo: nn.ModuleDict,
    optim_actor: torch.optim.Optimizer,
    optim_q1: torch.optim.Optimizer,
    optim_q2: torch.optim.Optimizer,
    buffer: Dict[str, object],
    dispositivo: torch.device,
    numero_actualizacion: int,
) -> Dict[str, float]:
    lote = base.muestrear_lote_buffer_sac(
        buffer=buffer,
        tamano_lote=TAMANO_LOTE,
        dispositivo=dispositivo,
    )
    p = lote["parches"]
    e = lote["escalares"]
    a = lote["acciones"]
    r = lote["recompensas"]
    sp = lote["siguientes_parches"]
    se = lote["siguientes_escalares"]
    d = lote["terminados"]

    with torch.no_grad():
        siguiente_accion = ejecutar_actor_td3(actor_objetivo, sp, se)
        ruido = torch.randn_like(siguiente_accion) * RUIDO_POLITICA_OBJETIVO
        ruido = ruido.clamp(-LIMITE_RUIDO_OBJETIVO, LIMITE_RUIDO_OBJETIVO)
        siguiente_accion = (siguiente_accion + ruido).clamp(-1.0, 1.0)
        q1t = base.ejecutar_critico_sac(sp, se, siguiente_accion, q1_objetivo)["valor_q"]
        q2t = base.ejecutar_critico_sac(sp, se, siguiente_accion, q2_objetivo)["valor_q"]
        objetivo = r + GAMMA * (1.0 - d) * torch.minimum(q1t, q2t)

    q1_pred = base.ejecutar_critico_sac(p, e, a, q1)["valor_q"]
    q2_pred = base.ejecutar_critico_sac(p, e, a, q2)["valor_q"]
    perdida_q1 = F.smooth_l1_loss(q1_pred, objetivo)
    perdida_q2 = F.smooth_l1_loss(q2_pred, objetivo)

    optim_q1.zero_grad(set_to_none=True)
    perdida_q1.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), RECORTE_GRADIENTE)
    optim_q1.step()

    optim_q2.zero_grad(set_to_none=True)
    perdida_q2.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), RECORTE_GRADIENTE)
    optim_q2.step()

    perdida_actor = float("nan")
    actor_actualizado = numero_actualizacion % RETARDO_ACTOR == 0
    if actor_actualizado:
        acciones_actor = ejecutar_actor_td3(actor, p, e)
        perdida = -base.ejecutar_critico_sac(p, e, acciones_actor, q1)["valor_q"].mean()
        optim_actor.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), RECORTE_GRADIENTE)
        optim_actor.step()
        perdida_actor = float(perdida.detach().cpu())
        actualizar_objetivo(actor, actor_objetivo)
        actualizar_objetivo(q1, q1_objetivo)
        actualizar_objetivo(q2, q2_objetivo)

    return {
        "perdida_q1": float(perdida_q1.detach().cpu()),
        "perdida_q2": float(perdida_q2.detach().cpu()),
        "perdida_actor": perdida_actor,
        "actor_actualizado": float(actor_actualizado),
    }

# ==========================================================
# EVALUACIÓN
# ==========================================================


def ejecutar_episodio(
    actor: nn.ModuleDict,
    semilla: int,
    dispositivo: torch.device,
) -> Dict[str, float | int | str]:
    entorno, observacion = base.reiniciar_entorno_sac(semilla)
    terminado = truncado = False
    recompensa_total = 0.0
    ultima_info: Dict[str, object] = {}
    while not (terminado or truncado):
        accion = accion_td3(actor, observacion, dispositivo)
        observacion, recompensa, terminado, truncado, ultima_info = (
            base.ejecutar_paso_entorno_sac(entorno, accion)
        )
        recompensa_total += float(recompensa)
    registro = entorno["registro"]
    clearances = registro.get("clearances", [])
    clearance_min = float(np.min(clearances)) if len(clearances) else float("inf")
    estado = entorno["estado_robot"]
    distancia_final = float(base.distancia_entre_puntos((estado[0], estado[1]), entorno["meta"]))
    return {
        "semilla": int(semilla),
        "resultado": str(entorno["resultado"]),
        "pasos": int(entorno["paso_actual"]),
        "recompensa": recompensa_total,
        "distancia_final": distancia_final,
        "clearance_minimo": clearance_min,
    }


def validar_actor(
    actor: nn.ModuleDict,
    semillas: Sequence[int],
    dispositivo: torch.device,
) -> Dict[str, object]:
    resultados = [ejecutar_episodio(actor, s, dispositivo) for s in semillas]
    conteos = {k: 0 for k in ("meta", "colision_estatica", "colision_dinamica", "fuera_mapa", "timeout")}
    for fila in resultados:
        conteos[str(fila["resultado"])] = conteos.get(str(fila["resultado"]), 0) + 1
    exitos = [f for f in resultados if f["resultado"] == "meta"]
    n = len(resultados)
    return {
        "resultados": resultados,
        "conteos": conteos,
        "tasa_exito": len(exitos) / n,
        "tasa_timeout": conteos.get("timeout", 0) / n,
        "tasa_colision_dinamica": conteos.get("colision_dinamica", 0) / n,
        "tasa_colision_estatica": conteos.get("colision_estatica", 0) / n,
        "tasa_fuera_mapa": conteos.get("fuera_mapa", 0) / n,
        "distancia_final_media": float(np.mean([f["distancia_final"] for f in resultados])),
        "clearance_exitos": float(np.mean([f["clearance_minimo"] for f in exitos])) if exitos else -float("inf"),
    }


def criterio_validacion(v: Mapping[str, object]) -> Tuple[float, ...]:
    return (
        float(v["tasa_exito"]),
        -float(v["tasa_timeout"]),
        -float(v["tasa_colision_dinamica"]),
        -float(v["tasa_colision_estatica"]),
        -float(v["tasa_fuera_mapa"]),
        -float(v["distancia_final_media"]),
        float(v["clearance_exitos"]),
    )


def guardar_checkpoint(
    ruta: Path,
    actor: nn.ModuleDict,
    episodio: int,
    validacion: Mapping[str, object],
    extras: Mapping[str, object],
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tipo": "mejor_actor_td3" if "mejor" in ruta.name else "td3_entrenamiento_completo",
        "variante": VARIANTE,
        "metodo": NOMBRE_METODO,
        "episodio": int(episodio),
        "actor_state_dict": actor.state_dict(),
        "validacion": {k: v for k, v in validacion.items() if k != "resultados"},
        "configuracion": {
            "gamma": GAMMA, "tau": TAU, "lr_actor": LR_ACTOR,
            "lr_criticos": LR_CRITICOS, "policy_delay": RETARDO_ACTOR,
            "policy_noise": RUIDO_POLITICA_OBJETIVO,
            "noise_clip": LIMITE_RUIDO_OBJETIVO,
            "observacion": "reactiva_actual",
            "perdida_criticos": "smooth_l1_huber",
            "transiciones_aleatorias": TRANSICIONES_ALEATORIAS,
            "transiciones_decaimiento_ruido": TRANSICIONES_DECAIMIENTO_RUIDO,
            "presupuesto_episodios": int(EPISODIOS_PREDETERMINADOS),
            "actor_inicial": "inicializacion_aleatoria_desde_cero",
            "semilla_inicializacion": int(SEMILLA_ENTRENAMIENTO),
        },
        **dict(extras),
    }, ruta)

# ==========================================================
# VERIFICACIÓN Y ENTRENAMIENTO
# ==========================================================

def verificar_integracion() -> None:
    dispositivo = base.DISPOSITIVO_SAC
    actor = crear_actor_td3(dispositivo)
    q1, q2 = base.crear_dos_criticos_sac(dispositivo)
    entorno, obs = base.reiniciar_entorno_sac(base.SEMILLA)
    a = accion_td3(actor, obs, dispositivo)
    siguiente, r, terminado, truncado, info = base.ejecutar_paso_entorno_sac(entorno, a, pasos_maximos=5)
    parche, escalares = tensores_observacion(obs, dispositivo)
    at = ejecutar_actor_td3(actor, parche, escalares)
    qt = base.ejecutar_critico_sac(parche, escalares, at, q1)["valor_q"]
    comprobaciones = {
        "accion_forma": a.shape == (base.DIMENSION_ACCION_SAC,),
        "accion_rango": bool(np.all(a >= -1.0) and np.all(a <= 1.0)),
        "observacion_compatible": siguiente["parche"].shape == obs["parche"].shape,
        "recompensa_finita": np.isfinite(r),
        "q_finito": bool(torch.isfinite(qt).all()),
        "entorno_reactivo": info.get("modo_sac", "reactivo") != "predictivo",
    }
    print("\n" + "=" * 80)
    print("VERIFICACIÓN TD3")
    print("=" * 80)
    for k, v in comprobaciones.items():
        print(f"{k}: {v}")
    if not all(comprobaciones.values()):
        raise RuntimeError("Falló la verificación TD3")
    print("RESULTADO DE VERIFICACIÓN: TODO CORRECTO")

def entrenar(episodios: int, mostrar: bool) -> None:
    fijar_semillas(SEMILLA_ENTRENAMIENTO)
    dispositivo = base.DISPOSITIVO_SAC
    # Inicialización completamente aleatoria y reproducible.
    # No se carga ningún actor de corridas TD3 anteriores.
    actor = crear_actor_td3(dispositivo)
    actor_obj = copy.deepcopy(actor).to(dispositivo).eval()
    q1, q2 = base.crear_dos_criticos_sac(dispositivo)
    q1_obj, q2_obj = base.crear_dos_criticos_objetivo_sac(q1, q2, dispositivo)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optim_q1 = torch.optim.Adam(q1.parameters(), lr=LR_CRITICOS)
    optim_q2 = torch.optim.Adam(q2.parameters(), lr=LR_CRITICOS)
    buffer = base.crear_buffer_repeticion_sac(CAPACIDAD_BUFFER, SEMILLA_ENTRENAMIENTO + 99)
    rng = np.random.default_rng(SEMILLA_ENTRENAMIENTO + 123)
    semillas_val = list(range(SEMILLA_VALIDACION, SEMILLA_VALIDACION + NUMERO_SEMILLAS_VALIDACION))

    historial: List[Dict[str, object]] = []
    recompensas: List[float] = []
    exitos: List[int] = []
    numero_actualizacion = 0
    mejor_criterio: Tuple[float, ...] | None = None
    mejor_validacion: Dict[str, object] | None = None
    mejor_episodio = 0
    ultima_validacion: Dict[str, object] | None = None

    print("\n" + "=" * 80)
    print("TERCERA CORRIDA TD3 ESTABILIZADA DESDE CERO")
    print("=" * 80)
    print(f"Dispositivo: {dispositivo}")
    print("Actor inicial: inicialización aleatoria desde cero")
    print(f"Semilla de inicialización: {SEMILLA_ENTRENAMIENTO}")
    print("Actor, críticos, objetivos, optimizadores y buffer: creados desde cero")
    print(f"Episodios: {episodios}")
    print(f"Semillas de entrenamiento: {SEMILLA_ENTRENAMIENTO} a {SEMILLA_ENTRENAMIENTO + episodios - 1}")
    print(f"Semillas nuevas de validación fija: {SEMILLA_VALIDACION} a {SEMILLA_VALIDACION + NUMERO_SEMILLAS_VALIDACION - 1}")

    for episodio in range(1, episodios + 1):
        entorno, observacion = base.reiniciar_entorno_sac(SEMILLA_ENTRENAMIENTO + episodio - 1)
        terminado = truncado = False
        recompensa_ep = 0.0
        while not (terminado or truncado):
            if int(buffer["total_transiciones_agregadas"]) < TRANSICIONES_ALEATORIAS:
                accion = rng.uniform(-1.0, 1.0, size=base.DIMENSION_ACCION_SAC).astype(np.float32)
            else:
                transiciones_politica = max(
                    0,
                    int(buffer["total_transiciones_agregadas"])
                    - TRANSICIONES_ALEATORIAS,
                )
                fraccion = min(
                    1.0,
                    transiciones_politica
                    / max(1, TRANSICIONES_DECAIMIENTO_RUIDO),
                )
                sigma = (
                    RUIDO_EXPLORACION_INICIAL
                    + fraccion
                    * (RUIDO_EXPLORACION_FINAL - RUIDO_EXPLORACION_INICIAL)
                )
                accion = accion_td3(actor, observacion, dispositivo, sigma, rng)
            siguiente, recompensa, terminado, truncado, info = base.ejecutar_paso_entorno_sac(entorno, accion)
            base.agregar_transicion_buffer_sac(
                buffer, observacion, accion, recompensa, siguiente, bool(terminado or truncado)
            )
            observacion = siguiente
            recompensa_ep += float(recompensa)
            if int(buffer["tamano"]) >= TAMANO_LOTE and int(buffer["total_transiciones_agregadas"]) >= TRANSICIONES_ALEATORIAS:
                numero_actualizacion += 1
                actualizar_td3(
                    actor, actor_obj, q1, q2, q1_obj, q2_obj,
                    optim_actor, optim_q1, optim_q2, buffer,
                    dispositivo, numero_actualizacion,
                )
        recompensas.append(recompensa_ep)
        exitos.append(int(entorno["resultado"] == "meta"))

        if episodio % INTERVALO_VALIDACION == 0 or episodio == episodios:
            val = validar_actor(actor, semillas_val, dispositivo)
            ultima_validacion = val
            criterio = criterio_validacion(val)
            nuevo = mejor_criterio is None or criterio > mejor_criterio
            if nuevo:
                mejor_criterio = criterio
                mejor_validacion = val
                mejor_episodio = episodio
                guardar_checkpoint(RUTA_MEJOR, actor, episodio, val, {"actualizaciones": numero_actualizacion})
            fila = {
                "episodio": episodio,
                "tasa_exito": val["tasa_exito"],
                "timeout": val["tasa_timeout"],
                "colision_dinamica": val["tasa_colision_dinamica"],
                "colision_estatica": val["tasa_colision_estatica"],
                "fuera_mapa": val["tasa_fuera_mapa"],
                "distancia_final_media": val["distancia_final_media"],
                "clearance_exitos": val["clearance_exitos"],
            }
            historial.append(fila)
            print(
                f"Episodios {episodio:4d}/{episodios:4d} | "
                f"R móvil={np.mean(recompensas[-25:]):9.2f} | "
                f"éxito train={100*np.mean(exitos[-25:]):6.2f}% | "
                f"validación={100*float(val['tasa_exito']):6.2f}% | "
                f"col. din={100*float(val['tasa_colision_dinamica']):5.2f}% | "
                f"{'NUEVO MEJOR' if nuevo else ''}"
            )

    if mejor_validacion is None:
        raise RuntimeError("No se realizó validación TD3")
    if ultima_validacion is None:
        raise RuntimeError("No existe validación final TD3")
    guardar_checkpoint(
        RUTA_FINAL, actor, episodios, ultima_validacion,
        {
            "actualizaciones": numero_actualizacion,
            "actor_objetivo_state_dict": actor_obj.state_dict(),
            "q1_state_dict": q1.state_dict(), "q2_state_dict": q2.state_dict(),
            "q1_objetivo_state_dict": q1_obj.state_dict(),
            "q2_objetivo_state_dict": q2_obj.state_dict(),
            "optimizador_actor": optim_actor.state_dict(),
            "optimizador_q1": optim_q1.state_dict(),
            "optimizador_q2": optim_q2.state_dict(),
        },
    )

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with RUTA_CSV.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        escritor.writeheader(); escritor.writerows(historial)

    figura = plt.figure(figsize=(12, 7))
    ax1 = figura.add_subplot(211)
    ax1.plot(range(1, len(recompensas)+1), recompensas)
    ax1.set_ylabel("Recompensa"); ax1.grid(True, alpha=.3)
    ax2 = figura.add_subplot(212)
    ax2.plot([h["episodio"] for h in historial], [100*h["tasa_exito"] for h in historial], marker="o")
    ax2.set_xlabel("Episodio"); ax2.set_ylabel("Éxito validación (%)"); ax2.grid(True, alpha=.3)
    figura.tight_layout(); figura.savefig(RUTA_GRAFICA, dpi=160)
    # La figura ya fue guardada. Se cierra para evitar que plt.show()
    # bloquee el programa después de terminar los 400 episodios.
    plt.close(figura)

    mejor = torch.load(RUTA_MEJOR, map_location=dispositivo, weights_only=False)
    actor.load_state_dict(mejor["actor_state_dict"], strict=True)
    diagnostico = ejecutar_episodio(actor, SEMILLA_DIAGNOSTICA, dispositivo)
    print("\n" + "=" * 80)
    print("RESULTADO DEL ENTRENAMIENTO TD3")
    print("=" * 80)
    print(f"Mejor episodio: {mejor_episodio}")
    print(f"Éxito validación: {100*float(mejor_validacion['tasa_exito']):.2f}%")
    print(f"Colisión dinámica: {100*float(mejor_validacion['tasa_colision_dinamica']):.2f}%")
    print(f"Colisión estática: {100*float(mejor_validacion['tasa_colision_estatica']):.2f}%")
    print(f"Timeout: {100*float(mejor_validacion['tasa_timeout']):.2f}%")
    print(f"Diagnóstico {SEMILLA_DIAGNOSTICA}: {diagnostico['resultado']}")
    print(f"Mejor actor: {RUTA_MEJOR}")
    print("RESULTADO TÉCNICO: ENTRENAMIENTO TD3 COMPLETADO")

def analizar_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline TD3 para el mismo entorno de navegación")
    p.add_argument("--solo-verificacion", action="store_true")
    p.add_argument("--episodios", type=int, default=EPISODIOS_PREDETERMINADOS)
    p.add_argument("--sin-graficas", action="store_true")
    return p.parse_args()

def main() -> None:
    args = analizar_argumentos()
    verificar_integracion()
    if not args.solo_verificacion:
        entrenar(args.episodios, not args.sin_graficas)

if __name__ == "__main__":
    main()
