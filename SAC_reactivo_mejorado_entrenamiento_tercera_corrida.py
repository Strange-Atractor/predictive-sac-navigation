import heapq
import math
import csv
import time
import random
import torch
from collections import Counter
from pathlib import Path
import csv
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, Rectangle
import torch.nn as nn
# ==========================================================
# VISUALIZACIÓN EN TIEMPO REAL
# ==========================================================
ANIMAR_SIMULACION = True
# Con PAUSA_ANIMACION = DT la reproducción es
# aproximadamente en tiempo real.
PAUSA_ANIMACION = 0.01
# Mantiene visibles los rastros recorridos.
MOSTRAR_RASTROS_DINAMICOS = True
# Muestra gráficas adicionales después de la animación.
MOSTRAR_GRAFICAS_FINALES = False
# ==========================================================
# PARÁMETROS DEL MAPA
# ==========================================================
ANCHO_MAPA = 15.0
ALTO_MAPA = 15.0
# ==========================================================
# PARÁMETROS DEL ROBOT
# ==========================================================
RADIO_ROBOT = 0.40
VELOCIDAD_MAXIMA = 1.0
VELOCIDAD_ANGULAR_MAXIMA = 1.2
DT = 0.1
DISTANCIA_META = 0.5
# ==========================================================
# MARGEN DE SEGURIDAD
# ==========================================================
MARGEN_MANIOBRA = 0.25
RADIO_INFLADO = RADIO_ROBOT + MARGEN_MANIOBRA
#  ==========================================================
# Obstaculos aleatorios y reproducibilidad
# ==========================================================
SEMILLA = 10
NUMERO_OBSTACULOS = 7
ANCHO_MIN_OBSTACULO = 0.6
ANCHO_MAX_OBSTACULO = 2.5
ALTO_MIN_OBSTACULO = 0.6
ALTO_MAX_OBSTACULO = 2.5
SEPARACION_OBSTACULOS = 0.8
MARGEN_BORDE_OBSTACULOS = 0.5
#  ==========================================================
# Punto inicial aleatorio del robot y meta aleatoria
# ==========================================================
MARGEN_BORDE_PUNTOS = RADIO_INFLADO + 0.1
DISTANCIA_MINIMA_INICIO_META = 7.0
INTENTOS_MAXIMOS_PUNTO = 1000
INTENTOS_MAXIMOS_INICIO_META = 500
# ==========================================================
# PARÁMETROS DE LA REJILLA
# ==========================================================
RESOLUCION_REJILLA = 0.25
NUMERO_COLUMNAS = int(
    math.ceil(ANCHO_MAPA / RESOLUCION_REJILLA)
)
NUMERO_FILAS = int(
    math.ceil(ALTO_MAPA / RESOLUCION_REJILLA)
)
MOVIMIENTOS = [
    (-1,  0),
    ( 1,  0),
    ( 0, -1),
    ( 0,  1),
    (-1, -1),
    (-1,  1),
    ( 1, -1),
    ( 1,  1),
]
INTENTOS_MAXIMOS_ESCENARIO = 100
# ==========================================================
# CONTROLADOR DE SEGUIMIENTO DE RUTA
# ==========================================================

DISTANCIA_LOOKAHEAD = 1.0
VELOCIDAD_NOMINAL = 0.80
GANANCIA_ANGULAR = 2.5
DISTANCIA_REDUCCION_VELOCIDAD = 1.0
PASOS_MAXIMOS_SEGUIMIENTO = 400


# ==========================================================
# OBSTÁCULOS DINÁMICOS
# ==========================================================

NUMERO_OBSTACULOS_DINAMICOS = 3
RADIO_MIN_OBSTACULO_DINAMICO = 0.25
RADIO_MAX_OBSTACULO_DINAMICO = 0.40
VELOCIDAD_MIN_OBSTACULO_DINAMICO = 0.25
VELOCIDAD_MAX_OBSTACULO_DINAMICO = 0.60
DISTANCIA_MINIMA_CRUCE = 1.0
DISTANCIA_MAXIMA_CRUCE = 2.0
MARGEN_DINAMICO_ESTATICO = 0.10
SEPARACION_OBSTACULOS_DINAMICOS = 0.20
INTENTOS_MAXIMOS_DINAMICOS = 1000
PASOS_SIMULACION_DINAMICA = 200
DESFASE_SEMILLA_DINAMICA = 50000

# ==========================================================
# SINCRONIZACIÓN TEMPORAL DE OBSTÁCULOS DINÁMICOS
# ==========================================================

DESFASE_MINIMO_TIEMPO_CRUCE = -0.40
DESFASE_MAXIMO_TIEMPO_CRUCE = 0.40

# ==========================================================
# SUBMETA LOCAL Y PERCEPCIÓN DINÁMICA
# ==========================================================

DISTANCIA_SUBMETA_LOCAL = 1.50

VENTANA_BUSQUEDA_PROGRESO = 25

RADIO_PERCEPCION_DINAMICA = 4.0

MAX_OBSTACULOS_DINAMICOS_OBSERVADOS = 5

# ==========================================================
# OBSERVACIÓN ESPACIAL SAC
# ==========================================================

TAMANO_PARCHE_SAC = 6.0

RESOLUCION_PARCHE_SAC = 32

CANALES_PARCHE_SAC = 2

DIMENSION_ESCALARES_SAC = 6
# ==========================================================
# DIMENSIONES DE SAC
# ==========================================================

DIMENSION_ACCION_SAC = 2
DIMENSION_OBSERVACION_SAC = (
    6
    + 7
    * MAX_OBSTACULOS_DINAMICOS_OBSERVADOS
)
# ==========================================================
# ARQUITECTURA DEL ACTOR SAC
# ==========================================================
# ==========================================================
# ARQUITECTURA DEL ACTOR SAC
# ==========================================================

# Rama CNN del parche.

CANALES_CNN_1_SAC = 16
CANALES_CNN_2_SAC = 32
CANALES_CNN_3_SAC = 64

DIMENSION_CARACTERISTICAS_CNN_SAC = 128

# Rama MLP de los escalares.

NEURONAS_MLP_ESCALARES_SAC = 64

DIMENSION_CARACTERISTICAS_MLP_SAC = 64


# Fusión de ambas ramas.

DIMENSION_CARACTERISTICAS_FUSIONADAS_SAC = (
    DIMENSION_CARACTERISTICAS_CNN_SAC
    + DIMENSION_CARACTERISTICAS_MLP_SAC
)


# Capas posteriores del actor.
NEURONAS_CAPA_1_ACTOR_SAC = 256
NEURONAS_CAPA_2_ACTOR_SAC = 256
LOG_DESVIACION_MINIMA_SAC = -20.0
LOG_DESVIACION_MAXIMA_SAC = 2.0
EPSILON_LOG_PROBABILIDAD_SAC = 1e-6
# ==========================================================
# OPTIMIZADOR DEL ACTOR SAC
# ==========================================================

TASA_APRENDIZAJE_ACTOR_SAC = 3e-4
BETA_1_ADAM_ACTOR_SAC = 0.9
BETA_2_ADAM_ACTOR_SAC = 0.999
EPSILON_ADAM_ACTOR_SAC = 1e-8
NEURONAS_RAMA_ACCION_CRITICO_SAC = 64
DIMENSION_CARACTERISTICAS_ACCION_CRITICO_SAC = 64
DIMENSION_ENTRADA_RED_Q_SAC = (
    DIMENSION_CARACTERISTICAS_FUSIONADAS_SAC
    + DIMENSION_CARACTERISTICAS_ACCION_CRITICO_SAC
)
NEURONAS_CAPA_1_CRITICO_SAC = 256
NEURONAS_CAPA_2_CRITICO_SAC = 256
DIMENSION_SALIDA_CRITICO_SAC = 1
CAPACIDAD_BUFFER_REPETICION_SAC = 50000

TAMANO_LOTE_SAC = 256

# ==========================================================
# ACTUALIZACIÓN DE LOS CRÍTICOS OBJETIVO
# ==========================================================

TAU_POLYAK_SAC = 0.005

# ==========================================================
# ENTRENAMIENTO DE LOS CRÍTICOS SAC
# ==========================================================

TASA_APRENDIZAJE_CRITICOS_SAC = 3e-4
BETA_1_ADAM_CRITICOS_SAC = 0.9
BETA_2_ADAM_CRITICOS_SAC = 0.999
EPSILON_ADAM_CRITICOS_SAC = 1e-8
FACTOR_DESCUENTO_SAC = 0.99
COEFICIENTE_ENTROPIA_INICIAL_SAC = 0.20

# ==========================================================
# AJUSTE AUTOMÁTICO DE LA TEMPERATURA SAC
# ==========================================================

ENTROPIA_OBJETIVO_SAC = -float(
    DIMENSION_ACCION_SAC
)

TASA_APRENDIZAJE_ALPHA_SAC = 3e-4

BETA_1_ADAM_ALPHA_SAC = 0.9

BETA_2_ADAM_ALPHA_SAC = 0.999

EPSILON_ADAM_ALPHA_SAC = 1e-8

# RECOMPENSA SAC
# ==========================================================

# ==========================================================
# RECOMPENSA SAC MEJORADA
# ==========================================================

PESO_SAC_PROGRESO = 2.0

PESO_SAC_ALINEACION = 0.25

PESO_SAC_RUTA = 0.50

PESO_SAC_SUAVIDAD = 0.10


# ----------------------------------------------------------
# SEGURIDAD ESTÁTICA
# ----------------------------------------------------------

CLEARANCE_OBJETIVO_SAC_ESTATICO = 0.75

PESO_SAC_SEGURIDAD_ESTATICA = 1.50


# ----------------------------------------------------------
# SEGURIDAD DINÁMICA
# ----------------------------------------------------------

CLEARANCE_OBJETIVO_SAC_DINAMICO = 1.20

PESO_SAC_SEGURIDAD_DINAMICA = 4.00

PESO_SAC_VELOCIDAD_RIESGOSA_DINAMICA = 2.50


# ----------------------------------------------------------
# COMPATIBILIDAD CON PRUEBAS ANTERIORES
# ----------------------------------------------------------

PESO_SAC_SEGURIDAD = 1.50


# ----------------------------------------------------------
# PENALIZACIONES GENERALES
# ----------------------------------------------------------

PENALIZACION_SAC_ESTANCAMIENTO = 0.10

PENALIZACION_SAC_PASO = 0.01


# ----------------------------------------------------------
# RECOMPENSAS TERMINALES
# ----------------------------------------------------------
PESO_SAC_PROGRESO = 2.0
PESO_SAC_ALINEACION = 0.25
PESO_SAC_SEGURIDAD = 1.50
PESO_SAC_RUTA = 0.50
PESO_SAC_SUAVIDAD = 0.10

PENALIZACION_SAC_ESTANCAMIENTO = 0.10
PENALIZACION_SAC_PASO = 0.01

RECOMPENSA_SAC_META = 100.0
RECOMPENSA_SAC_COLISION = -100.0
RECOMPENSA_SAC_FUERA_MAPA = -100.0
RECOMPENSA_SAC_TIMEOUT = -10.0
RECOMPENSA_SAC_META = 100.0

RECOMPENSA_SAC_COLISION_ESTATICA = -100.0

RECOMPENSA_SAC_COLISION_DINAMICA = -180.0

RECOMPENSA_SAC_FUERA_MAPA = -100.0

RECOMPENSA_SAC_TIMEOUT = -10.0


# Compatibilidad con verificaciones anteriores.

RECOMPENSA_SAC_COLISION = (
    RECOMPENSA_SAC_COLISION_DINAMICA
)
# ==========================================================
# DYNAMIC WINDOW APPROACH
# ==========================================================

VELOCIDAD_MINIMA_DWA = 0.0

ACELERACION_LINEAL_MAXIMA_DWA = 2.0
ACELERACION_ANGULAR_MAXIMA_DWA = 4.0

HORIZONTE_PREDICCION_DWA = 1.6

NUMERO_MUESTRAS_V_DWA = 3
NUMERO_MUESTRAS_OMEGA_DWA = 7

CLEARANCE_OBJETIVO_DWA = 0.75
CLEARANCE_SEGURIDAD_DWA = 0.20

VELOCIDAD_ESTANCAMIENTO_DWA = 0.05
PESO_DWA_ESTANCAMIENTO = 0.75
ERROR_RUTA_OBJETIVO_DWA = 0.75

PESO_DWA_PROGRESO = 3.0
PESO_DWA_ALINEACION = 1.0
PESO_DWA_CLEARANCE = 2.5
PESO_DWA_VELOCIDAD = 0.40
PESO_DWA_RUTA = 1.20
PESO_DWA_SUAVIDAD = 0.25
# ==========================================================
# CICLO DE ENTRENAMIENTO SAC
# ==========================================================

TRANSICIONES_ALEATORIAS_INICIALES_SAC = 5000

ACTUALIZACIONES_POR_PASO_SAC = 1
# ==========================================================
# BENCHMARK MULTISEMILLA
# ==========================================================

SEMILLAS_DESARROLLO = list(
    range(
        10,
        15,
    )
)

METODOS_BENCHMARK = [
    "lookahead",
    "dwa",
]

CARPETA_RESULTADOS = Path(
    "resultados_benchmark"
)

SEMILLAS_FIGURAS_TRAYECTORIA = SEMILLAS_DESARROLLO

GUARDAR_TRAYECTORIAS = True
GUARDAR_FIGURAS_TRAYECTORIA = True
MOSTRAR_FIGURAS_BENCHMARK = False

# ==========================================================
# ENTRENAMIENTO GLOBAL DEL AGENTE SAC
# ==========================================================

NUMERO_EPISODIOS_ENTRENAMIENTO_SAC = 500

VENTANA_PROMEDIO_MOVIL_SAC = 25

FRECUENCIA_IMPRESION_ENTRENAMIENTO_SAC = 10

# ==========================================================
# GRÁFICAS DEL ENTRENAMIENTO SAC
# ==========================================================

MOSTRAR_GRAFICAS_ENTRENAMIENTO_SAC = True

GUARDAR_GRAFICAS_ENTRENAMIENTO_SAC = True

DIRECTORIO_GRAFICAS_ENTRENAMIENTO_SAC = (
    "resultados_sac/graficas_entrenamiento"
)

DPI_GRAFICAS_ENTRENAMIENTO_SAC = 300

# ==========================================================
# GRÁFICAS TEMPORALES DE EVALUACIÓN SAC
# ==========================================================

MOSTRAR_GRAFICAS_TEMPORALES_SAC = True

GUARDAR_GRAFICAS_TEMPORALES_SAC = False

DIRECTORIO_GRAFICAS_TEMPORALES_SAC = (
    "resultados_sac/evaluacion_temporal"
)

DPI_GRAFICAS_TEMPORALES_SAC = 300

# ==========================================================
# GRÁFICA ESPACIAL DE EVALUACIÓN SAC
# ==========================================================

MOSTRAR_GRAFICA_ESPACIAL_SAC = True

GUARDAR_GRAFICA_ESPACIAL_SAC = False

RUTA_GRAFICA_ESPACIAL_SAC = (
    "resultados_sac/evaluacion/"
    "trayectoria_espacial_sac.png"
)

DPI_GRAFICA_ESPACIAL_SAC = 300

ANOTAR_SUBMETAS_ESPACIALES_SAC = True

# ==========================================================
# ENTRENAMIENTO REAL SAC
# ==========================================================

NUMERO_EPISODIOS_ENTRENAMIENTO_SAC = 500

TRANSICIONES_ALEATORIAS_INICIALES_SAC = 5000

ACTUALIZACIONES_POR_PASO_SAC = 1

VENTANA_PROMEDIO_MOVIL_SAC = 25

FRECUENCIA_IMPRESION_ENTRENAMIENTO_SAC = 10


# ==========================================================
# VALIDACIÓN DEL ACTOR SAC ACTUAL
# ==========================================================

NUMERO_SEMILLAS_VALIDACION_ACTOR_ACTUAL_SAC = 50

SEMILLA_BASE_VALIDACION_ACTOR_ACTUAL_SAC = 10000

FRECUENCIA_IMPRESION_VALIDACION_SAC = 5

RUTA_CHECKPOINT_ACTOR_ACTUAL_SAC = Path(
    "resultados_sac/entrenamiento_real/"
    "checkpoint_final_sac.pt"
)

DIRECTORIO_VALIDACION_ACTOR_ACTUAL_SAC = Path(
    "resultados_sac/validacion_actor_actual"
)

RUTA_CSV_VALIDACION_ACTOR_ACTUAL_SAC = (
    DIRECTORIO_VALIDACION_ACTOR_ACTUAL_SAC
    / "resultados_por_semilla.csv"
)

RUTA_RESUMEN_VALIDACION_ACTOR_ACTUAL_SAC = (
    DIRECTORIO_VALIDACION_ACTOR_ACTUAL_SAC
    / "resumen_validacion.txt"
)

RUTA_GRAFICA_VALIDACION_ACTOR_ACTUAL_SAC = (
    DIRECTORIO_VALIDACION_ACTOR_ACTUAL_SAC
    / "distribucion_resultados.png"
)

MOSTRAR_GRAFICA_VALIDACION_ACTOR_ACTUAL_SAC = True


# ==========================================================
# SEGUNDA CORRIDA SAC: AJUSTE DE SEGURIDAD DINÁMICA
# ==========================================================

NUMERO_EPISODIOS_SEGUNDA_CORRIDA_SAC = 400

SEMILLA_BASE_SEGUNDA_CORRIDA_SAC = 30000

TRANSICIONES_ALEATORIAS_SEGUNDA_CORRIDA_SAC = 5000

ACTUALIZACIONES_POR_PASO_SEGUNDA_CORRIDA_SAC = 1

EPISODIOS_ENTRE_VALIDACIONES_SEGUNDA_CORRIDA_SAC = 20

NUMERO_SEMILLAS_VALIDACION_SEGUNDA_CORRIDA_SAC = 20

SEMILLA_BASE_VALIDACION_SEGUNDA_CORRIDA_SAC = 10000

VENTANA_PROMEDIO_SEGUNDA_CORRIDA_SAC = 25


# ----------------------------------------------------------
# TASAS DE APRENDIZAJE
# ----------------------------------------------------------
#
# El actor comienza con conocimientos previos, por eso
# utilizamos una tasa menor.
#
# Los críticos comienzan desde cero y conservan 3e-4.

TASA_APRENDIZAJE_ACTOR_SEGUNDA_CORRIDA_SAC = 1e-4

TASA_APRENDIZAJE_CRITICOS_SEGUNDA_CORRIDA_SAC = 3e-4

TASA_APRENDIZAJE_ALPHA_SEGUNDA_CORRIDA_SAC = 3e-4


# ----------------------------------------------------------
# CHECKPOINT DE ORIGEN
# ----------------------------------------------------------

RUTA_ACTOR_INICIAL_SEGUNDA_CORRIDA_SAC = Path(
    "resultados_sac/entrenamiento_real/"
    "checkpoint_final_sac.pt"
)


# ----------------------------------------------------------
# DIRECTORIO DE SALIDA
# ----------------------------------------------------------

DIRECTORIO_SEGUNDA_CORRIDA_SAC = Path(
    "resultados_sac/segunda_corrida_seguridad_dinamica"
)

RUTA_MEJOR_ACTOR_SEGUNDA_CORRIDA_SAC = (
    DIRECTORIO_SEGUNDA_CORRIDA_SAC
    / "checkpoint_mejor_actor_sac.pt"
)

RUTA_CHECKPOINT_FINAL_SEGUNDA_CORRIDA_SAC = (
    DIRECTORIO_SEGUNDA_CORRIDA_SAC
    / "checkpoint_final_completo_sac.pt"
)

DIRECTORIO_GRAFICAS_SEGUNDA_CORRIDA_SAC = (
    DIRECTORIO_SEGUNDA_CORRIDA_SAC
    / "graficas_entrenamiento"
)

RUTA_CSV_VALIDACIONES_SEGUNDA_CORRIDA_SAC = (
    DIRECTORIO_SEGUNDA_CORRIDA_SAC
    / "validaciones_periodicas.csv"
)

RUTA_GRAFICA_VALIDACIONES_SEGUNDA_CORRIDA_SAC = (
    DIRECTORIO_SEGUNDA_CORRIDA_SAC
    / "validaciones_periodicas.png"
)

SEMILLA_DIAGNOSTICA_MEJOR_ACTOR_SEGUNDA_CORRIDA_SAC = (
    11000
)

RUTA_GRAFICA_DIAGNOSTICA_MEJOR_ACTOR_SAC = (
    DIRECTORIO_SEGUNDA_CORRIDA_SAC
    / "trayectoria_diagnostica_mejor_actor.png"
)

MOSTRAR_RESULTADOS_SEGUNDA_CORRIDA_SAC = True

# ==========================================================
# SELECCIÓN PERIÓDICA DEL MEJOR ACTOR SAC
# ==========================================================

EPISODIOS_ENTRE_VALIDACIONES_SAC = 20

NUMERO_SEMILLAS_VALIDACION_PERIODICA_SAC = 20

SEMILLA_BASE_VALIDACION_PERIODICA_SAC = 10000

DIRECTORIO_SELECCION_MEJOR_ACTOR_SAC = Path(
    "resultados_sac/seleccion_mejor_actor"
)

RUTA_MEJOR_ACTOR_SAC = (
    DIRECTORIO_SELECCION_MEJOR_ACTOR_SAC
    / "checkpoint_mejor_actor_sac.pt"
)
# ==========================================================
# ARCHIVOS DEL ENTRENAMIENTO REAL
# ==========================================================

DIRECTORIO_ENTRENAMIENTO_REAL_SAC = Path(
    "resultados_sac/entrenamiento_real"
)

RUTA_CHECKPOINT_FINAL_SAC = (
    DIRECTORIO_ENTRENAMIENTO_REAL_SAC
    / "checkpoint_final_sac.pt"
)

DIRECTORIO_GRAFICAS_REALES_SAC = (
    DIRECTORIO_ENTRENAMIENTO_REAL_SAC
    / "graficas_aprendizaje"
)

RUTA_EVALUACION_DIAGNOSTICA_SAC = (
    DIRECTORIO_ENTRENAMIENTO_REAL_SAC
    / "evaluacion_diagnostica_sac.png"
)

SEMILLA_EVALUACION_DIAGNOSTICA_SAC = (
    SEMILLA + 100000
)

MOSTRAR_RESULTADOS_ENTRENAMIENTO_REAL_SAC = True

def configurar_pytorch_sac(
    semilla,
):

    semilla = int(
        semilla
    )

    # Semilla del generador aleatorio normal de Python.
    random.seed(
        semilla
    )

    # Semilla de NumPy.
    np.random.seed(
        semilla
    )

    # Semilla de PyTorch para CPU.
    torch.manual_seed(
        semilla
    )

    # Comprobar si PyTorch puede utilizar CUDA.
    gpu_disponible = torch.cuda.is_available()

    if gpu_disponible:

        # Semilla para la GPU principal.
        torch.cuda.manual_seed(
            semilla
        )

        # Semilla para todas las GPU disponibles.
        torch.cuda.manual_seed_all(
            semilla
        )

        dispositivo = torch.device(
    "cuda:0"
)

        nombre_dispositivo = (
            torch.cuda.get_device_name(
                0
            )
        )

    else:

        dispositivo = torch.device(
            "cpu"
        )

        nombre_dispositivo = "CPU"

    informacion = {
        "semilla": semilla,

        "gpu_disponible": (
            gpu_disponible
        ),

        "dispositivo": dispositivo,

        "nombre_dispositivo": (
            nombre_dispositivo
        ),

        "numero_gpus": (
            torch.cuda.device_count()
            if gpu_disponible
            else 0
        ),
    }

    return informacion

INFORMACION_PYTORCH_SAC = configurar_pytorch_sac(
    semilla=SEMILLA
)

DISPOSITIVO_SAC = INFORMACION_PYTORCH_SAC[
    "dispositivo"
]
print(
    "\nDispositivo seleccionado para SAC:",
    DISPOSITIVO_SAC,
)
print(
    "GPU:",
    INFORMACION_PYTORCH_SAC[
        "nombre_dispositivo"
    ],
)

# ==========================================================
# PRUEBA INDEPENDIENTE DEL MEJOR ACTOR SAC
# ==========================================================

NUMERO_SEMILLAS_PRUEBA_INDEPENDIENTE_SAC = 100

SEMILLA_BASE_PRUEBA_INDEPENDIENTE_SAC = 50000

FRECUENCIA_IMPRESION_PRUEBA_INDEPENDIENTE_SAC = 10


# ----------------------------------------------------------
# CHECKPOINT DEL MEJOR ACTOR
# ----------------------------------------------------------

RUTA_MEJOR_ACTOR_PRUEBA_INDEPENDIENTE_SAC = Path(
    "resultados_sac/"
    "segunda_corrida_seguridad_dinamica/"
    "checkpoint_mejor_actor_sac.pt"
)


# ----------------------------------------------------------
# DIRECTORIO DE RESULTADOS
# ----------------------------------------------------------

DIRECTORIO_PRUEBA_INDEPENDIENTE_SAC = Path(
    "resultados_sac/"
    "prueba_independiente_mejor_actor"
)

RUTA_CSV_PRUEBA_INDEPENDIENTE_SAC = (
    DIRECTORIO_PRUEBA_INDEPENDIENTE_SAC
    / "resultados_100_semillas.csv"
)

RUTA_CSV_FALLOS_PRUEBA_INDEPENDIENTE_SAC = (
    DIRECTORIO_PRUEBA_INDEPENDIENTE_SAC
    / "semillas_fallidas.csv"
)

RUTA_RESUMEN_PRUEBA_INDEPENDIENTE_SAC = (
    DIRECTORIO_PRUEBA_INDEPENDIENTE_SAC
    / "resumen_prueba_independiente.txt"
)

RUTA_GRAFICA_PRUEBA_INDEPENDIENTE_SAC = (
    DIRECTORIO_PRUEBA_INDEPENDIENTE_SAC
    / "distribucion_resultados_100_semillas.png"
)


# ----------------------------------------------------------
# CRITERIOS PREVIOS PARA CONSIDERAR SAC APTO
# ----------------------------------------------------------

TASA_EXITO_MINIMA_PRUEBA_SAC = 0.85

TASA_COLISION_DINAMICA_MAXIMA_PRUEBA_SAC = 0.10

TASA_COLISION_ESTATICA_MAXIMA_PRUEBA_SAC = 0.10

TASA_FUERA_MAPA_MAXIMA_PRUEBA_SAC = 0.05

TASA_TIMEOUT_MAXIMA_PRUEBA_SAC = 0.05


MOSTRAR_GRAFICA_PRUEBA_INDEPENDIENTE_SAC = True


# ==========================================================
# DIAGNÓSTICO DE SEMILLAS FALLIDAS SAC
# ==========================================================

RUTA_CHECKPOINT_DIAGNOSTICO_FALLOS_SAC = Path(
    "resultados_sac/"
    "segunda_corrida_seguridad_dinamica/"
    "checkpoint_mejor_actor_sac.pt"
)

RUTA_CSV_FALLOS_ORIGINAL_SAC = Path(
    "resultados_sac/"
    "prueba_independiente_mejor_actor/"
    "semillas_fallidas.csv"
)

DIRECTORIO_DIAGNOSTICO_FALLOS_SAC = Path(
    "resultados_sac/"
    "diagnostico_semillas_fallidas"
)

DIRECTORIO_TRAYECTORIAS_FALLOS_SAC = (
    DIRECTORIO_DIAGNOSTICO_FALLOS_SAC
    / "trayectorias"
)

DIRECTORIO_SERIES_FALLOS_SAC = (
    DIRECTORIO_DIAGNOSTICO_FALLOS_SAC
    / "series_temporales"
)

RUTA_CSV_DIAGNOSTICO_FALLOS_SAC = (
    DIRECTORIO_DIAGNOSTICO_FALLOS_SAC
    / "diagnostico_por_semilla.csv"
)

RUTA_CSV_RESUMEN_TIPOS_FALLO_SAC = (
    DIRECTORIO_DIAGNOSTICO_FALLOS_SAC
    / "resumen_por_tipo_fallo.csv"
)

RUTA_GRAFICA_RESUMEN_FALLOS_SAC = (
    DIRECTORIO_DIAGNOSTICO_FALLOS_SAC
    / "resumen_diagnostico_fallos.png"
)

MOSTRAR_RESUMEN_DIAGNOSTICO_FALLOS_SAC = True


# ==========================================================
# UMBRALES DEL DIAGNÓSTICO
# ==========================================================

UMBRAL_VELOCIDAD_BAJA_DIAGNOSTICO_SAC = 0.08

UMBRAL_OMEGA_OSCILACION_DIAGNOSTICO_SAC = 0.15

PASOS_MINIMOS_ESTANCAMIENTO_DIAGNOSTICO_SAC = 20

TTC_RIESGOSO_DIAGNOSTICO_SAC = 1.0


# Respaldo por si el CSV no se encuentra.

SEMILLAS_FALLIDAS_RESPALDO_SAC = [
    50002,
    50015,
    50017,
    50019,
    50026,
    50028,
    50032,
    50033,
    50036,
    50037,
    50039,
    50042,
    50051,
    50052,
    50053,
    50054,
    50074,
    50085,
    50091,
    50092,
    50093,
    50096,
]
# ==========================================================
# DINÁMICA FÍSICA DEL SAC REACTIVO
# ==========================================================

ACELERACION_LINEAL_MAXIMA_SAC_REACTIVO = (
    ACELERACION_LINEAL_MAXIMA_DWA
)

ACELERACION_ANGULAR_MAXIMA_SAC_REACTIVO = (
    ACELERACION_ANGULAR_MAXIMA_DWA
)


# ==========================================================
# APROXIMACIÓN TERMINAL DEL SAC REACTIVO
# ==========================================================

DISTANCIA_INICIO_FRENADO_META_SAC_REACTIVO = 1.75

VELOCIDAD_MINIMA_APROXIMACION_META_SAC_REACTIVO = 0.08


# ==========================================================
# CHECKPOINT DEL SAC REACTIVO ACTUAL
# ==========================================================

RUTA_ACTOR_BASE_MEJORA_REACTIVA_SAC = Path(
    "resultados_sac/"
    "segunda_corrida_seguridad_dinamica/"
    "checkpoint_mejor_actor_sac.pt"
)
# ==========================================================
# PROGRESO HACIA LA META GLOBAL
# ==========================================================

PESO_SAC_PROGRESO_META_GLOBAL = 1.00


# ==========================================================
# ESTANCAMIENTO PERSISTENTE DEL SAC REACTIVO
# ==========================================================

PROGRESO_MINIMO_META_POR_PASO_SAC_REACTIVO = 0.005

MEJORA_MINIMA_MEJOR_DISTANCIA_SAC_REACTIVO = 0.03

PASOS_INICIO_PENALIZACION_ESTANCAMIENTO_SAC_REACTIVO = 25

PASOS_PENALIZACION_MAXIMA_ESTANCAMIENTO_SAC_REACTIVO = 100

PENALIZACION_ESTANCAMIENTO_PERSISTENTE_MAXIMA_SAC = 2.00

RECOMPENSA_RECUPERACION_ESTANCAMIENTO_SAC = 0.75


# ==========================================================
# TERCERA CORRIDA: SAC REACTIVO MEJORADO
# ==========================================================

NUMERO_EPISODIOS_TERCERA_CORRIDA_SAC = 400

SEMILLA_BASE_TERCERA_CORRIDA_SAC = 60000

TRANSICIONES_ALEATORIAS_TERCERA_CORRIDA_SAC = 5000

ACTUALIZACIONES_POR_PASO_TERCERA_CORRIDA_SAC = 1

EPISODIOS_ENTRE_VALIDACIONES_TERCERA_CORRIDA_SAC = 20

NUMERO_SEMILLAS_VALIDACION_TERCERA_CORRIDA_SAC = 50

SEMILLA_BASE_VALIDACION_TERCERA_CORRIDA_SAC = 40000

VENTANA_PROMEDIO_TERCERA_CORRIDA_SAC = 25

TASA_APRENDIZAJE_ACTOR_TERCERA_CORRIDA_SAC = 1e-4

TASA_APRENDIZAJE_CRITICOS_TERCERA_CORRIDA_SAC = 3e-4

TASA_APRENDIZAJE_ALPHA_TERCERA_CORRIDA_SAC = 3e-4

RUTA_ACTOR_INICIAL_TERCERA_CORRIDA_SAC = Path(
    "resultados_sac/"
    "segunda_corrida_seguridad_dinamica/"
    "checkpoint_mejor_actor_sac.pt"
)

DIRECTORIO_TERCERA_CORRIDA_SAC = Path(
    "resultados_sac/"
    "tercera_corrida_reactivo_mejorado"
)

RUTA_MEJOR_ACTOR_TERCERA_CORRIDA_SAC = (
    DIRECTORIO_TERCERA_CORRIDA_SAC
    / "checkpoint_mejor_actor_reactivo_mejorado.pt"
)

RUTA_CHECKPOINT_FINAL_TERCERA_CORRIDA_SAC = (
    DIRECTORIO_TERCERA_CORRIDA_SAC
    / "checkpoint_final_completo_reactivo_mejorado.pt"
)

DIRECTORIO_GRAFICAS_TERCERA_CORRIDA_SAC = (
    DIRECTORIO_TERCERA_CORRIDA_SAC
    / "graficas_entrenamiento"
)

RUTA_CSV_VALIDACIONES_TERCERA_CORRIDA_SAC = (
    DIRECTORIO_TERCERA_CORRIDA_SAC
    / "validaciones_periodicas_50_semillas.csv"
)

RUTA_GRAFICA_VALIDACIONES_TERCERA_CORRIDA_SAC = (
    DIRECTORIO_TERCERA_CORRIDA_SAC
    / "validaciones_periodicas_50_semillas.png"
)

SEMILLA_DIAGNOSTICA_TERCERA_CORRIDA_SAC = 45000

RUTA_GRAFICA_DIAGNOSTICA_TERCERA_CORRIDA_SAC = (
    DIRECTORIO_TERCERA_CORRIDA_SAC
    / "trayectoria_diagnostica_mejor_actor.png"
)

MOSTRAR_RESULTADOS_TERCERA_CORRIDA_SAC = True

EJECUTAR_VERIFICACION_PREVIA_TERCERA_CORRIDA_SAC = True



def mundo_a_rejilla(x, y):

    columna = int(
        x / RESOLUCION_REJILLA
    )

    fila = int(
        y / RESOLUCION_REJILLA
    )

    columna = int(
        np.clip(
            columna,
            0,
            NUMERO_COLUMNAS - 1,
        )
    )

    fila = int(
        np.clip(
            fila,
            0,
            NUMERO_FILAS - 1,
        )
    )

    return fila, columna
def rejilla_a_mundo(fila, columna):

    x = (
        columna + 0.5
    ) * RESOLUCION_REJILLA

    y = (
        fila + 0.5
    ) * RESOLUCION_REJILLA

    return x, y
def crear_rejilla_vacia():

    rejilla = np.zeros(
        (
            NUMERO_FILAS,
            NUMERO_COLUMNAS,
        ),
        dtype=np.uint8,
    )

    return rejilla
def construir_rejilla_ocupacion(obstaculos):

    rejilla = crear_rejilla_vacia()

    for fila in range(NUMERO_FILAS):

        for columna in range(NUMERO_COLUMNAS):

            x, y = rejilla_a_mundo(
                fila,
                columna,
            )

            cerca_del_borde = not punto_dentro_del_mapa(
                x,
                y,
                margen=RADIO_INFLADO,
            )

            cerca_de_obstaculo = punto_en_zona_inflada(
                x,
                y,
                obstaculos,
            )

            if cerca_del_borde or cerca_de_obstaculo:

                rejilla[fila, columna] = 1

    return rejilla
def celda_esta_libre(
    punto,
    rejilla,
):

    fila, columna = mundo_a_rejilla(
        punto[0],
        punto[1],
    )

    valor = rejilla[
        fila,
        columna,
    ]

    esta_libre = valor == 0

    return esta_libre
def dibujar_rejilla_ocupacion(
    ax,
    rejilla,
):

    ax.imshow(
        rejilla,
        origin="lower",
        extent=[
            0,
            ANCHO_MAPA,
            0,
            ALTO_MAPA,
        ],
        cmap="Greys",
        alpha=0.35,
        interpolation="nearest",
    )
def nodo_dentro_rejilla(nodo):

    fila = nodo[0]
    columna = nodo[1]

    fila_valida = (
        fila >= 0
        and fila < NUMERO_FILAS
    )

    columna_valida = (
        columna >= 0
        and columna < NUMERO_COLUMNAS
    )

    return fila_valida and columna_valida
def nodo_es_libre(nodo, rejilla):

    if not nodo_dentro_rejilla(nodo):
        return False

    fila = nodo[0]
    columna = nodo[1]

    valor_celda = rejilla[
        fila,
        columna,
    ]

    return valor_celda == 0
def calcular_costo_movimiento(
    cambio_fila,
    cambio_columna,
):

    movimiento_diagonal = (
        cambio_fila != 0
        and cambio_columna != 0
    )

    if movimiento_diagonal:

        costo = (
            math.sqrt(2.0)
            * RESOLUCION_REJILLA
        )

    else:

        costo = RESOLUCION_REJILLA

    return costo
def diagonal_permitida(
    nodo_actual,
    cambio_fila,
    cambio_columna,
    rejilla,
):

    es_diagonal = (
        cambio_fila != 0
        and cambio_columna != 0
    )

    if not es_diagonal:
        return True

    fila_actual = nodo_actual[0]
    columna_actual = nodo_actual[1]

    nodo_lateral_1 = (
        fila_actual + cambio_fila,
        columna_actual,
    )

    nodo_lateral_2 = (
        fila_actual,
        columna_actual + cambio_columna,
    )

    lateral_1_libre = nodo_es_libre(
        nodo_lateral_1,
        rejilla,
    )

    lateral_2_libre = nodo_es_libre(
        nodo_lateral_2,
        rejilla,
    )

    return lateral_1_libre and lateral_2_libre
def obtener_vecinos(
    nodo_actual,
    rejilla,
):

    vecinos = []

    fila_actual = nodo_actual[0]
    columna_actual = nodo_actual[1]

    for movimiento in MOVIMIENTOS:

        cambio_fila = movimiento[0]
        cambio_columna = movimiento[1]

        fila_vecino = (
            fila_actual
            + cambio_fila
        )

        columna_vecino = (
            columna_actual
            + cambio_columna
        )

        nodo_vecino = (
            fila_vecino,
            columna_vecino,
        )

        if not nodo_es_libre(
            nodo_vecino,
            rejilla,
        ):
            continue

        diagonal_valida = diagonal_permitida(
            nodo_actual,
            cambio_fila,
            cambio_columna,
            rejilla,
        )

        if not diagonal_valida:
            continue

        costo = calcular_costo_movimiento(
            cambio_fila,
            cambio_columna,
        )

        vecinos.append(
            (
                nodo_vecino,
                costo,
            )
        )

    return vecinos
def calcular_heuristica(
    nodo_actual,
    nodo_meta,
):

    diferencia_filas = (
        nodo_meta[0]
        - nodo_actual[0]
    )

    diferencia_columnas = (
        nodo_meta[1]
        - nodo_actual[1]
    )

    distancia_celdas = math.hypot(
        diferencia_filas,
        diferencia_columnas,
    )

    distancia_metros = (
        distancia_celdas
        * RESOLUCION_REJILLA
    )

    return distancia_metros
def dibujar_vecinos(
    ax,
    vecinos,
):

    for vecino, costo in vecinos:

        fila = vecino[0]
        columna = vecino[1]

        x, y = rejilla_a_mundo(
            fila,
            columna,
        )

        ax.scatter(
            x,
            y,
            marker="s",
            s=35,
            color="cyan",
            edgecolor="black",
            zorder=6,
        )
def buscar_astar(
    rejilla,
    nodo_inicio,
    nodo_meta,
):

    # Verificar que inicio y meta sean transitables.
    if not nodo_es_libre(
        nodo_inicio,
        rejilla,
    ):
        print(
            "El nodo inicial está ocupado."
        )

        return False, {}, {}, []

    if not nodo_es_libre(
        nodo_meta,
        rejilla,
    ):
        print(
            "El nodo meta está ocupado."
        )

        return False, {}, {}, []

    # ------------------------------------------------------
    # Lista abierta
    # ------------------------------------------------------

    lista_abierta = []

    heuristica_inicio = calcular_heuristica(
        nodo_inicio,
        nodo_meta,
    )

    heapq.heappush(
        lista_abierta,
        (
            heuristica_inicio,
            0.0,
            nodo_inicio,
        )
    )

    # ------------------------------------------------------
    # Costos acumulados g(n)
    # ------------------------------------------------------

    costos_g = {
        nodo_inicio: 0.0
    }

    # ------------------------------------------------------
    # Registro de padres
    # ------------------------------------------------------

    padres = {}

    # ------------------------------------------------------
    # Nodos que ya fueron expandidos
    # ------------------------------------------------------

    nodos_cerrados = set()

    orden_expansion = []

    # ------------------------------------------------------
    # Ciclo principal de A*
    # ------------------------------------------------------

    while len(lista_abierta) > 0:

        elemento_actual = heapq.heappop(
            lista_abierta
        )

        costo_f_actual = elemento_actual[0]
        costo_g_guardado = elemento_actual[1]
        nodo_actual = elemento_actual[2]

        # El mismo nodo puede aparecer varias veces en la
        # lista abierta. Si ya fue procesado, se ignora.
        if nodo_actual in nodos_cerrados:
            continue

        # Ignorar registros antiguos con un costo peor.
        mejor_costo_conocido = costos_g.get(
            nodo_actual,
            float("inf"),
        )

        if costo_g_guardado > mejor_costo_conocido:
            continue

        # Marcar el nodo como procesado.
        nodos_cerrados.add(
            nodo_actual
        )

        orden_expansion.append(
            nodo_actual
        )

        # --------------------------------------------------
        # Comprobar si se alcanzó la meta
        # --------------------------------------------------

        if nodo_actual == nodo_meta:

            return (
                True,
                padres,
                costos_g,
                orden_expansion,
            )

        # --------------------------------------------------
        # Explorar los vecinos
        # --------------------------------------------------

        vecinos = obtener_vecinos(
            nodo_actual,
            rejilla,
        )

        for nodo_vecino, costo_movimiento in vecinos:

            if nodo_vecino in nodos_cerrados:
                continue

            nuevo_costo_g = (
                costos_g[nodo_actual]
                + costo_movimiento
            )

            costo_anterior = costos_g.get(
                nodo_vecino,
                float("inf"),
            )

            # Solamente se actualiza cuando encontramos
            # una forma más económica de llegar al vecino.
            if nuevo_costo_g < costo_anterior:

                costos_g[nodo_vecino] = nuevo_costo_g

                padres[nodo_vecino] = nodo_actual

                heuristica = calcular_heuristica(
                    nodo_vecino,
                    nodo_meta,
                )

                nuevo_costo_f = (
                    nuevo_costo_g
                    + heuristica
                )

                heapq.heappush(
                    lista_abierta,
                    (
                        nuevo_costo_f,
                        nuevo_costo_g,
                        nodo_vecino,
                    )
                )

    # Si la lista abierta se vacía, no existe una ruta.
    return (
        False,
        padres,
        costos_g,
        orden_expansion,
    )
def dibujar_nodos_expandidos(
    ax,
    nodos_expandidos,
):

    coordenadas_x = []
    coordenadas_y = []

    for nodo in nodos_expandidos:

        fila = nodo[0]
        columna = nodo[1]

        x, y = rejilla_a_mundo(
            fila,
            columna,
        )

        coordenadas_x.append(
            x
        )

        coordenadas_y.append(
            y
        )

    if len(coordenadas_x) > 0:

        ax.scatter(
            coordenadas_x,
            coordenadas_y,
            marker="s",
            s=10,
            color="orange",
            alpha=0.35,
            label="Nodos expandidos por A*",
            zorder=2,
        )
def reconstruir_camino_nodos(
    padres,
    nodo_inicio,
    nodo_meta,
):

    camino_inverso = [
        nodo_meta
    ]

    nodo_actual = nodo_meta

    while nodo_actual != nodo_inicio:

        if nodo_actual not in padres:

            print(
                "No fue posible reconstruir el camino."
            )

            return None

        nodo_actual = padres[
            nodo_actual
        ]

        camino_inverso.append(
            nodo_actual
        )

    camino_nodos = list(
        reversed(
            camino_inverso
        )
    )

    return camino_nodos
def convertir_camino_a_mundo(
    camino_nodos,
    inicio,
    meta,
):

    camino_mundo = []

    for nodo in camino_nodos:

        fila = nodo[0]
        columna = nodo[1]

        x, y = rejilla_a_mundo(
            fila,
            columna,
        )

        camino_mundo.append(
            (x, y)
        )

    # Sustituir el centro de la primera celda por la
    # posición continua exacta de inicio.
    camino_mundo[0] = inicio

    # Sustituir el centro de la última celda por la
    # posición continua exacta de la meta.
    camino_mundo[-1] = meta

    return camino_mundo
def calcular_longitud_camino(
    camino_mundo,
):

    longitud_total = 0.0

    for indice in range(
        len(camino_mundo) - 1
    ):

        punto_actual = camino_mundo[
            indice
        ]

        punto_siguiente = camino_mundo[
            indice + 1
        ]

        longitud_segmento = distancia_entre_puntos(
            punto_actual,
            punto_siguiente,
        )

        longitud_total = (
            longitud_total
            + longitud_segmento
        )

    return longitud_total
def dibujar_camino_astar(
    ax,
    camino_mundo,
):

    coordenadas_x = []
    coordenadas_y = []

    for punto in camino_mundo:

        coordenadas_x.append(
            punto[0]
        )

        coordenadas_y.append(
            punto[1]
        )

    ax.plot(
        coordenadas_x,
        coordenadas_y,
        color="blue",
        linewidth=2.5,
        label="Ruta A*",
        zorder=6,
    )
def dibujar_nodos_camino(
    ax,
    camino_mundo,
):

    coordenadas_x = []
    coordenadas_y = []

    for punto in camino_mundo:

        coordenadas_x.append(
            punto[0]
        )

        coordenadas_y.append(
            punto[1]
        )

    ax.scatter(
        coordenadas_x,
        coordenadas_y,
        marker="o",
        s=12,
        color="blue",
        zorder=7,
    )
def generar_escenario_valido(
    semilla,
    intentos_maximos=INTENTOS_MAXIMOS_ESCENARIO,
):

    # El generador se crea una sola vez para este escenario.
    generador = np.random.default_rng(
        semilla
    )

    for intento in range(
        intentos_maximos
    ):

        # --------------------------------------------------
        # 1. Generar obstáculos
        # --------------------------------------------------

        obstaculos = generar_obstaculos_estaticos(
            NUMERO_OBSTACULOS,
            generador,
        )

        if (
            len(obstaculos)
            != NUMERO_OBSTACULOS
        ):
            continue

        mapa_valido = verificar_superposiciones(
            obstaculos
        )

        if not mapa_valido:
            continue

        # --------------------------------------------------
        # 2. Generar inicio y meta
        # --------------------------------------------------

        inicio, meta = generar_inicio_meta(
            obstaculos,
            generador,
        )

        if inicio is None or meta is None:
            continue

        # --------------------------------------------------
        # 3. Construir rejilla
        # --------------------------------------------------

        rejilla = construir_rejilla_ocupacion(
            obstaculos
        )

        # --------------------------------------------------
        # 4. Convertir a nodos
        # --------------------------------------------------

        nodo_inicio = mundo_a_rejilla(
            inicio[0],
            inicio[1],
        )

        nodo_meta = mundo_a_rejilla(
            meta[0],
            meta[1],
        )

        # --------------------------------------------------
        # 5. Verificar nodos
        # --------------------------------------------------

        if not nodo_es_libre(
            nodo_inicio,
            rejilla,
        ):
            continue

        if not nodo_es_libre(
            nodo_meta,
            rejilla,
        ):
            continue

        # --------------------------------------------------
        # 6. Ejecutar A*
        # --------------------------------------------------

        (
            camino_encontrado,
            padres,
            costos_g,
            nodos_expandidos,
        ) = buscar_astar(
            rejilla,
            nodo_inicio,
            nodo_meta,
        )

        if not camino_encontrado:
            continue

        # --------------------------------------------------
        # 7. Reconstruir camino
        # --------------------------------------------------

        camino_nodos = reconstruir_camino_nodos(
            padres,
            nodo_inicio,
            nodo_meta,
        )

        if camino_nodos is None:
            continue

        camino_mundo = convertir_camino_a_mundo(
            camino_nodos,
            inicio,
            meta,
        )

        # --------------------------------------------------
        # 8. Calcular métricas
        # --------------------------------------------------

        distancia_directa = distancia_entre_puntos(
            inicio,
            meta,
        )

        longitud_camino = calcular_longitud_camino(
            camino_mundo
        )

        costo_astar = costos_g[
            nodo_meta
        ]

        eficiencia_geometrica = (
            distancia_directa
            / longitud_camino
        )

        # --------------------------------------------------
        # 9. Guardar escenario
        # --------------------------------------------------

        escenario = {
            "semilla": semilla,
            "obstaculos": obstaculos,
            "inicio": inicio,
            "meta": meta,
            "rejilla": rejilla,
            "nodo_inicio": nodo_inicio,
            "nodo_meta": nodo_meta,
            "padres": padres,
            "costos_g": costos_g,
            "nodos_expandidos": nodos_expandidos,
            "camino_nodos": camino_nodos,
            "camino_mundo": camino_mundo,
            "distancia_directa": distancia_directa,
            "longitud_camino": longitud_camino,
            "costo_astar": costo_astar,
            "eficiencia_geometrica": eficiencia_geometrica,
            "numero_intento": intento + 1,
        }

        return escenario

    return None
def escenarios_son_iguales(
    escenario_1,
    escenario_2,
):

    if (
        escenario_1 is None
        or escenario_2 is None
    ):
        return False

    mismo_inicio = (
        escenario_1["inicio"]
        == escenario_2["inicio"]
    )

    misma_meta = (
        escenario_1["meta"]
        == escenario_2["meta"]
    )

    mismos_obstaculos = (
        escenario_1["obstaculos"]
        == escenario_2["obstaculos"]
    )

    mismo_camino = (
        escenario_1["camino_nodos"]
        == escenario_2["camino_nodos"]
    )

    son_iguales = (
        mismo_inicio
        and misma_meta
        and mismos_obstaculos
        and mismo_camino
    )

    return son_iguales
def normalizar_angulo(angulo):
    angulo_normalizado = (angulo + math.pi) % (2.0 * math.pi) - math.pi
    return angulo_normalizado
def punto_en_obstaculo(x, y, obstaculo, margen=0.0):

    dentro_horizontalmente = (
        x >= obstaculo["x_min"] - margen
        and x <= obstaculo["x_max"] + margen
    )

    dentro_verticalmente = (
        y >= obstaculo["y_min"] - margen
        and y <= obstaculo["y_max"] + margen
    )

    esta_dentro = dentro_horizontalmente and dentro_verticalmente

    return esta_dentro
def distancia_punto_obstaculo(x, y, obstaculo):
    x_cercano = np.clip(
        x,
        obstaculo["x_min"],
        obstaculo["x_max"],
    )
    y_cercano = np.clip(
        y,
        obstaculo["y_min"],
        obstaculo["y_max"],
    )
    distancia_x = x - x_cercano
    distancia_y = y - y_cercano
    distancia = math.hypot(distancia_x, distancia_y)
    return distancia
def robot_colisiona_obstaculo(x_robot, y_robot, obstaculo):

    distancia = distancia_punto_obstaculo(
        x_robot,
        y_robot,
        obstaculo,
    )

    hay_colision = distancia < RADIO_ROBOT

    return hay_colision
def robot_colisiona_obstaculos(x_robot, y_robot, obstaculos):

    for obstaculo in obstaculos:

        colision = robot_colisiona_obstaculo(
            x_robot,
            y_robot,
            obstaculo,
        )

        if colision:
            return True

    return False
def robot_fuera_del_mapa(x_robot, y_robot):

    limite_izquierdo = x_robot < RADIO_ROBOT
    limite_derecho = x_robot > ANCHO_MAPA - RADIO_ROBOT

    limite_inferior = y_robot < RADIO_ROBOT
    limite_superior = y_robot > ALTO_MAPA - RADIO_ROBOT

    fuera = (
        limite_izquierdo
        or limite_derecho
        or limite_inferior
        or limite_superior
    )

    return fuera
def inflar_obstaculo(obstaculo, margen):

    obstaculo_inflado = {
        "x_min": obstaculo["x_min"] - margen,
        "x_max": obstaculo["x_max"] + margen,
        "y_min": obstaculo["y_min"] - margen,
        "y_max": obstaculo["y_max"] + margen,
    }

    return obstaculo_inflado
def dibujar_obstaculo(ax, obstaculo):

    ancho = obstaculo["x_max"] - obstaculo["x_min"]
    alto = obstaculo["y_max"] - obstaculo["y_min"]

    rectangulo = Rectangle(
        (obstaculo["x_min"], obstaculo["y_min"]),
        ancho,
        alto,
        facecolor="gray",
        edgecolor="black",
        alpha=0.8,
    )

    ax.add_patch(rectangulo)
def dibujar_obstaculo_inflado(ax, obstaculo):

    obstaculo_inflado = inflar_obstaculo(
        obstaculo,
        RADIO_INFLADO,
    )

    ancho = (
        obstaculo_inflado["x_max"]
        - obstaculo_inflado["x_min"]
    )

    alto = (
        obstaculo_inflado["y_max"]
        - obstaculo_inflado["y_min"]
    )

    rectangulo_inflado = Rectangle(
        (
            obstaculo_inflado["x_min"],
            obstaculo_inflado["y_min"],
        ),
        ancho,
        alto,
        fill=False,
        edgecolor="red",
        linestyle="--",
        linewidth=1.5,
    )

    ax.add_patch(rectangulo_inflado)
def dibujar_obstaculo_completo(ax, obstaculo):

    dibujar_obstaculo(
        ax,
        obstaculo,
    )

    dibujar_obstaculo_inflado(
        ax,
        obstaculo,
    )
def dibujar_obstaculos(ax, obstaculos):

    for indice, obstaculo in enumerate(obstaculos):

        dibujar_obstaculo_completo(
            ax,
            obstaculo,
        )

        x_centro = (
            obstaculo["x_min"]
            + obstaculo["x_max"]
        ) / 2.0

        y_centro = (
            obstaculo["y_min"]
            + obstaculo["y_max"]
        ) / 2.0

        ax.text(
            x_centro,
            y_centro,
            str(indice + 1),
            horizontalalignment="center",
            verticalalignment="center",
        )
def configurar_mapa(ax):

    ax.set_xlim(
        0,
        ANCHO_MAPA,
    )

    ax.set_ylim(
        0,
        ALTO_MAPA,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("Posición x [m]")
    ax.set_ylabel("Posición y [m]")

    ax.set_title(
        "Mapa del entorno con obstáculos inflados"
    )

    ax.grid(
        True,
        alpha=0.3,
    )
def dibujar_robot(ax, x_robot, y_robot):

    circulo_robot = Circle(
        (x_robot, y_robot),
        radius=RADIO_ROBOT,
        facecolor="blue",
        edgecolor="black",
        alpha=0.7,
    )

    ax.add_patch(circulo_robot)
def punto_en_zona_inflada(x, y, obstaculos):

    for obstaculo in obstaculos:

        distancia = distancia_punto_obstaculo(
            x,
            y,
            obstaculo,
        )

        if distancia < RADIO_INFLADO:
            return True

    return False
def generar_obstaculo_candidato(
    generador,
):

    ancho = generador.uniform(
        ANCHO_MIN_OBSTACULO,
        ANCHO_MAX_OBSTACULO,
    )

    alto = generador.uniform(
        ALTO_MIN_OBSTACULO,
        ALTO_MAX_OBSTACULO,
    )

    x_min = generador.uniform(
        MARGEN_BORDE_OBSTACULOS,
        ANCHO_MAPA
        - ancho
        - MARGEN_BORDE_OBSTACULOS,
    )

    y_min = generador.uniform(
        MARGEN_BORDE_OBSTACULOS,
        ALTO_MAPA
        - alto
        - MARGEN_BORDE_OBSTACULOS,
    )

    obstaculo = {
        "x_min": x_min,
        "x_max": x_min + ancho,
        "y_min": y_min,
        "y_max": y_min + alto,
    }

    return obstaculo
def obstaculos_se_superponen(
    obstaculo_1,
    obstaculo_2,
    separacion=0.0,
):

    superposicion_horizontal = (
        obstaculo_1["x_min"]
        < obstaculo_2["x_max"] + separacion
        and
        obstaculo_1["x_max"]
        > obstaculo_2["x_min"] - separacion
    )

    superposicion_vertical = (
        obstaculo_1["y_min"]
        < obstaculo_2["y_max"] + separacion
        and
        obstaculo_1["y_max"]
        > obstaculo_2["y_min"] - separacion
    )

    se_superponen = (
        superposicion_horizontal
        and superposicion_vertical
    )

    return se_superponen
def candidato_es_valido(
    candidato,
    obstaculos_existentes,
):

    for obstaculo in obstaculos_existentes:

        se_superpone = obstaculos_se_superponen(
            candidato,
            obstaculo,
            separacion=SEPARACION_OBSTACULOS,
        )

        if se_superpone:
            return False

    return True
def generar_obstaculos_estaticos(
    numero_obstaculos,
    generador,
    intentos_maximos=500,
):

    obstaculos_generados = []

    intentos_realizados = 0

    while (
        len(obstaculos_generados)
        < numero_obstaculos
        and intentos_realizados
        < intentos_maximos
    ):

        candidato = generar_obstaculo_candidato(
            generador
        )

        valido = candidato_es_valido(
            candidato,
            obstaculos_generados,
        )

        if valido:

            obstaculos_generados.append(
                candidato
            )

        intentos_realizados = (
            intentos_realizados + 1
        )

    if (
        len(obstaculos_generados)
        < numero_obstaculos
    ):

        print(
            "Advertencia: solamente se generaron",
            len(obstaculos_generados),
            "de",
            numero_obstaculos,
            "obstáculos.",
        )

    return obstaculos_generados
def verificar_superposiciones(obstaculos):

    for i in range(len(obstaculos)):

        for j in range(i + 1, len(obstaculos)):

            se_superponen = obstaculos_se_superponen(
                obstaculos[i],
                obstaculos[j],
                separacion=SEPARACION_OBSTACULOS,
            )

            if se_superponen:

                print(
                    "Problema entre obstáculos",
                    i + 1,
                    "y",
                    j + 1,
                )

                return False

    return True
def punto_dentro_del_mapa(x, y, margen=0.0):

    dentro_horizontalmente = (
        x >= margen
        and x <= ANCHO_MAPA - margen
    )

    dentro_verticalmente = (
        y >= margen
        and y <= ALTO_MAPA - margen
    )

    esta_dentro = (
        dentro_horizontalmente
        and dentro_verticalmente
    )

    return esta_dentro
def punto_es_libre(x, y, obstaculos):

    dentro_mapa = punto_dentro_del_mapa(
        x,
        y,
        margen=MARGEN_BORDE_PUNTOS,
    )

    if not dentro_mapa:
        return False

    dentro_obstaculo_inflado = punto_en_zona_inflada(
        x,
        y,
        obstaculos,
    )

    if dentro_obstaculo_inflado:
        return False

    return True
def generar_punto_candidato(
    generador,
):

    x = generador.uniform(
        MARGEN_BORDE_PUNTOS,
        ANCHO_MAPA
        - MARGEN_BORDE_PUNTOS,
    )

    y = generador.uniform(
        MARGEN_BORDE_PUNTOS,
        ALTO_MAPA
        - MARGEN_BORDE_PUNTOS,
    )

    punto = (
        x,
        y,
    )

    return punto

def generar_punto_libre(
    obstaculos,
    generador,
    intentos_maximos=INTENTOS_MAXIMOS_PUNTO,
):

    for intento in range(
        intentos_maximos
    ):

        candidato = generar_punto_candidato(
            generador
        )

        x = candidato[0]
        y = candidato[1]

        libre = punto_es_libre(
            x,
            y,
            obstaculos,
        )

        if libre:

            return candidato

    return None
def distancia_entre_puntos(punto_1, punto_2):

    diferencia_x = punto_2[0] - punto_1[0]
    diferencia_y = punto_2[1] - punto_1[1]

    distancia = math.hypot(
        diferencia_x,
        diferencia_y,
    )

    return distancia
def generar_inicio_meta(
    obstaculos,
    generador,
    distancia_minima=DISTANCIA_MINIMA_INICIO_META,
    intentos_maximos=INTENTOS_MAXIMOS_INICIO_META,
):

    for intento in range(
        intentos_maximos
    ):

        inicio = generar_punto_libre(
            obstaculos,
            generador,
        )

        meta = generar_punto_libre(
            obstaculos,
            generador,
        )

        if inicio is None or meta is None:
            continue

        distancia = distancia_entre_puntos(
            inicio,
            meta,
        )

        if distancia >= distancia_minima:

            return inicio, meta

    return None, None
def dibujar_inicio(ax, inicio):

    ax.scatter(
        inicio[0],
        inicio[1],
        marker="o",
        s=80,
        color="green",
        edgecolor="black",
        label="Inicio",
        zorder=5,
    )
def dibujar_meta(ax, meta):

    ax.scatter(
        meta[0],
        meta[1],
        marker="*",
        s=180,
        color="red",
        edgecolor="black",
        label="Meta",
        zorder=5,
    )
def verificar_inicio_meta(
    inicio,
    meta,
    obstaculos,
):

    inicio_libre = punto_es_libre(
        inicio[0],
        inicio[1],
        obstaculos,
    )

    meta_libre = punto_es_libre(
        meta[0],
        meta[1],
        obstaculos,
    )

    distancia = distancia_entre_puntos(
        inicio,
        meta,
    )

    separacion_valida = (
        distancia
        >= DISTANCIA_MINIMA_INICIO_META
    )

    escenario_valido = (
        inicio_libre
        and meta_libre
        and separacion_valida
    )

    return escenario_valido
def limitar_control(
    velocidad_lineal,
    velocidad_angular,
):

    velocidad_lineal_limitada = np.clip(
        velocidad_lineal,
        -VELOCIDAD_MAXIMA,
        VELOCIDAD_MAXIMA,
    )

    velocidad_angular_limitada = np.clip(
        velocidad_angular,
        -VELOCIDAD_ANGULAR_MAXIMA,
        VELOCIDAD_ANGULAR_MAXIMA,
    )

    return (
        float(velocidad_lineal_limitada),
        float(velocidad_angular_limitada),
    )
def convertir_accion_sac_a_control(
    accion,
):

    accion = np.asarray(
        accion,
        dtype=float,
    ).reshape(-1)

    # SAC debe producir exactamente dos valores:
    # acción lineal y acción angular.
    if accion.size != DIMENSION_ACCION_SAC:

        raise ValueError(
            "La acción SAC debe contener exactamente "
            "dos valores: [accion_v, accion_omega]."
        )

    # Evita aceptar NaN o infinito.
    if not np.all(
        np.isfinite(
            accion
        )
    ):

        raise ValueError(
            "La acción SAC contiene valores no finitos."
        )

    # Garantizar el intervalo [-1, 1].
    accion_limitada = np.clip(
        accion,
        -1.0,
        1.0,
    )

    accion_velocidad = accion_limitada[
        0
    ]

    accion_angular = accion_limitada[
        1
    ]

    # ------------------------------------------------------
    # Velocidad lineal
    # ------------------------------------------------------
    #
    # Convierte:
    #
    # -1 → 0 m/s
    #  0 → 0.5 m/s
    #  1 → 1.0 m/s
    #
    # Así SAC utiliza todo el intervalo de la salida tanh,
    # pero el robot solamente se desplaza hacia adelante.

    velocidad_lineal = (
        accion_velocidad
        + 1.0
    ) * 0.5 * VELOCIDAD_MAXIMA

    # ------------------------------------------------------
    # Velocidad angular
    # ------------------------------------------------------
    #
    # Convierte:
    #
    # -1 → -1.2 rad/s
    #  0 →  0.0 rad/s
    #  1 →  1.2 rad/s

    velocidad_angular = (
        accion_angular
        * VELOCIDAD_ANGULAR_MAXIMA
    )

    return (
        float(
            velocidad_lineal
        ),
        float(
            velocidad_angular
        ),
    )
# ==========================================================
# LÍMITE DE VELOCIDAD CERCA DE LA META
# ==========================================================

def calcular_limite_velocidad_meta_sac_reactivo(
    distancia_meta,
):

    distancia_meta = float(
        distancia_meta
    )

    if not np.isfinite(
        distancia_meta
    ):

        raise ValueError(
            "La distancia a la meta no es finita."
        )

    if distancia_meta < 0.0:

        raise ValueError(
            "La distancia a la meta no puede ser negativa."
        )

    distancia_inicio_frenado = float(
        DISTANCIA_INICIO_FRENADO_META_SAC_REACTIVO
    )

    velocidad_minima = float(
        VELOCIDAD_MINIMA_APROXIMACION_META_SAC_REACTIVO
    )

    if (
        distancia_inicio_frenado
        <= DISTANCIA_META
    ):

        raise ValueError(
            "La distancia de inicio de frenado debe ser "
            "mayor que la tolerancia de llegada."
        )

    if (
        velocidad_minima < 0.0
        or velocidad_minima > VELOCIDAD_MAXIMA
    ):

        raise ValueError(
            "La velocidad mínima de aproximación no es "
            "válida."
        )

    # ------------------------------------------------------
    # 1. Robot dentro de la meta
    # ------------------------------------------------------

    if distancia_meta <= DISTANCIA_META:

        return {
            "limite_velocidad_meta": 0.0,

            "factor_frenado_meta": 0.0,

            "frenado_terminal_activo": True,

            "robot_dentro_meta": True,
        }

    # ------------------------------------------------------
    # 2. Robot lejos de la zona terminal
    # ------------------------------------------------------

    if (
        distancia_meta
        >= distancia_inicio_frenado
    ):

        return {
            "limite_velocidad_meta": float(
                VELOCIDAD_MAXIMA
            ),

            "factor_frenado_meta": 1.0,

            "frenado_terminal_activo": False,

            "robot_dentro_meta": False,
        }

    # ------------------------------------------------------
    # 3. Interpolación dentro de la zona de frenado
    # ------------------------------------------------------
    #
    # En DISTANCIA_META:
    #
    # límite ≈ velocidad mínima.
    #
    # En DISTANCIA_INICIO_FRENADO:
    #
    # límite = velocidad máxima.

    intervalo_frenado = (
        distancia_inicio_frenado
        - DISTANCIA_META
    )

    distancia_util = (
        distancia_meta
        - DISTANCIA_META
    )

    factor_frenado = float(
        np.clip(
            distancia_util
            / max(
                intervalo_frenado,
                1e-8,
            ),

            0.0,
            1.0,
        )
    )

    limite_velocidad = (
        velocidad_minima
        + (
            VELOCIDAD_MAXIMA
            - velocidad_minima
        )
        * factor_frenado
    )

    limite_velocidad = float(
        np.clip(
            limite_velocidad,
            velocidad_minima,
            VELOCIDAD_MAXIMA,
        )
    )

    return {
        "limite_velocidad_meta": (
            limite_velocidad
        ),

        "factor_frenado_meta": (
            factor_frenado
        ),

        "frenado_terminal_activo": True,

        "robot_dentro_meta": False,
    }

def convertir_control_a_accion_sac_prueba(
    velocidad_lineal,
    velocidad_angular,
):

    velocidad_lineal = float(
        np.clip(
            velocidad_lineal,
            0.0,
            VELOCIDAD_MAXIMA,
        )
    )

    velocidad_angular = float(
        np.clip(
            velocidad_angular,
            -VELOCIDAD_ANGULAR_MAXIMA,
            VELOCIDAD_ANGULAR_MAXIMA,
        )
    )

    accion_velocidad = (
        2.0
        * velocidad_lineal
        / max(
            VELOCIDAD_MAXIMA,
            1e-8,
        )
        - 1.0
    )

    accion_angular = (
        velocidad_angular
        / max(
            VELOCIDAD_ANGULAR_MAXIMA,
            1e-8,
        )
    )

    accion = np.array(
        [
            accion_velocidad,
            accion_angular,
        ],
        dtype=np.float32,
    )

    accion = np.clip(
        accion,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )

    return accion
def seleccionar_accion_politica_prueba_sac(
    entorno,
    observacion,
):

    # La observación todavía no se utiliza.
    # Se conserva como argumento para que esta función tenga
    # la misma interfaz que tendrá la red neuronal SAC.
    _ = observacion

    (
        velocidad_lineal,
        velocidad_angular,
        informacion_control,
    ) = controlador_submeta_local(
        estado_robot=entorno[
            "estado_robot"
        ],
        submeta=entorno[
            "submeta"
        ],
        meta=entorno[
            "meta"
        ],
    )

    accion = convertir_control_a_accion_sac_prueba(
        velocidad_lineal=velocidad_lineal,
        velocidad_angular=velocidad_angular,
    )

    informacion_politica = {
        "velocidad_lineal_deseada": (
            velocidad_lineal
        ),

        "velocidad_angular_deseada": (
            velocidad_angular
        ),

        "informacion_control": (
            informacion_control
        ),
    }

    return (
        accion,
        informacion_politica,
    )
def actualizar_estado_robot(
    estado,
    velocidad_lineal,
    velocidad_angular,
    dt=DT,
):

    x = estado[0]
    y = estado[1]
    theta = estado[2]

    velocidad_lineal, velocidad_angular = limitar_control(
        velocidad_lineal,
        velocidad_angular,
    )

    movimiento_recto = (
        abs(velocidad_angular) < 1e-8
    )

    if movimiento_recto:

        x_nuevo = (
            x
            + velocidad_lineal
            * math.cos(theta)
            * dt
        )

        y_nuevo = (
            y
            + velocidad_lineal
            * math.sin(theta)
            * dt
        )

    else:

        theta_futuro = (
            theta
            + velocidad_angular * dt
        )

        x_nuevo = (
            x
            + velocidad_lineal
            / velocidad_angular
            * (
                math.sin(theta_futuro)
                - math.sin(theta)
            )
        )

        y_nuevo = (
            y
            - velocidad_lineal
            / velocidad_angular
            * (
                math.cos(theta_futuro)
                - math.cos(theta)
            )
        )

    theta_nuevo = normalizar_angulo(
        theta
        + velocidad_angular * dt
    )

    estado_nuevo = (
        x_nuevo,
        y_nuevo,
        theta_nuevo,
    )

    return estado_nuevo
def simular_cinematica(
    estado_inicial,
    velocidad_lineal,
    velocidad_angular,
    numero_pasos,
    dt=DT,
):

    estados = [
        estado_inicial
    ]

    estado_actual = estado_inicial

    for paso in range(
        numero_pasos
    ):

        estado_actual = actualizar_estado_robot(
            estado_actual,
            velocidad_lineal,
            velocidad_angular,
            dt,
        )

        estados.append(
            estado_actual
        )

    return estados
def dibujar_trayectoria_estados(
    ax,
    estados,
    etiqueta,
    estilo="-",
):

    coordenadas_x = []
    coordenadas_y = []

    for estado in estados:

        coordenadas_x.append(
            estado[0]
        )

        coordenadas_y.append(
            estado[1]
        )

    ax.plot(
        coordenadas_x,
        coordenadas_y,
        linestyle=estilo,
        linewidth=2.0,
        label=etiqueta,
    )
def dibujar_orientacion_robot(
    ax,
    estado,
    longitud=0.7,
):

    x = estado[0]
    y = estado[1]
    theta = estado[2]

    desplazamiento_x = (
        longitud
        * math.cos(theta)
    )

    desplazamiento_y = (
        longitud
        * math.sin(theta)
    )

    ax.arrow(
        x,
        y,
        desplazamiento_x,
        desplazamiento_y,
        width=0.03,
        head_width=0.15,
        head_length=0.20,
        length_includes_head=True,
        zorder=8,
    )
def evaluar_estado_robot(
    estado,
    obstaculos,
):

    x_robot = estado[0]
    y_robot = estado[1]

    colision = robot_colisiona_obstaculos(
        x_robot,
        y_robot,
        obstaculos,
    )

    fuera_del_mapa = robot_fuera_del_mapa(
        x_robot,
        y_robot,
    )

    return colision, fuera_del_mapa
def simular_movimiento_con_colisiones(
    estado_inicial,
    velocidad_lineal,
    velocidad_angular,
    numero_pasos,
    obstaculos,
    dt=DT,
):

    estados = [
        estado_inicial
    ]

    estado_actual = estado_inicial

    resultado = "pasos_completados"

    pasos_ejecutados = 0

    # Comprobar primero que el estado inicial sea válido.
    colision_inicial, fuera_inicial = evaluar_estado_robot(
        estado_actual,
        obstaculos,
    )

    if colision_inicial:

        resultado = "colision_inicial"

        return (
            estados,
            resultado,
            pasos_ejecutados,
        )

    if fuera_inicial:

        resultado = "inicio_fuera_mapa"

        return (
            estados,
            resultado,
            pasos_ejecutados,
        )

    for paso in range(
        numero_pasos
    ):

        estado_nuevo = actualizar_estado_robot(
            estado_actual,
            velocidad_lineal,
            velocidad_angular,
            dt,
        )

        estados.append(
            estado_nuevo
        )

        pasos_ejecutados = paso + 1

        colision, fuera_del_mapa = evaluar_estado_robot(
            estado_nuevo,
            obstaculos,
        )

        if colision:

            resultado = "colision"

            break

        if fuera_del_mapa:

            resultado = "fuera_mapa"

            break

        estado_actual = estado_nuevo

    return (
        estados,
        resultado,
        pasos_ejecutados,
    )
def crear_estado_prueba_colision(
    obstaculos,
    separacion_inicial=0.60,
):

    distancia_centro = (
        RADIO_ROBOT
        + separacion_inicial
    )

    for obstaculo in obstaculos:

        x_centro = (
            obstaculo["x_min"]
            + obstaculo["x_max"]
        ) / 2.0

        y_centro = (
            obstaculo["y_min"]
            + obstaculo["y_max"]
        ) / 2.0

        candidatos = [
            (
                obstaculo["x_min"] - distancia_centro,
                y_centro,
                0.0,
            ),
            (
                obstaculo["x_max"] + distancia_centro,
                y_centro,
                math.pi,
            ),
            (
                x_centro,
                obstaculo["y_min"] - distancia_centro,
                math.pi / 2.0,
            ),
            (
                x_centro,
                obstaculo["y_max"] + distancia_centro,
                -math.pi / 2.0,
            ),
        ]

        for estado_candidato in candidatos:

            colision, fuera_del_mapa = evaluar_estado_robot(
                estado_candidato,
                obstaculos,
            )

            if not colision and not fuera_del_mapa:

                return (
                    estado_candidato,
                    obstaculo,
                )

    return None, None
def dibujar_resultado_simulacion(
    ax,
    estados,
    resultado,
):

    if len(estados) == 0:
        return

    estado_final = estados[-1]

    x_final = estado_final[0]
    y_final = estado_final[1]

    if resultado == "colision":

        marcador = "X"
        color = "red"
        etiqueta = "Punto de colisión"

    elif resultado == "fuera_mapa":

        marcador = "X"
        color = "orange"
        etiqueta = "Salida del mapa"

    else:

        marcador = "o"
        color = "green"
        etiqueta = "Final de simulación"

    ax.scatter(
        x_final,
        y_final,
        marker=marcador,
        s=120,
        color=color,
        edgecolor="black",
        label=etiqueta,
        zorder=10,
    )
def encontrar_indice_mas_cercano(
    estado,
    camino_mundo,
):

    x_robot = estado[0]
    y_robot = estado[1]

    distancia_minima = float("inf")

    indice_mas_cercano = 0

    for indice, punto in enumerate(
        camino_mundo
    ):

        distancia = math.hypot(
            x_robot - punto[0],
            y_robot - punto[1],
        )

        if distancia < distancia_minima:

            distancia_minima = distancia

            indice_mas_cercano = indice

    return indice_mas_cercano
def seleccionar_punto_lookahead(
    estado,
    camino_mundo,
    distancia_lookahead=DISTANCIA_LOOKAHEAD,
):

    indice_cercano = encontrar_indice_mas_cercano(
        estado,
        camino_mundo,
    )

    distancia_acumulada = 0.0

    punto_anterior = (
        estado[0],
        estado[1],
    )

    for indice in range(
        indice_cercano,
        len(camino_mundo),
    ):

        punto_actual = camino_mundo[
            indice
        ]

        distancia_segmento = distancia_entre_puntos(
            punto_anterior,
            punto_actual,
        )

        distancia_acumulada = (
            distancia_acumulada
            + distancia_segmento
        )

        if (
            distancia_acumulada
            >= distancia_lookahead
        ):

            return (
                punto_actual,
                indice,
            )

        punto_anterior = punto_actual

    # Si ya estamos cerca del final de la ruta,
    # se utiliza directamente la meta.
    return (
        camino_mundo[-1],
        len(camino_mundo) - 1,
    )
def controlador_seguimiento_astar(
    estado,
    camino_mundo,
    meta,
):

    x_robot = estado[0]
    y_robot = estado[1]
    theta_robot = estado[2]

    # ------------------------------------------------------
    # Seleccionar punto adelantado
    # ------------------------------------------------------

    punto_objetivo, indice_objetivo = seleccionar_punto_lookahead(
        estado,
        camino_mundo,
        DISTANCIA_LOOKAHEAD,
    )

    # ------------------------------------------------------
    # Calcular orientación deseada
    # ------------------------------------------------------

    diferencia_x = (
        punto_objetivo[0]
        - x_robot
    )

    diferencia_y = (
        punto_objetivo[1]
        - y_robot
    )

    angulo_deseado = math.atan2(
        diferencia_y,
        diferencia_x,
    )

    error_angular = normalizar_angulo(
        angulo_deseado
        - theta_robot
    )

    # ------------------------------------------------------
    # Control angular proporcional
    # ------------------------------------------------------

    velocidad_angular = (
        GANANCIA_ANGULAR
        * error_angular
    )

    velocidad_angular = float(
        np.clip(
            velocidad_angular,
            -VELOCIDAD_ANGULAR_MAXIMA,
            VELOCIDAD_ANGULAR_MAXIMA,
        )
    )

    # ------------------------------------------------------
    # Reducir velocidad cuando existe desalineación
    # ------------------------------------------------------

    factor_alineacion = max(
        0.0,
        math.cos(
            error_angular
        ),
    )

    velocidad_lineal = (
        VELOCIDAD_NOMINAL
        * (
            0.20
            + 0.80 * factor_alineacion
        )
    )

    # ------------------------------------------------------
    # Reducir velocidad cerca de la meta
    # ------------------------------------------------------

    distancia_meta = distancia_entre_puntos(
        (
            x_robot,
            y_robot,
        ),
        meta,
    )

    if (
        distancia_meta
        < DISTANCIA_REDUCCION_VELOCIDAD
    ):

        factor_meta = (
            distancia_meta
            / DISTANCIA_REDUCCION_VELOCIDAD
        )

        velocidad_lineal = (
            velocidad_lineal
            * factor_meta
        )

    velocidad_lineal, velocidad_angular = limitar_control(
        velocidad_lineal,
        velocidad_angular,
    )

    informacion_control = {
        "punto_objetivo": punto_objetivo,
        "indice_objetivo": indice_objetivo,
        "angulo_deseado": angulo_deseado,
        "error_angular": error_angular,
        "distancia_meta": distancia_meta,
    }

    return (
        velocidad_lineal,
        velocidad_angular,
        informacion_control,
    )
def robot_llego_meta(
    estado,
    meta,
):

    distancia = distancia_entre_puntos(
        (
            estado[0],
            estado[1],
        ),
        meta,
    )

    llego = (
        distancia
        <= DISTANCIA_META
    )

    return llego
def simular_seguimiento_astar(
    estado_inicial,
    camino_mundo,
    meta,
    obstaculos,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
):

    estados = [
        estado_inicial
    ]

    controles = []

    puntos_objetivo = []

    errores_angulares = []

    estado_actual = estado_inicial

    resultado = "timeout"

    pasos_ejecutados = 0

    for paso in range(
        pasos_maximos
    ):

        # --------------------------------------------------
        # Comprobar llegada antes de aplicar otro control
        # --------------------------------------------------

        if robot_llego_meta(
            estado_actual,
            meta,
        ):

            resultado = "meta"

            break

        # --------------------------------------------------
        # Calcular acción del controlador
        # --------------------------------------------------

        (
            velocidad_lineal,
            velocidad_angular,
            informacion_control,
        ) = controlador_seguimiento_astar(
            estado_actual,
            camino_mundo,
            meta,
        )

        # --------------------------------------------------
        # Actualizar estado
        # --------------------------------------------------

        estado_nuevo = actualizar_estado_robot(
            estado_actual,
            velocidad_lineal,
            velocidad_angular,
            dt,
        )

        estados.append(
            estado_nuevo
        )

        controles.append(
            (
                velocidad_lineal,
                velocidad_angular,
            )
        )

        puntos_objetivo.append(
            informacion_control[
                "punto_objetivo"
            ]
        )

        errores_angulares.append(
            informacion_control[
                "error_angular"
            ]
        )

        pasos_ejecutados = paso + 1

        # --------------------------------------------------
        # Comprobar seguridad
        # --------------------------------------------------

        colision, fuera_del_mapa = evaluar_estado_robot(
            estado_nuevo,
            obstaculos,
        )

        if colision:

            resultado = "colision"

            break

        if fuera_del_mapa:

            resultado = "fuera_mapa"

            break

        # --------------------------------------------------
        # Comprobar llegada después del movimiento
        # --------------------------------------------------

        if robot_llego_meta(
            estado_nuevo,
            meta,
        ):

            resultado = "meta"

            estado_actual = estado_nuevo

            break

        estado_actual = estado_nuevo

    simulacion = {
        "estados": estados,
        "controles": controles,
        "puntos_objetivo": puntos_objetivo,
        "errores_angulares": errores_angulares,
        "resultado": resultado,
        "pasos_ejecutados": pasos_ejecutados,
        "tiempo_total": pasos_ejecutados * dt,
    }

    return simulacion
def calcular_longitud_trayectoria_robot(
    estados,
):

    longitud_total = 0.0

    for indice in range(
        len(estados) - 1
    ):

        punto_actual = (
            estados[indice][0],
            estados[indice][1],
        )

        punto_siguiente = (
            estados[indice + 1][0],
            estados[indice + 1][1],
        )

        longitud_segmento = distancia_entre_puntos(
            punto_actual,
            punto_siguiente,
        )

        longitud_total = (
            longitud_total
            + longitud_segmento
        )

    return longitud_total
def dibujar_puntos_lookahead(
    ax,
    puntos_objetivo,
):

    if len(puntos_objetivo) == 0:
        return

    coordenadas_x = []
    coordenadas_y = []

    # Dibujamos uno de cada diez puntos para no saturar
    # la visualización.
    for indice in range(
        0,
        len(puntos_objetivo),
        10,
    ):

        punto = puntos_objetivo[
            indice
        ]

        coordenadas_x.append(
            punto[0]
        )

        coordenadas_y.append(
            punto[1]
        )

    ax.scatter(
        coordenadas_x,
        coordenadas_y,
        marker="x",
        s=45,
        label="Objetivos lookahead",
        zorder=8,
    )
def dibujar_controles(
    controles,
    dt=DT,
):

    if len(controles) == 0:
        return

    tiempos = []

    velocidades_lineales = []

    velocidades_angulares = []

    for indice, control in enumerate(
        controles
    ):

        tiempos.append(
            indice * dt
        )

        velocidades_lineales.append(
            control[0]
        )

        velocidades_angulares.append(
            control[1]
        )

    figura_velocidad, ax_velocidad = plt.subplots(
        figsize=(8, 4)
    )

    ax_velocidad.plot(
        tiempos,
        velocidades_lineales,
        linewidth=2.0,
    )

    ax_velocidad.set_xlabel(
        "Tiempo [s]"
    )

    ax_velocidad.set_ylabel(
        "Velocidad lineal [m/s]"
    )

    ax_velocidad.set_title(
        "Velocidad lineal del robot"
    )

    ax_velocidad.grid(
        True,
        alpha=0.3,
    )

    figura_omega, ax_omega = plt.subplots(
        figsize=(8, 4)
    )

    ax_omega.plot(
        tiempos,
        velocidades_angulares,
        linewidth=2.0,
    )

    ax_omega.set_xlabel(
        "Tiempo [s]"
    )

    ax_omega.set_ylabel(
        "Velocidad angular [rad/s]"
    )

    ax_omega.set_title(
        "Velocidad angular del robot"
    )

    ax_omega.grid(
        True,
        alpha=0.3,
    )
def distancia_punto_segmento(
    punto,
    punto_a,
    punto_b,
):

    x = punto[0]
    y = punto[1]

    x_a = punto_a[0]
    y_a = punto_a[1]

    x_b = punto_b[0]
    y_b = punto_b[1]

    desplazamiento_x = (
        x_b - x_a
    )

    desplazamiento_y = (
        y_b - y_a
    )

    longitud_cuadrada = (
        desplazamiento_x ** 2
        + desplazamiento_y ** 2
    )

    # Si ambos extremos son prácticamente iguales.
    if longitud_cuadrada <= 1e-12:

        return distancia_entre_puntos(
            punto,
            punto_a,
        )

    proyeccion = (
        (
            (x - x_a) * desplazamiento_x
            + (y - y_a) * desplazamiento_y
        )
        / longitud_cuadrada
    )

    proyeccion_limitada = float(
        np.clip(
            proyeccion,
            0.0,
            1.0,
        )
    )

    x_proyectado = (
        x_a
        + proyeccion_limitada
        * desplazamiento_x
    )

    y_proyectado = (
        y_a
        + proyeccion_limitada
        * desplazamiento_y
    )

    distancia = math.hypot(
        x - x_proyectado,
        y - y_proyectado,
    )

    return distancia
def distancia_punto_camino(
    punto,
    camino_mundo,
):

    if len(camino_mundo) == 0:
        return float("inf")

    if len(camino_mundo) == 1:

        return distancia_entre_puntos(
            punto,
            camino_mundo[0],
        )

    distancia_minima = float("inf")

    for indice in range(
        len(camino_mundo) - 1
    ):

        punto_a = camino_mundo[
            indice
        ]

        punto_b = camino_mundo[
            indice + 1
        ]

        distancia = distancia_punto_segmento(
            punto,
            punto_a,
            punto_b,
        )

        if distancia < distancia_minima:

            distancia_minima = distancia

    return distancia_minima
def calcular_errores_seguimiento(
    estados,
    camino_mundo,
):

    errores = []

    for estado in estados:

        punto_robot = (
            estado[0],
            estado[1],
        )

        error = distancia_punto_camino(
            punto_robot,
            camino_mundo,
        )

        errores.append(
            error
        )

    return errores
def calcular_clearance_estado(
    estado,
    obstaculos,
):

    x_robot = estado[0]
    y_robot = estado[1]

    # ------------------------------------------------------
    # Separación respecto de los obstáculos
    # ------------------------------------------------------

    clearance_obstaculos = float("inf")

    for obstaculo in obstaculos:

        distancia_centro_obstaculo = distancia_punto_obstaculo(
            x_robot,
            y_robot,
            obstaculo,
        )

        clearance_actual = (
            distancia_centro_obstaculo
            - RADIO_ROBOT
        )

        if clearance_actual < clearance_obstaculos:

            clearance_obstaculos = clearance_actual

    # ------------------------------------------------------
    # Separación respecto de las paredes
    # ------------------------------------------------------

    distancia_pared_izquierda = x_robot
    distancia_pared_derecha = (
        ANCHO_MAPA - x_robot
    )

    distancia_pared_inferior = y_robot
    distancia_pared_superior = (
        ALTO_MAPA - y_robot
    )

    distancia_centro_pared = min(
        distancia_pared_izquierda,
        distancia_pared_derecha,
        distancia_pared_inferior,
        distancia_pared_superior,
    )

    clearance_paredes = (
        distancia_centro_pared
        - RADIO_ROBOT
    )

    # ------------------------------------------------------
    # Clearance total
    # ------------------------------------------------------

    clearance_total = min(
        clearance_obstaculos,
        clearance_paredes,
    )

    return clearance_total
def calcular_clearances_trayectoria(
    estados,
    obstaculos,
):

    clearances = []

    for estado in estados:

        clearance = calcular_clearance_estado(
            estado,
            obstaculos,
        )

        clearances.append(
            clearance
        )

    return clearances
def calcular_metricas_control(
    controles,
    dt=DT,
):

    if len(controles) == 0:

        return {
            "variacion_total_v": 0.0,
            "variacion_total_omega": 0.0,
            "aceleracion_lineal_rms": 0.0,
            "aceleracion_angular_rms": 0.0,
            "esfuerzo_control": 0.0,
        }

    velocidades_lineales = np.array(
        [
            control[0]
            for control in controles
        ],
        dtype=float,
    )

    velocidades_angulares = np.array(
        [
            control[1]
            for control in controles
        ],
        dtype=float,
    )

    if len(controles) >= 2:

        cambios_v = np.diff(
            velocidades_lineales
        )

        cambios_omega = np.diff(
            velocidades_angulares
        )

        variacion_total_v = float(
            np.sum(
                np.abs(
                    cambios_v
                )
            )
        )

        variacion_total_omega = float(
            np.sum(
                np.abs(
                    cambios_omega
                )
            )
        )

        aceleraciones_lineales = (
            cambios_v / dt
        )

        aceleraciones_angulares = (
            cambios_omega / dt
        )

        aceleracion_lineal_rms = float(
            np.sqrt(
                np.mean(
                    aceleraciones_lineales ** 2
                )
            )
        )

        aceleracion_angular_rms = float(
            np.sqrt(
                np.mean(
                    aceleraciones_angulares ** 2
                )
            )
        )

    else:

        variacion_total_v = 0.0
        variacion_total_omega = 0.0
        aceleracion_lineal_rms = 0.0
        aceleracion_angular_rms = 0.0

    # Se normalizan las acciones para que v y omega
    # tengan escalas comparables.
    velocidades_normalizadas = (
        velocidades_lineales
        / VELOCIDAD_MAXIMA
    )

    omegas_normalizadas = (
        velocidades_angulares
        / VELOCIDAD_ANGULAR_MAXIMA
    )

    esfuerzo_control = float(
        np.sum(
            velocidades_normalizadas ** 2
            + omegas_normalizadas ** 2
        )
        * dt
    )

    metricas_control = {
        "variacion_total_v": variacion_total_v,
        "variacion_total_omega": variacion_total_omega,
        "aceleracion_lineal_rms": aceleracion_lineal_rms,
        "aceleracion_angular_rms": aceleracion_angular_rms,
        "esfuerzo_control": esfuerzo_control,
    }

    return metricas_control
def calcular_metricas_seguimiento(
    simulacion,
    camino_mundo,
    meta,
    obstaculos,
    distancia_directa,
    longitud_camino_astar,
    dt=DT,
):

    estados = simulacion[
        "estados"
    ]

    controles = simulacion[
        "controles"
    ]

    resultado = simulacion[
        "resultado"
    ]

    pasos_ejecutados = simulacion[
        "pasos_ejecutados"
    ]

    estado_final = estados[
        -1
    ]

    # ------------------------------------------------------
    # Error respecto de la ruta global
    # ------------------------------------------------------

    errores_seguimiento = calcular_errores_seguimiento(
        estados,
        camino_mundo,
    )

    errores_arreglo = np.array(
        errores_seguimiento,
        dtype=float,
    )

    error_medio = float(
        np.mean(
            errores_arreglo
        )
    )

    error_rmse = float(
        np.sqrt(
            np.mean(
                errores_arreglo ** 2
            )
        )
    )

    error_maximo = float(
        np.max(
            errores_arreglo
        )
    )

    # ------------------------------------------------------
    # Clearance
    # ------------------------------------------------------

    clearances = calcular_clearances_trayectoria(
        estados,
        obstaculos,
    )

    clearances_arreglo = np.array(
        clearances,
        dtype=float,
    )

    clearance_minimo = float(
        np.min(
            clearances_arreglo
        )
    )

    clearance_medio = float(
        np.mean(
            clearances_arreglo
        )
    )

    # ------------------------------------------------------
    # Longitud y distancia final
    # ------------------------------------------------------

    longitud_recorrida = calcular_longitud_trayectoria_robot(
        estados
    )

    distancia_final_meta = distancia_entre_puntos(
        (
            estado_final[0],
            estado_final[1],
        ),
        meta,
    )

    # Como el robot se detiene dentro de una tolerancia,
    # agregamos la distancia residual para comparar contra
    # la distancia completa inicio-meta.
    longitud_equivalente = (
        longitud_recorrida
        + distancia_final_meta
    )

    if longitud_equivalente > 0.0:

        eficiencia_navegacion = (
            distancia_directa
            / longitud_equivalente
        )

    else:

        eficiencia_navegacion = 0.0

    # ------------------------------------------------------
    # Métricas del control
    # ------------------------------------------------------

    metricas_control = calcular_metricas_control(
        controles,
        dt,
    )

    metricas = {
        "exito": resultado == "meta",
        "resultado": resultado,
        "pasos_ejecutados": pasos_ejecutados,
        "tiempo_total": pasos_ejecutados * dt,
        "estado_final": estado_final,
        "distancia_final_meta": distancia_final_meta,
        "longitud_recorrida": longitud_recorrida,
        "longitud_equivalente": longitud_equivalente,
        "longitud_astar": longitud_camino_astar,
        "eficiencia_navegacion": eficiencia_navegacion,
        "error_medio": error_medio,
        "error_rmse": error_rmse,
        "error_maximo": error_maximo,
        "clearance_minimo": clearance_minimo,
        "clearance_medio": clearance_medio,
        "variacion_total_v": metricas_control[
            "variacion_total_v"
        ],
        "variacion_total_omega": metricas_control[
            "variacion_total_omega"
        ],
        "aceleracion_lineal_rms": metricas_control[
            "aceleracion_lineal_rms"
        ],
        "aceleracion_angular_rms": metricas_control[
            "aceleracion_angular_rms"
        ],
        "esfuerzo_control": metricas_control[
            "esfuerzo_control"
        ],
        "errores_seguimiento": errores_seguimiento,
        "clearances": clearances,
    }

    return metricas
def dibujar_metricas_temporales(
    errores_seguimiento,
    clearances,
    dt=DT,
):

    # ======================================================
    # ERROR RESPECTO DE LA RUTA
    # ======================================================

    tiempos_error = np.arange(
        len(errores_seguimiento)
    ) * dt

    figura_error, ax_error = plt.subplots(
        figsize=(8, 4)
    )

    ax_error.plot(
        tiempos_error,
        errores_seguimiento,
        linewidth=2.0,
    )

    ax_error.set_xlabel(
        "Tiempo [s]"
    )

    ax_error.set_ylabel(
        "Error respecto de A* [m]"
    )

    ax_error.set_title(
        "Error de seguimiento de la ruta global"
    )

    ax_error.grid(
        True,
        alpha=0.3,
    )

    # ======================================================
    # CLEARANCE
    # ======================================================

    tiempos_clearance = np.arange(
        len(clearances)
    ) * dt

    figura_clearance, ax_clearance = plt.subplots(
        figsize=(8, 4)
    )

    ax_clearance.plot(
        tiempos_clearance,
        clearances,
        linewidth=2.0,
    )

    ax_clearance.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.5,
        label="Límite de colisión",
    )

    ax_clearance.set_xlabel(
        "Tiempo [s]"
    )

    ax_clearance.set_ylabel(
        "Clearance [m]"
    )

    ax_clearance.set_title(
        "Separación mínima durante la navegación"
    )

    ax_clearance.grid(
        True,
        alpha=0.3,
    )

    ax_clearance.legend()
def obstaculo_dinamico_colisiona_estaticos(
    x,
    y,
    radio,
    obstaculos_estaticos,
    margen=0.0,
):

    for obstaculo in obstaculos_estaticos:

        distancia = distancia_punto_obstaculo(
            x,
            y,
            obstaculo,
        )

        if distancia <= radio + margen:

            return True

    return False
def obstaculo_dinamico_fuera_mapa(
    x,
    y,
    radio,
    margen=0.0,
):

    limite = radio + margen

    fuera_izquierda = (
        x - limite < 0.0
    )

    fuera_derecha = (
        x + limite > ANCHO_MAPA
    )

    fuera_abajo = (
        y - limite < 0.0
    )

    fuera_arriba = (
        y + limite > ALTO_MAPA
    )

    return (
        fuera_izquierda
        or fuera_derecha
        or fuera_abajo
        or fuera_arriba
    )
def posicion_obstaculo_dinamico_es_valida(
    x,
    y,
    radio,
    obstaculos_estaticos,
    obstaculos_dinamicos,
    inicio,
    meta,
):

    # ------------------------------------------------------
    # Verificar límites
    # ------------------------------------------------------

    fuera_mapa = obstaculo_dinamico_fuera_mapa(
        x,
        y,
        radio,
        margen=MARGEN_DINAMICO_ESTATICO,
    )

    if fuera_mapa:
        return False

    # ------------------------------------------------------
    # Verificar obstáculos estáticos
    # ------------------------------------------------------

    colision_estatica = obstaculo_dinamico_colisiona_estaticos(
        x,
        y,
        radio,
        obstaculos_estaticos,
        margen=MARGEN_DINAMICO_ESTATICO,
    )

    if colision_estatica:
        return False

    # ------------------------------------------------------
    # Evitar que aparezca sobre el robot
    # ------------------------------------------------------

    distancia_inicio = distancia_entre_puntos(
        (x, y),
        inicio,
    )

    distancia_minima_inicio = (
        radio
        + RADIO_ROBOT
        + MARGEN_DINAMICO_ESTATICO
    )

    if distancia_inicio <= distancia_minima_inicio:
        return False

    # ------------------------------------------------------
    # Evitar que aparezca directamente sobre la meta
    # ------------------------------------------------------

    distancia_meta = distancia_entre_puntos(
        (x, y),
        meta,
    )

    distancia_minima_meta = (
        radio
        + RADIO_ROBOT
        + MARGEN_DINAMICO_ESTATICO
    )

    if distancia_meta <= distancia_minima_meta:
        return False

    # ------------------------------------------------------
    # Verificar otros obstáculos dinámicos
    # ------------------------------------------------------

    for otro in obstaculos_dinamicos:

        distancia = distancia_entre_puntos(
            (x, y),
            (
                otro["x"],
                otro["y"],
            ),
        )

        distancia_minima = (
            radio
            + otro["radio"]
            + SEPARACION_OBSTACULOS_DINAMICOS
        )

        if distancia <= distancia_minima:
            return False

    return True
def corredor_dinamico_es_libre(
    punto_inicial,
    punto_final,
    radio,
    obstaculos_estaticos,
    numero_muestras=30,
):

    for alpha in np.linspace(
        0.0,
        1.0,
        numero_muestras,
    ):

        x = (
            punto_inicial[0]
            + alpha
            * (
                punto_final[0]
                - punto_inicial[0]
            )
        )

        y = (
            punto_inicial[1]
            + alpha
            * (
                punto_final[1]
                - punto_inicial[1]
            )
        )

        fuera_mapa = obstaculo_dinamico_fuera_mapa(
            x,
            y,
            radio,
            margen=MARGEN_DINAMICO_ESTATICO,
        )

        if fuera_mapa:
            return False

        colision = obstaculo_dinamico_colisiona_estaticos(
            x,
            y,
            radio,
            obstaculos_estaticos,
            margen=MARGEN_DINAMICO_ESTATICO,
        )

        if colision:
            return False

    return True
def generar_obstaculo_dinamico_cruce(
    camino_mundo,
    obstaculos_estaticos,
    obstaculos_dinamicos,
    inicio,
    meta,
    generador,
):

    if len(camino_mundo) < 5:
        return None

    indice_minimo = max(
        2,
        int(
            0.20
            * len(camino_mundo)
        ),
    )

    indice_maximo = min(
        len(camino_mundo) - 3,
        int(
            0.80
            * len(camino_mundo)
        ),
    )

    if indice_maximo < indice_minimo:
        return None

    for intento in range(
        INTENTOS_MAXIMOS_DINAMICOS
    ):

        indice_cruce = int(
            generador.integers(
                indice_minimo,
                indice_maximo + 1,
            )
        )

        punto_anterior = camino_mundo[
            indice_cruce - 1
        ]

        punto_cruce = camino_mundo[
            indice_cruce
        ]

        punto_siguiente = camino_mundo[
            indice_cruce + 1
        ]

        # ==================================================
        # VECTOR TANGENTE DE LA RUTA
        # ==================================================

        tangente_x = (
            punto_siguiente[0]
            - punto_anterior[0]
        )

        tangente_y = (
            punto_siguiente[1]
            - punto_anterior[1]
        )

        norma_tangente = math.hypot(
            tangente_x,
            tangente_y,
        )

        if norma_tangente <= 1e-8:
            continue

        tangente_x /= norma_tangente
        tangente_y /= norma_tangente

        # ==================================================
        # VECTOR PERPENDICULAR
        # ==================================================

        normal_x = -tangente_y
        normal_y = tangente_x

        lado = float(
            generador.choice(
                [-1.0, 1.0]
            )
        )

        distancia_cruce = float(
            generador.uniform(
                DISTANCIA_MINIMA_CRUCE,
                DISTANCIA_MAXIMA_CRUCE,
            )
        )

        radio = float(
            generador.uniform(
                RADIO_MIN_OBSTACULO_DINAMICO,
                RADIO_MAX_OBSTACULO_DINAMICO,
            )
        )

        velocidad = float(
            generador.uniform(
                VELOCIDAD_MIN_OBSTACULO_DINAMICO,
                VELOCIDAD_MAX_OBSTACULO_DINAMICO,
            )
        )

        # ==================================================
        # POSICIÓN INICIAL
        # ==================================================

        x_inicial = (
            punto_cruce[0]
            + lado
            * distancia_cruce
            * normal_x
        )

        y_inicial = (
            punto_cruce[1]
            + lado
            * distancia_cruce
            * normal_y
        )

        # ==================================================
        # POSICIÓN FINAL PREVISTA DEL CRUCE
        # ==================================================

        x_final = (
            punto_cruce[0]
            - lado
            * distancia_cruce
            * normal_x
        )

        y_final = (
            punto_cruce[1]
            - lado
            * distancia_cruce
            * normal_y
        )

        posicion_valida = posicion_obstaculo_dinamico_es_valida(
            x_inicial,
            y_inicial,
            radio,
            obstaculos_estaticos,
            obstaculos_dinamicos,
            inicio,
            meta,
        )

        if not posicion_valida:
            continue

        corredor_valido = corredor_dinamico_es_libre(
            (
                x_inicial,
                y_inicial,
            ),
            (
                x_final,
                y_final,
            ),
            radio,
            obstaculos_estaticos,
        )

        if not corredor_valido:
            continue

        # ==================================================
        # VELOCIDAD HACIA LA RUTA
        # ==================================================

        velocidad_x = (
            -lado
            * velocidad
            * normal_x
        )

        velocidad_y = (
            -lado
            * velocidad
            * normal_y
        )

        # ==================================================
        # SINCRONIZACIÓN TEMPORAL
        # ==================================================

        distancia_robot_cruce = calcular_distancia_camino_hasta_indice(
            camino_mundo,
            indice_cruce,
        )

        tiempo_estimado_robot = (
            distancia_robot_cruce
            / VELOCIDAD_NOMINAL
        )

        tiempo_viaje_obstaculo = (
            distancia_cruce
            / velocidad
        )

        desfase_temporal = float(
            generador.uniform(
                DESFASE_MINIMO_TIEMPO_CRUCE,
                DESFASE_MAXIMO_TIEMPO_CRUCE,
            )
        )

        tiempo_inicio = max(
            0.0,
            tiempo_estimado_robot
            - tiempo_viaje_obstaculo
            + desfase_temporal,
        )

        obstaculo = {
            "x": float(x_inicial),
            "y": float(y_inicial),
            "vx": float(velocidad_x),
            "vy": float(velocidad_y),
            "radio": radio,
            "tipo": "cruce",
            "indice_cruce": indice_cruce,
            "punto_cruce": punto_cruce,
            "tiempo_inicio": float(tiempo_inicio),
            "tiempo_estimado_cruce": float(
                tiempo_inicio
                + tiempo_viaje_obstaculo
            ),
            "tiempo_estimado_robot": float(
                tiempo_estimado_robot
            ),
        }

        return obstaculo

    return None
def generar_obstaculo_dinamico_aleatorio(
    obstaculos_estaticos,
    obstaculos_dinamicos,
    inicio,
    meta,
    generador,
):

    for intento in range(
        INTENTOS_MAXIMOS_DINAMICOS
    ):

        radio = float(
            generador.uniform(
                RADIO_MIN_OBSTACULO_DINAMICO,
                RADIO_MAX_OBSTACULO_DINAMICO,
            )
        )

        x = float(
            generador.uniform(
                radio,
                ANCHO_MAPA - radio,
            )
        )

        y = float(
            generador.uniform(
                radio,
                ALTO_MAPA - radio,
            )
        )

        angulo = float(
            generador.uniform(
                -math.pi,
                math.pi,
            )
        )

        velocidad = float(
            generador.uniform(
                VELOCIDAD_MIN_OBSTACULO_DINAMICO,
                VELOCIDAD_MAX_OBSTACULO_DINAMICO,
            )
        )

        valido = posicion_obstaculo_dinamico_es_valida(
            x,
            y,
            radio,
            obstaculos_estaticos,
            obstaculos_dinamicos,
            inicio,
            meta,
        )

        if not valido:
            continue

        obstaculo = {
            "x": x,
            "y": y,
            "vx": float(
                velocidad
                * math.cos(angulo)
            ),
            "vy": float(
                velocidad
                * math.sin(angulo)
            ),
            "radio": radio,
            "tipo": "aleatorio",
            "indice_cruce": None,
            "punto_cruce": None,
            "tiempo_inicio": 0.0,
            "tiempo_estimado_cruce": None,
            "tiempo_estimado_robot": None,
        }

        return obstaculo

    return None
def generar_obstaculos_dinamicos(
    camino_mundo,
    obstaculos_estaticos,
    inicio,
    meta,
    numero_obstaculos,
    semilla,
):

    generador = np.random.default_rng(
        semilla
    )

    obstaculos_dinamicos = []

    for indice in range(
        numero_obstaculos
    ):

        obstaculo = generar_obstaculo_dinamico_cruce(
            camino_mundo,
            obstaculos_estaticos,
            obstaculos_dinamicos,
            inicio,
            meta,
            generador,
        )

        if obstaculo is None:

            obstaculo = generar_obstaculo_dinamico_aleatorio(
                obstaculos_estaticos,
                obstaculos_dinamicos,
                inicio,
                meta,
                generador,
            )

        if obstaculo is not None:

            obstaculos_dinamicos.append(
                obstaculo
            )

    return obstaculos_dinamicos
def actualizar_obstaculo_dinamico(
    obstaculo,
    obstaculos_estaticos,
    tiempo_actual,
    dt=DT,
):

    actualizado = obstaculo.copy()

    tiempo_inicio = obstaculo.get(
        "tiempo_inicio",
        0.0,
    )

    # El obstáculo permanece detenido hasta el instante
    # programado.
    if tiempo_actual < tiempo_inicio:

        return actualizado

    x = obstaculo["x"]
    y = obstaculo["y"]

    vx = obstaculo["vx"]
    vy = obstaculo["vy"]

    radio = obstaculo["radio"]

    x_nuevo = (
        x + vx * dt
    )

    y_nuevo = (
        y + vy * dt
    )

    # ======================================================
    # REBOTE CONTRA PAREDES
    # ======================================================

    if (
        x_nuevo - radio < 0.0
        or x_nuevo + radio > ANCHO_MAPA
    ):

        vx = -vx

        x_nuevo = (
            x + vx * dt
        )

    if (
        y_nuevo - radio < 0.0
        or y_nuevo + radio > ALTO_MAPA
    ):

        vy = -vy

        y_nuevo = (
            y + vy * dt
        )

    # ======================================================
    # REBOTE CONTRA OBSTÁCULOS ESTÁTICOS
    # ======================================================

    colision_total = obstaculo_dinamico_colisiona_estaticos(
        x_nuevo,
        y_nuevo,
        radio,
        obstaculos_estaticos,
    )

    if colision_total:

        colision_x = obstaculo_dinamico_colisiona_estaticos(
            x + vx * dt,
            y,
            radio,
            obstaculos_estaticos,
        )

        colision_y = obstaculo_dinamico_colisiona_estaticos(
            x,
            y + vy * dt,
            radio,
            obstaculos_estaticos,
        )

        if colision_x:
            vx = -vx

        if colision_y:
            vy = -vy

        if not colision_x and not colision_y:

            vx = -vx
            vy = -vy

        x_nuevo = (
            x + vx * dt
        )

        y_nuevo = (
            y + vy * dt
        )

    sigue_en_colision = obstaculo_dinamico_colisiona_estaticos(
        x_nuevo,
        y_nuevo,
        radio,
        obstaculos_estaticos,
    )

    sigue_fuera = obstaculo_dinamico_fuera_mapa(
        x_nuevo,
        y_nuevo,
        radio,
    )

    if sigue_en_colision or sigue_fuera:

        x_nuevo = x
        y_nuevo = y

        vx = -vx
        vy = -vy

    actualizado["x"] = float(x_nuevo)
    actualizado["y"] = float(y_nuevo)
    actualizado["vx"] = float(vx)
    actualizado["vy"] = float(vy)

    return actualizado
def actualizar_obstaculos_dinamicos(
    obstaculos_dinamicos,
    obstaculos_estaticos,
    tiempo_actual,
    dt=DT,
):

    estados_anteriores = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos
    ]

    actualizados = []

    for obstaculo in obstaculos_dinamicos:

        actualizado = actualizar_obstaculo_dinamico(
            obstaculo,
            obstaculos_estaticos,
            tiempo_actual,
            dt,
        )

        actualizados.append(
            actualizado
        )

    # ======================================================
    # COLISIONES ENTRE OBSTÁCULOS DINÁMICOS
    # ======================================================

    for indice_1 in range(
        len(actualizados)
    ):

        for indice_2 in range(
            indice_1 + 1,
            len(actualizados),
        ):

            obstaculo_1 = actualizados[
                indice_1
            ]

            obstaculo_2 = actualizados[
                indice_2
            ]

            distancia = distancia_entre_puntos(
                (
                    obstaculo_1["x"],
                    obstaculo_1["y"],
                ),
                (
                    obstaculo_2["x"],
                    obstaculo_2["y"],
                ),
            )

            distancia_colision = (
                obstaculo_1["radio"]
                + obstaculo_2["radio"]
            )

            if distancia <= distancia_colision:

                anterior_1 = estados_anteriores[
                    indice_1
                ]

                anterior_2 = estados_anteriores[
                    indice_2
                ]

                obstaculo_1["x"] = anterior_1["x"]
                obstaculo_1["y"] = anterior_1["y"]

                obstaculo_2["x"] = anterior_2["x"]
                obstaculo_2["y"] = anterior_2["y"]

                velocidad_1 = (
                    obstaculo_1["vx"],
                    obstaculo_1["vy"],
                )

                velocidad_2 = (
                    obstaculo_2["vx"],
                    obstaculo_2["vy"],
                )

                obstaculo_1["vx"] = velocidad_2[0]
                obstaculo_1["vy"] = velocidad_2[1]

                obstaculo_2["vx"] = velocidad_1[0]
                obstaculo_2["vy"] = velocidad_1[1]

    return actualizados
def robot_colisiona_obstaculo_dinamico(
    estado_robot,
    obstaculo_dinamico,
):

    distancia = distancia_entre_puntos(
        (
            estado_robot[0],
            estado_robot[1],
        ),
        (
            obstaculo_dinamico["x"],
            obstaculo_dinamico["y"],
        ),
    )

    distancia_colision = (
        RADIO_ROBOT
        + obstaculo_dinamico["radio"]
    )

    return distancia <= distancia_colision
def robot_colisiona_obstaculos_dinamicos(
    estado_robot,
    obstaculos_dinamicos,
):

    for obstaculo in obstaculos_dinamicos:

        colision = robot_colisiona_obstaculo_dinamico(
            estado_robot,
            obstaculo,
        )

        if colision:
            return True

    return False
def simular_obstaculos_dinamicos(
    obstaculos_dinamicos,
    obstaculos_estaticos,
    numero_pasos,
    dt=DT,
):

    obstaculos_actuales = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos
    ]

    trayectorias = []

    for obstaculo in obstaculos_actuales:

        trayectorias.append(
            [
                (
                    obstaculo["x"],
                    obstaculo["y"],
                )
            ]
        )

    for paso in range(
        numero_pasos
    ):

        tiempo_actual = (
            paso + 1
        ) * dt

        obstaculos_actuales = actualizar_obstaculos_dinamicos(
            obstaculos_actuales,
            obstaculos_estaticos,
            tiempo_actual,
            dt,
        )

        for indice, obstaculo in enumerate(
            obstaculos_actuales
        ):

            trayectorias[indice].append(
                (
                    obstaculo["x"],
                    obstaculo["y"],
                )
            )

    return (
        obstaculos_actuales,
        trayectorias,
    )
def obstaculos_dinamicos_son_iguales(
    obstaculos_1,
    obstaculos_2,
):

    if len(obstaculos_1) != len(
        obstaculos_2
    ):

        return False

    for indice in range(
        len(obstaculos_1)
    ):

        if (
            obstaculos_1[indice]
            != obstaculos_2[indice]
        ):

            return False

    return True
def dibujar_obstaculos_dinamicos(
    ax,
    obstaculos_dinamicos,
    etiqueta,
    relleno=True,
    transparencia=0.65,
):

    for indice, obstaculo in enumerate(
        obstaculos_dinamicos
    ):

        etiqueta_actual = None

        if indice == 0:
            etiqueta_actual = etiqueta

        circulo = Circle(
            (
                obstaculo["x"],
                obstaculo["y"],
            ),
            obstaculo["radio"],
            facecolor=(
                "purple"
                if relleno
                else "none"
            ),
            edgecolor="purple",
            linewidth=2.0,
            alpha=transparencia,
            label=etiqueta_actual,
            zorder=8,
        )

        ax.add_patch(
            circulo
        )

        ax.arrow(
            obstaculo["x"],
            obstaculo["y"],
            obstaculo["vx"] * 0.8,
            obstaculo["vy"] * 0.8,
            width=0.015,
            head_width=0.12,
            head_length=0.15,
            length_includes_head=True,
            zorder=9,
        )
def dibujar_trayectorias_dinamicas(
    ax,
    trayectorias,
):

    for indice, trayectoria in enumerate(
        trayectorias
    ):

        coordenadas_x = [
            punto[0]
            for punto in trayectoria
        ]

        coordenadas_y = [
            punto[1]
            for punto in trayectoria
        ]

        etiqueta = None

        if indice == 0:

            etiqueta = (
                "Trayectorias de obstáculos dinámicos"
            )

        ax.plot(
            coordenadas_x,
            coordenadas_y,
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            label=etiqueta,
            zorder=5,
        )
def calcular_distancia_camino_hasta_indice(
    camino_mundo,
    indice_objetivo,
):

    if len(camino_mundo) < 2:
        return 0.0

    indice_objetivo = int(
        np.clip(
            indice_objetivo,
            0,
            len(camino_mundo) - 1,
        )
    )

    distancia_acumulada = 0.0

    for indice in range(
        indice_objetivo
    ):

        punto_actual = camino_mundo[
            indice
        ]

        punto_siguiente = camino_mundo[
            indice + 1
        ]

        distancia_segmento = distancia_entre_puntos(
            punto_actual,
            punto_siguiente,
        )

        distancia_acumulada += distancia_segmento

    return distancia_acumulada
def calcular_clearance_dinamico_estado(
    estado_robot,
    obstaculos_dinamicos,
):

    if len(obstaculos_dinamicos) == 0:

        return float("inf")

    clearance_minimo = float("inf")

    for obstaculo in obstaculos_dinamicos:

        distancia_centros = distancia_entre_puntos(
            (
                estado_robot[0],
                estado_robot[1],
            ),
            (
                obstaculo["x"],
                obstaculo["y"],
            ),
        )

        clearance = (
            distancia_centros
            - RADIO_ROBOT
            - obstaculo["radio"]
        )

        if clearance < clearance_minimo:

            clearance_minimo = clearance

    return clearance_minimo
def obtener_indice_colision_dinamica(
    estado_robot,
    obstaculos_dinamicos,
):

    for indice, obstaculo in enumerate(
        obstaculos_dinamicos
    ):

        if robot_colisiona_obstaculo_dinamico(
            estado_robot,
            obstaculo,
        ):

            return indice

    return None
def simular_seguimiento_dinamico_lookahead(
    estado_inicial,
    camino_mundo,
    meta,
    obstaculos_estaticos,
    obstaculos_dinamicos_iniciales,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
):

    estado_robot = estado_inicial

    obstaculos_dinamicos = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos_iniciales
    ]

    estados_robot = [
        estado_inicial
    ]

    controles = []

    puntos_objetivo = []

    errores_angulares = []

    historial_dinamicos = [
        [
            obstaculo.copy()
            for obstaculo in obstaculos_dinamicos
        ]
    ]

    clearances_dinamicos = [
        calcular_clearance_dinamico_estado(
            estado_inicial,
            obstaculos_dinamicos,
        )
    ]

    historial_submetas = []

    historial_indices_progreso = []

    historial_indices_submeta = []

    historial_observaciones = []

    historial_obstaculos_observados = []

    indice_progreso = 0

    velocidad_lineal_actual = 0.0
    velocidad_angular_actual = 0.0

    resultado = "timeout"

    pasos_ejecutados = 0

    indice_obstaculo_colision = None

    # ======================================================
    # COMPROBACIÓN INICIAL
    # ======================================================

    colision_estatica, fuera_mapa = evaluar_estado_robot(
        estado_robot,
        obstaculos_estaticos,
    )

    if colision_estatica:

        resultado = "colision_estatica"

    elif fuera_mapa:

        resultado = "fuera_mapa"

    elif robot_colisiona_obstaculos_dinamicos(
        estado_robot,
        obstaculos_dinamicos,
    ):

        resultado = "colision_dinamica"

        indice_obstaculo_colision = obtener_indice_colision_dinamica(
            estado_robot,
            obstaculos_dinamicos,
        )

    # ======================================================
    # CICLO PRINCIPAL
    # ======================================================

    if resultado == "timeout":

        for paso in range(
            pasos_maximos
        ):

            if robot_llego_meta(
                estado_robot,
                meta,
            ):

                resultado = "meta"
                break

            tiempo_siguiente = (
                paso + 1
            ) * dt

            # ----------------------------------------------
            # SUBMETA LOCAL ESTABLE
            # ----------------------------------------------

            (
                submeta,
                indice_progreso,
                indice_submeta,
            ) = seleccionar_submeta_local_estable(
                estado_robot,
                camino_mundo,
                indice_progreso,
            )

            # ----------------------------------------------
            # OBSERVACIÓN LOCAL
            # ----------------------------------------------

            (
                observacion_local,
                obstaculos_observados,
            ) = construir_observacion_local(
                estado_robot=estado_robot,
                submeta=submeta,
                meta=meta,
                obstaculos_dinamicos=obstaculos_dinamicos,
                velocidad_lineal_actual=velocidad_lineal_actual,
                velocidad_angular_actual=velocidad_angular_actual,
                indice_progreso=indice_progreso,
                numero_puntos_camino=len(
                    camino_mundo
                ),
            )

            historial_submetas.append(
                submeta
            )

            historial_indices_progreso.append(
                indice_progreso
            )

            historial_indices_submeta.append(
                indice_submeta
            )

            historial_observaciones.append(
                observacion_local
            )

            historial_obstaculos_observados.append(
                obstaculos_observados
            )

            # ----------------------------------------------
            # CONTROL NOMINAL: NO USA LOS OBSTÁCULOS
            # ----------------------------------------------

            (
                velocidad_lineal,
                velocidad_angular,
                informacion_control,
            ) = controlador_submeta_local(
                estado_robot,
                submeta,
                meta,
            )

            # ----------------------------------------------
            # ACTUALIZAR ROBOT
            # ----------------------------------------------

            estado_robot_nuevo = actualizar_estado_robot(
                estado_robot,
                velocidad_lineal,
                velocidad_angular,
                dt,
            )

            # ----------------------------------------------
            # ACTUALIZAR OBSTÁCULOS DINÁMICOS
            # ----------------------------------------------

            obstaculos_dinamicos_nuevos = actualizar_obstaculos_dinamicos(
                obstaculos_dinamicos,
                obstaculos_estaticos,
                tiempo_siguiente,
                dt,
            )

            # ----------------------------------------------
            # GUARDAR RESULTADOS DEL PASO
            # ----------------------------------------------

            estados_robot.append(
                estado_robot_nuevo
            )

            controles.append(
                (
                    velocidad_lineal,
                    velocidad_angular,
                )
            )

            puntos_objetivo.append(
                submeta
            )

            errores_angulares.append(
                informacion_control[
                    "error_angular"
                ]
            )

            historial_dinamicos.append(
                [
                    obstaculo.copy()
                    for obstaculo in obstaculos_dinamicos_nuevos
                ]
            )

            clearance_dinamico = calcular_clearance_dinamico_estado(
                estado_robot_nuevo,
                obstaculos_dinamicos_nuevos,
            )

            clearances_dinamicos.append(
                clearance_dinamico
            )

            pasos_ejecutados = paso + 1

            # ----------------------------------------------
            # COMPROBACIONES DE SEGURIDAD
            # ----------------------------------------------

            (
                colision_estatica,
                fuera_mapa,
            ) = evaluar_estado_robot(
                estado_robot_nuevo,
                obstaculos_estaticos,
            )

            if colision_estatica:

                resultado = "colision_estatica"
                break

            if fuera_mapa:

                resultado = "fuera_mapa"
                break

            if robot_colisiona_obstaculos_dinamicos(
                estado_robot_nuevo,
                obstaculos_dinamicos_nuevos,
            ):

                resultado = "colision_dinamica"

                indice_obstaculo_colision = obtener_indice_colision_dinamica(
                    estado_robot_nuevo,
                    obstaculos_dinamicos_nuevos,
                )

                break

            if robot_llego_meta(
                estado_robot_nuevo,
                meta,
            ):

                resultado = "meta"

                estado_robot = estado_robot_nuevo
                obstaculos_dinamicos = obstaculos_dinamicos_nuevos

                break

            estado_robot = estado_robot_nuevo

            obstaculos_dinamicos = obstaculos_dinamicos_nuevos

            velocidad_lineal_actual = velocidad_lineal

            velocidad_angular_actual = velocidad_angular

    simulacion = {
        "estados": estados_robot,
        "controles": controles,
        "puntos_objetivo": puntos_objetivo,
        "errores_angulares": errores_angulares,
        "resultado": resultado,
        "pasos_ejecutados": pasos_ejecutados,
        "tiempo_total": pasos_ejecutados * dt,
        "historial_dinamicos": historial_dinamicos,
        "clearances_dinamicos": clearances_dinamicos,
        "obstaculos_dinamicos_finales": historial_dinamicos[-1],
        "indice_obstaculo_colision": indice_obstaculo_colision,
        "historial_submetas": historial_submetas,
        "historial_indices_progreso": historial_indices_progreso,
        "historial_indices_submeta": historial_indices_submeta,
        "historial_observaciones": historial_observaciones,
        "historial_obstaculos_observados": historial_obstaculos_observados,
    }

    return simulacion
def extraer_trayectorias_dinamicas(
    historial_dinamicos,
):

    if len(historial_dinamicos) == 0:
        return []

    numero_obstaculos = len(
        historial_dinamicos[0]
    )

    trayectorias = [
        []
        for indice in range(
            numero_obstaculos
        )
    ]

    for instante in historial_dinamicos:

        for indice, obstaculo in enumerate(
            instante
        ):

            trayectorias[indice].append(
                (
                    obstaculo["x"],
                    obstaculo["y"],
                )
            )

    return trayectorias
def dibujar_clearance_dinamico(
    clearances_dinamicos,
    dt=DT,
):

    tiempos = np.arange(
        len(clearances_dinamicos)
    ) * dt

    figura, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        tiempos,
        clearances_dinamicos,
        linewidth=2.0,
    )

    ax.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.5,
        label="Contacto o colisión",
    )

    ax.set_xlabel(
        "Tiempo [s]"
    )

    ax.set_ylabel(
        "Clearance dinámico [m]"
    )

    ax.set_title(
        "Separación respecto de obstáculos dinámicos"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()
def dibujar_resultado_dinamico(
    ax,
    estados_robot,
    resultado,
):

    estado_final = estados_robot[
        -1
    ]

    x_final = estado_final[0]
    y_final = estado_final[1]

    if resultado == "meta":

        marcador = "o"
        etiqueta = "Meta alcanzada"

    elif resultado == "colision_dinamica":

        marcador = "X"
        etiqueta = "Colisión dinámica"

    elif resultado == "colision_estatica":

        marcador = "X"
        etiqueta = "Colisión estática"

    elif resultado == "fuera_mapa":

        marcador = "X"
        etiqueta = "Salida del mapa"

    else:

        marcador = "s"
        etiqueta = "Tiempo agotado"

    ax.scatter(
        x_final,
        y_final,
        marker=marcador,
        s=140,
        edgecolor="black",
        label=etiqueta,
        zorder=12,
    )
def simular_seguimiento_dinamico_animado(
    estado_inicial,
    camino_mundo,
    meta,
    obstaculos_estaticos,
    obstaculos_dinamicos_iniciales,
    rejilla,
    inicio,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
    animar=ANIMAR_SIMULACION,
    pausa=PAUSA_ANIMACION,
):

    # ======================================================
    # 1. COPIAR ESTADOS INICIALES
    # ======================================================

    estado_robot = estado_inicial

    obstaculos_dinamicos = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos_iniciales
    ]

    # ======================================================
    # 2. HISTORIA DE LA SIMULACIÓN
    # ======================================================

    estados_robot = [
        estado_inicial
    ]

    controles = []

    puntos_objetivo = []

    errores_angulares = []

    historial_dinamicos = [
        [
            obstaculo.copy()
            for obstaculo in obstaculos_dinamicos
        ]
    ]

    clearances_dinamicos = [
        calcular_clearance_dinamico_estado(
            estado_robot,
            obstaculos_dinamicos,
        )
    ]

    trayectorias_obstaculos = []

    for obstaculo in obstaculos_dinamicos:

        trayectorias_obstaculos.append(
            [
                (
                    obstaculo["x"],
                    obstaculo["y"],
                )
            ]
        )

    resultado = "timeout"

    pasos_ejecutados = 0

    indice_obstaculo_colision = None

    # ======================================================
    # 3. VARIABLES GRÁFICAS
    # ======================================================

    figura = None
    ax = None

    robot_patch = None
    linea_orientacion = None
    linea_trayectoria_robot = None
    marcador_lookahead = None
    marcador_evento = None

    patches_dinamicos = []
    lineas_velocidad_dinamicos = []
    lineas_trayectorias_dinamicas = []

    # ======================================================
    # 4. CREAR LA FIGURA INTERACTIVA
    # ======================================================

    if animar:

        plt.ion()

        figura, ax = plt.subplots(
            figsize=(9, 9)
        )

        configurar_mapa(
            ax
        )

        dibujar_rejilla_ocupacion(
            ax,
            rejilla,
        )

        dibujar_obstaculos(
            ax,
            obstaculos_estaticos,
        )

        dibujar_camino_astar(
            ax,
            camino_mundo,
        )

        dibujar_inicio(
            ax,
            inicio,
        )

        dibujar_meta(
            ax,
            meta,
        )

        # ==================================================
        # ROBOT
        # ==================================================

        robot_patch = Circle(
            (
                estado_robot[0],
                estado_robot[1],
            ),
            RADIO_ROBOT,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
            label="Robot",
            zorder=10,
        )

        ax.add_patch(
            robot_patch
        )

        longitud_orientacion = 0.75

        x_orientacion = (
            estado_robot[0]
            + longitud_orientacion
            * math.cos(
                estado_robot[2]
            )
        )

        y_orientacion = (
            estado_robot[1]
            + longitud_orientacion
            * math.sin(
                estado_robot[2]
            )
        )

        linea_orientacion, = ax.plot(
            [
                estado_robot[0],
                x_orientacion,
            ],
            [
                estado_robot[1],
                y_orientacion,
            ],
            linewidth=2.5,
            label="Orientación",
            zorder=11,
        )

        linea_trayectoria_robot, = ax.plot(
            [
                estado_robot[0]
            ],
            [
                estado_robot[1]
            ],
            linewidth=2.5,
            label="Trayectoria recorrida",
            zorder=9,
        )

        marcador_lookahead, = ax.plot(
            [],
            [],
            marker="X",
            linestyle="None",
            markersize=9,
            label="Punto lookahead",
            zorder=12,
        )

        marcador_evento, = ax.plot(
            [],
            [],
            marker="X",
            linestyle="None",
            markersize=15,
            markeredgecolor="black",
            label="Evento final",
            zorder=15,
        )

        # ==================================================
        # OBSTÁCULOS DINÁMICOS
        # ==================================================

        for indice, obstaculo in enumerate(
            obstaculos_dinamicos
        ):

            etiqueta = None

            if indice == 0:

                etiqueta = (
                    "Obstáculos dinámicos"
                )

            patch = Circle(
                (
                    obstaculo["x"],
                    obstaculo["y"],
                ),
                obstaculo["radio"],
                alpha=0.75,
                label=etiqueta,
                zorder=10,
            )

            ax.add_patch(
                patch
            )

            patches_dinamicos.append(
                patch
            )

            longitud_vector = 0.8

            linea_velocidad, = ax.plot(
                [
                    obstaculo["x"],
                    obstaculo["x"]
                    + obstaculo["vx"]
                    * longitud_vector,
                ],
                [
                    obstaculo["y"],
                    obstaculo["y"]
                    + obstaculo["vy"]
                    * longitud_vector,
                ],
                linewidth=2.0,
                zorder=11,
            )

            lineas_velocidad_dinamicos.append(
                linea_velocidad
            )

            etiqueta_trayectoria = None

            if indice == 0:

                etiqueta_trayectoria = (
                    "Trayectorias dinámicas"
                )

            linea_trayectoria, = ax.plot(
                [
                    obstaculo["x"]
                ],
                [
                    obstaculo["y"]
                ],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=etiqueta_trayectoria,
                zorder=7,
            )

            lineas_trayectorias_dinamicas.append(
                linea_trayectoria
            )

        ax.set_title(
            "Inicializando simulación dinámica..."
        )

        ax.legend(
            loc="upper right"
        )

        plt.show(
            block=False
        )

        figura.canvas.draw()
        figura.canvas.flush_events()

        plt.pause(
            0.01
        )

    # ======================================================
    # 5. VERIFICAR ESTADO INICIAL
    # ======================================================

    colision_estatica, fuera_mapa = evaluar_estado_robot(
        estado_robot,
        obstaculos_estaticos,
    )

    if colision_estatica:

        resultado = "colision_estatica"

    elif fuera_mapa:

        resultado = "fuera_mapa"

    else:

        indice_obstaculo_colision = obtener_indice_colision_dinamica(
            estado_robot,
            obstaculos_dinamicos,
        )

        if indice_obstaculo_colision is not None:

            resultado = "colision_dinamica"

    # ======================================================
    # 6. CICLO DE SIMULACIÓN Y ANIMACIÓN
    # ======================================================

    if resultado == "timeout":

        for paso in range(
            pasos_maximos
        ):

            tiempo_actual = (
                paso * dt
            )

            tiempo_siguiente = (
                (paso + 1) * dt
            )

            # ==============================================
            # COMPROBAR META ANTES DEL MOVIMIENTO
            # ==============================================

            if robot_llego_meta(
                estado_robot,
                meta,
            ):

                resultado = "meta"
                break

            # ==============================================
            # CALCULAR CONTROL LOOKAHEAD
            # ==============================================

            (
                velocidad_lineal,
                velocidad_angular,
                informacion_control,
            ) = controlador_seguimiento_astar(
                estado_robot,
                camino_mundo,
                meta,
            )

            punto_objetivo = informacion_control[
                "punto_objetivo"
            ]

            error_angular = informacion_control[
                "error_angular"
            ]

            # ==============================================
            # ACTUALIZAR ROBOT
            # ==============================================

            estado_robot_nuevo = actualizar_estado_robot(
                estado_robot,
                velocidad_lineal,
                velocidad_angular,
                dt,
            )

            # ==============================================
            # ACTUALIZAR OBSTÁCULOS DINÁMICOS
            # ==============================================

            obstaculos_dinamicos_nuevos = actualizar_obstaculos_dinamicos(
                obstaculos_dinamicos,
                obstaculos_estaticos,
                tiempo_siguiente,
                dt,
            )

            # ==============================================
            # GUARDAR HISTORIA
            # ==============================================

            estados_robot.append(
                estado_robot_nuevo
            )

            controles.append(
                (
                    velocidad_lineal,
                    velocidad_angular,
                )
            )

            puntos_objetivo.append(
                punto_objetivo
            )

            errores_angulares.append(
                error_angular
            )

            historial_dinamicos.append(
                [
                    obstaculo.copy()
                    for obstaculo in obstaculos_dinamicos_nuevos
                ]
            )

            for indice, obstaculo in enumerate(
                obstaculos_dinamicos_nuevos
            ):

                trayectorias_obstaculos[
                    indice
                ].append(
                    (
                        obstaculo["x"],
                        obstaculo["y"],
                    )
                )

            clearance_dinamico = calcular_clearance_dinamico_estado(
                estado_robot_nuevo,
                obstaculos_dinamicos_nuevos,
            )

            clearances_dinamicos.append(
                clearance_dinamico
            )

            pasos_ejecutados = paso + 1

            # ==============================================
            # DETECTAR COLISIONES
            # ==============================================

            (
                colision_estatica,
                fuera_mapa,
            ) = evaluar_estado_robot(
                estado_robot_nuevo,
                obstaculos_estaticos,
            )

            indice_colision_actual = obtener_indice_colision_dinamica(
                estado_robot_nuevo,
                obstaculos_dinamicos_nuevos,
            )

            evento_actual = None

            if colision_estatica:

                resultado = "colision_estatica"
                evento_actual = resultado

            elif fuera_mapa:

                resultado = "fuera_mapa"
                evento_actual = resultado

            elif indice_colision_actual is not None:

                resultado = "colision_dinamica"

                indice_obstaculo_colision = (
                    indice_colision_actual
                )

                evento_actual = resultado

            elif robot_llego_meta(
                estado_robot_nuevo,
                meta,
            ):

                resultado = "meta"
                evento_actual = resultado

            # ==============================================
            # ACTUALIZAR VARIABLES PRINCIPALES
            # ==============================================

            estado_robot = estado_robot_nuevo

            obstaculos_dinamicos = obstaculos_dinamicos_nuevos

            # ==============================================
            # ACTUALIZAR ANIMACIÓN
            # ==============================================

            if animar:

                # ------------------------------------------
                # Robot
                # ------------------------------------------

                robot_patch.center = (
                    estado_robot[0],
                    estado_robot[1],
                )

                longitud_orientacion = 0.75

                x_orientacion = (
                    estado_robot[0]
                    + longitud_orientacion
                    * math.cos(
                        estado_robot[2]
                    )
                )

                y_orientacion = (
                    estado_robot[1]
                    + longitud_orientacion
                    * math.sin(
                        estado_robot[2]
                    )
                )

                linea_orientacion.set_data(
                    [
                        estado_robot[0],
                        x_orientacion,
                    ],
                    [
                        estado_robot[1],
                        y_orientacion,
                    ],
                )

                # ------------------------------------------
                # Rastro del robot
                # ------------------------------------------

                coordenadas_robot_x = [
                    estado[0]
                    for estado in estados_robot
                ]

                coordenadas_robot_y = [
                    estado[1]
                    for estado in estados_robot
                ]

                linea_trayectoria_robot.set_data(
                    coordenadas_robot_x,
                    coordenadas_robot_y,
                )

                # ------------------------------------------
                # Punto lookahead
                # ------------------------------------------

                marcador_lookahead.set_data(
                    [
                        punto_objetivo[0]
                    ],
                    [
                        punto_objetivo[1]
                    ],
                )

                # ------------------------------------------
                # Obstáculos dinámicos
                # ------------------------------------------

                for indice, obstaculo in enumerate(
                    obstaculos_dinamicos
                ):

                    patches_dinamicos[
                        indice
                    ].center = (
                        obstaculo["x"],
                        obstaculo["y"],
                    )

                    longitud_vector = 0.8

                    lineas_velocidad_dinamicos[
                        indice
                    ].set_data(
                        [
                            obstaculo["x"],
                            obstaculo["x"]
                            + obstaculo["vx"]
                            * longitud_vector,
                        ],
                        [
                            obstaculo["y"],
                            obstaculo["y"]
                            + obstaculo["vy"]
                            * longitud_vector,
                        ],
                    )

                    if MOSTRAR_RASTROS_DINAMICOS:

                        trayectoria = trayectorias_obstaculos[
                            indice
                        ]

                        coordenadas_x = [
                            punto[0]
                            for punto in trayectoria
                        ]

                        coordenadas_y = [
                            punto[1]
                            for punto in trayectoria
                        ]

                        lineas_trayectorias_dinamicas[
                            indice
                        ].set_data(
                            coordenadas_x,
                            coordenadas_y,
                        )

                # ------------------------------------------
                # Distancia a la meta
                # ------------------------------------------

                distancia_meta_actual = distancia_entre_puntos(
                    (
                        estado_robot[0],
                        estado_robot[1],
                    ),
                    meta,
                )

                # ------------------------------------------
                # Título durante movimiento
                # ------------------------------------------

                titulo_estado = (
                    "SIMULACIÓN EN MOVIMIENTO"
                )

                if evento_actual == "meta":

                    titulo_estado = (
                        "META ALCANZADA"
                    )

                elif evento_actual == "colision_dinamica":

                    titulo_estado = (
                        "COLISIÓN DINÁMICA"
                    )

                elif evento_actual == "colision_estatica":

                    titulo_estado = (
                        "COLISIÓN ESTÁTICA"
                    )

                elif evento_actual == "fuera_mapa":

                    titulo_estado = (
                        "ROBOT FUERA DEL MAPA"
                    )

                ax.set_title(
                    f"{titulo_estado}\n"
                    f"t = {tiempo_siguiente:.2f} s | "
                    f"v = {velocidad_lineal:.2f} m/s | "
                    f"ω = {velocidad_angular:.2f} rad/s | "
                    f"d_meta = {distancia_meta_actual:.2f} m | "
                    f"clearance dinámico = "
                    f"{clearance_dinamico:.3f} m"
                )

                # ------------------------------------------
                # Marcar evento
                # ------------------------------------------

                if evento_actual is not None:

                    marcador_evento.set_data(
                        [
                            estado_robot[0]
                        ],
                        [
                            estado_robot[1]
                        ],
                    )

                figura.canvas.draw()
                figura.canvas.flush_events()

                plt.pause(
                    pausa
                )

            # ==============================================
            # TERMINAR SI OCURRIÓ UN EVENTO
            # ==============================================

            if evento_actual is not None:
                break

    # ======================================================
    # 7. TÍTULO FINAL
    # ======================================================

    if animar:

        clearance_final = clearances_dinamicos[
            -1
        ]

        if resultado == "meta":

            texto_final = (
                "META ALCANZADA"
            )

        elif resultado == "colision_dinamica":

            texto_final = (
                "COLISIÓN DINÁMICA"
            )

        elif resultado == "colision_estatica":

            texto_final = (
                "COLISIÓN ESTÁTICA"
            )

        elif resultado == "fuera_mapa":

            texto_final = (
                "ROBOT FUERA DEL MAPA"
            )

        else:

            texto_final = (
                "TIEMPO AGOTADO"
            )

        ax.set_title(
            f"{texto_final}\n"
            f"Resultado = {resultado} | "
            f"t = {pasos_ejecutados * dt:.2f} s | "
            f"clearance dinámico final = "
            f"{clearance_final:.3f} m"
        )

        figura.canvas.draw()
        figura.canvas.flush_events()

        # Dejar visible el instante final.
        plt.pause(
            1.5
        )

        plt.ioff()

    # ======================================================
    # 8. DEVOLVER RESULTADOS
    # ======================================================

    simulacion = {
        "estados": estados_robot,
        "controles": controles,
        "puntos_objetivo": puntos_objetivo,
        "errores_angulares": errores_angulares,
        "resultado": resultado,
        "pasos_ejecutados": pasos_ejecutados,
        "tiempo_total": pasos_ejecutados * dt,
        "historial_dinamicos": historial_dinamicos,
        "clearances_dinamicos": clearances_dinamicos,
        "obstaculos_dinamicos_finales": obstaculos_dinamicos,
        "indice_obstaculo_colision": indice_obstaculo_colision,
        "figura_animacion": figura,
    }

    return simulacion
def transformar_punto_al_marco_robot(
    estado_robot,
    punto_mundo,
):

    x_robot = estado_robot[0]
    y_robot = estado_robot[1]
    theta_robot = estado_robot[2]

    diferencia_x = (
        punto_mundo[0]
        - x_robot
    )

    diferencia_y = (
        punto_mundo[1]
        - y_robot
    )

    coseno = math.cos(
        theta_robot
    )

    seno = math.sin(
        theta_robot
    )

    x_relativo = (
        coseno * diferencia_x
        + seno * diferencia_y
    )

    y_relativo = (
        -seno * diferencia_x
        + coseno * diferencia_y
    )

    return (
        x_relativo,
        y_relativo,
    )
def transformar_vector_al_marco_robot(
    estado_robot,
    vector_mundo,
):

    theta_robot = estado_robot[2]

    coseno = math.cos(
        theta_robot
    )

    seno = math.sin(
        theta_robot
    )

    velocidad_x_relativa = (
        coseno * vector_mundo[0]
        + seno * vector_mundo[1]
    )

    velocidad_y_relativa = (
        -seno * vector_mundo[0]
        + coseno * vector_mundo[1]
    )

    return (
        velocidad_x_relativa,
        velocidad_y_relativa,
    )
def actualizar_indice_progreso_ruta(
    estado_robot,
    camino_mundo,
    indice_anterior,
    ventana=VENTANA_BUSQUEDA_PROGRESO,
):

    if len(camino_mundo) == 0:
        return 0

    indice_inicial = int(
        np.clip(
            indice_anterior,
            0,
            len(camino_mundo) - 1,
        )
    )

    indice_final = min(
        len(camino_mundo) - 1,
        indice_inicial + ventana,
    )

    punto_robot = (
        estado_robot[0],
        estado_robot[1],
    )

    mejor_indice = indice_inicial

    mejor_distancia = distancia_entre_puntos(
        punto_robot,
        camino_mundo[indice_inicial],
    )

    for indice in range(
        indice_inicial + 1,
        indice_final + 1,
    ):

        distancia = distancia_entre_puntos(
            punto_robot,
            camino_mundo[indice],
        )

        if distancia < mejor_distancia:

            mejor_distancia = distancia
            mejor_indice = indice

    return mejor_indice
def seleccionar_submeta_local_estable(
    estado_robot,
    camino_mundo,
    indice_progreso_anterior,
    distancia_submeta=DISTANCIA_SUBMETA_LOCAL,
):

    indice_progreso = actualizar_indice_progreso_ruta(
        estado_robot,
        camino_mundo,
        indice_progreso_anterior,
    )

    punto_robot = (
        estado_robot[0],
        estado_robot[1],
    )

    distancia_acumulada = distancia_entre_puntos(
        punto_robot,
        camino_mundo[indice_progreso],
    )

    indice_submeta = indice_progreso

    for indice in range(
        indice_progreso,
        len(camino_mundo) - 1,
    ):

        if distancia_acumulada >= distancia_submeta:
            break

        punto_actual = camino_mundo[
            indice
        ]

        punto_siguiente = camino_mundo[
            indice + 1
        ]

        distancia_segmento = distancia_entre_puntos(
            punto_actual,
            punto_siguiente,
        )

        distancia_acumulada += distancia_segmento

        indice_submeta = indice + 1

    submeta = camino_mundo[
        indice_submeta
    ]

    return (
        submeta,
        indice_progreso,
        indice_submeta,
    )
def describir_obstaculo_dinamico_local(
    estado_robot,
    obstaculo,
):

    posicion_relativa = transformar_punto_al_marco_robot(
        estado_robot,
        (
            obstaculo["x"],
            obstaculo["y"],
        ),
    )

    velocidad_relativa = transformar_vector_al_marco_robot(
        estado_robot,
        (
            obstaculo["vx"],
            obstaculo["vy"],
        ),
    )

    distancia_centro = math.hypot(
        posicion_relativa[0],
        posicion_relativa[1],
    )

    clearance = (
        distancia_centro
        - RADIO_ROBOT
        - obstaculo["radio"]
    )

    angulo_relativo = math.atan2(
        posicion_relativa[1],
        posicion_relativa[0],
    )

    descripcion = {
        "x_relativo": posicion_relativa[0],
        "y_relativo": posicion_relativa[1],
        "vx_relativo": velocidad_relativa[0],
        "vy_relativo": velocidad_relativa[1],
        "distancia_centro": distancia_centro,
        "clearance": clearance,
        "angulo_relativo": angulo_relativo,
        "radio": obstaculo["radio"],
        "x_mundo": obstaculo["x"],
        "y_mundo": obstaculo["y"],
    }

    return descripcion
def obtener_obstaculos_dinamicos_observados(
    estado_robot,
    obstaculos_dinamicos,
    radio_percepcion=RADIO_PERCEPCION_DINAMICA,
    maximo_obstaculos=MAX_OBSTACULOS_DINAMICOS_OBSERVADOS,
):

    observados = []

    for obstaculo in obstaculos_dinamicos:

        descripcion = describir_obstaculo_dinamico_local(
            estado_robot,
            obstaculo,
        )

        visible = (
            descripcion["distancia_centro"]
            <= radio_percepcion
            + obstaculo["radio"]
        )

        if visible:

            observados.append(
                descripcion
            )

    observados.sort(
        key=lambda elemento: elemento[
            "clearance"
        ]
    )

    return observados[
        :maximo_obstaculos
    ]
def construir_parche_egocentrico_sac(
    estado_robot,
    obstaculos_estaticos,
    obstaculos_dinamicos,
):

    # ------------------------------------------------------
    # 1. Estado del robot
    # ------------------------------------------------------

    x_robot = estado_robot[0]
    y_robot = estado_robot[1]
    theta_robot = estado_robot[2]

    # ------------------------------------------------------
    # 2. Geometría del parche
    # ------------------------------------------------------

    mitad_parche = (
        TAMANO_PARCHE_SAC
        / 2.0
    )

    tamano_celda = (
        TAMANO_PARCHE_SAC
        / RESOLUCION_PARCHE_SAC
    )

    # Coordenada longitudinal:
    #
    # fila 0  -> zona situada delante del robot
    # fila 31 -> zona situada detrás del robot

    coordenadas_adelante = (
        mitad_parche
        - (
            np.arange(
                RESOLUCION_PARCHE_SAC
            )
            + 0.5
        )
        * tamano_celda
    )

    # Coordenada lateral:
    #
    # columna 0  -> lado izquierdo del robot
    # columna 31 -> lado derecho del robot

    coordenadas_izquierda = (
        mitad_parche
        - (
            np.arange(
                RESOLUCION_PARCHE_SAC
            )
            + 0.5
        )
        * tamano_celda
    )

    # Crear las coordenadas locales de los centros
    # de todas las celdas.

    (
        x_local,
        y_local,
    ) = np.meshgrid(
        coordenadas_adelante,
        coordenadas_izquierda,
        indexing="ij",
    )

    # ------------------------------------------------------
    # 3. Transformar las celdas al marco global
    # ------------------------------------------------------

    coseno = math.cos(
        theta_robot
    )

    seno = math.sin(
        theta_robot
    )

    x_mundo = (
        x_robot
        + coseno * x_local
        - seno * y_local
    )

    y_mundo = (
        y_robot
        + seno * x_local
        + coseno * y_local
    )

    # ------------------------------------------------------
    # 4. Inicializar los dos canales
    # ------------------------------------------------------

    canal_estatico = np.zeros(
        (
            RESOLUCION_PARCHE_SAC,
            RESOLUCION_PARCHE_SAC,
        ),
        dtype=np.float32,
    )

    canal_dinamico = np.zeros(
        (
            RESOLUCION_PARCHE_SAC,
            RESOLUCION_PARCHE_SAC,
        ),
        dtype=np.float32,
    )

    # ------------------------------------------------------
    # 5. Marcar los límites del mapa
    # ------------------------------------------------------

    fuera_del_mapa = (
        (x_mundo < RADIO_ROBOT)
        | (
            x_mundo
            > ANCHO_MAPA - RADIO_ROBOT
        )
        | (y_mundo < RADIO_ROBOT)
        | (
            y_mundo
            > ALTO_MAPA - RADIO_ROBOT
        )
    )

    canal_estatico[
        fuera_del_mapa
    ] = 1.0

    # ------------------------------------------------------
    # 6. Marcar obstáculos estáticos
    # ------------------------------------------------------

    for obstaculo in obstaculos_estaticos:

        # Se infla el obstáculo con el radio del robot.
        #
        # Así el parche representa el espacio prohibido
        # para el centro del robot, no solamente la forma
        # física del obstáculo.

        ocupado = (
            (
                x_mundo
                >= obstaculo["x_min"]
                - RADIO_ROBOT
            )
            & (
                x_mundo
                <= obstaculo["x_max"]
                + RADIO_ROBOT
            )
            & (
                y_mundo
                >= obstaculo["y_min"]
                - RADIO_ROBOT
            )
            & (
                y_mundo
                <= obstaculo["y_max"]
                + RADIO_ROBOT
            )
        )

        canal_estatico[
            ocupado
        ] = 1.0

    # ------------------------------------------------------
    # 7. Marcar obstáculos dinámicos
    # ------------------------------------------------------

    for obstaculo in obstaculos_dinamicos:

        radio_ocupado = (
            obstaculo["radio"]
            + RADIO_ROBOT
        )

        distancia_cuadrada = (
            (
                x_mundo
                - obstaculo["x"]
            ) ** 2
            + (
                y_mundo
                - obstaculo["y"]
            ) ** 2
        )

        ocupado = (
            distancia_cuadrada
            <= radio_ocupado ** 2
        )

        canal_dinamico[
            ocupado
        ] = 1.0

    # ------------------------------------------------------
    # 8. Unir los canales
    # ------------------------------------------------------

    parche = np.stack(
        [
            canal_estatico,
            canal_dinamico,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    return parche
def construir_escalares_sac(
    estado_robot,
    submeta,
    meta,
    velocidad_lineal_actual,
    velocidad_angular_actual,
    indice_progreso,
    numero_puntos_camino,
):

    # ------------------------------------------------------
    # 1. Submeta expresada en el marco del robot
    # ------------------------------------------------------

    submeta_local = transformar_punto_al_marco_robot(
        estado_robot,
        submeta,
    )

    x_submeta_local = submeta_local[
        0
    ]

    y_submeta_local = submeta_local[
        1
    ]

    # ------------------------------------------------------
    # 2. Distancia del robot a la meta global
    # ------------------------------------------------------

    distancia_meta = distancia_entre_puntos(
        (
            estado_robot[0],
            estado_robot[1],
        ),
        meta,
    )

    # Distancia máxima aproximada dentro del mapa.
    diagonal_mapa = math.hypot(
        ANCHO_MAPA,
        ALTO_MAPA,
    )

    # ------------------------------------------------------
    # 3. Progreso normalizado sobre la ruta A*
    # ------------------------------------------------------

    if numero_puntos_camino > 1:

        progreso_normalizado = (
            indice_progreso
            / (
                numero_puntos_camino
                - 1
            )
        )

    else:

        progreso_normalizado = 1.0

    # ------------------------------------------------------
    # 4. Normalizar los seis valores
    # ------------------------------------------------------

    escala_submeta = max(
        DISTANCIA_SUBMETA_LOCAL,
        1e-8,
    )

    x_submeta_normalizado = float(
        np.clip(
            x_submeta_local
            / escala_submeta,
            -1.0,
            1.0,
        )
    )

    y_submeta_normalizado = float(
        np.clip(
            y_submeta_local
            / escala_submeta,
            -1.0,
            1.0,
        )
    )

    distancia_meta_normalizada = float(
        np.clip(
            distancia_meta
            / max(
                diagonal_mapa,
                1e-8,
            ),
            0.0,
            1.0,
        )
    )

    velocidad_lineal_normalizada = float(
        np.clip(
            velocidad_lineal_actual
            / max(
                VELOCIDAD_MAXIMA,
                1e-8,
            ),
            0.0,
            1.0,
        )
    )

    velocidad_angular_normalizada = float(
        np.clip(
            velocidad_angular_actual
            / max(
                VELOCIDAD_ANGULAR_MAXIMA,
                1e-8,
            ),
            -1.0,
            1.0,
        )
    )

    progreso_normalizado = float(
        np.clip(
            progreso_normalizado,
            0.0,
            1.0,
        )
    )

    # ------------------------------------------------------
    # 5. Construir el vector escalar
    # ------------------------------------------------------

    escalares = np.array(
        [
            x_submeta_normalizado,
            y_submeta_normalizado,
            distancia_meta_normalizada,
            velocidad_lineal_normalizada,
            velocidad_angular_normalizada,
            progreso_normalizado,
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------
    # 6. Verificar la dimensión
    # ------------------------------------------------------

    if escalares.shape != (
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "La rama escalar SAC tiene dimensión "
            f"{escalares.shape}; se esperaba "
            f"({DIMENSION_ESCALARES_SAC},)."
        )

    if not np.all(
        np.isfinite(
            escalares
        )
    ):

        raise ValueError(
            "La rama escalar SAC contiene "
            "NaN o valores infinitos."
        )

    return escalares
def construir_observacion_sac(
    estado_robot,
    submeta,
    meta,
    obstaculos_estaticos,
    obstaculos_dinamicos,
    velocidad_lineal_actual,
    velocidad_angular_actual,
    indice_progreso,
    numero_puntos_camino,
):

    # ------------------------------------------------------
    # 1. Construir la rama espacial
    # ------------------------------------------------------

    parche = construir_parche_egocentrico_sac(
        estado_robot=estado_robot,
        obstaculos_estaticos=(
            obstaculos_estaticos
        ),
        obstaculos_dinamicos=(
            obstaculos_dinamicos
        ),
    )

    # ------------------------------------------------------
    # 2. Construir la rama escalar
    # ------------------------------------------------------

    escalares = construir_escalares_sac(
        estado_robot=estado_robot,
        submeta=submeta,
        meta=meta,
        velocidad_lineal_actual=(
            velocidad_lineal_actual
        ),
        velocidad_angular_actual=(
            velocidad_angular_actual
        ),
        indice_progreso=indice_progreso,
        numero_puntos_camino=(
            numero_puntos_camino
        ),
    )

    # ------------------------------------------------------
    # 3. Verificar la rama espacial
    # ------------------------------------------------------

    forma_parche_esperada = (
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if parche.shape != forma_parche_esperada:

        raise ValueError(
            "El parche SAC tiene forma "
            f"{parche.shape}; se esperaba "
            f"{forma_parche_esperada}."
        )

    if parche.dtype != np.float32:

        parche = parche.astype(
            np.float32
        )

    if not np.all(
        np.isfinite(
            parche
        )
    ):

        raise ValueError(
            "El parche SAC contiene NaN "
            "o valores infinitos."
        )

    # ------------------------------------------------------
    # 4. Verificar la rama escalar
    # ------------------------------------------------------

    forma_escalares_esperada = (
        DIMENSION_ESCALARES_SAC,
    )

    if escalares.shape != forma_escalares_esperada:

        raise ValueError(
            "Los escalares SAC tienen forma "
            f"{escalares.shape}; se esperaba "
            f"{forma_escalares_esperada}."
        )

    if escalares.dtype != np.float32:

        escalares = escalares.astype(
            np.float32
        )

    if not np.all(
        np.isfinite(
            escalares
        )
    ):

        raise ValueError(
            "Los escalares SAC contienen NaN "
            "o valores infinitos."
        )

    # ------------------------------------------------------
    # 5. Reunir ambas ramas
    # ------------------------------------------------------

    observacion = {
        "parche": parche.copy(),

        "escalares": escalares.copy(),
    }

    return observacion
def convertir_observacion_sac_a_tensores(
    observacion,
    dispositivo,
):

    # ------------------------------------------------------
    # 1. Verificar la estructura de la observación
    # ------------------------------------------------------

    if not isinstance(
        observacion,
        dict,
    ):

        raise TypeError(
            "La observación SAC debe ser un diccionario."
        )

    if (
        "parche" not in observacion
        or "escalares" not in observacion
    ):

        raise KeyError(
            "La observación debe contener "
            "'parche' y 'escalares'."
        )

    parche = observacion[
        "parche"
    ]

    escalares = observacion[
        "escalares"
    ]

    # ------------------------------------------------------
    # 2. Verificar las dimensiones NumPy
    # ------------------------------------------------------

    forma_parche_esperada = (
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if parche.shape != forma_parche_esperada:

        raise ValueError(
            "El parche tiene forma "
            f"{parche.shape}; se esperaba "
            f"{forma_parche_esperada}."
        )

    if escalares.shape != (
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "Los escalares tienen forma "
            f"{escalares.shape}; se esperaba "
            f"({DIMENSION_ESCALARES_SAC},)."
        )

    # ------------------------------------------------------
    # 3. Convertir el parche a tensor
    # ------------------------------------------------------

    tensor_parche = torch.as_tensor(
        parche,
        dtype=torch.float32,
        device=dispositivo,
    )

    # Agregar la dimensión del lote:
    #
    # (2, 32, 32)
    #       ↓
    # (1, 2, 32, 32)

    tensor_parche = tensor_parche.unsqueeze(
        0
    )

    # ------------------------------------------------------
    # 4. Convertir los escalares a tensor
    # ------------------------------------------------------

    tensor_escalares = torch.as_tensor(
        escalares,
        dtype=torch.float32,
        device=dispositivo,
    )

    # Agregar la dimensión del lote:
    #
    # (6,)
    #   ↓
    # (1, 6)

    tensor_escalares = tensor_escalares.unsqueeze(
        0
    )

    # ------------------------------------------------------
    # 5. Verificar los tensores resultantes
    # ------------------------------------------------------

    forma_tensor_parche = (
        1,
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if tensor_parche.shape != forma_tensor_parche:

        raise ValueError(
            "El tensor del parche tiene forma "
            f"{tuple(tensor_parche.shape)}; "
            f"se esperaba {forma_tensor_parche}."
        )

    forma_tensor_escalares = (
        1,
        DIMENSION_ESCALARES_SAC,
    )

    if tensor_escalares.shape != (
        forma_tensor_escalares
    ):

        raise ValueError(
            "El tensor escalar tiene forma "
            f"{tuple(tensor_escalares.shape)}; "
            f"se esperaba {forma_tensor_escalares}."
        )

    if not torch.isfinite(
        tensor_parche
    ).all():

        raise ValueError(
            "El tensor del parche contiene "
            "NaN o valores infinitos."
        )

    if not torch.isfinite(
        tensor_escalares
    ).all():

        raise ValueError(
            "El tensor escalar contiene "
            "NaN o valores infinitos."
        )

    return (
        tensor_parche,
        tensor_escalares,
    )
def crear_rama_cnn_sac(
    dispositivo,
):

    rama_cnn = nn.Sequential(

        # --------------------------------------------------
        # Entrada:
        # (lote, 2, 32, 32)
        #
        # Salida:
        # (lote, 16, 16, 16)
        # --------------------------------------------------

        nn.Conv2d(
            in_channels=(
                CANALES_PARCHE_SAC
            ),
            out_channels=(
                CANALES_CNN_1_SAC
            ),
            kernel_size=3,
            stride=2,
            padding=1,
        ),

        nn.ReLU(),

        # --------------------------------------------------
        # Entrada:
        # (lote, 16, 16, 16)
        #
        # Salida:
        # (lote, 32, 8, 8)
        # --------------------------------------------------

        nn.Conv2d(
            in_channels=(
                CANALES_CNN_1_SAC
            ),
            out_channels=(
                CANALES_CNN_2_SAC
            ),
            kernel_size=3,
            stride=2,
            padding=1,
        ),

        nn.ReLU(),

        # --------------------------------------------------
        # Entrada:
        # (lote, 32, 8, 8)
        #
        # Salida:
        # (lote, 64, 4, 4)
        # --------------------------------------------------

        nn.Conv2d(
            in_channels=(
                CANALES_CNN_2_SAC
            ),
            out_channels=(
                CANALES_CNN_3_SAC
            ),
            kernel_size=3,
            stride=2,
            padding=1,
        ),

        nn.ReLU(),

        # --------------------------------------------------
        # Entrada:
        # (lote, 64, 4, 4)
        #
        # Salida:
        # (lote, 1024)
        # --------------------------------------------------

        nn.Flatten(),

        # --------------------------------------------------
        # Entrada:
        # 64 * 4 * 4 = 1024
        #
        # Salida:
        # 128 características espaciales
        # --------------------------------------------------

        nn.Linear(
            in_features=(
                CANALES_CNN_3_SAC
                * 4
                * 4
            ),
            out_features=(
                DIMENSION_CARACTERISTICAS_CNN_SAC
            ),
        ),

        nn.ReLU(),
    )

    rama_cnn = rama_cnn.to(
        dispositivo
    )

    return rama_cnn
def crear_rama_mlp_escalares_sac(
    dispositivo,
):

    rama_mlp = nn.Sequential(

        # --------------------------------------------------
        # Entrada:
        # (lote, 6)
        #
        # Salida:
        # (lote, 64)
        # --------------------------------------------------

        nn.Linear(
            in_features=(
                DIMENSION_ESCALARES_SAC
            ),
            out_features=(
                NEURONAS_MLP_ESCALARES_SAC
            ),
        ),

        nn.ReLU(),

        # --------------------------------------------------
        # Entrada:
        # (lote, 64)
        #
        # Salida:
        # (lote, 64)
        # --------------------------------------------------

        nn.Linear(
            in_features=(
                NEURONAS_MLP_ESCALARES_SAC
            ),
            out_features=(
                DIMENSION_CARACTERISTICAS_MLP_SAC
            ),
        ),

        nn.ReLU(),
    )

    rama_mlp = rama_mlp.to(
        dispositivo
    )

    return rama_mlp
def fusionar_caracteristicas_sac(
    caracteristicas_cnn,
    caracteristicas_mlp,
):

    # ------------------------------------------------------
    # 1. Verificar que las entradas sean tensores
    # ------------------------------------------------------

    if not torch.is_tensor(
        caracteristicas_cnn
    ):

        raise TypeError(
            "Las características CNN deben ser "
            "un tensor de PyTorch."
        )

    if not torch.is_tensor(
        caracteristicas_mlp
    ):

        raise TypeError(
            "Las características MLP deben ser "
            "un tensor de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar dimensiones generales
    # ------------------------------------------------------

    if caracteristicas_cnn.ndim != 2:

        raise ValueError(
            "Las características CNN deben tener forma "
            "(lote, características)."
        )

    if caracteristicas_mlp.ndim != 2:

        raise ValueError(
            "Las características MLP deben tener forma "
            "(lote, características)."
        )

    numero_elementos_lote_cnn = (
        caracteristicas_cnn.shape[
            0
        ]
    )

    numero_elementos_lote_mlp = (
        caracteristicas_mlp.shape[
            0
        ]
    )

    if (
        numero_elementos_lote_cnn
        != numero_elementos_lote_mlp
    ):

        raise ValueError(
            "Las ramas CNN y MLP deben tener "
            "el mismo tamaño de lote."
        )

    # ------------------------------------------------------
    # 3. Verificar número de características
    # ------------------------------------------------------

    if caracteristicas_cnn.shape[1] != (
        DIMENSION_CARACTERISTICAS_CNN_SAC
    ):

        raise ValueError(
            "La salida CNN tiene "
            f"{caracteristicas_cnn.shape[1]} "
            "características; se esperaban "
            f"{DIMENSION_CARACTERISTICAS_CNN_SAC}."
        )

    if caracteristicas_mlp.shape[1] != (
        DIMENSION_CARACTERISTICAS_MLP_SAC
    ):

        raise ValueError(
            "La salida MLP tiene "
            f"{caracteristicas_mlp.shape[1]} "
            "características; se esperaban "
            f"{DIMENSION_CARACTERISTICAS_MLP_SAC}."
        )

    # ------------------------------------------------------
    # 4. Verificar dispositivo y tipo de datos
    # ------------------------------------------------------

    if (
        caracteristicas_cnn.device
        != caracteristicas_mlp.device
    ):

        raise ValueError(
            "Las características CNN y MLP deben "
            "estar en el mismo dispositivo."
        )

    if (
        caracteristicas_cnn.dtype
        != caracteristicas_mlp.dtype
    ):

        raise TypeError(
            "Las características CNN y MLP deben "
            "tener el mismo tipo de datos."
        )

    # ------------------------------------------------------
    # 5. Verificar valores finitos
    # ------------------------------------------------------

    if not torch.isfinite(
        caracteristicas_cnn
    ).all():

        raise ValueError(
            "Las características CNN contienen "
            "NaN o valores infinitos."
        )

    if not torch.isfinite(
        caracteristicas_mlp
    ).all():

        raise ValueError(
            "Las características MLP contienen "
            "NaN o valores infinitos."
        )

    # ------------------------------------------------------
    # 6. Concatenar las dos ramas
    # ------------------------------------------------------
    #
    # CNN:
    # (lote, 128)
    #
    # MLP:
    # (lote, 64)
    #
    # Resultado:
    # (lote, 192)

    caracteristicas_fusionadas = torch.cat(
        [
            caracteristicas_cnn,
            caracteristicas_mlp,
        ],
        dim=1,
    )

    # ------------------------------------------------------
    # 7. Verificar la salida
    # ------------------------------------------------------

    forma_esperada = (
        numero_elementos_lote_cnn,
        DIMENSION_CARACTERISTICAS_FUSIONADAS_SAC,
    )

    if tuple(
        caracteristicas_fusionadas.shape
    ) != forma_esperada:

        raise ValueError(
            "Las características fusionadas tienen forma "
            f"{tuple(caracteristicas_fusionadas.shape)}; "
            f"se esperaba {forma_esperada}."
        )

    if not torch.isfinite(
        caracteristicas_fusionadas
    ).all():

        raise ValueError(
            "Las características fusionadas contienen "
            "NaN o valores infinitos."
        )

    return caracteristicas_fusionadas
def crear_red_compartida_actor_sac(
    dispositivo,
):

    red_compartida = nn.Sequential(

        # --------------------------------------------------
        # Entrada:
        # (lote, 192)
        #
        # Salida:
        # (lote, 256)
        # --------------------------------------------------

        nn.Linear(
            in_features=(
                DIMENSION_CARACTERISTICAS_FUSIONADAS_SAC
            ),
            out_features=(
                NEURONAS_CAPA_1_ACTOR_SAC
            ),
        ),

        nn.ReLU(),

        # --------------------------------------------------
        # Entrada:
        # (lote, 256)
        #
        # Salida:
        # (lote, 256)
        # --------------------------------------------------

        nn.Linear(
            in_features=(
                NEURONAS_CAPA_1_ACTOR_SAC
            ),
            out_features=(
                NEURONAS_CAPA_2_ACTOR_SAC
            ),
        ),

        nn.ReLU(),
    )

    red_compartida = red_compartida.to(
        dispositivo
    )

    return red_compartida
def crear_cabezas_actor_sac(
    dispositivo,
):

    # ------------------------------------------------------
    # Cabeza de la media
    # ------------------------------------------------------
    #
    # Entrada:
    # (lote, 256)
    #
    # Salida:
    # (lote, 2)

    cabeza_media = nn.Linear(
        in_features=(
            NEURONAS_CAPA_2_ACTOR_SAC
        ),
        out_features=(
            DIMENSION_ACCION_SAC
        ),
    )

    # ------------------------------------------------------
    # Cabeza del logaritmo de la desviación estándar
    # ------------------------------------------------------
    #
    # Entrada:
    # (lote, 256)
    #
    # Salida:
    # (lote, 2)

    cabeza_log_desviacion = nn.Linear(
        in_features=(
            NEURONAS_CAPA_2_ACTOR_SAC
        ),
        out_features=(
            DIMENSION_ACCION_SAC
        ),
    )

    # Mover ambas cabezas al dispositivo seleccionado.

    cabeza_media = cabeza_media.to(
        dispositivo
    )

    cabeza_log_desviacion = (
        cabeza_log_desviacion.to(
            dispositivo
        )
    )

    return (
        cabeza_media,
        cabeza_log_desviacion,
    )
def crear_actor_sac(
    dispositivo,
):

    # ------------------------------------------------------
    # 1. Crear las dos cabezas
    # ------------------------------------------------------

    (
        cabeza_media,
        cabeza_log_desviacion,
    ) = crear_cabezas_actor_sac(
        dispositivo=dispositivo
    )

    # ------------------------------------------------------
    # 2. Reunir todos los módulos
    # ------------------------------------------------------
    #
    # ModuleDict funciona como un diccionario, pero registra
    # correctamente todos los parámetros de las redes.

    actor = nn.ModuleDict(
        {
            "rama_cnn": crear_rama_cnn_sac(
                dispositivo=dispositivo
            ),

            "rama_mlp": (
                crear_rama_mlp_escalares_sac(
                    dispositivo=dispositivo
                )
            ),

            "red_compartida": (
                crear_red_compartida_actor_sac(
                    dispositivo=dispositivo
                )
            ),

            "cabeza_media": cabeza_media,

            "cabeza_log_desviacion": (
                cabeza_log_desviacion
            ),
        }
    )

    # ------------------------------------------------------
    # 3. Mover todo el actor al dispositivo
    # ------------------------------------------------------

    actor = actor.to(
        dispositivo
    )

    # ------------------------------------------------------
    # 4. Verificar las claves
    # ------------------------------------------------------

    claves_esperadas = {
        "rama_cnn",
        "rama_mlp",
        "red_compartida",
        "cabeza_media",
        "cabeza_log_desviacion",
    }

    claves_obtenidas = set(
        actor.keys()
    )

    if claves_obtenidas != claves_esperadas:

        raise RuntimeError(
            "El actor SAC no contiene todos los "
            "módulos esperados."
        )

    # ------------------------------------------------------
    # 5. Verificar que existan parámetros
    # ------------------------------------------------------

    numero_parametros = sum(
        parametro.numel()
        for parametro in actor.parameters()
        if parametro.requires_grad
    )

    if numero_parametros <= 0:

        raise RuntimeError(
            "El actor SAC no contiene parámetros "
            "entrenables."
        )

    return actor
def crear_optimizador_actor_sac(
    actor,
    tasa_aprendizaje=(
        TASA_APRENDIZAJE_ACTOR_SAC
    ),
):

    # ------------------------------------------------------
    # 1. Verificar el actor
    # ------------------------------------------------------

    if not isinstance(
        actor,
        nn.Module,
    ):

        raise TypeError(
            "El actor debe ser un módulo de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar la tasa de aprendizaje
    # ------------------------------------------------------

    tasa_aprendizaje = float(
        tasa_aprendizaje
    )

    if not np.isfinite(
        tasa_aprendizaje
    ):

        raise ValueError(
            "La tasa de aprendizaje debe ser finita."
        )

    if tasa_aprendizaje <= 0.0:

        raise ValueError(
            "La tasa de aprendizaje debe ser "
            "estrictamente positiva."
        )

    # ------------------------------------------------------
    # 3. Obtener los parámetros entrenables
    # ------------------------------------------------------

    parametros_entrenables = [
        parametro
        for parametro in actor.parameters()
        if parametro.requires_grad
    ]

    if len(
        parametros_entrenables
    ) == 0:

        raise RuntimeError(
            "El actor no contiene parámetros entrenables."
        )

    # ------------------------------------------------------
    # 4. Verificar los parámetros
    # ------------------------------------------------------

    for parametro in parametros_entrenables:

        if not torch.isfinite(
            parametro
        ).all():

            raise ValueError(
                "El actor contiene parámetros con "
                "NaN o valores infinitos."
            )

    # ------------------------------------------------------
    # 5. Crear el optimizador Adam
    # ------------------------------------------------------

    optimizador = torch.optim.Adam(
        params=parametros_entrenables,

        lr=tasa_aprendizaje,

        betas=(
            BETA_1_ADAM_ACTOR_SAC,
            BETA_2_ADAM_ACTOR_SAC,
        ),

        eps=EPSILON_ADAM_ACTOR_SAC,

        weight_decay=0.0,
    )

    # ------------------------------------------------------
    # 6. Comprobar el registro de parámetros
    # ------------------------------------------------------

    identificadores_actor = {
        id(
            parametro
        )
        for parametro in parametros_entrenables
    }

    identificadores_optimizador = {
        id(
            parametro
        )
        for grupo in optimizador.param_groups
        for parametro in grupo[
            "params"
        ]
    }

    if (
        identificadores_actor
        != identificadores_optimizador
    ):

        raise RuntimeError(
            "El optimizador no contiene exactamente "
            "todos los parámetros entrenables del actor."
        )

    # ------------------------------------------------------
    # 7. Verificar la tasa registrada
    # ------------------------------------------------------

    for grupo in optimizador.param_groups:

        if not np.isclose(
            grupo[
                "lr"
            ],
            tasa_aprendizaje,
        ):

            raise RuntimeError(
                "La tasa de aprendizaje del optimizador "
                "no coincide con la solicitada."
            )

    return optimizador
def crear_rama_accion_critico_sac(
    dispositivo,
):

    rama_accion = nn.Sequential(

        # Entrada:
        # (lote, 2)

        nn.Linear(
            in_features=(
                DIMENSION_ACCION_SAC
            ),

            out_features=(
                NEURONAS_RAMA_ACCION_CRITICO_SAC
            ),
        ),

        nn.ReLU(),

        # Salida:
        # (lote, 64)

        nn.Linear(
            in_features=(
                NEURONAS_RAMA_ACCION_CRITICO_SAC
            ),

            out_features=(
                DIMENSION_CARACTERISTICAS_ACCION_CRITICO_SAC
            ),
        ),

        nn.ReLU(),
    )

    rama_accion = rama_accion.to(
        dispositivo
    )

    return rama_accion
def crear_red_valor_q_sac(
    dispositivo,
):

    red_valor_q = nn.Sequential(

        # Entrada:
        # características del estado y de la acción
        #
        # (lote, 256)

        nn.Linear(
            in_features=(
                DIMENSION_ENTRADA_RED_Q_SAC
            ),

            out_features=(
                NEURONAS_CAPA_1_CRITICO_SAC
            ),
        ),

        nn.ReLU(),

        nn.Linear(
            in_features=(
                NEURONAS_CAPA_1_CRITICO_SAC
            ),

            out_features=(
                NEURONAS_CAPA_2_CRITICO_SAC
            ),
        ),

        nn.ReLU(),

        # Salida escalar Q(s,a).
        #
        # No se agrega activación porque el valor Q
        # puede ser positivo o negativo.

        nn.Linear(
            in_features=(
                NEURONAS_CAPA_2_CRITICO_SAC
            ),

            out_features=(
                DIMENSION_SALIDA_CRITICO_SAC
            ),
        ),
    )

    red_valor_q = red_valor_q.to(
        dispositivo
    )

    return red_valor_q
def crear_critico_sac(
    dispositivo,
):

    critico = nn.ModuleDict(
        {
            # Codificación espacial del parche.

            "rama_cnn": crear_rama_cnn_sac(
                dispositivo=dispositivo
            ),

            # Codificación de los seis escalares.

            "rama_mlp": (
                crear_rama_mlp_escalares_sac(
                    dispositivo=dispositivo
                )
            ),

            # Codificación de la acción normalizada.

            "rama_accion": (
                crear_rama_accion_critico_sac(
                    dispositivo=dispositivo
                )
            ),

            # Red final Q(s,a).

            "red_valor_q": crear_red_valor_q_sac(
                dispositivo=dispositivo
            ),
        }
    )

    critico = critico.to(
        dispositivo
    )

    claves_esperadas = {
        "rama_cnn",
        "rama_mlp",
        "rama_accion",
        "red_valor_q",
    }

    if set(
        critico.keys()
    ) != claves_esperadas:

        raise RuntimeError(
            "El crítico SAC no contiene todos "
            "los módulos esperados."
        )

    numero_parametros = sum(
        parametro.numel()
        for parametro in critico.parameters()
        if parametro.requires_grad
    )

    if numero_parametros <= 0:

        raise RuntimeError(
            "El crítico no contiene parámetros "
            "entrenables."
        )

    return critico
def crear_dos_criticos_sac(
    dispositivo,
):

    critico_q1 = crear_critico_sac(
        dispositivo=dispositivo
    )

    critico_q2 = crear_critico_sac(
        dispositivo=dispositivo
    )

    # ------------------------------------------------------
    # Verificar que no compartan parámetros
    # ------------------------------------------------------

    identificadores_q1 = {
        id(
            parametro
        )
        for parametro in critico_q1.parameters()
    }

    identificadores_q2 = {
        id(
            parametro
        )
        for parametro in critico_q2.parameters()
    }

    parametros_compartidos = (
        identificadores_q1.intersection(
            identificadores_q2
        )
    )

    if len(
        parametros_compartidos
    ) != 0:

        raise RuntimeError(
            "Q1 y Q2 no deben compartir parámetros."
        )

    return (
        critico_q1,
        critico_q2,
    )
def crear_dos_criticos_objetivo_sac(
    critico_q1,
    critico_q2,
    dispositivo,
):

    # ------------------------------------------------------
    # 1. Verificar los críticos en línea
    # ------------------------------------------------------

    if not isinstance(
        critico_q1,
        nn.ModuleDict,
    ):

        raise TypeError(
            "Q1 debe ser un nn.ModuleDict."
        )

    if not isinstance(
        critico_q2,
        nn.ModuleDict,
    ):

        raise TypeError(
            "Q2 debe ser un nn.ModuleDict."
        )

    # ------------------------------------------------------
    # 2. Crear arquitecturas independientes
    # ------------------------------------------------------

    critico_q1_objetivo = crear_critico_sac(
        dispositivo=dispositivo
    )

    critico_q2_objetivo = crear_critico_sac(
        dispositivo=dispositivo
    )

    # ------------------------------------------------------
    # 3. Copiar exactamente los pesos iniciales
    # ------------------------------------------------------

    critico_q1_objetivo.load_state_dict(
        critico_q1.state_dict()
    )

    critico_q2_objetivo.load_state_dict(
        critico_q2.state_dict()
    )

    # ------------------------------------------------------
    # 4. Desactivar sus gradientes
    # ------------------------------------------------------

    for parametro in (
        critico_q1_objetivo.parameters()
    ):

        parametro.requires_grad_(
            False
        )

    for parametro in (
        critico_q2_objetivo.parameters()
    ):

        parametro.requires_grad_(
            False
        )

    # Los críticos objetivo se utilizan solamente
    # para inferencia del valor objetivo.

    critico_q1_objetivo.eval()

    critico_q2_objetivo.eval()

    # ------------------------------------------------------
    # 5. Verificar que las copias sean exactas
    # ------------------------------------------------------

    copia_q1_correcta = all(
        torch.equal(
            parametro_q1,
            parametro_objetivo,
        )
        for (
            parametro_q1,
            parametro_objetivo,
        ) in zip(
            critico_q1.parameters(),
            critico_q1_objetivo.parameters(),
        )
    )

    copia_q2_correcta = all(
        torch.equal(
            parametro_q2,
            parametro_objetivo,
        )
        for (
            parametro_q2,
            parametro_objetivo,
        ) in zip(
            critico_q2.parameters(),
            critico_q2_objetivo.parameters(),
        )
    )

    if not copia_q1_correcta:

        raise RuntimeError(
            "Q1 objetivo no es una copia exacta de Q1."
        )

    if not copia_q2_correcta:

        raise RuntimeError(
            "Q2 objetivo no es una copia exacta de Q2."
        )

    # ------------------------------------------------------
    # 6. Verificar que no compartan parámetros
    # ------------------------------------------------------

    identificadores_q1 = {
        id(
            parametro
        )
        for parametro in critico_q1.parameters()
    }

    identificadores_q2 = {
        id(
            parametro
        )
        for parametro in critico_q2.parameters()
    }

    identificadores_q1_objetivo = {
        id(
            parametro
        )
        for parametro in (
            critico_q1_objetivo.parameters()
        )
    }

    identificadores_q2_objetivo = {
        id(
            parametro
        )
        for parametro in (
            critico_q2_objetivo.parameters()
        )
    }

    todos_los_identificadores = [
        identificadores_q1,
        identificadores_q2,
        identificadores_q1_objetivo,
        identificadores_q2_objetivo,
    ]

    for indice_a in range(
        len(
            todos_los_identificadores
        )
    ):

        for indice_b in range(
            indice_a + 1,
            len(
                todos_los_identificadores
            ),
        ):

            parametros_compartidos = (
                todos_los_identificadores[
                    indice_a
                ].intersection(
                    todos_los_identificadores[
                        indice_b
                    ]
                )
            )

            if len(
                parametros_compartidos
            ) != 0:

                raise RuntimeError(
                    "Los críticos en línea y objetivo "
                    "no deben compartir parámetros."
                )

    return (
        critico_q1_objetivo,
        critico_q2_objetivo,
    )
def actualizar_critico_objetivo_sac(
    critico_en_linea,
    critico_objetivo,
    tau=TAU_POLYAK_SAC,
):

    # ------------------------------------------------------
    # 1. Verificar los modelos
    # ------------------------------------------------------

    if not isinstance(
        critico_en_linea,
        nn.Module,
    ):

        raise TypeError(
            "El crítico en línea debe ser "
            "un módulo de PyTorch."
        )

    if not isinstance(
        critico_objetivo,
        nn.Module,
    ):

        raise TypeError(
            "El crítico objetivo debe ser "
            "un módulo de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar tau
    # ------------------------------------------------------

    tau = float(
        tau
    )

    if not np.isfinite(
        tau
    ):

        raise ValueError(
            "Tau debe ser un número finito."
        )

    if tau <= 0.0 or tau > 1.0:

        raise ValueError(
            "Tau debe pertenecer al intervalo (0, 1]."
        )

    # ------------------------------------------------------
    # 3. Obtener los parámetros
    # ------------------------------------------------------

    parametros_en_linea = list(
        critico_en_linea.parameters()
    )

    parametros_objetivo = list(
        critico_objetivo.parameters()
    )

    if len(
        parametros_en_linea
    ) != len(
        parametros_objetivo
    ):

        raise ValueError(
            "El crítico en línea y el objetivo deben "
            "tener el mismo número de parámetros."
        )

    # ------------------------------------------------------
    # 4. Actualización suave
    # ------------------------------------------------------

    with torch.no_grad():

        for (
            parametro_en_linea,
            parametro_objetivo,
        ) in zip(
            parametros_en_linea,
            parametros_objetivo,
        ):

            if (
                parametro_en_linea.shape
                != parametro_objetivo.shape
            ):

                raise ValueError(
                    "Los parámetros correspondientes "
                    "deben tener la misma forma."
                )

            if (
                parametro_en_linea.device
                != parametro_objetivo.device
            ):

                raise ValueError(
                    "Los parámetros correspondientes "
                    "deben estar en el mismo dispositivo."
                )

            # θ_objetivo =
            # (1 - tau) θ_objetivo
            # + tau θ_en_linea

            parametro_objetivo.mul_(
                1.0 - tau
            )

            parametro_objetivo.add_(
                parametro_en_linea,
                alpha=tau,
            )

    # ------------------------------------------------------
    # 5. Mantener el crítico objetivo congelado
    # ------------------------------------------------------

    for parametro in (
        critico_objetivo.parameters()
    ):

        parametro.requires_grad_(
            False
        )

    critico_objetivo.eval()
def actualizar_dos_criticos_objetivo_sac(
    critico_q1,
    critico_q2,
    critico_q1_objetivo,
    critico_q2_objetivo,
    tau=TAU_POLYAK_SAC,
):

    actualizar_critico_objetivo_sac(
        critico_en_linea=critico_q1,
        critico_objetivo=(
            critico_q1_objetivo
        ),
        tau=tau,
    )

    actualizar_critico_objetivo_sac(
        critico_en_linea=critico_q2,
        critico_objetivo=(
            critico_q2_objetivo
        ),
        tau=tau,
    )
def ejecutar_critico_sac(
    tensor_parche,
    tensor_escalares,
    tensor_accion,
    critico,
):

    # ------------------------------------------------------
    # 1. Verificar el crítico
    # ------------------------------------------------------

    if not isinstance(
        critico,
        nn.ModuleDict,
    ):

        raise TypeError(
            "El crítico debe ser un nn.ModuleDict."
        )

    claves_necesarias = {
        "rama_cnn",
        "rama_mlp",
        "rama_accion",
        "red_valor_q",
    }

    if set(
        critico.keys()
    ) != claves_necesarias:

        raise ValueError(
            "El crítico no contiene los módulos "
            "necesarios."
        )

    # ------------------------------------------------------
    # 2. Verificar los tensores
    # ------------------------------------------------------

    if not torch.is_tensor(
        tensor_parche
    ):

        raise TypeError(
            "El parche debe ser un tensor."
        )

    if not torch.is_tensor(
        tensor_escalares
    ):

        raise TypeError(
            "Los escalares deben ser un tensor."
        )

    if not torch.is_tensor(
        tensor_accion
    ):

        raise TypeError(
            "La acción debe ser un tensor."
        )

    if tensor_parche.ndim != 4:

        raise ValueError(
            "El parche debe tener forma "
            "(lote, canales, alto, ancho)."
        )

    if tensor_escalares.ndim != 2:

        raise ValueError(
            "Los escalares deben tener forma "
            "(lote, escalares)."
        )

    if tensor_accion.ndim != 2:

        raise ValueError(
            "La acción debe tener forma "
            "(lote, acciones)."
        )

    # ------------------------------------------------------
    # 3. Verificar las dimensiones específicas
    # ------------------------------------------------------

    if tuple(
        tensor_parche.shape[1:]
    ) != (
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    ):

        raise ValueError(
            "El parche no tiene las dimensiones "
            "esperadas."
        )

    if tensor_escalares.shape[1] != (
        DIMENSION_ESCALARES_SAC
    ):

        raise ValueError(
            "El tensor de escalares tiene una "
            "dimensión incorrecta."
        )

    if tensor_accion.shape[1] != (
        DIMENSION_ACCION_SAC
    ):

        raise ValueError(
            "El tensor de acción tiene una "
            "dimensión incorrecta."
        )

    numero_elementos_lote = (
        tensor_parche.shape[0]
    )

    if (
        tensor_escalares.shape[0]
        != numero_elementos_lote
        or tensor_accion.shape[0]
        != numero_elementos_lote
    ):

        raise ValueError(
            "El parche, los escalares y la acción "
            "deben tener el mismo tamaño de lote."
        )

    # ------------------------------------------------------
    # 4. Verificar dispositivo y tipo de datos
    # ------------------------------------------------------

    if not (
        tensor_parche.device
        == tensor_escalares.device
        == tensor_accion.device
    ):

        raise ValueError(
            "Todos los tensores deben estar en "
            "el mismo dispositivo."
        )

    if not (
        tensor_parche.dtype
        == tensor_escalares.dtype
        == tensor_accion.dtype
    ):

        raise TypeError(
            "Todos los tensores deben tener el "
            "mismo tipo de datos."
        )

    # ------------------------------------------------------
    # 5. Verificar valores
    # ------------------------------------------------------

    if not torch.isfinite(
        tensor_parche
    ).all():

        raise ValueError(
            "El parche contiene NaN o infinitos."
        )

    if not torch.isfinite(
        tensor_escalares
    ).all():

        raise ValueError(
            "Los escalares contienen NaN o infinitos."
        )

    if not torch.isfinite(
        tensor_accion
    ).all():

        raise ValueError(
            "La acción contiene NaN o infinitos."
        )

    if not torch.all(
        tensor_accion >= -1.0
    ):

        raise ValueError(
            "La acción normalizada contiene valores "
            "menores que -1."
        )

    if not torch.all(
        tensor_accion <= 1.0
    ):

        raise ValueError(
            "La acción normalizada contiene valores "
            "mayores que 1."
        )

    # ------------------------------------------------------
    # 6. Procesar la observación
    # ------------------------------------------------------

    caracteristicas_cnn = critico[
        "rama_cnn"
    ](
        tensor_parche
    )

    caracteristicas_mlp = critico[
        "rama_mlp"
    ](
        tensor_escalares
    )

    caracteristicas_estado = (
        fusionar_caracteristicas_sac(
            caracteristicas_cnn=(
                caracteristicas_cnn
            ),

            caracteristicas_mlp=(
                caracteristicas_mlp
            ),
        )
    )

    # ------------------------------------------------------
    # 7. Procesar la acción
    # ------------------------------------------------------

    caracteristicas_accion = critico[
        "rama_accion"
    ](
        tensor_accion
    )

    if tuple(
        caracteristicas_accion.shape
    ) != (
        numero_elementos_lote,
        DIMENSION_CARACTERISTICAS_ACCION_CRITICO_SAC,
    ):

        raise ValueError(
            "La rama de acción produjo una forma "
            "incorrecta."
        )

    # ------------------------------------------------------
    # 8. Fusionar observación y acción
    # ------------------------------------------------------

    caracteristicas_estado_accion = torch.cat(
        [
            caracteristicas_estado,
            caracteristicas_accion,
        ],
        dim=1,
    )

    if tuple(
        caracteristicas_estado_accion.shape
    ) != (
        numero_elementos_lote,
        DIMENSION_ENTRADA_RED_Q_SAC,
    ):

        raise ValueError(
            "La fusión estado-acción produjo una "
            "forma incorrecta."
        )

    # ------------------------------------------------------
    # 9. Calcular Q(s,a)
    # ------------------------------------------------------

    valor_q = critico[
        "red_valor_q"
    ](
        caracteristicas_estado_accion
    )

    if tuple(
        valor_q.shape
    ) != (
        numero_elementos_lote,
        DIMENSION_SALIDA_CRITICO_SAC,
    ):

        raise ValueError(
            "El crítico debe producir un valor escalar "
            "por elemento del lote."
        )

    if not torch.isfinite(
        valor_q
    ).all():

        raise ValueError(
            "El valor Q contiene NaN o infinitos."
        )

    resultados = {
        "caracteristicas_cnn": (
            caracteristicas_cnn
        ),

        "caracteristicas_mlp": (
            caracteristicas_mlp
        ),

        "caracteristicas_estado": (
            caracteristicas_estado
        ),

        "caracteristicas_accion": (
            caracteristicas_accion
        ),

        "caracteristicas_estado_accion": (
            caracteristicas_estado_accion
        ),

        "valor_q": valor_q,
    }

    return resultados
def ejecutar_dos_criticos_sac(
    tensor_parche,
    tensor_escalares,
    tensor_accion,
    critico_q1,
    critico_q2,
):

    resultados_q1 = ejecutar_critico_sac(
        tensor_parche=tensor_parche,
        tensor_escalares=tensor_escalares,
        tensor_accion=tensor_accion,
        critico=critico_q1,
    )

    resultados_q2 = ejecutar_critico_sac(
        tensor_parche=tensor_parche,
        tensor_escalares=tensor_escalares,
        tensor_accion=tensor_accion,
        critico=critico_q2,
    )

    valor_q1 = resultados_q1[
        "valor_q"
    ]

    valor_q2 = resultados_q2[
        "valor_q"
    ]

    if valor_q1.shape != valor_q2.shape:

        raise ValueError(
            "Q1 y Q2 deben producir tensores "
            "con la misma forma."
        )

    valor_q_minimo = torch.minimum(
        valor_q1,
        valor_q2,
    )

    if not torch.isfinite(
        valor_q_minimo
    ).all():

        raise ValueError(
            "El mínimo entre Q1 y Q2 contiene "
            "NaN o infinitos."
        )

    return {
        "resultados_q1": resultados_q1,
        "resultados_q2": resultados_q2,
        "valor_q1": valor_q1,
        "valor_q2": valor_q2,
        "valor_q_minimo": valor_q_minimo,
    }
def calcular_desviacion_actor_sac(
    log_desviacion,
):

    # ------------------------------------------------------
    # 1. Verificar que la entrada sea un tensor
    # ------------------------------------------------------

    if not torch.is_tensor(
        log_desviacion
    ):

        raise TypeError(
            "El logaritmo de la desviación debe ser "
            "un tensor de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar sus dimensiones
    # ------------------------------------------------------

    if log_desviacion.ndim != 2:

        raise ValueError(
            "El logaritmo de la desviación debe tener "
            "forma (lote, acciones)."
        )

    if log_desviacion.shape[1] != (
        DIMENSION_ACCION_SAC
    ):

        raise ValueError(
            "El logaritmo de la desviación tiene "
            f"{log_desviacion.shape[1]} valores por elemento; "
            f"se esperaban {DIMENSION_ACCION_SAC}."
        )

    # ------------------------------------------------------
    # 3. Verificar valores finitos
    # ------------------------------------------------------

    if not torch.isfinite(
        log_desviacion
    ).all():

        raise ValueError(
            "El logaritmo de la desviación contiene "
            "NaN o valores infinitos."
        )

    # ------------------------------------------------------
    # 4. Limitar el logaritmo
    # ------------------------------------------------------

    log_desviacion_limitado = torch.clamp(
        log_desviacion,
        min=LOG_DESVIACION_MINIMA_SAC,
        max=LOG_DESVIACION_MAXIMA_SAC,
    )

    # ------------------------------------------------------
    # 5. Convertir a desviación estándar
    # ------------------------------------------------------
    #
    # La función exponencial garantiza:
    #
    # desviacion > 0

    desviacion = torch.exp(
        log_desviacion_limitado
    )

    # ------------------------------------------------------
    # 6. Verificar el resultado
    # ------------------------------------------------------

    if desviacion.shape != (
        log_desviacion.shape
    ):

        raise ValueError(
            "La desviación calculada no conserva "
            "la forma del tensor original."
        )

    if not torch.isfinite(
        desviacion
    ).all():

        raise ValueError(
            "La desviación calculada contiene "
            "NaN o valores infinitos."
        )

    if not torch.all(
        desviacion > 0.0
    ):

        raise ValueError(
            "Todos los valores de desviación deben "
            "ser estrictamente positivos."
        )

    return (
        log_desviacion_limitado,
        desviacion,
    )
def muestrear_accion_gaussiana_sac(
    media,
    desviacion,
):

    # ------------------------------------------------------
    # 1. Verificar que las entradas sean tensores
    # ------------------------------------------------------

    if not torch.is_tensor(
        media
    ):

        raise TypeError(
            "La media debe ser un tensor de PyTorch."
        )

    if not torch.is_tensor(
        desviacion
    ):

        raise TypeError(
            "La desviación debe ser un tensor de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar dimensiones
    # ------------------------------------------------------

    if media.ndim != 2:

        raise ValueError(
            "La media debe tener forma "
            "(lote, acciones)."
        )

    if desviacion.ndim != 2:

        raise ValueError(
            "La desviación debe tener forma "
            "(lote, acciones)."
        )

    if media.shape != desviacion.shape:

        raise ValueError(
            "La media y la desviación deben tener "
            "exactamente la misma forma."
        )

    if media.shape[1] != (
        DIMENSION_ACCION_SAC
    ):

        raise ValueError(
            "La media debe contener "
            f"{DIMENSION_ACCION_SAC} acciones "
            "por elemento del lote."
        )

    # ------------------------------------------------------
    # 3. Verificar dispositivo y tipo
    # ------------------------------------------------------

    if media.device != desviacion.device:

        raise ValueError(
            "La media y la desviación deben estar "
            "en el mismo dispositivo."
        )

    if media.dtype != desviacion.dtype:

        raise TypeError(
            "La media y la desviación deben tener "
            "el mismo tipo de datos."
        )

    # ------------------------------------------------------
    # 4. Verificar valores
    # ------------------------------------------------------

    if not torch.isfinite(
        media
    ).all():

        raise ValueError(
            "La media contiene NaN o valores infinitos."
        )

    if not torch.isfinite(
        desviacion
    ).all():

        raise ValueError(
            "La desviación contiene NaN "
            "o valores infinitos."
        )

    if not torch.all(
        desviacion > 0.0
    ):

        raise ValueError(
            "La desviación debe ser estrictamente positiva."
        )

    # ------------------------------------------------------
    # 5. Construir la distribución normal
    # ------------------------------------------------------

    distribucion = torch.distributions.Normal(
        loc=media,
        scale=desviacion,
    )

    # ------------------------------------------------------
    # 6. Muestrear mediante reparametrización
    # ------------------------------------------------------
    #
    # Internamente:
    #
    # accion = media + desviacion * ruido
    #
    # donde:
    #
    # ruido ~ N(0, 1)

    accion_pre_tanh = distribucion.rsample()

    # ------------------------------------------------------
    # 7. Verificar la muestra
    # ------------------------------------------------------

    if accion_pre_tanh.shape != media.shape:

        raise ValueError(
            "La acción muestreada no conserva "
            "la forma de la media."
        )

    if not torch.isfinite(
        accion_pre_tanh
    ).all():

        raise ValueError(
            "La acción muestreada contiene "
            "NaN o valores infinitos."
        )

    return (
        distribucion,
        accion_pre_tanh,
    )
def aplicar_tanh_accion_sac(
    accion_pre_tanh,
):

    # ------------------------------------------------------
    # 1. Verificar que la entrada sea un tensor
    # ------------------------------------------------------

    if not torch.is_tensor(
        accion_pre_tanh
    ):

        raise TypeError(
            "La acción previa a tanh debe ser "
            "un tensor de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar dimensiones
    # ------------------------------------------------------

    if accion_pre_tanh.ndim != 2:

        raise ValueError(
            "La acción previa a tanh debe tener forma "
            "(lote, acciones)."
        )

    if accion_pre_tanh.shape[1] != (
        DIMENSION_ACCION_SAC
    ):

        raise ValueError(
            "La acción previa a tanh contiene "
            f"{accion_pre_tanh.shape[1]} valores; "
            f"se esperaban {DIMENSION_ACCION_SAC}."
        )

    # ------------------------------------------------------
    # 3. Verificar valores finitos
    # ------------------------------------------------------

    if not torch.isfinite(
        accion_pre_tanh
    ).all():

        raise ValueError(
            "La acción previa a tanh contiene "
            "NaN o valores infinitos."
        )

    # ------------------------------------------------------
    # 4. Aplicar la saturación hiperbólica
    # ------------------------------------------------------
    #
    # Convierte cualquier número real al intervalo:
    #
    # -1 < accion < 1

    accion_normalizada = torch.tanh(
        accion_pre_tanh
    )

    # ------------------------------------------------------
    # 5. Verificar la salida
    # ------------------------------------------------------

    if accion_normalizada.shape != (
        accion_pre_tanh.shape
    ):

        raise ValueError(
            "La acción normalizada no conserva "
            "la forma de la acción original."
        )

    if not torch.isfinite(
        accion_normalizada
    ).all():

        raise ValueError(
            "La acción normalizada contiene "
            "NaN o valores infinitos."
        )

    if not torch.all(
        accion_normalizada >= -1.0
    ):

        raise ValueError(
            "La acción normalizada contiene valores "
            "menores que -1."
        )

    if not torch.all(
        accion_normalizada <= 1.0
    ):

        raise ValueError(
            "La acción normalizada contiene valores "
            "mayores que 1."
        )

    return accion_normalizada
def calcular_log_probabilidad_accion_sac(
    distribucion,
    accion_pre_tanh,
    accion_normalizada,
):

    # ------------------------------------------------------
    # 1. Verificar la distribución
    # ------------------------------------------------------

    if not isinstance(
        distribucion,
        torch.distributions.Normal,
    ):

        raise TypeError(
            "La distribución debe ser una distribución "
            "Normal de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar que las acciones sean tensores
    # ------------------------------------------------------

    if not torch.is_tensor(
        accion_pre_tanh
    ):

        raise TypeError(
            "La acción previa a tanh debe ser "
            "un tensor de PyTorch."
        )

    if not torch.is_tensor(
        accion_normalizada
    ):

        raise TypeError(
            "La acción normalizada debe ser "
            "un tensor de PyTorch."
        )

    # ------------------------------------------------------
    # 3. Verificar dimensiones
    # ------------------------------------------------------

    if accion_pre_tanh.ndim != 2:

        raise ValueError(
            "La acción previa a tanh debe tener forma "
            "(lote, acciones)."
        )

    if accion_normalizada.ndim != 2:

        raise ValueError(
            "La acción normalizada debe tener forma "
            "(lote, acciones)."
        )

    if (
        accion_pre_tanh.shape
        != accion_normalizada.shape
    ):

        raise ValueError(
            "La acción previa y la acción normalizada "
            "deben tener la misma forma."
        )

    if accion_pre_tanh.shape[1] != (
        DIMENSION_ACCION_SAC
    ):

        raise ValueError(
            "La acción debe contener "
            f"{DIMENSION_ACCION_SAC} componentes."
        )

    # ------------------------------------------------------
    # 4. Verificar dispositivo y tipo de datos
    # ------------------------------------------------------

    if (
        accion_pre_tanh.device
        != accion_normalizada.device
    ):

        raise ValueError(
            "Las acciones deben estar en el mismo "
            "dispositivo."
        )

    if (
        accion_pre_tanh.dtype
        != accion_normalizada.dtype
    ):

        raise TypeError(
            "Las acciones deben tener el mismo "
            "tipo de datos."
        )

    # ------------------------------------------------------
    # 5. Verificar valores finitos
    # ------------------------------------------------------

    if not torch.isfinite(
        accion_pre_tanh
    ).all():

        raise ValueError(
            "La acción previa a tanh contiene "
            "NaN o valores infinitos."
        )

    if not torch.isfinite(
        accion_normalizada
    ).all():

        raise ValueError(
            "La acción normalizada contiene "
            "NaN o valores infinitos."
        )

    # ------------------------------------------------------
    # 6. Log-probabilidad gaussiana
    # ------------------------------------------------------
    #
    # distribucion.log_prob() devuelve una probabilidad
    # independiente para cada componente de la acción:
    #
    # (lote, 2)

    log_probabilidad_por_accion = (
        distribucion.log_prob(
            accion_pre_tanh
        )
    )

    # Las dos acciones forman una sola decisión conjunta.
    # Por eso se suman sus log-probabilidades:
    #
    # (lote, 2) -> (lote, 1)

    log_probabilidad_normal = (
        log_probabilidad_por_accion.sum(
            dim=1,
            keepdim=True,
        )
    )

    # ------------------------------------------------------
    # 7. Corrección por la transformación tanh
    # ------------------------------------------------------
    #
    # Derivada de tanh:
    #
    # 1 - tanh(x)^2
    #
    # Como accion_normalizada = tanh(accion_pre_tanh):
    #
    # jacobiano = 1 - accion_normalizada^2

    jacobiano_tanh = (
        1.0
        - accion_normalizada.pow(
            2
        )
    )

    # Evitar valores iguales a cero.

    jacobiano_tanh = torch.clamp(
        jacobiano_tanh,
        min=EPSILON_LOG_PROBABILIDAD_SAC,
    )

    # La corrección también se suma sobre las dos acciones.

    correccion_tanh = torch.log(
        jacobiano_tanh
    ).sum(
        dim=1,
        keepdim=True,
    )

    # ------------------------------------------------------
    # 8. Log-probabilidad corregida
    # ------------------------------------------------------

    log_probabilidad_corregida = (
        log_probabilidad_normal
        - correccion_tanh
    )

    # ------------------------------------------------------
    # 9. Verificar las salidas
    # ------------------------------------------------------

    numero_elementos_lote = (
        accion_pre_tanh.shape[
            0
        ]
    )

    forma_esperada = (
        numero_elementos_lote,
        1,
    )

    if tuple(
        log_probabilidad_normal.shape
    ) != forma_esperada:

        raise ValueError(
            "La log-probabilidad normal tiene forma "
            f"{tuple(log_probabilidad_normal.shape)}; "
            f"se esperaba {forma_esperada}."
        )

    if tuple(
        correccion_tanh.shape
    ) != forma_esperada:

        raise ValueError(
            "La corrección tanh tiene forma "
            f"{tuple(correccion_tanh.shape)}; "
            f"se esperaba {forma_esperada}."
        )

    if tuple(
        log_probabilidad_corregida.shape
    ) != forma_esperada:

        raise ValueError(
            "La log-probabilidad corregida tiene forma "
            f"{tuple(log_probabilidad_corregida.shape)}; "
            f"se esperaba {forma_esperada}."
        )

    if not torch.isfinite(
        log_probabilidad_normal
    ).all():

        raise ValueError(
            "La log-probabilidad normal contiene "
            "NaN o valores infinitos."
        )

    if not torch.isfinite(
        correccion_tanh
    ).all():

        raise ValueError(
            "La corrección tanh contiene "
            "NaN o valores infinitos."
        )

    if not torch.isfinite(
        log_probabilidad_corregida
    ).all():

        raise ValueError(
            "La log-probabilidad corregida contiene "
            "NaN o valores infinitos."
        )

    return (
        log_probabilidad_normal,
        correccion_tanh,
        log_probabilidad_corregida,
    )
def ejecutar_actor_sac(
    tensor_parche,
    tensor_escalares,
    rama_cnn,
    rama_mlp,
    red_compartida,
    cabeza_media,
    cabeza_log_desviacion,
):

    # ------------------------------------------------------
    # 1. Verificar las entradas
    # ------------------------------------------------------

    if not torch.is_tensor(
        tensor_parche
    ):

        raise TypeError(
            "El parche debe ser un tensor de PyTorch."
        )

    if not torch.is_tensor(
        tensor_escalares
    ):

        raise TypeError(
            "Los escalares deben ser un tensor de PyTorch."
        )

    if tensor_parche.ndim != 4:

        raise ValueError(
            "El parche debe tener forma "
            "(lote, canales, alto, ancho)."
        )

    if tensor_escalares.ndim != 2:

        raise ValueError(
            "Los escalares deben tener forma "
            "(lote, escalares)."
        )

    if tensor_parche.shape[0] != (
        tensor_escalares.shape[0]
    ):

        raise ValueError(
            "El parche y los escalares deben tener "
            "el mismo tamaño de lote."
        )

    if tensor_parche.device != (
        tensor_escalares.device
    ):

        raise ValueError(
            "El parche y los escalares deben estar "
            "en el mismo dispositivo."
        )

    # ------------------------------------------------------
    # 2. Procesar el parche con la CNN
    # ------------------------------------------------------

    caracteristicas_cnn = rama_cnn(
        tensor_parche
    )

    # ------------------------------------------------------
    # 3. Procesar los escalares con la MLP
    # ------------------------------------------------------

    caracteristicas_mlp = rama_mlp(
        tensor_escalares
    )

    # ------------------------------------------------------
    # 4. Fusionar ambas representaciones
    # ------------------------------------------------------

    caracteristicas_fusionadas = (
        fusionar_caracteristicas_sac(
            caracteristicas_cnn=(
                caracteristicas_cnn
            ),

            caracteristicas_mlp=(
                caracteristicas_mlp
            ),
        )
    )

    # ------------------------------------------------------
    # 5. Red compartida del actor
    # ------------------------------------------------------

    caracteristicas_compartidas = (
        red_compartida(
            caracteristicas_fusionadas
        )
    )

    # ------------------------------------------------------
    # 6. Cabezas de media y log-desviación
    # ------------------------------------------------------

    media = cabeza_media(
        caracteristicas_compartidas
    )

    log_desviacion = (
        cabeza_log_desviacion(
            caracteristicas_compartidas
        )
    )

    # ------------------------------------------------------
    # 7. Calcular la desviación estándar
    # ------------------------------------------------------

    (
        log_desviacion_limitado,
        desviacion,
    ) = calcular_desviacion_actor_sac(
        log_desviacion=log_desviacion
    )

    # ------------------------------------------------------
    # 8. Crear la distribución y muestrear
    # ------------------------------------------------------

    (
        distribucion,
        accion_pre_tanh,
    ) = muestrear_accion_gaussiana_sac(
        media=media,
        desviacion=desviacion,
    )

    # ------------------------------------------------------
    # 9. Aplicar tanh
    # ------------------------------------------------------

    accion_normalizada = (
        aplicar_tanh_accion_sac(
            accion_pre_tanh=accion_pre_tanh
        )
    )

    # ------------------------------------------------------
    # 10. Calcular la log-probabilidad corregida
    # ------------------------------------------------------

    (
        log_probabilidad_normal,
        correccion_tanh,
        log_probabilidad_corregida,
    ) = calcular_log_probabilidad_accion_sac(
        distribucion=distribucion,

        accion_pre_tanh=(
            accion_pre_tanh
        ),

        accion_normalizada=(
            accion_normalizada
        ),
    )

    # ------------------------------------------------------
    # 11. Acción determinista para evaluación
    # ------------------------------------------------------
    #
    # Durante entrenamiento se utiliza la acción muestreada.
    #
    # Durante evaluación puede utilizarse:
    #
    # accion_determinista = tanh(media)

    accion_determinista = torch.tanh(
        media
    )

    # ------------------------------------------------------
    # 12. Verificaciones finales
    # ------------------------------------------------------

    numero_elementos_lote = (
        tensor_parche.shape[0]
    )

    forma_accion_esperada = (
        numero_elementos_lote,
        DIMENSION_ACCION_SAC,
    )

    if tuple(
        accion_normalizada.shape
    ) != forma_accion_esperada:

        raise ValueError(
            "La acción normalizada tiene forma "
            f"{tuple(accion_normalizada.shape)}; "
            f"se esperaba {forma_accion_esperada}."
        )

    if tuple(
        accion_determinista.shape
    ) != forma_accion_esperada:

        raise ValueError(
            "La acción determinista tiene forma "
            f"{tuple(accion_determinista.shape)}; "
            f"se esperaba {forma_accion_esperada}."
        )

    if tuple(
        log_probabilidad_corregida.shape
    ) != (
        numero_elementos_lote,
        1,
    ):

        raise ValueError(
            "La log-probabilidad corregida debe tener "
            "forma (lote, 1)."
        )

    if not torch.isfinite(
        accion_normalizada
    ).all():

        raise ValueError(
            "La acción normalizada contiene "
            "NaN o valores infinitos."
        )

    if not torch.isfinite(
        log_probabilidad_corregida
    ).all():

        raise ValueError(
            "La log-probabilidad corregida contiene "
            "NaN o valores infinitos."
        )

    # ------------------------------------------------------
    # 13. Reunir los resultados
    # ------------------------------------------------------
    #
    # No usamos detach() porque durante el entrenamiento
    # necesitamos conservar el grafo de gradientes.

    resultados = {
        "caracteristicas_cnn": (
            caracteristicas_cnn
        ),

        "caracteristicas_mlp": (
            caracteristicas_mlp
        ),

        "caracteristicas_fusionadas": (
            caracteristicas_fusionadas
        ),

        "caracteristicas_compartidas": (
            caracteristicas_compartidas
        ),

        "media": media,

        "log_desviacion": (
            log_desviacion
        ),

        "log_desviacion_limitado": (
            log_desviacion_limitado
        ),

        "desviacion": desviacion,

        "distribucion": distribucion,

        "accion_pre_tanh": (
            accion_pre_tanh
        ),

        "accion_normalizada": (
            accion_normalizada
        ),

        "accion_determinista": (
            accion_determinista
        ),

        "log_probabilidad_normal": (
            log_probabilidad_normal
        ),

        "correccion_tanh": (
            correccion_tanh
        ),

        "log_probabilidad_corregida": (
            log_probabilidad_corregida
        ),
    }

    return resultados
def seleccionar_accion_actor_sac(
    observacion,
    actor,
    dispositivo=DISPOSITIVO_SAC,
    determinista=False,
):
    # ------------------------------------------------------
    # 1. Verificar el actor
    # ------------------------------------------------------

    if not isinstance(
        actor,
        nn.ModuleDict,
    ):

        raise TypeError(
            "El actor SAC debe ser un nn.ModuleDict."
        )

    claves_necesarias = {
        "rama_cnn",
        "rama_mlp",
        "red_compartida",
        "cabeza_media",
        "cabeza_log_desviacion",
    }

    if set(
        actor.keys()
    ) != claves_necesarias:

        raise KeyError(
            "El actor SAC no contiene todos los módulos "
            "necesarios."
        )

    # ------------------------------------------------------
    # 2. Verificar el modo solicitado
    # ------------------------------------------------------

    if not isinstance(
        determinista,
        (
            bool,
            np.bool_,
        ),
    ):

        raise TypeError(
            "El indicador determinista debe ser booleano."
        )

    determinista = bool(
        determinista
    )

    dispositivo = torch.device(
        dispositivo
    )

    # ------------------------------------------------------
    # 3. Verificar el dispositivo del actor
    # ------------------------------------------------------

    dispositivos_actor = {
        parametro.device
        for parametro in actor.parameters()
    }

    if len(
        dispositivos_actor
    ) != 1:

        raise RuntimeError(
            "Todos los parámetros del actor deben estar "
            "en el mismo dispositivo."
        )

    dispositivo_actor = next(
        iter(
            dispositivos_actor
        )
    )

    if dispositivo_actor != dispositivo:

        raise ValueError(
            "El actor está en "
            f"{dispositivo_actor}, pero se solicitó "
            f"ejecutarlo en {dispositivo}."
        )

    # ------------------------------------------------------
    # 4. Convertir la observación a tensores
    # ------------------------------------------------------

    (
        tensor_parche,
        tensor_escalares,
    ) = convertir_observacion_sac_a_tensores(
        observacion=observacion,
        dispositivo=dispositivo,
    )

    # ------------------------------------------------------
    # 5. Guardar el modo anterior del actor
    # ------------------------------------------------------
    #
    # La función no debe cambiar permanentemente el modo
    # train/eval que tenga el actor.

    modo_entrenamiento_anterior = (
        actor.training
    )

    try:

        # --------------------------------------------------
        # 6. Seleccionar el modo de ejecución
        # --------------------------------------------------

        if determinista:

            actor.eval()

        else:

            actor.train()

        # --------------------------------------------------
        # 7. Ejecutar el actor sin construir gradientes
        # --------------------------------------------------
        #
        # La selección de acciones para interactuar con el
        # entorno no necesita backward().
        #
        # Los gradientes se construyen posteriormente al
        # entrenar lotes obtenidos del replay buffer.

        with torch.no_grad():

            resultados_actor = ejecutar_actor_sac(
                tensor_parche=tensor_parche,

                tensor_escalares=tensor_escalares,

                rama_cnn=actor[
                    "rama_cnn"
                ],

                rama_mlp=actor[
                    "rama_mlp"
                ],

                red_compartida=actor[
                    "red_compartida"
                ],

                cabeza_media=actor[
                    "cabeza_media"
                ],

                cabeza_log_desviacion=actor[
                    "cabeza_log_desviacion"
                ],
            )

        # --------------------------------------------------
        # 8. Elegir acción estocástica o determinista
        # --------------------------------------------------

        if determinista:

            tensor_accion = resultados_actor[
                "accion_determinista"
            ]

            modo_accion = "determinista"

        else:

            tensor_accion = resultados_actor[
                "accion_normalizada"
            ]

            modo_accion = "estocastica"

        # --------------------------------------------------
        # 9. Convertir la acción a NumPy
        # --------------------------------------------------

        accion = (
            tensor_accion[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        media = (
            resultados_actor[
                "media"
            ][
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        desviacion = (
            resultados_actor[
                "desviacion"
            ][
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if determinista:

            log_probabilidad = None

        else:

            log_probabilidad = float(
                resultados_actor[
                    "log_probabilidad_corregida"
                ][
                    0,
                    0
                ]
                .detach()
                .cpu()
                .item()
            )

    finally:

        # --------------------------------------------------
        # 10. Restaurar el modo anterior
        # --------------------------------------------------

        actor.train(
            modo_entrenamiento_anterior
        )

    # ------------------------------------------------------
    # 11. Verificar la acción resultante
    # ------------------------------------------------------

    if accion.shape != (
        DIMENSION_ACCION_SAC,
    ):

        raise ValueError(
            "La acción seleccionada tiene forma "
            f"{accion.shape}; se esperaba "
            f"({DIMENSION_ACCION_SAC},)."
        )

    if accion.dtype != np.float32:

        raise TypeError(
            "La acción seleccionada debe tener tipo "
            "np.float32."
        )

    if not np.all(
        np.isfinite(
            accion
        )
    ):

        raise ValueError(
            "La acción seleccionada contiene NaN "
            "o valores infinitos."
        )

    tolerancia = 1e-6

    if np.any(
        accion < -1.0 - tolerancia
    ):

        raise ValueError(
            "La acción contiene valores menores que -1."
        )

    if np.any(
        accion > 1.0 + tolerancia
    ):

        raise ValueError(
            "La acción contiene valores mayores que 1."
        )

    accion = np.clip(
        accion,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )

    # ------------------------------------------------------
    # 12. Convertir a control físico para diagnóstico
    # ------------------------------------------------------

    (
        velocidad_lineal,
        velocidad_angular,
    ) = convertir_accion_sac_a_control(
        accion=accion
    )

    informacion = {
        "modo": modo_accion,

        "determinista": determinista,

        "accion": accion.copy(),

        "media": media,

        "desviacion": desviacion,

        "log_probabilidad": (
            log_probabilidad
        ),

        "velocidad_lineal": (
            velocidad_lineal
        ),

        "velocidad_angular": (
            velocidad_angular
        ),

        "dispositivo": str(
            dispositivo
        ),
    }

    return (
        accion,
        informacion,
    )
def construir_observacion_local(
    estado_robot,
    submeta,
    meta,
    obstaculos_dinamicos,
    velocidad_lineal_actual,
    velocidad_angular_actual,
    indice_progreso,
    numero_puntos_camino,
):

    submeta_local = transformar_punto_al_marco_robot(
        estado_robot,
        submeta,
    )

    distancia_meta = distancia_entre_puntos(
        (
            estado_robot[0],
            estado_robot[1],
        ),
        meta,
    )

    diagonal_mapa = math.hypot(
        ANCHO_MAPA,
        ALTO_MAPA,
    )

    if numero_puntos_camino > 1:

        progreso_normalizado = (
            indice_progreso
            / (
                numero_puntos_camino - 1
            )
        )

    else:

        progreso_normalizado = 1.0

    observados = obtener_obstaculos_dinamicos_observados(
        estado_robot,
        obstaculos_dinamicos,
    )

    escala_submeta = max(
        DISTANCIA_SUBMETA_LOCAL,
        1e-8,
    )

    observacion = [
        float(
            np.clip(
                submeta_local[0]
                / escala_submeta,
                -1.0,
                1.0,
            )
        ),
        float(
            np.clip(
                submeta_local[1]
                / escala_submeta,
                -1.0,
                1.0,
            )
        ),
        float(
            np.clip(
                distancia_meta
                / diagonal_mapa,
                0.0,
                1.0,
            )
        ),
        float(
            np.clip(
                velocidad_lineal_actual
                / VELOCIDAD_MAXIMA,
                -1.0,
                1.0,
            )
        ),
        float(
            np.clip(
                velocidad_angular_actual
                / VELOCIDAD_ANGULAR_MAXIMA,
                -1.0,
                1.0,
            )
        ),
        float(
            np.clip(
                progreso_normalizado,
                0.0,
                1.0,
            )
        ),
    ]

    for descripcion in observados:

        observacion.extend(
            [
                1.0,
                float(
                    np.clip(
                        descripcion["x_relativo"]
                        / RADIO_PERCEPCION_DINAMICA,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        descripcion["y_relativo"]
                        / RADIO_PERCEPCION_DINAMICA,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        descripcion["vx_relativo"]
                        / VELOCIDAD_MAX_OBSTACULO_DINAMICO,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        descripcion["vy_relativo"]
                        / VELOCIDAD_MAX_OBSTACULO_DINAMICO,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        descripcion["radio"]
                        / RADIO_MAX_OBSTACULO_DINAMICO,
                        0.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        descripcion["clearance"]
                        / RADIO_PERCEPCION_DINAMICA,
                        -1.0,
                        1.0,
                    )
                ),
            ]
        )

    numero_faltantes = (
        MAX_OBSTACULOS_DINAMICOS_OBSERVADOS
        - len(observados)
    )

    for indice in range(
        numero_faltantes
    ):

        observacion.extend(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    observacion = np.array(
        observacion,
        dtype=np.float32,
    )

    return (
        observacion,
        observados,
    )
def controlador_submeta_local(
    estado_robot,
    submeta,
    meta,
):

    diferencia_x = (
        submeta[0]
        - estado_robot[0]
    )

    diferencia_y = (
        submeta[1]
        - estado_robot[1]
    )

    angulo_deseado = math.atan2(
        diferencia_y,
        diferencia_x,
    )

    error_angular = normalizar_angulo(
        angulo_deseado
        - estado_robot[2]
    )

    velocidad_angular = (
        GANANCIA_ANGULAR
        * error_angular
    )

    factor_alineacion = max(
        0.0,
        math.cos(
            error_angular
        ),
    )

    velocidad_lineal = (
        VELOCIDAD_NOMINAL
        * (
            0.20
            + 0.80
            * factor_alineacion
        )
    )

    distancia_meta = distancia_entre_puntos(
        (
            estado_robot[0],
            estado_robot[1],
        ),
        meta,
    )

    if (
        distancia_meta
        < DISTANCIA_REDUCCION_VELOCIDAD
    ):

        velocidad_lineal *= (
            distancia_meta
            / DISTANCIA_REDUCCION_VELOCIDAD
        )

    velocidad_lineal, velocidad_angular = limitar_control(
        velocidad_lineal,
        velocidad_angular,
    )

    informacion = {
        "angulo_deseado": angulo_deseado,
        "error_angular": error_angular,
        "distancia_meta": distancia_meta,
    }

    return (
        velocidad_lineal,
        velocidad_angular,
        informacion,
    )
def dibujar_historial_submetas(
    ax,
    historial_submetas,
):

    if len(historial_submetas) == 0:
        return

    coordenadas_x = []

    coordenadas_y = []

    for indice in range(
        0,
        len(historial_submetas),
        10,
    ):

        submeta = historial_submetas[
            indice
        ]

        coordenadas_x.append(
            submeta[0]
        )

        coordenadas_y.append(
            submeta[1]
        )

    ax.scatter(
        coordenadas_x,
        coordenadas_y,
        marker="*",
        s=70,
        label="Submetas locales",
        zorder=9,
    )
def dibujar_percepcion_dinamica(
    ax,
    estado_robot,
    submeta,
    obstaculos_observados,
):

    circulo_percepcion = Circle(
        (
            estado_robot[0],
            estado_robot[1],
        ),
        RADIO_PERCEPCION_DINAMICA,
        facecolor="none",
        edgecolor="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Radio de percepción",
        zorder=4,
    )

    ax.add_patch(
        circulo_percepcion
    )

    ax.scatter(
        submeta[0],
        submeta[1],
        marker="*",
        s=180,
        edgecolor="black",
        label="Submeta local final",
        zorder=11,
    )

    primera_linea = True

    for obstaculo in obstaculos_observados:

        etiqueta = None

        if primera_linea:

            etiqueta = "Obstáculos percibidos"

            primera_linea = False

        ax.plot(
            [
                estado_robot[0],
                obstaculo["x_mundo"],
            ],
            [
                estado_robot[1],
                obstaculo["y_mundo"],
            ],
            linestyle=":",
            linewidth=1.5,
            label=etiqueta,
            zorder=7,
        )
def obtener_obstaculos_dinamicos_visibles(
    estado_robot,
    obstaculos_dinamicos,
    radio_percepcion=RADIO_PERCEPCION_DINAMICA,
    maximo_obstaculos=MAX_OBSTACULOS_DINAMICOS_OBSERVADOS,
):

    visibles = []

    for obstaculo in obstaculos_dinamicos:

        distancia_centros = distancia_entre_puntos(
            (
                estado_robot[0],
                estado_robot[1],
            ),
            (
                obstaculo["x"],
                obstaculo["y"],
            ),
        )

        visible = (
            distancia_centros
            <= radio_percepcion
            + obstaculo["radio"]
        )

        if visible:

            clearance = (
                distancia_centros
                - RADIO_ROBOT
                - obstaculo["radio"]
            )

            visibles.append(
                (
                    clearance,
                    obstaculo.copy(),
                )
            )

    visibles.sort(
        key=lambda elemento: elemento[0]
    )

    obstaculos_ordenados = [
        elemento[1]
        for elemento in visibles[
            :maximo_obstaculos
        ]
    ]

    return obstaculos_ordenados
def generar_ventana_dinamica_dwa(
    velocidad_actual,
    omega_actual,
    dt=DT,
):

    cambio_maximo_v = (
        ACELERACION_LINEAL_MAXIMA_DWA
        * dt
    )

    cambio_maximo_omega = (
        ACELERACION_ANGULAR_MAXIMA_DWA
        * dt
    )

    velocidad_minima = max(
        VELOCIDAD_MINIMA_DWA,
        velocidad_actual
        - cambio_maximo_v,
    )

    velocidad_maxima = min(
        VELOCIDAD_MAXIMA,
        velocidad_actual
        + cambio_maximo_v,
    )

    omega_minima = max(
        -VELOCIDAD_ANGULAR_MAXIMA,
        omega_actual
        - cambio_maximo_omega,
    )

    omega_maxima = min(
        VELOCIDAD_ANGULAR_MAXIMA,
        omega_actual
        + cambio_maximo_omega,
    )

    ventana = {
        "velocidad_minima": velocidad_minima,
        "velocidad_maxima": velocidad_maxima,
        "omega_minima": omega_minima,
        "omega_maxima": omega_maxima,
    }

    return ventana
def generar_controles_candidatos_dwa(
    velocidad_actual,
    omega_actual,
    dt=DT,
):

    ventana = generar_ventana_dinamica_dwa(
        velocidad_actual,
        omega_actual,
        dt,
    )

    velocidades = np.linspace(
        ventana["velocidad_minima"],
        ventana["velocidad_maxima"],
        NUMERO_MUESTRAS_V_DWA,
    )

    omegas = np.linspace(
        ventana["omega_minima"],
        ventana["omega_maxima"],
        NUMERO_MUESTRAS_OMEGA_DWA,
    )

    # Incluir giro nulo cuando pertenece a la ventana.
    if (
        ventana["omega_minima"]
        <= 0.0
        <= ventana["omega_maxima"]
    ):

        omegas = np.append(
            omegas,
            0.0,
        )

    velocidades = np.unique(
        np.round(
            velocidades,
            decimals=10,
        )
    )

    omegas = np.unique(
        np.round(
            omegas,
            decimals=10,
        )
    )

    controles = []

    for velocidad in velocidades:

        for omega in omegas:

            controles.append(
                (
                    float(velocidad),
                    float(omega),
                )
            )

    return controles, ventana
def predecir_trayectoria_dwa(
    estado_inicial,
    velocidad_lineal,
    velocidad_angular,
    obstaculos_estaticos,
    obstaculos_dinamicos_visibles,
    tiempo_actual,
    horizonte=HORIZONTE_PREDICCION_DWA,
    dt=DT,
):

    numero_pasos = max(
        1,
        int(
            math.ceil(
                horizonte / dt
            )
        ),
    )

    estado_predicho = estado_inicial

    dinamicos_predichos = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos_visibles
    ]

    estados_predichos = [
        estado_inicial
    ]

    clearances_estaticos = []

    clearances_dinamicos = []

    trayectoria_valida = True

    tipo_colision = None

    for paso in range(
        numero_pasos
    ):

        tiempo_predicho = (
            tiempo_actual
            + (
                paso + 1
            ) * dt
        )

        # --------------------------------------------------
        # Predecir movimiento del robot
        # --------------------------------------------------

        estado_predicho = actualizar_estado_robot(
            estado_predicho,
            velocidad_lineal,
            velocidad_angular,
            dt,
        )

        # --------------------------------------------------
        # Predecir movimiento de obstáculos visibles
        # --------------------------------------------------

        dinamicos_predichos = actualizar_obstaculos_dinamicos(
            dinamicos_predichos,
            obstaculos_estaticos,
            tiempo_predicho,
            dt,
        )

        estados_predichos.append(
            estado_predicho
        )

        # --------------------------------------------------
        # Clearance estático
        # --------------------------------------------------

        clearance_estatico = calcular_clearance_estado(
            estado_predicho,
            obstaculos_estaticos,
        )

        clearances_estaticos.append(
            clearance_estatico
        )

        # --------------------------------------------------
        # Clearance dinámico
        # --------------------------------------------------

        clearance_dinamico = calcular_clearance_dinamico_estado(
            estado_predicho,
            dinamicos_predichos,
        )

        clearances_dinamicos.append(
            clearance_dinamico
        )

        # --------------------------------------------------
        # Rechazar trayectorias que no conserven el margen
        # de seguridad, aunque todavía no exista contacto.
        # --------------------------------------------------

        clearance_total_paso = min(
            clearance_estatico,
            clearance_dinamico,
        )

        if (
            clearance_total_paso
            < CLEARANCE_SEGURIDAD_DWA
        ):

            trayectoria_valida = False
            tipo_colision = "margen_seguridad"
            break

        # --------------------------------------------------
        # Comprobar obstáculos estáticos y paredes
        # --------------------------------------------------

        (
            colision_estatica,
            fuera_mapa,
        ) = evaluar_estado_robot(
            estado_predicho,
            obstaculos_estaticos,
        )

        if colision_estatica:

            trayectoria_valida = False
            tipo_colision = "estatica"
            break

        if fuera_mapa:

            trayectoria_valida = False
            tipo_colision = "fuera_mapa"
            break

        # --------------------------------------------------
        # Comprobar obstáculos dinámicos
        # --------------------------------------------------

        colision_dinamica = robot_colisiona_obstaculos_dinamicos(
            estado_predicho,
            dinamicos_predichos,
        )

        if colision_dinamica:

            trayectoria_valida = False
            tipo_colision = "dinamica"
            break

    if len(clearances_estaticos) > 0:

        clearance_estatico_minimo = min(
            clearances_estaticos
        )

    else:

        clearance_estatico_minimo = float(
            "inf"
        )

    if len(clearances_dinamicos) > 0:

        clearance_dinamico_minimo = min(
            clearances_dinamicos
        )

    else:

        clearance_dinamico_minimo = float(
            "inf"
        )

    clearance_total_minimo = min(
        clearance_estatico_minimo,
        clearance_dinamico_minimo,
    )

    prediccion = {
        "valida": trayectoria_valida,
        "tipo_colision": tipo_colision,
        "estados": estados_predichos,
        "estado_final": estados_predichos[-1],
        "clearance_estatico_minimo": clearance_estatico_minimo,
        "clearance_dinamico_minimo": clearance_dinamico_minimo,
        "clearance_total_minimo": clearance_total_minimo,
    }

    return prediccion
def evaluar_trayectoria_dwa(
    prediccion,
    estado_actual,
    submeta,
    camino_mundo,
    velocidad_candidata,
    omega_candidata,
    velocidad_actual,
    omega_actual,
):

    if not prediccion["valida"]:

        return (
            -float("inf"),
            None,
        )

    estados_predichos = prediccion[
        "estados"
    ]

    estado_final = prediccion[
        "estado_final"
    ]

    # ======================================================
    # PROGRESO HACIA LA SUBMETA
    # ======================================================

    distancia_inicial_submeta = distancia_entre_puntos(
        (
            estado_actual[0],
            estado_actual[1],
        ),
        submeta,
    )

    distancia_final_submeta = distancia_entre_puntos(
        (
            estado_final[0],
            estado_final[1],
        ),
        submeta,
    )

    progreso = (
        distancia_inicial_submeta
        - distancia_final_submeta
    )

    distancia_maxima_horizonte = max(
        VELOCIDAD_MAXIMA
        * HORIZONTE_PREDICCION_DWA,
        1e-8,
    )

    progreso_normalizado = float(
        np.clip(
            progreso
            / distancia_maxima_horizonte,
            -1.0,
            1.0,
        )
    )

    # ======================================================
    # ALINEACIÓN FINAL CON LA SUBMETA
    # ======================================================

    angulo_submeta = math.atan2(
        submeta[1]
        - estado_final[1],
        submeta[0]
        - estado_final[0],
    )

    error_angular_final = normalizar_angulo(
        angulo_submeta
        - estado_final[2]
    )

    alineacion = (
        1.0
        + math.cos(
            error_angular_final
        )
    ) / 2.0

    # ======================================================
    # CLEARANCE
    # ======================================================

    clearance_minimo = prediccion[
        "clearance_total_minimo"
    ]

    if math.isinf(
        clearance_minimo
    ):

        clearance_normalizado = 1.0

    else:

        clearance_normalizado = float(
            np.clip(
                clearance_minimo
                / CLEARANCE_OBJETIVO_DWA,
                0.0,
                1.0,
            )
        )

    # ======================================================
    # VELOCIDAD
    # ======================================================

    velocidad_normalizada = float(
        np.clip(
            velocidad_candidata
            / VELOCIDAD_MAXIMA,
            0.0,
            1.0,
        )
    )

    # ======================================================
    # PENALIZACIÓN DE ESTANCAMIENTO
    # ======================================================

    penalizacion_estancamiento = float(
        np.clip(
            1.0
            - velocidad_candidata
            / max(
                VELOCIDAD_ESTANCAMIENTO_DWA,
                1e-8,
            ),
            0.0,
            1.0,
        )
    )

    # ======================================================
    # DESVIACIÓN RESPECTO DE A*
    # ======================================================

    errores_ruta = []

    for estado in estados_predichos[
        1:
    ]:

        error = distancia_punto_camino(
            (
                estado[0],
                estado[1],
            ),
            camino_mundo,
        )

        errores_ruta.append(
            error
        )

    if len(errores_ruta) > 0:

        error_ruta_medio = float(
            np.mean(
                errores_ruta
            )
        )

    else:

        error_ruta_medio = 0.0

    error_ruta_normalizado = float(
        np.clip(
            error_ruta_medio
            / ERROR_RUTA_OBJETIVO_DWA,
            0.0,
            1.0,
        )
    )

    # ======================================================
    # SUAVIDAD DEL CAMBIO DE CONTROL
    # ======================================================

    cambio_v_normalizado = (
        abs(
            velocidad_candidata
            - velocidad_actual
        )
        / max(
            ACELERACION_LINEAL_MAXIMA_DWA
            * DT,
            1e-8,
        )
    )

    cambio_omega_normalizado = (
        abs(
            omega_candidata
            - omega_actual
        )
        / max(
            ACELERACION_ANGULAR_MAXIMA_DWA
            * DT,
            1e-8,
        )
    )

    penalizacion_suavidad = float(
        np.clip(
            (
                cambio_v_normalizado
                + cambio_omega_normalizado
            ) / 2.0,
            0.0,
            1.0,
        )
    )

    # ======================================================
    # FUNCIÓN OBJETIVO
    # ======================================================

    puntuacion = (
        PESO_DWA_PROGRESO
        * progreso_normalizado
        + PESO_DWA_ALINEACION
        * alineacion
        + PESO_DWA_CLEARANCE
        * clearance_normalizado
        + PESO_DWA_VELOCIDAD
        * velocidad_normalizada
        - PESO_DWA_RUTA
        * error_ruta_normalizado
        - PESO_DWA_SUAVIDAD
        * penalizacion_suavidad
        - PESO_DWA_ESTANCAMIENTO
        * penalizacion_estancamiento
    )

    componentes = {
        "progreso": progreso,
        "progreso_normalizado": progreso_normalizado,
        "alineacion": alineacion,
        "clearance_minimo": clearance_minimo,
        "clearance_normalizado": clearance_normalizado,
        "velocidad_normalizada": velocidad_normalizada,
        "penalizacion_estancamiento": penalizacion_estancamiento,
        "error_ruta_medio": error_ruta_medio,
        "error_ruta_normalizado": error_ruta_normalizado,
        "penalizacion_suavidad": penalizacion_suavidad,
        "distancia_final_submeta": distancia_final_submeta,
    }

    return puntuacion, componentes
def calcular_control_emergencia_dwa(
    velocidad_actual,
    omega_actual,
    dt=DT,
):

    reduccion_v = (
        ACELERACION_LINEAL_MAXIMA_DWA
        * dt
    )

    reduccion_omega = (
        ACELERACION_ANGULAR_MAXIMA_DWA
        * dt
    )

    velocidad_emergencia = max(
        0.0,
        velocidad_actual
        - reduccion_v,
    )

    if omega_actual > 0.0:

        omega_emergencia = max(
            0.0,
            omega_actual
            - reduccion_omega,
        )

    elif omega_actual < 0.0:

        omega_emergencia = min(
            0.0,
            omega_actual
            + reduccion_omega,
        )

    else:

        omega_emergencia = 0.0

    return (
        velocidad_emergencia,
        omega_emergencia,
    )
def controlador_dwa(
    estado_robot,
    velocidad_actual,
    omega_actual,
    submeta,
    camino_mundo,
    obstaculos_estaticos,
    obstaculos_dinamicos,
    tiempo_actual,
    dt=DT,
):

    dinamicos_visibles = obtener_obstaculos_dinamicos_visibles(
        estado_robot,
        obstaculos_dinamicos,
    )

    controles_candidatos, ventana = generar_controles_candidatos_dwa(
        velocidad_actual,
        omega_actual,
        dt,
    )

    mejor_puntuacion = -float(
        "inf"
    )

    mejor_control = None
    mejor_prediccion = None
    mejores_componentes = None

    candidatos_validos = 0
    candidatos_invalidos = 0

    for (
        velocidad_candidata,
        omega_candidata,
    ) in controles_candidatos:

        prediccion = predecir_trayectoria_dwa(
            estado_inicial=estado_robot,
            velocidad_lineal=velocidad_candidata,
            velocidad_angular=omega_candidata,
            obstaculos_estaticos=obstaculos_estaticos,
            obstaculos_dinamicos_visibles=dinamicos_visibles,
            tiempo_actual=tiempo_actual,
            horizonte=HORIZONTE_PREDICCION_DWA,
            dt=dt,
        )

        (
            puntuacion,
            componentes,
        ) = evaluar_trayectoria_dwa(
            prediccion=prediccion,
            estado_actual=estado_robot,
            submeta=submeta,
            camino_mundo=camino_mundo,
            velocidad_candidata=velocidad_candidata,
            omega_candidata=omega_candidata,
            velocidad_actual=velocidad_actual,
            omega_actual=omega_actual,
        )

        if not prediccion["valida"]:

            candidatos_invalidos += 1
            continue

        candidatos_validos += 1

        if puntuacion > mejor_puntuacion:

            mejor_puntuacion = puntuacion

            mejor_control = (
                velocidad_candidata,
                omega_candidata,
            )

            mejor_prediccion = prediccion

            mejores_componentes = componentes

    frenado_emergencia = False

    if mejor_control is None:

        frenado_emergencia = True

        mejor_control = calcular_control_emergencia_dwa(
            velocidad_actual,
            omega_actual,
            dt,
        )

        mejor_prediccion = predecir_trayectoria_dwa(
            estado_inicial=estado_robot,
            velocidad_lineal=mejor_control[0],
            velocidad_angular=mejor_control[1],
            obstaculos_estaticos=obstaculos_estaticos,
            obstaculos_dinamicos_visibles=dinamicos_visibles,
            tiempo_actual=tiempo_actual,
            horizonte=HORIZONTE_PREDICCION_DWA,
            dt=dt,
        )

        mejor_puntuacion = -float(
            "inf"
        )

        mejores_componentes = None

    informacion = {
        "puntuacion": mejor_puntuacion,
        "prediccion": mejor_prediccion,
        "componentes": mejores_componentes,
        "ventana": ventana,
        "candidatos_totales": len(
            controles_candidatos
        ),
        "candidatos_validos": candidatos_validos,
        "candidatos_invalidos": candidatos_invalidos,
        "dinamicos_visibles": len(
            dinamicos_visibles
        ),
        "frenado_emergencia": frenado_emergencia,
    }

    return (
        mejor_control[0],
        mejor_control[1],
        informacion,
    )
def simular_seguimiento_dinamico_dwa(
    estado_inicial,
    camino_mundo,
    meta,
    obstaculos_estaticos,
    obstaculos_dinamicos_iniciales,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
):

    estado_robot = estado_inicial

    obstaculos_dinamicos = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos_iniciales
    ]

    estados_robot = [
        estado_inicial
    ]

    controles = []

    puntos_objetivo = []

    errores_angulares = []

    historial_dinamicos = [
        [
            obstaculo.copy()
            for obstaculo in obstaculos_dinamicos
        ]
    ]

    clearances_dinamicos = [
        calcular_clearance_dinamico_estado(
            estado_inicial,
            obstaculos_dinamicos,
        )
    ]

    historial_submetas = []

    historial_indices_progreso = []

    historial_observaciones = []

    historial_trayectorias_predichas = []

    historial_puntuaciones = []

    historial_candidatos_validos = []

    historial_dinamicos_visibles = []

    historial_frenado_emergencia = []

    indice_progreso = 0

    velocidad_actual = 0.0
    omega_actual = 0.0

    resultado = "timeout"

    pasos_ejecutados = 0

    indice_obstaculo_colision = None

    for paso in range(
        pasos_maximos
    ):

        tiempo_actual = (
            paso * dt
        )

        if robot_llego_meta(
            estado_robot,
            meta,
        ):

            resultado = "meta"
            break

        # ==================================================
        # SUBMETA LOCAL
        # ==================================================

        (
            submeta,
            indice_progreso,
            indice_submeta,
        ) = seleccionar_submeta_local_estable(
            estado_robot,
            camino_mundo,
            indice_progreso,
        )

        # ==================================================
        # OBSERVACIÓN COMÚN CON SAC
        # ==================================================

        (
            observacion_local,
            obstaculos_observados,
        ) = construir_observacion_local(
            estado_robot=estado_robot,
            submeta=submeta,
            meta=meta,
            obstaculos_dinamicos=obstaculos_dinamicos,
            velocidad_lineal_actual=velocidad_actual,
            velocidad_angular_actual=omega_actual,
            indice_progreso=indice_progreso,
            numero_puntos_camino=len(
                camino_mundo
            ),
        )

        # ==================================================
        # CONTROL DWA
        # ==================================================

        (
            velocidad_lineal,
            velocidad_angular,
            informacion_dwa,
        ) = controlador_dwa(
            estado_robot=estado_robot,
            velocidad_actual=velocidad_actual,
            omega_actual=omega_actual,
            submeta=submeta,
            camino_mundo=camino_mundo,
            obstaculos_estaticos=obstaculos_estaticos,
            obstaculos_dinamicos=obstaculos_dinamicos,
            tiempo_actual=tiempo_actual,
            dt=dt,
        )

        # ==================================================
        # ACTUALIZAR ROBOT
        # ==================================================

        estado_nuevo = actualizar_estado_robot(
            estado_robot,
            velocidad_lineal,
            velocidad_angular,
            dt,
        )

        # ==================================================
        # ACTUALIZAR OBSTÁCULOS DINÁMICOS
        # ==================================================

        tiempo_siguiente = (
            paso + 1
        ) * dt

        dinamicos_nuevos = actualizar_obstaculos_dinamicos(
            obstaculos_dinamicos,
            obstaculos_estaticos,
            tiempo_siguiente,
            dt,
        )

        # ==================================================
        # GUARDAR INFORMACIÓN
        # ==================================================

        estados_robot.append(
            estado_nuevo
        )

        controles.append(
            (
                velocidad_lineal,
                velocidad_angular,
            )
        )

        puntos_objetivo.append(
            submeta
        )

        error_angular = normalizar_angulo(
            math.atan2(
                submeta[1]
                - estado_robot[1],
                submeta[0]
                - estado_robot[0],
            )
            - estado_robot[2]
        )

        errores_angulares.append(
            error_angular
        )

        historial_dinamicos.append(
            [
                obstaculo.copy()
                for obstaculo in dinamicos_nuevos
            ]
        )

        clearance_dinamico = calcular_clearance_dinamico_estado(
            estado_nuevo,
            dinamicos_nuevos,
        )

        clearances_dinamicos.append(
            clearance_dinamico
        )

        historial_submetas.append(
            submeta
        )

        historial_indices_progreso.append(
            indice_progreso
        )

        historial_observaciones.append(
            observacion_local
        )

        prediccion = informacion_dwa[
            "prediccion"
        ]

        if prediccion is not None:

            historial_trayectorias_predichas.append(
                prediccion["estados"]
            )

        else:

            historial_trayectorias_predichas.append(
                []
            )

        historial_puntuaciones.append(
            informacion_dwa[
                "puntuacion"
            ]
        )

        historial_candidatos_validos.append(
            informacion_dwa[
                "candidatos_validos"
            ]
        )

        historial_dinamicos_visibles.append(
            informacion_dwa[
                "dinamicos_visibles"
            ]
        )

        historial_frenado_emergencia.append(
            informacion_dwa[
                "frenado_emergencia"
            ]
        )

        pasos_ejecutados = paso + 1

        # ==================================================
        # COMPROBAR COLISIONES
        # ==================================================

        (
            colision_estatica,
            fuera_mapa,
        ) = evaluar_estado_robot(
            estado_nuevo,
            obstaculos_estaticos,
        )

        if colision_estatica:

            resultado = "colision_estatica"
            break

        if fuera_mapa:

            resultado = "fuera_mapa"
            break

        if robot_colisiona_obstaculos_dinamicos(
            estado_nuevo,
            dinamicos_nuevos,
        ):

            resultado = "colision_dinamica"

            indice_obstaculo_colision = obtener_indice_colision_dinamica(
                estado_nuevo,
                dinamicos_nuevos,
            )

            break

        if robot_llego_meta(
            estado_nuevo,
            meta,
        ):

            resultado = "meta"

            estado_robot = estado_nuevo
            obstaculos_dinamicos = dinamicos_nuevos

            break

        estado_robot = estado_nuevo
        obstaculos_dinamicos = dinamicos_nuevos

        velocidad_actual = velocidad_lineal
        omega_actual = velocidad_angular

    simulacion = {
        "estados": estados_robot,
        "controles": controles,
        "puntos_objetivo": puntos_objetivo,
        "errores_angulares": errores_angulares,
        "resultado": resultado,
        "pasos_ejecutados": pasos_ejecutados,
        "tiempo_total": pasos_ejecutados * dt,
        "historial_dinamicos": historial_dinamicos,
        "clearances_dinamicos": clearances_dinamicos,
        "obstaculos_dinamicos_finales": historial_dinamicos[-1],
        "indice_obstaculo_colision": indice_obstaculo_colision,
        "historial_submetas": historial_submetas,
        "historial_indices_progreso": historial_indices_progreso,
        "historial_observaciones": historial_observaciones,
        "historial_trayectorias_predichas": historial_trayectorias_predichas,
        "historial_puntuaciones": historial_puntuaciones,
        "historial_candidatos_validos": historial_candidatos_validos,
        "historial_dinamicos_visibles": historial_dinamicos_visibles,
        "historial_frenado_emergencia": historial_frenado_emergencia,
    }

    return simulacion
def calcular_metricas_dinamicas(
    simulacion,
    camino_mundo,
    meta,
    obstaculos_estaticos,
    distancia_directa,
    longitud_camino_astar,
    dt=DT,
):

    metricas = calcular_metricas_seguimiento(
        simulacion=simulacion,
        camino_mundo=camino_mundo,
        meta=meta,
        obstaculos=obstaculos_estaticos,
        distancia_directa=distancia_directa,
        longitud_camino_astar=longitud_camino_astar,
        dt=dt,
    )

    clearances_dinamicos = np.array(
        simulacion[
            "clearances_dinamicos"
        ],
        dtype=float,
    )

    if len(clearances_dinamicos) > 0:

        clearance_dinamico_minimo = float(
            np.min(
                clearances_dinamicos
            )
        )

        clearance_dinamico_medio = float(
            np.mean(
                clearances_dinamicos
            )
        )

    else:

        clearance_dinamico_minimo = float(
            "inf"
        )

        clearance_dinamico_medio = float(
            "inf"
        )

    clearance_total_minimo = min(
        metricas["clearance_minimo"],
        clearance_dinamico_minimo,
    )

    metricas[
        "clearance_dinamico_minimo"
    ] = clearance_dinamico_minimo

    metricas[
        "clearance_dinamico_medio"
    ] = clearance_dinamico_medio

    metricas[
        "clearance_total_minimo"
    ] = clearance_total_minimo

    return metricas
def dibujar_trayectoria_predicha_dwa(
    ax,
    trayectoria_predicha,
):

    if len(trayectoria_predicha) == 0:
        return

    coordenadas_x = [
        estado[0]
        for estado in trayectoria_predicha
    ]

    coordenadas_y = [
        estado[1]
        for estado in trayectoria_predicha
    ]

    ax.plot(
        coordenadas_x,
        coordenadas_y,
        linestyle="--",
        linewidth=2.0,
        label="Última predicción DWA",
        zorder=10,
    )
def dibujar_comparacion_clearance_dinamico(
    clearances_lookahead,
    clearances_dwa,
    dt=DT,
):

    tiempos_lookahead = np.arange(
        len(clearances_lookahead)
    ) * dt

    tiempos_dwa = np.arange(
        len(clearances_dwa)
    ) * dt

    figura, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        tiempos_lookahead,
        clearances_lookahead,
        linewidth=2.0,
        label="A* + seguimiento nominal",
    )

    ax.plot(
        tiempos_dwa,
        clearances_dwa,
        linewidth=2.0,
        label="A* + DWA",
    )

    ax.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.5,
        label="Límite de colisión",
    )

    ax.set_xlabel(
        "Tiempo [s]"
    )

    ax.set_ylabel(
        "Clearance dinámico [m]"
    )

    ax.set_title(
        "Comparación de seguridad dinámica"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()
def crear_registro_simulacion(estado_inicial):

    registro = {
        "estados": [
            estado_inicial
        ],
        "controles": [],
        "clearances_estaticos": [],
        "clearances_dinamicos": [],
        "submetas": [],
        "obstaculos_dinamicos": [],
        "resultado": "en_progreso",
        "pasos_ejecutados": 0,
    }

    return registro
def copiar_obstaculos_dinamicos(
    obstaculos_dinamicos,
):

    copia = []

    for obstaculo in obstaculos_dinamicos:

        obstaculo_copiado = {
            "x": obstaculo["x"],
            "y": obstaculo["y"],
            "vx": obstaculo["vx"],
            "vy": obstaculo["vy"],
            "radio": obstaculo["radio"],
        }

        copia.append(
            obstaculo_copiado
        )

    return copia
def registrar_paso_simulacion(
    registro,
    estado_nuevo,
    velocidad_lineal,
    velocidad_angular,
    clearance_estatico,
    clearance_dinamico,
    submeta,
    obstaculos_dinamicos,
):

    registro["estados"].append(
        estado_nuevo
    )

    registro["controles"].append(
        (
            velocidad_lineal,
            velocidad_angular,
        )
    )

    registro["clearances_estaticos"].append(
        clearance_estatico
    )

    registro["clearances_dinamicos"].append(
        clearance_dinamico
    )

    registro["submetas"].append(
        submeta
    )

    copia_obstaculos = copiar_obstaculos_dinamicos(
        obstaculos_dinamicos
    )

    registro["obstaculos_dinamicos"].append(
        copia_obstaculos
    )

    registro["pasos_ejecutados"] += 1

    return registro
def calcular_clearance_obstaculo(
    estado_robot,
    obstaculo,
):

    x_robot = estado_robot[0]
    y_robot = estado_robot[1]

    distancia_centro = distancia_punto_obstaculo(
        x_robot,
        y_robot,
        obstaculo,
    )

    clearance = (
        distancia_centro
        - RADIO_ROBOT
    )

    return clearance
def calcular_clearance_obstaculos_estaticos(
    estado_robot,
    obstaculos,
):

    clearance_minimo = float("inf")

    for obstaculo in obstaculos:

        clearance = calcular_clearance_obstaculo(
            estado_robot,
            obstaculo,
        )

        if clearance < clearance_minimo:

            clearance_minimo = clearance

    return clearance_minimo
def calcular_clearance_bordes(
    estado_robot,
):

    x_robot = estado_robot[0]
    y_robot = estado_robot[1]

    clearance_izquierdo = (
        x_robot
        - RADIO_ROBOT
    )

    clearance_derecho = (
        ANCHO_MAPA
        - x_robot
        - RADIO_ROBOT
    )

    clearance_inferior = (
        y_robot
        - RADIO_ROBOT
    )

    clearance_superior = (
        ALTO_MAPA
        - y_robot
        - RADIO_ROBOT
    )

    clearance_minimo = min(
        clearance_izquierdo,
        clearance_derecho,
        clearance_inferior,
        clearance_superior,
    )

    return clearance_minimo
def calcular_clearance_estatico(
    estado_robot,
    obstaculos,
):

    clearance_obstaculos = (
        calcular_clearance_obstaculos_estaticos(
            estado_robot,
            obstaculos,
        )
    )

    clearance_bordes = calcular_clearance_bordes(
        estado_robot
    )

    clearance_estatico = min(
        clearance_obstaculos,
        clearance_bordes,
    )

    return clearance_estatico
def calcular_clearance_obstaculo_dinamico(
    estado_robot,
    obstaculo_dinamico,
):

    x_robot = estado_robot[0]
    y_robot = estado_robot[1]

    x_obstaculo = obstaculo_dinamico["x"]
    y_obstaculo = obstaculo_dinamico["y"]
    radio_obstaculo = obstaculo_dinamico["radio"]

    distancia_centros = math.hypot(
        x_robot - x_obstaculo,
        y_robot - y_obstaculo,
    )

    clearance = (
        distancia_centros
        - RADIO_ROBOT
        - radio_obstaculo
    )

    return clearance
def calcular_clearance_dinamico(
    estado_robot,
    obstaculos_dinamicos,
):

    clearance_minimo = float("inf")

    for obstaculo in obstaculos_dinamicos:

        clearance = calcular_clearance_obstaculo_dinamico(
            estado_robot,
            obstaculo,
        )

        if clearance < clearance_minimo:

            clearance_minimo = clearance

    return clearance_minimo
def crear_directorios_benchmark(
    carpeta_base=CARPETA_RESULTADOS,
):

    rutas = {
        "base": Path(
            carpeta_base
        ),
        "csv": Path(
            carpeta_base
        ) / "csv",
        "figuras": Path(
            carpeta_base
        ) / "figuras",
        "trayectorias": Path(
            carpeta_base
        ) / "figuras" / "trayectorias",
        "boxplots": Path(
            carpeta_base
        ) / "figuras" / "boxplots",
    }

    for ruta in rutas.values():

        ruta.mkdir(
            parents=True,
            exist_ok=True,
        )

    return rutas
def crear_estado_inicial_escenario(
    escenario,
):

    inicio = escenario[
        "inicio"
    ]

    camino_mundo = escenario[
        "camino_mundo"
    ]

    if len(camino_mundo) >= 2:

        siguiente_punto = camino_mundo[
            1
        ]

        orientacion_inicial = math.atan2(
            siguiente_punto[1]
            - inicio[1],
            siguiente_punto[0]
            - inicio[0],
        )

    else:

        orientacion_inicial = 0.0

    estado_inicial = (
        inicio[0],
        inicio[1],
        orientacion_inicial,
    )

    return estado_inicial
def ejecutar_metodo_benchmark(
    metodo,
    escenario,
    obstaculos_dinamicos,
):

    estado_inicial = crear_estado_inicial_escenario(
        escenario
    )

    argumentos = {
        "estado_inicial": estado_inicial,
        "camino_mundo": escenario[
            "camino_mundo"
        ],
        "meta": escenario[
            "meta"
        ],
        "obstaculos_estaticos": escenario[
            "obstaculos"
        ],
        "obstaculos_dinamicos_iniciales": obstaculos_dinamicos,
        "pasos_maximos": PASOS_MAXIMOS_SEGUIMIENTO,
        "dt": DT,
    }

    inicio_computo = time.perf_counter()

    if metodo == "lookahead":

        simulacion = simular_seguimiento_dinamico_lookahead(
            **argumentos
        )

    elif metodo == "dwa":

        simulacion = simular_seguimiento_dinamico_dwa(
            **argumentos
        )

    else:

        raise ValueError(
            f"Método desconocido: {metodo}"
        )

    tiempo_computacional = (
        time.perf_counter()
        - inicio_computo
    )

    metricas = calcular_metricas_dinamicas(
        simulacion=simulacion,
        camino_mundo=escenario[
            "camino_mundo"
        ],
        meta=escenario[
            "meta"
        ],
        obstaculos_estaticos=escenario[
            "obstaculos"
        ],
        distancia_directa=escenario[
            "distancia_directa"
        ],
        longitud_camino_astar=escenario[
            "longitud_camino"
        ],
        dt=DT,
    )

    return (
        simulacion,
        metricas,
        tiempo_computacional,
    )
def construir_fila_resultado(
    semilla,
    semilla_dinamica,
    metodo,
    escenario,
    obstaculos_dinamicos,
    simulacion,
    metricas,
    tiempo_computacional,
):

    fila = {
        "semilla_estatica": semilla,
        "semilla_dinamica": semilla_dinamica,
        "metodo": metodo,
        "resultado": metricas[
            "resultado"
        ],
        "exito": int(
            metricas["exito"]
        ),
        "numero_obstaculos_estaticos": len(
            escenario["obstaculos"]
        ),
        "numero_obstaculos_dinamicos": len(
            obstaculos_dinamicos
        ),
        "pasos_ejecutados": metricas[
            "pasos_ejecutados"
        ],
        "tiempo_total_s": metricas[
            "tiempo_total"
        ],
        "tiempo_computacional_s": tiempo_computacional,
        "distancia_directa_m": escenario[
            "distancia_directa"
        ],
        "longitud_astar_m": escenario[
            "longitud_camino"
        ],
        "costo_astar_m": escenario[
            "costo_astar"
        ],
        "eficiencia_geometrica_astar": escenario[
            "eficiencia_geometrica"
        ],
        "distancia_final_meta_m": metricas[
            "distancia_final_meta"
        ],
        "longitud_recorrida_m": metricas[
            "longitud_recorrida"
        ],
        "longitud_equivalente_m": metricas[
            "longitud_equivalente"
        ],
        "eficiencia_navegacion": metricas[
            "eficiencia_navegacion"
        ],
        "error_medio_m": metricas[
            "error_medio"
        ],
        "error_rmse_m": metricas[
            "error_rmse"
        ],
        "error_maximo_m": metricas[
            "error_maximo"
        ],
        "clearance_estatico_minimo_m": metricas[
            "clearance_minimo"
        ],
        "clearance_estatico_medio_m": metricas[
            "clearance_medio"
        ],
        "clearance_dinamico_minimo_m": metricas[
            "clearance_dinamico_minimo"
        ],
        "clearance_dinamico_medio_m": metricas[
            "clearance_dinamico_medio"
        ],
        "clearance_total_minimo_m": metricas[
            "clearance_total_minimo"
        ],
        "variacion_total_v_m_s": metricas[
            "variacion_total_v"
        ],
        "variacion_total_omega_rad_s": metricas[
            "variacion_total_omega"
        ],
        "aceleracion_lineal_rms_m_s2": metricas[
            "aceleracion_lineal_rms"
        ],
        "aceleracion_angular_rms_rad_s2": metricas[
            "aceleracion_angular_rms"
        ],
        "esfuerzo_control": metricas[
            "esfuerzo_control"
        ],
        "frenados_emergencia": 0,
        "candidatos_validos_promedio": "",
        "horizonte_dwa_s": (
            HORIZONTE_PREDICCION_DWA
            if metodo == "dwa"
            else ""
        ),
        "muestras_v_dwa": (
            NUMERO_MUESTRAS_V_DWA
            if metodo == "dwa"
            else ""
        ),
        "muestras_omega_dwa": (
            NUMERO_MUESTRAS_OMEGA_DWA
            if metodo == "dwa"
            else ""
        ),
    }

    if metodo == "dwa":

        fila[
            "frenados_emergencia"
        ] = int(
            np.sum(
                simulacion.get(
                    "historial_frenado_emergencia",
                    [],
                )
            )
        )

        candidatos_validos = simulacion.get(
            "historial_candidatos_validos",
            [],
        )

        if len(candidatos_validos) > 0:

            fila[
                "candidatos_validos_promedio"
            ] = float(
                np.mean(
                    candidatos_validos
                )
            )

    return fila
def construir_filas_trayectoria_robot(
    semilla,
    semilla_dinamica,
    metodo,
    simulacion,
    metricas,
):

    filas = []

    estados = simulacion[
        "estados"
    ]

    controles = simulacion[
        "controles"
    ]

    submetas = simulacion.get(
        "puntos_objetivo",
        [],
    )

    indices_progreso = simulacion.get(
        "historial_indices_progreso",
        [],
    )

    clearances_dinamicos = simulacion.get(
        "clearances_dinamicos",
        [],
    )

    errores_ruta = metricas[
        "errores_seguimiento"
    ]

    clearances_estaticos = metricas[
        "clearances"
    ]

    for paso, estado in enumerate(
        estados
    ):

        indice_control = paso - 1

        if (
            indice_control >= 0
            and indice_control < len(
                controles
            )
        ):

            velocidad_lineal = controles[
                indice_control
            ][0]

            velocidad_angular = controles[
                indice_control
            ][1]

        else:

            velocidad_lineal = 0.0
            velocidad_angular = 0.0

        if (
            indice_control >= 0
            and indice_control < len(
                submetas
            )
        ):

            submeta_x = submetas[
                indice_control
            ][0]

            submeta_y = submetas[
                indice_control
            ][1]

        else:

            submeta_x = ""
            submeta_y = ""

        if (
            indice_control >= 0
            and indice_control < len(
                indices_progreso
            )
        ):

            indice_progreso = indices_progreso[
                indice_control
            ]

        else:

            indice_progreso = 0

        if paso < len(
            clearances_dinamicos
        ):

            clearance_dinamico = clearances_dinamicos[
                paso
            ]

        else:

            clearance_dinamico = ""

        fila = {
            "semilla_estatica": semilla,
            "semilla_dinamica": semilla_dinamica,
            "metodo": metodo,
            "paso": paso,
            "tiempo_s": paso * DT,
            "x_robot_m": estado[0],
            "y_robot_m": estado[1],
            "theta_robot_rad": estado[2],
            "velocidad_lineal_m_s": velocidad_lineal,
            "velocidad_angular_rad_s": velocidad_angular,
            "submeta_x_m": submeta_x,
            "submeta_y_m": submeta_y,
            "indice_progreso": indice_progreso,
            "error_ruta_m": (
                errores_ruta[paso]
                if paso < len(
                    errores_ruta
                )
                else ""
            ),
            "clearance_estatico_m": (
                clearances_estaticos[
                    paso
                ]
                if paso < len(
                    clearances_estaticos
                )
                else ""
            ),
            "clearance_dinamico_m": clearance_dinamico,
            "resultado_episodio": simulacion[
                "resultado"
            ],
        }

        filas.append(
            fila
        )

    return filas
def construir_filas_trayectoria_obstaculos(
    semilla,
    semilla_dinamica,
    metodo,
    simulacion,
):

    filas = []

    historial_dinamicos = simulacion[
        "historial_dinamicos"
    ]

    for paso, instante in enumerate(
        historial_dinamicos
    ):

        tiempo_actual = paso * DT

        for indice, obstaculo in enumerate(
            instante
        ):

            tiempo_inicio = obstaculo.get(
                "tiempo_inicio",
                0.0,
            )

            fila = {
                "semilla_estatica": semilla,
                "semilla_dinamica": semilla_dinamica,
                "metodo": metodo,
                "paso": paso,
                "tiempo_s": tiempo_actual,
                "indice_obstaculo": indice,
                "x_m": obstaculo[
                    "x"
                ],
                "y_m": obstaculo[
                    "y"
                ],
                "vx_m_s": obstaculo[
                    "vx"
                ],
                "vy_m_s": obstaculo[
                    "vy"
                ],
                "radio_m": obstaculo[
                    "radio"
                ],
                "tipo": obstaculo[
                    "tipo"
                ],
                "tiempo_inicio_s": tiempo_inicio,
                "activo": int(
                    tiempo_actual
                    >= tiempo_inicio
                ),
            }

            filas.append(
                fila
            )

    return filas
def guardar_filas_csv(
    ruta_archivo,
    filas,
):

    if len(filas) == 0:
        return

    campos = []

    for fila in filas:

        for campo in fila:

            if campo not in campos:

                campos.append(
                    campo
                )

    with open(
        ruta_archivo,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos,
        )

        escritor.writeheader()

        escritor.writerows(
            filas
        )
def calcular_resumen_benchmark(
    resultados,
):

    resumen = []

    metricas_continuas = [
        "tiempo_total_s",
        "longitud_recorrida_m",
        "eficiencia_navegacion",
        "error_rmse_m",
        "clearance_dinamico_minimo_m",
        "clearance_total_minimo_m",
        "aceleracion_angular_rms_rad_s2",
        "esfuerzo_control",
    ]

    for metodo in METODOS_BENCHMARK:

        filas_metodo = [
            fila
            for fila in resultados
            if fila[
                "metodo"
            ] == metodo
        ]

        filas_exitosas = [
            fila
            for fila in filas_metodo
            if fila[
                "exito"
            ] == 1
        ]

        conteo_resultados = Counter(
            fila[
                "resultado"
            ]
            for fila in filas_metodo
        )

        numero_episodios = len(
            filas_metodo
        )

        numero_exitos = len(
            filas_exitosas
        )

        fila_resumen = {
            "metodo": metodo,
            "episodios": numero_episodios,
            "exitos": numero_exitos,
            "tasa_exito": (
                numero_exitos
                / numero_episodios
                if numero_episodios > 0
                else 0.0
            ),
            "metas": conteo_resultados.get(
                "meta",
                0,
            ),
            "colisiones_dinamicas": conteo_resultados.get(
                "colision_dinamica",
                0,
            ),
            "colisiones_estaticas": conteo_resultados.get(
                "colision_estatica",
                0,
            ),
            "fuera_mapa": conteo_resultados.get(
                "fuera_mapa",
                0,
            ),
            "timeouts": conteo_resultados.get(
                "timeout",
                0,
            ),
        }

        for metrica in metricas_continuas:

            valores = np.array(
                [
                    float(
                        fila[
                            metrica
                        ]
                    )
                    for fila in filas_exitosas
                    if fila[
                        metrica
                    ] != ""
                    and np.isfinite(
                        float(
                            fila[
                                metrica
                            ]
                        )
                    )
                ],
                dtype=float,
            )

            if len(valores) > 0:

                fila_resumen[
                    f"{metrica}_n"
                ] = len(
                    valores
                )

                fila_resumen[
                    f"{metrica}_media"
                ] = float(
                    np.mean(
                        valores
                    )
                )

                fila_resumen[
                    f"{metrica}_std"
                ] = (
                    float(
                        np.std(
                            valores,
                            ddof=1,
                        )
                    )
                    if len(valores) > 1
                    else 0.0
                )

                fila_resumen[
                    f"{metrica}_mediana"
                ] = float(
                    np.median(
                        valores
                    )
                )

                fila_resumen[
                    f"{metrica}_q25"
                ] = float(
                    np.percentile(
                        valores,
                        25,
                    )
                )

                fila_resumen[
                    f"{metrica}_q75"
                ] = float(
                    np.percentile(
                        valores,
                        75,
                    )
                )

            else:

                fila_resumen[
                    f"{metrica}_n"
                ] = 0

                fila_resumen[
                    f"{metrica}_media"
                ] = ""

                fila_resumen[
                    f"{metrica}_std"
                ] = ""

                fila_resumen[
                    f"{metrica}_mediana"
                ] = ""

                fila_resumen[
                    f"{metrica}_q25"
                ] = ""

                fila_resumen[
                    f"{metrica}_q75"
                ] = ""

        resumen.append(
            fila_resumen
        )

    return resumen
def guardar_figura_episodio(
    semilla,
    metodo,
    escenario,
    obstaculos_dinamicos_iniciales,
    simulacion,
    metricas,
    ruta_archivo,
):

    figura, ax = plt.subplots(
        figsize=(
            9,
            9,
        )
    )

    configurar_mapa(
        ax
    )

    dibujar_rejilla_ocupacion(
        ax,
        escenario[
            "rejilla"
        ],
    )

    dibujar_obstaculos(
        ax,
        escenario[
            "obstaculos"
        ],
    )

    dibujar_camino_astar(
        ax,
        escenario[
            "camino_mundo"
        ],
    )

    dibujar_trayectoria_estados(
        ax,
        simulacion[
            "estados"
        ],
        etiqueta=(
            f"Trayectoria {metodo}"
        ),
        estilo="-",
    )

    trayectorias_dinamicas = extraer_trayectorias_dinamicas(
        simulacion[
            "historial_dinamicos"
        ]
    )

    dibujar_trayectorias_dinamicas(
        ax,
        trayectorias_dinamicas,
    )

    dibujar_obstaculos_dinamicos(
        ax,
        obstaculos_dinamicos_iniciales,
        etiqueta=(
            "Obstáculos dinámicos iniciales"
        ),
        relleno=False,
        transparencia=0.9,
    )

    dibujar_obstaculos_dinamicos(
        ax,
        simulacion[
            "obstaculos_dinamicos_finales"
        ],
        etiqueta=(
            "Obstáculos dinámicos finales"
        ),
        relleno=True,
        transparencia=0.7,
    )

    dibujar_inicio(
        ax,
        escenario[
            "inicio"
        ],
    )

    dibujar_meta(
        ax,
        escenario[
            "meta"
        ],
    )

    dibujar_resultado_dinamico(
        ax,
        simulacion[
            "estados"
        ],
        simulacion[
            "resultado"
        ],
    )

    ax.set_title(
        f"Semilla {semilla} | "
        f"{metodo}\n"
        f"Resultado = "
        f"{simulacion['resultado']} | "
        f"Clearance dinámico = "
        f"{metricas['clearance_dinamico_minimo']:.3f} m"
    )

    ax.legend(
        loc="upper right"
    )

    figura.savefig(
        ruta_archivo,
        dpi=200,
        bbox_inches="tight",
    )

    if MOSTRAR_FIGURAS_BENCHMARK:

        figura.show()

    else:

        plt.close(
            figura
        )
def crear_graficas_benchmark(
    resultados,
    carpeta_boxplots,
):

    tasas_exito = []

    for metodo in METODOS_BENCHMARK:

        filas_metodo = [
            fila
            for fila in resultados
            if fila[
                "metodo"
            ] == metodo
        ]

        if len(filas_metodo) > 0:

            tasa = (
                100.0
                * sum(
                    fila[
                        "exito"
                    ]
                    for fila in filas_metodo
                )
                / len(
                    filas_metodo
                )
            )

        else:

            tasa = 0.0

        tasas_exito.append(
            tasa
        )

    figura, ax = plt.subplots(
        figsize=(
            7,
            4,
        )
    )

    ax.bar(
        METODOS_BENCHMARK,
        tasas_exito,
    )

    ax.set_ylabel(
        "Tasa de éxito [%]"
    )

    ax.set_ylim(
        0.0,
        100.0,
    )

    ax.set_title(
        "Tasa de éxito por método"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figura.savefig(
        carpeta_boxplots
        / "tasa_exito.png",
        dpi=200,
        bbox_inches="tight",
    )

    if not MOSTRAR_FIGURAS_BENCHMARK:

        plt.close(
            figura
        )

    configuraciones = [
        (
            "tiempo_total_s",
            "Tiempo de navegación [s]",
            "boxplot_tiempo.png",
        ),
        (
            "longitud_recorrida_m",
            "Longitud recorrida [m]",
            "boxplot_longitud.png",
        ),
        (
            "eficiencia_navegacion",
            "Eficiencia de navegación",
            "boxplot_eficiencia.png",
        ),
        (
            "error_rmse_m",
            "RMSE respecto de A* [m]",
            "boxplot_rmse.png",
        ),
        (
            "clearance_dinamico_minimo_m",
            "Clearance dinámico mínimo [m]",
            "boxplot_clearance_dinamico.png",
        ),
        (
            "aceleracion_angular_rms_rad_s2",
            "Aceleración angular RMS [rad/s²]",
            "boxplot_aceleracion_angular.png",
        ),
        (
            "esfuerzo_control",
            "Esfuerzo de control",
            "boxplot_esfuerzo.png",
        ),
    ]

    for (
        campo,
        etiqueta_y,
        nombre_archivo,
    ) in configuraciones:

        datos = []
        etiquetas = []

        for metodo in METODOS_BENCHMARK:

            valores = [
                float(
                    fila[
                        campo
                    ]
                )
                for fila in resultados
                if fila[
                    "metodo"
                ] == metodo
                and fila[
                    "exito"
                ] == 1
                and fila[
                    campo
                ] != ""
                and np.isfinite(
                    float(
                        fila[
                            campo
                        ]
                    )
                )
            ]

            if len(valores) > 0:

                datos.append(
                    valores
                )

                etiquetas.append(
                    metodo
                )

        if len(datos) == 0:

            continue

        figura, ax = plt.subplots(
            figsize=(
                7,
                4,
            )
        )

        ax.boxplot(
            datos,
            labels=etiquetas,
            showmeans=True,
        )

        ax.set_ylabel(
            etiqueta_y
        )

        ax.set_title(
            "Episodios exitosos"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        figura.savefig(
            carpeta_boxplots
            / nombre_archivo,
            dpi=200,
            bbox_inches="tight",
        )

        if not MOSTRAR_FIGURAS_BENCHMARK:

            plt.close(
                figura
            )
def ejecutar_benchmark_multisemilla(
    semillas=SEMILLAS_DESARROLLO,
):

    rutas = crear_directorios_benchmark()

    resultados = []

    trayectorias_robot = []

    trayectorias_obstaculos = []

    numero_total_escenarios = len(
        semillas
    )

    for (
        numero_escenario,
        semilla,
    ) in enumerate(
        semillas,
        start=1,
    ):

        print()

        print(
            "========================================"
        )

        print(
            f"ESCENARIO "
            f"{numero_escenario}/"
            f"{numero_total_escenarios}"
        )

        print(
            "Semilla:",
            semilla,
        )

        print(
            "========================================"
        )

        escenario = generar_escenario_valido(
            semilla
        )

        if escenario is None:

            print(
                "No fue posible generar "
                "un escenario válido."
            )

            continue

        semilla_dinamica = (
            semilla
            + DESFASE_SEMILLA_DINAMICA
        )

        obstaculos_dinamicos = generar_obstaculos_dinamicos(
            camino_mundo=escenario[
                "camino_mundo"
            ],
            obstaculos_estaticos=escenario[
                "obstaculos"
            ],
            inicio=escenario[
                "inicio"
            ],
            meta=escenario[
                "meta"
            ],
            numero_obstaculos=NUMERO_OBSTACULOS_DINAMICOS,
            semilla=semilla_dinamica,
        )

        for metodo in METODOS_BENCHMARK:

            (
                simulacion,
                metricas,
                tiempo_computacional,
            ) = ejecutar_metodo_benchmark(
                metodo,
                escenario,
                obstaculos_dinamicos,
            )

            fila_resultado = construir_fila_resultado(
                semilla=semilla,
                semilla_dinamica=semilla_dinamica,
                metodo=metodo,
                escenario=escenario,
                obstaculos_dinamicos=obstaculos_dinamicos,
                simulacion=simulacion,
                metricas=metricas,
                tiempo_computacional=tiempo_computacional,
            )

            resultados.append(
                fila_resultado
            )

            if GUARDAR_TRAYECTORIAS:

                nuevas_filas_robot = construir_filas_trayectoria_robot(
                    semilla=semilla,
                    semilla_dinamica=semilla_dinamica,
                    metodo=metodo,
                    simulacion=simulacion,
                    metricas=metricas,
                )

                trayectorias_robot.extend(
                    nuevas_filas_robot
                )

                nuevas_filas_obstaculos = construir_filas_trayectoria_obstaculos(
                    semilla=semilla,
                    semilla_dinamica=semilla_dinamica,
                    metodo=metodo,
                    simulacion=simulacion,
                )

                trayectorias_obstaculos.extend(
                    nuevas_filas_obstaculos
                )

            if (
                GUARDAR_FIGURAS_TRAYECTORIA
                and semilla
                in SEMILLAS_FIGURAS_TRAYECTORIA
            ):

                ruta_figura = (
                    rutas[
                        "trayectorias"
                    ]
                    / (
                        f"semilla_"
                        f"{semilla:04d}_"
                        f"{metodo}.png"
                    )
                )

                guardar_figura_episodio(
                    semilla=semilla,
                    metodo=metodo,
                    escenario=escenario,
                    obstaculos_dinamicos_iniciales=obstaculos_dinamicos,
                    simulacion=simulacion,
                    metricas=metricas,
                    ruta_archivo=ruta_figura,
                )

            print(
                metodo,
                "| resultado:",
                metricas[
                    "resultado"
                ],
                "| clearance dinámico:",
                f"{metricas['clearance_dinamico_minimo']:.3f} m",
                "| cómputo:",
                f"{tiempo_computacional:.2f} s",
            )

        # Guardado incremental para no perder resultados
        # si una ejecución se interrumpe.
        guardar_filas_csv(
            rutas[
                "csv"
            ] / "resultados_episodios.csv",
            resultados,
        )

        if GUARDAR_TRAYECTORIAS:

            guardar_filas_csv(
                rutas[
                    "csv"
                ] / "trayectorias_robot.csv",
                trayectorias_robot,
            )

            guardar_filas_csv(
                rutas[
                    "csv"
                ] / "trayectorias_obstaculos.csv",
                trayectorias_obstaculos,
            )

    resumen = calcular_resumen_benchmark(
        resultados
    )

    guardar_filas_csv(
        rutas[
            "csv"
        ] / "resumen_metodos.csv",
        resumen,
    )

    crear_graficas_benchmark(
        resultados,
        rutas[
            "boxplots"
        ],
    )

    salida = {
        "resultados": resultados,
        "resumen": resumen,
        "rutas": rutas,
    }

    return salida
def calcular_clearance_total(
    estado_robot,
    obstaculos_estaticos,
    obstaculos_dinamicos,
):

    clearance_estatico = calcular_clearance_estatico(
        estado_robot,
        obstaculos_estaticos,
    )

    clearance_dinamico = calcular_clearance_dinamico(
        estado_robot,
        obstaculos_dinamicos,
    )

    clearance_total = min(
        clearance_estatico,
        clearance_dinamico,
    )

    return (
        clearance_total,
        clearance_estatico,
        clearance_dinamico,
    )
def determinar_resultado_episodio(
    estado_robot,
    meta,
    clearance_estatico,
    clearance_dinamico,
    paso_actual,
    pasos_maximos,
):

    x_robot = estado_robot[0]
    y_robot = estado_robot[1]

    # ------------------------------------------------------
    # 1. Salida del mapa
    # ------------------------------------------------------

    if robot_fuera_del_mapa(
        x_robot,
        y_robot,
    ):

        return True, "fuera_mapa"

    # ------------------------------------------------------
    # 2. Colisión contra obstáculos estáticos
    # ------------------------------------------------------

    if clearance_estatico <= 0.0:

        return True, "colision_estatica"

    # ------------------------------------------------------
    # 3. Colisión contra obstáculos dinámicos
    # ------------------------------------------------------

    if clearance_dinamico <= 0.0:

        return True, "colision_dinamica"

    # ------------------------------------------------------
    # 4. Llegada a la meta
    # ------------------------------------------------------

    if robot_llego_meta(
        estado_robot,
        meta,
    ):

        return True, "meta"

    # ------------------------------------------------------
    # 5. Límite de tiempo
    # ------------------------------------------------------

    if paso_actual >= pasos_maximos:

        return True, "timeout"

    # ------------------------------------------------------
    # 6. El episodio continúa
    # ------------------------------------------------------

    return False, "en_progreso"
def calcular_tiempo_navegacion(
    registro,
    dt=DT,
):

    numero_pasos = registro[
        "pasos_ejecutados"
    ]

    tiempo_total = (
        numero_pasos
        * dt
    )

    return tiempo_total
def calcular_longitud_recorrida(
    registro,
):

    estados = registro[
        "estados"
    ]

    longitud_total = 0.0

    for indice in range(
        len(estados) - 1
    ):

        estado_actual = estados[
            indice
        ]

        estado_siguiente = estados[
            indice + 1
        ]

        diferencia_x = (
            estado_siguiente[0]
            - estado_actual[0]
        )

        diferencia_y = (
            estado_siguiente[1]
            - estado_actual[1]
        )

        distancia = math.hypot(
            diferencia_x,
            diferencia_y,
        )

        longitud_total += distancia

    return longitud_total
def calcular_distancia_final_meta(
    registro,
    meta,
):

    estado_final = registro[
        "estados"
    ][-1]

    diferencia_x = (
        meta[0]
        - estado_final[0]
    )

    diferencia_y = (
        meta[1]
        - estado_final[1]
    )

    distancia_final = math.hypot(
        diferencia_x,
        diferencia_y,
    )

    return distancia_final
def calcular_eficiencia_navegacion(
    registro,
    longitud_camino_astar,
):

    # La métrica solamente es válida cuando el robot
    # realmente alcanza la meta.
    if registro["resultado"] != "meta":

        return np.nan

    longitud_recorrida = calcular_longitud_recorrida(
        registro
    )

    if longitud_recorrida <= 1e-8:

        return np.nan

    eficiencia = (
        longitud_camino_astar
        / longitud_recorrida
    )

    return eficiencia
def calcular_exceso_longitud_porcentual(
    registro,
    longitud_camino_astar,
):

    if registro["resultado"] != "meta":

        return np.nan

    if longitud_camino_astar <= 1e-8:

        return np.nan

    longitud_recorrida = calcular_longitud_recorrida(
        registro
    )

    exceso_porcentual = (
        (
            longitud_recorrida
            - longitud_camino_astar
        )
        / longitud_camino_astar
        * 100.0
    )

    return exceso_porcentual
def filtrar_clearances_validos(
    clearances,
):

    clearances_validos = []

    for clearance in clearances:

        if np.isfinite(clearance):

            clearances_validos.append(
                clearance
            )

    return clearances_validos
def calcular_clearance_minimo(
    clearances,
):

    clearances_validos = filtrar_clearances_validos(
        clearances
    )

    if len(clearances_validos) == 0:

        return np.nan

    clearance_minimo = min(
        clearances_validos
    )

    return clearance_minimo
def calcular_clearance_promedio(
    clearances,
):

    clearances_validos = filtrar_clearances_validos(
        clearances
    )

    if len(clearances_validos) == 0:

        return np.nan

    clearance_promedio = np.mean(
        clearances_validos
    )

    return float(
        clearance_promedio
    )
def calcular_metricas_clearance_estatico(
    registro,
):

    clearances = registro[
        "clearances_estaticos"
    ]

    clearance_minimo = calcular_clearance_minimo(
        clearances
    )

    clearance_promedio = calcular_clearance_promedio(
        clearances
    )

    return (
        clearance_minimo,
        clearance_promedio,
    )
def calcular_metricas_clearance_dinamico(
    registro,
):

    clearances = registro[
        "clearances_dinamicos"
    ]

    clearance_minimo = calcular_clearance_minimo(
        clearances
    )

    clearance_promedio = calcular_clearance_promedio(
        clearances
    )

    return (
        clearance_minimo,
        clearance_promedio,
    )
def construir_clearances_totales(
    registro,
):

    clearances_estaticos = registro[
        "clearances_estaticos"
    ]

    clearances_dinamicos = registro[
        "clearances_dinamicos"
    ]

    clearances_totales = []

    numero_pasos = min(
        len(clearances_estaticos),
        len(clearances_dinamicos),
    )

    for indice in range(
        numero_pasos
    ):

        clearance_estatico = clearances_estaticos[
            indice
        ]

        clearance_dinamico = clearances_dinamicos[
            indice
        ]

        clearance_total = min(
            clearance_estatico,
            clearance_dinamico,
        )

        clearances_totales.append(
            clearance_total
        )

    return clearances_totales
def calcular_metricas_clearance_total(
    registro,
):

    clearances_totales = construir_clearances_totales(
        registro
    )

    clearance_minimo = calcular_clearance_minimo(
        clearances_totales
    )

    clearance_promedio = calcular_clearance_promedio(
        clearances_totales
    )

    return (
        clearance_minimo,
        clearance_promedio,
    )
def verificar_registro_clearances(
    registro,
):

    numero_estaticos = len(
        registro["clearances_estaticos"]
    )

    numero_dinamicos = len(
        registro["clearances_dinamicos"]
    )

    numero_controles = len(
        registro["controles"]
    )

    son_consistentes = (
        numero_estaticos == numero_controles
        and numero_dinamicos == numero_controles
    )

    return son_consistentes
def calcular_errores_ruta(
    registro,
    camino_mundo,
):

    errores = calcular_errores_seguimiento(
        registro["estados"],
        camino_mundo,
    )

    return errores
def calcular_error_medio_ruta(
    registro,
    camino_mundo,
):

    errores = calcular_errores_ruta(
        registro,
        camino_mundo,
    )

    if len(errores) == 0:
        return np.nan

    error_medio = np.mean(
        errores
    )

    return float(
        error_medio
    )
def calcular_error_rmse_ruta(
    registro,
    camino_mundo,
):

    errores = calcular_errores_ruta(
        registro,
        camino_mundo,
    )

    if len(errores) == 0:
        return np.nan

    errores = np.array(
        errores,
        dtype=float,
    )

    error_rmse = np.sqrt(
        np.mean(
            errores ** 2
        )
    )

    return float(
        error_rmse
    )
def calcular_error_maximo_ruta(
    registro,
    camino_mundo,
):

    errores = calcular_errores_ruta(
        registro,
        camino_mundo,
    )

    if len(errores) == 0:
        return np.nan

    error_maximo = max(
        errores
    )

    return float(
        error_maximo
    )
def calcular_metricas_suavidad_control(
    registro,
    dt=DT,
):

    metricas_control = calcular_metricas_control(
        registro["controles"],
        dt,
    )

    return metricas_control
def calcular_metricas_episodio_estandar(
    registro,
    camino_mundo,
    meta,
    longitud_camino_astar,
    dt=DT,
):

    # ------------------------------------------------------
    # 1. Resultado general
    # ------------------------------------------------------

    resultado = registro[
        "resultado"
    ]

    exito = (
        resultado == "meta"
    )

    # ------------------------------------------------------
    # 2. Tiempo y recorrido
    # ------------------------------------------------------

    tiempo_navegacion = calcular_tiempo_navegacion(
        registro,
        dt,
    )

    longitud_recorrida = calcular_longitud_recorrida(
        registro
    )

    distancia_final_meta = calcular_distancia_final_meta(
        registro,
        meta,
    )

    eficiencia_navegacion = calcular_eficiencia_navegacion(
        registro,
        longitud_camino_astar,
    )

    exceso_longitud_porcentual = (
        calcular_exceso_longitud_porcentual(
            registro,
            longitud_camino_astar,
        )
    )

    # ------------------------------------------------------
    # 3. Error respecto de A*
    # ------------------------------------------------------

    error_medio_ruta = calcular_error_medio_ruta(
        registro,
        camino_mundo,
    )

    error_rmse_ruta = calcular_error_rmse_ruta(
        registro,
        camino_mundo,
    )

    error_maximo_ruta = calcular_error_maximo_ruta(
        registro,
        camino_mundo,
    )

    # ------------------------------------------------------
    # 4. Clearance estático
    # ------------------------------------------------------

    (
        clearance_estatico_minimo,
        clearance_estatico_promedio,
    ) = calcular_metricas_clearance_estatico(
        registro
    )

    # ------------------------------------------------------
    # 5. Clearance dinámico
    # ------------------------------------------------------

    (
        clearance_dinamico_minimo,
        clearance_dinamico_promedio,
    ) = calcular_metricas_clearance_dinamico(
        registro
    )

    # ------------------------------------------------------
    # 6. Clearance total
    # ------------------------------------------------------

    (
        clearance_total_minimo,
        clearance_total_promedio,
    ) = calcular_metricas_clearance_total(
        registro
    )

    # ------------------------------------------------------
    # 7. Suavidad del control
    # ------------------------------------------------------

    metricas_control = calcular_metricas_suavidad_control(
        registro,
        dt,
    )

    # ------------------------------------------------------
    # 8. Construir un solo diccionario
    # ------------------------------------------------------

    metricas = {
        "resultado": resultado,
        "exito": exito,

        "colision_estatica": (
            resultado == "colision_estatica"
        ),

        "colision_dinamica": (
            resultado == "colision_dinamica"
        ),

        "fuera_mapa": (
            resultado == "fuera_mapa"
        ),

        "timeout": (
            resultado == "timeout"
        ),

        "pasos_ejecutados": registro[
            "pasos_ejecutados"
        ],

        "tiempo_navegacion": tiempo_navegacion,

        "longitud_recorrida": longitud_recorrida,

        "longitud_astar": float(
            longitud_camino_astar
        ),

        "distancia_final_meta": distancia_final_meta,

        "eficiencia_navegacion": eficiencia_navegacion,

        "exceso_longitud_porcentual": (
            exceso_longitud_porcentual
        ),

        "error_medio_ruta": error_medio_ruta,

        "error_rmse_ruta": error_rmse_ruta,

        "error_maximo_ruta": error_maximo_ruta,

        "clearance_estatico_minimo": (
            clearance_estatico_minimo
        ),

        "clearance_estatico_promedio": (
            clearance_estatico_promedio
        ),

        "clearance_dinamico_minimo": (
            clearance_dinamico_minimo
        ),

        "clearance_dinamico_promedio": (
            clearance_dinamico_promedio
        ),

        "clearance_total_minimo": (
            clearance_total_minimo
        ),

        "clearance_total_promedio": (
            clearance_total_promedio
        ),

        "variacion_total_v": metricas_control[
            "variacion_total_v"
        ],

        "variacion_total_omega": metricas_control[
            "variacion_total_omega"
        ],

        "aceleracion_lineal_rms": metricas_control[
            "aceleracion_lineal_rms"
        ],

        "aceleracion_angular_rms": metricas_control[
            "aceleracion_angular_rms"
        ],

        "esfuerzo_control": metricas_control[
            "esfuerzo_control"
        ],

        "registro_consistente": (
            verificar_registro_clearances(
                registro
            )
        ),
    }

    return metricas
def convertir_simulacion_dwa_a_registro(
    simulacion_dwa,
    obstaculos_estaticos,
):

    estados = simulacion_dwa[
        "estados"
    ]

    controles = simulacion_dwa[
        "controles"
    ]

    historial_dinamicos = simulacion_dwa.get(
        "historial_dinamicos",
        [],
    )

    historial_submetas = simulacion_dwa.get(
        "historial_submetas",
        [],
    )

    # ------------------------------------------------------
    # 1. Comprobar que existe al menos el estado inicial
    # ------------------------------------------------------

    if len(estados) == 0:

        raise ValueError(
            "La simulación DWA no contiene estados."
        )

    # ------------------------------------------------------
    # 2. Crear el registro con el estado inicial
    # ------------------------------------------------------

    registro = crear_registro_simulacion(
        estados[0]
    )

    numero_pasos = min(
        len(controles),
        len(estados) - 1,
    )

    # ------------------------------------------------------
    # 3. Convertir cada paso
    # ------------------------------------------------------

    for indice in range(
        numero_pasos
    ):

        estado_nuevo = estados[
            indice + 1
        ]

        velocidad_lineal = controles[
            indice
        ][0]

        velocidad_angular = controles[
            indice
        ][1]

        # El historial dinámico normalmente contiene primero
        # el estado inicial de los obstáculos.
        indice_dinamicos = indice + 1

        if indice_dinamicos < len(
            historial_dinamicos
        ):

            obstaculos_dinamicos = historial_dinamicos[
                indice_dinamicos
            ]

        else:

            obstaculos_dinamicos = []

        if indice < len(
            historial_submetas
        ):

            submeta = historial_submetas[
                indice
            ]

        else:

            submeta = None

        clearance_estatico = calcular_clearance_estatico(
            estado_nuevo,
            obstaculos_estaticos,
        )

        clearance_dinamico = calcular_clearance_dinamico(
            estado_nuevo,
            obstaculos_dinamicos,
        )

        registro = registrar_paso_simulacion(
            registro=registro,
            estado_nuevo=estado_nuevo,
            velocidad_lineal=velocidad_lineal,
            velocidad_angular=velocidad_angular,
            clearance_estatico=clearance_estatico,
            clearance_dinamico=clearance_dinamico,
            submeta=submeta,
            obstaculos_dinamicos=obstaculos_dinamicos,
        )

    # ------------------------------------------------------
    # 4. Copiar el resultado final
    # ------------------------------------------------------

    registro["resultado"] = simulacion_dwa.get(
        "resultado",
        "en_progreso",
    )

    return registro
def ejecutar_episodio_dwa_estandar(
    semilla,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
):

    # ------------------------------------------------------
    # 1. Generar el escenario estático
    # ------------------------------------------------------

    escenario = generar_escenario_valido(
        semilla
    )

    if escenario is None:

        raise RuntimeError(
            "No fue posible generar un escenario válido "
            f"con la semilla {semilla}."
        )

    # ------------------------------------------------------
    # 2. Crear el estado inicial del robot
    # ------------------------------------------------------

    estado_inicial = crear_estado_inicial_escenario(
        escenario
    )

    # ------------------------------------------------------
    # 3. Generar los obstáculos dinámicos
    # ------------------------------------------------------

    semilla_dinamica = (
        semilla
        + DESFASE_SEMILLA_DINAMICA
    )

    obstaculos_dinamicos = generar_obstaculos_dinamicos(
        camino_mundo=escenario[
            "camino_mundo"
        ],
        obstaculos_estaticos=escenario[
            "obstaculos"
        ],
        inicio=escenario[
            "inicio"
        ],
        meta=escenario[
            "meta"
        ],
        numero_obstaculos=(
            NUMERO_OBSTACULOS_DINAMICOS
        ),
        semilla=semilla_dinamica,
    )

    # ------------------------------------------------------
    # 4. Ejecutar el controlador DWA
    # ------------------------------------------------------

    simulacion_dwa = simular_seguimiento_dinamico_dwa(
        estado_inicial=estado_inicial,
        camino_mundo=escenario[
            "camino_mundo"
        ],
        meta=escenario[
            "meta"
        ],
        obstaculos_estaticos=escenario[
            "obstaculos"
        ],
        obstaculos_dinamicos_iniciales=(
            obstaculos_dinamicos
        ),
        pasos_maximos=pasos_maximos,
        dt=dt,
    )

    # ------------------------------------------------------
    # 5. Convertir al registro común
    # ------------------------------------------------------

    registro_dwa = convertir_simulacion_dwa_a_registro(
        simulacion_dwa=simulacion_dwa,
        obstaculos_estaticos=escenario[
            "obstaculos"
        ],
    )

    # ------------------------------------------------------
    # 6. Calcular las métricas estándar
    # ------------------------------------------------------

    metricas_dwa = calcular_metricas_episodio_estandar(
        registro=registro_dwa,
        camino_mundo=escenario[
            "camino_mundo"
        ],
        meta=escenario[
            "meta"
        ],
        longitud_camino_astar=escenario[
            "longitud_camino"
        ],
        dt=dt,
    )

    # ------------------------------------------------------
    # 7. Añadir identificación del experimento
    # ------------------------------------------------------

    metricas_dwa[
        "metodo"
    ] = "dwa"

    metricas_dwa[
        "semilla"
    ] = int(
        semilla
    )

    metricas_dwa[
        "semilla_dinamica"
    ] = int(
        semilla_dinamica
    )

    metricas_dwa[
        "numero_obstaculos_dinamicos"
    ] = len(
        obstaculos_dinamicos
    )

    # ------------------------------------------------------
    # 8. Reunir todos los resultados
    # ------------------------------------------------------

    resultado_episodio = {
        "escenario": escenario,

        "estado_inicial": estado_inicial,

        "obstaculos_dinamicos_iniciales": (
            obstaculos_dinamicos
        ),

        "simulacion": simulacion_dwa,

        "registro": registro_dwa,

        "metricas": metricas_dwa,
    }

    return resultado_episodio
def reiniciar_entorno_sac(
    semilla,
):

    semilla = int(
        semilla
    )

    # ------------------------------------------------------
    # 1. Generar escenario estático
    # ------------------------------------------------------

    escenario = generar_escenario_valido(
        semilla
    )

    if escenario is None:

        raise RuntimeError(
            "No fue posible generar un escenario válido "
            f"con la semilla {semilla}."
        )

    # ------------------------------------------------------
    # 2. Crear estado inicial del robot
    # ------------------------------------------------------

    estado_inicial = crear_estado_inicial_escenario(
        escenario
    )

    # ------------------------------------------------------
    # 3. Generar obstáculos dinámicos
    # ------------------------------------------------------

    semilla_dinamica = (
        semilla
        + DESFASE_SEMILLA_DINAMICA
    )

    obstaculos_dinamicos = generar_obstaculos_dinamicos(
        camino_mundo=escenario[
            "camino_mundo"
        ],
        obstaculos_estaticos=escenario[
            "obstaculos"
        ],
        inicio=escenario[
            "inicio"
        ],
        meta=escenario[
            "meta"
        ],
        numero_obstaculos=(
            NUMERO_OBSTACULOS_DINAMICOS
        ),
        semilla=semilla_dinamica,
    )

    # Conservar una copia de la configuración inicial.
    obstaculos_dinamicos_iniciales = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos
    ]

    # Copia que podrá modificarse durante el episodio.
    obstaculos_dinamicos_actuales = [
        obstaculo.copy()
        for obstaculo in obstaculos_dinamicos
    ]

    # ------------------------------------------------------
    # 4. Estado inicial del control
    # ------------------------------------------------------

    velocidad_lineal_actual = 0.0

    velocidad_angular_actual = 0.0

    # ------------------------------------------------------
    # ESTADO INICIAL DEL PROGRESO GLOBAL
    # ------------------------------------------------------

    distancia_meta_inicial = distancia_entre_puntos(
        (
            estado_inicial[
                0
            ],
            estado_inicial[
                1
            ],
        ),

        escenario[
            "meta"
        ],
    )

    mejor_distancia_meta_inicial = (
        distancia_meta_inicial
    )

    pasos_sin_progreso_inicial = 0

    numero_recuperaciones_estancamiento_inicial = 0
    
    indice_progreso_inicial = 0

    # ------------------------------------------------------
    # 5. Seleccionar primera submeta
    # ------------------------------------------------------

    (
        submeta,
        indice_progreso,
        indice_submeta,
    ) = seleccionar_submeta_local_estable(
        estado_robot=estado_inicial,
        camino_mundo=escenario[
            "camino_mundo"
        ],
        indice_progreso_anterior=(
            indice_progreso_inicial
        ),
    )

    # ------------------------------------------------------
    # 6. Construir primera observación SAC
    # ------------------------------------------------------

    observacion = construir_observacion_sac(
        estado_robot=estado_inicial,

        submeta=submeta,

        meta=escenario[
            "meta"
        ],

        obstaculos_estaticos=escenario[
            "obstaculos"
        ],

        obstaculos_dinamicos=(
            obstaculos_dinamicos_actuales
        ),

        velocidad_lineal_actual=(
            velocidad_lineal_actual
        ),

        velocidad_angular_actual=(
            velocidad_angular_actual
        ),

        indice_progreso=indice_progreso,

        numero_puntos_camino=len(
            escenario[
                "camino_mundo"
            ]
        ),
    )

    # Se conserva esta lista porque otras funciones
    # de visualización y diagnóstico todavía la utilizan.

    obstaculos_observados = (
        obtener_obstaculos_dinamicos_observados(
            estado_robot=estado_inicial,

            obstaculos_dinamicos=(
                obstaculos_dinamicos_actuales
            ),
        )
    )

    # ------------------------------------------------------
    # 7. Verificar la observación SAC
    # ------------------------------------------------------

    if not isinstance(
        observacion,
        dict,
    ):

        raise TypeError(
            "La observación SAC debe ser un diccionario."
        )

    if (
        "parche" not in observacion
        or "escalares" not in observacion
    ):

        raise KeyError(
            "La observación SAC debe contener las claves "
            "'parche' y 'escalares'."
        )

    parche = observacion[
        "parche"
    ]

    escalares = observacion[
        "escalares"
    ]

    forma_parche_esperada = (
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if parche.shape != forma_parche_esperada:

        raise ValueError(
            "El parche inicial SAC tiene forma "
            f"{parche.shape}; se esperaba "
            f"{forma_parche_esperada}."
        )

    if escalares.shape != (
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "Los escalares iniciales SAC tienen forma "
            f"{escalares.shape}; se esperaba "
            f"({DIMENSION_ESCALARES_SAC},)."
        )

    if not np.all(
        np.isfinite(
            parche
        )
    ):

        raise ValueError(
            "El parche inicial SAC contiene NaN "
            "o valores infinitos."
        )

    if not np.all(
        np.isfinite(
            escalares
        )
    ):

        raise ValueError(
            "Los escalares iniciales SAC contienen NaN "
            "o valores infinitos."
        )
    # ------------------------------------------------------
    # 8. Crear registro del episodio
    # ------------------------------------------------------

    registro = crear_registro_simulacion(
        estado_inicial
    )

    # ------------------------------------------------------
    # SERIES DEL PROGRESO GLOBAL Y ESTANCAMIENTO
    # ------------------------------------------------------

    registro[
        "distancias_meta_globales"
    ] = []

    registro[
        "progresos_meta_globales"
    ] = []

    registro[
        "mejores_distancias_meta"
    ] = []

    registro[
        "pasos_sin_progreso"
    ] = []

    registro[
        "intensidades_estancamiento"
    ] = []

    registro[
        "penalizaciones_estancamiento_persistente"
    ] = []

    registro[
        "recuperaciones_estancamiento"
    ] = []

    # ------------------------------------------------------
    # 9. Reunir el estado interno del entorno
    # ------------------------------------------------------

    entorno = {
        "semilla": semilla,

        "semilla_dinamica": (
            semilla_dinamica
        ),

        "escenario": escenario,

        "estado_inicial": estado_inicial,

        "estado_robot": estado_inicial,

        "obstaculos_estaticos": escenario[
            "obstaculos"
        ],

        "obstaculos_dinamicos_iniciales": (
            obstaculos_dinamicos_iniciales
        ),

        "obstaculos_dinamicos": (
            obstaculos_dinamicos_actuales
        ),

        "camino_mundo": escenario[
            "camino_mundo"
        ],

        "meta": escenario[
            "meta"
        ],

        "submeta": submeta,

        "indice_progreso": indice_progreso,

        "indice_submeta": indice_submeta,

        "velocidad_lineal_actual": (
            velocidad_lineal_actual
        ),

        "velocidad_angular_actual": (
            velocidad_angular_actual
        ),

        "paso_actual": 0,

        "tiempo_actual": 0.0,

        "terminado": False,

        "resultado": "en_progreso",

        "observacion": {
            "parche": observacion[
                "parche"
            ].copy(),

            "escalares": observacion[
                "escalares"
            ].copy(),
        },
        "obstaculos_observados": [
            descripcion.copy()
            for descripcion in obstaculos_observados
        ],

        "registro": registro,
        "distancia_meta_anterior": (
            distancia_meta_inicial
        ),

        "mejor_distancia_meta": (
            mejor_distancia_meta_inicial
        ),

        "pasos_sin_progreso": (
            pasos_sin_progreso_inicial
        ),

        "numero_recuperaciones_estancamiento": (
            numero_recuperaciones_estancamiento_inicial
        ),
    }

    observacion_salida = {
        "parche": observacion[
            "parche"
        ].copy(),

        "escalares": observacion[
            "escalares"
        ].copy(),
    }

    return (
        entorno,
        observacion_salida,
    )
# ==========================================================
# RECOMPENSA DEL ENTORNO SAC
# ==========================================================
def calcular_recompensa_sac(
    estado_anterior,
    estado_nuevo,
    submeta,
    meta,
    camino_mundo,
    velocidad_lineal_anterior,
    velocidad_angular_anterior,
    velocidad_lineal_nueva,
    velocidad_angular_nueva,
    clearance_total,
    resultado,
    dt=DT,
    clearance_estatico=None,
    clearance_dinamico=None,
    distancia_meta_anterior=None,
    pasos_sin_progreso=0,
    recuperacion_estancamiento=False,
):

    # ------------------------------------------------------
    # 1. Validaciones básicas
    # ------------------------------------------------------

    dt = float(
        dt
    )

    if (
        not np.isfinite(
            dt
        )
        or dt <= 0.0
    ):

        raise ValueError(
            "El periodo de muestreo debe ser positivo."
        )

    velocidad_lineal_anterior = float(
        velocidad_lineal_anterior
    )

    velocidad_angular_anterior = float(
        velocidad_angular_anterior
    )

    velocidad_lineal_nueva = float(
        velocidad_lineal_nueva
    )

    velocidad_angular_nueva = float(
        velocidad_angular_nueva
    )

    valores_control = [
        velocidad_lineal_anterior,
        velocidad_angular_anterior,
        velocidad_lineal_nueva,
        velocidad_angular_nueva,
    ]

    if not np.all(
        np.isfinite(
            valores_control
        )
    ):

        raise ValueError(
            "Las velocidades de la recompensa contienen "
            "NaN o valores infinitos."
        )

    # ------------------------------------------------------
    # 2. Determinar el modo de clearances
    # ------------------------------------------------------
    #
    # Modo nuevo:
    #
    # clearance estático y dinámico por separado.
    #
    # Modo heredado:
    #
    # solamente clearance_total, para mantener compatibles
    # las verificaciones anteriores.

    clearances_separados = (
        clearance_estatico is not None
        or clearance_dinamico is not None
    )

    if clearances_separados:

        if (
            clearance_estatico is None
            or clearance_dinamico is None
        ):

            raise ValueError(
                "Para utilizar clearances separados deben "
                "entregarse tanto el estático como el "
                "dinámico."
            )

        clearance_estatico = float(
            clearance_estatico
        )

        clearance_dinamico = float(
            clearance_dinamico
        )

        if np.isnan(
            clearance_estatico
        ):

            raise ValueError(
                "El clearance estático contiene NaN."
            )

        if np.isnan(
            clearance_dinamico
        ):

            raise ValueError(
                "El clearance dinámico contiene NaN."
            )

        clearance_total_efectivo = min(
            clearance_estatico,
            clearance_dinamico,
        )

        modo_clearance = "separado"

    else:

        clearance_total_efectivo = float(
            clearance_total
        )

        if np.isnan(
            clearance_total_efectivo
        ):

            raise ValueError(
                "El clearance total contiene NaN."
            )

        clearance_estatico = float(
            "inf"
        )

        clearance_dinamico = float(
            "inf"
        )

        modo_clearance = "legacy_total"

    # ------------------------------------------------------
    # 3. Progreso hacia la submeta local
    # ------------------------------------------------------

    punto_anterior = (
        estado_anterior[
            0
        ],
        estado_anterior[
            1
        ],
    )

    punto_nuevo = (
        estado_nuevo[
            0
        ],
        estado_nuevo[
            1
        ],
    )

    distancia_anterior_submeta = (
        distancia_entre_puntos(
            punto_anterior,
            submeta,
        )
    )

    distancia_nueva_submeta = (
        distancia_entre_puntos(
            punto_nuevo,
            submeta,
        )
    )

    progreso = (
        distancia_anterior_submeta
        - distancia_nueva_submeta
    )

    desplazamiento_maximo_paso = max(
        VELOCIDAD_MAXIMA
        * dt,
        1e-8,
    )

    progreso_normalizado = float(
        np.clip(
            progreso
            / desplazamiento_maximo_paso,

            -1.0,
            1.0,
        )
    )

    recompensa_progreso = (
        PESO_SAC_PROGRESO
        * progreso_normalizado
    )

    # ------------------------------------------------------
    # PROGRESO HACIA LA META GLOBAL
    # ------------------------------------------------------

    distancia_meta_nueva = distancia_entre_puntos(
        punto_nuevo,
        meta,
    )

    if distancia_meta_anterior is None:

        distancia_meta_anterior_efectiva = (
            distancia_entre_puntos(
                punto_anterior,
                meta,
            )
        )

    else:

        distancia_meta_anterior_efectiva = float(
            distancia_meta_anterior
        )

    if (
        not np.isfinite(
            distancia_meta_anterior_efectiva
        )
        or distancia_meta_anterior_efectiva < 0.0
    ):

        raise ValueError(
            "La distancia anterior a la meta no es válida."
        )

    progreso_meta_global = (
        distancia_meta_anterior_efectiva
        - distancia_meta_nueva
    )

    progreso_meta_global_normalizado = float(
        np.clip(
            progreso_meta_global
            / desplazamiento_maximo_paso,

            -1.0,
            1.0,
        )
    )

    recompensa_progreso_meta_global = (
        PESO_SAC_PROGRESO_META_GLOBAL
        * progreso_meta_global_normalizado
    )

    # ------------------------------------------------------
    # 4. Alineación con la submeta
    # ------------------------------------------------------

    angulo_deseado = math.atan2(
        submeta[
            1
        ]
        - estado_nuevo[
            1
        ],

        submeta[
            0
        ]
        - estado_nuevo[
            0
        ],
    )

    error_angular = normalizar_angulo(
        angulo_deseado
        - estado_nuevo[
            2
        ]
    )

    alineacion = math.cos(
        error_angular
    )

    velocidad_normalizada = float(
        np.clip(
            velocidad_lineal_nueva
            / max(
                VELOCIDAD_MAXIMA,
                1e-8,
            ),

            0.0,
            1.0,
        )
    )

    alineacion_con_movimiento = (
        alineacion
        * velocidad_normalizada
    )

    recompensa_alineacion = (
        PESO_SAC_ALINEACION
        * alineacion_con_movimiento
    )

    # ------------------------------------------------------
    # 5. Seguridad estática y dinámica
    # ------------------------------------------------------

    riesgo_clearance_estatico = 0.0

    riesgo_clearance_dinamico = 0.0

    penalizacion_seguridad_estatica = 0.0

    penalizacion_seguridad_dinamica = 0.0

    penalizacion_velocidad_riesgosa_dinamica = 0.0

    if clearances_separados:

        # --------------------------------------------------
        # 5.1. Riesgo estático
        # --------------------------------------------------

        if np.isfinite(
            clearance_estatico
        ):

            riesgo_clearance_estatico = float(
                np.clip(
                    (
                        CLEARANCE_OBJETIVO_SAC_ESTATICO
                        - clearance_estatico
                    )
                    / max(
                        CLEARANCE_OBJETIVO_SAC_ESTATICO,
                        1e-8,
                    ),

                    0.0,
                    1.0,
                )
            )

        penalizacion_seguridad_estatica = (
            PESO_SAC_SEGURIDAD_ESTATICA
            * riesgo_clearance_estatico
        )

        # --------------------------------------------------
        # 5.2. Riesgo dinámico
        # --------------------------------------------------

        if np.isfinite(
            clearance_dinamico
        ):

            riesgo_clearance_dinamico = float(
                np.clip(
                    (
                        CLEARANCE_OBJETIVO_SAC_DINAMICO
                        - clearance_dinamico
                    )
                    / max(
                        CLEARANCE_OBJETIVO_SAC_DINAMICO,
                        1e-8,
                    ),

                    0.0,
                    1.0,
                )
            )

        penalizacion_seguridad_dinamica = (
            PESO_SAC_SEGURIDAD_DINAMICA
            * riesgo_clearance_dinamico
        )

        # --------------------------------------------------
        # 5.3. Penalizar velocidad alta bajo riesgo dinámico
        # --------------------------------------------------
        #
        # No se penaliza simplemente avanzar rápido.
        #
        # La penalización aparece únicamente cuando existe
        # proximidad dinámica y aumenta de forma cuadrática
        # con el riesgo y con la velocidad.

        penalizacion_velocidad_riesgosa_dinamica = (
            PESO_SAC_VELOCIDAD_RIESGOSA_DINAMICA
            * (
                riesgo_clearance_dinamico
                ** 2
            )
            * (
                velocidad_normalizada
                ** 2
            )
        )

        riesgo_clearance = max(
            riesgo_clearance_estatico,
            riesgo_clearance_dinamico,
        )

        penalizacion_seguridad = (
            penalizacion_seguridad_estatica
            + penalizacion_seguridad_dinamica
            + penalizacion_velocidad_riesgosa_dinamica
        )

    else:

        # --------------------------------------------------
        # 5.4. Compatibilidad con la recompensa anterior
        # --------------------------------------------------

        if np.isfinite(
            clearance_total_efectivo
        ):

            riesgo_clearance = float(
                np.clip(
                    (
                        CLEARANCE_OBJETIVO_DWA
                        - clearance_total_efectivo
                    )
                    / max(
                        CLEARANCE_OBJETIVO_DWA,
                        1e-8,
                    ),

                    0.0,
                    1.0,
                )
            )

        else:

            riesgo_clearance = 0.0

        penalizacion_seguridad = (
            PESO_SAC_SEGURIDAD
            * riesgo_clearance
        )

    # ------------------------------------------------------
    # 6. Penalización por alejarse de la ruta A*
    # ------------------------------------------------------

    error_ruta = distancia_punto_camino(
        punto_nuevo,
        camino_mundo,
    )

    error_ruta_normalizado = float(
        np.clip(
            error_ruta
            / max(
                ERROR_RUTA_OBJETIVO_DWA,
                1e-8,
            ),

            0.0,
            1.0,
        )
    )

    penalizacion_ruta = (
        PESO_SAC_RUTA
        * error_ruta_normalizado
    )

    # ------------------------------------------------------
    # 7. Penalización por cambios bruscos de control
    # ------------------------------------------------------

    cambio_v_normalizado = float(
        np.clip(
            abs(
                velocidad_lineal_nueva
                - velocidad_lineal_anterior
            )
            / max(
                VELOCIDAD_MAXIMA,
                1e-8,
            ),

            0.0,
            1.0,
        )
    )

    cambio_omega_normalizado = float(
        np.clip(
            abs(
                velocidad_angular_nueva
                - velocidad_angular_anterior
            )
            / max(
                2.0
                * VELOCIDAD_ANGULAR_MAXIMA,

                1e-8,
            ),

            0.0,
            1.0,
        )
    )

    cambio_control_normalizado = (
        cambio_v_normalizado
        + cambio_omega_normalizado
    ) / 2.0

    penalizacion_suavidad = (
        PESO_SAC_SUAVIDAD
        * cambio_control_normalizado
    )

    # ------------------------------------------------------
    # 8. Penalización por permanecer detenido
    # ------------------------------------------------------

    distancia_meta = distancia_meta_nueva

    robot_estancado = (
        velocidad_lineal_nueva
        < VELOCIDAD_ESTANCAMIENTO_DWA

        and distancia_meta
        > DISTANCIA_META
    )

    if robot_estancado:

        penalizacion_estancamiento = (
            PENALIZACION_SAC_ESTANCAMIENTO
        )

    else:

        penalizacion_estancamiento = 0.0

    # ------------------------------------------------------
    # 9. Penalización por estancamiento persistente
    # ------------------------------------------------------

    pasos_sin_progreso = int(
        pasos_sin_progreso
    )

    if pasos_sin_progreso < 0:

        raise ValueError(
            "Los pasos sin progreso no pueden ser negativos."
        )

    inicio_penalizacion_persistente = (
        PASOS_INICIO_PENALIZACION_ESTANCAMIENTO_SAC_REACTIVO
    )

    pasos_penalizacion_maxima = (
        PASOS_PENALIZACION_MAXIMA_ESTANCAMIENTO_SAC_REACTIVO
    )

    if (
        pasos_sin_progreso
        <= inicio_penalizacion_persistente
    ):

        intensidad_estancamiento_persistente = 0.0

    else:

        pasos_excedentes = (
            pasos_sin_progreso
            - inicio_penalizacion_persistente
        )

        intervalo_crecimiento = max(
            pasos_penalizacion_maxima
            - inicio_penalizacion_persistente,
            1,
        )

        intensidad_estancamiento_persistente = float(
            np.clip(
                pasos_excedentes
                / intervalo_crecimiento,
                0.0,
                1.0,
            )
        )

    penalizacion_estancamiento_persistente = (
        PENALIZACION_ESTANCAMIENTO_PERSISTENTE_MAXIMA_SAC
        * (
            intensidad_estancamiento_persistente
            ** 2
        )
    )

    recuperacion_estancamiento = bool(
        recuperacion_estancamiento
    )

    if recuperacion_estancamiento:

        recompensa_recuperacion_estancamiento = (
            RECOMPENSA_RECUPERACION_ESTANCAMIENTO_SAC
        )

    else:

        recompensa_recuperacion_estancamiento = 0.0

    # ------------------------------------------------------
    # 10. Recompensa terminal diferenciada
    # ------------------------------------------------------

    recompensa_terminal = 0.0

    if resultado == "meta":

        recompensa_terminal = (
            RECOMPENSA_SAC_META
        )

    elif resultado == "colision_estatica":

        recompensa_terminal = (
            RECOMPENSA_SAC_COLISION_ESTATICA
        )

    elif resultado == "colision_dinamica":

        recompensa_terminal = (
            RECOMPENSA_SAC_COLISION_DINAMICA
        )

    elif resultado == "fuera_mapa":

        recompensa_terminal = (
            RECOMPENSA_SAC_FUERA_MAPA
        )

    elif resultado == "timeout":

        recompensa_terminal = (
            RECOMPENSA_SAC_TIMEOUT
        )

    # ------------------------------------------------------
    # 11. Suma total
    # ------------------------------------------------------

    recompensa = (
        recompensa_progreso
        + recompensa_progreso_meta_global
        + recompensa_alineacion
        + recompensa_recuperacion_estancamiento
        - penalizacion_seguridad
        - penalizacion_ruta
        - penalizacion_suavidad
        - penalizacion_estancamiento
        - penalizacion_estancamiento_persistente
        - PENALIZACION_SAC_PASO
        + recompensa_terminal
    )

    if not np.isfinite(
        recompensa
    ):

        raise ValueError(
            "La recompensa SAC resultante no es finita."
        )

    # ------------------------------------------------------
    # 12. Componentes para diagnóstico
    # ------------------------------------------------------

    componentes = {
        "modo_clearance": modo_clearance,

        "progreso": progreso,

        "progreso_normalizado": (
            progreso_normalizado
        ),

        "recompensa_progreso": (
            recompensa_progreso
        ),

        "distancia_meta_anterior": (
            distancia_meta_anterior_efectiva
        ),

        "distancia_meta_nueva": (
            distancia_meta_nueva
        ),

        "progreso_meta_global": (
            progreso_meta_global
        ),

        "progreso_meta_global_normalizado": (
            progreso_meta_global_normalizado
        ),

        "recompensa_progreso_meta_global": (
            recompensa_progreso_meta_global
        ),

        "error_angular": error_angular,

        "alineacion": alineacion,

        "alineacion_con_movimiento": (
            alineacion_con_movimiento
        ),

        "recompensa_alineacion": (
            recompensa_alineacion
        ),

        "clearance_estatico": (
            clearance_estatico
        ),

        "clearance_dinamico": (
            clearance_dinamico
        ),

        "clearance_total": (
            clearance_total_efectivo
        ),

        "riesgo_clearance": (
            riesgo_clearance
        ),

        "riesgo_clearance_estatico": (
            riesgo_clearance_estatico
        ),

        "riesgo_clearance_dinamico": (
            riesgo_clearance_dinamico
        ),

        "penalizacion_seguridad_estatica": (
            penalizacion_seguridad_estatica
        ),

        "penalizacion_seguridad_dinamica": (
            penalizacion_seguridad_dinamica
        ),

        "penalizacion_velocidad_riesgosa_dinamica": (
            penalizacion_velocidad_riesgosa_dinamica
        ),

        "penalizacion_seguridad": (
            penalizacion_seguridad
        ),

        "velocidad_normalizada": (
            velocidad_normalizada
        ),

        "error_ruta": error_ruta,

        "error_ruta_normalizado": (
            error_ruta_normalizado
        ),

        "penalizacion_ruta": (
            penalizacion_ruta
        ),

        "cambio_v_normalizado": (
            cambio_v_normalizado
        ),

        "cambio_omega_normalizado": (
            cambio_omega_normalizado
        ),

        "penalizacion_suavidad": (
            penalizacion_suavidad
        ),

        "penalizacion_estancamiento": (
            penalizacion_estancamiento
        ),

        "pasos_sin_progreso": (
            pasos_sin_progreso
        ),

        "intensidad_estancamiento_persistente": (
            intensidad_estancamiento_persistente
        ),

        "penalizacion_estancamiento_persistente": (
            penalizacion_estancamiento_persistente
        ),

        "recuperacion_estancamiento": (
            recuperacion_estancamiento
        ),

        "recompensa_recuperacion_estancamiento": (
            recompensa_recuperacion_estancamiento
        ),

        "penalizacion_paso": (
            PENALIZACION_SAC_PASO
        ),

        "recompensa_terminal": (
            recompensa_terminal
        ),

        "recompensa_total": float(
            recompensa
        ),
    }

    return (
        float(
            recompensa
        ),

        componentes,
    )
def ejecutar_paso_entorno_sac(
    entorno,
    accion,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
):

    # ------------------------------------------------------
    # 1. Impedir acciones después de terminar el episodio
    # ------------------------------------------------------

    if entorno["terminado"]:

        raise RuntimeError(
            "El episodio SAC ya terminó. "
            "Debes reiniciar el entorno."
        )

    estado_actual = entorno[
        "estado_robot"
    ]
    velocidad_lineal_anterior = entorno[
        "velocidad_lineal_actual"
    ]

    velocidad_angular_anterior = entorno[
        "velocidad_angular_actual"
    ]
    obstaculos_estaticos = entorno[
        "obstaculos_estaticos"
    ]

    obstaculos_dinamicos_actuales = entorno[
        "obstaculos_dinamicos"
    ]

    submeta_usada = entorno[
        "submeta"
    ]

    # ------------------------------------------------------
    # 2. Convertir la acción normalizada en control
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 2. Convertir acción en control deseado
    # ------------------------------------------------------

    (
        velocidad_lineal_deseada,
        velocidad_angular_deseada,
    ) = convertir_accion_sac_a_control(
        accion
    )

    # ------------------------------------------------------
    # 2.1. Distancia actual a la meta
    # ------------------------------------------------------

    distancia_meta_actual = distancia_entre_puntos(
        (
            estado_actual[
                0
            ],
            estado_actual[
                1
            ],
        ),

        entorno[
            "meta"
        ],
    )

    # ------------------------------------------------------
    # 2.2. Aplicar dinámica física y frenado terminal
    # ------------------------------------------------------

    (
        velocidad_lineal,
        velocidad_angular,
        informacion_control_reactivo,
    ) = aplicar_dinamica_control_sac_reactivo(
        velocidad_lineal_deseada=(
            velocidad_lineal_deseada
        ),

        velocidad_angular_deseada=(
            velocidad_angular_deseada
        ),

        velocidad_lineal_actual=(
            velocidad_lineal_anterior
        ),

        velocidad_angular_actual=(
            velocidad_angular_anterior
        ),

        distancia_meta=(
            distancia_meta_actual
        ),

        dt=dt,
    )

    # ------------------------------------------------------
    # 3. Avanzar el contador temporal
    # ------------------------------------------------------

    paso_nuevo = (
        entorno["paso_actual"]
        + 1
    )

    tiempo_nuevo = (
        paso_nuevo
        * dt
    )

    # ------------------------------------------------------
    # 4. Actualizar el estado del robot
    # ------------------------------------------------------

    estado_nuevo = actualizar_estado_robot(
        estado=estado_actual,
        velocidad_lineal=velocidad_lineal,
        velocidad_angular=velocidad_angular,
        dt=dt,
    )

    # ------------------------------------------------------
    # 4.1. Actualizar progreso hacia la meta global
    # ------------------------------------------------------

    distancia_meta_anterior = float(
        entorno.get(
            "distancia_meta_anterior",
            distancia_meta_actual,
        )
    )

    distancia_meta_nueva = distancia_entre_puntos(
        (
            estado_nuevo[
                0
            ],
            estado_nuevo[
                1
            ],
        ),
        entorno[
            "meta"
        ],
    )

    mejor_distancia_meta_anterior = float(
        entorno.get(
            "mejor_distancia_meta",
            distancia_meta_anterior,
        )
    )

    pasos_sin_progreso_anterior = int(
        entorno.get(
            "pasos_sin_progreso",
            0,
        )
    )

    estado_estancamiento = (
        actualizar_estado_estancamiento_sac_reactivo(
            distancia_meta_anterior=(
                distancia_meta_anterior
            ),
            distancia_meta_nueva=(
                distancia_meta_nueva
            ),
            mejor_distancia_meta_anterior=(
                mejor_distancia_meta_anterior
            ),
            pasos_sin_progreso_anterior=(
                pasos_sin_progreso_anterior
            ),
        )
    )

    # ------------------------------------------------------
    # 5. Actualizar los obstáculos dinámicos
    # ------------------------------------------------------

    obstaculos_dinamicos_nuevos = (
        actualizar_obstaculos_dinamicos(
            obstaculos_dinamicos=(
                obstaculos_dinamicos_actuales
            ),
            obstaculos_estaticos=(
                obstaculos_estaticos
            ),
            tiempo_actual=tiempo_nuevo,
            dt=dt,
        )
    )

    # ------------------------------------------------------
    # 6. Calcular clearances
    # ------------------------------------------------------

    clearance_estatico = calcular_clearance_estatico(
        estado_nuevo,
        obstaculos_estaticos,
    )

    clearance_dinamico = calcular_clearance_dinamico(
        estado_nuevo,
        obstaculos_dinamicos_nuevos,
    )

    clearance_total = min(
        clearance_estatico,
        clearance_dinamico,
    )

    # ------------------------------------------------------
    # 7. Determinar si terminó el episodio
    # ------------------------------------------------------

    (
        episodio_finalizado,
        resultado,
    ) = determinar_resultado_episodio(
        estado_robot=estado_nuevo,
        meta=entorno["meta"],
        clearance_estatico=clearance_estatico,
        clearance_dinamico=clearance_dinamico,
        paso_actual=paso_nuevo,
        pasos_maximos=pasos_maximos,
    )

    # Para aprendizaje por refuerzo conviene distinguir:
    #
    # terminado: meta, colisión o salida del mapa.
    # truncado: límite máximo de pasos.

    terminado = (
        episodio_finalizado
        and resultado != "timeout"
    )

    truncado = (
        resultado == "timeout"
    )

    # ------------------------------------------------------
    # 8. Registrar la transición
    # ------------------------------------------------------

    registro = registrar_paso_simulacion(
        registro=entorno["registro"],
        estado_nuevo=estado_nuevo,
        velocidad_lineal=velocidad_lineal,
        velocidad_angular=velocidad_angular,
        clearance_estatico=clearance_estatico,
        clearance_dinamico=clearance_dinamico,
        submeta=submeta_usada,
        obstaculos_dinamicos=(
            obstaculos_dinamicos_nuevos
        ),
    )

    registro["resultado"] = resultado

    # ------------------------------------------------------
    # 9. Seleccionar la submeta para el siguiente paso
    # ------------------------------------------------------

    (
        submeta_nueva,
        indice_progreso_nuevo,
        indice_submeta_nuevo,
    ) = seleccionar_submeta_local_estable(
        estado_robot=estado_nuevo,
        camino_mundo=entorno[
            "camino_mundo"
        ],
        indice_progreso_anterior=entorno[
            "indice_progreso"
        ],
    )

    # ------------------------------------------------------
    # 10. Construir la siguiente observación SAC
    # ------------------------------------------------------

    observacion_nueva = construir_observacion_sac(
        estado_robot=estado_nuevo,

        submeta=submeta_nueva,

        meta=entorno[
            "meta"
        ],

        obstaculos_estaticos=entorno[
            "obstaculos_estaticos"
        ],

        obstaculos_dinamicos=(
            obstaculos_dinamicos_nuevos
        ),

        velocidad_lineal_actual=(
            velocidad_lineal
        ),

        velocidad_angular_actual=(
            velocidad_angular
        ),

        indice_progreso=(
            indice_progreso_nuevo
        ),

        numero_puntos_camino=len(
            entorno[
                "camino_mundo"
            ]
        ),
    )

    # Esta información se conserva para visualización
    # y diagnóstico, pero no forma parte directamente
    # de la observación de la red neuronal.

    obstaculos_observados = (
        obtener_obstaculos_dinamicos_observados(
            estado_robot=estado_nuevo,

            obstaculos_dinamicos=(
                obstaculos_dinamicos_nuevos
            ),
        )
    )
    # ------------------------------------------------------
    # 11. Validar la nueva observación SAC
    # ------------------------------------------------------

    if not isinstance(
        observacion_nueva,
        dict,
    ):

        raise TypeError(
            "La nueva observación SAC debe ser "
            "un diccionario."
        )

    if (
        "parche" not in observacion_nueva
        or "escalares" not in observacion_nueva
    ):

        raise KeyError(
            "La nueva observación SAC debe contener "
            "las claves 'parche' y 'escalares'."
        )

    parche_nuevo = observacion_nueva[
        "parche"
    ]

    escalares_nuevos = observacion_nueva[
        "escalares"
    ]

    forma_parche_esperada = (
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if parche_nuevo.shape != forma_parche_esperada:

        raise ValueError(
            "El nuevo parche SAC tiene forma "
            f"{parche_nuevo.shape}; se esperaba "
            f"{forma_parche_esperada}."
        )

    if escalares_nuevos.shape != (
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "Los nuevos escalares SAC tienen forma "
            f"{escalares_nuevos.shape}; se esperaba "
            f"({DIMENSION_ESCALARES_SAC},)."
        )

    if parche_nuevo.dtype != np.float32:

        raise TypeError(
            "El nuevo parche SAC debe tener "
            "tipo np.float32."
        )

    if escalares_nuevos.dtype != np.float32:

        raise TypeError(
            "Los nuevos escalares SAC deben tener "
            "tipo np.float32."
        )

    if not np.all(
        np.isfinite(
            parche_nuevo
        )
    ):

        raise ValueError(
            "El nuevo parche SAC contiene NaN "
            "o valores infinitos."
        )

    if not np.all(
        np.isfinite(
            escalares_nuevos
        )
    ):

        raise ValueError(
            "Los nuevos escalares SAC contienen NaN "
            "o valores infinitos."
        )
    # ------------------------------------------------------
    # 12. Actualizar el estado interno
    # ------------------------------------------------------

    entorno["estado_robot"] = estado_nuevo

    entorno["obstaculos_dinamicos"] = [
        obstaculo.copy()
        for obstaculo
        in obstaculos_dinamicos_nuevos
    ]

    entorno["submeta"] = submeta_nueva

    entorno["indice_progreso"] = (
        indice_progreso_nuevo
    )

    entorno["indice_submeta"] = (
        indice_submeta_nuevo
    )

    entorno["velocidad_lineal_actual"] = (
        velocidad_lineal
    )

    entorno["velocidad_angular_actual"] = (
        velocidad_angular
    )

    entorno["paso_actual"] = paso_nuevo

    entorno["tiempo_actual"] = tiempo_nuevo

    entorno["terminado"] = episodio_finalizado

    entorno["resultado"] = resultado

    entorno["observacion"] = {
        "parche": observacion_nueva[
            "parche"
        ].copy(),

        "escalares": observacion_nueva[
            "escalares"
        ].copy(),
    }
    entorno["obstaculos_observados"] = [
        descripcion.copy()
        for descripcion
        in obstaculos_observados
    ]

    entorno["registro"] = registro

    # ------------------------------------------------------
    # 13. Calcular recompensa SAC
    # ------------------------------------------------------

    (
        recompensa,
        componentes_recompensa,
    ) = calcular_recompensa_sac(
        estado_anterior=estado_actual,
        estado_nuevo=estado_nuevo,
        submeta=submeta_usada,
        meta=entorno["meta"],
        camino_mundo=entorno[
            "camino_mundo"
        ],
        velocidad_lineal_anterior=(
            velocidad_lineal_anterior
        ),

        velocidad_angular_anterior=(
            velocidad_angular_anterior
        ),
        velocidad_lineal_nueva=(
            velocidad_lineal
        ),
        velocidad_angular_nueva=(
            velocidad_angular
        ),
        clearance_total=clearance_total,
        clearance_estatico=clearance_estatico,
        clearance_dinamico=clearance_dinamico,
        distancia_meta_anterior=(
            distancia_meta_anterior
        ),
        pasos_sin_progreso=(
            estado_estancamiento[
                "pasos_sin_progreso_nuevos"
            ]
        ),
        recuperacion_estancamiento=(
            estado_estancamiento[
                "recuperacion_estancamiento"
            ]
        ),
        resultado=resultado,
        dt=dt,
    )

    # ------------------------------------------------------
    # 14. Registrar progreso global y estancamiento
    # ------------------------------------------------------

    registro[
        "distancias_meta_globales"
    ].append(
        float(
            distancia_meta_nueva
        )
    )

    registro[
        "progresos_meta_globales"
    ].append(
        float(
            estado_estancamiento[
                "progreso_meta_global"
            ]
        )
    )

    registro[
        "mejores_distancias_meta"
    ].append(
        float(
            estado_estancamiento[
                "mejor_distancia_meta_nueva"
            ]
        )
    )

    registro[
        "pasos_sin_progreso"
    ].append(
        int(
            estado_estancamiento[
                "pasos_sin_progreso_nuevos"
            ]
        )
    )

    registro[
        "intensidades_estancamiento"
    ].append(
        float(
            estado_estancamiento[
                "intensidad_estancamiento"
            ]
        )
    )

    registro[
        "penalizaciones_estancamiento_persistente"
    ].append(
        float(
            componentes_recompensa[
                "penalizacion_estancamiento_persistente"
            ]
        )
    )

    registro[
        "recuperaciones_estancamiento"
    ].append(
        int(
            bool(
                estado_estancamiento[
                    "recuperacion_estancamiento"
                ]
            )
        )
    )

    entorno[
        "distancia_meta_anterior"
    ] = float(
        distancia_meta_nueva
    )

    entorno[
        "mejor_distancia_meta"
    ] = float(
        estado_estancamiento[
            "mejor_distancia_meta_nueva"
        ]
    )

    entorno[
        "pasos_sin_progreso"
    ] = int(
        estado_estancamiento[
            "pasos_sin_progreso_nuevos"
        ]
    )

    if estado_estancamiento[
        "recuperacion_estancamiento"
    ]:

        entorno[
            "numero_recuperaciones_estancamiento"
        ] = (
            int(
                entorno.get(
                    "numero_recuperaciones_estancamiento",
                    0,
                )
            )
            + 1
        )

    entorno[
        "registro"
    ] = registro

    informacion = {
        "resultado": resultado,

        "paso": paso_nuevo,

        "tiempo": tiempo_nuevo,
        "velocidad_lineal_deseada": (
            velocidad_lineal_deseada
        ),

        "velocidad_angular_deseada": (
            velocidad_angular_deseada
        ),

        "control_reactivo": (
            informacion_control_reactivo.copy()
        ),
        "velocidad_lineal": (
            velocidad_lineal
        ),

        "velocidad_angular": (
            velocidad_angular
        ),

        "clearance_estatico": (
            clearance_estatico
        ),

        "clearance_dinamico": (
            clearance_dinamico
        ),

        "clearance_total": (
            clearance_total
        ),

        "distancia_meta_anterior": (
            distancia_meta_anterior
        ),

        "distancia_meta_nueva": (
            distancia_meta_nueva
        ),

        "progreso_meta_global": (
            estado_estancamiento[
                "progreso_meta_global"
            ]
        ),

        "mejor_distancia_meta": (
            estado_estancamiento[
                "mejor_distancia_meta_nueva"
            ]
        ),

        "pasos_sin_progreso": (
            estado_estancamiento[
                "pasos_sin_progreso_nuevos"
            ]
        ),

        "intensidad_estancamiento": (
            estado_estancamiento[
                "intensidad_estancamiento"
            ]
        ),

        "recuperacion_estancamiento": (
            estado_estancamiento[
                "recuperacion_estancamiento"
            ]
        ),

        "penalizacion_estancamiento_persistente": (
            componentes_recompensa[
                "penalizacion_estancamiento_persistente"
            ]
        ),

        "submeta_usada": submeta_usada,

        "submeta_nueva": submeta_nueva,

        "indice_progreso": (
            indice_progreso_nuevo
        ),

        "terminado": terminado,

        "truncado": truncado,
        "componentes_recompensa": (
            componentes_recompensa
        ),

        "recompensa": recompensa,
    }

    observacion_salida = {
        "parche": observacion_nueva[
            "parche"
        ].copy(),

        "escalares": observacion_nueva[
            "escalares"
        ].copy(),
    }

    return (
        observacion_salida,
        recompensa,
        terminado,
        truncado,
        informacion,
    )
def ejecutar_episodio_sac_prueba(
    semilla,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
):

    # ------------------------------------------------------
    # 1. Reiniciar el entorno
    # ------------------------------------------------------

    (
        entorno,
        observacion,
    ) = reiniciar_entorno_sac(
        semilla=semilla
    )

    acciones = []
    recompensas = []
    informaciones = []

    recompensa_acumulada = 0.0

    terminado = False
    truncado = False

    # ------------------------------------------------------
    # 2. Ejecutar transiciones
    # ------------------------------------------------------

    while not (
        terminado
        or truncado
    ):

        (
            accion,
            informacion_politica,
        ) = seleccionar_accion_politica_prueba_sac(
            entorno=entorno,
            observacion=observacion,
        )

        (
            observacion_nueva,
            recompensa,
            terminado,
            truncado,
            informacion_paso,
        ) = ejecutar_paso_entorno_sac(
            entorno=entorno,
            accion=accion,
            pasos_maximos=pasos_maximos,
            dt=dt,
        )

        acciones.append(
            accion.copy()
        )

        recompensas.append(
            float(
                recompensa
            )
        )

        informacion_completa = (
            informacion_paso.copy()
        )

        informacion_completa[
            "politica"
        ] = informacion_politica

        informaciones.append(
            informacion_completa
        )

        recompensa_acumulada += (
            recompensa
        )

        observacion = (
            observacion_nueva
        )

    # ------------------------------------------------------
    # 3. Obtener registro y métricas
    # ------------------------------------------------------

    registro = entorno[
        "registro"
    ]

    escenario = entorno[
        "escenario"
    ]

    metricas = calcular_metricas_episodio_estandar(
        registro=registro,
        camino_mundo=entorno[
            "camino_mundo"
        ],
        meta=entorno[
            "meta"
        ],
        longitud_camino_astar=escenario[
            "longitud_camino"
        ],
        dt=dt,
    )

    metricas[
        "metodo"
    ] = "sac_prueba"

    metricas[
        "semilla"
    ] = int(
        semilla
    )

    metricas[
        "recompensa_acumulada"
    ] = float(
        recompensa_acumulada
    )

    metricas[
        "recompensa_promedio"
    ] = float(
        np.mean(
            recompensas
        )
    ) if len(
        recompensas
    ) > 0 else 0.0

    # ------------------------------------------------------
    # 4. Reunir el resultado completo
    # ------------------------------------------------------

    resultado_episodio = {
        "entorno": entorno,

        "escenario": escenario,

        "registro": registro,

        "acciones": acciones,

        "recompensas": recompensas,

        "informaciones": informaciones,

        "observacion_final": (
            observacion.copy()
        ),

        "terminado": terminado,

        "truncado": truncado,

        "metricas": metricas,
    }

    return resultado_episodio
def ejecutar_pruebas_integradas():

    print("\n" + "=" * 68)
    print("PRUEBAS INTEGRADAS DEL REGISTRO Y LAS MÉTRICAS")
    print("=" * 68)

    resultados_pruebas = []

    def verificar(
        nombre,
        condicion,
        obtenido=None,
        esperado=None,
    ):

        resultados_pruebas.append(
            bool(condicion)
        )

        if condicion:

            print(
                "[OK]",
                nombre,
            )

        else:

            print(
                "[ERROR]",
                nombre,
            )

            print(
                "        obtenido:",
                obtenido,
            )

            print(
                "        esperado:",
                esperado,
            )

    # ======================================================
    # 1. REGISTRO DE SIMULACIÓN
    # ======================================================

    registro = crear_registro_simulacion(
        (
            0.0,
            0.0,
            0.0,
        )
    )

    obstaculos_dinamicos = [
        {
            "x": 5.0,
            "y": 6.0,
            "vx": 0.2,
            "vy": -0.1,
            "radio": 0.30,
        }
    ]

    resultado_registro = registrar_paso_simulacion(
        registro=registro,
        estado_nuevo=(
            3.0,
            0.0,
            0.0,
        ),
        velocidad_lineal=0.8,
        velocidad_angular=0.2,
        clearance_estatico=1.2,
        clearance_dinamico=float("inf"),
        submeta=(
            4.0,
            0.0,
        ),
        obstaculos_dinamicos=obstaculos_dinamicos,
    )

    registrar_paso_simulacion(
        registro=registro,
        estado_nuevo=(
            3.0,
            4.0,
            1.57,
        ),
        velocidad_lineal=0.7,
        velocidad_angular=0.1,
        clearance_estatico=0.7,
        clearance_dinamico=0.4,
        submeta=(
            3.0,
            5.0,
        ),
        obstaculos_dinamicos=obstaculos_dinamicos,
    )

    verificar(
        "registrar_paso_simulacion devuelve el registro",
        resultado_registro is registro,
        resultado_registro,
        "el mismo diccionario registro",
    )

    verificar(
        "cantidad de estados, controles y pasos",
        (
            len(registro["estados"]) == 3
            and len(registro["controles"]) == 2
            and registro["pasos_ejecutados"] == 2
        ),
        (
            len(registro["estados"]),
            len(registro["controles"]),
            registro["pasos_ejecutados"],
        ),
        (
            3,
            2,
            2,
        ),
    )

    # Comprobar que el historial tiene una copia independiente.

    obstaculos_dinamicos[0]["x"] = 100.0

    posicion_guardada = registro[
        "obstaculos_dinamicos"
    ][0][0]["x"]

    verificar(
        "la copia de obstáculos es independiente",
        math.isclose(
            posicion_guardada,
            5.0,
            abs_tol=1e-9,
        ),
        posicion_guardada,
        5.0,
    )

    registro_consistente = verificar_registro_clearances(
        registro
    )

    verificar(
        "registro de clearances consistente",
        registro_consistente,
        registro_consistente,
        True,
    )

    # ======================================================
    # 2. CLEARANCES GEOMÉTRICOS
    # ======================================================

    obstaculo_estatico = {
        "x_min": 4.0,
        "x_max": 6.0,
        "y_min": 3.0,
        "y_max": 7.0,
    }

    obstaculo_dinamico = {
        "x": 5.0,
        "y": 5.0,
        "vx": 0.0,
        "vy": 0.0,
        "radio": 0.30,
    }

    estado_clearance = (
        3.0,
        5.0,
        0.0,
    )

    (
        clearance_total,
        clearance_estatico,
        clearance_dinamico,
    ) = calcular_clearance_total(
        estado_clearance,
        [
            obstaculo_estatico
        ],
        [
            obstaculo_dinamico
        ],
    )

    verificar(
        "clearance estático",
        math.isclose(
            clearance_estatico,
            0.6,
            abs_tol=1e-9,
        ),
        clearance_estatico,
        0.6,
    )

    verificar(
        "clearance dinámico",
        math.isclose(
            clearance_dinamico,
            1.3,
            abs_tol=1e-9,
        ),
        clearance_dinamico,
        1.3,
    )

    verificar(
        "clearance total",
        math.isclose(
            clearance_total,
            0.6,
            abs_tol=1e-9,
        ),
        clearance_total,
        0.6,
    )

    # ======================================================
    # 3. TERMINACIÓN DEL EPISODIO
    # ======================================================

    terminado, resultado = determinar_resultado_episodio(
        estado_robot=(
            9.8,
            10.0,
            0.0,
        ),
        meta=(
            10.0,
            10.0,
        ),
        clearance_estatico=1.0,
        clearance_dinamico=1.0,
        paso_actual=20,
        pasos_maximos=400,
    )

    verificar(
        "detección de llegada a la meta",
        (
            terminado is True
            and resultado == "meta"
        ),
        (
            terminado,
            resultado,
        ),
        (
            True,
            "meta",
        ),
    )

    terminado, resultado = determinar_resultado_episodio(
        estado_robot=(
            3.0,
            5.0,
            0.0,
        ),
        meta=(
            10.0,
            10.0,
        ),
        clearance_estatico=1.0,
        clearance_dinamico=-0.05,
        paso_actual=20,
        pasos_maximos=400,
    )

    verificar(
        "detección de colisión dinámica",
        (
            terminado is True
            and resultado == "colision_dinamica"
        ),
        (
            terminado,
            resultado,
        ),
        (
            True,
            "colision_dinamica",
        ),
    )

    # ======================================================
    # 4. MÉTRICAS DE NAVEGACIÓN
    # ======================================================

    registro["resultado"] = "meta"

    tiempo = calcular_tiempo_navegacion(
        registro
    )

    longitud = calcular_longitud_recorrida(
        registro
    )

    distancia_final = calcular_distancia_final_meta(
        registro,
        (
            6.0,
            8.0,
        ),
    )

    eficiencia = calcular_eficiencia_navegacion(
        registro,
        longitud_camino_astar=5.0,
    )

    exceso = calcular_exceso_longitud_porcentual(
        registro,
        longitud_camino_astar=5.0,
    )

    verificar(
        "tiempo de navegación",
        math.isclose(
            tiempo,
            0.2,
            abs_tol=1e-9,
        ),
        tiempo,
        0.2,
    )

    verificar(
        "longitud recorrida",
        math.isclose(
            longitud,
            7.0,
            abs_tol=1e-9,
        ),
        longitud,
        7.0,
    )

    verificar(
        "distancia final a la meta",
        math.isclose(
            distancia_final,
            5.0,
            abs_tol=1e-9,
        ),
        distancia_final,
        5.0,
    )

    verificar(
        "eficiencia de navegación",
        math.isclose(
            eficiencia,
            5.0 / 7.0,
            abs_tol=1e-9,
        ),
        eficiencia,
        5.0 / 7.0,
    )

    verificar(
        "exceso porcentual de longitud",
        math.isclose(
            exceso,
            40.0,
            abs_tol=1e-9,
        ),
        exceso,
        40.0,
    )

    # ======================================================
    # 5. MÉTRICAS DE CLEARANCE DEL EPISODIO
    # ======================================================

    (
        estatico_minimo,
        estatico_promedio,
    ) = calcular_metricas_clearance_estatico(
        registro
    )

    (
        dinamico_minimo,
        dinamico_promedio,
    ) = calcular_metricas_clearance_dinamico(
        registro
    )

    (
        total_minimo,
        total_promedio,
    ) = calcular_metricas_clearance_total(
        registro
    )

    verificar(
        "métricas de clearance estático",
        (
            math.isclose(
                estatico_minimo,
                0.7,
                abs_tol=1e-9,
            )
            and math.isclose(
                estatico_promedio,
                0.95,
                abs_tol=1e-9,
            )
        ),
        (
            estatico_minimo,
            estatico_promedio,
        ),
        (
            0.7,
            0.95,
        ),
    )

    verificar(
        "métricas de clearance dinámico",
        (
            math.isclose(
                dinamico_minimo,
                0.4,
                abs_tol=1e-9,
            )
            and math.isclose(
                dinamico_promedio,
                0.4,
                abs_tol=1e-9,
            )
        ),
        (
            dinamico_minimo,
            dinamico_promedio,
        ),
        (
            0.4,
            0.4,
        ),
    )

    verificar(
        "métricas de clearance total",
        (
            math.isclose(
                total_minimo,
                0.4,
                abs_tol=1e-9,
            )
            and math.isclose(
                total_promedio,
                0.8,
                abs_tol=1e-9,
            )
        ),
        (
            total_minimo,
            total_promedio,
        ),
        (
            0.4,
            0.8,
        ),
    )

    # ======================================================
    # 6. ERROR RESPECTO DE LA RUTA A*
    # ======================================================

    registro_error = {
        "estados": [
            (0.0, 0.0, 0.0),
            (2.0, 0.5, 0.0),
            (4.0, 1.0, 0.0),
            (6.0, 0.5, 0.0),
            (8.0, 0.0, 0.0),
        ]
    }

    camino_recto = [
        (0.0, 0.0),
        (10.0, 0.0),
    ]

    errores = calcular_errores_ruta(
        registro_error,
        camino_recto,
    )

    error_medio = calcular_error_medio_ruta(
        registro_error,
        camino_recto,
    )

    error_rmse = calcular_error_rmse_ruta(
        registro_error,
        camino_recto,
    )

    error_maximo = calcular_error_maximo_ruta(
        registro_error,
        camino_recto,
    )

    verificar(
        "lista de errores respecto de la ruta",
        np.allclose(
            errores,
            [
                0.0,
                0.5,
                1.0,
                0.5,
                0.0,
            ],
            atol=1e-9,
        ),
        errores,
        [
            0.0,
            0.5,
            1.0,
            0.5,
            0.0,
        ],
    )

    verificar(
        "error medio, RMSE y máximo",
        (
            math.isclose(
                error_medio,
                0.4,
                abs_tol=1e-9,
            )
            and math.isclose(
                error_rmse,
                math.sqrt(0.3),
                abs_tol=1e-9,
            )
            and math.isclose(
                error_maximo,
                1.0,
                abs_tol=1e-9,
            )
        ),
        (
            error_medio,
            error_rmse,
            error_maximo,
        ),
        (
            0.4,
            math.sqrt(0.3),
            1.0,
        ),
    )
    # ======================================================
    # 7. SUAVIDAD Y ESFUERZO DEL CONTROL
    # ======================================================

    registro_control = {
        "controles": [
            (
                0.2,
                0.0,
            ),
            (
                0.5,
                0.3,
            ),
            (
                0.4,
                -0.3,
            ),
        ]
    }

    metricas_control = calcular_metricas_suavidad_control(
        registro_control,
        dt=0.1,
    )

    variacion_total_v = metricas_control[
        "variacion_total_v"
    ]

    variacion_total_omega = metricas_control[
        "variacion_total_omega"
    ]

    aceleracion_lineal_rms = metricas_control[
        "aceleracion_lineal_rms"
    ]

    aceleracion_angular_rms = metricas_control[
        "aceleracion_angular_rms"
    ]

    esfuerzo_control = metricas_control[
        "esfuerzo_control"
    ]

    verificar(
        "variación total de velocidad lineal",
        math.isclose(
            variacion_total_v,
            0.4,
            abs_tol=1e-9,
        ),
        variacion_total_v,
        0.4,
    )

    verificar(
        "variación total de velocidad angular",
        math.isclose(
            variacion_total_omega,
            0.9,
            abs_tol=1e-9,
        ),
        variacion_total_omega,
        0.9,
    )

    verificar(
        "aceleración lineal RMS",
        math.isclose(
            aceleracion_lineal_rms,
            math.sqrt(5.0),
            abs_tol=1e-9,
        ),
        aceleracion_lineal_rms,
        math.sqrt(5.0),
    )

    verificar(
        "aceleración angular RMS",
        math.isclose(
            aceleracion_angular_rms,
            math.sqrt(22.5),
            abs_tol=1e-9,
        ),
        aceleracion_angular_rms,
        math.sqrt(22.5),
    )

    verificar(
        "esfuerzo de control normalizado",
        math.isclose(
            esfuerzo_control,
            0.0575,
            abs_tol=1e-9,
        ),
        esfuerzo_control,
        0.0575,
    )
    # ======================================================
    # 8. RESUMEN ESTÁNDAR DE MÉTRICAS
    # ======================================================

    camino_prueba_metricas = [
        (
            0.0,
            0.0,
        ),
        (
            3.0,
            0.0,
        ),
        (
            3.0,
            4.0,
        ),
    ]

    meta_prueba_metricas = (
        6.0,
        8.0,
    )

    registro["resultado"] = "meta"

    metricas_episodio = (
        calcular_metricas_episodio_estandar(
            registro=registro,
            camino_mundo=camino_prueba_metricas,
            meta=meta_prueba_metricas,
            longitud_camino_astar=5.0,
            dt=0.1,
        )
    )

    claves_obligatorias = [
        "resultado",
        "exito",
        "pasos_ejecutados",
        "tiempo_navegacion",
        "longitud_recorrida",
        "longitud_astar",
        "distancia_final_meta",
        "eficiencia_navegacion",
        "exceso_longitud_porcentual",
        "error_medio_ruta",
        "error_rmse_ruta",
        "error_maximo_ruta",
        "clearance_estatico_minimo",
        "clearance_dinamico_minimo",
        "clearance_total_minimo",
        "variacion_total_v",
        "variacion_total_omega",
        "aceleracion_lineal_rms",
        "aceleracion_angular_rms",
        "esfuerzo_control",
        "registro_consistente",
    ]

    contiene_todas_las_claves = all(
        clave in metricas_episodio
        for clave in claves_obligatorias
    )

    valores_generales_correctos = (
        metricas_episodio["resultado"] == "meta"
        and metricas_episodio["exito"] is True
        and metricas_episodio["pasos_ejecutados"] == 2
        and math.isclose(
            metricas_episodio[
                "tiempo_navegacion"
            ],
            0.2,
            abs_tol=1e-9,
        )
        and math.isclose(
            metricas_episodio[
                "longitud_recorrida"
            ],
            7.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            metricas_episodio[
                "distancia_final_meta"
            ],
            5.0,
            abs_tol=1e-9,
        )
    )

    errores_ruta_correctos = (
        math.isclose(
            metricas_episodio[
                "error_medio_ruta"
            ],
            0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            metricas_episodio[
                "error_rmse_ruta"
            ],
            0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            metricas_episodio[
                "error_maximo_ruta"
            ],
            0.0,
            abs_tol=1e-9,
        )
    )

    verificar(
        "diccionario estándar de métricas",
        (
            contiene_todas_las_claves
            and valores_generales_correctos
            and errores_ruta_correctos
            and metricas_episodio[
                "registro_consistente"
            ] is True
        ),
        metricas_episodio,
        "diccionario completo y valores correctos",
    )
    # ======================================================
    # 9. CONVERSIÓN DE DWA AL REGISTRO ESTÁNDAR
    # ======================================================

    simulacion_dwa_prueba = {
        "estados": [
            (
                2.0,
                2.0,
                0.0,
            ),
            (
                2.1,
                2.0,
                0.0,
            ),
            (
                2.2,
                2.0,
                0.0,
            ),
        ],

        "controles": [
            (
                1.0,
                0.0,
            ),
            (
                1.0,
                0.0,
            ),
        ],

        "resultado": "meta",

        "pasos_ejecutados": 2,

        "historial_dinamicos": [
            [],
            [],
            [],
        ],

        "historial_submetas": [
            (
                3.0,
                2.0,
            ),
            (
                3.0,
                2.0,
            ),
        ],
    }

    registro_dwa_prueba = (
        convertir_simulacion_dwa_a_registro(
            simulacion_dwa=simulacion_dwa_prueba,
            obstaculos_estaticos=[],
        )
    )

    conversion_dwa_correcta = (
        registro_dwa_prueba["resultado"] == "meta"
        and registro_dwa_prueba[
            "pasos_ejecutados"
        ] == 2
        and len(
            registro_dwa_prueba["estados"]
        ) == 3
        and len(
            registro_dwa_prueba["controles"]
        ) == 2
        and len(
            registro_dwa_prueba[
                "clearances_estaticos"
            ]
        ) == 2
        and len(
            registro_dwa_prueba[
                "clearances_dinamicos"
            ]
        ) == 2
        and verificar_registro_clearances(
            registro_dwa_prueba
        )
    )

    verificar(
        "conversión de DWA al registro estándar",
        conversion_dwa_correcta,
        registro_dwa_prueba,
        "registro DWA consistente con dos pasos",
    )
    # ======================================================
    # 10. EJECUCIÓN REAL DE DWA CON FORMATO ESTÁNDAR
    # ======================================================

    # Se usan solamente 30 pasos para comprobar la conexión
    # entre funciones sin ejecutar todavía un experimento
    # completo de 400 pasos.

    episodio_dwa_real = ejecutar_episodio_dwa_estandar(
        semilla=SEMILLA,
        pasos_maximos=30,
        dt=DT,
    )

    escenario_dwa_real = episodio_dwa_real[
        "escenario"
    ]

    simulacion_dwa_real = episodio_dwa_real[
        "simulacion"
    ]

    registro_dwa_real = episodio_dwa_real[
        "registro"
    ]

    metricas_dwa_real = episodio_dwa_real[
        "metricas"
    ]

    estructura_dwa_correcta = (
        escenario_dwa_real is not None

        and metricas_dwa_real[
            "metodo"
        ] == "dwa"

        and metricas_dwa_real[
            "semilla"
        ] == SEMILLA

        and registro_dwa_real[
            "resultado"
        ] == simulacion_dwa_real[
            "resultado"
        ]

        and metricas_dwa_real[
            "resultado"
        ] == simulacion_dwa_real[
            "resultado"
        ]

        and len(
            registro_dwa_real[
                "estados"
            ]
        ) == (
            registro_dwa_real[
                "pasos_ejecutados"
            ]
            + 1
        )

        and len(
            registro_dwa_real[
                "controles"
            ]
        ) == registro_dwa_real[
            "pasos_ejecutados"
        ]

        and verificar_registro_clearances(
            registro_dwa_real
        )
    )

    metricas_dwa_finitas = (
        np.isfinite(
            metricas_dwa_real[
                "tiempo_navegacion"
            ]
        )

        and np.isfinite(
            metricas_dwa_real[
                "longitud_recorrida"
            ]
        )

        and np.isfinite(
            metricas_dwa_real[
                "distancia_final_meta"
            ]
        )

        and np.isfinite(
            metricas_dwa_real[
                "error_medio_ruta"
            ]
        )

        and np.isfinite(
            metricas_dwa_real[
                "error_rmse_ruta"
            ]
        )

        and np.isfinite(
            metricas_dwa_real[
                "clearance_estatico_minimo"
            ]
        )
    )

    verificar(
        "ejecución real de DWA con métricas estándar",
        (
            estructura_dwa_correcta
            and metricas_dwa_finitas
        ),
        {
            "resultado": metricas_dwa_real[
                "resultado"
            ],

            "pasos": metricas_dwa_real[
                "pasos_ejecutados"
            ],

            "tiempo": metricas_dwa_real[
                "tiempo_navegacion"
            ],

            "registro_consistente": (
                metricas_dwa_real[
                    "registro_consistente"
                ]
            ),
        },
        "episodio DWA válido y registro consistente",
    )

    print(
        "\nResultado de la prueba DWA:",
        metricas_dwa_real[
            "resultado"
        ],
    )

    print(
        "Pasos ejecutados:",
        metricas_dwa_real[
            "pasos_ejecutados"
        ],
    )

    print(
        "Longitud recorrida:",
        metricas_dwa_real[
            "longitud_recorrida"
        ],
    )

    print(
        "Clearance mínimo:",
        metricas_dwa_real[
            "clearance_total_minimo"
        ],
    )
    # ======================================================
    # 11. CONVERSIÓN DE ACCIONES SAC
    # ======================================================

    control_reposo = convertir_accion_sac_a_control(
        [
            -1.0,
            0.0,
        ]
    )

    control_medio = convertir_accion_sac_a_control(
        [
            0.0,
            0.0,
        ]
    )

    control_maximo = convertir_accion_sac_a_control(
        [
            1.0,
            1.0,
        ]
    )

    control_fuera_rango = (
        convertir_accion_sac_a_control(
            [
                2.0,
                -2.0,
            ]
        )
    )

    verificar(
        "acción SAC de reposo",
        (
            math.isclose(
                control_reposo[0],
                0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                control_reposo[1],
                0.0,
                abs_tol=1e-9,
            )
        ),
        control_reposo,
        (
            0.0,
            0.0,
        ),
    )

    verificar(
        "acción SAC media",
        (
            math.isclose(
                control_medio[0],
                0.5,
                abs_tol=1e-9,
            )
            and math.isclose(
                control_medio[1],
                0.0,
                abs_tol=1e-9,
            )
        ),
        control_medio,
        (
            0.5,
            0.0,
        ),
    )

    verificar(
        "acción SAC máxima",
        (
            math.isclose(
                control_maximo[0],
                VELOCIDAD_MAXIMA,
                abs_tol=1e-9,
            )
            and math.isclose(
                control_maximo[1],
                VELOCIDAD_ANGULAR_MAXIMA,
                abs_tol=1e-9,
            )
        ),
        control_maximo,
        (
            VELOCIDAD_MAXIMA,
            VELOCIDAD_ANGULAR_MAXIMA,
        ),
    )

    verificar(
        "limitación de una acción SAC fuera de rango",
        (
            math.isclose(
                control_fuera_rango[0],
                VELOCIDAD_MAXIMA,
                abs_tol=1e-9,
            )
            and math.isclose(
                control_fuera_rango[1],
                -VELOCIDAD_ANGULAR_MAXIMA,
                abs_tol=1e-9,
            )
        ),
        control_fuera_rango,
        (
            VELOCIDAD_MAXIMA,
            -VELOCIDAD_ANGULAR_MAXIMA,
        ),
    )
    # ======================================================
    # 12. REINICIO DEL ENTORNO SAC
    # ======================================================

    (
        entorno_sac_1,
        observacion_sac_1,
    ) = reiniciar_entorno_sac(
        semilla=SEMILLA
    )

    (
        entorno_sac_2,
        observacion_sac_2,
    ) = reiniciar_entorno_sac(
        semilla=SEMILLA
    )

    # ------------------------------------------------------
    # Estructura del entorno
    # ------------------------------------------------------

    claves_entorno_sac = [
        "semilla",
        "semilla_dinamica",
        "escenario",
        "estado_inicial",
        "estado_robot",
        "obstaculos_estaticos",
        "obstaculos_dinamicos_iniciales",
        "obstaculos_dinamicos",
        "camino_mundo",
        "meta",
        "submeta",
        "indice_progreso",
        "indice_submeta",
        "velocidad_lineal_actual",
        "velocidad_angular_actual",
        "paso_actual",
        "tiempo_actual",
        "terminado",
        "resultado",
        "observacion",
        "obstaculos_observados",
        "registro",
    ]

    estructura_entorno_correcta = all(
        clave in entorno_sac_1
        for clave in claves_entorno_sac
    )

    verificar(
        "estructura inicial del entorno SAC",
        estructura_entorno_correcta,
        list(
            entorno_sac_1.keys()
        ),
        claves_entorno_sac,
    )

    # ------------------------------------------------------
    # Dimensión y tipo de la observación
    # ------------------------------------------------------

    observacion_inicial_correcta = (
        isinstance(
            observacion_sac_1,
            np.ndarray,
        )
        and observacion_sac_1.dtype == np.float32
        and observacion_sac_1.shape == (
            DIMENSION_OBSERVACION_SAC,
        )
        and np.all(
            np.isfinite(
                observacion_sac_1
            )
        )
    )

    verificar(
        "observación inicial de SAC",
        observacion_inicial_correcta,
        {
            "tipo": type(
                observacion_sac_1
            ),
            "dtype": observacion_sac_1.dtype,
            "shape": observacion_sac_1.shape,
        },
        {
            "tipo": np.ndarray,
            "dtype": np.float32,
            "shape": (
                DIMENSION_OBSERVACION_SAC,
            ),
        },
    )

    # ------------------------------------------------------
    # Estado inicial y registro
    # ------------------------------------------------------

    registro_sac_inicial = entorno_sac_1[
        "registro"
    ]

    estado_inicial_correcto = (
        entorno_sac_1[
            "estado_robot"
        ] == entorno_sac_1[
            "estado_inicial"
        ]
        and entorno_sac_1[
            "velocidad_lineal_actual"
        ] == 0.0
        and entorno_sac_1[
            "velocidad_angular_actual"
        ] == 0.0
        and entorno_sac_1[
            "paso_actual"
        ] == 0
        and entorno_sac_1[
            "tiempo_actual"
        ] == 0.0
        and entorno_sac_1[
            "terminado"
        ] is False
        and entorno_sac_1[
            "resultado"
        ] == "en_progreso"
        and registro_sac_inicial[
            "pasos_ejecutados"
        ] == 0
        and len(
            registro_sac_inicial[
                "estados"
            ]
        ) == 1
        and len(
            registro_sac_inicial[
                "controles"
            ]
        ) == 0
    )

    verificar(
        "estado inicial y registro SAC",
        estado_inicial_correcto,
        {
            "paso": entorno_sac_1[
                "paso_actual"
            ],
            "resultado": entorno_sac_1[
                "resultado"
            ],
            "estados": len(
                registro_sac_inicial[
                    "estados"
                ]
            ),
            "controles": len(
                registro_sac_inicial[
                    "controles"
                ]
            ),
        },
        {
            "paso": 0,
            "resultado": "en_progreso",
            "estados": 1,
            "controles": 0,
        },
    )

    # ------------------------------------------------------
    # Reproducibilidad
    # ------------------------------------------------------

    reinicio_reproducible = (
        escenarios_son_iguales(
            entorno_sac_1[
                "escenario"
            ],
            entorno_sac_2[
                "escenario"
            ],
        )
        and entorno_sac_1[
            "obstaculos_dinamicos_iniciales"
        ] == entorno_sac_2[
            "obstaculos_dinamicos_iniciales"
        ]
        and np.allclose(
            observacion_sac_1,
            observacion_sac_2,
            atol=1e-9,
        )
    )

    verificar(
        "reinicio SAC reproducible",
        reinicio_reproducible,
        {
            "observaciones_iguales": (
                np.allclose(
                    observacion_sac_1,
                    observacion_sac_2,
                )
            ),
            "numero_dinamicos_1": len(
                entorno_sac_1[
                    "obstaculos_dinamicos"
                ]
            ),
            "numero_dinamicos_2": len(
                entorno_sac_2[
                    "obstaculos_dinamicos"
                ]
            ),
        },
        "mismo escenario, obstáculos y observación",
    )
    # ======================================================
    # 13. EJECUCIÓN DE UN PASO SAC
    # ======================================================

    (
        entorno_paso_sac,
        observacion_inicial_paso,
    ) = reiniciar_entorno_sac(
        semilla=SEMILLA
    )

    estado_antes_paso = entorno_paso_sac[
        "estado_robot"
    ]

    (
        observacion_despues_paso,
        recompensa_paso,
        terminado_paso,
        truncado_paso,
        informacion_paso,
    ) = ejecutar_paso_entorno_sac(
        entorno=entorno_paso_sac,
        accion=[
            0.0,
            0.0,
        ],
        pasos_maximos=(
            PASOS_MAXIMOS_SEGUIMIENTO
        ),
        dt=DT,
    )

    estado_despues_paso = entorno_paso_sac[
        "estado_robot"
    ]

    registro_paso_sac = entorno_paso_sac[
        "registro"
    ]

    # ------------------------------------------------------
    # Movimiento y contadores
    # ------------------------------------------------------

    distancia_movida = distancia_entre_puntos(
        (
            estado_antes_paso[0],
            estado_antes_paso[1],
        ),
        (
            estado_despues_paso[0],
            estado_despues_paso[1],
        ),
    )

    paso_sac_actualizado = (
        entorno_paso_sac[
            "paso_actual"
        ] == 1

        and math.isclose(
            entorno_paso_sac[
                "tiempo_actual"
            ],
            DT,
            abs_tol=1e-9,
        )

        and distancia_movida > 0.0
    )

    verificar(
        "actualización de un paso SAC",
        paso_sac_actualizado,
        {
            "paso": entorno_paso_sac[
                "paso_actual"
            ],
            "tiempo": entorno_paso_sac[
                "tiempo_actual"
            ],
            "distancia_movida": (
                distancia_movida
            ),
        },
        {
            "paso": 1,
            "tiempo": DT,
            "distancia_movida": "> 0",
        },
    )

    # ------------------------------------------------------
    # Registro
    # ------------------------------------------------------

    registro_paso_correcto = (
        registro_paso_sac[
            "pasos_ejecutados"
        ] == 1

        and len(
            registro_paso_sac[
                "estados"
            ]
        ) == 2

        and len(
            registro_paso_sac[
                "controles"
            ]
        ) == 1

        and len(
            registro_paso_sac[
                "clearances_estaticos"
            ]
        ) == 1

        and len(
            registro_paso_sac[
                "clearances_dinamicos"
            ]
        ) == 1

        and verificar_registro_clearances(
            registro_paso_sac
        )
    )

    verificar(
        "registro después de un paso SAC",
        registro_paso_correcto,
        {
            "estados": len(
                registro_paso_sac[
                    "estados"
                ]
            ),
            "controles": len(
                registro_paso_sac[
                    "controles"
                ]
            ),
            "pasos": registro_paso_sac[
                "pasos_ejecutados"
            ],
        },
        {
            "estados": 2,
            "controles": 1,
            "pasos": 1,
        },
    )

    # ------------------------------------------------------
    # Observación y recompensa provisional
    # ------------------------------------------------------

    salida_paso_correcta = (
        observacion_despues_paso.shape
        == (
            DIMENSION_OBSERVACION_SAC,
        )

        and observacion_despues_paso.dtype
        == np.float32

        and np.all(
            np.isfinite(
                observacion_despues_paso
            )
        )

        and np.isfinite(
            recompensa_paso
        )

        and isinstance(
            informacion_paso[
                "componentes_recompensa"
            ],
            dict,
        )

        and isinstance(
            terminado_paso,
            bool,
        )

        and isinstance(
            truncado_paso,
            bool,
        )
    )

    verificar(
        "salida de un paso SAC",
        salida_paso_correcta,
        {
            "shape": (
                observacion_despues_paso.shape
            ),
            "dtype": (
                observacion_despues_paso.dtype
            ),
            "recompensa": recompensa_paso,
            "terminado": terminado_paso,
            "truncado": truncado_paso,
        },
        {
            "shape": (
                DIMENSION_OBSERVACION_SAC,
            ),
            "dtype": np.float32,
            "recompensa": "valor finito",
        },
    )

    # ------------------------------------------------------
    # Timeout separado de terminado
    # ------------------------------------------------------

    (
        entorno_timeout_sac,
        observacion_timeout_sac,
    ) = reiniciar_entorno_sac(
        semilla=SEMILLA
    )

    (
        observacion_timeout_sac,
        recompensa_timeout_sac,
        terminado_timeout_sac,
        truncado_timeout_sac,
        informacion_timeout_sac,
    ) = ejecutar_paso_entorno_sac(
        entorno=entorno_timeout_sac,
        accion=[
            -1.0,
            0.0,
        ],
        pasos_maximos=1,
        dt=DT,
    )

    timeout_correcto = (
        terminado_timeout_sac is False

        and truncado_timeout_sac is True

        and entorno_timeout_sac[
            "terminado"
        ] is True

        and entorno_timeout_sac[
            "resultado"
        ] == "timeout"

        and informacion_timeout_sac[
            "resultado"
        ] == "timeout"
    )

    verificar(
        "timeout SAC tratado como truncamiento",
        timeout_correcto,
        {
            "terminado": (
                terminado_timeout_sac
            ),
            "truncado": (
                truncado_timeout_sac
            ),
            "resultado": (
                entorno_timeout_sac[
                    "resultado"
                ]
            ),
        },
        {
            "terminado": False,
            "truncado": True,
            "resultado": "timeout",
        },
    )
    # ======================================================
    # 14. FUNCIÓN DE RECOMPENSA SAC
    # ======================================================

    camino_recompensa = [
        (
            0.0,
            0.0,
        ),
        (
            10.0,
            0.0,
        ),
    ]

    submeta_recompensa = (
        5.0,
        0.0,
    )

    meta_recompensa = (
        10.0,
        0.0,
    )

    estado_anterior_recompensa = (
        1.0,
        0.0,
        0.0,
    )

    estado_avance_recompensa = (
        1.1,
        0.0,
        0.0,
    )

    estado_retroceso_recompensa = (
        0.9,
        0.0,
        math.pi,
    )

    # ------------------------------------------------------
    # Avanzar debe ser mejor que retroceder
    # ------------------------------------------------------

    (
        recompensa_avance,
        componentes_avance,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=1.0,
        resultado="en_progreso",
        dt=0.1,
    )

    (
        recompensa_retroceso,
        componentes_retroceso,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_retroceso_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=1.0,
        resultado="en_progreso",
        dt=0.1,
    )

    verificar(
        "recompensa SAC favorece el progreso",
        (
            recompensa_avance
            > recompensa_retroceso

            and componentes_avance[
                "progreso"
            ] > 0.0

            and componentes_retroceso[
                "progreso"
            ] < 0.0
        ),
        {
            "avance": recompensa_avance,
            "retroceso": (
                recompensa_retroceso
            ),
        },
        "recompensa de avance mayor",
    )

    # ------------------------------------------------------
    # El riesgo debe reducir la recompensa
    # ------------------------------------------------------

    (
        recompensa_segura,
        componentes_segura,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=1.0,
        resultado="en_progreso",
        dt=0.1,
    )

    (
        recompensa_riesgo,
        componentes_riesgo,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=0.05,
        resultado="en_progreso",
        dt=0.1,
    )

    verificar(
        "recompensa SAC penaliza el riesgo",
        (
            recompensa_segura
            > recompensa_riesgo

            and componentes_riesgo[
                "penalizacion_seguridad"
            ] > componentes_segura[
                "penalizacion_seguridad"
            ]
        ),
        {
            "segura": recompensa_segura,
            "riesgo": recompensa_riesgo,
        },
        "recompensa segura mayor",
    )

    # ------------------------------------------------------
    # Meta y colisión
    # ------------------------------------------------------

    (
        recompensa_meta,
        componentes_meta,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=1.0,
        resultado="meta",
        dt=0.1,
    )

    (
        recompensa_colision,
        componentes_colision,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=-0.01,
        resultado="colision_dinamica",
        dt=0.1,
    )

    verificar(
        "recompensas terminales de SAC",
        (
            recompensa_meta > 90.0

            and recompensa_colision < -90.0

            and componentes_meta[
                "recompensa_terminal"
            ] == RECOMPENSA_SAC_META

            and componentes_colision[
                "recompensa_terminal"
            ] == RECOMPENSA_SAC_COLISION
        ),
        {
            "meta": recompensa_meta,
            "colision": recompensa_colision,
        },
        {
            "meta": "> 90",
            "colision": "< -90",
        },
    )

    # ------------------------------------------------------
    # Cambios bruscos deben ser penalizados
    # ------------------------------------------------------

    (
        recompensa_suave,
        componentes_suave,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=1.0,
        resultado="en_progreso",
        dt=0.1,
    )

    (
        recompensa_brusca,
        componentes_brusca,
    ) = calcular_recompensa_sac(
        estado_anterior=(
            estado_anterior_recompensa
        ),
        estado_nuevo=(
            estado_avance_recompensa
        ),
        submeta=submeta_recompensa,
        meta=meta_recompensa,
        camino_mundo=camino_recompensa,
        velocidad_lineal_anterior=0.0,
        velocidad_angular_anterior=(
            -VELOCIDAD_ANGULAR_MAXIMA
        ),
        velocidad_lineal_nueva=1.0,
        velocidad_angular_nueva=(
            VELOCIDAD_ANGULAR_MAXIMA
        ),
        clearance_total=1.0,
        resultado="en_progreso",
        dt=0.1,
    )

    verificar(
        "recompensa SAC penaliza cambios bruscos",
        (
            componentes_brusca[
                "penalizacion_suavidad"
            ] > componentes_suave[
                "penalizacion_suavidad"
            ]
        ),
        {
            "suave": componentes_suave[
                "penalizacion_suavidad"
            ],
            "brusca": componentes_brusca[
                "penalizacion_suavidad"
            ],
        },
        "penalización brusca mayor",
    )
    # ======================================================
    # 15. EPISODIO SAC COMPLETO DE PRUEBA
    # ======================================================

    episodio_sac_prueba = (
        ejecutar_episodio_sac_prueba(
            semilla=SEMILLA,
            pasos_maximos=5,
            dt=DT,
        )
    )

    registro_episodio_sac = (
        episodio_sac_prueba[
            "registro"
        ]
    )

    metricas_episodio_sac = (
        episodio_sac_prueba[
            "metricas"
        ]
    )

    acciones_episodio_sac = (
        episodio_sac_prueba[
            "acciones"
        ]
    )

    recompensas_episodio_sac = (
        episodio_sac_prueba[
            "recompensas"
        ]
    )

    numero_pasos_sac = registro_episodio_sac[
        "pasos_ejecutados"
    ]

    # ------------------------------------------------------
    # Ciclo completo
    # ------------------------------------------------------

    ciclo_sac_completo = (
        numero_pasos_sac >= 1

        and numero_pasos_sac <= 5

        and len(
            registro_episodio_sac[
                "estados"
            ]
        ) == numero_pasos_sac + 1

        and len(
            acciones_episodio_sac
        ) == numero_pasos_sac

        and len(
            recompensas_episodio_sac
        ) == numero_pasos_sac

        and (
            episodio_sac_prueba[
                "terminado"
            ]
            or episodio_sac_prueba[
                "truncado"
            ]
        )

        and registro_episodio_sac[
            "resultado"
        ] != "en_progreso"

        and verificar_registro_clearances(
            registro_episodio_sac
        )
    )

    verificar(
        "ciclo completo de un episodio SAC",
        ciclo_sac_completo,
        {
            "resultado": registro_episodio_sac[
                "resultado"
            ],
            "pasos": numero_pasos_sac,
            "terminado": episodio_sac_prueba[
                "terminado"
            ],
            "truncado": episodio_sac_prueba[
                "truncado"
            ],
        },
        "episodio finalizado o truncado",
    )

    # ------------------------------------------------------
    # Acciones válidas
    # ------------------------------------------------------

    acciones_sac_validas = all(
        isinstance(
            accion,
            np.ndarray,
        )
        and accion.shape == (
            DIMENSION_ACCION_SAC,
        )
        and accion.dtype == np.float32
        and np.all(
            np.isfinite(
                accion
            )
        )
        and np.all(
            accion >= -1.0
        )
        and np.all(
            accion <= 1.0
        )
        for accion in acciones_episodio_sac
    )

    verificar(
        "acciones del episodio SAC válidas",
        acciones_sac_validas,
        acciones_episodio_sac,
        "acciones float32 dentro de [-1, 1]",
    )

    # ------------------------------------------------------
    # Recompensas y métricas
    # ------------------------------------------------------

    recompensas_sac_validas = (
        len(
            recompensas_episodio_sac
        ) > 0

        and np.all(
            np.isfinite(
                recompensas_episodio_sac
            )
        )

        and np.isfinite(
            metricas_episodio_sac[
                "recompensa_acumulada"
            ]
        )

        and np.isfinite(
            metricas_episodio_sac[
                "recompensa_promedio"
            ]
        )

        and metricas_episodio_sac[
            "metodo"
        ] == "sac_prueba"

        and metricas_episodio_sac[
            "registro_consistente"
        ] is True
    )

    verificar(
        "recompensas y métricas del episodio SAC",
        recompensas_sac_validas,
        {
            "recompensa_acumulada": (
                metricas_episodio_sac[
                    "recompensa_acumulada"
                ]
            ),
            "recompensa_promedio": (
                metricas_episodio_sac[
                    "recompensa_promedio"
                ]
            ),
            "resultado": (
                metricas_episodio_sac[
                    "resultado"
                ]
            ),
        },
        "recompensas finitas y métricas consistentes",
    )

    print(
        "\nResultado del episodio SAC de prueba:",
        metricas_episodio_sac[
            "resultado"
        ],
    )

    print(
        "Pasos ejecutados:",
        numero_pasos_sac,
    )

    print(
        "Recompensa acumulada:",
        metricas_episodio_sac[
            "recompensa_acumulada"
        ],
    )
    # ======================================================
    # RESUMEN
    # ======================================================

    pruebas_aprobadas = sum(
        resultados_pruebas
    )

    total_pruebas = len(
        resultados_pruebas
    )

    pruebas_fallidas = (
        total_pruebas
        - pruebas_aprobadas
    )

    print("-" * 68)

    print(
        "Pruebas aprobadas:",
        pruebas_aprobadas,
    )

    print(
        "Pruebas fallidas:",
        pruebas_fallidas,
    )

    if pruebas_fallidas == 0:

        print(
            "RESULTADO GENERAL: TODO CORRECTO"
        )

    else:

        print(
            "RESULTADO GENERAL: HAY FUNCIONES POR CORREGIR"
        )

    print("=" * 68)
def crear_optimizador_critico_sac(
    critico,
    tasa_aprendizaje=(
        TASA_APRENDIZAJE_CRITICOS_SAC
    ),
):

    if not isinstance(
        critico,
        nn.Module,
    ):

        raise TypeError(
            "El crítico debe ser un módulo de PyTorch."
        )

    tasa_aprendizaje = float(
        tasa_aprendizaje
    )

    if not np.isfinite(
        tasa_aprendizaje
    ):

        raise ValueError(
            "La tasa de aprendizaje debe ser finita."
        )

    if tasa_aprendizaje <= 0.0:

        raise ValueError(
            "La tasa de aprendizaje debe ser positiva."
        )

    parametros_entrenables = [
        parametro
        for parametro in critico.parameters()
        if parametro.requires_grad
    ]

    if len(
        parametros_entrenables
    ) == 0:

        raise RuntimeError(
            "El crítico no contiene parámetros entrenables."
        )

    optimizador = torch.optim.Adam(
        params=parametros_entrenables,

        lr=tasa_aprendizaje,

        betas=(
            BETA_1_ADAM_CRITICOS_SAC,
            BETA_2_ADAM_CRITICOS_SAC,
        ),

        eps=EPSILON_ADAM_CRITICOS_SAC,

        weight_decay=0.0,
    )

    identificadores_critico = {
        id(
            parametro
        )
        for parametro in parametros_entrenables
    }

    identificadores_optimizador = {
        id(
            parametro
        )
        for grupo in optimizador.param_groups
        for parametro in grupo[
            "params"
        ]
    }

    if (
        identificadores_critico
        != identificadores_optimizador
    ):

        raise RuntimeError(
            "El optimizador no contiene exactamente "
            "los parámetros entrenables del crítico."
        )

    return optimizador
def crear_optimizadores_dos_criticos_sac(
    critico_q1,
    critico_q2,
    tasa_aprendizaje=(
        TASA_APRENDIZAJE_CRITICOS_SAC
    ),
):

    optimizador_q1 = (
        crear_optimizador_critico_sac(
            critico=critico_q1,

            tasa_aprendizaje=(
                tasa_aprendizaje
            ),
        )
    )

    optimizador_q2 = (
        crear_optimizador_critico_sac(
            critico=critico_q2,

            tasa_aprendizaje=(
                tasa_aprendizaje
            ),
        )
    )

    identificadores_q1 = {
        id(
            parametro
        )
        for grupo in optimizador_q1.param_groups
        for parametro in grupo[
            "params"
        ]
    }

    identificadores_q2 = {
        id(
            parametro
        )
        for grupo in optimizador_q2.param_groups
        for parametro in grupo[
            "params"
        ]
    }

    if len(
        identificadores_q1.intersection(
            identificadores_q2
        )
    ) != 0:

        raise RuntimeError(
            "Los optimizadores Q1 y Q2 no deben "
            "compartir parámetros."
        )

    return (
        optimizador_q1,
        optimizador_q2,
    )
def calcular_objetivo_criticos_sac(
    recompensas,
    terminados,
    valor_q_objetivo_minimo,
    log_probabilidad_siguiente,
    factor_descuento=FACTOR_DESCUENTO_SAC,
    coeficiente_entropia=(
        COEFICIENTE_ENTROPIA_INICIAL_SAC
    ),
):

    tensores = [
        recompensas,
        terminados,
        valor_q_objetivo_minimo,
        log_probabilidad_siguiente,
    ]

    if not all(
        torch.is_tensor(
            tensor
        )
        for tensor in tensores
    ):

        raise TypeError(
            "Las entradas del objetivo SAC deben ser "
            "tensores de PyTorch."
        )

    if recompensas.ndim != 2:

        raise ValueError(
            "Las recompensas deben tener forma (lote, 1)."
        )

    if recompensas.shape[1] != 1:

        raise ValueError(
            "Debe existir una recompensa por transición."
        )

    forma_esperada = recompensas.shape

    if not all(
        tensor.shape == forma_esperada
        for tensor in tensores
    ):

        raise ValueError(
            "Todos los tensores deben tener la misma "
            "forma (lote, 1)."
        )

    dispositivo = recompensas.device

    tipo_datos = recompensas.dtype

    if not all(
        tensor.device == dispositivo
        for tensor in tensores
    ):

        raise ValueError(
            "Todos los tensores deben estar en el mismo "
            "dispositivo."
        )

    if not all(
        tensor.dtype == tipo_datos
        for tensor in tensores
    ):

        raise TypeError(
            "Todos los tensores deben tener el mismo "
            "tipo de datos."
        )

    if not all(
        torch.isfinite(
            tensor
        ).all().item()
        for tensor in tensores
    ):

        raise ValueError(
            "El objetivo recibió NaN o valores infinitos."
        )

    if not torch.all(
        terminados >= 0.0
    ).item():

        raise ValueError(
            "El indicador terminado no puede ser negativo."
        )

    if not torch.all(
        terminados <= 1.0
    ).item():

        raise ValueError(
            "El indicador terminado no puede ser mayor "
            "que uno."
        )

    factor_descuento = float(
        factor_descuento
    )

    if (
        not np.isfinite(
            factor_descuento
        )
        or factor_descuento < 0.0
        or factor_descuento > 1.0
    ):

        raise ValueError(
            "El factor de descuento debe pertenecer "
            "al intervalo [0, 1]."
        )

    # ------------------------------------------------------
    # Preparar el alpha aprendido
    # ------------------------------------------------------

    alpha = preparar_coeficiente_entropia_sac(
        coeficiente_entropia=(
            coeficiente_entropia
        ),
        dispositivo=dispositivo,
        tipo_datos=tipo_datos,
    )

    # ------------------------------------------------------
    # Calcular el objetivo temporal
    # ------------------------------------------------------

    with torch.no_grad():

        valor_siguiente_con_entropia = (
            valor_q_objetivo_minimo
            -
            alpha
            * log_probabilidad_siguiente
        )

        objetivo_q = (
            recompensas
            +
            factor_descuento
            * (
                1.0
                - terminados
            )
            * valor_siguiente_con_entropia
        )

    if objetivo_q.requires_grad:

        raise RuntimeError(
            "El objetivo temporal no debe conservar "
            "gradientes."
        )

    if not torch.isfinite(
        objetivo_q
    ).all().item():

        raise ValueError(
            "El objetivo Q contiene NaN o infinitos."
        )

    return objetivo_q
def calcular_perdidas_criticos_sac(
    valor_q1,
    valor_q2,
    objetivo_q,
):

    if not all(
        torch.is_tensor(
            tensor
        )
        for tensor in [
            valor_q1,
            valor_q2,
            objetivo_q,
        ]
    ):

        raise TypeError(
            "Q1, Q2 y el objetivo deben ser tensores."
        )

    if not (
        valor_q1.shape
        == valor_q2.shape
        == objetivo_q.shape
    ):

        raise ValueError(
            "Q1, Q2 y el objetivo deben tener "
            "la misma forma."
        )

    if valor_q1.ndim != 2:

        raise ValueError(
            "Los valores Q deben tener forma (lote, 1)."
        )

    if valor_q1.shape[1] != 1:

        raise ValueError(
            "Cada crítico debe producir un valor escalar."
        )

    if not (
        valor_q1.device
        == valor_q2.device
        == objetivo_q.device
    ):

        raise ValueError(
            "Q1, Q2 y el objetivo deben estar en "
            "el mismo dispositivo."
        )

    if not (
        valor_q1.dtype
        == valor_q2.dtype
        == objetivo_q.dtype
    ):

        raise TypeError(
            "Q1, Q2 y el objetivo deben tener "
            "el mismo tipo de datos."
        )

    perdida_q1 = (
        valor_q1
        - objetivo_q
    ).pow(
        2
    ).mean()

    perdida_q2 = (
        valor_q2
        - objetivo_q
    ).pow(
        2
    ).mean()

    perdida_total_criticos = (
        perdida_q1
        + perdida_q2
    )

    if not torch.isfinite(
        perdida_total_criticos
    ).item():

        raise ValueError(
            "La pérdida de los críticos no es finita."
        )

    return (
        perdida_q1,
        perdida_q2,
        perdida_total_criticos,
    )
def actualizar_dos_criticos_sac(
    valor_q1,
    valor_q2,
    objetivo_q,
    critico_q1,
    critico_q2,
    optimizador_q1,
    optimizador_q2,
):

    if not isinstance(
        optimizador_q1,
        torch.optim.Optimizer,
    ):

        raise TypeError(
            "El optimizador Q1 no es válido."
        )

    if not isinstance(
        optimizador_q2,
        torch.optim.Optimizer,
    ):

        raise TypeError(
            "El optimizador Q2 no es válido."
        )

    optimizador_q1.zero_grad(
        set_to_none=True
    )

    optimizador_q2.zero_grad(
        set_to_none=True
    )

    (
        perdida_q1,
        perdida_q2,
        perdida_total_criticos,
    ) = calcular_perdidas_criticos_sac(
        valor_q1=valor_q1,
        valor_q2=valor_q2,
        objetivo_q=objetivo_q,
    )

    perdida_total_criticos.backward()

    parametros_q1 = [
        parametro
        for parametro in critico_q1.parameters()
        if parametro.requires_grad
    ]

    parametros_q2 = [
        parametro
        for parametro in critico_q2.parameters()
        if parametro.requires_grad
    ]

    gradientes_q1_completos = all(
        parametro.grad is not None
        for parametro in parametros_q1
    )

    gradientes_q2_completos = all(
        parametro.grad is not None
        for parametro in parametros_q2
    )

    gradientes_q1_finitos = all(
        torch.isfinite(
            parametro.grad
        ).all().item()
        for parametro in parametros_q1
        if parametro.grad is not None
    )

    gradientes_q2_finitos = all(
        torch.isfinite(
            parametro.grad
        ).all().item()
        for parametro in parametros_q2
        if parametro.grad is not None
    )

    sumas_gradientes_q1 = [
        parametro.grad
        .detach()
        .pow(
            2
        )
        .sum()
        for parametro in parametros_q1
        if parametro.grad is not None
    ]

    sumas_gradientes_q2 = [
        parametro.grad
        .detach()
        .pow(
            2
        )
        .sum()
        for parametro in parametros_q2
        if parametro.grad is not None
    ]

    norma_gradientes_q1 = torch.sqrt(
        torch.stack(
            sumas_gradientes_q1
        ).sum()
    ).item()

    norma_gradientes_q2 = torch.sqrt(
        torch.stack(
            sumas_gradientes_q2
        ).sum()
    ).item()

    optimizador_q1.step()

    optimizador_q2.step()

    return {
        "perdida_q1": perdida_q1.detach(),
        "perdida_q2": perdida_q2.detach(),
        "perdida_total": (
            perdida_total_criticos.detach()
        ),
        "gradientes_q1_completos": (
            gradientes_q1_completos
        ),
        "gradientes_q2_completos": (
            gradientes_q2_completos
        ),
        "gradientes_q1_finitos": (
            gradientes_q1_finitos
        ),
        "gradientes_q2_finitos": (
            gradientes_q2_finitos
        ),
        "norma_gradientes_q1": (
            norma_gradientes_q1
        ),
        "norma_gradientes_q2": (
            norma_gradientes_q2
        ),
    }
def calcular_perdida_actor_sac(
    log_probabilidad,
    valor_q1,
    valor_q2,
    coeficiente_entropia,
):

    entradas = [
        log_probabilidad,
        valor_q1,
        valor_q2,
    ]

    if not all(
        torch.is_tensor(
            entrada
        )
        for entrada in entradas
    ):

        raise TypeError(
            "La log-probabilidad, Q1 y Q2 deben ser "
            "tensores."
        )

    if not (
        log_probabilidad.shape
        == valor_q1.shape
        == valor_q2.shape
    ):

        raise ValueError(
            "La log-probabilidad, Q1 y Q2 deben tener "
            "la misma forma."
        )

    if (
        log_probabilidad.ndim != 2
        or log_probabilidad.shape[1] != 1
    ):

        raise ValueError(
            "Las entradas deben tener forma (lote, 1)."
        )

    if not (
        log_probabilidad.device
        == valor_q1.device
        == valor_q2.device
    ):

        raise ValueError(
            "Las entradas deben estar en el mismo "
            "dispositivo."
        )

    if not (
        log_probabilidad.dtype
        == valor_q1.dtype
        == valor_q2.dtype
    ):

        raise TypeError(
            "Las entradas deben tener el mismo tipo."
        )

    if not all(
        torch.isfinite(
            entrada
        ).all().item()
        for entrada in entradas
    ):

        raise ValueError(
            "La pérdida del actor recibió valores "
            "no finitos."
        )

    # ------------------------------------------------------
    # Utilizar alpha aprendido, pero desacoplado
    # ------------------------------------------------------

    alpha = preparar_coeficiente_entropia_sac(
        coeficiente_entropia=(
            coeficiente_entropia
        ),
        dispositivo=(
            log_probabilidad.device
        ),
        tipo_datos=(
            log_probabilidad.dtype
        ),
    )

    valor_q_minimo = torch.minimum(
        valor_q1,
        valor_q2,
    )

    termino_entropia = (
        alpha
        * log_probabilidad
    )

    perdida_por_elemento = (
        termino_entropia
        - valor_q_minimo
    )

    perdida_actor = (
        perdida_por_elemento.mean()
    )

    if perdida_actor.ndim != 0:

        raise ValueError(
            "La pérdida del actor debe ser escalar."
        )

    if not torch.isfinite(
        perdida_actor
    ).item():

        raise ValueError(
            "La pérdida del actor no es finita."
        )

    return {
        "perdida_actor": perdida_actor,
        "valor_q_minimo": valor_q_minimo,
        "termino_entropia": termino_entropia,
        "perdida_por_elemento": (
            perdida_por_elemento
        ),
        "coeficiente_entropia": alpha,
    }
def actualizar_actor_sac(
    tensor_parche,
    tensor_escalares,
    actor,
    critico_q1,
    critico_q2,
    optimizador_actor,
    temperatura,
):

    # ------------------------------------------------------
    # 1. Verificar los modelos y el optimizador
    # ------------------------------------------------------

    if not isinstance(
        actor,
        nn.ModuleDict,
    ):

        raise TypeError(
            "El actor debe ser un nn.ModuleDict."
        )

    if not isinstance(
        critico_q1,
        nn.ModuleDict,
    ):

        raise TypeError(
            "Q1 debe ser un nn.ModuleDict."
        )

    if not isinstance(
        critico_q2,
        nn.ModuleDict,
    ):

        raise TypeError(
            "Q2 debe ser un nn.ModuleDict."
        )

    if not isinstance(
        optimizador_actor,
        torch.optim.Optimizer,
    ):

        raise TypeError(
            "El optimizador del actor no es válido."
        )
    if not isinstance(
        temperatura,
        nn.ParameterDict,
    ):

        raise TypeError(
            "La temperatura debe ser un "
            "nn.ParameterDict."
        )

    coeficiente_entropia = (
        obtener_coeficiente_entropia_sac(
            temperatura=temperatura
        )
    )
    # ------------------------------------------------------
    # 2. Guardar el estado de requires_grad de los críticos
    # ------------------------------------------------------

    parametros_q1 = list(
        critico_q1.parameters()
    )

    parametros_q2 = list(
        critico_q2.parameters()
    )

    estados_gradiente_q1 = [
        parametro.requires_grad
        for parametro in parametros_q1
    ]

    estados_gradiente_q2 = [
        parametro.requires_grad
        for parametro in parametros_q2
    ]

    # ------------------------------------------------------
    # 3. Limpiar gradientes anteriores
    # ------------------------------------------------------

    optimizador_actor.zero_grad(
        set_to_none=True
    )

    critico_q1.zero_grad(
        set_to_none=True
    )

    critico_q2.zero_grad(
        set_to_none=True
    )

    try:

        # --------------------------------------------------
        # 4. Congelar temporalmente Q1 y Q2
        # --------------------------------------------------

        for parametro in parametros_q1:

            parametro.requires_grad_(
                False
            )

        for parametro in parametros_q2:

            parametro.requires_grad_(
                False
            )

        # --------------------------------------------------
        # 5. Ejecutar el actor
        # --------------------------------------------------

        resultados_actor = ejecutar_actor_sac(
            tensor_parche=tensor_parche,

            tensor_escalares=tensor_escalares,

            rama_cnn=actor[
                "rama_cnn"
            ],

            rama_mlp=actor[
                "rama_mlp"
            ],

            red_compartida=actor[
                "red_compartida"
            ],

            cabeza_media=actor[
                "cabeza_media"
            ],

            cabeza_log_desviacion=actor[
                "cabeza_log_desviacion"
            ],
        )

        accion_actor = resultados_actor[
            "accion_normalizada"
        ]

        log_probabilidad = resultados_actor[
            "log_probabilidad_corregida"
        ]

        # La acción NO se separa del grafo.

        if not accion_actor.requires_grad:

            raise RuntimeError(
                "La acción del actor debe conservar "
                "el grafo de gradientes."
            )

        # --------------------------------------------------
        # 6. Evaluar la acción con Q1 y Q2
        # --------------------------------------------------

        resultados_criticos = (
            ejecutar_dos_criticos_sac(
                tensor_parche=tensor_parche,

                tensor_escalares=tensor_escalares,

                tensor_accion=accion_actor,

                critico_q1=critico_q1,

                critico_q2=critico_q2,
            )
        )

        valor_q1 = resultados_criticos[
            "valor_q1"
        ]

        valor_q2 = resultados_criticos[
            "valor_q2"
        ]

        # --------------------------------------------------
        # 7. Calcular la pérdida real del actor
        # --------------------------------------------------

        resultados_perdida = (
            calcular_perdida_actor_sac(
                log_probabilidad=(
                    log_probabilidad
                ),

                valor_q1=valor_q1,

                valor_q2=valor_q2,

                coeficiente_entropia=(
                    coeficiente_entropia
                ),
            )
        )

        perdida_actor = resultados_perdida[
            "perdida_actor"
        ]

        valor_q_minimo = resultados_perdida[
            "valor_q_minimo"
        ]

        # --------------------------------------------------
        # 8. Verificar que Q depende de la acción
        # --------------------------------------------------
        #
        # Esta operación obtiene solamente el gradiente
        # de -Q respecto a la acción, sin acumularlo todavía
        # en los parámetros del actor.

        gradiente_q_respecto_accion = (
            torch.autograd.grad(
                outputs=(
                    -valor_q_minimo.mean()
                ),

                inputs=accion_actor,

                retain_graph=True,

                create_graph=False,

                allow_unused=False,
            )[
                0
            ]
        )

        gradiente_q_accion_finito = (
            torch.isfinite(
                gradiente_q_respecto_accion
            ).all().item()
        )

        norma_gradiente_q_accion = (
            gradiente_q_respecto_accion
            .detach()
            .pow(
                2
            )
            .sum()
            .sqrt()
            .item()
        )

        # --------------------------------------------------
        # 9. Retropropagación
        # --------------------------------------------------

        perdida_actor.backward()

        parametros_actor = [
            parametro
            for parametro in actor.parameters()
            if parametro.requires_grad
        ]

        gradientes_actor_completos = all(
            parametro.grad is not None
            for parametro in parametros_actor
        )

        gradientes_actor_finitos = all(
            torch.isfinite(
                parametro.grad
            ).all().item()
            for parametro in parametros_actor
            if parametro.grad is not None
        )

        sumas_gradientes_actor = [
            parametro.grad
            .detach()
            .pow(
                2
            )
            .sum()
            for parametro in parametros_actor
            if parametro.grad is not None
        ]

        if len(
            sumas_gradientes_actor
        ) == 0:

            norma_gradientes_actor = 0.0

        else:

            norma_gradientes_actor = (
                torch.sqrt(
                    torch.stack(
                        sumas_gradientes_actor
                    ).sum()
                ).item()
            )

        # Los críticos deben seguir sin gradientes.

        gradientes_q1_ausentes = all(
            parametro.grad is None
            for parametro in parametros_q1
        )

        gradientes_q2_ausentes = all(
            parametro.grad is None
            for parametro in parametros_q2
        )

        # --------------------------------------------------
        # 10. Actualizar únicamente el actor
        # --------------------------------------------------

        optimizador_actor.step()

    finally:

        # --------------------------------------------------
        # 11. Restaurar Q1 y Q2
        # --------------------------------------------------
        #
        # De esta forma vuelven a quedar disponibles para
        # su propia actualización supervisada.

        for (
            parametro,
            estado_original,
        ) in zip(
            parametros_q1,
            estados_gradiente_q1,
        ):

            parametro.requires_grad_(
                estado_original
            )

        for (
            parametro,
            estado_original,
        ) in zip(
            parametros_q2,
            estados_gradiente_q2,
        ):

            parametro.requires_grad_(
                estado_original
            )

    # ------------------------------------------------------
    # 12. Regresar resultados separados del grafo
    # ------------------------------------------------------

    return {
        "perdida_actor": (
            perdida_actor.detach()
        ),
        "coeficiente_entropia": (
            resultados_perdida[
                "coeficiente_entropia"
            ].detach()
        ),
        "valor_q1": valor_q1.detach(),

        "valor_q2": valor_q2.detach(),

        "valor_q_minimo": (
            valor_q_minimo.detach()
        ),

        "log_probabilidad": (
            log_probabilidad.detach()
        ),

        "accion_actor": (
            accion_actor.detach()
        ),

        "termino_entropia": (
            resultados_perdida[
                "termino_entropia"
            ].detach()
        ),

        "gradientes_actor_completos": (
            gradientes_actor_completos
        ),

        "gradientes_actor_finitos": (
            gradientes_actor_finitos
        ),

        "norma_gradientes_actor": (
            norma_gradientes_actor
        ),

        "gradientes_q1_ausentes": (
            gradientes_q1_ausentes
        ),

        "gradientes_q2_ausentes": (
            gradientes_q2_ausentes
        ),

        "gradiente_q_accion_finito": (
            gradiente_q_accion_finito
        ),

        "norma_gradiente_q_accion": (
            norma_gradiente_q_accion
        ),
    }
def crear_temperatura_entropia_sac(
    dispositivo,
    coeficiente_inicial=(
        COEFICIENTE_ENTROPIA_INICIAL_SAC
    ),
):

    # ------------------------------------------------------
    # 1. Verificar el valor inicial
    # ------------------------------------------------------

    coeficiente_inicial = float(
        coeficiente_inicial
    )

    if not np.isfinite(
        coeficiente_inicial
    ):

        raise ValueError(
            "El coeficiente de entropía inicial "
            "debe ser finito."
        )

    if coeficiente_inicial <= 0.0:

        raise ValueError(
            "El coeficiente de entropía inicial "
            "debe ser estrictamente positivo."
        )

    # ------------------------------------------------------
    # 2. Calcular log(alpha)
    # ------------------------------------------------------

    log_alpha_inicial = np.log(
        coeficiente_inicial
    )

    # ------------------------------------------------------
    # 3. Crear un parámetro entrenable persistente
    # ------------------------------------------------------
    #
    # ParameterDict registra log_alpha como un parámetro
    # de PyTorch sin necesitar una clase personalizada.

    temperatura = nn.ParameterDict(
        {
            "log_alpha": nn.Parameter(
                torch.tensor(
                    log_alpha_inicial,
                    dtype=torch.float32,
                    device=dispositivo,
                ),
                requires_grad=True,
            )
        }
    )

    temperatura = temperatura.to(
        dispositivo
    )

    # ------------------------------------------------------
    # 4. Verificaciones
    # ------------------------------------------------------

    if "log_alpha" not in temperatura:

        raise RuntimeError(
            "No se creó el parámetro log_alpha."
        )

    if temperatura[
        "log_alpha"
    ].ndim != 0:

        raise RuntimeError(
            "log_alpha debe ser un tensor escalar."
        )

    if not temperatura[
        "log_alpha"
    ].requires_grad:

        raise RuntimeError(
            "log_alpha debe ser entrenable."
        )

    if not torch.isfinite(
        temperatura[
            "log_alpha"
        ]
    ).item():

        raise ValueError(
            "log_alpha no es finito."
        )

    return temperatura
def obtener_coeficiente_entropia_sac(
    temperatura,
):

    if not isinstance(
        temperatura,
        nn.ParameterDict,
    ):

        raise TypeError(
            "La temperatura SAC debe ser un "
            "nn.ParameterDict."
        )

    if "log_alpha" not in temperatura:

        raise KeyError(
            "La temperatura no contiene log_alpha."
        )

    log_alpha = temperatura[
        "log_alpha"
    ]

    if log_alpha.ndim != 0:

        raise ValueError(
            "log_alpha debe ser un tensor escalar."
        )

    if not torch.isfinite(
        log_alpha
    ).item():

        raise ValueError(
            "log_alpha contiene un valor no finito."
        )

    # alpha siempre será positivo.

    alpha = torch.exp(
        log_alpha
    )

    if not torch.isfinite(
        alpha
    ).item():

        raise ValueError(
            "El coeficiente alpha no es finito."
        )

    if alpha.item() <= 0.0:

        raise ValueError(
            "El coeficiente alpha debe ser positivo."
        )

    return alpha
def preparar_coeficiente_entropia_sac(
    coeficiente_entropia,
    dispositivo,
    tipo_datos,
):

    # ------------------------------------------------------
    # 1. Convertir alpha a tensor escalar
    # ------------------------------------------------------

    if torch.is_tensor(
        coeficiente_entropia
    ):

        if coeficiente_entropia.numel() != 1:

            raise ValueError(
                "El coeficiente de entropía debe contener "
                "un único valor."
            )

        # Se separa del grafo porque las pérdidas del actor
        # y de los críticos no deben modificar log_alpha.

        alpha = (
            coeficiente_entropia
            .reshape(
                ()
            )
            .detach()
            .to(
                device=dispositivo,
                dtype=tipo_datos,
            )
        )

    else:

        coeficiente_entropia = float(
            coeficiente_entropia
        )

        if not np.isfinite(
            coeficiente_entropia
        ):

            raise ValueError(
                "El coeficiente de entropía debe ser finito."
            )

        alpha = torch.tensor(
            coeficiente_entropia,
            dtype=tipo_datos,
            device=dispositivo,
        )

    # ------------------------------------------------------
    # 2. Verificar el resultado
    # ------------------------------------------------------

    if alpha.ndim != 0:

        raise ValueError(
            "Alpha debe ser un tensor escalar."
        )

    if not torch.isfinite(
        alpha
    ).item():

        raise ValueError(
            "Alpha contiene un valor no finito."
        )

    if alpha.item() <= 0.0:

        raise ValueError(
            "Alpha debe ser estrictamente positivo."
        )

    if alpha.requires_grad:

        raise RuntimeError(
            "Alpha debe estar desacoplado durante esta "
            "actualización."
        )

    return alpha
def crear_optimizador_temperatura_sac(
    temperatura,
    tasa_aprendizaje=(
        TASA_APRENDIZAJE_ALPHA_SAC
    ),
):

    if not isinstance(
        temperatura,
        nn.ParameterDict,
    ):

        raise TypeError(
            "La temperatura debe ser un "
            "nn.ParameterDict."
        )

    if "log_alpha" not in temperatura:

        raise KeyError(
            "La temperatura no contiene log_alpha."
        )

    tasa_aprendizaje = float(
        tasa_aprendizaje
    )

    if not np.isfinite(
        tasa_aprendizaje
    ):

        raise ValueError(
            "La tasa de aprendizaje debe ser finita."
        )

    if tasa_aprendizaje <= 0.0:

        raise ValueError(
            "La tasa de aprendizaje debe ser positiva."
        )

    parametros_entrenables = [
        parametro
        for parametro in temperatura.parameters()
        if parametro.requires_grad
    ]

    if len(
        parametros_entrenables
    ) != 1:

        raise RuntimeError(
            "La temperatura SAC debe contener exactamente "
            "un parámetro entrenable."
        )

    optimizador_alpha = torch.optim.Adam(
        params=parametros_entrenables,

        lr=tasa_aprendizaje,

        betas=(
            BETA_1_ADAM_ALPHA_SAC,
            BETA_2_ADAM_ALPHA_SAC,
        ),

        eps=EPSILON_ADAM_ALPHA_SAC,

        weight_decay=0.0,
    )

    parametros_optimizador = [
        parametro
        for grupo in optimizador_alpha.param_groups
        for parametro in grupo[
            "params"
        ]
    ]

    if len(
        parametros_optimizador
    ) != 1:

        raise RuntimeError(
            "El optimizador de alpha debe contener "
            "exactamente un parámetro."
        )

    if id(
        parametros_optimizador[
            0
        ]
    ) != id(
        temperatura[
            "log_alpha"
        ]
    ):

        raise RuntimeError(
            "El optimizador no contiene el parámetro "
            "log_alpha correcto."
        )

    return optimizador_alpha
def calcular_perdida_temperatura_sac(
    temperatura,
    log_probabilidad,
    entropia_objetivo=(
        ENTROPIA_OBJETIVO_SAC
    ),
):

    # ------------------------------------------------------
    # 1. Verificar la temperatura
    # ------------------------------------------------------

    if not isinstance(
        temperatura,
        nn.ParameterDict,
    ):

        raise TypeError(
            "La temperatura debe ser un "
            "nn.ParameterDict."
        )

    if "log_alpha" not in temperatura:

        raise KeyError(
            "La temperatura no contiene log_alpha."
        )

    # ------------------------------------------------------
    # 2. Verificar la log-probabilidad
    # ------------------------------------------------------

    if not torch.is_tensor(
        log_probabilidad
    ):

        raise TypeError(
            "La log-probabilidad debe ser un tensor."
        )

    if log_probabilidad.ndim != 2:

        raise ValueError(
            "La log-probabilidad debe tener forma "
            "(lote, 1)."
        )

    if log_probabilidad.shape[1] != 1:

        raise ValueError(
            "Debe existir una log-probabilidad "
            "por elemento del lote."
        )

    if not torch.isfinite(
        log_probabilidad
    ).all().item():

        raise ValueError(
            "La log-probabilidad contiene NaN "
            "o valores infinitos."
        )

    # ------------------------------------------------------
    # 3. Verificar la entropía objetivo
    # ------------------------------------------------------

    entropia_objetivo = float(
        entropia_objetivo
    )

    if not np.isfinite(
        entropia_objetivo
    ):

        raise ValueError(
            "La entropía objetivo debe ser finita."
        )

    # ------------------------------------------------------
    # 4. Obtener log_alpha
    # ------------------------------------------------------

    log_alpha = temperatura[
        "log_alpha"
    ]

    if (
        log_alpha.device
        != log_probabilidad.device
    ):

        raise ValueError(
            "log_alpha y la log-probabilidad deben "
            "estar en el mismo dispositivo."
        )

    # ------------------------------------------------------
    # 5. Calcular el error de entropía
    # ------------------------------------------------------
    #
    # Se utiliza detach() para impedir que esta pérdida
    # modifique el actor.
    #
    # Solamente log_alpha debe recibir gradientes.

    error_entropia = (
        log_probabilidad
        + entropia_objetivo
    ).detach()

    # ------------------------------------------------------
    # 6. Calcular la pérdida
    # ------------------------------------------------------

    perdida_temperatura = -(
        log_alpha
        * error_entropia
    ).mean()

    alpha = torch.exp(
        log_alpha
    )

    if perdida_temperatura.ndim != 0:

        raise ValueError(
            "La pérdida de temperatura debe ser escalar."
        )

    if not torch.isfinite(
        perdida_temperatura
    ).item():

        raise ValueError(
            "La pérdida de temperatura no es finita."
        )

    if not torch.isfinite(
        error_entropia
    ).all().item():

        raise ValueError(
            "El error de entropía no es finito."
        )

    return {
        "perdida_temperatura": (
            perdida_temperatura
        ),

        "error_entropia": error_entropia,

        "error_entropia_promedio": (
            error_entropia.mean()
        ),

        "log_alpha": log_alpha,

        "alpha": alpha,
    }
def actualizar_temperatura_sac(
    temperatura,
    optimizador_alpha,
    log_probabilidad,
    entropia_objetivo=(
        ENTROPIA_OBJETIVO_SAC
    ),
):

    if not isinstance(
        optimizador_alpha,
        torch.optim.Optimizer,
    ):

        raise TypeError(
            "El optimizador de alpha no es válido."
        )

    # ------------------------------------------------------
    # 1. Guardar los valores anteriores
    # ------------------------------------------------------

    log_alpha_antes = temperatura[
        "log_alpha"
    ].detach().clone()

    alpha_antes = torch.exp(
        log_alpha_antes
    )

    # ------------------------------------------------------
    # 2. Limpiar gradientes
    # ------------------------------------------------------

    optimizador_alpha.zero_grad(
        set_to_none=True
    )

    # ------------------------------------------------------
    # 3. Calcular la pérdida
    # ------------------------------------------------------

    resultados_perdida = (
        calcular_perdida_temperatura_sac(
            temperatura=temperatura,

            log_probabilidad=(
                log_probabilidad
            ),

            entropia_objetivo=(
                entropia_objetivo
            ),
        )
    )

    perdida_temperatura = (
        resultados_perdida[
            "perdida_temperatura"
        ]
    )

    # ------------------------------------------------------
    # 4. Retropropagación
    # ------------------------------------------------------

    perdida_temperatura.backward()

    gradiente_log_alpha = temperatura[
        "log_alpha"
    ].grad

    if gradiente_log_alpha is None:

        raise RuntimeError(
            "log_alpha no recibió gradiente."
        )

    gradiente_finito = torch.isfinite(
        gradiente_log_alpha
    ).item()

    if not gradiente_finito:

        raise ValueError(
            "El gradiente de log_alpha no es finito."
        )

    valor_gradiente = (
        gradiente_log_alpha
        .detach()
        .clone()
    )

    # ------------------------------------------------------
    # 5. Actualizar log_alpha
    # ------------------------------------------------------

    optimizador_alpha.step()

    log_alpha_despues = temperatura[
        "log_alpha"
    ].detach().clone()

    alpha_despues = torch.exp(
        log_alpha_despues
    )

    # ------------------------------------------------------
    # 6. Verificar valores posteriores
    # ------------------------------------------------------

    if not torch.isfinite(
        log_alpha_despues
    ).item():

        raise ValueError(
            "log_alpha dejó de ser finito."
        )

    if not torch.isfinite(
        alpha_despues
    ).item():

        raise ValueError(
            "alpha dejó de ser finito."
        )

    if alpha_despues.item() <= 0.0:

        raise ValueError(
            "alpha debe seguir siendo positivo."
        )

    return {
        "perdida_temperatura": (
            perdida_temperatura.detach()
        ),

        "error_entropia": (
            resultados_perdida[
                "error_entropia"
            ].detach()
        ),

        "error_entropia_promedio": (
            resultados_perdida[
                "error_entropia_promedio"
            ].detach()
        ),

        "log_alpha_antes": (
            log_alpha_antes
        ),

        "log_alpha_despues": (
            log_alpha_despues
        ),

        "alpha_antes": alpha_antes,

        "alpha_despues": alpha_despues,

        "gradiente_log_alpha": (
            valor_gradiente
        ),

        "gradiente_finito": (
            gradiente_finito
        ),
    }
def validar_observacion_buffer_sac(
    observacion,
):

    if not isinstance(
        observacion,
        dict,
    ):

        raise TypeError(
            "La observación debe ser un diccionario."
        )

    claves_necesarias = {
        "parche",
        "escalares",
    }

    if not claves_necesarias.issubset(
        observacion.keys()
    ):

        raise KeyError(
            "La observación debe contener las claves "
            "'parche' y 'escalares'."
        )

    parche = np.asarray(
        observacion[
            "parche"
        ],
        dtype=np.float32,
    )

    escalares = np.asarray(
        observacion[
            "escalares"
        ],
        dtype=np.float32,
    )

    forma_parche_esperada = (
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if parche.shape != forma_parche_esperada:

        raise ValueError(
            "El parche tiene forma "
            f"{parche.shape}; se esperaba "
            f"{forma_parche_esperada}."
        )

    if escalares.shape != (
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "Los escalares tienen forma "
            f"{escalares.shape}; se esperaba "
            f"({DIMENSION_ESCALARES_SAC},)."
        )

    if not np.isfinite(
        parche
    ).all():

        raise ValueError(
            "El parche contiene NaN o infinitos."
        )

    if not np.isfinite(
        escalares
    ).all():

        raise ValueError(
            "Los escalares contienen NaN o infinitos."
        )

    # El parche se codificará como uint8.
    # Por eso debe estar normalizado en [0, 1].

    tolerancia = 1e-6

    if parche.min() < -tolerancia:

        raise ValueError(
            "El parche contiene valores menores que cero."
        )

    if parche.max() > 1.0 + tolerancia:

        raise ValueError(
            "El parche contiene valores mayores que uno."
        )

    parche = np.clip(
        parche,
        0.0,
        1.0,
    )

    return (
        parche,
        escalares,
    )
def convertir_accion_buffer_sac(
    accion,
):

    if torch.is_tensor(
        accion
    ):

        accion = (
            accion
            .detach()
            .cpu()
            .numpy()
        )

    accion = np.asarray(
        accion,
        dtype=np.float32,
    )

    # Aceptar tanto:
    #
    # (2,)
    #
    # como:
    #
    # (1, 2)

    if accion.shape == (
        1,
        DIMENSION_ACCION_SAC,
    ):

        accion = accion[
            0
        ]

    if accion.shape != (
        DIMENSION_ACCION_SAC,
    ):

        raise ValueError(
            "La acción debe tener forma "
            f"({DIMENSION_ACCION_SAC},)."
        )

    if not np.isfinite(
        accion
    ).all():

        raise ValueError(
            "La acción contiene NaN o infinitos."
        )

    if np.any(
        accion < -1.0
    ):

        raise ValueError(
            "La acción contiene valores menores que -1."
        )

    if np.any(
        accion > 1.0
    ):

        raise ValueError(
            "La acción contiene valores mayores que 1."
        )

    return accion
def crear_buffer_repeticion_sac(
    capacidad=CAPACIDAD_BUFFER_REPETICION_SAC,
    semilla=SEMILLA,
):

    capacidad = int(
        capacidad
    )

    semilla = int(
        semilla
    )

    if capacidad <= 0:

        raise ValueError(
            "La capacidad del buffer debe ser positiva."
        )

    forma_parches = (
        capacidad,
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    forma_escalares = (
        capacidad,
        DIMENSION_ESCALARES_SAC,
    )

    forma_acciones = (
        capacidad,
        DIMENSION_ACCION_SAC,
    )

    forma_escalares_individuales = (
        capacidad,
        1,
    )

    buffer = {
        # Observaciones actuales.

        "parches": np.zeros(
            forma_parches,
            dtype=np.uint8,
        ),

        "escalares": np.zeros(
            forma_escalares,
            dtype=np.float32,
        ),

        # Acciones y resultados.

        "acciones": np.zeros(
            forma_acciones,
            dtype=np.float32,
        ),

        "recompensas": np.zeros(
            forma_escalares_individuales,
            dtype=np.float32,
        ),

        # Observaciones siguientes.

        "siguientes_parches": np.zeros(
            forma_parches,
            dtype=np.uint8,
        ),

        "siguientes_escalares": np.zeros(
            forma_escalares,
            dtype=np.float32,
        ),

        # Indicadores terminales.

        "terminados": np.zeros(
            forma_escalares_individuales,
            dtype=np.float32,
        ),

        # Estado circular del buffer.

        "capacidad": capacidad,

        "tamano": 0,

        "indice_escritura": 0,

        "total_transiciones_agregadas": 0,

        # Generador independiente para el muestreo.

        "generador": np.random.default_rng(
            semilla
        ),

        "semilla": semilla,
    }

    return buffer
def agregar_transicion_buffer_sac(
    buffer,
    observacion,
    accion,
    recompensa,
    siguiente_observacion,
    terminado,
):

    if not isinstance(
        buffer,
        dict,
    ):

        raise TypeError(
            "El buffer debe ser un diccionario."
        )

    claves_necesarias = {
        "parches",
        "escalares",
        "acciones",
        "recompensas",
        "siguientes_parches",
        "siguientes_escalares",
        "terminados",
        "capacidad",
        "tamano",
        "indice_escritura",
        "total_transiciones_agregadas",
    }

    if not claves_necesarias.issubset(
        buffer.keys()
    ):

        raise KeyError(
            "El buffer no contiene todas las estructuras "
            "necesarias."
        )

    (
        parche,
        escalares,
    ) = validar_observacion_buffer_sac(
        observacion=observacion
    )

    (
        siguiente_parche,
        siguientes_escalares,
    ) = validar_observacion_buffer_sac(
        observacion=siguiente_observacion
    )

    accion = convertir_accion_buffer_sac(
        accion=accion
    )

    recompensa = float(
        recompensa
    )

    if not np.isfinite(
        recompensa
    ):

        raise ValueError(
            "La recompensa debe ser finita."
        )

    if isinstance(
        terminado,
        np.ndarray,
    ):

        if terminado.size != 1:

            raise ValueError(
                "El indicador terminado debe contener "
                "un único valor."
            )

        terminado = terminado.item()

    terminado = float(
        terminado
    )

    if terminado not in (
        0.0,
        1.0,
    ):

        raise ValueError(
            "El indicador terminado debe ser 0, 1, "
            "False o True."
        )

    indice = int(
        buffer[
            "indice_escritura"
        ]
    )

    capacidad = int(
        buffer[
            "capacidad"
        ]
    )

    # ------------------------------------------------------
    # Codificar los parches como uint8
    # ------------------------------------------------------

    parche_codificado = np.rint(
        parche
        * 255.0
    ).astype(
        np.uint8
    )

    siguiente_parche_codificado = np.rint(
        siguiente_parche
        * 255.0
    ).astype(
        np.uint8
    )

    # ------------------------------------------------------
    # Escribir la transición
    # ------------------------------------------------------

    buffer[
        "parches"
    ][
        indice
    ] = parche_codificado

    buffer[
        "escalares"
    ][
        indice
    ] = escalares

    buffer[
        "acciones"
    ][
        indice
    ] = accion

    buffer[
        "recompensas"
    ][
        indice,
        0
    ] = recompensa

    buffer[
        "siguientes_parches"
    ][
        indice
    ] = siguiente_parche_codificado

    buffer[
        "siguientes_escalares"
    ][
        indice
    ] = siguientes_escalares

    buffer[
        "terminados"
    ][
        indice,
        0
    ] = terminado

    # ------------------------------------------------------
    # Actualizar el índice circular
    # ------------------------------------------------------

    buffer[
        "indice_escritura"
    ] = (
        indice + 1
    ) % capacidad

    buffer[
        "tamano"
    ] = min(
        int(
            buffer[
                "tamano"
            ]
        )
        + 1,
        capacidad,
    )

    buffer[
        "total_transiciones_agregadas"
    ] = (
        int(
            buffer[
                "total_transiciones_agregadas"
            ]
        )
        + 1
    )
def muestrear_lote_buffer_sac(
    buffer,
    tamano_lote=TAMANO_LOTE_SAC,
    dispositivo=DISPOSITIVO_SAC,
):

    if not isinstance(
        buffer,
        dict,
    ):

        raise TypeError(
            "El buffer debe ser un diccionario."
        )

    if "generador" not in buffer:

        raise KeyError(
            "El buffer no contiene su generador aleatorio."
        )

    tamano_lote = int(
        tamano_lote
    )

    if tamano_lote <= 0:

        raise ValueError(
            "El tamaño de lote debe ser positivo."
        )

    tamano_actual = int(
        buffer[
            "tamano"
        ]
    )

    if tamano_actual < tamano_lote:

        raise ValueError(
            "El buffer contiene "
            f"{tamano_actual} transiciones, pero se "
            f"solicitaron {tamano_lote}."
        )

    # ------------------------------------------------------
    # Elegir índices sin reemplazo
    # ------------------------------------------------------

    indices = buffer[
        "generador"
    ].choice(
        tamano_actual,
        size=tamano_lote,
        replace=False,
    )

    # ------------------------------------------------------
    # Decodificar los parches
    # ------------------------------------------------------

    parches_numpy = (
        buffer[
            "parches"
        ][
            indices
        ]
        .astype(
            np.float32
        )
        / 255.0
    )

    siguientes_parches_numpy = (
        buffer[
            "siguientes_parches"
        ][
            indices
        ]
        .astype(
            np.float32
        )
        / 255.0
    )

    # ------------------------------------------------------
    # Convertir el lote a tensores
    # ------------------------------------------------------

    lote = {
        "parches": torch.from_numpy(
            parches_numpy
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "escalares": torch.from_numpy(
            buffer[
                "escalares"
            ][
                indices
            ].copy()
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "acciones": torch.from_numpy(
            buffer[
                "acciones"
            ][
                indices
            ].copy()
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "recompensas": torch.from_numpy(
            buffer[
                "recompensas"
            ][
                indices
            ].copy()
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "siguientes_parches": torch.from_numpy(
            siguientes_parches_numpy
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "siguientes_escalares": torch.from_numpy(
            buffer[
                "siguientes_escalares"
            ][
                indices
            ].copy()
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "terminados": torch.from_numpy(
            buffer[
                "terminados"
            ][
                indices
            ].copy()
        ).to(
            device=dispositivo,
            dtype=torch.float32,
        ),

        "indices": torch.from_numpy(
            indices.astype(
                np.int64
            )
        ).to(
            device=dispositivo,
            dtype=torch.long,
        ),
    }

    # Los datos del buffer no deben requerir gradientes.

    for clave, tensor in lote.items():

        if tensor.requires_grad:

            raise RuntimeError(
                f"El tensor '{clave}' no debe requerir "
                "gradientes."
            )

    return lote
def entrenar_lote_sac(
    lote,
    actor,
    critico_q1,
    critico_q2,
    critico_q1_objetivo,
    critico_q2_objetivo,
    temperatura,
    optimizador_actor,
    optimizador_q1,
    optimizador_q2,
    optimizador_alpha,
    factor_descuento=FACTOR_DESCUENTO_SAC,
    tau=TAU_POLYAK_SAC,
    entropia_objetivo=ENTROPIA_OBJETIVO_SAC,
):

    # ------------------------------------------------------
    # 1. Verificar el lote
    # ------------------------------------------------------

    if not isinstance(
        lote,
        dict,
    ):

        raise TypeError(
            "El lote SAC debe ser un diccionario."
        )

    claves_necesarias = {
        "parches",
        "escalares",
        "acciones",
        "recompensas",
        "siguientes_parches",
        "siguientes_escalares",
        "terminados",
    }

    if not claves_necesarias.issubset(
        lote.keys()
    ):

        claves_faltantes = (
            claves_necesarias
            - set(
                lote.keys()
            )
        )

        raise KeyError(
            "Faltan elementos en el lote SAC: "
            f"{claves_faltantes}."
        )

    parches = lote[
        "parches"
    ]

    escalares = lote[
        "escalares"
    ]

    acciones = lote[
        "acciones"
    ]

    recompensas = lote[
        "recompensas"
    ]

    siguientes_parches = lote[
        "siguientes_parches"
    ]

    siguientes_escalares = lote[
        "siguientes_escalares"
    ]

    terminados = lote[
        "terminados"
    ]

    tensores_lote = [
        parches,
        escalares,
        acciones,
        recompensas,
        siguientes_parches,
        siguientes_escalares,
        terminados,
    ]

    if not all(
        torch.is_tensor(
            tensor
        )
        for tensor in tensores_lote
    ):

        raise TypeError(
            "Todos los elementos del lote SAC deben "
            "ser tensores de PyTorch."
        )

    # ------------------------------------------------------
    # 2. Verificar dimensiones
    # ------------------------------------------------------

    if parches.ndim != 4:

        raise ValueError(
            "Los parches deben tener forma "
            "(lote, canales, alto, ancho)."
        )

    tamano_lote = parches.shape[
        0
    ]

    if tamano_lote <= 0:

        raise ValueError(
            "El lote SAC no puede estar vacío."
        )

    forma_parches_esperada = (
        tamano_lote,
        CANALES_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
        RESOLUCION_PARCHE_SAC,
    )

    if tuple(
        parches.shape
    ) != forma_parches_esperada:

        raise ValueError(
            "Los parches actuales tienen una forma "
            "incorrecta."
        )

    if tuple(
        siguientes_parches.shape
    ) != forma_parches_esperada:

        raise ValueError(
            "Los siguientes parches tienen una forma "
            "incorrecta."
        )

    if tuple(
        escalares.shape
    ) != (
        tamano_lote,
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "Los escalares actuales tienen una forma "
            "incorrecta."
        )

    if tuple(
        siguientes_escalares.shape
    ) != (
        tamano_lote,
        DIMENSION_ESCALARES_SAC,
    ):

        raise ValueError(
            "Los siguientes escalares tienen una forma "
            "incorrecta."
        )

    if tuple(
        acciones.shape
    ) != (
        tamano_lote,
        DIMENSION_ACCION_SAC,
    ):

        raise ValueError(
            "Las acciones tienen una forma incorrecta."
        )

    if tuple(
        recompensas.shape
    ) != (
        tamano_lote,
        1,
    ):

        raise ValueError(
            "Las recompensas deben tener forma "
            "(lote, 1)."
        )

    if tuple(
        terminados.shape
    ) != (
        tamano_lote,
        1,
    ):

        raise ValueError(
            "Los indicadores terminados deben tener "
            "forma (lote, 1)."
        )

    # ------------------------------------------------------
    # 3. Verificar dispositivo y tipo
    # ------------------------------------------------------

    dispositivo = parches.device

    if not all(
        tensor.device == dispositivo
        for tensor in tensores_lote
    ):

        raise ValueError(
            "Todos los tensores del lote deben estar "
            "en el mismo dispositivo."
        )

    if not all(
        tensor.dtype == torch.float32
        for tensor in tensores_lote
    ):

        raise TypeError(
            "Todos los tensores del lote deben ser "
            "torch.float32."
        )

    if not all(
        torch.isfinite(
            tensor
        ).all().item()
        for tensor in tensores_lote
    ):

        raise ValueError(
            "El lote contiene NaN o valores infinitos."
        )

    if not torch.all(
        acciones >= -1.0
    ).item():

        raise ValueError(
            "El lote contiene acciones menores que -1."
        )

    if not torch.all(
        acciones <= 1.0
    ).item():

        raise ValueError(
            "El lote contiene acciones mayores que 1."
        )

    if not torch.all(
        terminados >= 0.0
    ).item():

        raise ValueError(
            "El lote contiene indicadores terminales "
            "negativos."
        )

    if not torch.all(
        terminados <= 1.0
    ).item():

        raise ValueError(
            "El lote contiene indicadores terminales "
            "mayores que uno."
        )

    # ------------------------------------------------------
    # 4. Preparar las redes
    # ------------------------------------------------------

    actor.train()

    critico_q1.train()

    critico_q2.train()

    critico_q1_objetivo.eval()

    critico_q2_objetivo.eval()

    # Alpha que se utilizará en el objetivo temporal y
    # en la actualización del actor.

    alpha_antes_actualizacion = (
        obtener_coeficiente_entropia_sac(
            temperatura=temperatura
        )
    )

    alpha_usado_objetivo = (
        alpha_antes_actualizacion
        .detach()
        .clone()
    )

    # ------------------------------------------------------
    # 5. Calcular la acción del siguiente estado
    # ------------------------------------------------------
    #
    # El objetivo temporal no conserva gradientes.

    with torch.no_grad():

        resultados_actor_siguiente = (
            ejecutar_actor_sac(
                tensor_parche=(
                    siguientes_parches
                ),

                tensor_escalares=(
                    siguientes_escalares
                ),

                rama_cnn=actor[
                    "rama_cnn"
                ],

                rama_mlp=actor[
                    "rama_mlp"
                ],

                red_compartida=actor[
                    "red_compartida"
                ],

                cabeza_media=actor[
                    "cabeza_media"
                ],

                cabeza_log_desviacion=(
                    actor[
                        "cabeza_log_desviacion"
                    ]
                ),
            )
        )

        accion_siguiente = (
            resultados_actor_siguiente[
                "accion_normalizada"
            ]
        )

        log_probabilidad_siguiente = (
            resultados_actor_siguiente[
                "log_probabilidad_corregida"
            ]
        )

        resultados_criticos_objetivo = (
            ejecutar_dos_criticos_sac(
                tensor_parche=(
                    siguientes_parches
                ),

                tensor_escalares=(
                    siguientes_escalares
                ),

                tensor_accion=(
                    accion_siguiente
                ),

                critico_q1=(
                    critico_q1_objetivo
                ),

                critico_q2=(
                    critico_q2_objetivo
                ),
            )
        )

        valor_q_objetivo_minimo = (
            resultados_criticos_objetivo[
                "valor_q_minimo"
            ]
        )

    # ------------------------------------------------------
    # 6. Calcular el objetivo temporal SAC
    # ------------------------------------------------------

    objetivo_q = calcular_objetivo_criticos_sac(
        recompensas=recompensas,

        terminados=terminados,

        valor_q_objetivo_minimo=(
            valor_q_objetivo_minimo
        ),

        log_probabilidad_siguiente=(
            log_probabilidad_siguiente
        ),

        factor_descuento=(
            factor_descuento
        ),

        coeficiente_entropia=(
            alpha_antes_actualizacion
        ),
    )

    if objetivo_q.requires_grad:

        raise RuntimeError(
            "El objetivo temporal SAC no debe "
            "conservar gradientes."
        )

    # ------------------------------------------------------
    # 7. Evaluar Q1(s,a) y Q2(s,a)
    # ------------------------------------------------------
    #
    # Aquí se usan las acciones almacenadas en el replay
    # buffer, no acciones nuevas del actor.

    resultados_criticos_actuales = (
        ejecutar_dos_criticos_sac(
            tensor_parche=parches,

            tensor_escalares=escalares,

            tensor_accion=acciones,

            critico_q1=critico_q1,

            critico_q2=critico_q2,
        )
    )

    valor_q1_actual = (
        resultados_criticos_actuales[
            "valor_q1"
        ]
    )

    valor_q2_actual = (
        resultados_criticos_actuales[
            "valor_q2"
        ]
    )

    # ------------------------------------------------------
    # 8. Actualizar Q1 y Q2
    # ------------------------------------------------------

    resultados_actualizacion_criticos = (
        actualizar_dos_criticos_sac(
            valor_q1=valor_q1_actual,

            valor_q2=valor_q2_actual,

            objetivo_q=objetivo_q,

            critico_q1=critico_q1,

            critico_q2=critico_q2,

            optimizador_q1=(
                optimizador_q1
            ),

            optimizador_q2=(
                optimizador_q2
            ),
        )
    )

    # ------------------------------------------------------
    # 9. Actualizar el actor
    # ------------------------------------------------------
    #
    # Esta actualización vuelve a ejecutar Q1 y Q2,
    # ahora con acciones generadas por el actor.

    resultados_actualizacion_actor = (
        actualizar_actor_sac(
            tensor_parche=parches,

            tensor_escalares=escalares,

            actor=actor,

            critico_q1=critico_q1,

            critico_q2=critico_q2,

            optimizador_actor=(
                optimizador_actor
            ),

            temperatura=temperatura,
        )
    )

    # Alpha usado por el actor antes de actualizar
    # log_alpha.

    alpha_usado_actor = (
        resultados_actualizacion_actor[
            "coeficiente_entropia"
        ]
        .detach()
        .clone()
    )

    # ------------------------------------------------------
    # 10. Actualizar la temperatura alpha
    # ------------------------------------------------------
    #
    # La log-probabilidad regresada por actualizar_actor_sac
    # ya está separada del grafo del actor.

    resultados_actualizacion_alpha = (
        actualizar_temperatura_sac(
            temperatura=temperatura,

            optimizador_alpha=(
                optimizador_alpha
            ),

            log_probabilidad=(
                resultados_actualizacion_actor[
                    "log_probabilidad"
                ]
            ),

            entropia_objetivo=(
                entropia_objetivo
            ),
        )
    )

    alpha_despues_actualizacion = (
        obtener_coeficiente_entropia_sac(
            temperatura=temperatura
        )
        .detach()
        .clone()
    )

    # ------------------------------------------------------
    # 11. Actualizar los críticos objetivo
    # ------------------------------------------------------

    actualizar_dos_criticos_objetivo_sac(
        critico_q1=critico_q1,

        critico_q2=critico_q2,

        critico_q1_objetivo=(
            critico_q1_objetivo
        ),

        critico_q2_objetivo=(
            critico_q2_objetivo
        ),

        tau=tau,
    )

    # ------------------------------------------------------
    # 12. Verificaciones posteriores
    # ------------------------------------------------------

    objetivos_congelados = all(
        not parametro.requires_grad
        for parametro in (
            critico_q1_objetivo.parameters()
        )
    ) and all(
        not parametro.requires_grad
        for parametro in (
            critico_q2_objetivo.parameters()
        )
    )

    objetivos_en_evaluacion = (
        not critico_q1_objetivo.training
        and not critico_q2_objetivo.training
    )

    resultados_finitos = all(
        [
            torch.isfinite(
                objetivo_q
            ).all().item(),

            torch.isfinite(
                resultados_actualizacion_criticos[
                    "perdida_q1"
                ]
            ).item(),

            torch.isfinite(
                resultados_actualizacion_criticos[
                    "perdida_q2"
                ]
            ).item(),

            torch.isfinite(
                resultados_actualizacion_actor[
                    "perdida_actor"
                ]
            ).item(),

            torch.isfinite(
                resultados_actualizacion_alpha[
                    "perdida_temperatura"
                ]
            ).item(),

            torch.isfinite(
                alpha_despues_actualizacion
            ).item(),
        ]
    )

    if not resultados_finitos:

        raise ValueError(
            "El entrenamiento SAC produjo valores "
            "no finitos."
        )

    # ------------------------------------------------------
    # 13. Regresar métricas del paso completo
    # ------------------------------------------------------

    return {
        "tamano_lote": tamano_lote,

        "objetivo_q": (
            objetivo_q.detach()
        ),

        "accion_siguiente": (
            accion_siguiente.detach()
        ),

        "log_probabilidad_siguiente": (
            log_probabilidad_siguiente.detach()
        ),

        "valor_q_objetivo_minimo": (
            valor_q_objetivo_minimo.detach()
        ),

        "valor_q1_antes": (
            valor_q1_actual.detach()
        ),

        "valor_q2_antes": (
            valor_q2_actual.detach()
        ),

        "perdida_q1": (
            resultados_actualizacion_criticos[
                "perdida_q1"
            ]
        ),

        "perdida_q2": (
            resultados_actualizacion_criticos[
                "perdida_q2"
            ]
        ),

        "perdida_total_criticos": (
            resultados_actualizacion_criticos[
                "perdida_total"
            ]
        ),

        "norma_gradientes_q1": (
            resultados_actualizacion_criticos[
                "norma_gradientes_q1"
            ]
        ),

        "norma_gradientes_q2": (
            resultados_actualizacion_criticos[
                "norma_gradientes_q2"
            ]
        ),

        "gradientes_q1_completos": (
            resultados_actualizacion_criticos[
                "gradientes_q1_completos"
            ]
        ),

        "gradientes_q2_completos": (
            resultados_actualizacion_criticos[
                "gradientes_q2_completos"
            ]
        ),

        "gradientes_q1_finitos": (
            resultados_actualizacion_criticos[
                "gradientes_q1_finitos"
            ]
        ),

        "gradientes_q2_finitos": (
            resultados_actualizacion_criticos[
                "gradientes_q2_finitos"
            ]
        ),

        "perdida_actor": (
            resultados_actualizacion_actor[
                "perdida_actor"
            ]
        ),

        "accion_actor": (
            resultados_actualizacion_actor[
                "accion_actor"
            ]
        ),

        "log_probabilidad_actor": (
            resultados_actualizacion_actor[
                "log_probabilidad"
            ]
        ),

        "norma_gradientes_actor": (
            resultados_actualizacion_actor[
                "norma_gradientes_actor"
            ]
        ),

        "gradientes_actor_completos": (
            resultados_actualizacion_actor[
                "gradientes_actor_completos"
            ]
        ),

        "gradientes_actor_finitos": (
            resultados_actualizacion_actor[
                "gradientes_actor_finitos"
            ]
        ),

        "perdida_temperatura": (
            resultados_actualizacion_alpha[
                "perdida_temperatura"
            ]
        ),

        "error_entropia_promedio": (
            resultados_actualizacion_alpha[
                "error_entropia_promedio"
            ]
        ),

        "gradiente_log_alpha": (
            resultados_actualizacion_alpha[
                "gradiente_log_alpha"
            ]
        ),

        "gradiente_alpha_finito": (
            resultados_actualizacion_alpha[
                "gradiente_finito"
            ]
        ),

        "alpha_usado_objetivo": (
            alpha_usado_objetivo
        ),

        "alpha_usado_actor": (
            alpha_usado_actor
        ),

        "alpha_antes": (
            resultados_actualizacion_alpha[
                "alpha_antes"
            ]
        ),

        "alpha_despues": (
            alpha_despues_actualizacion
        ),

        "objetivos_congelados": (
            objetivos_congelados
        ),

        "objetivos_en_evaluacion": (
            objetivos_en_evaluacion
        ),

        "resultados_finitos": (
            resultados_finitos
        ),
    }
def entrenar_episodio_sac(
    numero_episodio,
    semilla,
    actor,
    critico_q1,
    critico_q2,
    critico_q1_objetivo,
    critico_q2_objetivo,
    temperatura,
    optimizador_actor,
    optimizador_q1,
    optimizador_q2,
    optimizador_alpha,
    buffer,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    tamano_lote=TAMANO_LOTE_SAC,
    transiciones_aleatorias_iniciales=(
        TRANSICIONES_ALEATORIAS_INICIALES_SAC
    ),
    actualizaciones_por_paso=(
        ACTUALIZACIONES_POR_PASO_SAC
    ),
    factor_descuento=FACTOR_DESCUENTO_SAC,
    tau=TAU_POLYAK_SAC,
    entropia_objetivo=ENTROPIA_OBJETIVO_SAC,
    dt=DT,
    dispositivo=DISPOSITIVO_SAC,
):

    # ------------------------------------------------------
    # 1. Validaciones generales
    # ------------------------------------------------------

    numero_episodio = int(
        numero_episodio
    )

    semilla = int(
        semilla
    )

    pasos_maximos = int(
        pasos_maximos
    )

    tamano_lote = int(
        tamano_lote
    )

    transiciones_aleatorias_iniciales = int(
        transiciones_aleatorias_iniciales
    )

    actualizaciones_por_paso = int(
        actualizaciones_por_paso
    )

    dt = float(
        dt
    )

    if numero_episodio < 0:

        raise ValueError(
            "El número de episodio no puede ser negativo."
        )

    if pasos_maximos <= 0:

        raise ValueError(
            "El número máximo de pasos debe ser positivo."
        )

    if tamano_lote <= 0:

        raise ValueError(
            "El tamaño del lote debe ser positivo."
        )

    if transiciones_aleatorias_iniciales < 0:

        raise ValueError(
            "Las transiciones aleatorias iniciales no "
            "pueden ser negativas."
        )

    if actualizaciones_por_paso < 0:

        raise ValueError(
            "Las actualizaciones por paso no pueden "
            "ser negativas."
        )

    if not np.isfinite(
        dt
    ) or dt <= 0.0:

        raise ValueError(
            "El periodo de muestreo debe ser positivo."
        )

    if not isinstance(
        buffer,
        dict,
    ):

        raise TypeError(
            "El replay buffer debe ser un diccionario."
        )

    if "generador" not in buffer:

        raise KeyError(
            "El replay buffer no contiene su generador."
        )

    if tamano_lote > buffer[
        "capacidad"
    ]:

        raise ValueError(
            "El tamaño del lote no puede superar la "
            "capacidad del replay buffer."
        )

    modelos = [
        actor,
        critico_q1,
        critico_q2,
        critico_q1_objetivo,
        critico_q2_objetivo,
        temperatura,
    ]

    if not all(
        isinstance(
            modelo,
            nn.Module,
        )
        for modelo in modelos
    ):

        raise TypeError(
            "Todos los modelos SAC deben ser módulos "
            "de PyTorch."
        )

    optimizadores = [
        optimizador_actor,
        optimizador_q1,
        optimizador_q2,
        optimizador_alpha,
    ]

    if not all(
        isinstance(
            optimizador,
            torch.optim.Optimizer,
        )
        for optimizador in optimizadores
    ):

        raise TypeError(
            "Todos los optimizadores SAC deben ser válidos."
        )

    # ------------------------------------------------------
    # 2. Normalizar el dispositivo
    # ------------------------------------------------------
    #
    # Se toma como referencia el dispositivo real del actor.
    # Esto también evita el problema cuda contra cuda:0.

    dispositivo_actor = next(
        actor.parameters()
    ).device

    dispositivo_solicitado = torch.device(
        dispositivo
    )

    mismo_tipo_dispositivo = (
        dispositivo_solicitado.type
        == dispositivo_actor.type
    )

    indice_compatible = (
        dispositivo_solicitado.index is None
        or dispositivo_solicitado.index
        == dispositivo_actor.index
    )

    if not (
        mismo_tipo_dispositivo
        and indice_compatible
    ):

        raise ValueError(
            "El dispositivo solicitado no coincide con "
            "el dispositivo real del actor."
        )

    dispositivo = dispositivo_actor

    for modelo in modelos:

        dispositivos_modelo = {
            parametro.device
            for parametro in modelo.parameters()
        }

        if len(
            dispositivos_modelo
        ) != 1:

            raise RuntimeError(
                "Cada modelo SAC debe estar completamente "
                "en un único dispositivo."
            )

        if next(
            iter(
                dispositivos_modelo
            )
        ) != dispositivo:

            raise ValueError(
                "Todos los modelos SAC deben estar en el "
                "mismo dispositivo."
            )

    # ------------------------------------------------------
    # 3. Reiniciar el entorno
    # ------------------------------------------------------

    (
        entorno,
        observacion,
    ) = reiniciar_entorno_sac(
        semilla=semilla
    )

    observacion_inicial = {
        "parche": observacion[
            "parche"
        ].copy(),

        "escalares": observacion[
            "escalares"
        ].copy(),
    }

    # ------------------------------------------------------
    # 4. Historiales de navegación
    # ------------------------------------------------------

    acciones = []

    modos_accion = []

    informaciones_accion = []

    recompensas = []

    recompensas_acumuladas = []

    componentes_recompensa = []

    informaciones_paso = []

    submetas_usadas = []

    submetas_nuevas = []

    indices_progreso = []

    clearances_estaticos = []

    clearances_dinamicos = []

    clearances_totales = []

    velocidades_lineales = []

    velocidades_angulares = []

    indicadores_terminado = []

    indicadores_truncado = []

    # ------------------------------------------------------
    # 5. Historiales de entrenamiento
    # ------------------------------------------------------

    pasos_actualizacion = []

    perdidas_q1 = []

    perdidas_q2 = []

    perdidas_totales_criticos = []

    perdidas_actor = []

    perdidas_temperatura = []

    normas_gradientes_q1 = []

    normas_gradientes_q2 = []

    normas_gradientes_actor = []

    objetivos_q_promedio = []

    log_probabilidades_actor = []

    errores_entropia = []

    valores_alpha = []

    recompensa_acumulada = 0.0

    numero_actualizaciones = 0

    numero_acciones_aleatorias = 0

    numero_acciones_actor = 0

    terminado = False

    truncado = False

    # ------------------------------------------------------
    # 6. Ciclo del episodio
    # ------------------------------------------------------

    while not (
        terminado
        or truncado
    ):

        observacion_actual = {
            "parche": observacion[
                "parche"
            ].copy(),

            "escalares": observacion[
                "escalares"
            ].copy(),
        }

        total_transiciones = int(
            buffer[
                "total_transiciones_agregadas"
            ]
        )

        usar_accion_aleatoria = (
            total_transiciones
            < transiciones_aleatorias_iniciales
        )

        # --------------------------------------------------
        # 6.1. Seleccionar acción
        # --------------------------------------------------

        if usar_accion_aleatoria:

            accion = buffer[
                "generador"
            ].uniform(
                low=-1.0,
                high=1.0,
                size=(
                    DIMENSION_ACCION_SAC,
                ),
            ).astype(
                np.float32
            )

            (
                velocidad_lineal,
                velocidad_angular,
            ) = convertir_accion_sac_a_control(
                accion=accion
            )

            informacion_accion = {
                "modo": "aleatoria_inicial",

                "determinista": False,

                "accion": accion.copy(),

                "media": None,

                "desviacion": None,

                "log_probabilidad": None,

                "velocidad_lineal": (
                    velocidad_lineal
                ),

                "velocidad_angular": (
                    velocidad_angular
                ),

                "dispositivo": str(
                    dispositivo
                ),
            }

            numero_acciones_aleatorias += 1

        else:

            (
                accion,
                informacion_accion,
            ) = seleccionar_accion_actor_sac(
                observacion=observacion_actual,

                actor=actor,

                dispositivo=dispositivo,

                determinista=False,
            )

            numero_acciones_actor += 1

        # --------------------------------------------------
        # 6.2. Ejecutar el paso del entorno
        # --------------------------------------------------

        (
            siguiente_observacion,
            recompensa,
            terminado,
            truncado,
            informacion_paso,
        ) = ejecutar_paso_entorno_sac(
            entorno=entorno,

            accion=accion,

            pasos_maximos=pasos_maximos,

            dt=dt,
        )

        # --------------------------------------------------
        # 6.3. Guardar transición
        # --------------------------------------------------
        #
        # El timeout no se almacena como terminal verdadero.
        # Así el objetivo Q puede continuar haciendo
        # bootstrap cuando el episodio solamente terminó por
        # el límite artificial de tiempo.

        agregar_transicion_buffer_sac(
            buffer=buffer,

            observacion=observacion_actual,

            accion=accion,

            recompensa=recompensa,

            siguiente_observacion=(
                siguiente_observacion
            ),

            terminado=float(
                terminado
            ),
        )

        # --------------------------------------------------
        # 6.4. Registrar información de navegación
        # --------------------------------------------------

        recompensa_acumulada += float(
            recompensa
        )

        acciones.append(
            accion.copy()
        )

        modos_accion.append(
            informacion_accion[
                "modo"
            ]
        )

        informaciones_accion.append(
            informacion_accion.copy()
        )

        recompensas.append(
            float(
                recompensa
            )
        )

        recompensas_acumuladas.append(
            float(
                recompensa_acumulada
            )
        )

        componentes_recompensa.append(
            informacion_paso[
                "componentes_recompensa"
            ].copy()
        )

        informaciones_paso.append(
            informacion_paso.copy()
        )

        submetas_usadas.append(
            tuple(
                informacion_paso[
                    "submeta_usada"
                ]
            )
        )

        submetas_nuevas.append(
            tuple(
                informacion_paso[
                    "submeta_nueva"
                ]
            )
        )

        indices_progreso.append(
            int(
                informacion_paso[
                    "indice_progreso"
                ]
            )
        )

        clearances_estaticos.append(
            float(
                informacion_paso[
                    "clearance_estatico"
                ]
            )
        )

        clearances_dinamicos.append(
            float(
                informacion_paso[
                    "clearance_dinamico"
                ]
            )
        )

        clearances_totales.append(
            float(
                informacion_paso[
                    "clearance_total"
                ]
            )
        )

        velocidades_lineales.append(
            float(
                informacion_paso[
                    "velocidad_lineal"
                ]
            )
        )

        velocidades_angulares.append(
            float(
                informacion_paso[
                    "velocidad_angular"
                ]
            )
        )

        indicadores_terminado.append(
            bool(
                terminado
            )
        )

        indicadores_truncado.append(
            bool(
                truncado
            )
        )

        # --------------------------------------------------
        # 6.5. Entrenar con el replay buffer
        # --------------------------------------------------

        buffer_suficiente = (
            int(
                buffer[
                    "tamano"
                ]
            )
            >= tamano_lote
        )

        exploracion_inicial_completada = (
            int(
                buffer[
                    "total_transiciones_agregadas"
                ]
            )
            >= transiciones_aleatorias_iniciales
        )

        puede_entrenar = (
            buffer_suficiente
            and exploracion_inicial_completada
            and actualizaciones_por_paso > 0
        )

        if puede_entrenar:

            for _ in range(
                actualizaciones_por_paso
            ):

                lote = muestrear_lote_buffer_sac(
                    buffer=buffer,

                    tamano_lote=tamano_lote,

                    dispositivo=dispositivo,
                )

                resultados_entrenamiento = (
                    entrenar_lote_sac(
                        lote=lote,

                        actor=actor,

                        critico_q1=critico_q1,

                        critico_q2=critico_q2,

                        critico_q1_objetivo=(
                            critico_q1_objetivo
                        ),

                        critico_q2_objetivo=(
                            critico_q2_objetivo
                        ),

                        temperatura=temperatura,

                        optimizador_actor=(
                            optimizador_actor
                        ),

                        optimizador_q1=(
                            optimizador_q1
                        ),

                        optimizador_q2=(
                            optimizador_q2
                        ),

                        optimizador_alpha=(
                            optimizador_alpha
                        ),

                        factor_descuento=(
                            factor_descuento
                        ),

                        tau=tau,

                        entropia_objetivo=(
                            entropia_objetivo
                        ),
                    )
                )

                numero_actualizaciones += 1

                pasos_actualizacion.append(
                    int(
                        entorno[
                            "paso_actual"
                        ]
                    )
                )

                perdidas_q1.append(
                    float(
                        resultados_entrenamiento[
                            "perdida_q1"
                        ].item()
                    )
                )

                perdidas_q2.append(
                    float(
                        resultados_entrenamiento[
                            "perdida_q2"
                        ].item()
                    )
                )

                perdidas_totales_criticos.append(
                    float(
                        resultados_entrenamiento[
                            "perdida_total_criticos"
                        ].item()
                    )
                )

                perdidas_actor.append(
                    float(
                        resultados_entrenamiento[
                            "perdida_actor"
                        ].item()
                    )
                )

                perdidas_temperatura.append(
                    float(
                        resultados_entrenamiento[
                            "perdida_temperatura"
                        ].item()
                    )
                )

                normas_gradientes_q1.append(
                    float(
                        resultados_entrenamiento[
                            "norma_gradientes_q1"
                        ]
                    )
                )

                normas_gradientes_q2.append(
                    float(
                        resultados_entrenamiento[
                            "norma_gradientes_q2"
                        ]
                    )
                )

                normas_gradientes_actor.append(
                    float(
                        resultados_entrenamiento[
                            "norma_gradientes_actor"
                        ]
                    )
                )

                objetivos_q_promedio.append(
                    float(
                        resultados_entrenamiento[
                            "objetivo_q"
                        ].mean().item()
                    )
                )

                log_probabilidades_actor.append(
                    float(
                        resultados_entrenamiento[
                            "log_probabilidad_actor"
                        ].mean().item()
                    )
                )

                errores_entropia.append(
                    float(
                        resultados_entrenamiento[
                            "error_entropia_promedio"
                        ].item()
                    )
                )

                valores_alpha.append(
                    float(
                        resultados_entrenamiento[
                            "alpha_despues"
                        ].item()
                    )
                )

        # --------------------------------------------------
        # 6.6. Avanzar observación
        # --------------------------------------------------

        observacion = {
            "parche": siguiente_observacion[
                "parche"
            ].copy(),

            "escalares": siguiente_observacion[
                "escalares"
            ].copy(),
        }

    # ------------------------------------------------------
    # 7. Obtener registro y métricas estándar
    # ------------------------------------------------------

    registro = entorno[
        "registro"
    ]

    escenario = entorno[
        "escenario"
    ]

    metricas = calcular_metricas_episodio_estandar(
        registro=registro,

        camino_mundo=entorno[
            "camino_mundo"
        ],

        meta=entorno[
            "meta"
        ],

        longitud_camino_astar=escenario[
            "longitud_camino"
        ],

        dt=dt,
    )

    alpha_final = (
        obtener_coeficiente_entropia_sac(
            temperatura=temperatura
        )
        .detach()
        .item()
    )

    metricas[
        "metodo"
    ] = "sac_entrenamiento"

    metricas[
        "numero_episodio"
    ] = numero_episodio

    metricas[
        "semilla"
    ] = semilla

    metricas[
        "recompensa_acumulada"
    ] = float(
        recompensa_acumulada
    )

    metricas[
        "recompensa_promedio"
    ] = (
        float(
            np.mean(
                recompensas
            )
        )
        if len(
            recompensas
        ) > 0
        else 0.0
    )

    metricas[
        "numero_actualizaciones"
    ] = numero_actualizaciones

    metricas[
        "acciones_aleatorias"
    ] = numero_acciones_aleatorias

    metricas[
        "acciones_actor"
    ] = numero_acciones_actor

    metricas[
        "alpha_final"
    ] = float(
        alpha_final
    )

    metricas[
        "tamano_buffer_final"
    ] = int(
        buffer[
            "tamano"
        ]
    )

    metricas[
        "total_transiciones_buffer"
    ] = int(
        buffer[
            "total_transiciones_agregadas"
        ]
    )

    # ------------------------------------------------------
    # 8. Historial preparado para las gráficas
    # ------------------------------------------------------

    historial_navegacion = {
        "acciones": acciones,

        "modos_accion": modos_accion,

        "informaciones_accion": (
            informaciones_accion
        ),

        "recompensas": recompensas,

        "recompensas_acumuladas": (
            recompensas_acumuladas
        ),

        "componentes_recompensa": (
            componentes_recompensa
        ),

        "informaciones_paso": (
            informaciones_paso
        ),

        "submetas_usadas": submetas_usadas,

        "submetas_nuevas": submetas_nuevas,

        "indices_progreso": indices_progreso,

        "clearances_estaticos": (
            clearances_estaticos
        ),

        "clearances_dinamicos": (
            clearances_dinamicos
        ),

        "clearances_totales": (
            clearances_totales
        ),

        "velocidades_lineales": (
            velocidades_lineales
        ),

        "velocidades_angulares": (
            velocidades_angulares
        ),

        "terminados": (
            indicadores_terminado
        ),

        "truncados": (
            indicadores_truncado
        ),
    }

    historial_entrenamiento = {
        "pasos_actualizacion": (
            pasos_actualizacion
        ),

        "perdidas_q1": perdidas_q1,

        "perdidas_q2": perdidas_q2,

        "perdidas_totales_criticos": (
            perdidas_totales_criticos
        ),

        "perdidas_actor": perdidas_actor,

        "perdidas_temperatura": (
            perdidas_temperatura
        ),

        "normas_gradientes_q1": (
            normas_gradientes_q1
        ),

        "normas_gradientes_q2": (
            normas_gradientes_q2
        ),

        "normas_gradientes_actor": (
            normas_gradientes_actor
        ),

        "objetivos_q_promedio": (
            objetivos_q_promedio
        ),

        "log_probabilidades_actor": (
            log_probabilidades_actor
        ),

        "errores_entropia": (
            errores_entropia
        ),

        "valores_alpha": valores_alpha,
    }

    # ------------------------------------------------------
    # 9. Resultado completo del episodio
    # ------------------------------------------------------

    resultado_episodio = {
        "numero_episodio": numero_episodio,

        "semilla": semilla,

        "entorno": entorno,

        "escenario": escenario,

        "registro": registro,

        "observacion_inicial": (
            observacion_inicial
        ),

        "observacion_final": {
            "parche": observacion[
                "parche"
            ].copy(),

            "escalares": observacion[
                "escalares"
            ].copy(),
        },

        "terminado": bool(
            terminado
        ),

        "truncado": bool(
            truncado
        ),

        "resultado": entorno[
            "resultado"
        ],

        "recompensa_acumulada": float(
            recompensa_acumulada
        ),

        "numero_actualizaciones": (
            numero_actualizaciones
        ),

        "numero_acciones_aleatorias": (
            numero_acciones_aleatorias
        ),

        "numero_acciones_actor": (
            numero_acciones_actor
        ),

        "alpha_final": float(
            alpha_final
        ),

        "historial_navegacion": (
            historial_navegacion
        ),

        "historial_entrenamiento": (
            historial_entrenamiento
        ),

        "metricas": metricas,
    }

    return resultado_episodio
def entrenar_agente_sac(
    numero_episodios,
    semilla_base,
    actor,
    critico_q1,
    critico_q2,
    critico_q1_objetivo,
    critico_q2_objetivo,
    temperatura,
    optimizador_actor,
    optimizador_q1,
    optimizador_q2,
    optimizador_alpha,
    buffer,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    tamano_lote=TAMANO_LOTE_SAC,
    transiciones_aleatorias_iniciales=(
        TRANSICIONES_ALEATORIAS_INICIALES_SAC
    ),
    actualizaciones_por_paso=(
        ACTUALIZACIONES_POR_PASO_SAC
    ),
    factor_descuento=FACTOR_DESCUENTO_SAC,
    tau=TAU_POLYAK_SAC,
    entropia_objetivo=ENTROPIA_OBJETIVO_SAC,
    ventana_promedio=VENTANA_PROMEDIO_MOVIL_SAC,
    frecuencia_impresion=(
        FRECUENCIA_IMPRESION_ENTRENAMIENTO_SAC
    ),
    guardar_episodios_completos=False,
    dt=DT,
    dispositivo=DISPOSITIVO_SAC,
):

    # ------------------------------------------------------
    # 1. Validar parámetros generales
    # ------------------------------------------------------

    numero_episodios = int(
        numero_episodios
    )

    semilla_base = int(
        semilla_base
    )

    pasos_maximos = int(
        pasos_maximos
    )

    tamano_lote = int(
        tamano_lote
    )

    transiciones_aleatorias_iniciales = int(
        transiciones_aleatorias_iniciales
    )

    actualizaciones_por_paso = int(
        actualizaciones_por_paso
    )

    ventana_promedio = int(
        ventana_promedio
    )

    frecuencia_impresion = int(
        frecuencia_impresion
    )

    if numero_episodios <= 0:

        raise ValueError(
            "El número de episodios debe ser positivo."
        )

    if pasos_maximos <= 0:

        raise ValueError(
            "El número máximo de pasos debe ser positivo."
        )

    if tamano_lote <= 0:

        raise ValueError(
            "El tamaño del lote debe ser positivo."
        )

    if ventana_promedio <= 0:

        raise ValueError(
            "La ventana del promedio móvil debe ser "
            "positiva."
        )

    if frecuencia_impresion < 0:

        raise ValueError(
            "La frecuencia de impresión no puede ser "
            "negativa."
        )

    if not isinstance(
        guardar_episodios_completos,
        (
            bool,
            np.bool_,
        ),
    ):

        raise TypeError(
            "guardar_episodios_completos debe ser "
            "booleano."
        )

    guardar_episodios_completos = bool(
        guardar_episodios_completos
    )

    if not isinstance(
        buffer,
        dict,
    ):

        raise TypeError(
            "El replay buffer debe ser un diccionario."
        )

    # ------------------------------------------------------
    # 2. Historial global para las futuras gráficas
    # ------------------------------------------------------

    historial_global = {
        "episodios": [],

        "semillas": [],

        "resultados": [],

        "exitos": [],

        "recompensas_acumuladas": [],

        "recompensas_promedio": [],

        "recompensas_promedio_movil": [],

        "tasas_exito_acumuladas": [],

        "tasas_exito_moviles": [],

        "pasos_ejecutados": [],

        "pasos_promedio_movil": [],

        "tiempos_navegacion": [],

        "longitudes_recorridas": [],

        "distancias_finales_meta": [],

        "errores_medios_ruta": [],

        "errores_rmse_ruta": [],

        "clearances_totales_minimos": [],

        "clearances_totales_promedio": [],

        "variaciones_totales_v": [],

        "variaciones_totales_omega": [],

        "esfuerzos_control": [],

        "alphas_finales": [],

        "actualizaciones_por_episodio": [],

        "actualizaciones_acumuladas": [],

        "tamanos_buffer": [],

        "transiciones_totales_buffer": [],

        "acciones_aleatorias": [],

        "acciones_actor": [],

        "perdidas_q1_promedio": [],

        "perdidas_q2_promedio": [],

        "perdidas_criticos_promedio": [],

        "perdidas_actor_promedio": [],

        "perdidas_temperatura_promedio": [],

        "objetivos_q_promedio": [],

        "log_probabilidades_promedio": [],

        "errores_entropia_promedio": [],
    }

    # ------------------------------------------------------
    # 3. Contadores de resultados
    # ------------------------------------------------------

    conteos_resultados = {
        "meta": 0,

        "colision_estatica": 0,

        "colision_dinamica": 0,

        "fuera_mapa": 0,

        "timeout": 0,
    }

    episodios_completos = []

    mejor_resultado_episodio = None

    mejor_criterio = None

    ultimo_resultado_episodio = None

    actualizaciones_acumuladas = 0

    # ------------------------------------------------------
    # 4. Función interna para promediar listas
    # ------------------------------------------------------

    def promedio_seguro(
        valores,
    ):

        if len(
            valores
        ) == 0:

            return float(
                "nan"
            )

        valores_numpy = np.asarray(
            valores,
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                valores_numpy
            )
        ):

            return float(
                "nan"
            )

        return float(
            np.mean(
                valores_numpy
            )
        )

    # ------------------------------------------------------
    # 5. Ejecutar los episodios
    # ------------------------------------------------------

    for indice_episodio in range(
        numero_episodios
    ):

        numero_episodio = (
            indice_episodio + 1
        )

        semilla_episodio = (
            semilla_base
            + indice_episodio
        )

        resultado_episodio = (
            entrenar_episodio_sac(
                numero_episodio=(
                    numero_episodio
                ),

                semilla=(
                    semilla_episodio
                ),

                actor=actor,

                critico_q1=critico_q1,

                critico_q2=critico_q2,

                critico_q1_objetivo=(
                    critico_q1_objetivo
                ),

                critico_q2_objetivo=(
                    critico_q2_objetivo
                ),

                temperatura=temperatura,

                optimizador_actor=(
                    optimizador_actor
                ),

                optimizador_q1=(
                    optimizador_q1
                ),

                optimizador_q2=(
                    optimizador_q2
                ),

                optimizador_alpha=(
                    optimizador_alpha
                ),

                buffer=buffer,

                pasos_maximos=(
                    pasos_maximos
                ),

                tamano_lote=(
                    tamano_lote
                ),

                transiciones_aleatorias_iniciales=(
                    transiciones_aleatorias_iniciales
                ),

                actualizaciones_por_paso=(
                    actualizaciones_por_paso
                ),

                factor_descuento=(
                    factor_descuento
                ),

                tau=tau,

                entropia_objetivo=(
                    entropia_objetivo
                ),

                dt=dt,

                dispositivo=dispositivo,
            )
        )

        ultimo_resultado_episodio = (
            resultado_episodio
        )

        metricas = resultado_episodio[
            "metricas"
        ]

        historial_entrenamiento = (
            resultado_episodio[
                "historial_entrenamiento"
            ]
        )

        resultado = resultado_episodio[
            "resultado"
        ]

        exito = bool(
            metricas[
                "exito"
            ]
        )

        recompensa_acumulada = float(
            resultado_episodio[
                "recompensa_acumulada"
            ]
        )

        # --------------------------------------------------
        # 5.1. Actualizar contadores de resultados
        # --------------------------------------------------

        if resultado not in conteos_resultados:

            conteos_resultados[
                resultado
            ] = 0

        conteos_resultados[
            resultado
        ] += 1

        # --------------------------------------------------
        # 5.2. Seleccionar el mejor episodio
        # ------------------------------------------------------
        #
        # Primero se prioriza alcanzar la meta.
        # En igualdad de éxito, se elige la mayor recompensa.

        criterio_actual = (
            int(
                exito
            ),
            recompensa_acumulada,
        )

        if (
            mejor_criterio is None
            or criterio_actual
            > mejor_criterio
        ):

            mejor_criterio = (
                criterio_actual
            )

            mejor_resultado_episodio = (
                resultado_episodio
            )

        if guardar_episodios_completos:

            episodios_completos.append(
                resultado_episodio
            )

        # --------------------------------------------------
        # 5.3. Resumir pérdidas del episodio
        # --------------------------------------------------

        perdida_q1_media = promedio_seguro(
            historial_entrenamiento[
                "perdidas_q1"
            ]
        )

        perdida_q2_media = promedio_seguro(
            historial_entrenamiento[
                "perdidas_q2"
            ]
        )

        perdida_criticos_media = (
            promedio_seguro(
                historial_entrenamiento[
                    "perdidas_totales_criticos"
                ]
            )
        )

        perdida_actor_media = promedio_seguro(
            historial_entrenamiento[
                "perdidas_actor"
            ]
        )

        perdida_temperatura_media = (
            promedio_seguro(
                historial_entrenamiento[
                    "perdidas_temperatura"
                ]
            )
        )

        objetivo_q_medio = promedio_seguro(
            historial_entrenamiento[
                "objetivos_q_promedio"
            ]
        )

        log_probabilidad_media = (
            promedio_seguro(
                historial_entrenamiento[
                    "log_probabilidades_actor"
                ]
            )
        )

        error_entropia_medio = (
            promedio_seguro(
                historial_entrenamiento[
                    "errores_entropia"
                ]
            )
        )

        numero_actualizaciones = int(
            resultado_episodio[
                "numero_actualizaciones"
            ]
        )

        actualizaciones_acumuladas += (
            numero_actualizaciones
        )

        # --------------------------------------------------
        # 5.4. Agregar métricas al historial global
        # --------------------------------------------------

        historial_global[
            "episodios"
        ].append(
            numero_episodio
        )

        historial_global[
            "semillas"
        ].append(
            semilla_episodio
        )

        historial_global[
            "resultados"
        ].append(
            resultado
        )

        historial_global[
            "exitos"
        ].append(
            float(
                exito
            )
        )

        historial_global[
            "recompensas_acumuladas"
        ].append(
            recompensa_acumulada
        )

        historial_global[
            "recompensas_promedio"
        ].append(
            float(
                metricas[
                    "recompensa_promedio"
                ]
            )
        )

        historial_global[
            "pasos_ejecutados"
        ].append(
            int(
                metricas[
                    "pasos_ejecutados"
                ]
            )
        )

        historial_global[
            "tiempos_navegacion"
        ].append(
            float(
                metricas[
                    "tiempo_navegacion"
                ]
            )
        )

        historial_global[
            "longitudes_recorridas"
        ].append(
            float(
                metricas[
                    "longitud_recorrida"
                ]
            )
        )

        historial_global[
            "distancias_finales_meta"
        ].append(
            float(
                metricas[
                    "distancia_final_meta"
                ]
            )
        )

        historial_global[
            "errores_medios_ruta"
        ].append(
            float(
                metricas[
                    "error_medio_ruta"
                ]
            )
        )

        historial_global[
            "errores_rmse_ruta"
        ].append(
            float(
                metricas[
                    "error_rmse_ruta"
                ]
            )
        )

        historial_global[
            "clearances_totales_minimos"
        ].append(
            float(
                metricas[
                    "clearance_total_minimo"
                ]
            )
        )

        historial_global[
            "clearances_totales_promedio"
        ].append(
            float(
                metricas[
                    "clearance_total_promedio"
                ]
            )
        )

        historial_global[
            "variaciones_totales_v"
        ].append(
            float(
                metricas[
                    "variacion_total_v"
                ]
            )
        )

        historial_global[
            "variaciones_totales_omega"
        ].append(
            float(
                metricas[
                    "variacion_total_omega"
                ]
            )
        )

        historial_global[
            "esfuerzos_control"
        ].append(
            float(
                metricas[
                    "esfuerzo_control"
                ]
            )
        )

        historial_global[
            "alphas_finales"
        ].append(
            float(
                resultado_episodio[
                    "alpha_final"
                ]
            )
        )

        historial_global[
            "actualizaciones_por_episodio"
        ].append(
            numero_actualizaciones
        )

        historial_global[
            "actualizaciones_acumuladas"
        ].append(
            actualizaciones_acumuladas
        )

        historial_global[
            "tamanos_buffer"
        ].append(
            int(
                buffer[
                    "tamano"
                ]
            )
        )

        historial_global[
            "transiciones_totales_buffer"
        ].append(
            int(
                buffer[
                    "total_transiciones_agregadas"
                ]
            )
        )

        historial_global[
            "acciones_aleatorias"
        ].append(
            int(
                resultado_episodio[
                    "numero_acciones_aleatorias"
                ]
            )
        )

        historial_global[
            "acciones_actor"
        ].append(
            int(
                resultado_episodio[
                    "numero_acciones_actor"
                ]
            )
        )

        historial_global[
            "perdidas_q1_promedio"
        ].append(
            perdida_q1_media
        )

        historial_global[
            "perdidas_q2_promedio"
        ].append(
            perdida_q2_media
        )

        historial_global[
            "perdidas_criticos_promedio"
        ].append(
            perdida_criticos_media
        )

        historial_global[
            "perdidas_actor_promedio"
        ].append(
            perdida_actor_media
        )

        historial_global[
            "perdidas_temperatura_promedio"
        ].append(
            perdida_temperatura_media
        )

        historial_global[
            "objetivos_q_promedio"
        ].append(
            objetivo_q_medio
        )

        historial_global[
            "log_probabilidades_promedio"
        ].append(
            log_probabilidad_media
        )

        historial_global[
            "errores_entropia_promedio"
        ].append(
            error_entropia_medio
        )

        # --------------------------------------------------
        # 5.5. Calcular estadísticas móviles
        # --------------------------------------------------

        inicio_ventana = max(
            0,
            len(
                historial_global[
                    "episodios"
                ]
            )
            - ventana_promedio,
        )

        recompensas_ventana = (
            historial_global[
                "recompensas_acumuladas"
            ][
                inicio_ventana:
            ]
        )

        exitos_ventana = historial_global[
            "exitos"
        ][
            inicio_ventana:
        ]

        pasos_ventana = historial_global[
            "pasos_ejecutados"
        ][
            inicio_ventana:
        ]

        recompensa_media_movil = float(
            np.mean(
                recompensas_ventana
            )
        )

        tasa_exito_movil = float(
            np.mean(
                exitos_ventana
            )
        )

        pasos_media_movil = float(
            np.mean(
                pasos_ventana
            )
        )

        tasa_exito_acumulada = float(
            np.mean(
                historial_global[
                    "exitos"
                ]
            )
        )

        historial_global[
            "recompensas_promedio_movil"
        ].append(
            recompensa_media_movil
        )

        historial_global[
            "tasas_exito_moviles"
        ].append(
            tasa_exito_movil
        )

        historial_global[
            "tasas_exito_acumuladas"
        ].append(
            tasa_exito_acumulada
        )

        historial_global[
            "pasos_promedio_movil"
        ].append(
            pasos_media_movil
        )

        # --------------------------------------------------
        # 5.6. Mostrar progreso
        # --------------------------------------------------

        imprimir_episodio = (
            frecuencia_impresion > 0
            and (
                numero_episodio == 1
                or numero_episodio
                % frecuencia_impresion == 0
                or numero_episodio
                == numero_episodios
            )
        )

        if imprimir_episodio:

            print(
                f"Episodio "
                f"{numero_episodio:4d}/"
                f"{numero_episodios:4d} | "
                f"resultado = {resultado:18s} | "
                f"R = {recompensa_acumulada:9.3f} | "
                f"R_media = "
                f"{recompensa_media_movil:9.3f} | "
                f"éxito móvil = "
                f"{100.0 * tasa_exito_movil:6.2f}% | "
                f"pasos = "
                f"{metricas['pasos_ejecutados']:4d} | "
                f"alpha = "
                f"{resultado_episodio['alpha_final']:.5f} | "
                f"buffer = "
                f"{buffer['tamano']:6d}"
            )

    # ------------------------------------------------------
    # 6. Resumen global
    # ------------------------------------------------------

    numero_exitos = int(
        sum(
            historial_global[
                "exitos"
            ]
        )
    )

    tasa_exito_final = (
        numero_exitos
        / numero_episodios
    )

    numero_colisiones = (
        conteos_resultados.get(
            "colision_estatica",
            0,
        )
        +
        conteos_resultados.get(
            "colision_dinamica",
            0,
        )
    )

    resumen_global = {
        "numero_episodios": (
            numero_episodios
        ),

        "semilla_base": semilla_base,

        "numero_exitos": numero_exitos,

        "tasa_exito": float(
            tasa_exito_final
        ),

        "numero_colisiones": int(
            numero_colisiones
        ),

        "conteos_resultados": (
            conteos_resultados.copy()
        ),

        "recompensa_media": float(
            np.mean(
                historial_global[
                    "recompensas_acumuladas"
                ]
            )
        ),

        "recompensa_maxima": float(
            np.max(
                historial_global[
                    "recompensas_acumuladas"
                ]
            )
        ),

        "recompensa_minima": float(
            np.min(
                historial_global[
                    "recompensas_acumuladas"
                ]
            )
        ),

        "pasos_promedio": float(
            np.mean(
                historial_global[
                    "pasos_ejecutados"
                ]
            )
        ),

        "actualizaciones_totales": int(
            actualizaciones_acumuladas
        ),

        "tamano_buffer_final": int(
            buffer[
                "tamano"
            ]
        ),

        "transiciones_totales_buffer": int(
            buffer[
                "total_transiciones_agregadas"
            ]
        ),

        "alpha_final": float(
            obtener_coeficiente_entropia_sac(
                temperatura=temperatura
            )
            .detach()
            .item()
        ),
    }

    # ------------------------------------------------------
    # 7. Resultado completo del entrenamiento
    # ------------------------------------------------------

    resultado_entrenamiento = {
        "historial_global": historial_global,

        "resumen_global": resumen_global,

        "mejor_resultado_episodio": (
            mejor_resultado_episodio
        ),

        "ultimo_resultado_episodio": (
            ultimo_resultado_episodio
        ),

        "episodios_completos": (
            episodios_completos
        ),

        "actor": actor,

        "critico_q1": critico_q1,

        "critico_q2": critico_q2,

        "critico_q1_objetivo": (
            critico_q1_objetivo
        ),

        "critico_q2_objetivo": (
            critico_q2_objetivo
        ),

        "temperatura": temperatura,

        "buffer": buffer,
    }

    return resultado_entrenamiento
def obtener_serie_grafica_sac(
    historial,
    clave,
    numero_episodios,
    permitir_nan=False,
):

    if not isinstance(
        historial,
        dict,
    ):

        raise TypeError(
            "El historial SAC debe ser un diccionario."
        )

    if clave not in historial:

        raise KeyError(
            f"El historial SAC no contiene la clave "
            f"'{clave}'."
        )

    serie = np.asarray(
        historial[
            clave
        ],
        dtype=float,
    ).reshape(
        -1
    )

    if serie.size != numero_episodios:

        raise ValueError(
            f"La serie '{clave}' contiene "
            f"{serie.size} elementos; se esperaban "
            f"{numero_episodios}."
        )

    if np.any(
        np.isinf(
            serie
        )
    ):

        raise ValueError(
            f"La serie '{clave}' contiene valores "
            "infinitos."
        )

    if (
        not permitir_nan
        and np.any(
            np.isnan(
                serie
            )
        )
    ):

        raise ValueError(
            f"La serie '{clave}' contiene NaN."
        )

    return serie
def guardar_figuras_entrenamiento_sac(
    figuras,
    directorio_salida,
    dpi=DPI_GRAFICAS_ENTRENAMIENTO_SAC,
):

    if not isinstance(
        figuras,
        dict,
    ):

        raise TypeError(
            "Las figuras deben estar dentro de un "
            "diccionario."
        )

    if len(
        figuras
    ) == 0:

        raise ValueError(
            "No existen figuras para guardar."
        )

    dpi = int(
        dpi
    )

    if dpi <= 0:

        raise ValueError(
            "La resolución DPI debe ser positiva."
        )

    directorio = Path(
        directorio_salida
    )

    directorio.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas_guardadas = {}

    for nombre, figura in figuras.items():

        if not isinstance(
            figura,
            matplotlib.figure.Figure,
        ):

            raise TypeError(
                f"El elemento '{nombre}' no es una "
                "figura de Matplotlib."
            )

        ruta = directorio / (
            f"{nombre}.png"
        )

        figura.savefig(
            ruta,
            dpi=dpi,
            bbox_inches="tight",
        )

        rutas_guardadas[
            nombre
        ] = str(
            ruta
        )

    return rutas_guardadas
def graficar_entrenamiento_sac(
    resultado_entrenamiento,
    mostrar=True,
    directorio_salida=None,
    dpi=DPI_GRAFICAS_ENTRENAMIENTO_SAC,
):

    # ------------------------------------------------------
    # 1. Verificar el resultado del entrenamiento
    # ------------------------------------------------------

    if not isinstance(
        resultado_entrenamiento,
        dict,
    ):

        raise TypeError(
            "El resultado del entrenamiento debe ser "
            "un diccionario."
        )

    if (
        "historial_global"
        not in resultado_entrenamiento
    ):

        raise KeyError(
            "Falta el historial global del entrenamiento."
        )

    if (
        "resumen_global"
        not in resultado_entrenamiento
    ):

        raise KeyError(
            "Falta el resumen global del entrenamiento."
        )

    historial = resultado_entrenamiento[
        "historial_global"
    ]

    resumen = resultado_entrenamiento[
        "resumen_global"
    ]

    if not isinstance(
        historial,
        dict,
    ):

        raise TypeError(
            "El historial global debe ser un diccionario."
        )

    if not isinstance(
        resumen,
        dict,
    ):

        raise TypeError(
            "El resumen global debe ser un diccionario."
        )

    if "episodios" not in historial:

        raise KeyError(
            "El historial no contiene los episodios."
        )

    episodios = np.asarray(
        historial[
            "episodios"
        ],
        dtype=int,
    ).reshape(
        -1
    )

    numero_episodios = episodios.size

    if numero_episodios <= 0:

        raise ValueError(
            "No existen episodios para graficar."
        )

    if np.any(
        episodios <= 0
    ):

        raise ValueError(
            "Los números de episodio deben ser positivos."
        )

    if numero_episodios > 1:

        episodios_crecientes = np.all(
            np.diff(
                episodios
            )
            > 0
        )

        if not episodios_crecientes:

            raise ValueError(
                "Los episodios deben estar ordenados "
                "de forma estrictamente creciente."
            )

    # ------------------------------------------------------
    # 2. Obtener las series principales
    # ------------------------------------------------------

    recompensas = obtener_serie_grafica_sac(
        historial=historial,

        clave="recompensas_acumuladas",

        numero_episodios=numero_episodios,
    )

    recompensas_moviles = obtener_serie_grafica_sac(
        historial=historial,

        clave="recompensas_promedio_movil",

        numero_episodios=numero_episodios,
    )

    tasas_exito_acumuladas = (
        obtener_serie_grafica_sac(
            historial=historial,

            clave="tasas_exito_acumuladas",

            numero_episodios=numero_episodios,
        )
    )

    tasas_exito_moviles = (
        obtener_serie_grafica_sac(
            historial=historial,

            clave="tasas_exito_moviles",

            numero_episodios=numero_episodios,
        )
    )

    perdidas_q1 = obtener_serie_grafica_sac(
        historial=historial,

        clave="perdidas_q1_promedio",

        numero_episodios=numero_episodios,

        permitir_nan=True,
    )

    perdidas_q2 = obtener_serie_grafica_sac(
        historial=historial,

        clave="perdidas_q2_promedio",

        numero_episodios=numero_episodios,

        permitir_nan=True,
    )

    perdidas_actor = obtener_serie_grafica_sac(
        historial=historial,

        clave="perdidas_actor_promedio",

        numero_episodios=numero_episodios,

        permitir_nan=True,
    )

    alphas = obtener_serie_grafica_sac(
        historial=historial,

        clave="alphas_finales",

        numero_episodios=numero_episodios,
    )

    pasos = obtener_serie_grafica_sac(
        historial=historial,

        clave="pasos_ejecutados",

        numero_episodios=numero_episodios,
    )

    pasos_moviles = obtener_serie_grafica_sac(
        historial=historial,

        clave="pasos_promedio_movil",

        numero_episodios=numero_episodios,
    )

    # ------------------------------------------------------
    # 3. Verificaciones numéricas
    # ------------------------------------------------------

    if np.any(
        tasas_exito_acumuladas < 0.0
    ) or np.any(
        tasas_exito_acumuladas > 1.0
    ):

        raise ValueError(
            "La tasa de éxito acumulada debe estar "
            "entre cero y uno."
        )

    if np.any(
        tasas_exito_moviles < 0.0
    ) or np.any(
        tasas_exito_moviles > 1.0
    ):

        raise ValueError(
            "La tasa de éxito móvil debe estar entre "
            "cero y uno."
        )

    if np.any(
        alphas <= 0.0
    ):

        raise ValueError(
            "Todos los valores de alpha deben ser "
            "positivos."
        )

    if np.any(
        pasos < 0.0
    ):

        raise ValueError(
            "Los pasos ejecutados no pueden ser "
            "negativos."
        )

    figuras = {}

    # ======================================================
    # 4. RECOMPENSA
    # ======================================================

    figura_recompensa, ax_recompensa = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_recompensa.plot(
        episodios,
        recompensas,
        linewidth=1.2,
        alpha=0.65,
        label="Recompensa acumulada",
    )

    ax_recompensa.plot(
        episodios,
        recompensas_moviles,
        linewidth=2.4,
        label="Promedio móvil",
    )

    ax_recompensa.set_xlabel(
        "Episodio"
    )

    ax_recompensa.set_ylabel(
        "Recompensa"
    )

    ax_recompensa.set_title(
        "Aprendizaje de la política SAC"
    )

    ax_recompensa.grid(
        True,
        alpha=0.3,
    )

    ax_recompensa.legend()

    figura_recompensa.tight_layout()

    figuras[
        "01_recompensa_sac"
    ] = figura_recompensa

    # ======================================================
    # 5. TASA DE ÉXITO
    # ======================================================

    figura_exito, ax_exito = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_exito.plot(
        episodios,
        100.0
        * tasas_exito_acumuladas,
        linewidth=2.0,
        label="Tasa acumulada",
    )

    ax_exito.plot(
        episodios,
        100.0
        * tasas_exito_moviles,
        linewidth=2.0,
        label="Tasa móvil",
    )

    ax_exito.set_xlabel(
        "Episodio"
    )

    ax_exito.set_ylabel(
        "Tasa de éxito [%]"
    )

    ax_exito.set_title(
        "Tasa de llegada a la meta"
    )

    ax_exito.set_ylim(
        -2.0,
        102.0,
    )

    ax_exito.grid(
        True,
        alpha=0.3,
    )

    ax_exito.legend()

    figura_exito.tight_layout()

    figuras[
        "02_tasa_exito_sac"
    ] = figura_exito

    # ======================================================
    # 6. PÉRDIDAS
    # ======================================================

    figura_perdidas, ax_perdidas = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    mascara_q1 = np.isfinite(
        perdidas_q1
    )

    mascara_q2 = np.isfinite(
        perdidas_q2
    )

    mascara_actor = np.isfinite(
        perdidas_actor
    )

    existe_alguna_perdida = (
        np.any(
            mascara_q1
        )
        or np.any(
            mascara_q2
        )
        or np.any(
            mascara_actor
        )
    )

    if np.any(
        mascara_q1
    ):

        ax_perdidas.plot(
            episodios[
                mascara_q1
            ],

            perdidas_q1[
                mascara_q1
            ],

            linewidth=1.8,

            label="Pérdida Q1",
        )

    if np.any(
        mascara_q2
    ):

        ax_perdidas.plot(
            episodios[
                mascara_q2
            ],

            perdidas_q2[
                mascara_q2
            ],

            linewidth=1.8,

            label="Pérdida Q2",
        )

    if np.any(
        mascara_actor
    ):

        ax_perdidas.plot(
            episodios[
                mascara_actor
            ],

            perdidas_actor[
                mascara_actor
            ],

            linewidth=1.8,

            label="Pérdida del actor",
        )

    if existe_alguna_perdida:

        ax_perdidas.legend()

    else:

        ax_perdidas.text(
            0.5,
            0.5,
            (
                "Todavía no existen actualizaciones "
                "del agente"
            ),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_perdidas.transAxes,
        )

    ax_perdidas.set_xlabel(
        "Episodio"
    )

    ax_perdidas.set_ylabel(
        "Pérdida promedio"
    )

    ax_perdidas.set_title(
        "Pérdidas de las redes SAC"
    )

    ax_perdidas.grid(
        True,
        alpha=0.3,
    )

    figura_perdidas.tight_layout()

    figuras[
        "03_perdidas_sac"
    ] = figura_perdidas

    # ======================================================
    # 7. ALPHA
    # ======================================================

    figura_alpha, ax_alpha = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_alpha.plot(
        episodios,
        alphas,
        linewidth=2.0,
        label="Alpha aprendido",
    )

    ax_alpha.axhline(
        y=COEFICIENTE_ENTROPIA_INICIAL_SAC,
        linestyle="--",
        linewidth=1.5,
        label="Alpha inicial",
    )

    ax_alpha.set_xlabel(
        "Episodio"
    )

    ax_alpha.set_ylabel(
        "Coeficiente de entropía alpha"
    )

    ax_alpha.set_title(
        "Evolución de la exploración SAC"
    )

    ax_alpha.grid(
        True,
        alpha=0.3,
    )

    ax_alpha.legend()

    figura_alpha.tight_layout()

    figuras[
        "04_alpha_sac"
    ] = figura_alpha

    # ======================================================
    # 8. DURACIÓN DE LOS EPISODIOS
    # ======================================================

    figura_pasos, ax_pasos = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_pasos.plot(
        episodios,
        pasos,
        linewidth=1.2,
        alpha=0.65,
        label="Pasos ejecutados",
    )

    ax_pasos.plot(
        episodios,
        pasos_moviles,
        linewidth=2.4,
        label="Promedio móvil",
    )

    ax_pasos.axhline(
        y=PASOS_MAXIMOS_SEGUIMIENTO,
        linestyle="--",
        linewidth=1.5,
        label="Límite máximo configurado",
    )

    ax_pasos.set_xlabel(
        "Episodio"
    )

    ax_pasos.set_ylabel(
        "Pasos"
    )

    ax_pasos.set_title(
        "Duración de los episodios SAC"
    )

    ax_pasos.grid(
        True,
        alpha=0.3,
    )

    ax_pasos.legend()

    figura_pasos.tight_layout()

    figuras[
        "05_pasos_sac"
    ] = figura_pasos

    # ======================================================
    # 9. DISTRIBUCIÓN DE RESULTADOS
    # ======================================================

    if "conteos_resultados" not in resumen:

        raise KeyError(
            "El resumen global no contiene los conteos "
            "de resultados."
        )

    conteos_resultados = resumen[
        "conteos_resultados"
    ]

    orden_resultados = [
        "meta",
        "colision_estatica",
        "colision_dinamica",
        "fuera_mapa",
        "timeout",
    ]

    etiquetas_resultados = [
        "Meta",
        "Colisión\nestática",
        "Colisión\ndinámica",
        "Fuera del\nmapa",
        "Timeout",
    ]

    valores_resultados = [
        int(
            conteos_resultados.get(
                resultado,
                0,
            )
        )
        for resultado in orden_resultados
    ]

    if any(
        valor < 0
        for valor in valores_resultados
    ):

        raise ValueError(
            "Los conteos de resultados no pueden ser "
            "negativos."
        )

    if sum(
        valores_resultados
    ) != numero_episodios:

        raise ValueError(
            "La suma de los resultados no coincide con "
            "el número de episodios."
        )

    figura_resultados, ax_resultados = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    barras = ax_resultados.bar(
        etiquetas_resultados,
        valores_resultados,
    )

    maximo_conteo = max(
        valores_resultados
    )

    margen_texto = max(
        0.05
        * maximo_conteo,
        0.1,
    )

    for barra, valor in zip(
        barras,
        valores_resultados,
    ):

        ax_resultados.text(
            barra.get_x()
            + barra.get_width()
            / 2.0,

            barra.get_height()
            + margen_texto,

            str(
                valor
            ),

            horizontalalignment="center",

            verticalalignment="bottom",
        )

    ax_resultados.set_xlabel(
        "Resultado del episodio"
    )

    ax_resultados.set_ylabel(
        "Número de episodios"
    )

    ax_resultados.set_title(
        "Distribución de resultados del entrenamiento"
    )

    ax_resultados.set_ylim(
        0.0,
        max(
            1.0,
            maximo_conteo
            + 3.0
            * margen_texto,
        ),
    )

    ax_resultados.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figura_resultados.tight_layout()

    figuras[
        "06_resultados_sac"
    ] = figura_resultados

    # ------------------------------------------------------
    # 10. Guardar opcionalmente
    # ------------------------------------------------------

    rutas_guardadas = {}

    if directorio_salida is not None:

        rutas_guardadas = (
            guardar_figuras_entrenamiento_sac(
                figuras=figuras,

                directorio_salida=(
                    directorio_salida
                ),

                dpi=dpi,
            )
        )

    # ------------------------------------------------------
    # 11. Mostrar opcionalmente
    # ------------------------------------------------------

    if mostrar:

        plt.show()

    return {
        "figuras": figuras,

        "rutas_guardadas": rutas_guardadas,

        "numero_figuras": len(
            figuras
        ),

        "numero_episodios": (
            numero_episodios
        ),

        "hay_perdidas": (
            existe_alguna_perdida
        ),
    }
def evaluar_episodio_sac(
    semilla,
    actor,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
    dispositivo=DISPOSITIVO_SAC,
):

    # ------------------------------------------------------
    # 1. Validar entradas
    # ------------------------------------------------------

    semilla = int(
        semilla
    )

    pasos_maximos = int(
        pasos_maximos
    )

    dt = float(
        dt
    )

    if pasos_maximos <= 0:

        raise ValueError(
            "El número máximo de pasos debe ser positivo."
        )

    if (
        not np.isfinite(
            dt
        )
        or dt <= 0.0
    ):

        raise ValueError(
            "El periodo de muestreo debe ser positivo."
        )

    if not isinstance(
        actor,
        nn.ModuleDict,
    ):

        raise TypeError(
            "El actor SAC debe ser un nn.ModuleDict."
        )

    claves_actor_necesarias = {
        "rama_cnn",
        "rama_mlp",
        "red_compartida",
        "cabeza_media",
        "cabeza_log_desviacion",
    }

    if set(
        actor.keys()
    ) != claves_actor_necesarias:

        raise KeyError(
            "El actor no contiene todos los módulos "
            "necesarios."
        )

    parametros_actor = list(
        actor.parameters()
    )

    if len(
        parametros_actor
    ) == 0:

        raise RuntimeError(
            "El actor SAC no contiene parámetros."
        )

    # ------------------------------------------------------
    # 2. Verificar el dispositivo real
    # ------------------------------------------------------

    dispositivo_actor = parametros_actor[
        0
    ].device

    if not all(
        parametro.device
        == dispositivo_actor
        for parametro in parametros_actor
    ):

        raise RuntimeError(
            "Todos los parámetros del actor deben estar "
            "en el mismo dispositivo."
        )

    dispositivo_solicitado = torch.device(
        dispositivo
    )

    mismo_tipo_dispositivo = (
        dispositivo_solicitado.type
        == dispositivo_actor.type
    )

    indice_compatible = (
        dispositivo_solicitado.index is None
        or dispositivo_solicitado.index
        == dispositivo_actor.index
    )

    if not (
        mismo_tipo_dispositivo
        and indice_compatible
    ):

        raise ValueError(
            "El dispositivo solicitado no coincide con "
            "el dispositivo real del actor."
        )

    dispositivo = dispositivo_actor

    # ------------------------------------------------------
    # 3. Guardar el modo original del actor
    # ------------------------------------------------------

    modo_actor_anterior = actor.training

    # ------------------------------------------------------
    # 4. Reiniciar el entorno
    # ------------------------------------------------------

    (
        entorno,
        observacion,
    ) = reiniciar_entorno_sac(
        semilla=semilla
    )

    observacion_inicial = {
        "parche": observacion[
            "parche"
        ].copy(),

        "escalares": observacion[
            "escalares"
        ].copy(),
    }

    # ------------------------------------------------------
    # 5. Crear historiales de evaluación
    # ------------------------------------------------------

    acciones = []

    informaciones_accion = []

    recompensas = []

    recompensas_acumuladas = []

    componentes_recompensa = []

    informaciones_paso = []

    tiempos_pasos = []

    submetas_usadas = []

    submetas_nuevas = []

    indices_progreso = []

    distancias_meta = []

    distancias_submeta = []

    clearances_estaticos = []

    clearances_dinamicos = []

    clearances_totales = []

    velocidades_lineales = []

    velocidades_angulares = []

    recompensa_acumulada = 0.0

    terminado = False

    truncado = False

    try:

        # --------------------------------------------------
        # 6. Colocar el actor en modo evaluación
        # --------------------------------------------------

        actor.eval()

        # --------------------------------------------------
        # 7. Ejecutar el episodio completo
        # --------------------------------------------------

        while not (
            terminado
            or truncado
        ):

            observacion_actual = {
                "parche": observacion[
                    "parche"
                ].copy(),

                "escalares": observacion[
                    "escalares"
                ].copy(),
            }

            # ----------------------------------------------
            # 7.1. Acción determinista
            # ----------------------------------------------

            (
                accion,
                informacion_accion,
            ) = seleccionar_accion_actor_sac(
                observacion=observacion_actual,

                actor=actor,

                dispositivo=dispositivo,

                determinista=True,
            )

            if informacion_accion[
                "modo"
            ] != "determinista":

                raise RuntimeError(
                    "La evaluación debe utilizar solamente "
                    "acciones deterministas."
                )

            # ----------------------------------------------
            # 7.2. Ejecutar un paso del entorno
            # ----------------------------------------------

            (
                siguiente_observacion,
                recompensa,
                terminado,
                truncado,
                informacion_paso,
            ) = ejecutar_paso_entorno_sac(
                entorno=entorno,

                accion=accion,

                pasos_maximos=pasos_maximos,

                dt=dt,
            )

            # ----------------------------------------------
            # 7.3. Estado resultante
            # ----------------------------------------------

            estado_nuevo = entorno[
                "estado_robot"
            ]

            punto_robot = (
                estado_nuevo[
                    0
                ],
                estado_nuevo[
                    1
                ],
            )

            distancia_meta_actual = (
                distancia_entre_puntos(
                    punto_robot,

                    entorno[
                        "meta"
                    ],
                )
            )

            distancia_submeta_actual = (
                distancia_entre_puntos(
                    punto_robot,

                    informacion_paso[
                        "submeta_usada"
                    ],
                )
            )

            # ----------------------------------------------
            # 7.4. Registrar datos
            # ----------------------------------------------

            recompensa_acumulada += float(
                recompensa
            )

            acciones.append(
                accion.copy()
            )

            informaciones_accion.append(
                informacion_accion.copy()
            )

            recompensas.append(
                float(
                    recompensa
                )
            )

            recompensas_acumuladas.append(
                float(
                    recompensa_acumulada
                )
            )

            componentes_recompensa.append(
                informacion_paso[
                    "componentes_recompensa"
                ].copy()
            )

            informaciones_paso.append(
                informacion_paso.copy()
            )

            tiempos_pasos.append(
                float(
                    informacion_paso[
                        "tiempo"
                    ]
                )
            )

            submetas_usadas.append(
                tuple(
                    informacion_paso[
                        "submeta_usada"
                    ]
                )
            )

            submetas_nuevas.append(
                tuple(
                    informacion_paso[
                        "submeta_nueva"
                    ]
                )
            )

            indices_progreso.append(
                int(
                    informacion_paso[
                        "indice_progreso"
                    ]
                )
            )

            distancias_meta.append(
                float(
                    distancia_meta_actual
                )
            )

            distancias_submeta.append(
                float(
                    distancia_submeta_actual
                )
            )

            clearances_estaticos.append(
                float(
                    informacion_paso[
                        "clearance_estatico"
                    ]
                )
            )

            clearances_dinamicos.append(
                float(
                    informacion_paso[
                        "clearance_dinamico"
                    ]
                )
            )

            clearances_totales.append(
                float(
                    informacion_paso[
                        "clearance_total"
                    ]
                )
            )

            velocidades_lineales.append(
                float(
                    informacion_paso[
                        "velocidad_lineal"
                    ]
                )
            )

            velocidades_angulares.append(
                float(
                    informacion_paso[
                        "velocidad_angular"
                    ]
                )
            )

            # ----------------------------------------------
            # 7.5. Avanzar la observación
            # ----------------------------------------------

            observacion = {
                "parche": siguiente_observacion[
                    "parche"
                ].copy(),

                "escalares": siguiente_observacion[
                    "escalares"
                ].copy(),
            }

    finally:

        # --------------------------------------------------
        # 8. Restaurar el modo original del actor
        # --------------------------------------------------

        actor.train(
            modo_actor_anterior
        )

    # ------------------------------------------------------
    # 9. Obtener el registro estándar
    # ------------------------------------------------------

    registro = entorno[
        "registro"
    ]

    escenario = entorno[
        "escenario"
    ]

    camino_mundo = entorno[
        "camino_mundo"
    ]

    numero_pasos = int(
        registro[
            "pasos_ejecutados"
        ]
    )

    # ------------------------------------------------------
    # 10. Calcular métricas estándar
    # ------------------------------------------------------

    metricas = calcular_metricas_episodio_estandar(
        registro=registro,

        camino_mundo=camino_mundo,

        meta=entorno[
            "meta"
        ],

        longitud_camino_astar=escenario[
            "longitud_camino"
        ],

        dt=dt,
    )

    metricas[
        "metodo"
    ] = "sac"

    metricas[
        "modo_evaluacion"
    ] = "determinista"

    metricas[
        "semilla"
    ] = semilla

    metricas[
        "recompensa_acumulada"
    ] = float(
        recompensa_acumulada
    )

    metricas[
        "recompensa_promedio"
    ] = (
        float(
            np.mean(
                recompensas
            )
        )
        if len(
            recompensas
        ) > 0
        else 0.0
    )

    # ------------------------------------------------------
    # 11. Calcular errores temporales respecto de A*
    # ------------------------------------------------------

    errores_ruta = []

    for estado in registro[
        "estados"
    ]:

        punto = (
            estado[
                0
            ],
            estado[
                1
            ],
        )

        error_ruta = distancia_punto_camino(
            punto=punto,
            camino_mundo=camino_mundo,
        )

        errores_ruta.append(
            float(
                error_ruta
            )
        )

    tiempos_estados = (
        np.arange(
            len(
                registro[
                    "estados"
                ]
            ),
            dtype=float,
        )
        * dt
    )

    # ------------------------------------------------------
    # 12. Obtener solamente los cambios de submeta
    # ------------------------------------------------------

    submetas_distintas = []

    for submeta in submetas_usadas:

        if len(
            submetas_distintas
        ) == 0:

            submetas_distintas.append(
                tuple(
                    submeta
                )
            )

        else:

            submeta_anterior = np.asarray(
                submetas_distintas[
                    -1
                ],
                dtype=float,
            )

            submeta_actual = np.asarray(
                submeta,
                dtype=float,
            )

            if not np.allclose(
                submeta_actual,
                submeta_anterior,
                rtol=0.0,
                atol=1e-9,
            ):

                submetas_distintas.append(
                    tuple(
                        submeta
                    )
                )

    # Agregar la submeta final, cuando cambió después del
    # último movimiento.

    submeta_final = tuple(
        entorno[
            "submeta"
        ]
    )

    if len(
        submetas_distintas
    ) == 0:

        submetas_distintas.append(
            submeta_final
        )

    elif not np.allclose(
        np.asarray(
            submetas_distintas[
                -1
            ],
            dtype=float,
        ),

        np.asarray(
            submeta_final,
            dtype=float,
        ),

        rtol=0.0,
        atol=1e-9,
    ):

        submetas_distintas.append(
            submeta_final
        )

    # ------------------------------------------------------
    # 13. Construir trayectorias de obstáculos dinámicos
    # ------------------------------------------------------

    obstaculos_dinamicos_iniciales = entorno[
        "obstaculos_dinamicos_iniciales"
    ]

    numero_obstaculos_dinamicos = len(
        obstaculos_dinamicos_iniciales
    )

    trayectorias_obstaculos_dinamicos = []

    for indice_obstaculo in range(
        numero_obstaculos_dinamicos
    ):

        obstaculo_inicial = (
            obstaculos_dinamicos_iniciales[
                indice_obstaculo
            ]
        )

        trayectoria_obstaculo = [
            (
                float(
                    obstaculo_inicial[
                        "x"
                    ]
                ),

                float(
                    obstaculo_inicial[
                        "y"
                    ]
                ),
            )
        ]

        for obstaculos_paso in registro[
            "obstaculos_dinamicos"
        ]:

            if indice_obstaculo >= len(
                obstaculos_paso
            ):

                raise RuntimeError(
                    "El historial de obstáculos dinámicos "
                    "es inconsistente."
                )

            obstaculo = obstaculos_paso[
                indice_obstaculo
            ]

            trayectoria_obstaculo.append(
                (
                    float(
                        obstaculo[
                            "x"
                        ]
                    ),

                    float(
                        obstaculo[
                            "y"
                        ]
                    ),
                )
            )

        trayectorias_obstaculos_dinamicos.append(
            trayectoria_obstaculo
        )

    # ------------------------------------------------------
    # 14. Historial preparado para las gráficas
    # ------------------------------------------------------

    historial_visualizacion = {
        "tiempos_pasos": tiempos_pasos,

        "tiempos_estados": (
            tiempos_estados
        ),

        "estados_robot": [
            tuple(
                estado
            )
            for estado in registro[
                "estados"
            ]
        ],

        "acciones": acciones,

        "informaciones_accion": (
            informaciones_accion
        ),

        "recompensas": recompensas,

        "recompensas_acumuladas": (
            recompensas_acumuladas
        ),

        "componentes_recompensa": (
            componentes_recompensa
        ),

        "informaciones_paso": (
            informaciones_paso
        ),

        "submetas_usadas": (
            submetas_usadas
        ),

        "submetas_nuevas": (
            submetas_nuevas
        ),

        "submetas_distintas": (
            submetas_distintas
        ),

        "indices_progreso": (
            indices_progreso
        ),

        "distancias_meta": (
            distancias_meta
        ),

        "distancias_submeta": (
            distancias_submeta
        ),

        "errores_ruta": (
            errores_ruta
        ),

        "clearances_estaticos": (
            clearances_estaticos
        ),

        "clearances_dinamicos": (
            clearances_dinamicos
        ),

        "clearances_totales": (
            clearances_totales
        ),

        "velocidades_lineales": (
            velocidades_lineales
        ),

        "velocidades_angulares": (
            velocidades_angulares
        ),

        "trayectorias_obstaculos_dinamicos": (
            trayectorias_obstaculos_dinamicos
        ),
    }

    # ------------------------------------------------------
    # 15. Verificaciones finales
    # ------------------------------------------------------

    claves_por_paso = [
        "tiempos_pasos",
        "acciones",
        "informaciones_accion",
        "recompensas",
        "recompensas_acumuladas",
        "componentes_recompensa",
        "informaciones_paso",
        "submetas_usadas",
        "submetas_nuevas",
        "indices_progreso",
        "distancias_meta",
        "distancias_submeta",
        "clearances_estaticos",
        "clearances_dinamicos",
        "clearances_totales",
        "velocidades_lineales",
        "velocidades_angulares",
    ]

    historial_por_paso_consistente = all(
        len(
            historial_visualizacion[
                clave
            ]
        )
        == numero_pasos
        for clave in claves_por_paso
    )

    historial_estados_consistente = (
        len(
            historial_visualizacion[
                "estados_robot"
            ]
        )
        == numero_pasos + 1

        and len(
            historial_visualizacion[
                "tiempos_estados"
            ]
        )
        == numero_pasos + 1

        and len(
            historial_visualizacion[
                "errores_ruta"
            ]
        )
        == numero_pasos + 1
    )

    if not historial_por_paso_consistente:

        raise RuntimeError(
            "El historial temporal de la evaluación SAC "
            "es inconsistente."
        )

    if not historial_estados_consistente:

        raise RuntimeError(
            "El historial de estados de la evaluación SAC "
            "es inconsistente."
        )

    if not verificar_registro_clearances(
        registro
    ):

        raise RuntimeError(
            "El registro de clearances es inconsistente."
        )

    # ------------------------------------------------------
    # 16. Construir resultado completo
    # ------------------------------------------------------

    resultado_evaluacion = {
        "metodo": "sac",

        "modo": "determinista",

        "semilla": semilla,

        "entorno": entorno,

        "escenario": escenario,

        "camino_mundo": camino_mundo,

        "inicio": escenario[
            "inicio"
        ],

        "meta": entorno[
            "meta"
        ],

        "obstaculos_estaticos": entorno[
            "obstaculos_estaticos"
        ],

        "obstaculos_dinamicos_iniciales": (
            obstaculos_dinamicos_iniciales
        ),

        "registro": registro,

        "observacion_inicial": (
            observacion_inicial
        ),

        "observacion_final": {
            "parche": observacion[
                "parche"
            ].copy(),

            "escalares": observacion[
                "escalares"
            ].copy(),
        },

        "resultado": entorno[
            "resultado"
        ],

        "terminado": bool(
            terminado
        ),

        "truncado": bool(
            truncado
        ),

        "recompensa_acumulada": float(
            recompensa_acumulada
        ),

        "numero_pasos": numero_pasos,

        "historial_visualizacion": (
            historial_visualizacion
        ),

        "metricas": metricas,
    }

    return resultado_evaluacion
def graficar_episodio_sac_espacial(
    resultado_evaluacion,
    mostrar=True,
    ruta_guardado=None,
    dpi=DPI_GRAFICA_ESPACIAL_SAC,
    anotar_submetas=(
        ANOTAR_SUBMETAS_ESPACIALES_SAC
    ),
):

    # ------------------------------------------------------
    # 1. Validar el resultado de evaluación
    # ------------------------------------------------------

    if not isinstance(
        resultado_evaluacion,
        dict,
    ):

        raise TypeError(
            "El resultado de evaluación SAC debe ser "
            "un diccionario."
        )

    claves_necesarias = {
        "metodo",
        "modo",
        "semilla",
        "escenario",
        "camino_mundo",
        "inicio",
        "meta",
        "obstaculos_estaticos",
        "obstaculos_dinamicos_iniciales",
        "registro",
        "historial_visualizacion",
        "resultado",
        "metricas",
        "entorno",
    }

    if not claves_necesarias.issubset(
        resultado_evaluacion.keys()
    ):

        claves_faltantes = (
            claves_necesarias
            - set(
                resultado_evaluacion.keys()
            )
        )

        raise KeyError(
            "Faltan datos para la gráfica espacial SAC: "
            f"{claves_faltantes}."
        )

    if resultado_evaluacion[
        "metodo"
    ] != "sac":

        raise ValueError(
            "La gráfica espacial recibió un resultado "
            "que no pertenece a SAC."
        )

    if resultado_evaluacion[
        "modo"
    ] != "determinista":

        raise ValueError(
            "La gráfica espacial debe usar una evaluación "
            "determinista."
        )

    mostrar = bool(
        mostrar
    )

    anotar_submetas = bool(
        anotar_submetas
    )

    dpi = int(
        dpi
    )

    if dpi <= 0:

        raise ValueError(
            "El valor DPI debe ser positivo."
        )

    # ------------------------------------------------------
    # 2. Extraer los datos
    # ------------------------------------------------------

    semilla = int(
        resultado_evaluacion[
            "semilla"
        ]
    )

    resultado = resultado_evaluacion[
        "resultado"
    ]

    camino_mundo = resultado_evaluacion[
        "camino_mundo"
    ]

    inicio = resultado_evaluacion[
        "inicio"
    ]

    meta = resultado_evaluacion[
        "meta"
    ]

    obstaculos_estaticos = (
        resultado_evaluacion[
            "obstaculos_estaticos"
        ]
    )

    obstaculos_dinamicos_iniciales = (
        resultado_evaluacion[
            "obstaculos_dinamicos_iniciales"
        ]
    )

    entorno = resultado_evaluacion[
        "entorno"
    ]

    obstaculos_dinamicos_finales = entorno[
        "obstaculos_dinamicos"
    ]

    registro = resultado_evaluacion[
        "registro"
    ]

    historial = resultado_evaluacion[
        "historial_visualizacion"
    ]

    metricas = resultado_evaluacion[
        "metricas"
    ]

    estados_robot = registro[
        "estados"
    ]

    submetas_distintas = historial[
        "submetas_distintas"
    ]

    trayectorias_dinamicas = historial[
        "trayectorias_obstaculos_dinamicos"
    ]

    # ------------------------------------------------------
    # 3. Validar datos espaciales
    # ------------------------------------------------------

    if len(
        camino_mundo
    ) < 2:

        raise ValueError(
            "La ruta global A* debe contener al menos "
            "dos puntos."
        )

    if len(
        estados_robot
    ) < 2:

        raise ValueError(
            "La evaluación debe contener al menos dos "
            "estados del robot."
        )

    if len(
        inicio
    ) != 2:

        raise ValueError(
            "El punto inicial debe contener x e y."
        )

    if len(
        meta
    ) != 2:

        raise ValueError(
            "La meta debe contener x e y."
        )

    camino_numpy = np.asarray(
        camino_mundo,
        dtype=float,
    )

    estados_numpy = np.asarray(
        estados_robot,
        dtype=float,
    )

    if camino_numpy.ndim != 2:

        raise ValueError(
            "La ruta A* tiene una estructura incorrecta."
        )

    if camino_numpy.shape[
        1
    ] < 2:

        raise ValueError(
            "Cada punto de la ruta A* debe contener x e y."
        )

    if estados_numpy.ndim != 2:

        raise ValueError(
            "Los estados del robot tienen una estructura "
            "incorrecta."
        )

    if estados_numpy.shape[
        1
    ] < 3:

        raise ValueError(
            "Cada estado del robot debe contener "
            "x, y y theta."
        )

    if not np.all(
        np.isfinite(
            camino_numpy
        )
    ):

        raise ValueError(
            "La ruta A* contiene NaN o infinitos."
        )

    if not np.all(
        np.isfinite(
            estados_numpy
        )
    ):

        raise ValueError(
            "La trayectoria SAC contiene NaN o infinitos."
        )

    if len(
        trayectorias_dinamicas
    ) != len(
        obstaculos_dinamicos_iniciales
    ):

        raise ValueError(
            "La cantidad de trayectorias dinámicas no "
            "coincide con la cantidad de obstáculos."
        )

    # ------------------------------------------------------
    # 4. Crear la figura
    # ------------------------------------------------------

    figura, ax = plt.subplots(
        figsize=(
            11,
            9,
        )
    )

    # ------------------------------------------------------
    # 5. Dibujar obstáculos estáticos
    # ------------------------------------------------------

    dibujar_obstaculos(
        ax=ax,

        obstaculos=obstaculos_estaticos,
    )

    # ------------------------------------------------------
    # 6. Dibujar la ruta global A*
    # ------------------------------------------------------

    dibujar_camino_astar(
        ax=ax,

        camino_mundo=camino_mundo,
    )

    # ------------------------------------------------------
    # 7. Dibujar trayectoria real del robot
    # ------------------------------------------------------

    dibujar_trayectoria_estados(
        ax=ax,

        estados=estados_robot,

        etiqueta="Trayectoria SAC determinista",

        estilo="-",
    )

    # ------------------------------------------------------
    # 8. Dibujar las submetas locales distintas
    # ------------------------------------------------------

    if len(
        submetas_distintas
    ) > 0:

        submetas_numpy = np.asarray(
            submetas_distintas,
            dtype=float,
        )

        if (
            submetas_numpy.ndim != 2
            or submetas_numpy.shape[
                1
            ] != 2
        ):

            raise ValueError(
                "Las submetas locales tienen una "
                "estructura incorrecta."
            )

        if not np.all(
            np.isfinite(
                submetas_numpy
            )
        ):

            raise ValueError(
                "Las submetas contienen NaN o infinitos."
            )

        ax.scatter(
            submetas_numpy[
                :,
                0
            ],

            submetas_numpy[
                :,
                1
            ],

            marker="D",

            s=65,

            edgecolor="black",

            linewidth=0.8,

            label="Submetas locales",

            zorder=11,
        )

        # Unir las submetas en el orden en que fueron usadas.

        if len(
            submetas_distintas
        ) > 1:

            ax.plot(
                submetas_numpy[
                    :,
                    0
                ],

                submetas_numpy[
                    :,
                    1
                ],

                linestyle=":",

                linewidth=1.2,

                alpha=0.65,

                label="Secuencia de submetas",

                zorder=7,
            )

        if anotar_submetas:

            for indice, submeta in enumerate(
                submetas_distintas
            ):

                ax.annotate(
                    f"S{indice + 1}",

                    xy=(
                        submeta[
                            0
                        ],
                        submeta[
                            1
                        ],
                    ),

                    xytext=(
                        5,
                        5,
                    ),

                    textcoords="offset points",

                    fontsize=8,

                    zorder=12,
                )

    # ------------------------------------------------------
    # 9. Dibujar trayectorias dinámicas
    # ------------------------------------------------------

    dibujar_trayectorias_dinamicas(
        ax=ax,

        trayectorias=trayectorias_dinamicas,
    )

    # ------------------------------------------------------
    # 10. Posiciones iniciales de obstáculos dinámicos
    # ------------------------------------------------------

    if len(
        obstaculos_dinamicos_iniciales
    ) > 0:

        dibujar_obstaculos_dinamicos(
            ax=ax,

            obstaculos_dinamicos=(
                obstaculos_dinamicos_iniciales
            ),

            etiqueta=(
                "Obstáculos dinámicos al inicio"
            ),

            relleno=True,

            transparencia=0.30,
        )

    # ------------------------------------------------------
    # 11. Posiciones finales de obstáculos dinámicos
    # ------------------------------------------------------

    if len(
        obstaculos_dinamicos_finales
    ) > 0:

        dibujar_obstaculos_dinamicos(
            ax=ax,

            obstaculos_dinamicos=(
                obstaculos_dinamicos_finales
            ),

            etiqueta=(
                "Obstáculos dinámicos al final"
            ),

            relleno=False,

            transparencia=0.90,
        )

    # ------------------------------------------------------
    # 12. Dibujar inicio y meta
    # ------------------------------------------------------

    dibujar_inicio(
        ax=ax,

        inicio=inicio,
    )

    dibujar_meta(
        ax=ax,

        meta=meta,
    )

    # ------------------------------------------------------
    # 13. Dibujar posición final del robot
    # ------------------------------------------------------

    estado_final = estados_robot[
        -1
    ]

    x_final = float(
        estado_final[
            0
        ]
    )

    y_final = float(
        estado_final[
            1
        ]
    )

    theta_final = float(
        estado_final[
            2
        ]
    )

    circulo_robot_final = Circle(
        (
            x_final,
            y_final,
        ),

        radius=RADIO_ROBOT,

        facecolor="none",

        edgecolor="black",

        linewidth=2.0,

        label="Robot al final",

        zorder=13,
    )

    ax.add_patch(
        circulo_robot_final
    )

    dibujar_orientacion_robot(
        ax=ax,

        estado=estado_final,

        longitud=0.65,
    )

    # ------------------------------------------------------
    # 14. Marcar el resultado terminal
    # ------------------------------------------------------

    es_colision = resultado in {
        "colision_estatica",
        "colision_dinamica",
    }

    if es_colision:

        ax.scatter(
            x_final,
            y_final,

            marker="X",

            s=180,

            edgecolor="black",

            linewidth=1.2,

            label="Punto de colisión",

            zorder=15,
        )

    elif resultado == "meta":

        ax.scatter(
            x_final,
            y_final,

            marker="o",

            s=90,

            facecolor="none",

            edgecolor="black",

            linewidth=2.0,

            label="Llegada a la meta",

            zorder=15,
        )

    elif resultado == "fuera_mapa":

        ax.scatter(
            x_final,
            y_final,

            marker="X",

            s=180,

            edgecolor="black",

            linewidth=1.2,

            label="Salida del mapa",

            zorder=15,
        )

    else:

        ax.scatter(
            x_final,
            y_final,

            marker="s",

            s=80,

            edgecolor="black",

            linewidth=1.2,

            label="Posición final",

            zorder=15,
        )

    # ------------------------------------------------------
    # 15. Configurar el mapa
    # ------------------------------------------------------

    configurar_mapa(
        ax
    )

    ax.set_title(
        "Evaluación espacial A* + SAC\n"
        f"Semilla: {semilla} | "
        f"Resultado: {resultado} | "
        f"Pasos: {registro['pasos_ejecutados']}"
    )

    # ------------------------------------------------------
    # 16. Agregar resumen numérico
    # ------------------------------------------------------

    texto_metricas = (
        f"Longitud: "
        f"{metricas['longitud_recorrida']:.2f} m\n"
        f"Distancia final: "
        f"{metricas['distancia_final_meta']:.2f} m\n"
        f"Error medio A*: "
        f"{metricas['error_medio_ruta']:.2f} m\n"
        f"Clearance mínimo: "
        f"{metricas['clearance_total_minimo']:.2f} m"
    )

    ax.text(
        0.02,
        0.98,

        texto_metricas,

        transform=ax.transAxes,

        horizontalalignment="left",

        verticalalignment="top",

        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.80,
        },

        zorder=20,
    )

    # ------------------------------------------------------
    # 17. Eliminar etiquetas repetidas de la leyenda
    # ------------------------------------------------------

    manejadores, etiquetas = (
        ax.get_legend_handles_labels()
    )

    manejadores_unicos = []

    etiquetas_unicas = []

    etiquetas_encontradas = set()

    for manejador, etiqueta in zip(
        manejadores,
        etiquetas,
    ):

        if (
            etiqueta
            and etiqueta
            not in etiquetas_encontradas
        ):

            manejadores_unicos.append(
                manejador
            )

            etiquetas_unicas.append(
                etiqueta
            )

            etiquetas_encontradas.add(
                etiqueta
            )

    ax.legend(
        manejadores_unicos,
        etiquetas_unicas,

        loc="upper right",

        fontsize=8,
    )

    figura.tight_layout()

    # ------------------------------------------------------
    # 18. Guardar opcionalmente
    # ------------------------------------------------------

    ruta_guardada = None

    if ruta_guardado is not None:

        ruta = Path(
            ruta_guardado
        )

        ruta.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figura.savefig(
            ruta,

            dpi=dpi,

            bbox_inches="tight",
        )

        ruta_guardada = str(
            ruta
        )

    # ------------------------------------------------------
    # 19. Mostrar opcionalmente
    # ------------------------------------------------------

    if mostrar:

        plt.show()

    # ------------------------------------------------------
    # 20. Regresar la figura y sus datos
    # ------------------------------------------------------

    return {
        "figura": figura,

        "eje": ax,

        "ruta_guardada": ruta_guardada,

        "resultado": resultado,

        "numero_estados": len(
            estados_robot
        ),

        "numero_submetas_distintas": len(
            submetas_distintas
        ),

        "numero_trayectorias_dinamicas": len(
            trayectorias_dinamicas
        ),

        "estado_final": tuple(
            estado_final
        ),

        "es_colision": es_colision,
    }
def obtener_serie_temporal_sac(
    historial,
    clave,
    longitud_esperada,
    permitir_infinito=False,
):

    # ------------------------------------------------------
    # 1. Verificar el historial
    # ------------------------------------------------------

    if not isinstance(
        historial,
        dict,
    ):

        raise TypeError(
            "El historial temporal SAC debe ser un "
            "diccionario."
        )

    if clave not in historial:

        raise KeyError(
            f"El historial temporal no contiene "
            f"la clave '{clave}'."
        )

    longitud_esperada = int(
        longitud_esperada
    )

    if longitud_esperada < 0:

        raise ValueError(
            "La longitud esperada no puede ser negativa."
        )

    # ------------------------------------------------------
    # 2. Convertir a arreglo NumPy
    # ------------------------------------------------------

    serie = np.asarray(
        historial[
            clave
        ],
        dtype=float,
    ).reshape(
        -1
    )

    # ------------------------------------------------------
    # 3. Verificar longitud
    # ------------------------------------------------------

    if serie.size != longitud_esperada:

        raise ValueError(
            f"La serie '{clave}' contiene "
            f"{serie.size} valores; se esperaban "
            f"{longitud_esperada}."
        )

    # ------------------------------------------------------
    # 4. Verificar NaN
    # ------------------------------------------------------

    if np.any(
        np.isnan(
            serie
        )
    ):

        raise ValueError(
            f"La serie '{clave}' contiene NaN."
        )

    # ------------------------------------------------------
    # 5. Verificar infinitos
    # ------------------------------------------------------

    if (
        not permitir_infinito
        and np.any(
            np.isinf(
                serie
            )
        )
    ):

        raise ValueError(
            f"La serie '{clave}' contiene valores "
            "infinitos."
        )

    return serie
def guardar_figuras_temporales_sac(
    figuras,
    directorio_salida,
    dpi=DPI_GRAFICAS_TEMPORALES_SAC,
):

    # ------------------------------------------------------
    # 1. Validar entradas
    # ------------------------------------------------------

    if not isinstance(
        figuras,
        dict,
    ):

        raise TypeError(
            "Las figuras temporales deben estar dentro "
            "de un diccionario."
        )

    if len(
        figuras
    ) == 0:

        raise ValueError(
            "No existen figuras temporales para guardar."
        )

    dpi = int(
        dpi
    )

    if dpi <= 0:

        raise ValueError(
            "La resolución DPI debe ser positiva."
        )

    directorio = Path(
        directorio_salida
    )

    directorio.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas_guardadas = {}

    # ------------------------------------------------------
    # 2. Guardar cada figura
    # ------------------------------------------------------

    for nombre, figura in figuras.items():

        if not isinstance(
            figura,
            matplotlib.figure.Figure,
        ):

            raise TypeError(
                f"El elemento '{nombre}' no es una "
                "figura válida de Matplotlib."
            )

        ruta = directorio / (
            f"{nombre}.png"
        )

        figura.savefig(
            ruta,
            dpi=dpi,
            bbox_inches="tight",
        )

        rutas_guardadas[
            nombre
        ] = str(
            ruta
        )

    return rutas_guardadas
def graficar_episodio_sac_temporal(
    resultado_evaluacion,
    mostrar=True,
    directorio_salida=None,
    dpi=DPI_GRAFICAS_TEMPORALES_SAC,
):

    # ------------------------------------------------------
    # 1. Verificar el resultado de evaluación
    # ------------------------------------------------------

    if not isinstance(
        resultado_evaluacion,
        dict,
    ):

        raise TypeError(
            "El resultado de evaluación SAC debe ser "
            "un diccionario."
        )

    claves_necesarias = {
        "metodo",
        "modo",
        "semilla",
        "resultado",
        "numero_pasos",
        "historial_visualizacion",
        "metricas",
    }

    if not claves_necesarias.issubset(
        resultado_evaluacion.keys()
    ):

        claves_faltantes = (
            claves_necesarias
            - set(
                resultado_evaluacion.keys()
            )
        )

        raise KeyError(
            "Faltan datos para las gráficas temporales: "
            f"{claves_faltantes}."
        )

    if resultado_evaluacion[
        "metodo"
    ] != "sac":

        raise ValueError(
            "El resultado recibido no corresponde a SAC."
        )

    if resultado_evaluacion[
        "modo"
    ] != "determinista":

        raise ValueError(
            "Las gráficas temporales deben construirse "
            "con una evaluación determinista."
        )

    mostrar = bool(
        mostrar
    )

    dpi = int(
        dpi
    )

    if dpi <= 0:

        raise ValueError(
            "El valor DPI debe ser positivo."
        )

    # ------------------------------------------------------
    # 2. Extraer datos generales
    # ------------------------------------------------------

    historial = resultado_evaluacion[
        "historial_visualizacion"
    ]

    metricas = resultado_evaluacion[
        "metricas"
    ]

    numero_pasos = int(
        resultado_evaluacion[
            "numero_pasos"
        ]
    )

    semilla = int(
        resultado_evaluacion[
            "semilla"
        ]
    )

    resultado = resultado_evaluacion[
        "resultado"
    ]

    if numero_pasos <= 0:

        raise ValueError(
            "La evaluación debe contener al menos "
            "un paso."
        )

    # ------------------------------------------------------
    # 3. Obtener tiempos
    # ------------------------------------------------------

    tiempos_pasos = obtener_serie_temporal_sac(
        historial=historial,

        clave="tiempos_pasos",

        longitud_esperada=numero_pasos,
    )

    tiempos_estados = obtener_serie_temporal_sac(
        historial=historial,

        clave="tiempos_estados",

        longitud_esperada=(
            numero_pasos + 1
        ),
    )

    if numero_pasos > 1:

        if not np.all(
            np.diff(
                tiempos_pasos
            )
            > 0.0
        ):

            raise ValueError(
                "Los tiempos de los pasos deben crecer "
                "estrictamente."
            )

    if numero_pasos > 0:

        if not np.all(
            np.diff(
                tiempos_estados
            )
            > 0.0
        ):

            raise ValueError(
                "Los tiempos de los estados deben crecer "
                "estrictamente."
            )

    # ------------------------------------------------------
    # 4. Obtener series por paso
    # ------------------------------------------------------

    distancias_meta = obtener_serie_temporal_sac(
        historial=historial,

        clave="distancias_meta",

        longitud_esperada=numero_pasos,
    )

    distancias_submeta = obtener_serie_temporal_sac(
        historial=historial,

        clave="distancias_submeta",

        longitud_esperada=numero_pasos,
    )

    indices_progreso = obtener_serie_temporal_sac(
        historial=historial,

        clave="indices_progreso",

        longitud_esperada=numero_pasos,
    )

    clearances_estaticos = obtener_serie_temporal_sac(
        historial=historial,

        clave="clearances_estaticos",

        longitud_esperada=numero_pasos,

        permitir_infinito=True,
    )

    clearances_dinamicos = obtener_serie_temporal_sac(
        historial=historial,

        clave="clearances_dinamicos",

        longitud_esperada=numero_pasos,

        permitir_infinito=True,
    )

    clearances_totales = obtener_serie_temporal_sac(
        historial=historial,

        clave="clearances_totales",

        longitud_esperada=numero_pasos,

        permitir_infinito=True,
    )

    velocidades_lineales = obtener_serie_temporal_sac(
        historial=historial,

        clave="velocidades_lineales",

        longitud_esperada=numero_pasos,
    )

    velocidades_angulares = obtener_serie_temporal_sac(
        historial=historial,

        clave="velocidades_angulares",

        longitud_esperada=numero_pasos,
    )

    recompensas = obtener_serie_temporal_sac(
        historial=historial,

        clave="recompensas",

        longitud_esperada=numero_pasos,
    )

    recompensas_acumuladas = (
        obtener_serie_temporal_sac(
            historial=historial,

            clave="recompensas_acumuladas",

            longitud_esperada=numero_pasos,
        )
    )

    # ------------------------------------------------------
    # 5. Obtener serie por estado
    # ------------------------------------------------------

    errores_ruta = obtener_serie_temporal_sac(
        historial=historial,

        clave="errores_ruta",

        longitud_esperada=(
            numero_pasos + 1
        ),
    )

    # ------------------------------------------------------
    # 6. Validaciones físicas
    # ------------------------------------------------------

    if np.any(
        distancias_meta < 0.0
    ):

        raise ValueError(
            "La distancia a la meta no puede ser negativa."
        )

    if np.any(
        distancias_submeta < 0.0
    ):

        raise ValueError(
            "La distancia a la submeta no puede ser "
            "negativa."
        )

    if np.any(
        errores_ruta < 0.0
    ):

        raise ValueError(
            "El error respecto de A* no puede ser negativo."
        )

    tolerancia = 1e-6

    if np.any(
        velocidades_lineales
        < -tolerancia
    ):

        raise ValueError(
            "La velocidad lineal contiene valores "
            "negativos."
        )

    if np.any(
        velocidades_lineales
        > VELOCIDAD_MAXIMA
        + tolerancia
    ):

        raise ValueError(
            "La velocidad lineal supera su límite."
        )

    if np.any(
        velocidades_angulares
        < -VELOCIDAD_ANGULAR_MAXIMA
        - tolerancia
    ):

        raise ValueError(
            "La velocidad angular es menor que su límite."
        )

    if np.any(
        velocidades_angulares
        > VELOCIDAD_ANGULAR_MAXIMA
        + tolerancia
    ):

        raise ValueError(
            "La velocidad angular supera su límite."
        )

    # ------------------------------------------------------
    # 7. Preparar clearances para Matplotlib
    # ------------------------------------------------------
    #
    # Un clearance infinito significa que no existe un
    # obstáculo limitante. Para evitar que la escala vertical
    # se vuelva infinita, esos valores se representan como
    # espacios vacíos en la curva.

    clearances_estaticos_grafica = (
        clearances_estaticos.copy()
    )

    clearances_dinamicos_grafica = (
        clearances_dinamicos.copy()
    )

    clearances_totales_grafica = (
        clearances_totales.copy()
    )

    clearances_estaticos_grafica[
        np.isinf(
            clearances_estaticos_grafica
        )
    ] = np.nan

    clearances_dinamicos_grafica[
        np.isinf(
            clearances_dinamicos_grafica
        )
    ] = np.nan

    clearances_totales_grafica[
        np.isinf(
            clearances_totales_grafica
        )
    ] = np.nan

    figuras = {}

    titulo_sufijo = (
        f"Semilla {semilla} | "
        f"Resultado: {resultado}"
    )

    # ======================================================
    # 8. DISTANCIA A META Y SUBMETA
    # ======================================================

    figura_distancias, ax_distancias = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_distancias.plot(
        tiempos_pasos,
        distancias_meta,
        linewidth=2.0,
        label="Distancia a la meta",
    )

    ax_distancias.plot(
        tiempos_pasos,
        distancias_submeta,
        linewidth=2.0,
        label="Distancia a la submeta",
    )

    ax_distancias.axhline(
        y=DISTANCIA_META,
        linestyle="--",
        linewidth=1.5,
        label="Tolerancia de llegada",
    )

    ax_distancias.set_xlabel(
        "Tiempo [s]"
    )

    ax_distancias.set_ylabel(
        "Distancia [m]"
    )

    ax_distancias.set_title(
        "Distancia a los objetivos de navegación\n"
        + titulo_sufijo
    )

    ax_distancias.grid(
        True,
        alpha=0.3,
    )

    ax_distancias.legend()

    figura_distancias.tight_layout()

    figuras[
        "01_distancias_sac"
    ] = figura_distancias

    # ======================================================
    # 9. ÍNDICE DE PROGRESO SOBRE A*
    # ======================================================

    figura_progreso, ax_progreso = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_progreso.step(
        tiempos_pasos,
        indices_progreso,
        where="post",
        linewidth=2.0,
        label="Índice de progreso A*",
    )

    ax_progreso.set_xlabel(
        "Tiempo [s]"
    )

    ax_progreso.set_ylabel(
        "Índice de la ruta global"
    )

    ax_progreso.set_title(
        "Progreso sobre la ruta A*\n"
        + titulo_sufijo
    )

    ax_progreso.grid(
        True,
        alpha=0.3,
    )

    ax_progreso.legend()

    figura_progreso.tight_layout()

    figuras[
        "02_progreso_astar_sac"
    ] = figura_progreso

    # ======================================================
    # 10. ERROR RESPECTO DE A*
    # ======================================================

    figura_error, ax_error = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_error.plot(
        tiempos_estados,
        errores_ruta,
        linewidth=2.0,
        label="Error respecto de A*",
    )

    ax_error.axhline(
        y=metricas[
            "error_medio_ruta"
        ],
        linestyle="--",
        linewidth=1.5,
        label="Error medio",
    )

    ax_error.set_xlabel(
        "Tiempo [s]"
    )

    ax_error.set_ylabel(
        "Error lateral [m]"
    )

    ax_error.set_title(
        "Separación respecto de la ruta global A*\n"
        + titulo_sufijo
    )

    ax_error.grid(
        True,
        alpha=0.3,
    )

    ax_error.legend()

    figura_error.tight_layout()

    figuras[
        "03_error_ruta_sac"
    ] = figura_error

    # ======================================================
    # 11. CLEARANCES
    # ======================================================

    figura_clearances, ax_clearances = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_clearances.plot(
        tiempos_pasos,
        clearances_estaticos_grafica,
        linewidth=1.8,
        label="Clearance estático",
    )

    ax_clearances.plot(
        tiempos_pasos,
        clearances_dinamicos_grafica,
        linewidth=1.8,
        label="Clearance dinámico",
    )

    ax_clearances.plot(
        tiempos_pasos,
        clearances_totales_grafica,
        linewidth=2.2,
        label="Clearance total",
    )

    ax_clearances.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.5,
        label="Límite de colisión",
    )

    ax_clearances.set_xlabel(
        "Tiempo [s]"
    )

    ax_clearances.set_ylabel(
        "Clearance [m]"
    )

    ax_clearances.set_title(
        "Seguridad durante la navegación SAC\n"
        + titulo_sufijo
    )

    ax_clearances.grid(
        True,
        alpha=0.3,
    )

    ax_clearances.legend()

    figura_clearances.tight_layout()

    figuras[
        "04_clearances_sac"
    ] = figura_clearances

    # ======================================================
    # 12. VELOCIDAD LINEAL
    # ======================================================

    figura_velocidad_lineal, ax_v = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_v.plot(
        tiempos_pasos,
        velocidades_lineales,
        linewidth=2.0,
        label="Velocidad lineal",
    )

    ax_v.axhline(
        y=VELOCIDAD_MAXIMA,
        linestyle="--",
        linewidth=1.5,
        label="Límite máximo",
    )

    ax_v.axhline(
        y=0.0,
        linestyle=":",
        linewidth=1.2,
        label="Robot detenido",
    )

    ax_v.set_xlabel(
        "Tiempo [s]"
    )

    ax_v.set_ylabel(
        "Velocidad lineal [m/s]"
    )

    ax_v.set_title(
        "Control lineal generado por SAC\n"
        + titulo_sufijo
    )

    ax_v.set_ylim(
        min(
            -0.05,
            float(
                np.min(
                    velocidades_lineales
                )
            )
            - 0.05,
        ),

        max(
            VELOCIDAD_MAXIMA
            + 0.10,

            float(
                np.max(
                    velocidades_lineales
                )
            )
            + 0.10,
        ),
    )

    ax_v.grid(
        True,
        alpha=0.3,
    )

    ax_v.legend()

    figura_velocidad_lineal.tight_layout()

    figuras[
        "05_velocidad_lineal_sac"
    ] = figura_velocidad_lineal

    # ======================================================
    # 13. VELOCIDAD ANGULAR
    # ======================================================

    figura_velocidad_angular, ax_omega = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_omega.plot(
        tiempos_pasos,
        velocidades_angulares,
        linewidth=2.0,
        label="Velocidad angular",
    )

    ax_omega.axhline(
        y=VELOCIDAD_ANGULAR_MAXIMA,
        linestyle="--",
        linewidth=1.5,
        label="Límite superior",
    )

    ax_omega.axhline(
        y=-VELOCIDAD_ANGULAR_MAXIMA,
        linestyle="--",
        linewidth=1.5,
        label="Límite inferior",
    )

    ax_omega.axhline(
        y=0.0,
        linestyle=":",
        linewidth=1.2,
        label="Sin giro",
    )

    ax_omega.set_xlabel(
        "Tiempo [s]"
    )

    ax_omega.set_ylabel(
        "Velocidad angular [rad/s]"
    )

    ax_omega.set_title(
        "Control angular generado por SAC\n"
        + titulo_sufijo
    )

    ax_omega.set_ylim(
        -VELOCIDAD_ANGULAR_MAXIMA
        - 0.15,

        VELOCIDAD_ANGULAR_MAXIMA
        + 0.15,
    )

    ax_omega.grid(
        True,
        alpha=0.3,
    )

    ax_omega.legend()

    figura_velocidad_angular.tight_layout()

    figuras[
        "06_velocidad_angular_sac"
    ] = figura_velocidad_angular

    # ======================================================
    # 14. RECOMPENSA POR PASO
    # ======================================================

    figura_recompensa, ax_recompensa = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax_recompensa.plot(
        tiempos_pasos,
        recompensas,
        linewidth=1.8,
        label="Recompensa por paso",
    )

    ax_recompensa.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.2,
        label="Recompensa neutra",
    )

    ax_recompensa.set_xlabel(
        "Tiempo [s]"
    )

    ax_recompensa.set_ylabel(
        "Recompensa"
    )

    ax_recompensa.set_title(
        "Recompensa instantánea SAC\n"
        + titulo_sufijo
    )

    ax_recompensa.grid(
        True,
        alpha=0.3,
    )

    ax_recompensa.legend()

    figura_recompensa.tight_layout()

    figuras[
        "07_recompensa_paso_sac"
    ] = figura_recompensa

    # ======================================================
    # 15. RECOMPENSA ACUMULADA
    # ======================================================

    figura_recompensa_acumulada, ax_acumulada = (
        plt.subplots(
            figsize=(
                10,
                5,
            )
        )
    )

    ax_acumulada.plot(
        tiempos_pasos,
        recompensas_acumuladas,
        linewidth=2.2,
        label="Recompensa acumulada",
    )

    ax_acumulada.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.2,
        label="Referencia cero",
    )

    ax_acumulada.set_xlabel(
        "Tiempo [s]"
    )

    ax_acumulada.set_ylabel(
        "Recompensa acumulada"
    )

    ax_acumulada.set_title(
        "Retorno acumulado del episodio SAC\n"
        + titulo_sufijo
    )

    ax_acumulada.grid(
        True,
        alpha=0.3,
    )

    ax_acumulada.legend()

    figura_recompensa_acumulada.tight_layout()

    figuras[
        "08_recompensa_acumulada_sac"
    ] = figura_recompensa_acumulada

    # ------------------------------------------------------
    # 16. Guardar opcionalmente
    # ------------------------------------------------------

    rutas_guardadas = {}

    if directorio_salida is not None:

        rutas_guardadas = (
            guardar_figuras_temporales_sac(
                figuras=figuras,

                directorio_salida=(
                    directorio_salida
                ),

                dpi=dpi,
            )
        )

    # ------------------------------------------------------
    # 17. Mostrar opcionalmente
    # ------------------------------------------------------

    if mostrar:

        plt.show()

    # ------------------------------------------------------
    # 18. Regresar resultados
    # ------------------------------------------------------

    return {
        "figuras": figuras,

        "rutas_guardadas": rutas_guardadas,

        "numero_figuras": len(
            figuras
        ),

        "numero_pasos": numero_pasos,

        "resultado": resultado,

        "clearances_con_infinito": bool(
            np.any(
                np.isinf(
                    clearances_estaticos
                )
            )
            or np.any(
                np.isinf(
                    clearances_dinamicos
                )
            )
            or np.any(
                np.isinf(
                    clearances_totales
                )
            )
        ),
    }
# ==========================================================
# GUARDADO DEL CHECKPOINT FINAL SAC
# ==========================================================
def guardar_checkpoint_final_sac(
    ruta_checkpoint,
    resultado_entrenamiento,
    optimizador_actor,
    optimizador_q1,
    optimizador_q2,
    optimizador_alpha,
):

    # ------------------------------------------------------
    # 1. Validar el resultado del entrenamiento
    # ------------------------------------------------------

    if not isinstance(
        resultado_entrenamiento,
        dict,
    ):

        raise TypeError(
            "El resultado del entrenamiento debe ser "
            "un diccionario."
        )

    claves_necesarias = {
        "actor",
        "critico_q1",
        "critico_q2",
        "critico_q1_objetivo",
        "critico_q2_objetivo",
        "temperatura",
        "buffer",
        "historial_global",
        "resumen_global",
    }

    if not claves_necesarias.issubset(
        resultado_entrenamiento.keys()
    ):

        claves_faltantes = (
            claves_necesarias
            - set(
                resultado_entrenamiento.keys()
            )
        )

        raise KeyError(
            "Faltan elementos para guardar el checkpoint: "
            f"{claves_faltantes}."
        )

    # ------------------------------------------------------
    # 2. Obtener los componentes
    # ------------------------------------------------------

    actor = resultado_entrenamiento[
        "actor"
    ]

    critico_q1 = resultado_entrenamiento[
        "critico_q1"
    ]

    critico_q2 = resultado_entrenamiento[
        "critico_q2"
    ]

    critico_q1_objetivo = (
        resultado_entrenamiento[
            "critico_q1_objetivo"
        ]
    )

    critico_q2_objetivo = (
        resultado_entrenamiento[
            "critico_q2_objetivo"
        ]
    )

    temperatura = resultado_entrenamiento[
        "temperatura"
    ]

    buffer = resultado_entrenamiento[
        "buffer"
    ]

    historial_global = resultado_entrenamiento[
        "historial_global"
    ]

    resumen_global = resultado_entrenamiento[
        "resumen_global"
    ]

    # ------------------------------------------------------
    # 3. Validar módulos y optimizadores
    # ------------------------------------------------------

    modelos = [
        actor,
        critico_q1,
        critico_q2,
        critico_q1_objetivo,
        critico_q2_objetivo,
        temperatura,
    ]

    if not all(
        isinstance(
            modelo,
            nn.Module,
        )
        for modelo in modelos
    ):

        raise TypeError(
            "Todos los modelos del checkpoint deben ser "
            "módulos de PyTorch."
        )

    optimizadores = [
        optimizador_actor,
        optimizador_q1,
        optimizador_q2,
        optimizador_alpha,
    ]

    if not all(
        isinstance(
            optimizador,
            torch.optim.Optimizer,
        )
        for optimizador in optimizadores
    ):

        raise TypeError(
            "Todos los optimizadores del checkpoint deben "
            "ser válidos."
        )

    # ------------------------------------------------------
    # 4. Validar el replay buffer
    # ------------------------------------------------------

    claves_buffer = {
        "parches",
        "escalares",
        "acciones",
        "recompensas",
        "siguientes_parches",
        "siguientes_escalares",
        "terminados",
        "capacidad",
        "tamano",
        "indice_escritura",
        "total_transiciones_agregadas",
        "generador",
        "semilla",
    }

    if not claves_buffer.issubset(
        buffer.keys()
    ):

        claves_faltantes = (
            claves_buffer
            - set(
                buffer.keys()
            )
        )

        raise KeyError(
            "Faltan elementos del replay buffer: "
            f"{claves_faltantes}."
        )

    capacidad = int(
        buffer[
            "capacidad"
        ]
    )

    tamano = int(
        buffer[
            "tamano"
        ]
    )

    if tamano < capacidad:

        filas_guardadas = tamano

    else:

        filas_guardadas = capacidad

    # ------------------------------------------------------
    # 5. Guardar solamente las filas utilizadas
    # ------------------------------------------------------

    estado_buffer = {
        "parches": buffer[
            "parches"
        ][
            :filas_guardadas
        ].copy(),

        "escalares": buffer[
            "escalares"
        ][
            :filas_guardadas
        ].copy(),

        "acciones": buffer[
            "acciones"
        ][
            :filas_guardadas
        ].copy(),

        "recompensas": buffer[
            "recompensas"
        ][
            :filas_guardadas
        ].copy(),

        "siguientes_parches": buffer[
            "siguientes_parches"
        ][
            :filas_guardadas
        ].copy(),

        "siguientes_escalares": buffer[
            "siguientes_escalares"
        ][
            :filas_guardadas
        ].copy(),

        "terminados": buffer[
            "terminados"
        ][
            :filas_guardadas
        ].copy(),

        "capacidad": capacidad,

        "tamano": tamano,

        "indice_escritura": int(
            buffer[
                "indice_escritura"
            ]
        ),

        "total_transiciones_agregadas": int(
            buffer[
                "total_transiciones_agregadas"
            ]
        ),

        "filas_guardadas": filas_guardadas,

        "semilla": int(
            buffer[
                "semilla"
            ]
        ),

        "estado_generador": (
            buffer[
                "generador"
            ].bit_generator.state
        ),
    }

    # ------------------------------------------------------
    # 6. Construir el checkpoint
    # ------------------------------------------------------

    dispositivo_actor = next(
        actor.parameters()
    ).device

    checkpoint = {
        "version_checkpoint": 1,

        "tipo": "sac_entrenamiento_completo",

        "episodios_entrenados": int(
            resumen_global[
                "numero_episodios"
            ]
        ),

        "semilla_base": int(
            resumen_global[
                "semilla_base"
            ]
        ),

        "dispositivo_original": str(
            dispositivo_actor
        ),

        "actor_state_dict": (
            actor.state_dict()
        ),

        "critico_q1_state_dict": (
            critico_q1.state_dict()
        ),

        "critico_q2_state_dict": (
            critico_q2.state_dict()
        ),

        "critico_q1_objetivo_state_dict": (
            critico_q1_objetivo.state_dict()
        ),

        "critico_q2_objetivo_state_dict": (
            critico_q2_objetivo.state_dict()
        ),

        "temperatura_state_dict": (
            temperatura.state_dict()
        ),

        "optimizador_actor_state_dict": (
            optimizador_actor.state_dict()
        ),

        "optimizador_q1_state_dict": (
            optimizador_q1.state_dict()
        ),

        "optimizador_q2_state_dict": (
            optimizador_q2.state_dict()
        ),

        "optimizador_alpha_state_dict": (
            optimizador_alpha.state_dict()
        ),

        "buffer_repeticion": estado_buffer,

        "historial_global": historial_global,

        "resumen_global": resumen_global,

        "configuracion": {
            "numero_episodios": int(resumen_global["numero_episodios"]
            ),

            "pasos_maximos": (
                PASOS_MAXIMOS_SEGUIMIENTO
            ),

            "tamano_lote": (
                TAMANO_LOTE_SAC
            ),

            "capacidad_buffer": (
                CAPACIDAD_BUFFER_REPETICION_SAC
            ),

            "transiciones_aleatorias_iniciales": (
                TRANSICIONES_ALEATORIAS_INICIALES_SAC
            ),

            "actualizaciones_por_paso": (
                ACTUALIZACIONES_POR_PASO_SAC
            ),

            "factor_descuento": (
                FACTOR_DESCUENTO_SAC
            ),

            "tau_polyak": (
                TAU_POLYAK_SAC
            ),

            "entropia_objetivo": (
                ENTROPIA_OBJETIVO_SAC
            ),

            "dt": DT,

            "dimension_accion": (
                DIMENSION_ACCION_SAC
            ),

            "canales_parche": (
                CANALES_PARCHE_SAC
            ),

            "resolucion_parche": (
                RESOLUCION_PARCHE_SAC
            ),

            "dimension_escalares": (
                DIMENSION_ESCALARES_SAC
            ),
        },

        "estado_aleatorio_torch": (
            torch.get_rng_state()
        ),

        "estado_aleatorio_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    # ------------------------------------------------------
    # 7. Crear el directorio
    # ------------------------------------------------------

    ruta = Path(
        ruta_checkpoint
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_temporal = ruta.with_suffix(
        ruta.suffix
        + ".temporal"
    )

    # ------------------------------------------------------
    # 8. Guardado atómico
    # ------------------------------------------------------

    torch.save(
        checkpoint,
        ruta_temporal,
    )

    ruta_temporal.replace(
        ruta
    )

    # ------------------------------------------------------
    # 9. Verificar el archivo
    # ------------------------------------------------------

    if not ruta.is_file():

        raise RuntimeError(
            "No fue posible crear el checkpoint SAC."
        )

    tamano_bytes = int(
        ruta.stat().st_size
    )

    if tamano_bytes <= 0:

        raise RuntimeError(
            "El checkpoint SAC está vacío."
        )

    return {
        "ruta": str(
            ruta
        ),

        "tamano_bytes": tamano_bytes,

        "episodios_entrenados": int(
            resumen_global[
                "numero_episodios"
            ]
        ),

        "transiciones_guardadas": (
            filas_guardadas
        ),

        "buffer_completo": (
            tamano == capacidad
        ),
    }
def evaluar_actor_sac_multisemilla(
    actor,
    semilla_base,
    numero_semillas,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    dt=DT,
    dispositivo=DISPOSITIVO_SAC,
    frecuencia_impresion=(
        FRECUENCIA_IMPRESION_VALIDACION_SAC
    ),
):

    # ------------------------------------------------------
    # 1. Validar entradas
    # ------------------------------------------------------

    if not isinstance(
        actor,
        nn.Module,
    ):

        raise TypeError(
            "El actor debe ser un módulo de PyTorch."
        )

    semilla_base = int(
        semilla_base
    )

    numero_semillas = int(
        numero_semillas
    )

    pasos_maximos = int(
        pasos_maximos
    )

    frecuencia_impresion = int(
        frecuencia_impresion
    )

    dt = float(
        dt
    )

    if numero_semillas <= 0:

        raise ValueError(
            "El número de semillas debe ser positivo."
        )

    if pasos_maximos <= 0:

        raise ValueError(
            "El número máximo de pasos debe ser positivo."
        )

    if frecuencia_impresion < 0:

        raise ValueError(
            "La frecuencia de impresión no puede ser "
            "negativa."
        )

    if (
        not np.isfinite(
            dt
        )
        or dt <= 0.0
    ):

        raise ValueError(
            "El periodo de muestreo debe ser positivo."
        )

    # ------------------------------------------------------
    # 2. Preparar contadores
    # ------------------------------------------------------

    conteos_resultados = {
        "meta": 0,
        "colision_estatica": 0,
        "colision_dinamica": 0,
        "fuera_mapa": 0,
        "timeout": 0,
    }

    resultados_individuales = []

    modo_actor_anterior = actor.training

    # ------------------------------------------------------
    # 3. Función local de promedio finito
    # ------------------------------------------------------

    def promedio_finito(
        valores,
    ):

        arreglo = np.asarray(
            valores,
            dtype=float,
        )

        mascara = np.isfinite(
            arreglo
        )

        if not np.any(
            mascara
        ):

            return float(
                "nan"
            )

        return float(
            np.mean(
                arreglo[
                    mascara
                ]
            )
        )

    # ------------------------------------------------------
    # 4. Evaluar todas las semillas
    # ------------------------------------------------------

    try:

        actor.eval()

        for indice in range(
            numero_semillas
        ):

            numero_evaluacion = (
                indice + 1
            )

            semilla = (
                semilla_base
                + indice
            )

            resultado_evaluacion = (
                evaluar_episodio_sac(
                    semilla=semilla,

                    actor=actor,

                    pasos_maximos=(
                        pasos_maximos
                    ),

                    dt=dt,

                    dispositivo=dispositivo,
                )
            )

            metricas = resultado_evaluacion[
                "metricas"
            ]

            resultado = resultado_evaluacion[
                "resultado"
            ]

            if resultado not in conteos_resultados:

                raise ValueError(
                    "La evaluación produjo un resultado "
                    f"desconocido: {resultado}."
                )

            conteos_resultados[
                resultado
            ] += 1

            registro_semilla = {
                "numero_evaluacion": (
                    numero_evaluacion
                ),

                "semilla": semilla,

                "resultado": resultado,

                "exito": int(
                    bool(
                        metricas[
                            "exito"
                        ]
                    )
                ),

                "colision_estatica": int(
                    bool(
                        metricas[
                            "colision_estatica"
                        ]
                    )
                ),

                "colision_dinamica": int(
                    bool(
                        metricas[
                            "colision_dinamica"
                        ]
                    )
                ),

                "fuera_mapa": int(
                    bool(
                        metricas[
                            "fuera_mapa"
                        ]
                    )
                ),

                "timeout": int(
                    bool(
                        metricas[
                            "timeout"
                        ]
                    )
                ),

                "pasos_ejecutados": int(
                    metricas[
                        "pasos_ejecutados"
                    ]
                ),

                "tiempo_navegacion": float(
                    metricas[
                        "tiempo_navegacion"
                    ]
                ),

                "recompensa_acumulada": float(
                    resultado_evaluacion[
                        "recompensa_acumulada"
                    ]
                ),

                "longitud_recorrida": float(
                    metricas[
                        "longitud_recorrida"
                    ]
                ),

                "distancia_final_meta": float(
                    metricas[
                        "distancia_final_meta"
                    ]
                ),

                "eficiencia_navegacion": float(
                    metricas[
                        "eficiencia_navegacion"
                    ]
                ),

                "error_medio_ruta": float(
                    metricas[
                        "error_medio_ruta"
                    ]
                ),

                "error_rmse_ruta": float(
                    metricas[
                        "error_rmse_ruta"
                    ]
                ),

                "error_maximo_ruta": float(
                    metricas[
                        "error_maximo_ruta"
                    ]
                ),

                "clearance_estatico_minimo": float(
                    metricas[
                        "clearance_estatico_minimo"
                    ]
                ),

                "clearance_dinamico_minimo": float(
                    metricas[
                        "clearance_dinamico_minimo"
                    ]
                ),

                "clearance_total_minimo": float(
                    metricas[
                        "clearance_total_minimo"
                    ]
                ),

                "clearance_total_promedio": float(
                    metricas[
                        "clearance_total_promedio"
                    ]
                ),

                "variacion_total_v": float(
                    metricas[
                        "variacion_total_v"
                    ]
                ),

                "variacion_total_omega": float(
                    metricas[
                        "variacion_total_omega"
                    ]
                ),

                "esfuerzo_control": float(
                    metricas[
                        "esfuerzo_control"
                    ]
                ),
            }

            resultados_individuales.append(
                registro_semilla
            )

            imprimir = (
                frecuencia_impresion > 0
                and (
                    numero_evaluacion == 1
                    or numero_evaluacion
                    % frecuencia_impresion == 0
                    or numero_evaluacion
                    == numero_semillas
                )
            )

            if imprimir:

                exitos_actuales = conteos_resultados[
                    "meta"
                ]

                tasa_actual = (
                    exitos_actuales
                    / numero_evaluacion
                )

                print(
                    f"Validación "
                    f"{numero_evaluacion:3d}/"
                    f"{numero_semillas:3d} | "
                    f"semilla = {semilla:6d} | "
                    f"resultado = {resultado:18s} | "
                    f"éxito acumulado = "
                    f"{100.0 * tasa_actual:6.2f}% | "
                    f"pasos = "
                    f"{metricas['pasos_ejecutados']:4d}"
                )

    finally:

        actor.train(
            modo_actor_anterior
        )

    # ------------------------------------------------------
    # 5. Calcular tasas
    # ------------------------------------------------------

    numero_exitos = conteos_resultados[
        "meta"
    ]

    tasa_exito = (
        numero_exitos
        / numero_semillas
    )

    tasa_colision_estatica = (
        conteos_resultados[
            "colision_estatica"
        ]
        / numero_semillas
    )

    tasa_colision_dinamica = (
        conteos_resultados[
            "colision_dinamica"
        ]
        / numero_semillas
    )

    tasa_fuera_mapa = (
        conteos_resultados[
            "fuera_mapa"
        ]
        / numero_semillas
    )

    tasa_timeout = (
        conteos_resultados[
            "timeout"
        ]
        / numero_semillas
    )

    # ------------------------------------------------------
    # 6. Intervalo de confianza de Wilson para éxito
    # ------------------------------------------------------

    z_95 = 1.959963984540054

    denominador = (
        1.0
        + (
            z_95 ** 2
        )
        / numero_semillas
    )

    centro_wilson = (
        tasa_exito
        + (
            z_95 ** 2
        )
        / (
            2.0
            * numero_semillas
        )
    ) / denominador

    semiancho_wilson = (
        z_95
        * math.sqrt(
            (
                tasa_exito
                * (
                    1.0
                    - tasa_exito
                )
                + (
                    z_95 ** 2
                )
                / (
                    4.0
                    * numero_semillas
                )
            )
            / numero_semillas
        )
        / denominador
    )

    limite_inferior_exito_95 = max(
        0.0,
        centro_wilson
        - semiancho_wilson,
    )

    limite_superior_exito_95 = min(
        1.0,
        centro_wilson
        + semiancho_wilson,
    )

    # ------------------------------------------------------
    # 7. Separar resultados exitosos
    # ------------------------------------------------------

    resultados_exitosos = [
        registro
        for registro in resultados_individuales
        if registro[
            "exito"
        ] == 1
    ]

    # ------------------------------------------------------
    # 8. Estadísticas globales
    # ------------------------------------------------------

    resumen = {
        "semilla_base": semilla_base,

        "numero_semillas": numero_semillas,

        "numero_exitos": numero_exitos,

        "tasa_exito": float(
            tasa_exito
        ),

        "intervalo_exito_95_inferior": float(
            limite_inferior_exito_95
        ),

        "intervalo_exito_95_superior": float(
            limite_superior_exito_95
        ),

        "conteos_resultados": (
            conteos_resultados.copy()
        ),

        "tasa_colision_estatica": float(
            tasa_colision_estatica
        ),

        "tasa_colision_dinamica": float(
            tasa_colision_dinamica
        ),

        "tasa_fuera_mapa": float(
            tasa_fuera_mapa
        ),

        "tasa_timeout": float(
            tasa_timeout
        ),

        "recompensa_media": promedio_finito(
            [
                registro[
                    "recompensa_acumulada"
                ]
                for registro in resultados_individuales
            ]
        ),

        "pasos_promedio": promedio_finito(
            [
                registro[
                    "pasos_ejecutados"
                ]
                for registro in resultados_individuales
            ]
        ),

        "distancia_final_media": promedio_finito(
            [
                registro[
                    "distancia_final_meta"
                ]
                for registro in resultados_individuales
            ]
        ),

        "error_medio_ruta_promedio": promedio_finito(
            [
                registro[
                    "error_medio_ruta"
                ]
                for registro in resultados_individuales
            ]
        ),

        "clearance_minimo_promedio": promedio_finito(
            [
                registro[
                    "clearance_total_minimo"
                ]
                for registro in resultados_individuales
            ]
        ),

        "clearance_minimo_absoluto": float(
            np.min(
                np.asarray(
                    [
                        registro[
                            "clearance_total_minimo"
                        ]
                        for registro
                        in resultados_individuales
                        if np.isfinite(
                            registro[
                                "clearance_total_minimo"
                            ]
                        )
                    ],
                    dtype=float,
                )
            )
        ),

        "tiempo_promedio_exitos": promedio_finito(
            [
                registro[
                    "tiempo_navegacion"
                ]
                for registro in resultados_exitosos
            ]
        ),

        "longitud_promedio_exitos": promedio_finito(
            [
                registro[
                    "longitud_recorrida"
                ]
                for registro in resultados_exitosos
            ]
        ),

        "error_ruta_promedio_exitos": promedio_finito(
            [
                registro[
                    "error_medio_ruta"
                ]
                for registro in resultados_exitosos
            ]
        ),

        "clearance_minimo_promedio_exitos": (
            promedio_finito(
                [
                    registro[
                        "clearance_total_minimo"
                    ]
                    for registro in resultados_exitosos
                ]
            )
        ),
    }

    return {
        "resultados_individuales": (
            resultados_individuales
        ),

        "resumen": resumen,
    }
def guardar_resultados_validacion_sac_csv(
    resultado_validacion,
    ruta_csv,
):

    if not isinstance(
        resultado_validacion,
        dict,
    ):

        raise TypeError(
            "El resultado de validación debe ser un "
            "diccionario."
        )

    if (
        "resultados_individuales"
        not in resultado_validacion
    ):

        raise KeyError(
            "Faltan los resultados individuales."
        )

    registros = resultado_validacion[
        "resultados_individuales"
    ]

    if len(
        registros
    ) == 0:

        raise ValueError(
            "No existen resultados para guardar."
        )

    ruta = Path(
        ruta_csv
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nombres_columnas = list(
        registros[
            0
        ].keys()
    )

    with ruta.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=nombres_columnas,
        )

        escritor.writeheader()

        escritor.writerows(
            registros
        )

    if not ruta.is_file():

        raise RuntimeError(
            "No fue posible crear el archivo CSV."
        )

    return str(
        ruta
    )
def guardar_checkpoint_mejor_actor_sac(
    ruta_checkpoint,
    actor,
    episodio,
    resumen_validacion,
    historial_validaciones,
    criterio_seleccion,
):

    # ------------------------------------------------------
    # 1. Validar entradas
    # ------------------------------------------------------

    if not isinstance(
        actor,
        nn.Module,
    ):

        raise TypeError(
            "El actor debe ser un módulo de PyTorch."
        )

    episodio = int(
        episodio
    )

    if episodio <= 0:

        raise ValueError(
            "El episodio del mejor actor debe ser "
            "positivo."
        )

    if not isinstance(
        resumen_validacion,
        dict,
    ):

        raise TypeError(
            "El resumen de validación debe ser un "
            "diccionario."
        )

    if not isinstance(
        historial_validaciones,
        list,
    ):

        raise TypeError(
            "El historial de validaciones debe ser una "
            "lista."
        )

    if not isinstance(
        criterio_seleccion,
        tuple,
    ):

        raise TypeError(
            "El criterio de selección debe ser una tupla."
        )

    if len(
        criterio_seleccion
    ) != 7:

        raise ValueError(
            "El criterio de selección del SAC reactivo "
            "mejorado debe contener siete elementos."
        )

    # ------------------------------------------------------
    # 2. Copiar los parámetros del actor a CPU
    # ------------------------------------------------------
    #
    # Esto permite cargar posteriormente el checkpoint
    # tanto en CPU como en CUDA.

    estado_actor_cpu = {
        nombre: tensor
        .detach()
        .cpu()
        .clone()

        for nombre, tensor in actor.state_dict().items()
    }

    # ------------------------------------------------------
    # 3. Copiar el historial de validaciones
    # ------------------------------------------------------

    historial_validaciones_copia = []

    for registro in historial_validaciones:

        registro_copia = {}

        for clave, valor in registro.items():

            if isinstance(
                valor,
                dict,
            ):

                registro_copia[
                    clave
                ] = valor.copy()

            elif isinstance(
                valor,
                list,
            ):

                registro_copia[
                    clave
                ] = valor.copy()

            elif isinstance(
                valor,
                tuple,
            ):

                registro_copia[
                    clave
                ] = tuple(
                    valor
                )

            else:

                registro_copia[
                    clave
                ] = valor

        historial_validaciones_copia.append(
            registro_copia
        )

    # ------------------------------------------------------
    # 4. Construir checkpoint
    # ------------------------------------------------------

    checkpoint = {
        "version_checkpoint": 2,

        "tipo": "mejor_actor_sac_reactivo_mejorado",

        "variante_sac": "reactivo_mejorado",

        "observacion_predictiva": False,

        "usa_ttc": False,

        "usa_limites_aceleracion": True,

        "usa_frenado_terminal": True,

        "usa_estancamiento_persistente": True,

        "episodio": episodio,

        "actor_state_dict": estado_actor_cpu,

        "resumen_validacion": (
            resumen_validacion.copy()
        ),

        "criterio_seleccion": tuple(
            float(
                valor
            )
            for valor in criterio_seleccion
        ),

        "historial_validaciones": (
            historial_validaciones_copia
        ),

        "arquitectura": {
            "dimension_accion": (
                DIMENSION_ACCION_SAC
            ),

            "canales_parche": (
                CANALES_PARCHE_SAC
            ),

            "resolucion_parche": (
                RESOLUCION_PARCHE_SAC
            ),

            "dimension_escalares": (
                DIMENSION_ESCALARES_SAC
            ),
        },

        "validacion": {
            "semilla_base": int(
                resumen_validacion[
                    "semilla_base"
                ]
            ),

            "numero_semillas": int(
                resumen_validacion[
                    "numero_semillas"
                ]
            ),
        },
    }

    # ------------------------------------------------------
    # 5. Crear directorio y ruta temporal
    # ------------------------------------------------------

    ruta = Path(
        ruta_checkpoint
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_temporal = ruta.with_suffix(
        ruta.suffix
        + ".temporal"
    )

    # ------------------------------------------------------
    # 6. Guardado atómico
    # ------------------------------------------------------

    torch.save(
        checkpoint,
        ruta_temporal,
    )

    ruta_temporal.replace(
        ruta
    )

    # ------------------------------------------------------
    # 7. Verificar archivo
    # ------------------------------------------------------

    if not ruta.is_file():

        raise RuntimeError(
            "No fue posible guardar el mejor actor SAC."
        )

    tamano_bytes = int(
        ruta.stat().st_size
    )

    if tamano_bytes <= 0:

        raise RuntimeError(
            "El checkpoint del mejor actor está vacío."
        )

    return {
        "ruta": str(
            ruta
        ),

        "episodio": episodio,

        "tamano_bytes": tamano_bytes,

        "criterio_seleccion": tuple(
            criterio_seleccion
        ),

        "tasa_exito": float(
            resumen_validacion[
                "tasa_exito"
            ]
        ),

        "tasa_colision_dinamica": float(
            resumen_validacion[
                "tasa_colision_dinamica"
            ]
        ),

        "tasa_timeout": float(
            resumen_validacion[
                "tasa_timeout"
            ]
        ),
    }
def entrenar_agente_sac_con_validacion_periodica(
    numero_episodios,
    episodios_entre_validaciones,
    semilla_entrenamiento,
    semilla_validacion,
    numero_semillas_validacion,
    ruta_mejor_actor,
    actor,
    critico_q1,
    critico_q2,
    critico_q1_objetivo,
    critico_q2_objetivo,
    temperatura,
    optimizador_actor,
    optimizador_q1,
    optimizador_q2,
    optimizador_alpha,
    buffer,
    pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
    tamano_lote=TAMANO_LOTE_SAC,
    transiciones_aleatorias_iniciales=(
        TRANSICIONES_ALEATORIAS_INICIALES_SAC
    ),
    actualizaciones_por_paso=(
        ACTUALIZACIONES_POR_PASO_SAC
    ),
    factor_descuento=FACTOR_DESCUENTO_SAC,
    tau=TAU_POLYAK_SAC,
    entropia_objetivo=ENTROPIA_OBJETIVO_SAC,
    ventana_promedio=VENTANA_PROMEDIO_MOVIL_SAC,
    dt=DT,
    dispositivo=DISPOSITIVO_SAC,
):

    # ------------------------------------------------------
    # 1. Validar parámetros
    # ------------------------------------------------------

    numero_episodios = int(
        numero_episodios
    )

    episodios_entre_validaciones = int(
        episodios_entre_validaciones
    )

    semilla_entrenamiento = int(
        semilla_entrenamiento
    )

    semilla_validacion = int(
        semilla_validacion
    )

    numero_semillas_validacion = int(
        numero_semillas_validacion
    )

    ventana_promedio = int(
        ventana_promedio
    )

    if numero_episodios <= 0:

        raise ValueError(
            "El número de episodios debe ser positivo."
        )

    if episodios_entre_validaciones <= 0:

        raise ValueError(
            "El intervalo de validación debe ser "
            "positivo."
        )

    if numero_semillas_validacion <= 0:

        raise ValueError(
            "El número de semillas de validación debe "
            "ser positivo."
        )

    if ventana_promedio <= 0:

        raise ValueError(
            "La ventana de promedio debe ser positiva."
        )

    if not isinstance(
        buffer,
        dict,
    ):

        raise TypeError(
            "El replay buffer debe ser un diccionario."
        )

    # ------------------------------------------------------
    # 2. Función local para normalizar valores
    # ------------------------------------------------------

    def valor_finito_o_menos_infinito(
        valor,
    ):

        valor = float(
            valor
        )

        if np.isfinite(
            valor
        ):

            return valor

        return float(
            "-inf"
        )

    # ------------------------------------------------------
    # 3. Construir criterio de selección
    # ------------------------------------------------------
    #
    # Python compara las tuplas de izquierda a derecha.
    #
    # 1. Mayor tasa de éxito.
    # 2. Menor tasa de timeout.
    # 3. Menor tasa de colisión dinámica.
    # 4. Menor tasa de colisión estática.
    # 5. Menor tasa de salida del mapa.
    # 6. Menor distancia final media a la meta.
    # 7. Mayor clearance mínimo medio en episodios exitosos.
    #
    # La recompensa moldeada no se utiliza para desempatar,
    # porque cambió con las mejoras del entorno reactivo.

    def construir_criterio(
        resumen,
    ):

        tasa_exito = float(
            resumen[
                "tasa_exito"
            ]
        )

        tasa_timeout = float(
            resumen[
                "tasa_timeout"
            ]
        )

        tasa_colision_dinamica = float(
            resumen[
                "tasa_colision_dinamica"
            ]
        )

        tasa_colision_estatica = float(
            resumen[
                "tasa_colision_estatica"
            ]
        )

        tasa_fuera_mapa = float(
            resumen[
                "tasa_fuera_mapa"
            ]
        )

        distancia_final_media = float(
            resumen[
                "distancia_final_media"
            ]
        )

        if not np.isfinite(
            distancia_final_media
        ):

            distancia_final_media = float(
                "inf"
            )

        clearance_exitos = (
            valor_finito_o_menos_infinito(
                resumen[
                    "clearance_minimo_promedio_exitos"
                ]
            )
        )

        return (
            tasa_exito,

            -tasa_timeout,

            -tasa_colision_dinamica,

            -tasa_colision_estatica,

            -tasa_fuera_mapa,

            -distancia_final_media,

            clearance_exitos,
        )

    # ------------------------------------------------------
    # 4. Historial global
    # ------------------------------------------------------

    historial_global = None

    historial_validaciones = []

    episodios_completados = 0

    actualizaciones_acumuladas = 0

    mejor_criterio = None

    mejor_resumen_validacion = None

    mejor_episodio = None

    informacion_mejor_checkpoint = None

    ultimo_resultado_episodio = None

    # ------------------------------------------------------
    # 5. Series que se recalculan globalmente
    # ------------------------------------------------------

    claves_series_derivadas = {
        "recompensas_promedio_movil",
        "tasas_exito_acumuladas",
        "tasas_exito_moviles",
        "pasos_promedio_movil",
        "actualizaciones_acumuladas",
    }

    # ------------------------------------------------------
    # 6. Función local para recalcular medias
    # ------------------------------------------------------

    def recalcular_series_globales():

        historial_global[
            "recompensas_promedio_movil"
        ] = []

        historial_global[
            "tasas_exito_acumuladas"
        ] = []

        historial_global[
            "tasas_exito_moviles"
        ] = []

        historial_global[
            "pasos_promedio_movil"
        ] = []

        historial_global[
            "actualizaciones_acumuladas"
        ] = []

        acumulado_actualizaciones = 0

        numero_registros = len(
            historial_global[
                "episodios"
            ]
        )

        for indice in range(
            numero_registros
        ):

            inicio_ventana = max(
                0,
                indice
                + 1
                - ventana_promedio,
            )

            recompensas_ventana = np.asarray(
                historial_global[
                    "recompensas_acumuladas"
                ][
                    inicio_ventana:
                    indice + 1
                ],
                dtype=float,
            )

            exitos_ventana = np.asarray(
                historial_global[
                    "exitos"
                ][
                    inicio_ventana:
                    indice + 1
                ],
                dtype=float,
            )

            pasos_ventana = np.asarray(
                historial_global[
                    "pasos_ejecutados"
                ][
                    inicio_ventana:
                    indice + 1
                ],
                dtype=float,
            )

            exitos_acumulados = np.asarray(
                historial_global[
                    "exitos"
                ][
                    :indice + 1
                ],
                dtype=float,
            )

            historial_global[
                "recompensas_promedio_movil"
            ].append(
                float(
                    np.mean(
                        recompensas_ventana
                    )
                )
            )

            historial_global[
                "tasas_exito_moviles"
            ].append(
                float(
                    np.mean(
                        exitos_ventana
                    )
                )
            )

            historial_global[
                "tasas_exito_acumuladas"
            ].append(
                float(
                    np.mean(
                        exitos_acumulados
                    )
                )
            )

            historial_global[
                "pasos_promedio_movil"
            ].append(
                float(
                    np.mean(
                        pasos_ventana
                    )
                )
            )

            acumulado_actualizaciones += int(
                historial_global[
                    "actualizaciones_por_episodio"
                ][
                    indice
                ]
            )

            historial_global[
                "actualizaciones_acumuladas"
            ].append(
                acumulado_actualizaciones
            )

    # ------------------------------------------------------
    # 7. Entrenar por bloques
    # ------------------------------------------------------

    while episodios_completados < numero_episodios:

        episodios_restantes = (
            numero_episodios
            - episodios_completados
        )

        episodios_bloque = min(
            episodios_entre_validaciones,
            episodios_restantes,
        )

        semilla_bloque = (
            semilla_entrenamiento
            + episodios_completados
        )

        # --------------------------------------------------
        # 7.1. Entrenar el bloque
        # --------------------------------------------------

        resultado_bloque = entrenar_agente_sac(
            numero_episodios=episodios_bloque,

            semilla_base=semilla_bloque,

            actor=actor,

            critico_q1=critico_q1,

            critico_q2=critico_q2,

            critico_q1_objetivo=(
                critico_q1_objetivo
            ),

            critico_q2_objetivo=(
                critico_q2_objetivo
            ),

            temperatura=temperatura,

            optimizador_actor=(
                optimizador_actor
            ),

            optimizador_q1=(
                optimizador_q1
            ),

            optimizador_q2=(
                optimizador_q2
            ),

            optimizador_alpha=(
                optimizador_alpha
            ),

            buffer=buffer,

            pasos_maximos=pasos_maximos,

            tamano_lote=tamano_lote,

            transiciones_aleatorias_iniciales=(
                transiciones_aleatorias_iniciales
            ),

            actualizaciones_por_paso=(
                actualizaciones_por_paso
            ),

            factor_descuento=(
                factor_descuento
            ),

            tau=tau,

            entropia_objetivo=(
                entropia_objetivo
            ),

            ventana_promedio=(
                ventana_promedio
            ),

            frecuencia_impresion=0,

            guardar_episodios_completos=False,

            dt=dt,

            dispositivo=dispositivo,
        )

        historial_bloque = resultado_bloque[
            "historial_global"
        ]

        ultimo_resultado_episodio = (
            resultado_bloque[
                "ultimo_resultado_episodio"
            ]
        )

        # --------------------------------------------------
        # 7.2. Inicializar historial global
        # --------------------------------------------------

        if historial_global is None:

            historial_global = {
                clave: []
                for clave in historial_bloque.keys()
            }

        # --------------------------------------------------
        # 7.3. Agregar número global de episodio y semillas
        # --------------------------------------------------

        for indice_local in range(
            episodios_bloque
        ):

            numero_episodio_global = (
                episodios_completados
                + indice_local
                + 1
            )

            historial_global[
                "episodios"
            ].append(
                numero_episodio_global
            )

            historial_global[
                "semillas"
            ].append(
                semilla_entrenamiento
                + numero_episodio_global
                - 1
            )

        # --------------------------------------------------
        # 7.4. Combinar las demás series
        # --------------------------------------------------

        for clave, valores in historial_bloque.items():

            if clave in {
                "episodios",
                "semillas",
            }:

                continue

            if clave in claves_series_derivadas:

                continue

            historial_global[
                clave
            ].extend(
                list(
                    valores
                )
            )

        episodios_completados += episodios_bloque

        recalcular_series_globales()

        actualizaciones_acumuladas = int(
            historial_global[
                "actualizaciones_acumuladas"
            ][
                -1
            ]
        )

        # --------------------------------------------------
        # 7.5. Validar el actor actual
        # --------------------------------------------------

        resultado_validacion = (
            evaluar_actor_sac_multisemilla(
                actor=actor,

                semilla_base=(
                    semilla_validacion
                ),

                numero_semillas=(
                    numero_semillas_validacion
                ),

                pasos_maximos=(
                    pasos_maximos
                ),

                dt=dt,

                dispositivo=dispositivo,

                frecuencia_impresion=0,
            )
        )

        resumen_validacion = (
            resultado_validacion[
                "resumen"
            ]
        )

        criterio_actual = construir_criterio(
            resumen_validacion
        )

        nuevo_mejor = (
            mejor_criterio is None
            or criterio_actual
            > mejor_criterio
        )

        # --------------------------------------------------
        # 7.6. Registrar validación
        # --------------------------------------------------

        registro_validacion = {
            "episodio": episodios_completados,

            "tasa_exito": float(
                resumen_validacion[
                    "tasa_exito"
                ]
            ),

            "tasa_colision_dinamica": float(
                resumen_validacion[
                    "tasa_colision_dinamica"
                ]
            ),

            "tasa_colision_estatica": float(
                resumen_validacion[
                    "tasa_colision_estatica"
                ]
            ),

            "tasa_fuera_mapa": float(
                resumen_validacion[
                    "tasa_fuera_mapa"
                ]
            ),

            "tasa_timeout": float(
                resumen_validacion[
                    "tasa_timeout"
                ]
            ),

            "recompensa_media": float(
                resumen_validacion[
                    "recompensa_media"
                ]
            ),

            "distancia_final_media": float(
                resumen_validacion[
                    "distancia_final_media"
                ]
            ),

            "clearance_minimo_promedio_exitos": float(
                resumen_validacion[
                    "clearance_minimo_promedio_exitos"
                ]
            ),

            "conteos_resultados": (
                resumen_validacion[
                    "conteos_resultados"
                ].copy()
            ),

            "criterio": tuple(
                criterio_actual
            ),

            "nuevo_mejor": bool(
                nuevo_mejor
            ),
        }

        historial_validaciones.append(
            registro_validacion
        )

        # --------------------------------------------------
        # 7.7. Guardar cuando mejora
        # --------------------------------------------------

        if nuevo_mejor:

            mejor_criterio = tuple(
                criterio_actual
            )

            mejor_resumen_validacion = (
                resumen_validacion.copy()
            )

            mejor_episodio = (
                episodios_completados
            )

            informacion_mejor_checkpoint = (
                guardar_checkpoint_mejor_actor_sac(
                    ruta_checkpoint=(
                        ruta_mejor_actor
                    ),

                    actor=actor,

                    episodio=(
                        episodios_completados
                    ),

                    resumen_validacion=(
                        resumen_validacion
                    ),

                    historial_validaciones=(
                        historial_validaciones
                    ),

                    criterio_seleccion=(
                        mejor_criterio
                    ),
                )
            )

        # --------------------------------------------------
        # 7.8. Mostrar progreso
        # --------------------------------------------------

        recompensa_movil = float(
            historial_global[
                "recompensas_promedio_movil"
            ][
                -1
            ]
        )

        exito_movil = float(
            historial_global[
                "tasas_exito_moviles"
            ][
                -1
            ]
        )

        marca_mejor = (
            "NUEVO MEJOR"
            if nuevo_mejor
            else ""
        )

        print(
            f"Episodios "
            f"{episodios_completados:4d}/"
            f"{numero_episodios:4d} | "
            f"R móvil = "
            f"{recompensa_movil:9.3f} | "
            f"éxito entrenamiento = "
            f"{100.0 * exito_movil:6.2f}% | "
            f"validación = "
            f"{100.0 * resumen_validacion['tasa_exito']:6.2f}% | "
            f"timeout = "
            f"{100.0 * resumen_validacion['tasa_timeout']:6.2f}% | "
            f"col. dinámica = "
            f"{100.0 * resumen_validacion['tasa_colision_dinamica']:6.2f}% | "
            f"{marca_mejor}"
        )

    # ------------------------------------------------------
    # 8. Verificar que se guardó un mejor actor
    # ------------------------------------------------------

    if mejor_criterio is None:

        raise RuntimeError(
            "No fue posible seleccionar un mejor actor."
        )

    if informacion_mejor_checkpoint is None:

        raise RuntimeError(
            "No se guardó el checkpoint del mejor actor."
        )

    # ------------------------------------------------------
    # 9. Construir resumen global del entrenamiento
    # ------------------------------------------------------

    conteos_resultados = {
        "meta": 0,
        "colision_estatica": 0,
        "colision_dinamica": 0,
        "fuera_mapa": 0,
        "timeout": 0,
    }

    for resultado in historial_global[
        "resultados"
    ]:

        if resultado not in conteos_resultados:

            conteos_resultados[
                resultado
            ] = 0

        conteos_resultados[
            resultado
        ] += 1

    numero_exitos = int(
        conteos_resultados.get(
            "meta",
            0,
        )
    )

    recompensas_numpy = np.asarray(
        historial_global[
            "recompensas_acumuladas"
        ],
        dtype=float,
    )

    pasos_numpy = np.asarray(
        historial_global[
            "pasos_ejecutados"
        ],
        dtype=float,
    )

    resumen_global = {
        "numero_episodios": numero_episodios,

        "semilla_base": semilla_entrenamiento,

        "numero_exitos": numero_exitos,

        "tasa_exito": float(
            numero_exitos
            / numero_episodios
        ),

        "numero_colisiones": int(
            conteos_resultados.get(
                "colision_estatica",
                0,
            )
            + conteos_resultados.get(
                "colision_dinamica",
                0,
            )
        ),

        "conteos_resultados": (
            conteos_resultados.copy()
        ),

        "recompensa_media": float(
            np.mean(
                recompensas_numpy
            )
        ),

        "recompensa_maxima": float(
            np.max(
                recompensas_numpy
            )
        ),

        "recompensa_minima": float(
            np.min(
                recompensas_numpy
            )
        ),

        "pasos_promedio": float(
            np.mean(
                pasos_numpy
            )
        ),

        "actualizaciones_totales": int(
            actualizaciones_acumuladas
        ),

        "tamano_buffer_final": int(
            buffer[
                "tamano"
            ]
        ),

        "transiciones_totales_buffer": int(
            buffer[
                "total_transiciones_agregadas"
            ]
        ),

        "alpha_final": float(
            obtener_coeficiente_entropia_sac(
                temperatura=temperatura
            )
            .detach()
            .item()
        ),
    }

    # ------------------------------------------------------
    # 10. Resultado completo
    # ------------------------------------------------------

    return {
        "historial_global": historial_global,

        "resumen_global": resumen_global,

        "historial_validaciones": (
            historial_validaciones
        ),

        "mejor_episodio": mejor_episodio,

        "mejor_criterio": mejor_criterio,

        "mejor_resumen_validacion": (
            mejor_resumen_validacion
        ),

        "informacion_mejor_checkpoint": (
            informacion_mejor_checkpoint
        ),

        "ultimo_resultado_episodio": (
            ultimo_resultado_episodio
        ),

        "actor": actor,

        "critico_q1": critico_q1,

        "critico_q2": critico_q2,

        "critico_q1_objetivo": (
            critico_q1_objetivo
        ),

        "critico_q2_objetivo": (
            critico_q2_objetivo
        ),

        "temperatura": temperatura,

        "buffer": buffer,
    }
def guardar_historial_validaciones_periodicas_sac(
    historial_validaciones,
    ruta_csv,
    ruta_grafica,
    mejor_episodio,
    dpi=300,
):

    # ------------------------------------------------------
    # 1. Validar entradas
    # ------------------------------------------------------

    if not isinstance(
        historial_validaciones,
        list,
    ):

        raise TypeError(
            "El historial de validaciones debe ser una "
            "lista."
        )

    if len(
        historial_validaciones
    ) == 0:

        raise ValueError(
            "El historial de validaciones está vacío."
        )

    mejor_episodio = int(
        mejor_episodio
    )

    dpi = int(
        dpi
    )

    if dpi <= 0:

        raise ValueError(
            "El valor DPI debe ser positivo."
        )

    # ------------------------------------------------------
    # 2. Preparar registros planos
    # ------------------------------------------------------

    registros_csv = []

    for registro in historial_validaciones:

        conteos = registro.get(
            "conteos_resultados",
            {},
        )

        criterio = tuple(
            registro.get(
                "criterio",
                (),
            )
        )

        registro_plano = {
            "episodio": int(
                registro[
                    "episodio"
                ]
            ),

            "tasa_exito": float(
                registro[
                    "tasa_exito"
                ]
            ),

            "tasa_colision_dinamica": float(
                registro[
                    "tasa_colision_dinamica"
                ]
            ),

            "tasa_colision_estatica": float(
                registro[
                    "tasa_colision_estatica"
                ]
            ),

            "tasa_fuera_mapa": float(
                registro[
                    "tasa_fuera_mapa"
                ]
            ),

            "tasa_timeout": float(
                registro[
                    "tasa_timeout"
                ]
            ),

            "recompensa_media": float(
                registro[
                    "recompensa_media"
                ]
            ),

            "distancia_final_media": float(
                registro[
                    "distancia_final_media"
                ]
            ),

            "clearance_minimo_promedio_exitos": float(
                registro[
                    "clearance_minimo_promedio_exitos"
                ]
            ),

            "numero_metas": int(
                conteos.get(
                    "meta",
                    0,
                )
            ),

            "numero_colisiones_dinamicas": int(
                conteos.get(
                    "colision_dinamica",
                    0,
                )
            ),

            "numero_colisiones_estaticas": int(
                conteos.get(
                    "colision_estatica",
                    0,
                )
            ),

            "numero_fuera_mapa": int(
                conteos.get(
                    "fuera_mapa",
                    0,
                )
            ),

            "numero_timeout": int(
                conteos.get(
                    "timeout",
                    0,
                )
            ),

            "nuevo_mejor": int(
                bool(
                    registro[
                        "nuevo_mejor"
                    ]
                )
            ),

            "criterio_1_exito": (
                float(
                    criterio[
                        0
                    ]
                )
                if len(
                    criterio
                ) > 0
                else float(
                    "nan"
                )
            ),

            "criterio_2_timeout": (
                float(
                    criterio[
                        1
                    ]
                )
                if len(
                    criterio
                ) > 1
                else float(
                    "nan"
                )
            ),

            "criterio_3_colision_dinamica": (
                float(
                    criterio[
                        2
                    ]
                )
                if len(
                    criterio
                ) > 2
                else float(
                    "nan"
                )
            ),

            "criterio_4_colision_estatica": (
                float(
                    criterio[
                        3
                    ]
                )
                if len(
                    criterio
                ) > 3
                else float(
                    "nan"
                )
            ),

            "criterio_5_fuera_mapa": (
                float(
                    criterio[
                        4
                    ]
                )
                if len(
                    criterio
                ) > 4
                else float(
                    "nan"
                )
            ),

            "criterio_6_distancia_final": (
                float(
                    criterio[
                        5
                    ]
                )
                if len(
                    criterio
                ) > 5
                else float(
                    "nan"
                )
            ),

            "criterio_7_clearance": (
                float(
                    criterio[
                        6
                    ]
                )
                if len(
                    criterio
                ) > 6
                else float(
                    "nan"
                )
            ),
        }

        registros_csv.append(
            registro_plano
        )

    # ------------------------------------------------------
    # 3. Guardar CSV
    # ------------------------------------------------------

    ruta_csv = Path(
        ruta_csv
    )

    ruta_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ruta_csv.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,

            fieldnames=list(
                registros_csv[
                    0
                ].keys()
            ),
        )

        escritor.writeheader()

        escritor.writerows(
            registros_csv
        )

    # ------------------------------------------------------
    # 4. Preparar series
    # ------------------------------------------------------

    episodios = np.asarray(
        [
            registro[
                "episodio"
            ]
            for registro in registros_csv
        ],
        dtype=int,
    )

    tasas_exito = np.asarray(
        [
            registro[
                "tasa_exito"
            ]
            for registro in registros_csv
        ],
        dtype=float,
    )

    tasas_colision_dinamica = np.asarray(
        [
            registro[
                "tasa_colision_dinamica"
            ]
            for registro in registros_csv
        ],
        dtype=float,
    )

    tasas_colision_estatica = np.asarray(
        [
            registro[
                "tasa_colision_estatica"
            ]
            for registro in registros_csv
        ],
        dtype=float,
    )

    tasas_timeout = np.asarray(
        [
            registro[
                "tasa_timeout"
            ]
            for registro in registros_csv
        ],
        dtype=float,
    )

    tasas_fuera_mapa = np.asarray(
        [
            registro[
                "tasa_fuera_mapa"
            ]
            for registro in registros_csv
        ],
        dtype=float,
    )

    # ------------------------------------------------------
    # 5. Crear gráfica
    # ------------------------------------------------------

    figura, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    ax.plot(
        episodios,
        100.0
        * tasas_exito,

        marker="o",

        linewidth=2.2,

        label="Éxito de validación",
    )

    ax.plot(
        episodios,
        100.0
        * tasas_colision_dinamica,

        marker="s",

        linewidth=1.8,

        label="Colisión dinámica",
    )

    ax.plot(
        episodios,
        100.0
        * tasas_colision_estatica,

        marker="^",

        linewidth=1.8,

        label="Colisión estática",
    )

    ax.plot(
        episodios,
        100.0
        * tasas_timeout,

        marker="D",

        linewidth=1.8,

        label="Timeout",
    )

    ax.plot(
        episodios,
        100.0
        * tasas_fuera_mapa,

        marker="x",

        linewidth=1.5,

        label="Fuera del mapa",
    )

    ax.axvline(
        x=mejor_episodio,

        linestyle="--",

        linewidth=1.5,

        label=(
            "Mejor actor: episodio "
            f"{mejor_episodio}"
        ),
    )

    ax.set_xlabel(
        "Episodio de entrenamiento"
    )

    ax.set_ylabel(
        "Tasa sobre validación [%]"
    )

    ax.set_title(
        "Validación determinista periódica\n"
        "SAC reactivo mejorado: 50 semillas"
    )

    ax.set_ylim(
        0.0,
        105.0,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    figura.tight_layout()

    # ------------------------------------------------------
    # 6. Guardar gráfica
    # ------------------------------------------------------

    ruta_grafica = Path(
        ruta_grafica
    )

    ruta_grafica.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figura.savefig(
        ruta_grafica,

        dpi=dpi,

        bbox_inches="tight",
    )

    # ------------------------------------------------------
    # 7. Verificar archivos
    # ------------------------------------------------------

    if not ruta_csv.is_file():

        raise RuntimeError(
            "No se creó el CSV de validaciones."
        )

    if not ruta_grafica.is_file():

        raise RuntimeError(
            "No se creó la gráfica de validaciones."
        )

    return {
        "ruta_csv": str(
            ruta_csv
        ),

        "ruta_grafica": str(
            ruta_grafica
        ),

        "figura": figura,

        "numero_validaciones": len(
            registros_csv
        ),

        "mejor_episodio": mejor_episodio,

        "tasa_exito_maxima": float(
            np.max(
                tasas_exito
            )
        ),

        "tasa_colision_dinamica_minima": float(
            np.min(
                tasas_colision_dinamica
            )
        ),
    }
# ==========================================================
# GUARDAR SOLAMENTE LAS SEMILLAS FALLIDAS
# ==========================================================

def guardar_fallos_prueba_independiente_sac(
    resultado_validacion,
    ruta_csv,
):

    if not isinstance(
        resultado_validacion,
        dict,
    ):

        raise TypeError(
            "El resultado de validación debe ser un "
            "diccionario."
        )

    if (
        "resultados_individuales"
        not in resultado_validacion
    ):

        raise KeyError(
            "Faltan los resultados individuales."
        )

    registros = resultado_validacion[
        "resultados_individuales"
    ]

    fallos = [
        registro.copy()
        for registro in registros
        if int(
            registro[
                "exito"
            ]
        )
        == 0
    ]

    ruta = Path(
        ruta_csv
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(
        registros
    ) == 0:

        raise ValueError(
            "No existen evaluaciones para analizar."
        )

    nombres_columnas = list(
        registros[
            0
        ].keys()
    )

    with ruta.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=nombres_columnas,
        )

        escritor.writeheader()

        escritor.writerows(
            fallos
        )

    if not ruta.is_file():

        raise RuntimeError(
            "No se pudo crear el CSV de semillas fallidas."
        )

    return {
        "ruta": str(
            ruta
        ),

        "numero_fallos": len(
            fallos
        ),

        "semillas_fallidas": [
            int(
                registro[
                    "semilla"
                ]
            )
            for registro in fallos
        ],
    }

# ==========================================================
# CARGAR SEMILLAS FALLIDAS
# ==========================================================

def cargar_semillas_fallidas_sac(
    ruta_csv,
    semillas_respaldo,
):

    ruta_csv = Path(
        ruta_csv
    )

    semillas = []

    if ruta_csv.is_file():

        with ruta_csv.open(
            mode="r",
            newline="",
            encoding="utf-8",
        ) as archivo:

            lector = csv.DictReader(
                archivo
            )

            for fila in lector:

                if (
                    "semilla" in fila
                    and fila[
                        "semilla"
                    ] not in (
                        None,
                        "",
                    )
                ):

                    semillas.append(
                        int(
                            fila[
                                "semilla"
                            ]
                        )
                    )

    if len(
        semillas
    ) == 0:

        semillas = [
            int(
                semilla
            )
            for semilla in semillas_respaldo
        ]

    semillas = sorted(
        set(
            semillas
        )
    )

    if len(
        semillas
    ) == 0:

        raise ValueError(
            "No existen semillas fallidas para analizar."
        )

    return semillas
# ==========================================================
# BUSCAR UNA SERIE EN EL RESULTADO DE EVALUACIÓN
# ==========================================================
def buscar_serie_resultado_sac(
    resultado_evaluacion,
    posibles_claves,
):

    diccionarios = [
        resultado_evaluacion.get(
            "historial_visualizacion",
            {},
        ),

        resultado_evaluacion.get(
            "registro",
            {},
        ),

        resultado_evaluacion,
    ]

    for diccionario in diccionarios:

        if not isinstance(
            diccionario,
            dict,
        ):

            continue

        for clave in posibles_claves:

            if (
                clave in diccionario
                and diccionario[
                    clave
                ] is not None
            ):

                return (
                    diccionario[
                        clave
                    ],
                    clave,
                )

    return (
        None,
        None,
    )
# ==========================================================
# AJUSTAR UNA SERIE AL NÚMERO DE PASOS
# ==========================================================
def ajustar_serie_a_pasos_sac(
    valores,
    numero_pasos,
    valor_relleno=np.nan,
):

    numero_pasos = int(
        numero_pasos
    )

    if numero_pasos <= 0:

        raise ValueError(
            "El número de pasos debe ser positivo."
        )

    arreglo = np.asarray(
        valores,
        dtype=float,
    ).reshape(
        -1
    )

    # Algunas series incluyen el estado inicial.

    if arreglo.size == numero_pasos + 1:

        arreglo = arreglo[
            1:
        ]

    elif arreglo.size > numero_pasos:

        arreglo = arreglo[
            :numero_pasos
        ]

    elif arreglo.size < numero_pasos:

        faltantes = (
            numero_pasos
            - arreglo.size
        )

        arreglo = np.concatenate(
            [
                arreglo,

                np.full(
                    faltantes,
                    valor_relleno,
                    dtype=float,
                ),
            ]
        )

    return arreglo
# ==========================================================
# MAYOR BLOQUE CONTINUO DE VALORES VERDADEROS
# ==========================================================

def obtener_mayor_bloque_verdadero_sac(
    mascara,
):

    mascara = np.asarray(
        mascara,
        dtype=bool,
    ).reshape(
        -1
    )

    mejor_inicio = -1

    mejor_longitud = 0

    inicio_actual = -1

    longitud_actual = 0

    for indice, valor in enumerate(
        mascara
    ):

        if valor:

            if longitud_actual == 0:

                inicio_actual = indice

            longitud_actual += 1

            if longitud_actual > mejor_longitud:

                mejor_longitud = longitud_actual

                mejor_inicio = inicio_actual

        else:

            inicio_actual = -1

            longitud_actual = 0

    return (
        int(
            mejor_inicio
        ),

        int(
            mejor_longitud
        ),
    )
# ==========================================================
# EXTRAER SERIES TEMPORALES PARA DIAGNÓSTICO
# ==========================================================

def extraer_series_diagnostico_fallo_sac(
    resultado_evaluacion,
    dt=DT,
):

    if not isinstance(
        resultado_evaluacion,
        dict,
    ):

        raise TypeError(
            "El resultado de evaluación debe ser un "
            "diccionario."
        )

    dt = float(
        dt
    )

    metricas = resultado_evaluacion[
        "metricas"
    ]

    numero_pasos = int(
        resultado_evaluacion.get(
            "numero_pasos",

            metricas[
                "pasos_ejecutados"
            ],
        )
    )

    if numero_pasos <= 0:

        raise ValueError(
            "La evaluación no contiene pasos."
        )

    # ------------------------------------------------------
    # 1. Estados del robot
    # ------------------------------------------------------

    estados_brutos, clave_estados = (
        buscar_serie_resultado_sac(
            resultado_evaluacion,

            [
                "estados",
                "trayectoria_estados",
                "trayectoria",
            ],
        )
    )

    if estados_brutos is None:

        raise KeyError(
            "No se encontró la trayectoria de estados."
        )

    estados = np.asarray(
        estados_brutos,
        dtype=float,
    )

    if (
        estados.ndim != 2
        or estados.shape[
            1
        ] < 3
    ):

        raise ValueError(
            "La trayectoria del robot tiene una forma "
            "incorrecta."
        )

    if estados.shape[
        0
    ] == numero_pasos + 1:

        estados_pasos = estados[
            1:
        ]

    elif estados.shape[
        0
    ] >= numero_pasos:

        estados_pasos = estados[
            :numero_pasos
        ]

    else:

        raise ValueError(
            "La trayectoria contiene menos estados que "
            "pasos."
        )

    # ------------------------------------------------------
    # 2. Velocidades
    # ------------------------------------------------------

    velocidades_lineales_brutas, _ = (
        buscar_serie_resultado_sac(
            resultado_evaluacion,

            [
                "velocidades_lineales",
                "velocidad_lineal",
            ],
        )
    )

    velocidades_angulares_brutas, _ = (
        buscar_serie_resultado_sac(
            resultado_evaluacion,

            [
                "velocidades_angulares",
                "velocidad_angular",
            ],
        )
    )

    if (
        velocidades_lineales_brutas is None
        or velocidades_angulares_brutas is None
    ):

        controles_brutos, _ = (
            buscar_serie_resultado_sac(
                resultado_evaluacion,

                [
                    "controles",
                    "acciones_fisicas",
                ],
            )
        )

        if controles_brutos is None:

            raise KeyError(
                "No se encontraron los controles físicos."
            )

        controles = np.asarray(
            controles_brutos,
            dtype=float,
        )

        if (
            controles.ndim != 2
            or controles.shape[
                1
            ] < 2
        ):

            raise ValueError(
                "Los controles tienen una forma incorrecta."
            )

        velocidades_lineales_brutas = controles[
            :,
            0
        ]

        velocidades_angulares_brutas = controles[
            :,
            1
        ]

    velocidades_lineales = (
        ajustar_serie_a_pasos_sac(
            velocidades_lineales_brutas,
            numero_pasos,
        )
    )

    velocidades_angulares = (
        ajustar_serie_a_pasos_sac(
            velocidades_angulares_brutas,
            numero_pasos,
        )
    )

    # ------------------------------------------------------
    # 3. Clearances
    # ------------------------------------------------------

    clearances_dinamicos_brutos, _ = (
        buscar_serie_resultado_sac(
            resultado_evaluacion,

            [
                "clearances_dinamicos",
                "clearance_dinamico",
            ],
        )
    )

    clearances_estaticos_brutos, _ = (
        buscar_serie_resultado_sac(
            resultado_evaluacion,

            [
                "clearances_estaticos",
                "clearance_estatico",
            ],
        )
    )

    if clearances_dinamicos_brutos is None:

        raise KeyError(
            "No se encontró el historial de clearance "
            "dinámico."
        )

    clearances_dinamicos = (
        ajustar_serie_a_pasos_sac(
            clearances_dinamicos_brutos,
            numero_pasos,
            valor_relleno=float(
                "inf"
            ),
        )
    )

    if clearances_estaticos_brutos is None:

        clearances_estaticos = np.full(
            numero_pasos,
            float(
                "inf"
            ),
            dtype=float,
        )

    else:

        clearances_estaticos = (
            ajustar_serie_a_pasos_sac(
                clearances_estaticos_brutos,
                numero_pasos,
                valor_relleno=float(
                    "inf"
                ),
            )
        )

    clearances_totales = np.minimum(
        clearances_estaticos,
        clearances_dinamicos,
    )

    # ------------------------------------------------------
    # 4. Distancia a la meta
    # ------------------------------------------------------

    meta = resultado_evaluacion[
        "meta"
    ]

    distancias_meta = np.asarray(
        [
            distancia_entre_puntos(
                (
                    estado[
                        0
                    ],
                    estado[
                        1
                    ],
                ),

                meta,
            )

            for estado in estados_pasos
        ],
        dtype=float,
    )

    # ------------------------------------------------------
    # 5. Progreso sobre A*
    # ------------------------------------------------------

    indices_progreso_brutos, _ = (
        buscar_serie_resultado_sac(
            resultado_evaluacion,

            [
                "indices_progreso",
                "historial_indices_progreso",
                "indice_progreso",
            ],
        )
    )

    camino_mundo = resultado_evaluacion[
        "camino_mundo"
    ]

    if indices_progreso_brutos is None:

        indices_progreso = []

        for estado in estados_pasos:

            distancias_camino = [
                distancia_entre_puntos(
                    (
                        estado[
                            0
                        ],
                        estado[
                            1
                        ],
                    ),

                    punto_camino,
                )

                for punto_camino in camino_mundo
            ]

            indices_progreso.append(
                int(
                    np.argmin(
                        distancias_camino
                    )
                )
            )

        indices_progreso = np.maximum.accumulate(
            np.asarray(
                indices_progreso,
                dtype=float,
            )
        )

    else:

        indices_progreso = (
            ajustar_serie_a_pasos_sac(
                indices_progreso_brutos,
                numero_pasos,
                valor_relleno=0.0,
            )
        )

    denominador_progreso = max(
        len(
            camino_mundo
        )
        - 1,

        1,
    )

    progreso_porcentual = (
        100.0
        * indices_progreso
        / denominador_progreso
    )

    progreso_porcentual = np.clip(
        progreso_porcentual,
        0.0,
        100.0,
    )

    tiempos = (
        np.arange(
            1,
            numero_pasos + 1,
            dtype=float,
        )
        * dt
    )

    return {
        "numero_pasos": numero_pasos,

        "tiempos": tiempos,

        "estados": estados_pasos,

        "velocidades_lineales": (
            velocidades_lineales
        ),

        "velocidades_angulares": (
            velocidades_angulares
        ),

        "clearances_estaticos": (
            clearances_estaticos
        ),

        "clearances_dinamicos": (
            clearances_dinamicos
        ),

        "clearances_totales": (
            clearances_totales
        ),

        "distancias_meta": distancias_meta,

        "indices_progreso": indices_progreso,

        "progreso_porcentual": (
            progreso_porcentual
        ),

        "clave_estados": clave_estados,
    }
# ==========================================================
# CALCULAR DIAGNÓSTICO DE UNA SEMILLA FALLIDA
# ==========================================================
def calcular_diagnostico_fallo_sac(
    resultado_evaluacion,
    dt=DT,
):

    series = extraer_series_diagnostico_fallo_sac(
        resultado_evaluacion=resultado_evaluacion,
        dt=dt,
    )

    resultado = resultado_evaluacion[
        "resultado"
    ]

    metricas = resultado_evaluacion[
        "metricas"
    ]

    tiempos = series[
        "tiempos"
    ]

    velocidades_lineales = series[
        "velocidades_lineales"
    ]

    velocidades_angulares = series[
        "velocidades_angulares"
    ]

    clearances_dinamicos = series[
        "clearances_dinamicos"
    ]

    distancias_meta = series[
        "distancias_meta"
    ]

    progreso_porcentual = series[
        "progreso_porcentual"
    ]

    numero_pasos = series[
        "numero_pasos"
    ]

    # ------------------------------------------------------
    # 1. Clearance dinámico mínimo
    # ------------------------------------------------------

    indices_finitos = np.where(
        np.isfinite(
            clearances_dinamicos
        )
    )[
        0
    ]

    if indices_finitos.size > 0:

        indice_minimo_local = int(
            np.argmin(
                clearances_dinamicos[
                    indices_finitos
                ]
            )
        )

        indice_clearance_minimo = int(
            indices_finitos[
                indice_minimo_local
            ]
        )

        clearance_dinamico_minimo = float(
            clearances_dinamicos[
                indice_clearance_minimo
            ]
        )

    else:

        indice_clearance_minimo = (
            numero_pasos - 1
        )

        clearance_dinamico_minimo = float(
            "inf"
        )

    # ------------------------------------------------------
    # 2. Velocidad de cierre
    # ------------------------------------------------------

    velocidades_cierre = np.zeros(
        numero_pasos,
        dtype=float,
    )

    for indice in range(
        1,
        numero_pasos,
    ):

        clearance_anterior = (
            clearances_dinamicos[
                indice - 1
            ]
        )

        clearance_actual = (
            clearances_dinamicos[
                indice
            ]
        )

        if (
            np.isfinite(
                clearance_anterior
            )
            and np.isfinite(
                clearance_actual
            )
        ):

            derivada = (
                clearance_actual
                - clearance_anterior
            ) / dt

            velocidades_cierre[
                indice
            ] = max(
                0.0,
                -derivada,
            )

    velocidad_cierre_maxima = float(
        np.max(
            velocidades_cierre
        )
    )

    # ------------------------------------------------------
    # 3. Tiempo a colisión aproximado
    # ------------------------------------------------------

    tiempos_colision_aproximados = np.full(
        numero_pasos,
        float(
            "inf"
        ),
        dtype=float,
    )

    for indice in range(
        numero_pasos
    ):

        clearance = clearances_dinamicos[
            indice
        ]

        velocidad_cierre = velocidades_cierre[
            indice
        ]

        if (
            np.isfinite(
                clearance
            )
            and clearance > 0.0
            and velocidad_cierre > 1e-6
        ):

            tiempos_colision_aproximados[
                indice
            ] = (
                clearance
                / velocidad_cierre
            )

    ttc_finitos = tiempos_colision_aproximados[
        np.isfinite(
            tiempos_colision_aproximados
        )
    ]

    if ttc_finitos.size > 0:

        ttc_minimo = float(
            np.min(
                ttc_finitos
            )
        )

    else:

        ttc_minimo = float(
            "inf"
        )

    # ------------------------------------------------------
    # 4. Estancamiento
    # ------------------------------------------------------

    mascara_velocidad_baja = (
        velocidades_lineales
        < UMBRAL_VELOCIDAD_BAJA_DIAGNOSTICO_SAC
    )

    proporcion_velocidad_baja = float(
        np.mean(
            mascara_velocidad_baja
        )
    )

    (
        inicio_mayor_estancamiento,
        pasos_mayor_estancamiento,
    ) = obtener_mayor_bloque_verdadero_sac(
        mascara_velocidad_baja
    )

    duracion_mayor_estancamiento = (
        pasos_mayor_estancamiento
        * dt
    )

    # ------------------------------------------------------
    # 5. Oscilaciones angulares
    # ------------------------------------------------------

    signos_omega = np.sign(
        velocidades_angulares
    )

    signos_omega[
        np.abs(
            velocidades_angulares
        )
        < UMBRAL_OMEGA_OSCILACION_DIAGNOSTICO_SAC
    ] = 0.0

    signos_no_cero = signos_omega[
        signos_omega != 0.0
    ]

    if signos_no_cero.size >= 2:

        numero_cambios_signo_omega = int(
            np.sum(
                signos_no_cero[
                    1:
                ]
                != signos_no_cero[
                    :-1
                ]
            )
        )

    else:

        numero_cambios_signo_omega = 0

    # ------------------------------------------------------
    # 6. Elegir instante crítico
    # ------------------------------------------------------

    if resultado == "colision_dinamica":

        indice_critico = (
            indice_clearance_minimo
        )

    elif resultado == "timeout":

        if inicio_mayor_estancamiento >= 0:

            indice_critico = (
                inicio_mayor_estancamiento
                + pasos_mayor_estancamiento
                - 1
            )

        else:

            indice_critico = (
                numero_pasos - 1
            )

    else:

        indice_critico = (
            numero_pasos - 1
        )

    indice_critico = int(
        np.clip(
            indice_critico,
            0,
            numero_pasos - 1,
        )
    )

    # ------------------------------------------------------
    # 7. Clasificación heurística
    # ------------------------------------------------------

    progreso_final = float(
        progreso_porcentual[
            -1
        ]
    )

    velocidad_en_clearance_minimo = float(
        velocidades_lineales[
            indice_clearance_minimo
        ]
    )

    if resultado == "colision_dinamica":

        if (
            np.isfinite(
                ttc_minimo
            )
            and ttc_minimo
            < TTC_RIESGOSO_DIAGNOSTICO_SAC
        ):

            causa_probable = (
                "aproximacion_dinamica_rapida"
            )

        elif velocidad_en_clearance_minimo > 0.45:

            causa_probable = (
                "velocidad_alta_cerca_del_obstaculo"
            )

        else:

            causa_probable = (
                "maniobra_evasiva_tardia_o_insuficiente"
            )

    elif resultado == "timeout":

        if duracion_mayor_estancamiento >= 4.0:

            causa_probable = (
                "estancamiento_prolongado"
            )

        elif numero_cambios_signo_omega >= 15:

            causa_probable = (
                "oscilacion_angular"
            )

        elif progreso_final < 70.0:

            causa_probable = (
                "progreso_insuficiente"
            )

        else:

            causa_probable = (
                "avance_excesivamente_lento"
            )

    elif resultado == "fuera_mapa":

        causa_probable = (
            "desviacion_excesiva_de_la_ruta"
        )

    elif resultado == "colision_estatica":

        causa_probable = (
            "margen_estatico_insuficiente"
        )

    else:

        causa_probable = (
            "resultado_no_clasificado"
        )

    diagnostico = {
        "semilla": int(
            resultado_evaluacion[
                "semilla"
            ]
        ),

        "resultado": resultado,

        "causa_probable": causa_probable,

        "pasos_ejecutados": int(
            numero_pasos
        ),

        "tiempo_total_s": float(
            numero_pasos
            * dt
        ),

        "distancia_final_meta_m": float(
            metricas[
                "distancia_final_meta"
            ]
        ),

        "error_medio_ruta_m": float(
            metricas[
                "error_medio_ruta"
            ]
        ),

        "clearance_dinamico_minimo_m": (
            clearance_dinamico_minimo
        ),

        "tiempo_clearance_minimo_s": float(
            tiempos[
                indice_clearance_minimo
            ]
        ),

        "velocidad_en_clearance_minimo_m_s": (
            velocidad_en_clearance_minimo
        ),

        "omega_en_clearance_minimo_rad_s": float(
            velocidades_angulares[
                indice_clearance_minimo
            ]
        ),

        "velocidad_cierre_maxima_m_s": (
            velocidad_cierre_maxima
        ),

        "ttc_minimo_aproximado_s": (
            ttc_minimo
        ),

        "proporcion_velocidad_baja": (
            proporcion_velocidad_baja
        ),

        "porcentaje_velocidad_baja": float(
            100.0
            * proporcion_velocidad_baja
        ),

        "pasos_mayor_estancamiento": int(
            pasos_mayor_estancamiento
        ),

        "duracion_mayor_estancamiento_s": float(
            duracion_mayor_estancamiento
        ),

        "numero_cambios_signo_omega": int(
            numero_cambios_signo_omega
        ),

        "progreso_final_astar_pct": (
            progreso_final
        ),

        "tiempo_critico_s": float(
            tiempos[
                indice_critico
            ]
        ),

        "velocidad_critica_m_s": float(
            velocidades_lineales[
                indice_critico
            ]
        ),

        "omega_critica_rad_s": float(
            velocidades_angulares[
                indice_critico
            ]
        ),

        "clearance_dinamico_critico_m": float(
            clearances_dinamicos[
                indice_critico
            ]
        ),

        "distancia_meta_critica_m": float(
            distancias_meta[
                indice_critico
            ]
        ),

        "progreso_critico_astar_pct": float(
            progreso_porcentual[
                indice_critico
            ]
        ),

        "indice_critico": int(
            indice_critico
        ),
    }

    return (
        diagnostico,
        series,
    )
# ==========================================================
# GRÁFICA TEMPORAL DE UNA SEMILLA FALLIDA
# ==========================================================
def graficar_diagnostico_temporal_fallo_sac(
    diagnostico,
    series,
    ruta_guardado,
):

    tiempos = series[
        "tiempos"
    ]

    tiempo_critico = diagnostico[
        "tiempo_critico_s"
    ]

    figura, ejes = plt.subplots(
        5,
        1,
        figsize=(
            12,
            15,
        ),
        sharex=True,
    )

    # ------------------------------------------------------
    # 1. Clearance dinámico
    # ------------------------------------------------------

    ejes[
        0
    ].plot(
        tiempos,
        series[
            "clearances_dinamicos"
        ],
        linewidth=2.0,
        label="Clearance dinámico",
    )

    ejes[
        0
    ].axhline(
        0.0,
        linestyle="--",
        label="Colisión",
    )

    ejes[
        0
    ].axhline(
        CLEARANCE_SEGURIDAD_DWA,
        linestyle=":",
        label="Margen DWA de referencia",
    )

    ejes[
        0
    ].set_ylabel(
        "Clearance [m]"
    )

    ejes[
        0
    ].legend()

    ejes[
        0
    ].grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------
    # 2. Velocidad lineal
    # ------------------------------------------------------

    ejes[
        1
    ].plot(
        tiempos,
        series[
            "velocidades_lineales"
        ],
        linewidth=2.0,
        label="Velocidad lineal",
    )

    ejes[
        1
    ].axhline(
        UMBRAL_VELOCIDAD_BAJA_DIAGNOSTICO_SAC,
        linestyle="--",
        label="Umbral de velocidad baja",
    )

    ejes[
        1
    ].set_ylabel(
        "v [m/s]"
    )

    ejes[
        1
    ].legend()

    ejes[
        1
    ].grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------
    # 3. Velocidad angular
    # ------------------------------------------------------

    ejes[
        2
    ].plot(
        tiempos,
        series[
            "velocidades_angulares"
        ],
        linewidth=2.0,
        label="Velocidad angular",
    )

    ejes[
        2
    ].axhline(
        0.0,
        linestyle="--",
    )

    ejes[
        2
    ].set_ylabel(
        "ω [rad/s]"
    )

    ejes[
        2
    ].legend()

    ejes[
        2
    ].grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------
    # 4. Distancia a meta
    # ------------------------------------------------------

    ejes[
        3
    ].plot(
        tiempos,
        series[
            "distancias_meta"
        ],
        linewidth=2.0,
        label="Distancia a la meta",
    )

    ejes[
        3
    ].axhline(
        DISTANCIA_META,
        linestyle="--",
        label="Tolerancia de llegada",
    )

    ejes[
        3
    ].set_ylabel(
        "Distancia [m]"
    )

    ejes[
        3
    ].legend()

    ejes[
        3
    ].grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------
    # 5. Progreso A*
    # ------------------------------------------------------

    ejes[
        4
    ].plot(
        tiempos,
        series[
            "progreso_porcentual"
        ],
        linewidth=2.0,
        label="Progreso sobre A*",
    )

    ejes[
        4
    ].set_xlabel(
        "Tiempo [s]"
    )

    ejes[
        4
    ].set_ylabel(
        "Progreso [%]"
    )

    ejes[
        4
    ].set_ylim(
        0.0,
        105.0,
    )

    ejes[
        4
    ].legend()

    ejes[
        4
    ].grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------
    # 6. Instante crítico común
    # ------------------------------------------------------

    for eje in ejes:

        eje.axvline(
            tiempo_critico,
            linestyle="--",
            linewidth=1.5,
            label="_nolegend_",
        )

    figura.suptitle(
        "Diagnóstico temporal SAC\n"
        f"Semilla {diagnostico['semilla']} | "
        f"{diagnostico['resultado']} | "
        f"{diagnostico['causa_probable']}"
    )

    figura.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.96,
        )
    )

    ruta = Path(
        ruta_guardado
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figura.savefig(
        ruta,
        dpi=250,
        bbox_inches="tight",
    )

    return {
        "figura": figura,

        "ruta": str(
            ruta
        ),
    }
# ==========================================================
# GUARDAR DIAGNÓSTICOS EN CSV
# ==========================================================
def guardar_diagnosticos_fallos_sac_csv(
    diagnosticos,
    ruta_csv,
):

    if len(
        diagnosticos
    ) == 0:

        raise ValueError(
            "No existen diagnósticos para guardar."
        )

    ruta = Path(
        ruta_csv
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ruta.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,

            fieldnames=list(
                diagnosticos[
                    0
                ].keys()
            ),
        )

        escritor.writeheader()

        escritor.writerows(
            diagnosticos
        )

    return str(
        ruta
    )
# ==========================================================
# RESUMEN POR TIPO DE FALLO
# ==========================================================
def crear_resumen_tipos_fallo_sac(
    diagnosticos,
    ruta_csv,
):

    tipos = sorted(
        set(
            diagnostico[
                "resultado"
            ]
            for diagnostico in diagnosticos
        )
    )

    resumenes = []

    for tipo in tipos:

        grupo = [
            diagnostico
            for diagnostico in diagnosticos
            if diagnostico[
                "resultado"
            ]
            == tipo
        ]

        def promedio_finito(
            clave,
        ):

            valores = np.asarray(
                [
                    registro[
                        clave
                    ]
                    for registro in grupo
                ],
                dtype=float,
            )

            valores = valores[
                np.isfinite(
                    valores
                )
            ]

            if valores.size == 0:

                return float(
                    "nan"
                )

            return float(
                np.mean(
                    valores
                )
            )

        resumenes.append(
            {
                "resultado": tipo,

                "numero_casos": len(
                    grupo
                ),

                "pasos_promedio": promedio_finito(
                    "pasos_ejecutados"
                ),

                "clearance_dinamico_minimo_promedio_m": (
                    promedio_finito(
                        "clearance_dinamico_minimo_m"
                    )
                ),

                "velocidad_cierre_maxima_promedio_m_s": (
                    promedio_finito(
                        "velocidad_cierre_maxima_m_s"
                    )
                ),

                "ttc_minimo_promedio_s": promedio_finito(
                    "ttc_minimo_aproximado_s"
                ),

                "porcentaje_velocidad_baja_promedio": (
                    promedio_finito(
                        "porcentaje_velocidad_baja"
                    )
                ),

                "estancamiento_maximo_promedio_s": (
                    promedio_finito(
                        "duracion_mayor_estancamiento_s"
                    )
                ),

                "progreso_final_promedio_pct": (
                    promedio_finito(
                        "progreso_final_astar_pct"
                    )
                ),

                "distancia_final_promedio_m": (
                    promedio_finito(
                        "distancia_final_meta_m"
                    )
                ),
            }
        )

    ruta = Path(
        ruta_csv
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ruta.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(
                resumenes[
                    0
                ].keys()
            ),
        )

        escritor.writeheader()

        escritor.writerows(
            resumenes
        )

    return (
        resumenes,
        str(
            ruta
        ),
    )
# ==========================================================
# GRÁFICA RESUMEN DE LOS FALLOS
# ==========================================================
def graficar_resumen_diagnostico_fallos_sac(
    diagnosticos,
    ruta_guardado,
):

    semillas = np.asarray(
        [
            diagnostico[
                "semilla"
            ]
            for diagnostico in diagnosticos
        ],
        dtype=int,
    )

    clearances = np.asarray(
        [
            diagnostico[
                "clearance_dinamico_minimo_m"
            ]
            for diagnostico in diagnosticos
        ],
        dtype=float,
    )

    velocidad_baja = np.asarray(
        [
            diagnostico[
                "porcentaje_velocidad_baja"
            ]
            for diagnostico in diagnosticos
        ],
        dtype=float,
    )

    progreso = np.asarray(
        [
            diagnostico[
                "progreso_final_astar_pct"
            ]
            for diagnostico in diagnosticos
        ],
        dtype=float,
    )

    tipos = [
        diagnostico[
            "resultado"
        ]
        for diagnostico in diagnosticos
    ]

    tipos_unicos = [
        "colision_dinamica",
        "timeout",
        "fuera_mapa",
        "colision_estatica",
    ]

    conteos = [
        sum(
            tipo == tipo_objetivo
            for tipo in tipos
        )
        for tipo_objetivo in tipos_unicos
    ]

    etiquetas = [
        "Colisión\ndinámica",
        "Timeout",
        "Fuera del\nmapa",
        "Colisión\nestática",
    ]

    figura, ejes = plt.subplots(
        3,
        1,
        figsize=(
            13,
            14,
        ),
    )

    # ------------------------------------------------------
    # 1. Conteos
    # ------------------------------------------------------

    barras = ejes[
        0
    ].bar(
        etiquetas,
        conteos,
    )

    for barra, valor in zip(
        barras,
        conteos,
    ):

        ejes[
            0
        ].text(
            barra.get_x()
            + barra.get_width()
            / 2.0,

            barra.get_height()
            + 0.15,

            str(
                valor
            ),

            horizontalalignment="center",
        )

    ejes[
        0
    ].set_ylabel(
        "Número de casos"
    )

    ejes[
        0
    ].set_title(
        "Distribución de las semillas fallidas"
    )

    ejes[
        0
    ].grid(
        True,
        axis="y",
        alpha=0.3,
    )

    # ------------------------------------------------------
    # 2. Clearance mínimo
    # ------------------------------------------------------

    ejes[
        1
    ].plot(
        semillas,
        clearances,
        marker="o",
        linewidth=1.5,
        label="Clearance dinámico mínimo",
    )

    ejes[
        1
    ].axhline(
        0.0,
        linestyle="--",
        label="Colisión",
    )

    ejes[
        1
    ].set_ylabel(
        "Clearance [m]"
    )

    ejes[
        1
    ].set_title(
        "Mayor aproximación dinámica por semilla"
    )

    ejes[
        1
    ].grid(
        True,
        alpha=0.3,
    )

    ejes[
        1
    ].legend()

    # ------------------------------------------------------
    # 3. Velocidad baja y progreso
    # ------------------------------------------------------

    ejes[
        2
    ].plot(
        semillas,
        velocidad_baja,
        marker="s",
        linewidth=1.5,
        label="Tiempo a velocidad baja",
    )

    ejes[
        2
    ].plot(
        semillas,
        progreso,
        marker="^",
        linewidth=1.5,
        label="Progreso final sobre A*",
    )

    ejes[
        2
    ].set_xlabel(
        "Semilla"
    )

    ejes[
        2
    ].set_ylabel(
        "Porcentaje [%]"
    )

    ejes[
        2
    ].set_ylim(
        0.0,
        105.0,
    )

    ejes[
        2
    ].set_title(
        "Relación entre estancamiento y progreso"
    )

    ejes[
        2
    ].grid(
        True,
        alpha=0.3,
    )

    ejes[
        2
    ].legend()

    figura.tight_layout()

    ruta = Path(
        ruta_guardado
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figura.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight",
    )

    return {
        "figura": figura,

        "ruta": str(
            ruta
        ),
    }
# ==========================================================
# APLICAR LÍMITES FÍSICOS AL CONTROL SAC REACTIVO
# ==========================================================

def aplicar_dinamica_control_sac_reactivo(
    velocidad_lineal_deseada,
    velocidad_angular_deseada,
    velocidad_lineal_actual,
    velocidad_angular_actual,
    distancia_meta,
    dt=DT,
):

    # ------------------------------------------------------
    # 1. Convertir y validar
    # ------------------------------------------------------

    velocidad_lineal_deseada = float(
        velocidad_lineal_deseada
    )

    velocidad_angular_deseada = float(
        velocidad_angular_deseada
    )

    velocidad_lineal_actual = float(
        velocidad_lineal_actual
    )

    velocidad_angular_actual = float(
        velocidad_angular_actual
    )

    distancia_meta = float(
        distancia_meta
    )

    dt = float(
        dt
    )

    valores = [
        velocidad_lineal_deseada,
        velocidad_angular_deseada,
        velocidad_lineal_actual,
        velocidad_angular_actual,
        distancia_meta,
        dt,
    ]

    if not np.all(
        np.isfinite(
            valores
        )
    ):

        raise ValueError(
            "El control reactivo contiene valores "
            "no finitos."
        )

    if dt <= 0.0:

        raise ValueError(
            "El periodo de muestreo debe ser positivo."
        )

    # ------------------------------------------------------
    # 2. Limitar controles deseados globalmente
    # ------------------------------------------------------

    velocidad_lineal_deseada = float(
        np.clip(
            velocidad_lineal_deseada,
            0.0,
            VELOCIDAD_MAXIMA,
        )
    )

    velocidad_angular_deseada = float(
        np.clip(
            velocidad_angular_deseada,
            -VELOCIDAD_ANGULAR_MAXIMA,
            VELOCIDAD_ANGULAR_MAXIMA,
        )
    )

    velocidad_lineal_actual = float(
        np.clip(
            velocidad_lineal_actual,
            0.0,
            VELOCIDAD_MAXIMA,
        )
    )

    velocidad_angular_actual = float(
        np.clip(
            velocidad_angular_actual,
            -VELOCIDAD_ANGULAR_MAXIMA,
            VELOCIDAD_ANGULAR_MAXIMA,
        )
    )

    velocidad_lineal_original = (
        velocidad_lineal_deseada
    )

    velocidad_angular_original = (
        velocidad_angular_deseada
    )

    # ------------------------------------------------------
    # 3. Aplicar frenado terminal a la orden deseada
    # ------------------------------------------------------

    informacion_meta = (
        calcular_limite_velocidad_meta_sac_reactivo(
            distancia_meta=distancia_meta
        )
    )

    limite_velocidad_meta = float(
        informacion_meta[
            "limite_velocidad_meta"
        ]
    )

    velocidad_lineal_deseada_meta = min(
        velocidad_lineal_deseada,
        limite_velocidad_meta,
    )

    # ------------------------------------------------------
    # 4. Límites de cambio por periodo
    # ------------------------------------------------------

    cambio_lineal_maximo = (
        ACELERACION_LINEAL_MAXIMA_SAC_REACTIVO
        * dt
    )

    cambio_angular_maximo = (
        ACELERACION_ANGULAR_MAXIMA_SAC_REACTIVO
        * dt
    )

    velocidad_lineal_minima_fisica = max(
        0.0,

        velocidad_lineal_actual
        - cambio_lineal_maximo,
    )

    velocidad_lineal_maxima_fisica = min(
        VELOCIDAD_MAXIMA,

        velocidad_lineal_actual
        + cambio_lineal_maximo,
    )

    velocidad_angular_minima_fisica = max(
        -VELOCIDAD_ANGULAR_MAXIMA,

        velocidad_angular_actual
        - cambio_angular_maximo,
    )

    velocidad_angular_maxima_fisica = min(
        VELOCIDAD_ANGULAR_MAXIMA,

        velocidad_angular_actual
        + cambio_angular_maximo,
    )

    # ------------------------------------------------------
    # 5. Aplicar dinámica física
    # ------------------------------------------------------

    velocidad_lineal_aplicada = float(
        np.clip(
            velocidad_lineal_deseada_meta,

            velocidad_lineal_minima_fisica,
            velocidad_lineal_maxima_fisica,
        )
    )

    velocidad_angular_aplicada = float(
        np.clip(
            velocidad_angular_deseada,

            velocidad_angular_minima_fisica,
            velocidad_angular_maxima_fisica,
        )
    )

    # ------------------------------------------------------
    # 6. Diagnóstico de limitaciones
    # ------------------------------------------------------

    tolerancia = 1e-9

    frenado_meta_modifico_control = (
        velocidad_lineal_deseada_meta
        < velocidad_lineal_original
        - tolerancia
    )

    aceleracion_lineal_limitada = (
        abs(
            velocidad_lineal_aplicada
            - velocidad_lineal_deseada_meta
        )
        > tolerancia
    )

    aceleracion_angular_limitada = (
        abs(
            velocidad_angular_aplicada
            - velocidad_angular_original
        )
        > tolerancia
    )

    cambio_lineal_aplicado = (
        velocidad_lineal_aplicada
        - velocidad_lineal_actual
    )

    cambio_angular_aplicado = (
        velocidad_angular_aplicada
        - velocidad_angular_actual
    )

    # ------------------------------------------------------
    # 7. Validación final
    # ------------------------------------------------------

    if (
        abs(
            cambio_lineal_aplicado
        )
        > cambio_lineal_maximo
        + 1e-8
    ):

        raise RuntimeError(
            "El cambio lineal aplicado supera el límite."
        )

    if (
        abs(
            cambio_angular_aplicado
        )
        > cambio_angular_maximo
        + 1e-8
    ):

        raise RuntimeError(
            "El cambio angular aplicado supera el límite."
        )

    informacion = {
        "distancia_meta": distancia_meta,

        "velocidad_lineal_deseada_original": (
            velocidad_lineal_original
        ),

        "velocidad_angular_deseada_original": (
            velocidad_angular_original
        ),

        "velocidad_lineal_deseada_meta": (
            velocidad_lineal_deseada_meta
        ),

        "limite_velocidad_meta": (
            limite_velocidad_meta
        ),

        "factor_frenado_meta": (
            informacion_meta[
                "factor_frenado_meta"
            ]
        ),

        "frenado_terminal_activo": (
            informacion_meta[
                "frenado_terminal_activo"
            ]
        ),

        "frenado_meta_modifico_control": (
            frenado_meta_modifico_control
        ),

        "robot_dentro_meta": (
            informacion_meta[
                "robot_dentro_meta"
            ]
        ),

        "velocidad_lineal_minima_fisica": (
            velocidad_lineal_minima_fisica
        ),

        "velocidad_lineal_maxima_fisica": (
            velocidad_lineal_maxima_fisica
        ),

        "velocidad_angular_minima_fisica": (
            velocidad_angular_minima_fisica
        ),

        "velocidad_angular_maxima_fisica": (
            velocidad_angular_maxima_fisica
        ),

        "cambio_lineal_maximo": (
            cambio_lineal_maximo
        ),

        "cambio_angular_maximo": (
            cambio_angular_maximo
        ),

        "cambio_lineal_aplicado": (
            cambio_lineal_aplicado
        ),

        "cambio_angular_aplicado": (
            cambio_angular_aplicado
        ),

        "aceleracion_lineal_limitada": (
            aceleracion_lineal_limitada
        ),

        "aceleracion_angular_limitada": (
            aceleracion_angular_limitada
        ),

        "velocidad_lineal_aplicada": (
            velocidad_lineal_aplicada
        ),

        "velocidad_angular_aplicada": (
            velocidad_angular_aplicada
        ),
    }

    return (
        velocidad_lineal_aplicada,
        velocidad_angular_aplicada,
        informacion,
    )
# ==========================================================
# ACTUALIZAR ESTADO DE ESTANCAMIENTO DEL SAC REACTIVO
# ==========================================================
def actualizar_estado_estancamiento_sac_reactivo(
    distancia_meta_anterior,
    distancia_meta_nueva,
    mejor_distancia_meta_anterior,
    pasos_sin_progreso_anterior,
):

    # ------------------------------------------------------
    # 1. Convertir y validar
    # ------------------------------------------------------

    distancia_meta_anterior = float(
        distancia_meta_anterior
    )

    distancia_meta_nueva = float(
        distancia_meta_nueva
    )

    mejor_distancia_meta_anterior = float(
        mejor_distancia_meta_anterior
    )

    pasos_sin_progreso_anterior = int(
        pasos_sin_progreso_anterior
    )

    valores = [
        distancia_meta_anterior,
        distancia_meta_nueva,
        mejor_distancia_meta_anterior,
    ]

    if not np.all(
        np.isfinite(
            valores
        )
    ):

        raise ValueError(
            "Las distancias del estado de estancamiento "
            "deben ser finitas."
        )

    if any(
        valor < 0.0
        for valor in valores
    ):

        raise ValueError(
            "Las distancias no pueden ser negativas."
        )

    if pasos_sin_progreso_anterior < 0:

        raise ValueError(
            "El contador de estancamiento no puede ser "
            "negativo."
        )

    # ------------------------------------------------------
    # 2. Progreso instantáneo hacia la meta
    # ------------------------------------------------------

    progreso_meta_global = (
        distancia_meta_anterior
        - distancia_meta_nueva
    )

    progreso_instantaneo_suficiente = (
        progreso_meta_global
        >= PROGRESO_MINIMO_META_POR_PASO_SAC_REACTIVO
    )

    # ------------------------------------------------------
    # 3. Detectar un nuevo mejor acercamiento
    # ------------------------------------------------------

    mejora_respecto_mejor = (
        mejor_distancia_meta_anterior
        - distancia_meta_nueva
    )

    nuevo_mejor_significativo = (
        mejora_respecto_mejor
        >= MEJORA_MINIMA_MEJOR_DISTANCIA_SAC_REACTIVO
    )

    progreso_suficiente = (
        progreso_instantaneo_suficiente
        or nuevo_mejor_significativo
    )

    # ------------------------------------------------------
    # 4. Determinar si se recuperó de un bloqueo
    # ------------------------------------------------------

    habia_estancamiento_persistente = (
        pasos_sin_progreso_anterior
        >= PASOS_INICIO_PENALIZACION_ESTANCAMIENTO_SAC_REACTIVO
    )

    recuperacion_estancamiento = (
        habia_estancamiento_persistente
        and progreso_suficiente
    )

    # ------------------------------------------------------
    # 5. Actualizar contador
    # ------------------------------------------------------

    if progreso_suficiente:

        pasos_sin_progreso_nuevos = 0

    else:

        pasos_sin_progreso_nuevos = (
            pasos_sin_progreso_anterior
            + 1
        )

    # ------------------------------------------------------
    # 6. Actualizar mejor distancia
    # ------------------------------------------------------
    #
    # Se conserva el mínimo absoluto observado, aunque la
    # mejora de un solo paso sea menor que el umbral.

    mejor_distancia_meta_nueva = min(
        mejor_distancia_meta_anterior,
        distancia_meta_nueva,
    )

    # ------------------------------------------------------
    # 7. Calcular intensidad del estancamiento
    # ------------------------------------------------------

    inicio_penalizacion = (
        PASOS_INICIO_PENALIZACION_ESTANCAMIENTO_SAC_REACTIVO
    )

    pasos_penalizacion_maxima = (
        PASOS_PENALIZACION_MAXIMA_ESTANCAMIENTO_SAC_REACTIVO
    )

    if pasos_sin_progreso_nuevos <= inicio_penalizacion:

        intensidad_estancamiento = 0.0

    else:

        pasos_excedentes = (
            pasos_sin_progreso_nuevos
            - inicio_penalizacion
        )

        intervalo_crecimiento = max(
            pasos_penalizacion_maxima
            - inicio_penalizacion,
            1,
        )

        intensidad_estancamiento = float(
            np.clip(
                pasos_excedentes
                / intervalo_crecimiento,

                0.0,
                1.0,
            )
        )

    # Crecimiento cuadrático:
    #
    # al principio es moderado;
    # después aumenta con rapidez.

    intensidad_estancamiento_cuadratica = (
        intensidad_estancamiento ** 2
    )

    penalizacion_estancamiento_persistente = (
        PENALIZACION_ESTANCAMIENTO_PERSISTENTE_MAXIMA_SAC
        * intensidad_estancamiento_cuadratica
    )

    if recuperacion_estancamiento:

        recompensa_recuperacion = (
            RECOMPENSA_RECUPERACION_ESTANCAMIENTO_SAC
        )

    else:

        recompensa_recuperacion = 0.0

    return {
        "distancia_meta_anterior": (
            distancia_meta_anterior
        ),

        "distancia_meta_nueva": (
            distancia_meta_nueva
        ),

        "progreso_meta_global": (
            progreso_meta_global
        ),

        "progreso_instantaneo_suficiente": (
            progreso_instantaneo_suficiente
        ),

        "nuevo_mejor_significativo": (
            nuevo_mejor_significativo
        ),

        "progreso_suficiente": (
            progreso_suficiente
        ),

        "mejor_distancia_meta_anterior": (
            mejor_distancia_meta_anterior
        ),

        "mejor_distancia_meta_nueva": (
            mejor_distancia_meta_nueva
        ),

        "pasos_sin_progreso_anterior": (
            pasos_sin_progreso_anterior
        ),

        "pasos_sin_progreso_nuevos": (
            pasos_sin_progreso_nuevos
        ),

        "intensidad_estancamiento": (
            intensidad_estancamiento
        ),

        "penalizacion_estancamiento_persistente": (
            penalizacion_estancamiento_persistente
        ),

        "recuperacion_estancamiento": (
            recuperacion_estancamiento
        ),

        "recompensa_recuperacion_estancamiento": (
            recompensa_recuperacion
        ),
    }
def verificar_sac_reactivo_mejorado():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VERIFICACIÓN COMPLETA DEL SAC REACTIVO MEJORADO"
    )

    print(
        "=" * 80
    )

    # ======================================================
    # 1. DINÁMICA FÍSICA DESDE REPOSO
    # ======================================================

    (
        velocidad_reposo,
        omega_reposo,
        informacion_reposo,
    ) = aplicar_dinamica_control_sac_reactivo(
        velocidad_lineal_deseada=(
            VELOCIDAD_MAXIMA
        ),
        velocidad_angular_deseada=(
            VELOCIDAD_ANGULAR_MAXIMA
        ),
        velocidad_lineal_actual=0.0,
        velocidad_angular_actual=0.0,
        distancia_meta=5.0,
        dt=DT,
    )

    velocidad_reposo_esperada = (
        ACELERACION_LINEAL_MAXIMA_SAC_REACTIVO
        * DT
    )

    omega_reposo_esperada = (
        ACELERACION_ANGULAR_MAXIMA_SAC_REACTIVO
        * DT
    )

    aceleracion_desde_reposo_correcta = (
        np.isclose(
            velocidad_reposo,
            velocidad_reposo_esperada,
        )
        and np.isclose(
            omega_reposo,
            omega_reposo_esperada,
        )
    )

    # ======================================================
    # 2. INVERSIÓN ANGULAR LIMITADA
    # ======================================================

    (
        _,
        omega_inversion,
        informacion_inversion,
    ) = aplicar_dinamica_control_sac_reactivo(
        velocidad_lineal_deseada=0.5,
        velocidad_angular_deseada=(
            -VELOCIDAD_ANGULAR_MAXIMA
        ),
        velocidad_lineal_actual=0.5,
        velocidad_angular_actual=(
            VELOCIDAD_ANGULAR_MAXIMA
        ),
        distancia_meta=5.0,
        dt=DT,
    )

    omega_inversion_esperada = (
        VELOCIDAD_ANGULAR_MAXIMA
        - ACELERACION_ANGULAR_MAXIMA_SAC_REACTIVO
        * DT
    )

    inversion_angular_correcta = np.isclose(
        omega_inversion,
        omega_inversion_esperada,
    )

    # ======================================================
    # 3. FRENADO TERMINAL
    # ======================================================

    distancia_meta_cercana = 0.60

    limite_meta = (
        calcular_limite_velocidad_meta_sac_reactivo(
            distancia_meta=distancia_meta_cercana
        )
    )

    (
        velocidad_cerca_meta,
        _,
        informacion_cerca_meta,
    ) = aplicar_dinamica_control_sac_reactivo(
        velocidad_lineal_deseada=(
            VELOCIDAD_MAXIMA
        ),
        velocidad_angular_deseada=0.0,
        velocidad_lineal_actual=0.0,
        velocidad_angular_actual=0.0,
        distancia_meta=distancia_meta_cercana,
        dt=DT,
    )

    frenado_terminal_correcto = (
        limite_meta[
            "frenado_terminal_activo"
        ]
        and limite_meta[
            "limite_velocidad_meta"
        ]
        < VELOCIDAD_MAXIMA
        and velocidad_cerca_meta
        <= limite_meta[
            "limite_velocidad_meta"
        ]
        + 1e-8
        and informacion_cerca_meta[
            "frenado_meta_modifico_control"
        ]
    )

    # ======================================================
    # 4. PROGRESO GLOBAL NORMAL
    # ======================================================

    estado_progreso = (
        actualizar_estado_estancamiento_sac_reactivo(
            distancia_meta_anterior=5.00,
            distancia_meta_nueva=4.98,
            mejor_distancia_meta_anterior=5.00,
            pasos_sin_progreso_anterior=10,
        )
    )

    progreso_reinicia_contador = (
        estado_progreso[
            "progreso_suficiente"
        ]
        and estado_progreso[
            "pasos_sin_progreso_nuevos"
        ]
        == 0
        and np.isclose(
            estado_progreso[
                "progreso_meta_global"
            ],
            0.02,
        )
    )

    # ======================================================
    # 5. ESTANCAMIENTO PERSISTENTE
    # ======================================================

    pasos_previos_bloqueo = (
        PASOS_INICIO_PENALIZACION_ESTANCAMIENTO_SAC_REACTIVO
        + 20
    )

    estado_sin_progreso = (
        actualizar_estado_estancamiento_sac_reactivo(
            distancia_meta_anterior=5.00,
            distancia_meta_nueva=5.00,
            mejor_distancia_meta_anterior=4.90,
            pasos_sin_progreso_anterior=(
                pasos_previos_bloqueo
            ),
        )
    )

    estancamiento_incrementa = (
        not estado_sin_progreso[
            "progreso_suficiente"
        ]
        and estado_sin_progreso[
            "pasos_sin_progreso_nuevos"
        ]
        == pasos_previos_bloqueo + 1
        and estado_sin_progreso[
            "penalizacion_estancamiento_persistente"
        ]
        > 0.0
    )

    # ======================================================
    # 6. RECUPERACIÓN DESPUÉS DEL ESTANCAMIENTO
    # ======================================================

    estado_recuperacion = (
        actualizar_estado_estancamiento_sac_reactivo(
            distancia_meta_anterior=5.00,
            distancia_meta_nueva=4.90,
            mejor_distancia_meta_anterior=4.95,
            pasos_sin_progreso_anterior=60,
        )
    )

    recuperacion_detectada = (
        estado_recuperacion[
            "recuperacion_estancamiento"
        ]
        and estado_recuperacion[
            "pasos_sin_progreso_nuevos"
        ]
        == 0
        and np.isclose(
            estado_recuperacion[
                "recompensa_recuperacion_estancamiento"
            ],
            RECOMPENSA_RECUPERACION_ESTANCAMIENTO_SAC,
        )
    )

    # ======================================================
    # 7. RECOMPENSA POR PROGRESO HACIA LA META GLOBAL
    # ======================================================

    estado_anterior = (
        1.0,
        1.0,
        0.0,
    )

    estado_nuevo = (
        1.1,
        1.0,
        0.0,
    )

    submeta = (
        3.0,
        1.0,
    )

    meta = (
        5.0,
        1.0,
    )

    camino_mundo = [
        (
            1.0,
            1.0,
        ),
        (
            5.0,
            1.0,
        ),
    ]

    (
        recompensa_acercamiento,
        componentes_acercamiento,
    ) = calcular_recompensa_sac(
        estado_anterior=estado_anterior,
        estado_nuevo=estado_nuevo,
        submeta=submeta,
        meta=meta,
        camino_mundo=camino_mundo,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=2.0,
        clearance_estatico=2.0,
        clearance_dinamico=2.0,
        resultado="en_progreso",
        distancia_meta_anterior=4.0,
        pasos_sin_progreso=0,
        recuperacion_estancamiento=False,
        dt=DT,
    )

    (
        recompensa_alejamiento,
        componentes_alejamiento,
    ) = calcular_recompensa_sac(
        estado_anterior=estado_anterior,
        estado_nuevo=estado_nuevo,
        submeta=submeta,
        meta=meta,
        camino_mundo=camino_mundo,
        velocidad_lineal_anterior=0.5,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.5,
        velocidad_angular_nueva=0.0,
        clearance_total=2.0,
        clearance_estatico=2.0,
        clearance_dinamico=2.0,
        resultado="en_progreso",
        distancia_meta_anterior=3.8,
        pasos_sin_progreso=0,
        recuperacion_estancamiento=False,
        dt=DT,
    )

    progreso_global_premiado = (
        componentes_acercamiento[
            "recompensa_progreso_meta_global"
        ]
        > 0.0
        and componentes_alejamiento[
            "recompensa_progreso_meta_global"
        ]
        < 0.0
        and recompensa_acercamiento
        > recompensa_alejamiento
    )

    # ======================================================
    # 8. PENALIZACIÓN PERSISTENTE EN LA RECOMPENSA
    # ======================================================

    (
        recompensa_bloqueo,
        componentes_bloqueo,
    ) = calcular_recompensa_sac(
        estado_anterior=estado_anterior,
        estado_nuevo=estado_anterior,
        submeta=submeta,
        meta=meta,
        camino_mundo=camino_mundo,
        velocidad_lineal_anterior=0.0,
        velocidad_angular_anterior=0.0,
        velocidad_lineal_nueva=0.0,
        velocidad_angular_nueva=0.0,
        clearance_total=2.0,
        clearance_estatico=2.0,
        clearance_dinamico=2.0,
        resultado="en_progreso",
        distancia_meta_anterior=4.0,
        pasos_sin_progreso=80,
        recuperacion_estancamiento=False,
        dt=DT,
    )

    penalizacion_persistente_aplicada = (
        componentes_bloqueo[
            "penalizacion_estancamiento_persistente"
        ]
        > 0.0
        and componentes_bloqueo[
            "intensidad_estancamiento_persistente"
        ]
        > 0.0
    )

    # ======================================================
    # 9. INTEGRACIÓN EN UN ENTORNO REAL
    # ======================================================

    (
        entorno,
        observacion,
    ) = reiniciar_entorno_sac(
        semilla=SEMILLA
    )

    claves_entorno_nuevas = {
        "distancia_meta_anterior",
        "mejor_distancia_meta",
        "pasos_sin_progreso",
        "numero_recuperaciones_estancamiento",
    }

    claves_registro_nuevas = {
        "distancias_meta_globales",
        "progresos_meta_globales",
        "mejores_distancias_meta",
        "pasos_sin_progreso",
        "intensidades_estancamiento",
        "penalizaciones_estancamiento_persistente",
        "recuperaciones_estancamiento",
    }

    estado_inicial_integrado = (
        claves_entorno_nuevas.issubset(
            entorno.keys()
        )
        and claves_registro_nuevas.issubset(
            entorno[
                "registro"
            ].keys()
        )
    )

    accion_prueba = np.array(
        [
            -1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    (
        observacion_nueva,
        recompensa_paso,
        terminado,
        truncado,
        informacion_paso,
    ) = ejecutar_paso_entorno_sac(
        entorno=entorno,
        accion=accion_prueba,
        pasos_maximos=20,
        dt=DT,
    )

    claves_informacion_nuevas = {
        "velocidad_lineal_deseada",
        "velocidad_angular_deseada",
        "control_reactivo",
        "distancia_meta_anterior",
        "distancia_meta_nueva",
        "progreso_meta_global",
        "mejor_distancia_meta",
        "pasos_sin_progreso",
        "intensidad_estancamiento",
        "recuperacion_estancamiento",
        "penalizacion_estancamiento_persistente",
    }

    informacion_integrada = (
        claves_informacion_nuevas.issubset(
            informacion_paso.keys()
        )
    )

    registro_actualizado = all(
        len(
            entorno[
                "registro"
            ][
                clave
            ]
        )
        == 1
        for clave in claves_registro_nuevas
    )

    entorno_actualizado = (
        np.isclose(
            entorno[
                "distancia_meta_anterior"
            ],
            informacion_paso[
                "distancia_meta_nueva"
            ],
        )
        and entorno[
            "pasos_sin_progreso"
        ]
        == informacion_paso[
            "pasos_sin_progreso"
        ]
    )

    canal_dinamico = observacion_nueva[
        "parche"
    ][
        1
    ]

    observacion_continua_reactiva = np.all(
        np.isin(
            np.unique(
                canal_dinamico
            ),
            [
                0.0,
                1.0,
            ],
        )
    )

    informacion_sin_prediccion = not any(
        (
            "ttc"
            in str(
                clave
            ).lower()
            or "predich"
            in str(
                clave
            ).lower()
        )
        for clave in informacion_paso.keys()
    )

    paso_real_correcto = (
        estado_inicial_integrado
        and informacion_integrada
        and registro_actualizado
        and entorno_actualizado
        and observacion_continua_reactiva
        and informacion_sin_prediccion
        and np.isfinite(
            recompensa_paso
        )
        and isinstance(
            terminado,
            bool,
        )
        and isinstance(
            truncado,
            bool,
        )
    )

    # ======================================================
    # 10. RESULTADO GENERAL
    # ======================================================

    todo_correcto = (
        aceleracion_desde_reposo_correcta
        and inversion_angular_correcta
        and frenado_terminal_correcto
        and progreso_reinicia_contador
        and estancamiento_incrementa
        and recuperacion_detectada
        and progreso_global_premiado
        and penalizacion_persistente_aplicada
        and paso_real_correcto
    )

    print(
        "\nDINÁMICA FÍSICA"
    )

    print(
        "Velocidad desde reposo:",
        velocidad_reposo,
    )

    print(
        "Omega desde reposo:",
        omega_reposo,
    )

    print(
        "Omega durante inversión:",
        omega_inversion,
    )

    print(
        "Límite cerca de la meta:",
        limite_meta[
            "limite_velocidad_meta"
        ],
    )

    print(
        "Velocidad aplicada cerca de la meta:",
        velocidad_cerca_meta,
    )

    print(
        "\nPROGRESO Y ESTANCAMIENTO"
    )

    print(
        "Progreso global normal:",
        estado_progreso[
            "progreso_meta_global"
        ],
    )

    print(
        "Pasos sin progreso:",
        estado_sin_progreso[
            "pasos_sin_progreso_nuevos"
        ],
    )

    print(
        "Penalización persistente:",
        estado_sin_progreso[
            "penalizacion_estancamiento_persistente"
        ],
    )

    print(
        "Recuperación detectada:",
        estado_recuperacion[
            "recuperacion_estancamiento"
        ],
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Aceleración desde reposo correcta:",
        aceleracion_desde_reposo_correcta,
    )

    print(
        "Inversión angular correcta:",
        inversion_angular_correcta,
    )

    print(
        "Frenado terminal correcto:",
        frenado_terminal_correcto,
    )

    print(
        "Progreso reinicia contador:",
        progreso_reinicia_contador,
    )

    print(
        "Estancamiento incrementa contador:",
        estancamiento_incrementa,
    )

    print(
        "Recuperación detectada correctamente:",
        recuperacion_detectada,
    )

    print(
        "Progreso global correctamente premiado:",
        progreso_global_premiado,
    )

    print(
        "Penalización persistente aplicada:",
        penalizacion_persistente_aplicada,
    )

    print(
        "Estado y registro iniciales integrados:",
        estado_inicial_integrado,
    )

    print(
        "Información del paso integrada:",
        informacion_integrada,
    )

    print(
        "Registro actualizado:",
        registro_actualizado,
    )

    print(
        "Entorno actualizado:",
        entorno_actualizado,
    )

    print(
        "Observación continúa reactiva:",
        observacion_continua_reactiva,
    )

    print(
        "Información sin predicción:",
        informacion_sin_prediccion,
    )

    print(
        "Paso real correcto:",
        paso_real_correcto,
    )

    if todo_correcto:

        print(
            "\nRESULTADO DEL MAIN: TODO CORRECTO"
        )

    else:

        print(
            "\nRESULTADO DEL MAIN: HAY ALGO POR CORREGIR"
        )

    print(
        "=" * 80
    )
# ==========================================================
# AGREGAR METADATOS AL CHECKPOINT FINAL REACTIVO MEJORADO
# ==========================================================
def agregar_metadatos_checkpoint_final_reactivo_mejorado(
    ruta_checkpoint,
    resultado_entrenamiento,
):

    ruta = Path(
        ruta_checkpoint
    )

    if not ruta.is_file():

        raise FileNotFoundError(
            "No se encontró el checkpoint final que se "
            "desea etiquetar."
        )

    checkpoint = torch.load(
        ruta,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint[
        "version_checkpoint"
    ] = 2

    checkpoint[
        "tipo"
    ] = "sac_reactivo_mejorado_entrenamiento_completo"

    checkpoint[
        "variante_sac"
    ] = "reactivo_mejorado"

    checkpoint[
        "observacion_predictiva"
    ] = False

    checkpoint[
        "usa_ttc"
    ] = False

    checkpoint[
        "usa_limites_aceleracion"
    ] = True

    checkpoint[
        "usa_frenado_terminal"
    ] = True

    checkpoint[
        "usa_estancamiento_persistente"
    ] = True

    checkpoint[
        "historial_validaciones"
    ] = resultado_entrenamiento[
        "historial_validaciones"
    ]

    checkpoint[
        "mejor_episodio"
    ] = int(
        resultado_entrenamiento[
            "mejor_episodio"
        ]
    )

    checkpoint[
        "mejor_criterio"
    ] = tuple(
        float(
            valor
        )
        for valor in resultado_entrenamiento[
            "mejor_criterio"
        ]
    )

    checkpoint[
        "mejor_resumen_validacion"
    ] = resultado_entrenamiento[
        "mejor_resumen_validacion"
    ].copy()

    checkpoint[
        "configuracion_reactivo_mejorado"
    ] = {
        "episodios": int(
            NUMERO_EPISODIOS_TERCERA_CORRIDA_SAC
        ),

        "semilla_entrenamiento": int(
            SEMILLA_BASE_TERCERA_CORRIDA_SAC
        ),

        "semilla_validacion": int(
            SEMILLA_BASE_VALIDACION_TERCERA_CORRIDA_SAC
        ),

        "numero_semillas_validacion": int(
            NUMERO_SEMILLAS_VALIDACION_TERCERA_CORRIDA_SAC
        ),

        "episodios_entre_validaciones": int(
            EPISODIOS_ENTRE_VALIDACIONES_TERCERA_CORRIDA_SAC
        ),

        "tasa_actor": float(
            TASA_APRENDIZAJE_ACTOR_TERCERA_CORRIDA_SAC
        ),

        "tasa_criticos": float(
            TASA_APRENDIZAJE_CRITICOS_TERCERA_CORRIDA_SAC
        ),

        "tasa_alpha": float(
            TASA_APRENDIZAJE_ALPHA_TERCERA_CORRIDA_SAC
        ),
    }

    ruta_temporal = ruta.with_suffix(
        ruta.suffix
        + ".metadatos.temporal"
    )

    torch.save(
        checkpoint,
        ruta_temporal,
    )

    ruta_temporal.replace(
        ruta
    )

    return {
        "ruta": str(
            ruta
        ),

        "tamano_bytes": int(
            ruta.stat().st_size
        ),

        "tipo": checkpoint[
            "tipo"
        ],
    }
# ==========================================================
# TERCERA CORRIDA DEL SAC REACTIVO MEJORADO
# ==========================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TERCERA CORRIDA: ENTRENAMIENTO DEL SAC REACTIVO MEJORADO"
    )

    print(
        "=" * 80
    )

    plt.close(
        "all"
    )

    # ------------------------------------------------------
    # 1. Verificación previa del entorno mejorado
    # ------------------------------------------------------

    if EJECUTAR_VERIFICACION_PREVIA_TERCERA_CORRIDA_SAC:

        verificar_sac_reactivo_mejorado()

        plt.close(
            "all"
        )

    # ------------------------------------------------------
    # 2. Crear directorios
    # ------------------------------------------------------

    DIRECTORIO_TERCERA_CORRIDA_SAC.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIRECTORIO_GRAFICAS_TERCERA_CORRIDA_SAC.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # 3. Configurar semillas del nuevo entrenamiento
    # ------------------------------------------------------

    configurar_pytorch_sac(
        semilla=SEMILLA_BASE_TERCERA_CORRIDA_SAC
    )

    # ------------------------------------------------------
    # 4. Cargar solamente el actor reactivo anterior
    # ------------------------------------------------------

    if not RUTA_ACTOR_INICIAL_TERCERA_CORRIDA_SAC.is_file():

        raise FileNotFoundError(
            "No se encontró el actor inicial de la tercera "
            "corrida:\n"
            f"{RUTA_ACTOR_INICIAL_TERCERA_CORRIDA_SAC}"
        )

    checkpoint_inicial = torch.load(
        RUTA_ACTOR_INICIAL_TERCERA_CORRIDA_SAC,
        map_location="cpu",
        weights_only=False,
    )

    if "actor_state_dict" not in checkpoint_inicial:

        raise KeyError(
            "El checkpoint inicial no contiene los pesos "
            "del actor."
        )

    episodio_actor_inicial = checkpoint_inicial.get(
        "episodio",
        None,
    )

    actor = crear_actor_sac(
        dispositivo=DISPOSITIVO_SAC
    )

    resultado_carga_actor = actor.load_state_dict(
        checkpoint_inicial[
            "actor_state_dict"
        ],
        strict=True,
    )

    del checkpoint_inicial

    actor.train()

    carga_actor_correcta = (
        len(
            resultado_carga_actor.missing_keys
        )
        == 0
        and len(
            resultado_carga_actor.unexpected_keys
        )
        == 0
    )

    if not carga_actor_correcta:

        raise RuntimeError(
            "El actor inicial no se cargó correctamente."
        )

    # ------------------------------------------------------
    # 5. Reiniciar críticos, objetivos y temperatura
    # ------------------------------------------------------

    (
        critico_q1,
        critico_q2,
    ) = crear_dos_criticos_sac(
        dispositivo=DISPOSITIVO_SAC
    )

    (
        critico_q1_objetivo,
        critico_q2_objetivo,
    ) = crear_dos_criticos_objetivo_sac(
        critico_q1=critico_q1,
        critico_q2=critico_q2,
        dispositivo=DISPOSITIVO_SAC,
    )

    temperatura = crear_temperatura_entropia_sac(
        dispositivo=DISPOSITIVO_SAC,
        coeficiente_inicial=(
            COEFICIENTE_ENTROPIA_INICIAL_SAC
        ),
    )

    # ------------------------------------------------------
    # 6. Crear optimizadores nuevos
    # ------------------------------------------------------

    optimizador_actor = crear_optimizador_actor_sac(
        actor=actor,
        tasa_aprendizaje=(
            TASA_APRENDIZAJE_ACTOR_TERCERA_CORRIDA_SAC
        ),
    )

    (
        optimizador_q1,
        optimizador_q2,
    ) = crear_optimizadores_dos_criticos_sac(
        critico_q1=critico_q1,
        critico_q2=critico_q2,
        tasa_aprendizaje=(
            TASA_APRENDIZAJE_CRITICOS_TERCERA_CORRIDA_SAC
        ),
    )

    optimizador_alpha = crear_optimizador_temperatura_sac(
        temperatura=temperatura,
        tasa_aprendizaje=(
            TASA_APRENDIZAJE_ALPHA_TERCERA_CORRIDA_SAC
        ),
    )

    # ------------------------------------------------------
    # 7. Crear replay buffer completamente nuevo
    # ------------------------------------------------------

    buffer = crear_buffer_repeticion_sac(
        capacidad=CAPACIDAD_BUFFER_REPETICION_SAC,
        semilla=(
            SEMILLA_BASE_TERCERA_CORRIDA_SAC
            + 12345
        ),
    )

    # ------------------------------------------------------
    # 8. Mostrar configuración
    # ------------------------------------------------------

    print(
        "\n1. Actor inicial"
    )

    print(
        "Checkpoint:",
        RUTA_ACTOR_INICIAL_TERCERA_CORRIDA_SAC,
    )

    print(
        "Episodio del actor inicial:",
        episodio_actor_inicial,
    )

    print(
        "Dispositivo:",
        DISPOSITIVO_SAC,
    )

    print(
        "Número de parámetros del actor:",
        sum(
            parametro.numel()
            for parametro in actor.parameters()
        ),
    )

    print(
        "\n2. Entrenamiento"
    )

    print(
        "Episodios:",
        NUMERO_EPISODIOS_TERCERA_CORRIDA_SAC,
    )

    print(
        "Semillas de entrenamiento:",
        f"{SEMILLA_BASE_TERCERA_CORRIDA_SAC} a "
        f"{SEMILLA_BASE_TERCERA_CORRIDA_SAC + NUMERO_EPISODIOS_TERCERA_CORRIDA_SAC - 1}",
    )

    print(
        "Transiciones aleatorias iniciales:",
        TRANSICIONES_ALEATORIAS_TERCERA_CORRIDA_SAC,
    )

    print(
        "\n3. Validación periódica"
    )

    print(
        "Cada:",
        EPISODIOS_ENTRE_VALIDACIONES_TERCERA_CORRIDA_SAC,
        "episodios",
    )

    print(
        "Número de semillas:",
        NUMERO_SEMILLAS_VALIDACION_TERCERA_CORRIDA_SAC,
    )

    print(
        "Rango:",
        f"{SEMILLA_BASE_VALIDACION_TERCERA_CORRIDA_SAC} a "
        f"{SEMILLA_BASE_VALIDACION_TERCERA_CORRIDA_SAC + NUMERO_SEMILLAS_VALIDACION_TERCERA_CORRIDA_SAC - 1}",
    )

    print(
        "Criterio: éxito, timeout, colisión dinámica, "
        "colisión estática, salida del mapa, distancia "
        "final y clearance"
    )

    # ------------------------------------------------------
    # 9. Entrenamiento real con validación periódica
    # ------------------------------------------------------

    print(
        "\n4. Comienza el entrenamiento"
    )

    resultado_entrenamiento = (
        entrenar_agente_sac_con_validacion_periodica(
            numero_episodios=(
                NUMERO_EPISODIOS_TERCERA_CORRIDA_SAC
            ),

            episodios_entre_validaciones=(
                EPISODIOS_ENTRE_VALIDACIONES_TERCERA_CORRIDA_SAC
            ),

            semilla_entrenamiento=(
                SEMILLA_BASE_TERCERA_CORRIDA_SAC
            ),

            semilla_validacion=(
                SEMILLA_BASE_VALIDACION_TERCERA_CORRIDA_SAC
            ),

            numero_semillas_validacion=(
                NUMERO_SEMILLAS_VALIDACION_TERCERA_CORRIDA_SAC
            ),

            ruta_mejor_actor=(
                RUTA_MEJOR_ACTOR_TERCERA_CORRIDA_SAC
            ),

            actor=actor,
            critico_q1=critico_q1,
            critico_q2=critico_q2,
            critico_q1_objetivo=critico_q1_objetivo,
            critico_q2_objetivo=critico_q2_objetivo,
            temperatura=temperatura,
            optimizador_actor=optimizador_actor,
            optimizador_q1=optimizador_q1,
            optimizador_q2=optimizador_q2,
            optimizador_alpha=optimizador_alpha,
            buffer=buffer,
            pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
            tamano_lote=TAMANO_LOTE_SAC,

            transiciones_aleatorias_iniciales=(
                TRANSICIONES_ALEATORIAS_TERCERA_CORRIDA_SAC
            ),

            actualizaciones_por_paso=(
                ACTUALIZACIONES_POR_PASO_TERCERA_CORRIDA_SAC
            ),

            factor_descuento=FACTOR_DESCUENTO_SAC,
            tau=TAU_POLYAK_SAC,
            entropia_objetivo=ENTROPIA_OBJETIVO_SAC,

            ventana_promedio=(
                VENTANA_PROMEDIO_TERCERA_CORRIDA_SAC
            ),

            dt=DT,
            dispositivo=DISPOSITIVO_SAC,
        )
    )

    # ------------------------------------------------------
    # 10. Guardar checkpoint final completo
    # ------------------------------------------------------

    informacion_checkpoint_final = guardar_checkpoint_final_sac(
        ruta_checkpoint=(
            RUTA_CHECKPOINT_FINAL_TERCERA_CORRIDA_SAC
        ),
        resultado_entrenamiento=resultado_entrenamiento,
        optimizador_actor=optimizador_actor,
        optimizador_q1=optimizador_q1,
        optimizador_q2=optimizador_q2,
        optimizador_alpha=optimizador_alpha,
    )

    informacion_metadatos_final = (
        agregar_metadatos_checkpoint_final_reactivo_mejorado(
            ruta_checkpoint=(
                RUTA_CHECKPOINT_FINAL_TERCERA_CORRIDA_SAC
            ),
            resultado_entrenamiento=resultado_entrenamiento,
        )
    )

    # ------------------------------------------------------
    # 11. Guardar gráficas de entrenamiento
    # ------------------------------------------------------

    resultado_graficas_entrenamiento = graficar_entrenamiento_sac(
        resultado_entrenamiento=resultado_entrenamiento,
        mostrar=False,
        directorio_salida=(
            DIRECTORIO_GRAFICAS_TERCERA_CORRIDA_SAC
        ),
        dpi=300,
    )

    # ------------------------------------------------------
    # 12. Guardar historial de 50 semillas
    # ------------------------------------------------------

    resultado_historial_validaciones = (
        guardar_historial_validaciones_periodicas_sac(
            historial_validaciones=(
                resultado_entrenamiento[
                    "historial_validaciones"
                ]
            ),
            ruta_csv=(
                RUTA_CSV_VALIDACIONES_TERCERA_CORRIDA_SAC
            ),
            ruta_grafica=(
                RUTA_GRAFICA_VALIDACIONES_TERCERA_CORRIDA_SAC
            ),
            mejor_episodio=(
                resultado_entrenamiento[
                    "mejor_episodio"
                ]
            ),
            dpi=300,
        )
    )

    # ------------------------------------------------------
    # 13. Cargar y diagnosticar el mejor actor
    # ------------------------------------------------------

    checkpoint_mejor = torch.load(
        RUTA_MEJOR_ACTOR_TERCERA_CORRIDA_SAC,
        map_location="cpu",
        weights_only=False,
    )

    actor_mejor = crear_actor_sac(
        dispositivo=DISPOSITIVO_SAC
    )

    resultado_carga_mejor = actor_mejor.load_state_dict(
        checkpoint_mejor[
            "actor_state_dict"
        ],
        strict=True,
    )

    actor_mejor.eval()

    resultado_diagnostico = evaluar_episodio_sac(
        semilla=SEMILLA_DIAGNOSTICA_TERCERA_CORRIDA_SAC,
        actor=actor_mejor,
        pasos_maximos=PASOS_MAXIMOS_SEGUIMIENTO,
        dt=DT,
        dispositivo=DISPOSITIVO_SAC,
    )

    resultado_grafica_diagnostica = graficar_episodio_sac_espacial(
        resultado_evaluacion=resultado_diagnostico,
        mostrar=False,
        ruta_guardado=str(
            RUTA_GRAFICA_DIAGNOSTICA_TERCERA_CORRIDA_SAC
        ),
        dpi=300,
        anotar_submetas=False,
    )

    # ------------------------------------------------------
    # 14. Verificaciones técnicas
    # ------------------------------------------------------

    mejor_resumen = resultado_entrenamiento[
        "mejor_resumen_validacion"
    ]

    resumen_global = resultado_entrenamiento[
        "resumen_global"
    ]

    historial_validaciones = resultado_entrenamiento[
        "historial_validaciones"
    ]

    numero_validaciones_esperado = int(
        math.ceil(
            NUMERO_EPISODIOS_TERCERA_CORRIDA_SAC
            / EPISODIOS_ENTRE_VALIDACIONES_TERCERA_CORRIDA_SAC
        )
    )

    checkpoint_mejor_correcto = (
        RUTA_MEJOR_ACTOR_TERCERA_CORRIDA_SAC.is_file()
        and checkpoint_mejor.get(
            "tipo"
        )
        == "mejor_actor_sac_reactivo_mejorado"
        and checkpoint_mejor.get(
            "variante_sac"
        )
        == "reactivo_mejorado"
        and checkpoint_mejor.get(
            "observacion_predictiva"
        )
        is False
        and checkpoint_mejor.get(
            "usa_ttc"
        )
        is False
    )

    carga_mejor_correcta = (
        len(
            resultado_carga_mejor.missing_keys
        )
        == 0
        and len(
            resultado_carga_mejor.unexpected_keys
        )
        == 0
    )

    parametros_mejor_finitos = all(
        torch.isfinite(
            parametro
        ).all().item()
        for parametro in actor_mejor.parameters()
    )

    numero_validaciones_correcto = (
        len(
            historial_validaciones
        )
        == numero_validaciones_esperado
    )

    validaciones_de_50_semillas = all(
        sum(
            registro[
                "conteos_resultados"
            ].values()
        )
        == NUMERO_SEMILLAS_VALIDACION_TERCERA_CORRIDA_SAC
        for registro in historial_validaciones
    )

    archivos_guardados = all(
        Path(
            ruta
        ).is_file()
        for ruta in [
            RUTA_MEJOR_ACTOR_TERCERA_CORRIDA_SAC,
            RUTA_CHECKPOINT_FINAL_TERCERA_CORRIDA_SAC,
            RUTA_CSV_VALIDACIONES_TERCERA_CORRIDA_SAC,
            RUTA_GRAFICA_VALIDACIONES_TERCERA_CORRIDA_SAC,
            RUTA_GRAFICA_DIAGNOSTICA_TERCERA_CORRIDA_SAC,
        ]
    )

    observacion_no_predictiva = (
        checkpoint_mejor[
            "observacion_predictiva"
        ]
        is False
        and checkpoint_mejor[
            "usa_ttc"
        ]
        is False
    )

    todo_correcto = (
        carga_actor_correcta
        and checkpoint_mejor_correcto
        and carga_mejor_correcta
        and parametros_mejor_finitos
        and numero_validaciones_correcto
        and validaciones_de_50_semillas
        and archivos_guardados
        and observacion_no_predictiva
    )

    # ------------------------------------------------------
    # 15. Mostrar resultados
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESULTADO DE LA TERCERA CORRIDA REACTIVA"
    )

    print(
        "=" * 80
    )

    print(
        "\nENTRENAMIENTO"
    )

    print(
        "Episodios:",
        resumen_global[
            "numero_episodios"
        ],
    )

    print(
        "Conteos:",
        resumen_global[
            "conteos_resultados"
        ],
    )

    print(
        "Éxito acumulado de entrenamiento:",
        f"{100.0 * resumen_global['tasa_exito']:.2f}%",
    )

    print(
        "Actualizaciones totales:",
        resumen_global[
            "actualizaciones_totales"
        ],
    )

    print(
        "Transiciones totales:",
        resumen_global[
            "transiciones_totales_buffer"
        ],
    )

    print(
        "Alpha final:",
        resumen_global[
            "alpha_final"
        ],
    )

    print(
        "\nMEJOR ACTOR SELECCIONADO"
    )

    print(
        "Episodio:",
        resultado_entrenamiento[
            "mejor_episodio"
        ],
    )

    print(
        "Éxito de validación:",
        f"{100.0 * mejor_resumen['tasa_exito']:.2f}%",
    )

    print(
        "Timeout:",
        f"{100.0 * mejor_resumen['tasa_timeout']:.2f}%",
    )

    print(
        "Colisión dinámica:",
        f"{100.0 * mejor_resumen['tasa_colision_dinamica']:.2f}%",
    )

    print(
        "Colisión estática:",
        f"{100.0 * mejor_resumen['tasa_colision_estatica']:.2f}%",
    )

    print(
        "Fuera del mapa:",
        f"{100.0 * mejor_resumen['tasa_fuera_mapa']:.2f}%",
    )

    print(
        "Distancia final media:",
        f"{mejor_resumen['distancia_final_media']:.3f} m",
    )

    print(
        "Clearance medio en éxitos:",
        f"{mejor_resumen['clearance_minimo_promedio_exitos']:.3f} m",
    )

    print(
        "\nDIAGNÓSTICO INDEPENDIENTE DE UNA SEMILLA"
    )

    print(
        "Semilla:",
        SEMILLA_DIAGNOSTICA_TERCERA_CORRIDA_SAC,
    )

    print(
        "Resultado:",
        resultado_diagnostico[
            "resultado"
        ],
    )

    print(
        "Pasos:",
        resultado_diagnostico[
            "numero_pasos"
        ],
    )

    print(
        "Distancia final:",
        resultado_diagnostico[
            "metricas"
        ][
            "distancia_final_meta"
        ],
        "m",
    )

    print(
        "\nARCHIVOS"
    )

    print(
        "Mejor actor:",
        RUTA_MEJOR_ACTOR_TERCERA_CORRIDA_SAC,
    )

    print(
        "Checkpoint final:",
        informacion_metadatos_final[
            "ruta"
        ],
    )

    print(
        "CSV de validaciones:",
        resultado_historial_validaciones[
            "ruta_csv"
        ],
    )

    print(
        "Gráfica de validaciones:",
        resultado_historial_validaciones[
            "ruta_grafica"
        ],
    )

    print(
        "Gráficas de entrenamiento:",
        DIRECTORIO_GRAFICAS_TERCERA_CORRIDA_SAC,
    )

    print(
    "Trayectoria diagnóstica:",
    resultado_grafica_diagnostica[
        "ruta_guardada"],
    )

    print(
        "\nVERIFICACIONES TÉCNICAS"
    )

    print(
        "Actor inicial cargado:",
        carga_actor_correcta,
    )

    print(
        "Checkpoint mejor actor correcto:",
        checkpoint_mejor_correcto,
    )

    print(
        "Mejor actor cargado estrictamente:",
        carga_mejor_correcta,
    )

    print(
        "Parámetros finitos:",
        parametros_mejor_finitos,
    )

    print(
        "Número de validaciones correcto:",
        numero_validaciones_correcto,
    )

    print(
        "Cada validación usa 50 semillas:",
        validaciones_de_50_semillas,
    )

    print(
        "Archivos guardados:",
        archivos_guardados,
    )

    print(
        "SAC continúa sin predicción:",
        observacion_no_predictiva,
    )

    if todo_correcto:

        print(
            "\nRESULTADO TÉCNICO: TERCERA CORRIDA COMPLETADA"
        )

    else:

        print(
            "\nRESULTADO TÉCNICO: HAY ALGO POR CORREGIR"
        )

    print(
        "=" * 80
    )

    if MOSTRAR_RESULTADOS_TERCERA_CORRIDA_SAC:

        plt.show()

    else:

        plt.close(
            "all"
        )


if __name__ == "__main__":
    main()
