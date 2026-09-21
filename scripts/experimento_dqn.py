"""Entrena desde cero y guarda las evidencias de un experimento DQN."""

import argparse
import contextlib
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents.dqn import DQNAgent


class RegistroConsola:
    """Escribe el mismo mensaje en pantalla y en el archivo de registro."""

    def __init__(self, pantalla, archivo):
        self.pantalla = pantalla
        self.archivo = archivo

    def write(self, texto):
        self.pantalla.write(texto)
        self.archivo.write(texto)
        self.flush()

    def flush(self):
        self.pantalla.flush()
        self.archivo.flush()


def entero_positivo(texto):
    valor = int(texto)
    if valor < 1:
        raise argparse.ArgumentTypeError("El valor debe ser mayor o igual a 1.")
    return valor


def evaluar(agente, episodios, semilla):
    """Evalúa sin exploración y comprueba la llegada mediante terminated."""
    entorno = gym.make(agente.env_id)
    resultados = []
    try:
        for episodio in range(episodios):
            observacion, _ = entorno.reset(seed=semilla + episodio)
            recompensa_total = 0.0
            pasos = 0
            while True:
                accion, _ = agente.predict(observacion, deterministic=True)
                observacion, recompensa, terminado, truncado, _ = entorno.step(accion)
                recompensa_total += float(recompensa)
                pasos += 1
                if terminado or truncado:
                    resultados.append(
                        {
                            "episodio": episodio + 1,
                            "semilla": semilla + episodio,
                            "recompensa": recompensa_total,
                            "pasos": pasos,
                            "llego_a_la_bandera": bool(terminado),
                            "limite_de_tiempo": bool(truncado),
                        }
                    )
                    break
    finally:
        entorno.close()
    return resultados


def guardar_curva(recompensas, carpeta):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("No se creó la gráfica: falta matplotlib. El CSV sí está guardado.")
        return False

    figura, eje = plt.subplots(figsize=(9, 4.5))
    episodios = np.arange(1, len(recompensas) + 1)
    eje.plot(episodios, recompensas, alpha=0.4, linewidth=0.8, label="Por episodio")
    ventana = min(50, len(recompensas))
    if ventana > 1:
        promedio = np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")
        eje.plot(
            episodios[ventana - 1 :],
            promedio,
            marker="o" if len(promedio) == 1 else None,
            label=f"Media de {ventana} episodios",
        )
    eje.set(
        xlabel="Episodio",
        ylabel="Recompensa total",
        title="Entrenamiento DQN en MountainCar",
    )
    eje.grid(alpha=0.25)
    eje.legend()
    figura.tight_layout()
    figura.savefig(carpeta / "curva_entrenamiento.png", dpi=160)
    plt.close(figura)
    return True


def ejecutar(argumentos, carpeta):
    inicio = time.perf_counter()
    random.seed(argumentos.semilla)
    np.random.seed(argumentos.semilla)
    torch.manual_seed(argumentos.semilla)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(argumentos.semilla)

    agente = DQNAgent("MountainCar-v0", pasos_exploracion=argumentos.pasos_exploracion)
    codigo = Path(__file__).resolve().parents[1] / "src/mountain_car/agents/dqn.py"
    configuracion = {
        "inicio": datetime.now().astimezone().isoformat(),
        "episodios_solicitados": argumentos.episodios,
        "episodios_evaluacion": argumentos.evaluaciones,
        "semilla_entrenamiento": argumentos.semilla,
        "semilla_evaluacion": argumentos.semilla + 100_000,
        "semilla_validacion": argumentos.semilla + 200_000,
        "episodios_validacion": argumentos.validaciones,
        "intervalo_validacion": argumentos.intervalo_validacion,
        "epsilon_inicial": agente.epsilon,
        "hiperparametros": {clave: getattr(agente, clave) for clave in agente._HPARAMS},
        "versiones": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
        },
        "dispositivo": str(agente.device),
        "hilos_torch": torch.get_num_threads(),
        "sha256_codigo_dqn": hashlib.sha256(codigo.read_bytes()).hexdigest(),
    }
    (carpeta / "configuracion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Experimento nuevo: se comienza con un agente sin entrenar.")
    print(agente.info())
    recompensas = []
    historial_validacion = []
    mejor_media = -float("inf")
    mejor_episodio = 0
    ruta_mejor = carpeta / "dqn_mejor.pt"
    ruta_historial = carpeta / "validacion.csv"
    while agente.training_episodes < argumentos.episodios:
        bloque = min(
            argumentos.intervalo_validacion,
            argumentos.episodios - agente.training_episodes,
        )
        recompensas.extend(
            agente.train(
                total_episodes=bloque,
                log_interval=min(50, bloque),
                semilla=argumentos.semilla,
            )
        )
        validaciones = evaluar(
            agente, argumentos.validaciones, argumentos.semilla + 200_000
        )
        media = float(np.mean([fila["recompensa"] for fila in validaciones]))
        exitos_validacion = sum(fila["llego_a_la_bandera"] for fila in validaciones)
        historial_validacion.append(
            {
                "episodios_entrenados": agente.training_episodes,
                "recompensa_media": media,
                "exitos": exitos_validacion,
                "episodios_validacion": argumentos.validaciones,
                "segundos_acumulados": round(time.perf_counter() - inicio, 2),
            }
        )
        with ruta_historial.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(historial_validacion[0]))
            escritor.writeheader()
            escritor.writerows(historial_validacion)
        if media > mejor_media:
            mejor_media = media
            mejor_episodio = agente.training_episodes
            ruta_control = carpeta / f"dqn_episodio_{mejor_episodio}.pt"
            agente.save(ruta_control)
            shutil.copyfile(ruta_control, ruta_mejor)
        print(
            f"Total entrenado: {agente.training_episodes}/{argumentos.episodios} | "
            f"Validación sin exploración: {media:.2f} | "
            f"Llegadas: {exitos_validacion}/{argumentos.validaciones} | "
            f"Mejor modelo: episodio {mejor_episodio}"
        )
    with (carpeta / "entrenamiento.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["episodio", "recompensa"])
        escritor.writerows(enumerate(recompensas, 1))

    ruta_modelo = carpeta / "dqn_final.pt"
    agente.save(ruta_modelo)
    # Evaluar la copia cargada comprueba que el archivo guardado se puede usar.
    agente_cargado = DQNAgent.load(ruta_mejor)
    resultados = evaluar(
        agente_cargado, argumentos.evaluaciones, argumentos.semilla + 100_000
    )
    with (carpeta / "evaluacion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0]))
        escritor.writeheader()
        escritor.writerows(resultados)

    recompensas_evaluacion = [fila["recompensa"] for fila in resultados]
    exitos = sum(fila["llego_a_la_bandera"] for fila in resultados)
    resumen = {
        "recompensa_media": float(np.mean(recompensas_evaluacion)),
        "desviacion_estandar": float(np.std(recompensas_evaluacion)),
        "exitos": exitos,
        "episodios_evaluados": argumentos.evaluaciones,
        "epsilon_final_entrenamiento": agente.epsilon,
        "exploracion_en_evaluacion": False,
        "modelo_evaluado": "dqn_mejor.pt",
        "episodios_del_modelo_evaluado": mejor_episodio,
        "recompensa_validacion_del_modelo": mejor_media,
        "segundos_totales": round(time.perf_counter() - inicio, 2),
        "curva_generada": guardar_curva(recompensas, carpeta),
    }
    (carpeta / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    texto = (
        f"Episodios de entrenamiento: {argumentos.episodios}\n"
        f"Pasos por bloque exploratorio: {argumentos.pasos_exploracion}\n"
        f"Semilla del experimento: {argumentos.semilla}\n"
        f"Evaluación sin exploración: {argumentos.evaluaciones} episodios\n"
        f"Recompensa media: {resumen['recompensa_media']:.2f}\n"
        f"Desviación estándar: {resumen['desviacion_estandar']:.2f}\n"
        f"Llegadas a la bandera: {exitos}/{argumentos.evaluaciones}\n"
        f"Modelo seleccionado en validación: episodio {mejor_episodio}\n"
        "La evaluación final utiliza semillas distintas de la validación.\n"
    )
    (carpeta / "resultado.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Evidencias guardadas en: {carpeta.resolve()}")


def main():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--episodios", type=entero_positivo, default=20)
    analizador.add_argument("--pasos-exploracion", type=entero_positivo, default=20)
    analizador.add_argument("--evaluaciones", type=entero_positivo, default=10)
    analizador.add_argument("--validaciones", type=entero_positivo, default=20)
    analizador.add_argument("--intervalo-validacion", type=entero_positivo, default=250)
    analizador.add_argument("--semilla", type=int, default=42)
    analizador.add_argument("--salida", type=Path)
    argumentos = analizador.parse_args()
    if argumentos.semilla < 0:
        analizador.error("La semilla debe ser mayor o igual a cero.")
    carpeta = argumentos.salida or (
        Path("evidencias/dqn")
        / datetime.now().astimezone().strftime("experimento_%Y%m%d_%H%M%S_%f")
    )
    # Cada ejecución usa una carpeta nueva para conservar las pruebas previas.
    carpeta.mkdir(parents=True, exist_ok=False)
    with (
        (carpeta / "registro.txt").open("w", encoding="utf-8") as archivo,
        contextlib.redirect_stdout(RegistroConsola(sys.stdout, archivo)),
    ):
        ejecutar(argumentos, carpeta)


if __name__ == "__main__":
    main()
