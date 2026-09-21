"""Evalúa un modelo guardado sin exploración y conserva los resultados."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from experimento_dqn import entero_positivo, evaluar

from mountain_car.agents.dqn import DQNAgent


def main():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("modelo", type=Path)
    analizador.add_argument("--episodios", type=entero_positivo, default=100)
    analizador.add_argument("--semilla", type=int, default=100042)
    analizador.add_argument("--salida", type=Path, required=True)
    argumentos = analizador.parse_args()
    if argumentos.semilla < 0:
        analizador.error("La semilla debe ser mayor o igual a cero.")
    torch.set_num_threads(1)
    agente = DQNAgent.load(argumentos.modelo)
    resultados = evaluar(agente, argumentos.episodios, argumentos.semilla)
    argumentos.salida.mkdir(parents=True, exist_ok=False)
    with (argumentos.salida / "evaluacion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0]))
        escritor.writeheader()
        escritor.writerows(resultados)
    recompensas = [fila["recompensa"] for fila in resultados]
    resumen = {
        "archivo_modelo": argumentos.modelo.as_posix(),
        "sha256_modelo": hashlib.sha256(argumentos.modelo.read_bytes()).hexdigest(),
        "episodios_entrenados": agente.training_episodes,
        "pasos_exploracion_entrenamiento": agente.pasos_exploracion,
        "episodios_evaluados": argumentos.episodios,
        "semilla_inicial": argumentos.semilla,
        "recompensa_media": float(np.mean(recompensas)),
        "desviacion_estandar": float(np.std(recompensas)),
        "exitos": sum(fila["llego_a_la_bandera"] for fila in resultados),
        "exploracion_en_evaluacion": False,
    }
    texto = json.dumps(resumen, ensure_ascii=False, indent=2)
    (argumentos.salida / "resumen.json").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
