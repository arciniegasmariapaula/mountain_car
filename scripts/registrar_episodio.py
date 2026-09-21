"""Graba un episodio real de un modelo guardado y registra sus decisiones."""

import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import torch
from PIL import Image

from mountain_car.agents.dqn import DQNAgent


def main():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("modelo", type=Path)
    analizador.add_argument("--semilla", type=int, default=100042)
    analizador.add_argument("--salida", type=Path, required=True)
    argumentos = analizador.parse_args()
    if argumentos.semilla < 0:
        analizador.error("La semilla debe ser mayor o igual a cero.")
    argumentos.salida.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    torch.set_num_threads(1)
    agente = DQNAgent.load(argumentos.modelo)
    entorno = gym.make(agente.env_id, render_mode="rgb_array")
    filas = []
    imagenes = []
    try:
        observacion, _ = entorno.reset(seed=argumentos.semilla)
        imagenes.append(Image.fromarray(entorno.render()))
        while True:
            accion, _ = agente.predict(observacion, deterministic=True)
            anterior = observacion.copy()
            observacion, recompensa, terminado, truncado, _ = entorno.step(accion)
            filas.append(
                {
                    "paso": len(filas) + 1,
                    "posicion": float(anterior[0]),
                    "velocidad": float(anterior[1]),
                    "accion": accion,
                    "recompensa": float(recompensa),
                    "siguiente_posicion": float(observacion[0]),
                    "siguiente_velocidad": float(observacion[1]),
                    "llego_a_la_bandera": bool(terminado),
                    "limite_de_tiempo": bool(truncado),
                }
            )
            if len(filas) % 2 == 0 or terminado or truncado:
                imagenes.append(Image.fromarray(entorno.render()))
            if terminado or truncado:
                break
    finally:
        entorno.close()
    imagenes[0].save(
        argumentos.salida / "episodio.gif",
        save_all=True,
        append_images=imagenes[1:],
        duration=[67] * (len(imagenes) - 1) + [700],
        loop=0,
    )
    with (argumentos.salida / "trayectoria.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    resumen = {
        "modelo": argumentos.modelo.as_posix(),
        "semilla": argumentos.semilla,
        "pasos": len(filas),
        "recompensa": sum(fila["recompensa"] for fila in filas),
        "llego_a_la_bandera": filas[-1]["llego_a_la_bandera"],
        "exploracion": False,
        "nota": "Grabación de un episodio; el desempeño general se informa en la evaluación completa.",
    }
    texto = json.dumps(resumen, ensure_ascii=False, indent=2)
    (argumentos.salida / "resumen.json").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
