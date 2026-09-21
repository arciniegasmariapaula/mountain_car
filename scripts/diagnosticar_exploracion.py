"""Compara la llegada a la meta durante exploración pura, sin aprendizaje."""

import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym


def probar(pasos_bloque, episodios, semilla):
    azar = random.Random(semilla)
    entorno = gym.make("MountainCar-v0")
    resultados = []
    try:
        for episodio in range(episodios):
            entorno.reset(seed=semilla + episodio)
            restantes = 0
            pasos = 0
            recompensa_total = 0.0
            while True:
                if restantes == 0:
                    accion = azar.randrange(3)
                    restantes = pasos_bloque
                restantes -= 1
                _, recompensa, terminado, truncado, _ = entorno.step(accion)
                recompensa_total += float(recompensa)
                pasos += 1
                if terminado or truncado:
                    resultados.append(
                        {
                            "pasos_bloque": pasos_bloque,
                            "episodio": episodio + 1,
                            "semilla_entorno": semilla + episodio,
                            "pasos": pasos,
                            "recompensa": recompensa_total,
                            "llego_a_la_bandera": bool(terminado),
                        }
                    )
                    break
    finally:
        entorno.close()
    return resultados


def main():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--episodios", type=int, default=300)
    analizador.add_argument("--semilla", type=int, default=42)
    analizador.add_argument("--salida", type=Path, required=True)
    argumentos = analizador.parse_args()
    if argumentos.episodios < 1 or argumentos.semilla < 0:
        analizador.error("Se requieren episodios positivos y una semilla no negativa.")
    argumentos.salida.mkdir(parents=True, exist_ok=False)
    resultados = []
    resumen = []
    for pasos in (1, 20):
        filas = probar(pasos, argumentos.episodios, argumentos.semilla)
        resultados.extend(filas)
        resumen.append(
            {
                "pasos_bloque": pasos,
                "episodios": argumentos.episodios,
                "exitos": sum(fila["llego_a_la_bandera"] for fila in filas),
                "semilla": argumentos.semilla,
                "aprendizaje": False,
            }
        )
    with (argumentos.salida / "episodios.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0]))
        escritor.writeheader()
        escritor.writerows(resultados)
    texto = json.dumps(resumen, ensure_ascii=False, indent=2)
    (argumentos.salida / "resumen.json").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
