"""Resume experimentos terminados y crea las figuras de comparación."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def leer_csv(ruta):
    with ruta.open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def main():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("experimentos", nargs="+", type=Path)
    analizador.add_argument("--salida", required=True, type=Path)
    argumentos = analizador.parse_args()
    argumentos.salida.mkdir(parents=True, exist_ok=False)
    filas = []
    registros = []
    for carpeta in argumentos.experimentos:
        configuracion = json.loads(
            (carpeta / "configuracion.json").read_text(encoding="utf-8")
        )
        resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
        entrenamiento = leer_csv(carpeta / "entrenamiento.csv")
        validacion = leer_csv(carpeta / "validacion.csv")
        evaluacion = leer_csv(carpeta / "evaluacion.csv")
        pasos = configuracion["hiperparametros"]["pasos_exploracion"]
        nombre = (
            "Exploración independiente" if pasos == 1 else f"Bloques de {pasos} pasos"
        )
        recompensas = [float(fila["recompensa"]) for fila in evaluacion]
        filas.append(
            {
                "configuracion": nombre,
                "episodios_entrenados": len(entrenamiento),
                "episodio_modelo_seleccionado": resumen[
                    "episodios_del_modelo_evaluado"
                ],
                "validacion_media": resumen["recompensa_validacion_del_modelo"],
                "evaluacion_media": resumen["recompensa_media"],
                "evaluacion_desviacion": resumen["desviacion_estandar"],
                "exitos_evaluacion": resumen["exitos"],
                "episodios_evaluados": len(evaluacion),
                "mejor_episodio_evaluacion": max(recompensas),
                "peor_episodio_evaluacion": min(recompensas),
                "media_primeras_10_evaluaciones": float(np.mean(recompensas[:10])),
                "exitos_primeras_10_evaluaciones": sum(
                    fila["llego_a_la_bandera"] == "True" for fila in evaluacion[:10]
                ),
                "minutos_ejecucion": round(resumen["segundos_totales"] / 60, 2),
                "carpeta": carpeta.name,
            }
        )
        registros.append((nombre, entrenamiento, validacion, recompensas))

    with (argumentos.salida / "comparacion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    (argumentos.salida / "comparacion.json").write_text(
        json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    colores = ["#65758b", "#147d92", "#b86423", "#7750a7"]
    figura, ejes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for indice, (nombre, entrenamiento, validacion, _) in enumerate(registros):
        color = colores[indice % len(colores)]
        recompensas = np.array([float(fila["recompensa"]) for fila in entrenamiento])
        ventana = min(100, len(recompensas))
        promedio = np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")
        ejes[0].plot(
            np.arange(ventana, len(recompensas) + 1),
            promedio,
            label=nombre,
            color=color,
        )
        ejes[1].plot(
            [int(fila["episodios_entrenados"]) for fila in validacion],
            [float(fila["recompensa_media"]) for fila in validacion],
            marker="o",
            label=nombre,
            color=color,
        )
    ejes[0].set(
        title="Entrenamiento con exploración",
        ylabel="Recompensa media de 100 episodios",
    )
    ejes[1].set(
        title="Validación sin exploración cada 250 episodios",
        ylabel="Recompensa media de 20 episodios",
    )
    for eje in ejes:
        eje.set_xlabel("Episodios entrenados")
        eje.grid(alpha=0.2)
        eje.legend()
    figura.savefig(argumentos.salida / "comparacion_aprendizaje.png", dpi=180)
    plt.close(figura)

    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for indice, (nombre, _, _, recompensas) in enumerate(registros):
        color = colores[indice % len(colores)]
        valores = np.array(recompensas)
        posiciones = np.arange(1, len(valores) + 1)
        ejes[0].plot(
            posiciones, valores, alpha=0.6, linewidth=1, label=nombre, color=color
        )
        ejes[1].errorbar(
            indice,
            valores.mean(),
            yerr=valores.std(),
            fmt="o",
            capsize=7,
            markersize=8,
            color=color,
        )
    ejes[0].set(
        title="Resultado de cada episodio",
        xlabel="Episodio de evaluación",
        ylabel="Recompensa total",
    )
    ejes[0].legend()
    ejes[1].set(title="Media y desviación estándar", ylabel="Recompensa total")
    ejes[1].set_xticks(
        range(len(registros)),
        [registro[0].replace(" ", "\n", 1) for registro in registros],
    )
    ejes[1].margins(x=0.4)
    for eje in ejes:
        eje.grid(alpha=0.2)
    figura.suptitle(
        "Evaluación independiente sin exploración · 100 episodios", fontsize=13
    )
    figura.savefig(argumentos.salida / "comparacion_evaluacion.png", dpi=180)
    plt.close(figura)
    print(json.dumps(filas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
