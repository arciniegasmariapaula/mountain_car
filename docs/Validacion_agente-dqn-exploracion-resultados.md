# Comprobaciones realizadas

Fecha: 20 de septiembre de 2026.

Base del proyecto: `6cad235d7329ec009e2fc9d2fad58bfc63c4ad9e` del repositorio del equipo.

## Código y compatibilidad

Pasaron siete pruebas: duración exacta de los bloques, evaluación sin exploración, recuperación de la exploración independiente con un paso, reinicio al cambiar de episodio, conservación de configuración y predicciones al guardar y cargar, carga de modelos anteriores y rechazo de duraciones inválidas.

También se construyeron correctamente el paquete de código fuente y el paquete instalable del proyecto. El control de estilo de los archivos nuevos y modificados de Python pasó.

En el cuaderno previo se corrigieron únicamente una separación de importaciones y una importación sin uso para que pase el control de estilo del repositorio. Sus resultados y su implementación anterior se conservaron.

## Prueba corta

Se ejecutó `scripts/experimento_dqn.py` con 20 episodios, bloques de 20 pasos y semilla 42. Se evaluó la copia del modelo que se volvió a cargar desde el archivo guardado.

| Dato | Valor observado |
| --- | --- |
| Entorno | MountainCar-v0 |
| Dispositivo | CPU |
| Episodios entrenados | 20 |
| Media de recompensa durante el entrenamiento | -189.20 |
| Episodios de evaluación sin exploración | 10 |
| Semillas de evaluación | 100042 a 100051 |
| Recompensa media de evaluación | -200.00 |
| Desviación estándar de evaluación | 0.00 |
| Llegadas a la bandera en evaluación | 0 de 10 |

Las evidencias están en `evidencias/dqn/prueba_funcionamiento_persona4/`. Se verificó la creación del modelo, la configuración, el registro, los CSV, el resumen y la gráfica.

Esta ejecución demuestra el funcionamiento del programa. No demuestra que el agente haya aprendido a resolver MountainCar. El entrenamiento completo posterior se describe a continuación.

## Versiones utilizadas

- Python 3.11.16.
- PyTorch 2.13.0.
- Gymnasium 1.3.0.
- NumPy 2.4.6.
- Matplotlib 3.11.2.

La configuración exacta y la huella del archivo DQN están guardadas en `configuracion.json`.

## Entrenamiento completo y archivos entregados

Se completaron 2.500 episodios por configuración. Los archivos contienen 2.500 recompensas de entrenamiento, diez validaciones y cien episodios de evaluación por experimento. La evaluación recarga el modelo guardado y utiliza únicamente las acciones de la red.

| Exploración | Episodios entrenados | Modelo seleccionado | Recompensa media ± desviación | Llegadas a la meta |
| --- | ---: | ---: | ---: | ---: |
| Exploración independiente | 2.500 | Episodio 250 | -200,00 ± 0,00 | 0/100 |
| Bloques de 20 pasos | 2.500 | Episodio 1.750 | -102,23 ± 12,59 | 100/100 |

También se evaluaron los modelos del episodio 2.500 y se registró una animación de un episodio real. El [informe de resultados](RESULTADOS_PERSONA_4.md) distingue los modelos seleccionados y los últimos modelos. En el ZIP del proyecto completo se comprobó que la copia de `saves/dqn_mountaincar.pt` coincide con `dqn_mejor.pt` del experimento con bloques. Este paquete conserva el modelo dentro de las evidencias.

La selección por validación se comprobó antes de los experimentos completos mediante una ejecución corta de dos episodios. La configuración se guardó antes de entrenar y la huella del código DQN coincide con la del archivo entregado.
