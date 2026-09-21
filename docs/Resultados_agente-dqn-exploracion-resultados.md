# Resultado: DQN en MountainCar

El entrenamiento completo terminó el 20 de septiembre de 2026. Se ejecutaron **5.000 episodios en total: 2.500 por configuración**. El agente con bloques de veinte pasos obtuvo **-102,23 ± 12,59** de recompensa y **100/100** llegadas en la evaluación independiente.

## Comparación principal

| Exploración | Episodios entrenados | Modelo seleccionado | Recompensa media ± desviación | Llegadas a la meta |
| --- | ---: | ---: | ---: | ---: |
| Exploración independiente | 2.500 | Episodio 250 | -200,00 ± 0,00 | 0/100 |
| Bloques de 20 pasos | 2.500 | Episodio 1.750 | -102,23 ± 12,59 | 100/100 |

La desviación estándar describe la variación entre los cien episodios de evaluación; no es un intervalo de confianza. La recompensa media equivale al número medio de pasos con signo negativo en este entorno.

![Evaluación independiente](../evidencias/dqn/comparacion_persona4/comparacion_evaluacion.png)

## Cómo se obtuvieron estos resultados

Se partió de la revisión `6cad235d7329ec009e2fc9d2fad58bfc63c4ad9e` del repositorio del equipo. La red tiene dos entradas, dos capas ocultas de 128 neuronas y tres salidas, con 17.283 parámetros. Se mantuvieron Adam (0.001), descuento 0.99, lote de 64, memoria de 100.000 experiencias y sincronización de la red objetivo cada diez episodios. Epsilon pasó de 1.0 a un mínimo de 0.01 con factor 0.995 por episodio.

Ambos agentes comenzaron desde cero con semilla 42. Se realizaron diez comprobaciones de validación, una cada 250 episodios, sobre veinte semillas fijas (200042–200061). Se conservó el modelo de mayor media; en empates se conservó el primero. La evaluación principal utilizó cien semillas diferentes (100042–100141), después de cargar el archivo guardado. Ninguna de esas cien evaluaciones se usó para elegir el modelo.

El parámetro que diferencia las configuraciones es `pasos_exploracion`: 1 y 20. En un bloque no se vuelve a sortear epsilon en cada paso, de modo que también cambia la proporción efectiva de pasos exploratorios. La comparación mide el efecto de esta implementación completa de exploración por bloques.

## Evolución durante el entrenamiento

![Entrenamiento y validación](../evidencias/dqn/comparacion_persona4/comparacion_aprendizaje.png)

La primera gráfica promedia ventanas de cien episodios de entrenamiento, que incluyen exploración. La segunda mide el comportamiento de la red sin explorar. Son medidas distintas y no tienen por qué coincidir.

El modelo con bloques se seleccionó en el episodio 1750, con una media de validación de -99,35. Esto no detuvo el entrenamiento: se completaron los 2.500 episodios y se conservan ambos archivos, `dqn_mejor.pt` y `dqn_final.pt`.

## Comprobación del último modelo

Además del modelo seleccionado, se evaluaron los archivos guardados exactamente al terminar el episodio 2.500, sobre las mismas cien semillas de prueba:

| Modelo del episodio 2.500 | Recompensa media ± desviación | Llegadas |
| --- | ---: | ---: |
| Exploración independiente | -200,00 ± 0,00 | 0/100 |
| Bloques de veinte pasos | -139,32 ± 50,96 | 59/100 |

Esta comprobación describe el último modelo. La selección del modelo entregado sigue siendo la establecida mediante validación, sin cambiarla a partir de la prueba final.

## Diagnóstico de la exploración

Antes de analizar el agente aprendido se midieron 300 episodios de acciones aleatorias por estrategia, sin entrenar una red. La exploración independiente llegó a la meta 0 veces; los bloques de veinte pasos llegaron 31 veces (10,33 %). Este diagnóstico indica que los bloques pueden producir experiencias de llegada, pero no equivale al rendimiento de un agente entrenado.

## Qué significa el resultado

El DQN con bloques aprendió a llegar a la meta en las posiciones iniciales evaluadas. La referencia independiente no lo consiguió en esta ejecución. Una recompensa menos negativa significa que el carrito emplea menos pasos, porque cada paso cuesta −1. No se añadió ninguna recompensa artificial ni una regla que le indique hacia dónde empujar según su velocidad.

Para la tabla de diez episodios que pide el reparto del grupo, las primeras diez evaluaciones guardadas dieron **-99,00** y **10/10** llegadas. Se informa también la muestra completa de cien para no escoger únicamente episodios favorables.

Se usó una semilla de entrenamiento por configuración. No se concluye que todos los entrenamientos futuros darán el mismo resultado. Los tiempos incluyen validaciones y ambas ejecuciones compartieron la CPU; no se utilizan como una comparación precisa de velocidad.

## Archivos para revisar

- [Protocolo y parámetros](METODOLOGIA_DQN.md).
- [Código del DQN](../src/mountain_car/agents/dqn.py).
- [Resumen comparativo en CSV](../evidencias/dqn/comparacion_persona4/comparacion.csv).
- [Experimento con bloques](../evidencias/dqn/bloques20_semilla42_2500/): configuración, 2.500 recompensas, validaciones, cien evaluaciones y modelos.
- [Experimento independiente](../evidencias/dqn/independiente_semilla42_2500/): los mismos registros para la referencia.
- [Diagnóstico sin aprendizaje](../evidencias/dqn/diagnostico_exploracion/resumen.json).
- [Episodio grabado](../evidencias/dqn/demostracion_persona4/episodio.gif) y [decisiones de ese episodio](../evidencias/dqn/demostracion_persona4/trayectoria.csv).
- [Guía para repetir o integrar el aporte](GUIA_PERSONA_4.md).

El modelo seleccionado está en `evidencias/dqn/bloques20_semilla42_2500/dqn_mejor.pt`. Para usar los comandos habituales del proyecto, copia ese archivo a `saves/dqn_mountaincar.pt`; conserva antes cualquier modelo previo. Los archivos `.pt` contienen pesos, configuración y optimizador; no guardan toda la memoria de experiencias ni el estado de los generadores aleatorios. Sirven para evaluar o continuar aprendiendo, pero no para reanudar exactamente el mismo experimento interrumpido.
