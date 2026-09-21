## Exploración y resultados del DQN

La exploración original elegía una acción aleatoria en cada paso. Se añadió `pasos_exploracion` para mantener esa acción durante un bloque de pasos: 20 en la propuesta y 1 en la referencia. Esto facilita que aparezcan empujes sostenidos con los que el carrito pueda tomar impulso. Se conservaron la red del equipo, la recompensa y la actualización de Bellman.

Los bloques se reinician al comenzar cada episodio. La evaluación utiliza `deterministic=True`: las acciones salen de la red, sin exploración ni repetición forzada.

### Resultado del entrenamiento completo

Se entrenaron ambos agentes desde cero durante **2.500 episodios**, con semilla 42 y los mismos hiperparámetros. Cada 250 episodios se hicieron 20 validaciones para elegir el modelo. Después se cargó el modelo elegido y se evaluó en **100 episodios con semillas distintas**, sin exploración.

| Exploración | Episodios entrenados | Modelo seleccionado | Recompensa media ± desviación | Llegadas a la meta |
| --- | ---: | ---: | ---: | ---: |
| Exploración independiente | 2.500 | Episodio 250 | -200,00 ± 0,00 | 0/100 |
| Bloques de 20 pasos | 2.500 | Episodio 1.750 | -102,23 ± 12,59 | 100/100 |

Las primeras diez evaluaciones del agente con bloques dieron **-99,00** de recompensa media y **10/10** llegadas. Son un subconjunto de las cien evaluaciones, no un experimento adicional.

### Interpretación

En esta ejecución, repetir las acciones durante la exploración permitió aprender un comportamiento que llega a la bandera. El DQN con exploración independiente se mantuvo en −200. En un diagnóstico adicional sin aprendizaje, las acciones aleatorias independientes llegaron 0 de 300 veces; al mantenerlas veinte pasos, llegaron 31 de 300. El cambio favoreció que la memoria contuviera experiencias de llegada a la meta.

MountainCar entrega −1 por paso, también al llegar. Por eso una recompensa menos negativa indica menos pasos; no existe un premio positivo adicional en la bandera. Los éxitos se cuentan mediante `terminated`, incluyendo una posible llegada en el paso 200.

El mejor modelo según la validación se obtuvo en el episodio **1750**. El entrenamiento continuó hasta 2.500 y también se conserva ese último modelo. Su evaluación obtuvo **-139,32 ± 50,96**, con **59/100** llegadas. Entrenar más no asegura que cada versión supere a la anterior.

Esta comparación utiliza una semilla de entrenamiento por configuración. Los cien episodios muestran el desempeño en distintas posiciones iniciales; para medir la variación entre entrenamientos habría que repetir con varias semillas.

### Evidencias

- [Informe completo](RESULTADOS_PERSONA_4.md).
- [Curvas de entrenamiento y validación](../evidencias/dqn/comparacion_persona4/comparacion_aprendizaje.png).
- [Comparación de las cien evaluaciones](../evidencias/dqn/comparacion_persona4/comparacion_evaluacion.png).
- [Resultados del agente con bloques](../evidencias/dqn/bloques20_semilla42_2500/resultado.txt).
- [Registro y modelos del agente con bloques](../evidencias/dqn/bloques20_semilla42_2500/).
- [Registro y modelos de la referencia](../evidencias/dqn/independiente_semilla42_2500/).
- [Grabación de un episodio real](../evidencias/dqn/demostracion_persona4/episodio.gif).
