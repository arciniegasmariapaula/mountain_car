# Protocolo del entrenamiento completo

Se comparan dos agentes DQN con la misma arquitectura y los mismos hiperparámetros. La diferencia es la duración de la acción exploratoria: un paso en la referencia y veinte pasos en la propuesta. Cada ejecución comienza desde cero y completa 2.500 episodios.

La red tiene dos entradas, dos capas ocultas de 128 neuronas con ReLU y tres salidas. Se usa Adam con tasa de aprendizaje 0.001, descuento 0.99, lotes de 64, memoria de 100.000 experiencias y actualización de la red objetivo cada diez episodios. Epsilon comienza en 1.0, se multiplica por 0.995 al finalizar cada episodio y tiene un mínimo de 0.01. La recompensa del entorno y la regla de Bellman se mantienen.

## Entrenamiento, selección y evaluación

La semilla de entrenamiento es 42 para Python, NumPy y PyTorch. El entorno utiliza una semilla por episodio: 42, 43 y así sucesivamente. Cada agente se entrena con un hilo de PyTorch en CPU.

Cada 250 episodios se evalúa la política sin exploración en veinte episodios con semillas 200042 a 200061. Se conserva el modelo con mayor recompensa media en esta validación. Si dos modelos empatan, se conserva el primero. La selección no interrumpe el entrenamiento: ambas ejecuciones llegan a los 2.500 episodios.

Al terminar se carga el modelo seleccionado desde el archivo y se evalúa en cien episodios adicionales, con semillas 100042 a 100141. Estas semillas no se usan para seleccionar el modelo. Ambos agentes comparten las posiciones iniciales de evaluación, lo que facilita una comparación bajo las mismas condiciones.

El número de llegadas se obtiene de `terminated`. Agotar los 200 pasos detiene el episodio, pero no se interpreta como una terminación real en la actualización de Bellman. La evaluación utiliza exclusivamente la acción de mayor valor estimado por la red.

## Evidencias

Cada carpeta de experimento conserva la configuración, el historial completo de recompensas, las validaciones intermedias, el modelo seleccionado, el modelo del último episodio y cien registros de evaluación. Se guarda una copia de cada modelo que mejora la validación.

La curva de entrenamiento muestra el desempeño mientras existe exploración. La validación y la evaluación final miden el comportamiento sin exploración; por eso sus recompensas pueden diferir de la curva de entrenamiento.

## Alcance de las conclusiones

La comparación corresponde a una semilla de entrenamiento por configuración. Los cien episodios de evaluación exploran distintas posiciones iniciales, pero no sustituyen cien entrenamientos independientes. Los resultados permiten describir estas ejecuciones; para establecer la estabilidad entre entrenamientos se necesitan varias semillas adicionales.

El tiempo registrado incluye entrenamiento y comprobaciones. Las dos ejecuciones comparten el equipo y se ejecutan de forma simultánea, de modo que su duración no se usa como una medida precisa de eficiencia entre algoritmos.
