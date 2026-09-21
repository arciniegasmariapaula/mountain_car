# agente-dqn-exploracion-resultados

El objetivo es comprobar si una exploración más persistente ayuda al carrito a descubrir cómo tomar impulso. Se mantiene la red neuronal y la actualización de Bellman del equipo. El cambio está en la selección de acciones durante el entrenamiento.

## Por dónde comenzar

1. Abre en VS Code la carpeta del repositorio `mountain_car`. Es la que contiene `README.md`, `pyproject.toml` y `src`.
2. Comprueba con tu equipo que tienes su versión más reciente. Este aporte se preparó sobre la revisión `6cad235d7329ec009e2fc9d2fad58bfc63c4ad9e`. Si ya modificaron el DQN después de esa revisión, compara los archivos antes de reemplazarlo.
3. Si abriste el proyecto preparado que acompaña esta guía, los cambios ya están incorporados. Si trabajas en otra copia del grupo, conserva una copia del DQN actual fuera del repositorio y copia los archivos de este aporte respetando sus carpetas. El archivo que reemplaza al anterior es `src/mountain_car/agents/dqn.py`. En el README del grupo integra solo el apartado de Persona 4.
4. Abre **Terminal > Nueva terminal** en VS Code. Los comandos siguientes se ejecutan allí, no dentro de una celda de Python.

```powershell
uv sync
uv run mountaincar inspect --steps 3
uv run python -m unittest discover -s tests -p "test_dqn_persona4.py" -v
```

El proyecto utiliza Python 3.11 y `uv` prepara su entorno. Si el equipo todavía no tiene `uv`, sigue su instalación oficial en https://docs.astral.sh/uv/getting-started/installation/ y abre de nuevo la terminal. No cambies a Python 3.12 dentro de este proyecto: `pyproject.toml` pide la versión 3.11.

## Qué cambió en el código

- `pasos_exploracion`: duración de cada bloque. El valor inicial es 20; con 1 se obtiene el comportamiento anterior.
- `select_action`: cuando comienza a explorar, elige una acción al azar y la conserva durante el bloque completo. Al terminar, vuelve a decidir si explora o utiliza la red.
- `_reiniciar_exploracion`: borra la acción pendiente al iniciar cada episodio.
- `_HPARAMS`: incluye el nuevo parámetro para guardarlo y recuperarlo.
- `load`: acepta modelos antiguos sin ese parámetro. Para ellos conserva el valor 1, correspondiente al comportamiento original.
- `train`: acepta una semilla opcional para las posiciones iniciales y sincroniza la red objetivo según el total de episodios acumulados.

La elección aleatoria sigue incluyendo las tres acciones del entorno. No se obliga al carro a seguir una estrategia basada en su velocidad. Se mantiene una acción varios pasos para que pueda aparecer un empuje sostenido.

Durante un bloque no se vuelve a sortear epsilon en cada paso. Por eso epsilon expresa la probabilidad de iniciar otro bloque cuando se toma una nueva decisión; no equivale exactamente al porcentaje de pasos exploratorios. Esto ayuda a explicar por qué la recompensa de entrenamiento puede ser peor que la de evaluación.

Cuando `deterministic=True`, la acción sale siempre de la red, aunque haya un bloque pendiente. La evaluación no utiliza la ayuda de la exploración.

## Español y compatibilidad

Los comentarios, las explicaciones, los mensajes y las variables internas nuevas están en español. Se conservan los nombres públicos que usa el proyecto, como `DQNAgent`, `QNetwork`, `ReplayBuffer`, `select_action`, `train`, `save` y `load`; también las claves que ya existen en los modelos guardados. Las palabras de Python y los métodos de PyTorch pertenecen al lenguaje y a las bibliotecas.

La arquitectura sigue siendo 2 entradas, dos capas ocultas de 128 neuronas con ReLU y 3 salidas. El aprendizaje conserva la memoria de experiencias, el lote de 64, la red objetivo y el error cuadrático medio de la implementación del equipo.

El cuaderno `evidencias/dqn/DQN_MountainCar.ipynb` se conserva como evidencia previa del equipo y contiene la versión anterior de la clase. Las pruebas nuevas importan el archivo `src/mountain_car/agents/dqn.py`; editar ese cuaderno antiguo no modifica el agente que utiliza el proyecto.

## Primera prueba de funcionamiento

```powershell
uv run --with matplotlib python scripts/experimento_dqn.py --episodios 20 --pasos-exploracion 20 --semilla 42 --salida evidencias/dqn/prueba_20_episodios
```

Este comando crea un agente nuevo y una carpeta nueva. Si la carpeta ya existe, utiliza otro nombre: así conservas los resultados anteriores.

Una prueba de veinte episodios permite comprobar la ejecución, el guardado, la carga, la evaluación y la creación de evidencias. Un resultado de -200 en esta prueba corta no determina el resultado del entrenamiento completo.

`--with matplotlib` permite generar la curva sin editar las dependencias del proyecto. El script puede funcionar sin matplotlib, pero en ese caso guarda el CSV y avisa de que no se generó la imagen.

## Experimento para la entrega

El entrenamiento de la entrega ya está terminado; consulta `docs/RESULTADOS_PERSONA_4.md`. Los siguientes comandos sirven para repetirlo desde cero:

```powershell
uv run --with matplotlib python scripts/experimento_dqn.py --episodios 2500 --pasos-exploracion 20 --semilla 42 --evaluaciones 100 --salida evidencias/dqn/bloques20_repeticion
```

Para medir el efecto de la modificación, ejecuta también la referencia con el mismo número de episodios y la misma semilla:

```powershell
uv run --with matplotlib python scripts/experimento_dqn.py --episodios 2500 --pasos-exploracion 1 --semilla 42 --evaluaciones 100 --salida evidencias/dqn/independiente_repeticion
```

Ambos comandos comienzan desde cero. El cuaderno existente del equipo sirve como antecedente, pero compara 1.000 episodios; no equivale a una ejecución nueva de 2.500.

Si necesitas ajustar la duración, prueba 40 pasos conservando los demás parámetros y usando otra carpeta. No se garantiza obtener -106: ese valor es una referencia del ejercicio. Informa los resultados reales y las configuraciones probadas. Una sola semilla y cien evaluaciones describen esa ejecución; varias semillas de entrenamiento permiten comprobar si la mejora se repite.

El script fija semillas para Python, NumPy, PyTorch y el entorno; registra las versiones y utiliza un hilo de PyTorch. Aun así, equipos, versiones y dispositivos distintos pueden producir diferencias.

## Archivos que quedan como evidencia

| Archivo | Contenido |
| --- | --- |
| `configuracion.json` | Hiperparámetros, semillas, versiones y huella del código utilizado. |
| `entrenamiento.csv` | Recompensa de cada episodio. |
| `curva_entrenamiento.png` | Recompensas y media móvil de hasta 50 episodios. |
| `evaluacion.csv` | Recompensa, pasos, semilla y llegada a la bandera en cada evaluación. |
| `resultado.txt` | Resumen legible del resultado. |
| `resumen.json` | El mismo resultado en formato estructurado. |
| `registro.txt` | Mensajes impresos durante la ejecución. |
| `validacion.csv` | Evaluaciones intermedias que se usan para seleccionar el modelo. |
| `dqn_mejor.pt` | Modelo con mayor recompensa media de validación. |
| `dqn_final.pt` | Modelo después de completar todos los episodios. |
| `dqn_episodio_N.pt` | Copia de un modelo que mejoró la validación. |

El número de éxitos se cuenta mediante la señal de llegada del entorno (`terminated`). No se cuenta con `recompensa > -200`, porque un episodio que llega exactamente en el paso 200 también es un éxito.

Cada 250 episodios se valida sin exploración en veinte posiciones iniciales. Se guarda el modelo con mayor recompensa media; en caso de empate se conserva el primero. La evaluación final se realiza después de volver a cargar ese modelo y utiliza cien episodios con semillas distintas. El entrenamiento siempre completa el número solicitado, aunque el modelo seleccionado corresponda a un episodio anterior. El protocolo se explica en `docs/METODOLOGIA_DQN.md`.

Para evaluar por separado cualquier modelo guardado, sin volver a entrenarlo:

```powershell
uv run python scripts/evaluar_modelo.py evidencias/dqn/bloques20_semilla42_2500/dqn_mejor.pt --episodios 100 --semilla 100042 --salida evidencias/dqn/mi_evaluacion
```

Para volver a generar la comparación de los experimentos completos:

```powershell
uv run --with matplotlib python scripts/generar_comparacion.py evidencias/dqn/independiente_semilla42_2500 evidencias/dqn/bloques20_semilla42_2500 --salida evidencias/dqn/mi_comparacion
```

## Cómo dejarlo plasmado en el README

El README de esta copia ya incorpora la explicación del cambio. El archivo `docs/APORTE_PERSONA_4_README.md` contiene el texto del aporte con los resultados medidos, listo para integrarlo después de revisar las evidencias. Al integrar en otra copia del grupo, combina únicamente los cambios de la sección DQN.

En la conclusión explica:

1. Qué observaste con la exploración independiente.
2. Qué duración de bloque usaste y por qué la probaste.
3. Qué recompensa media y cuántas llegadas obtuviste sin exploración.
4. Si la mejora fue suficiente y qué limitaciones observaste.

En MountainCar la recompensa es -1 por paso, incluso al llegar a la meta. La ventaja de llegar antes es acumular menos penalizaciones. No hay un premio positivo adicional por llegar a la bandera.

## Compartir el aporte con el grupo

Los archivos para incorporar son:

```text
src/mountain_car/agents/dqn.py
scripts/experimento_dqn.py
scripts/evaluar_modelo.py
scripts/generar_comparacion.py
scripts/diagnosticar_exploracion.py
scripts/registrar_episodio.py
tests/test_dqn_persona4.py
docs/GUIA_PERSONA_4.md
docs/APORTE_PERSONA_4_README.md
docs/METODOLOGIA_DQN.md
docs/RESULTADOS_PERSONA_4.md
docs/VALIDACION_PERSONA_4.md
EMPIEZA_AQUI.md
evidencias/dqn/ (experimentos completos, comparación, diagnóstico y animación)
saves/dqn_mountaincar.pt (copia para los comandos del proyecto)
README.md (solo los cambios de la sección DQN)
```

Después añade las evidencias del experimento elegido y actualiza la sección DQN del README. El modelo puede ser grande; coordina con el grupo dónde conservarlo. Los comandos originales guardan en `saves/`, carpeta que el repositorio excluye de Git. El script de experimentos guarda el modelo dentro de su propia carpeta de evidencia.

El comando original `uv run mountaincar train dqn` retoma el modelo de `saves/` si existe. Para comparar desde cero utiliza el script incluido. No borres el modelo previo para hacer la comparación.

El ZIP del proyecto completo contiene una copia del modelo en `saves/dqn_mountaincar.pt`. Este paquete para GitHub conserva el modelo dentro de las evidencias; cópialo a esa ubicación si quieres usar `uv run mountaincar render dqn`. Los experimentos futuros no actualizan esa copia automáticamente. Si integras el modelo en otra carpeta, conserva primero cualquier modelo anterior.

## Fuentes del proyecto

- Repositorio: https://github.com/arciniegasmariapaula/mountain_car
- Ejercicio 3: https://github.com/arciniegasmariapaula/mountain_car/blob/main/EXERCISES.md
- Entorno: https://gymnasium.farama.org/environments/classic_control/mountain_car/
