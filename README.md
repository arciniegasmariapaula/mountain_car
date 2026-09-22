# MountainCar-v0 — Q-Learning tabular vs. Deep Q-Network (DQN)

Maestría en Inteligencia Artificial — Simulación y Aprendizaje por Refuerzo

Trabajo en equipo: implementación y comparación de dos enfoques de Aprendizaje
por Refuerzo para resolver el entorno [`MountainCar-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
de Gymnasium — un agente de **Q-Learning tabular** (con discretización del
espacio de estados) y un agente de **Deep Q-Network (DQN)**.

Repositorio base del curso: [emiliomunozai/mountain_car](https://github.com/emiliomunozai/mountain_car)

## Equipo

| Persona | Responsabilidad |
|---|---|
| Persona 1- Maria Paula Arciniegas Longas | Repositorio, entorno y README (coordinación) |
| Persona 2- Camilo Andres Briceno Leon | Agente Q-Learning: código, entrenamiento y evidencia |
| Persona 3- Sabrina Anais Miranda Franco| Agente DQN: red neuronal y aprendizaje |
| Persona 4- Carlos Eduardo Caicedo Hortua| Agente DQN: exploración y resultados |
| Persona 5- Cesar Camilo Rojas Sarmiento| Esquemas propios (dibujos a mano) |
| Persona 6-Cesar Hernan Garcia Afanador | Documentación y comparación final |

## Objetivo

Entender las diferencias entre los métodos tabulares clásicos y el Deep RL,
analizando el proceso de entrenamiento, el desempeño obtenido y las
limitaciones de cada aproximación, sobre el mismo entorno.

## Estructura del repositorio

```
mountain_car/
├── src/mountain_car/
│   ├── cli.py
│   └── agents/
│       ├── qlearning.py   # Ejercicio 1: Q-Learning tabular
│       └── dqn.py         # Ejercicios 2 y 3: DQN
├── evidencias/
│   ├── qlearning/         # Curva/score del mejor resultado de Q-Learning
│   ├── dqn/                # Curva/score del mejor resultado de DQN
│   └── esquemas/           # Dibujos propios: ciclo de Q-Learning y de DQN
├── saves/                  # Agentes entrenados guardados
├── EXERCISES.md            # Enunciado original de los ejercicios (curso)
└── pyproject.toml
```

## Instalación y ejecución

El proyecto usa [`uv`](https://docs.astral.sh/uv/) para manejar el entorno y las dependencias.

```bash
uv sync                                   # instala el entorno y las dependencias
uv run mountaincar inspect --steps 3      # inspecciona el entorno
```

Comandos principales:

```bash
# Q-Learning
uv run mountaincar train qlearning --episodes 10000
uv run mountaincar load qlearning --eval
uv run mountaincar render qlearning --episodes 3

# DQN
uv run mountaincar train dqn --episodes 1000
uv run mountaincar load dqn --eval
uv run mountaincar render dqn --episodes 3
```

Otros comandos útiles: `list`, `init <agente>`, `sim <agente>`, `delete <agente>`.

## 1. Agente Q-Learning tabular

*(Persona 2 completa esta sección)*

**Discretización:** `n_bins = 20` por dimensión (posición, velocidad) → 400 estados posibles.

**Hiperparámetros:**

| Hiperparámetro | Valor |
|---|---|
| `n_bins` | 20 |
| `lr` (α) | _pendiente_ |
| `gamma` (γ) | _pendiente_ |
| `epsilon_start` | _pendiente_ |
| `epsilon_end` | _pendiente_ |
| `epsilon_decay` | _pendiente_ |
| Episodios de entrenamiento | _pendiente_ |

**Resultado del mejor agente:** _pendiente_ (score de evaluación, episodios exitosos de 10)


**Verificación de la red y del aprendizaje (Ejercicio 2):**

| Prueba | Resultado |
|---|---|
| CartPole-v1 (200 episodios) | recompensa promedio de 20.68 (episodios 1-25) a 178.56 (episodios 176-200) |
| MountainCar-v0 (1000 episodios, exploración epsilon-greedy normal) | -200 en todos los episodios, 0 de 1000 llegaron a la bandera |
| Promedio de los valores Q en MountainCar | -63.53 |
| Diferencia promedio entre la mejor y la peor acción | 0.008 |


<img width="592" height="289" alt="image" src="https://github.com/user-attachments/assets/75ac7b7f-e8b7-47f8-8040-6e0601e11bd0" />

<img width="595" height="284" alt="image" src="https://github.com/user-attachments/assets/abd7b4c3-72a4-4b34-b639-532b4c5f106e" />


**Comentario:** En MountainCar el agente no aprendió. En los 1000 episodios nunca llegó a la bandera. Por eso solo recibió -1 en cada paso y nunca vio una recompensa diferente. La red le da casi el mismo valor Q en las tres acciones (la diferencia es de solo 0.008). Es decir, para la red da igual ir a la izquierda, a la derecha o no hacer nada ya que como nunca llegó a la meta, nada de lo que hizo le dio un mejor resultado. Para subir la montaña, el carro tiene que empujar muchas veces seguidas hacia el mismo lado. Pero al explorar, el agente elige una acción al azar en cada paso, así que casi nunca repite la misma varias veces. Esto se corrige en el Ejercicio 3.


**Evidencia:** ver [`evidencias/qlearning/`](evidencias/qlearning/)

**Comentario:** _pendiente — breve análisis del comportamiento aprendido_

## 2. Agente Deep Q-Network (DQN)

**Arquitectura de la red:** MLP `state_dim → hidden → hidden → action_dim`

**Componentes:** replay buffer, target network, exploración con acciones
temporalmente correlacionadas (necesaria para que el agente logre secuencias
sostenidas de empuje y pueda escapar el valle — ver `EXERCISES.md`, Ejercicio 3).

**Hiperparámetros:**

| Hiperparámetro | Valor |
| --- | --- |
| Tamaño de capas ocultas | 2 capas de 128 neuronas, activación ReLU |
| Learning rate | 0.001 (optimizador Adam) |
| `gamma` (γ) | 0.99 |
| Tamaño del replay buffer | 100.000 transiciones |
| Tamaño de batch | 64 |
| Frecuencia de actualización de target network | cada 10 episodios |
| Epsilon | de 1.0 a 0.01, decaimiento de 0.995 por episodio |
| Función de pérdida | MSE (error cuadrático medio) |
| Hiperparámetros de exploración correlacionada | `pasos_exploracion = 20`: repetir la misma acción exploratoria durante 20 pasos; referencia con `pasos_exploracion = 1` |
| Episodios de entrenamiento | 2.500 por configuración, con semilla 42 |

**Resultado del mejor agente:** DQN con exploración correlacionada, seleccionado mediante validación en el episodio 1.750 de un entrenamiento completo de 2.500 episodios. Obtuvo un score medio de **−99,00** y **10/10 episodios exitosos** en las primeras diez evaluaciones. Estas forman parte de una evaluación de 100 episodios, cuyo resultado completo fue **−102,23 ± 12,59** (media ± desviación estándar) y **100/100 episodios exitosos**, sin exploración.

**Evidencia:** ver [evidencias/dqn/](evidencias/dqn/).

- [Informe completo](docs/Resultados_agente-dqn-exploracion-resultados.md).
- [Curvas de entrenamiento y validación](evidencias/dqn/comparacion_persona4/comparacion_aprendizaje.png).
- [Comparación de las cien evaluaciones](evidencias/dqn/comparacion_persona4/comparacion_evaluacion.png).
- [Resultados del agente con bloques](evidencias/dqn/bloques20_semilla42_2500/resultado.txt).
- [Registro y modelos del agente con bloques](evidencias/dqn/bloques20_semilla42_2500/).
- [Registro y modelos de la referencia](evidencias/dqn/independiente_semilla42_2500/).
- [Grabación de un episodio real](evidencias/dqn/demostracion_persona4/episodio.gif).

**Comentario:** Mantener una acción exploratoria durante 20 pasos facilitó secuencias sostenidas de empuje durante el entrenamiento. El agente logró llegar a la meta en todos los episodios evaluados, mientras que la referencia con exploración independiente obtuvo −200,00 y 0/100 llegadas. El parámetro `pasos_exploracion` se añadió a `_HPARAMS` para guardarlo y cargarlo con el modelo, y el estado de exploración se reinicia al comenzar cada episodio en `train()`. La repetición se aplica en `select_action` durante la exploración; la evaluación utiliza las decisiones de la red sin repetición forzada. Los resultados corresponden a una semilla de entrenamiento por configuración.

## 3. Esquemas del proceso de entrenamiento

### Q-Learning — tabla Q

El esquema muestra cómo el agente observa el estado, selecciona una acción, recibe una recompensa y actualiza el valor correspondiente en la tabla Q, la exploración permite probar otras acciones y descubrir mejores estrategias.

![Esquema de Q-Learning](evidencias/esquemas/qlearning.png)

### DQN — redes y memoria

El esquema muestra la red principal, la memoria de experiencias, la red objetivo y la actualización de Bellman. La memoria permite reutilizar experiencias y seleccionar muestras al azar para entrenar, la red objetivo proporciona una referencia más estable para calcular los valores que la red principal aprende a estimar.

![Esquema de DQN](evidencias/esquemas/dqn.png)


## 4. Comparación entre Q-Learning y DQN

*(Persona 6 completa esta sección)*

| Aspecto | Q-Learning tabular | DQN |
|---|---|---|
| Estabilidad del entrenamiento | _pendiente_ | _pendiente_ |
| Velocidad de aprendizaje | _pendiente_ | _pendiente_ |
| Desempeño final | _pendiente_ | _pendiente_ |
| Ventajas | _pendiente_ | _pendiente_ |
| Limitaciones | _pendiente_ | _pendiente_ |
| Dificultad de implementación | _pendiente_ | _pendiente_ |

_pendiente — análisis escrito, apoyado en los números reales de cada agente._

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed., caps. 4–6). MIT Press.
- Documentación de [Gymnasium — MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/).
- Lapan, M. (2020). *Deep Reinforcement Learning Hands-On* (2.ª ed.). Packt.
git config --global user.name "Maria Paula Arciniegas"
git config --global user.email "arciniegasmariapaula@gmail.com"
