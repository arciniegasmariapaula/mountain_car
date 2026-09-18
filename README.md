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
| Persona 1- Maria Paula Arciniegas | Repositorio, entorno y README (coordinación) |
| Persona 2- Camilo Briceño | Agente Q-Learning: código, entrenamiento y evidencia |
| Persona 3- Sabrina Miranda| Agente DQN: red neuronal y aprendizaje |
| Persona 4 | Agente DQN: exploración y resultados |
| Persona 5- Camilo Rojas| Esquemas propios (dibujos a mano) |
| Persona 6 | Documentación y comparación final |

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

**Evidencia:** ver [`evidencias/qlearning/`](evidencias/qlearning/)

**Comentario:** _pendiente — breve análisis del comportamiento aprendido_

## 2. Agente Deep Q-Network (DQN)

*(Personas 3 y 4 completan esta sección)*

**Arquitectura de la red:** MLP `state_dim → hidden → hidden → action_dim`

**Componentes:** replay buffer, target network, exploración con acciones
temporalmente correlacionadas (necesaria para que el agente logre secuencias
sostenidas de empuje y pueda escapar el valle — ver `EXERCISES.md`, Ejercicio 3).

**Hiperparámetros:**

| Hiperparámetro | Valor |
|---|---|
| Tamaño de capas ocultas | _pendiente_ |
| Learning rate | _pendiente_ |
| `gamma` (γ) | _pendiente_ |
| Tamaño del replay buffer | _pendiente_ |
| Tamaño de batch | _pendiente_ |
| Frecuencia de actualización de target network | _pendiente_ |
| Hiperparámetros de exploración correlacionada | _pendiente_ |
| Episodios de entrenamiento | _pendiente_ |

**Resultado del mejor agente:** _pendiente_ (score de evaluación, episodios exitosos de 10)

**Evidencia:** ver [`evidencias/dqn/`](evidencias/dqn/)

**Comentario:** _pendiente — breve análisis del comportamiento aprendido_

## 3. Esquemas del proceso de entrenamiento

*(Persona 5 completa esta sección — dibujos propios, no generados por IA)*

### Q-Learning

![Esquema Q-Learning](evidencias/esquemas/qlearning.jpg)
*(reemplazar por la imagen escaneada)*

### DQN

![Esquema DQN](evidencias/esquemas/dqn.jpg)
*(reemplazar por la imagen escaneada)*

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
