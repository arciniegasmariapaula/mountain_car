"""Agente DQN para aprender a controlar el carrito de MountainCar.

La red estima el valor de cada acción. Las experiencias se guardan en una
memoria y se reutilizan para entrenar. Una segunda red calcula los objetivos
de aprendizaje y recibe una copia de los pesos cada cierto número de episodios.

Se conservan los nombres públicos del proyecto y las claves de los modelos
guardados para mantener la compatibilidad con los demás archivos.
"""

import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim


class QNetwork(nn.Module):
    """Red con dos capas ocultas que devuelve un valor Q por acción.

    La salida no lleva una activación: los valores Q pueden ser negativos
    y no representan probabilidades.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    """Memoria de experiencias; al llenarse, descarta las más antiguas."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN con exploración mediante bloques de acciones repetidas.

    pasos_exploracion indica cuántos pasos dura una acción exploratoria.
    Con 1 se recupera la exploración independiente del código original.
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
        pasos_exploracion: int = 20,
    ) -> None:
        if (
            isinstance(pasos_exploracion, bool)
            or not isinstance(pasos_exploracion, int)
            or pasos_exploracion < 1
        ):
            raise ValueError("pasos_exploracion debe ser un entero mayor o igual a 1.")

        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.hidden = hidden
        self.pasos_exploracion = pasos_exploracion
        self.training_episodes = 0
        self._reiniciar_exploracion()

        entorno = gym.make(env_id)
        self.state_dim = int(entorno.observation_space.shape[0])
        self.action_dim = int(entorno.action_space.n)
        entorno.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    def _reiniciar_exploracion(self) -> None:
        """Evita que un episodio herede una acción pendiente del anterior."""
        self._accion_exploratoria = 0
        self._pasos_restantes = 0

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Elige una acción de la red o inicia un bloque de exploración."""
        if not deterministic:
            if self._pasos_restantes > 0:
                self._pasos_restantes -= 1
                return self._accion_exploratoria

            if random.random() < self.epsilon:
                self._accion_exploratoria = random.randrange(self.action_dim)
                # Este primer paso ya cuenta dentro del bloque.
                self._pasos_restantes = self.pasos_exploracion - 1
                return self._accion_exploratoria

        # La evaluación ignora tanto epsilon como cualquier bloque pendiente.
        with torch.no_grad():
            estado = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return int(self.q_net(estado).argmax(dim=1).item())

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    def _tensor(self, valores, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.array(valores), dtype=dtype, device=self.device)

    def _learn(self) -> float:
        """Actualiza la red principal a partir de un lote de experiencias."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        lote = self.buffer.sample(self.batch_size)
        estados, acciones, recompensas, siguientes_estados, terminaciones = zip(*lote)

        estados_t = self._tensor(estados)
        acciones_t = self._tensor(acciones, torch.int64).unsqueeze(1)
        recompensas_t = self._tensor(recompensas).unsqueeze(1)
        siguientes_estados_t = self._tensor(siguientes_estados)
        terminaciones_t = self._tensor(terminaciones).unsqueeze(1)

        valores_actuales = self.q_net(estados_t)
        q_actual = valores_actuales.gather(1, acciones_t)

        # La red objetivo aporta una referencia que no cambia en cada paso.
        with torch.no_grad():
            valores_siguientes = self.target_net(siguientes_estados_t)
            q_siguiente = valores_siguientes.max(dim=1, keepdim=True).values
            q_objetivo = recompensas_t + self.gamma * q_siguiente * (
                1.0 - terminaciones_t
            )

        perdida = self.loss_fn(q_actual, q_objetivo)
        self.optimizer.zero_grad()
        perdida.backward()
        self.optimizer.step()
        return perdida.item()

    def train(
        self,
        total_episodes: int = 500,
        log_interval: int = 10,
        *,
        semilla: int | None = None,
    ) -> list[float]:
        """Entrena y devuelve la recompensa total de cada episodio.

        La semilla opcional controla las posiciones iniciales del entorno.
        Para repetir un experimento también se deben fijar las semillas de
        random, NumPy y PyTorch antes de crear el agente.
        """
        entorno = gym.make(self.env_id)
        historial_recompensas: list[float] = []

        try:
            for episodio in range(1, total_episodes + 1):
                semilla_episodio = (
                    None if semilla is None else semilla + self.training_episodes
                )
                observacion, _ = entorno.reset(seed=semilla_episodio)
                self._reiniciar_exploracion()
                recompensa_total = 0.0
                finalizado = False

                while not finalizado:
                    accion = self.select_action(observacion)
                    siguiente_observacion, recompensa, terminado, truncado, _ = (
                        entorno.step(accion)
                    )
                    finalizado = terminado or truncado

                    # Agotar el tiempo detiene el episodio, pero no elimina
                    # el valor futuro del estado en la actualización de Bellman.
                    self.buffer.push(
                        observacion,
                        accion,
                        float(recompensa),
                        siguiente_observacion,
                        terminado,
                    )
                    self._learn()
                    observacion = siguiente_observacion
                    recompensa_total += float(recompensa)

                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                self.training_episodes += 1
                historial_recompensas.append(recompensa_total)

                if self.training_episodes % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                if episodio % log_interval == 0:
                    promedio = np.mean(historial_recompensas[-log_interval:])
                    print(
                        f"Episodio {episodio}/{total_episodes} | "
                        f"Recompensa media: {promedio:.2f} | "
                        f"Epsilon: {self.epsilon:.4f} | "
                        f"Experiencias: {len(self.buffer)}"
                    )
        finally:
            entorno.close()

        return historial_recompensas

    _HPARAMS = (
        "env_id",
        "lr",
        "gamma",
        "epsilon_end",
        "epsilon_decay",
        "batch_size",
        "buffer_capacity",
        "target_update_freq",
        "hidden",
        "pasos_exploracion",
    )

    def save(self, path: Path) -> None:
        """Guarda los pesos, el optimizador y la configuración del agente."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        datos = {clave: getattr(self, clave) for clave in self._HPARAMS}
        datos["q_net_state"] = self.q_net.state_dict()
        datos["optimizer_state"] = self.optimizer.state_dict()
        datos["epsilon"] = self.epsilon
        datos["training_episodes"] = self.training_episodes
        torch.save(datos, path)
        print(f"Agente DQN guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Carga también los modelos anteriores al cambio de exploración."""
        datos = torch.load(path, map_location="cpu", weights_only=True)
        configuracion = {
            clave: datos[clave]
            for clave in cls._HPARAMS
            if clave != "env_id" and clave in datos
        }
        # Un modelo antiguo conserva la exploración independiente que usaba.
        configuracion.setdefault("pasos_exploracion", 1)
        agente = cls(datos["env_id"], epsilon_start=datos["epsilon"], **configuracion)
        agente.q_net.load_state_dict(datos["q_net_state"])
        agente.target_net.load_state_dict(datos["q_net_state"])
        agente.optimizer.load_state_dict(datos["optimizer_state"])
        agente.training_episodes = datos["training_episodes"]
        return agente

    def info(self) -> str:
        parametros = sum(parametro.numel() for parametro in self.q_net.parameters())
        return (
            f"Agente DQN para {self.env_id}\n"
            f"  Episodios entrenados : {self.training_episodes}\n"
            f"  Parámetros de la red : {parametros:,}\n"
            f"  Epsilon              : {self.epsilon:.4f}\n"
            f"  Tasa de aprendizaje  : {self.lr}\n"
            f"  Factor de descuento  : {self.gamma}\n"
            f"  Tamaño del lote      : {self.batch_size}\n"
            f"  Actualización objetivo: cada {self.target_update_freq} episodios\n"
            f"  Pasos de exploración : {self.pasos_exploracion}\n"
            f"  Dispositivo          : {self.device}"
        )
