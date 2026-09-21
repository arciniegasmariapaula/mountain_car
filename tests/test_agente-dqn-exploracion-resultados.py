"""Pruebas de la exploración y de la compatibilidad con modelos guardados."""

import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch

from mountain_car.agents.dqn import DQNAgent


@contextmanager
def carpeta_temporal():
    """Crea una carpeta con permisos heredados, también en Windows."""
    raiz = Path(tempfile.gettempdir()).resolve()
    carpeta = raiz / f"prueba_dqn_{uuid4().hex}"
    carpeta.mkdir()
    try:
        yield carpeta
    finally:
        if carpeta.resolve().parent != raiz:
            raise RuntimeError("La carpeta temporal cambió de ubicación.")
        shutil.rmtree(carpeta)


class PruebasExploracionDQN(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.agente = DQNAgent("MountainCar-v0", pasos_exploracion=3)
        self.estado = np.array([-0.5, 0.0], dtype=np.float32)

    def test_repite_la_accion_el_numero_exacto_de_pasos(self):
        with patch("random.randrange", side_effect=[2, 0]) as sorteo:
            acciones = [self.agente.select_action(self.estado) for _ in range(6)]
        self.assertEqual(acciones, [2, 2, 2, 0, 0, 0])
        self.assertEqual(sorteo.call_count, 2)

    def test_evaluacion_ignora_un_bloque_pendiente(self):
        with torch.no_grad():
            for parametro in self.agente.q_net.parameters():
                parametro.zero_()
            self.agente.q_net.layers[-1].bias[2] = 5.0
        self.agente._accion_exploratoria = 0
        self.agente._pasos_restantes = 2
        with patch("random.random", side_effect=AssertionError("No debe explorar")):
            accion = self.agente.select_action(self.estado, deterministic=True)
        self.assertEqual(accion, 2)
        self.assertEqual(self.agente._pasos_restantes, 2)

    def test_un_paso_equivale_a_exploracion_independiente(self):
        self.agente.pasos_exploracion = 1
        with patch("random.randrange", side_effect=[0, 2, 1]):
            acciones = [self.agente.select_action(self.estado) for _ in range(3)]
        self.assertEqual(acciones, [0, 2, 1])

    def test_reinicia_el_bloque_al_cambiar_de_episodio(self):
        estado = self.estado
        contador_al_iniciar = []
        agente = self.agente

        class EntornoCorto:
            def reset(self, seed=None):
                return estado, {}

            def step(self, accion):
                return estado, -1.0, True, False, {}

            def close(self):
                pass

        seleccion_original = agente.select_action

        def observar(observacion):
            contador_al_iniciar.append(agente._pasos_restantes)
            return seleccion_original(observacion)

        with (
            patch("mountain_car.agents.dqn.gym.make", return_value=EntornoCorto()),
            patch.object(agente, "select_action", side_effect=observar),
        ):
            agente.train(total_episodes=2)
        self.assertEqual(contador_al_iniciar, [0, 0])

    def test_guardar_y_cargar_conserva_la_configuracion_y_las_predicciones(self):
        self.agente.training_episodes = 7
        with carpeta_temporal() as temporal:
            ruta = Path(temporal) / "modelo.pt"
            self.agente.save(ruta)
            cargado = DQNAgent.load(ruta)
        self.assertEqual(cargado.pasos_exploracion, 3)
        self.assertEqual(cargado.training_episodes, 7)
        estado = torch.as_tensor(self.estado, device=self.agente.device)
        with torch.no_grad():
            self.assertTrue(
                torch.equal(self.agente.q_net(estado), cargado.q_net(estado))
            )

    def test_modelo_antiguo_conserva_la_exploracion_original(self):
        with carpeta_temporal() as temporal:
            ruta = Path(temporal) / "modelo_antiguo.pt"
            self.agente.save(ruta)
            datos = torch.load(ruta, weights_only=True)
            datos.pop("pasos_exploracion")
            torch.save(datos, ruta)
            cargado = DQNAgent.load(ruta)
        self.assertEqual(cargado.pasos_exploracion, 1)
        self.assertEqual(cargado._pasos_restantes, 0)

    def test_rechaza_duraciones_invalidas(self):
        for valor in [0, -1, 1.5, True]:
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                DQNAgent("MountainCar-v0", pasos_exploracion=valor)


if __name__ == "__main__":
    unittest.main()
