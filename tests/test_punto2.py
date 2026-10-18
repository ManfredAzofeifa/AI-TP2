"""!
@file test_punto2.py
@brief Pruebas del punto 2: densidades por histograma y umbral de verosimilitud.
"""

import torch

from tp2 import punto2


def test_predict_anomalies_uses_threshold():
    """!
    @brief Comprueba que solo los puntajes menores que el umbral se marcan como spam.
    @return None.
    """
    predictions = punto2.predict_anomalies(torch.tensor([-5.0, -1.0, 0.0]), threshold=-1.0)

    torch.testing.assert_close(predictions, torch.tensor([1, 0, 0]))


def test_histogram_densities_integrate_to_one():
    """!
    @brief (Recomendada, 2.1) La densidad de cada dimensión debe integrar 1 y los puntos
           fuera del rango de entrenamiento deben recibir el puntaje más bajo.
    @return None.
    """
    raise NotImplementedError("Punto 2.1: diseñar la prueba de fit_histogram_densities.")
