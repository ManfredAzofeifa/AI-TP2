"""!
@file test_punto1.py
@brief Pruebas del punto 1: regresión logística, datos sintéticos y PCA.

Las pruebas 1.1a y 1.2a son obligatorias (al menos dos para `forward` y dos
para `train_logistic_regression`). Cada prueba lanza `NotImplementedError`
hasta que el equipo la implemente; mientras tanto se reporta como pendiente.
"""

import torch

from tp2 import punto1


# ---------------------------------------------------------------------------
# 1.1a: pruebas de forward (obligatorias)
# ---------------------------------------------------------------------------


def test_forward_first_case():
    """!
    @brief (Punto 1.1a) Primera prueba de `forward`.

    Idea: con pesos y sesgo nulos, toda probabilidad debe ser 0.5 y la salida
    debe tener forma (n,).

    @return None.
    """
    raise NotImplementedError("Punto 1.1a: diseñar la primera prueba de forward.")


def test_forward_second_case():
    """!
    @brief (Punto 1.1a) Segunda prueba de `forward`.

    Idea: comparar contra la sigmoide calculada a mano para valores conocidos
    y verificar que las probabilidades quedan en [0, 1] para entradas grandes.

    @return None.
    """
    raise NotImplementedError("Punto 1.1a: diseñar la segunda prueba de forward.")


# ---------------------------------------------------------------------------
# 1.2a: pruebas de train_logistic_regression (obligatorias)
# ---------------------------------------------------------------------------


def test_train_first_case():
    """!
    @brief (Punto 1.2a) Primera prueba de `train_logistic_regression`.

    Idea: la pérdida registrada en `loss_history` debe disminuir.

    @return None.
    """
    raise NotImplementedError("Punto 1.2a: diseñar la primera prueba del entrenamiento.")


def test_train_second_case():
    """!
    @brief (Punto 1.2a) Segunda prueba de `train_logistic_regression`.

    Idea: sobre un conjunto pequeño linealmente separable, el modelo entrenado
    debe clasificar correctamente todos los puntos; la misma semilla debe dar
    los mismos parámetros.

    @return None.
    """
    raise NotImplementedError("Punto 1.2a: diseñar la segunda prueba del entrenamiento.")


# ---------------------------------------------------------------------------
# 1.3 y 1.5: pruebas recomendadas
# ---------------------------------------------------------------------------


def test_make_linearly_separable_shapes_and_reproducibility():
    """!
    @brief (Recomendada, 1.3) Verificar formas, tipos, ambas clases y reproducibilidad con semilla.
    @return None.
    """
    raise NotImplementedError("Punto 1.3: diseñar la prueba del generador sintético.")


def test_components_for_fraction():
    """!
    @brief (1.5) Comprueba la traducción de porcentajes de dimensiones a cantidad de componentes.
    @return None.
    """
    assert punto1.components_for_fraction(384, 0.5) == 192
    assert punto1.components_for_fraction(384, 0.25) == 96
    assert punto1.components_for_fraction(3, 0.1) == 1


def test_fit_pca_recovers_dominant_direction():
    """!
    @brief (Recomendada, 1.5) Con datos alargados sobre una recta conocida, la primera
           componente debe ser esa dirección y explicar casi toda la varianza.
    @return None.
    """
    raise NotImplementedError("Punto 1.5: diseñar la prueba de fit_pca.")
