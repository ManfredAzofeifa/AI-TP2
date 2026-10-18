"""!
@file test_punto1.py
@brief Pruebas del punto 1: regresión logística, datos sintéticos y PCA.

Las pruebas 1.1a y 1.2a son obligatorias (al menos dos para `forward` y dos
para `train_logistic_regression`). Las pruebas que todavía lanzan
`NotImplementedError` se reportan como pendientes.
"""

import math

import torch
import torch.nn.functional as F

from tp2 import punto1


def _noisy_dataset():
    """!
    @brief Genera un conjunto reproducible de 200 puntos en 3D con clases que se traslapan.
    @return Tupla `(X, y)` con `X` de forma (200, 3) e `y` de forma (200,) en `int64`.
    """
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(200, 3, generator=generator)
    noise = 0.5 * torch.randn(200, generator=generator)
    y = (X @ torch.tensor([1.0, -2.0, 0.5]) + noise > 0).to(torch.int64)
    return X, y


# ---------------------------------------------------------------------------
# 1.1a: pruebas de forward (obligatorias)
# ---------------------------------------------------------------------------


def test_forward_zero_parameters_returns_one_half():
    """!
    @brief (Punto 1.1a) Con pesos y sesgo nulos, toda probabilidad es 0.5.

    El logit es 0 para cualquier entrada y sigmoide(0) = 0.5. La prueba verifica
    además que la salida tenga una probabilidad por muestra, con forma (n,).

    @return None.
    """
    X = torch.randn(5, 3, generator=torch.Generator().manual_seed(1))
    probabilities = punto1.forward(X, torch.zeros(3), torch.tensor(0.0))

    assert probabilities.shape == (5,)
    torch.testing.assert_close(probabilities, torch.full((5,), 0.5))


def test_forward_matches_sigmoid_and_is_stable():
    """!
    @brief (Punto 1.1a) Coincide con la sigmoide calculada a mano y no se desborda.

    Para X = [[1, 2], [0, 0], [-1, -1]], w = (0.5, -1) y b = 0.25 los logits son
    -1.25, 0.25 y 0.75. Con logits de +-1000, la forma directa 1 / (1 + e^(-z))
    desborda e^1000 y su gradiente con Autograd da NaN; la salida debe valer 1 y
    0 y el gradiente respecto de los pesos debe ser finito.

    @return None.
    """
    X = torch.tensor([[1.0, 2.0], [0.0, 0.0], [-1.0, -1.0]])
    probabilities = punto1.forward(X, torch.tensor([0.5, -1.0]), torch.tensor(0.25))
    expected = torch.tensor([1.0 / (1.0 + math.exp(-z)) for z in (-1.25, 0.25, 0.75)])
    torch.testing.assert_close(probabilities, expected)

    weights = torch.tensor([1.0], requires_grad=True)
    extreme = punto1.forward(torch.tensor([[1000.0], [-1000.0]]), weights, torch.tensor(0.0))
    extreme.sum().backward()
    torch.testing.assert_close(extreme.detach(), torch.tensor([1.0, 0.0]))
    assert torch.isfinite(weights.grad).all()


# ---------------------------------------------------------------------------
# 1.2a: pruebas de train_logistic_regression (obligatorias)
# ---------------------------------------------------------------------------


def test_train_follows_cross_entropy_gradient():
    """!
    @brief (Punto 1.2a) El entrenamiento desciende por el gradiente de la entropía cruzada.

    El gradiente de referencia se calcula con Autograd sobre
    `binary_cross_entropy_with_logits` de PyTorch, independiente de la fórmula
    cerrada que usa la implementación. Se verifica que el primer paso sea
    w1 = w0 - alpha * dL/dw (y lo mismo para el sesgo), que la pérdida inicial
    registrada coincida con la de referencia y que la pérdida baje en cada época.

    @return None.
    """
    X, y = _noisy_dataset()
    learning_rate = 0.1
    initial = punto1.train_logistic_regression(X, y, learning_rate, epochs=0, seed=7)
    one_step = punto1.train_logistic_regression(X, y, learning_rate, epochs=1, seed=7)
    trained = punto1.train_logistic_regression(X, y, learning_rate, epochs=50, seed=7)

    weights = initial.weights.clone().requires_grad_(True)
    bias = initial.bias.clone().requires_grad_(True)
    loss = F.binary_cross_entropy_with_logits(X @ weights + bias, y.to(torch.float32))
    loss.backward()

    torch.testing.assert_close(one_step.weights, initial.weights - learning_rate * weights.grad)
    torch.testing.assert_close(one_step.bias, initial.bias - learning_rate * bias.grad)
    assert math.isclose(initial.loss_history[0], loss.item(), rel_tol=1e-5)

    history = torch.tensor(trained.loss_history)
    assert len(history) == 51
    assert (history[1:] <= history[:-1] + 1e-6).all()
    assert history[-1] < history[0]


def test_train_separates_linearly_separable_data():
    """!
    @brief (Punto 1.2a) Clasifica sin errores datos separables y es reproducible.

    Con dos grupos separables por una recta, el modelo entrenado debe acertar en
    todos los puntos. Entrenar dos veces con la misma semilla debe dar
    exactamente los mismos parámetros.

    @return None.
    """
    X = torch.tensor([[-2.0, -1.0], [-1.5, -2.0], [-1.0, -1.5], [1.0, 1.5], [1.5, 2.0], [2.0, 1.0]])
    y = torch.tensor([0, 0, 0, 1, 1, 1])

    model = punto1.train_logistic_regression(X, y, learning_rate=0.5, epochs=200, seed=3)
    repeated = punto1.train_logistic_regression(X, y, learning_rate=0.5, epochs=200, seed=3)

    torch.testing.assert_close(punto1.predict(model, X), y)
    assert torch.equal(model.weights, repeated.weights)
    assert torch.equal(model.bias, repeated.bias)


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
