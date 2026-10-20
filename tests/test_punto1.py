"""!
@file test_punto1.py
@brief Pruebas del punto 1: regresión logística, datos sintéticos y PCA.

Las pruebas 1.1a y 1.2a son obligatorias (al menos dos para `forward` y dos
para `train_logistic_regression`). Las pruebas que todavía lanzan
`NotImplementedError` se reportan como pendientes.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
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
# 1.3: datos sintéticos linealmente separables
# ---------------------------------------------------------------------------


def test_make_linearly_separable_shapes_and_reproducibility():
    """!
    @brief (1.3) Verifica formas, tipos, balance de clases, margen y reproducibilidad.

    La distancia con signo w* · x + b* a la frontera verdadera debe ser al menos
    `margin` en valor absoluto y tener el signo de la clase (positiva para la
    clase 1), lo que garantiza que los datos son linealmente separables.

    @return None.
    """
    X, y = punto1.make_linearly_separable(n_samples=101, margin=0.75, noise=0.5, seed=4)
    X_again, y_again = punto1.make_linearly_separable(n_samples=101, margin=0.75, noise=0.5, seed=4)
    X_other, _ = punto1.make_linearly_separable(n_samples=101, margin=0.75, noise=0.5, seed=5)

    assert X.shape == (101, 2) and X.dtype == torch.float32
    assert y.shape == (101,) and y.dtype == torch.int64
    assert sorted(torch.bincount(y).tolist()) == [50, 51]

    distance = X @ torch.tensor(punto1.SYNTHETIC_TRUE_WEIGHTS) + punto1.SYNTHETIC_TRUE_BIAS
    assert (distance.abs() >= 0.75 - 1e-5).all()
    assert torch.equal(distance > 0, y == 1)

    assert torch.equal(X, X_again) and torch.equal(y, y_again)
    assert not torch.equal(X, X_other)


def test_boundary_plots_return_figures():
    """!
    @brief (1.3) Las cuatro gráficas devuelven una figura con el título pedido, sin mostrarla.
    @return None.
    """
    X, y = punto1.make_linearly_separable(n_samples=40, seed=0)
    model = punto1.train_logistic_regression(X, y, learning_rate=0.1, epochs=20, seed=0)
    true_boundary = (torch.tensor(punto1.SYNTHETIC_TRUE_WEIGHTS), punto1.SYNTHETIC_TRUE_BIAS)

    figures = [
        punto1.plot_dataset(X, y, title="A", true_boundary=true_boundary),
        punto1.plot_decision_boundary(
            X, y, lambda grid: punto1.predict_proba(model, grid), title="B", grid_points=30,
            true_boundary=true_boundary,
        ),
        punto1.plot_boundary_evolution(X, y, {0: model, 20: model}, title="C"),
        punto1.plot_loss_history(model, title="D"),
    ]
    try:
        assert [figure.get_suptitle() for figure in figures] == ["A", "B", "C", "D"]
    finally:
        for figure in figures:
            plt.close(figure)


# ---------------------------------------------------------------------------
# 1.4: extracción de características y calibración
# ---------------------------------------------------------------------------


class _FakeEncoder:
    """!
    @brief Codificador de prueba: devuelve (longitud del texto, 1) sin descargar ningún modelo.
    """

    def __init__(self):
        """!
        @brief Inicializa el registro de llamadas.
        @return None.
        """
        self.calls = []

    def encode(self, texts, batch_size, convert_to_tensor, show_progress_bar):
        """!
        @brief Imita `SentenceTransformer.encode` con vectores deterministas en float64.
        @param texts Lista de textos.
        @param batch_size Tamaño de lote solicitado.
        @param convert_to_tensor Si se pidió un tensor.
        @param show_progress_bar Si se pidió barra de progreso.
        @return Tensor de forma (n, 2).
        """
        self.calls.append((list(texts), batch_size, convert_to_tensor, show_progress_bar))
        return torch.tensor([[float(len(text)), 1.0] for text in texts], dtype=torch.float64)


def test_extract_features_keeps_order_and_contract():
    """!
    @brief (1.4) Las características salen en float32, forma (n, D) y en el orden de los textos.
    @return None.
    """
    encoder = _FakeEncoder()
    features = punto1.extract_features(("hola", "ok", "mensaje largo"), encoder, batch_size=2)

    assert features.dtype == torch.float32
    torch.testing.assert_close(features, torch.tensor([[4.0, 1.0], [2.0, 1.0], [13.0, 1.0]]))
    assert encoder.calls == [(["hola", "ok", "mensaje largo"], 2, True, False)]


def test_grid_search_covers_grid_and_orders_by_score():
    """!
    @brief (1.4) La búsqueda prueba todas las combinaciones y deja primero la de mayor puntaje.

    A igual puntaje, debe quedar primero la de menor entropía cruzada en la validación.

    @return None.
    """
    X, y = _noisy_dataset()
    accuracy = lambda y_true, y_pred: float((y_true == y_pred).to(torch.float32).mean())
    grid = punto1.grid_search_logistic_regression(
        X[:150], y[:150], X[150:], y[150:], [0.01, 0.5], [1, 50], accuracy, seed=0
    )

    assert len(grid) == 4
    assert set(zip(grid["learning_rate"], grid["epochs"])) == {(0.01, 1), (0.01, 50), (0.5, 1), (0.5, 50)}
    assert grid["score"].is_monotonic_decreasing
    assert grid.iloc[0]["score"] > 0.8
    for _, ties in grid.groupby("score"):
        assert ties["validation_loss"].is_monotonic_increasing


def test_run_plots_return_figures():
    """!
    @brief (1.4) Las gráficas de calibración, métricas y matriz de confusión devuelven figuras.
    @return None.
    """
    grid = pd.DataFrame(
        {"learning_rate": [1.0, 1.0, 2.0, 2.0], "epochs": [10, 20, 10, 20],
         "score": [0.8, 0.9, 0.7, 0.85], "validation_loss": [0.5, 0.4, 0.6, 0.45],
         "train_loss": [0.4, 0.3, 0.5, 0.35]}
    ).sort_values("score", ascending=False)
    runs = pd.DataFrame(
        {"tasa_aciertos": [0.97, 0.98], "f1": [0.90, 0.92], "tn": [480, 482], "fp": [3, 1],
         "fn": [12, 10], "tp": [60, 62]}
    )
    figures = [
        punto1.plot_calibration(grid, title="E"),
        punto1.plot_run_metrics(runs, title="F"),
        punto1.plot_confusion_matrix(runs, title="G"),
    ]
    try:
        assert [figure.get_suptitle() for figure in figures] == ["E", "F", "G"]
    finally:
        for figure in figures:
            plt.close(figure)


# ---------------------------------------------------------------------------
# 1.5: PCA
# ---------------------------------------------------------------------------


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
