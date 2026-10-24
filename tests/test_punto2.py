"""!
@file test_punto2.py
@brief Pruebas del punto 2: densidades por histograma y umbral de verosimilitud.
"""

import math

import matplotlib.pyplot as plt
import pytest
import torch

from notebook_loader import notebook as punto2


def test_predict_anomalies_uses_threshold():
    """!
    @brief Comprueba que solo los puntajes menores que el umbral se marcan como spam.
    @return None.
    """
    predictions = punto2.predict_anomalies(torch.tensor([-5.0, -1.0, 0.0]), threshold=-1.0)

    torch.testing.assert_close(predictions, torch.tensor([1, 0, 0]))


def test_histogram_densities_integrate_to_one():
    """!
    @brief (2.1) Cada marginal integra uno y conserva una densidad por intervalo y dimensión.
    @return None.
    """
    X = torch.tensor([[0.0, -2.0], [0.5, -1.0], [1.0, 1.0], [2.0, 2.0]])
    model = punto2.fit_histogram_densities(X, n_bins=3)

    assert model.bin_edges.shape == (2, 4)
    assert model.densities.shape == (2, 3)
    widths = model.bin_edges.diff(dim=1)
    assert torch.all(widths > 0)
    assert torch.all(model.densities >= 0)
    assert torch.isfinite(model.densities).all()
    torch.testing.assert_close(
        (model.densities * widths).sum(dim=1), torch.ones(2), check_dtype=False
    )


def test_histogram_edges_include_the_maximum_and_use_right_bin_at_inner_edges():
    """!
    @brief (2.1) Los bordes internos pertenecen al intervalo derecho y el máximo no se pierde.
    @return None.
    """
    model = punto2.fit_histogram_densities(
        torch.tensor([[0.0], [1.0], [1.0], [2.0]]), n_bins=2
    )

    torch.testing.assert_close(model.bin_edges, torch.tensor([[0.0, 1.0, 2.0]]), check_dtype=False)
    torch.testing.assert_close(model.densities, torch.tensor([[0.25, 0.75]]), check_dtype=False)
    scores = punto2.log_likelihood(model, torch.tensor([[0.0], [1.0], [2.0]]))
    expected = torch.tensor([math.log(0.25), math.log(0.75), math.log(0.75)])
    torch.testing.assert_close(scores, expected, check_dtype=False)


def test_constant_dimensions_have_positive_width_and_finite_scores():
    """!
    @brief (2.1) Las columnas constantes reciben un soporte no nulo e histogramas normalizados.
    @return None.
    """
    values = torch.tensor([0.0, 4.0, -6.0])
    X = values.repeat(6, 1)
    model = punto2.fit_histogram_densities(X, n_bins=4)
    radius = 0.5 * values.abs().clamp_min(1.0)

    torch.testing.assert_close(model.bin_edges[:, 0], values - radius, check_dtype=False)
    torch.testing.assert_close(model.bin_edges[:, -1], values + radius, check_dtype=False)
    widths = model.bin_edges.diff(dim=1)
    assert torch.all(widths > 0)
    torch.testing.assert_close(
        (model.densities * widths).sum(dim=1), torch.ones(3), check_dtype=False
    )
    scores = punto2.log_likelihood(model, X)
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, scores[0].expand_as(scores))


def test_log_likelihood_is_the_sum_of_marginal_log_densities():
    """!
    @brief (2.1) El puntaje suma los logaritmos de las densidades, con una salida por mensaje.
    @return None.
    """
    model = punto2.HistogramDensities(
        bin_edges=torch.tensor([[0.0, 1.0, 2.0], [10.0, 12.0, 14.0]]),
        densities=torch.tensor([[0.75, 0.25], [0.125, 0.375]]),
    )
    X = torch.tensor([[0.2, 10.5], [1.2, 12.5], [2.0, 14.0]])
    scores = punto2.log_likelihood(model, X)
    expected = torch.tensor(
        [math.log(0.75) + math.log(0.125),
         math.log(0.25) + math.log(0.375),
         math.log(0.25) + math.log(0.375)]
    )

    assert scores.shape == (3,)
    torch.testing.assert_close(scores, expected, check_dtype=False)


def test_empty_bins_and_outside_support_use_epsilon_without_clipping():
    """!
    @brief (2.1) Los intervalos vacíos y ambos lados fuera del soporte usan la densidad mínima.
    @return None.
    """
    model = punto2.HistogramDensities(
        bin_edges=torch.tensor([[0.0, 1.0, 2.0]]),
        densities=torch.tensor([[1.0, 0.0]]),
    )
    epsilon = 1e-8
    X = torch.tensor([[0.5], [1.5], [-1.0], [3.0], [1.0], [2.0]])
    scores = punto2.log_likelihood(model, X, epsilon=epsilon)
    expected = torch.tensor([0.0] + [math.log(epsilon)] * 5)

    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, expected, check_dtype=False)
    assert scores[2] < scores[0]


def test_threshold_uses_linear_quantile_and_strict_rule_with_ties():
    """!
    @brief (2.1) El umbral interpola el cuantil; los empates no garantizan la fracción solicitada.
    @return None.
    """
    scores = torch.tensor([-10.0, -4.0, 2.0, 8.0, 14.0])
    threshold = punto2.select_threshold(scores, quantile=0.3)

    assert isinstance(threshold, float)
    assert math.isclose(threshold, -2.8, abs_tol=1e-6)
    assert punto2.select_threshold(scores, quantile=0.0) == -10.0
    assert punto2.select_threshold(scores, quantile=1.0) == 14.0

    tied_scores = torch.tensor([-8.0, -4.0, -4.0, 4.0])
    tied_threshold = punto2.select_threshold(tied_scores, quantile=0.5)
    assert tied_threshold == -4.0
    predictions = punto2.predict_anomalies(tied_scores, tied_threshold)
    torch.testing.assert_close(predictions, torch.tensor([1, 0, 0, 0]))
    assert predictions.float().mean() < 0.5


def test_histogram_estimation_rejects_invalid_data_and_bin_counts():
    """!
    @brief (2.1) Rechaza matrices vacías, mal formadas o no finitas e intervalos inválidos.
    @return None.
    """
    for X in (torch.empty(0, 2), torch.empty(2, 0), torch.ones(3),
              torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])):
        with pytest.raises(ValueError):
            punto2.fit_histogram_densities(X, n_bins=2)
    for n_bins in (0, -1, 1.5):
        with pytest.raises(ValueError):
            punto2.fit_histogram_densities(torch.ones(3, 1), n_bins=n_bins)


def test_threshold_rejects_invalid_scores_and_quantiles():
    """!
    @brief (2.1) El cuantil exige puntajes finitos no vacíos y una fracción entre cero y uno.
    @return None.
    """
    for scores in (torch.empty(0), torch.ones(2, 1),
                   torch.tensor([float("nan")]), torch.tensor([float("inf")])):
        with pytest.raises(ValueError):
            punto2.select_threshold(scores)
    for quantile in (-0.1, 1.1):
        with pytest.raises(ValueError):
            punto2.select_threshold(torch.tensor([-2.0, 1.0]), quantile=quantile)


def test_anomaly_plots_return_figures_with_exportable_titles():
    """!
    @brief (2.1) Las gráficas devuelven figuras con títulos centrados para exportarlas sin colisiones.
    @return None.
    """
    X = torch.tensor([[0.0, 0.0], [0.2, 0.6], [0.8, 0.3], [1.0, 1.0]])
    model = punto2.fit_histogram_densities(X, n_bins=3)
    scores = torch.tensor([-8.0, -7.0, -2.0, -1.0])
    targets = torch.tensor([1, 1, 0, 0])
    figures = []
    try:
        figures.append(punto2.plot_densities(model, max_dimensions=2, title="Densidades de prueba"))
        figures.append(
            punto2.plot_score_distribution(scores, targets, threshold=-5.0, title="Umbral de prueba")
        )
        assert [figure.get_suptitle() for figure in figures] == [
            "Densidades de prueba", "Umbral de prueba"
        ]
    finally:
        for figure in figures:
            plt.close(figure)
