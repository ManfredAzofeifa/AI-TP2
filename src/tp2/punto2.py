"""!
@file punto2.py
@brief Punto 2 (30 puntos): detección de anomalías con densidades por histograma y umbral de verosimilitud.

Cada dimensión d se modela con un histograma p_d. El puntaje de un mensaje es
la suma de log p_d(x_d) sobre las D dimensiones; un puntaje menor que el
umbral marca el mensaje como anomalía (spam = 1).

Contrato: `X` es un tensor `float32` de forma (n, D), normalmente ya reducido
con el PCA del punto 1 a la D óptima. Este archivo no importa ningún otro
archivo del proyecto: el notebook le entrega los datos ya reducidos.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch


@dataclass
class HistogramDensities:
    """!
    @brief Histogramas normalizados, uno por dimensión.
    """

    ## Bordes de los intervalos de cada dimensión, forma (D, B + 1).
    bin_edges: torch.Tensor
    ## Densidad de cada intervalo (integra 1 en cada dimensión), forma (D, B).
    densities: torch.Tensor


# ---------------------------------------------------------------------------
# 2.1: estimación de densidades y umbral
# ---------------------------------------------------------------------------


def fit_histogram_densities(X_train, n_bins=30):
    """!
    @brief Estima la densidad de cada dimensión con un histograma.
    @param X_train Tensor de entrenamiento (n, D). Decisión del equipo: usar solo mensajes
           ham (distribución normal) o todo el entrenamiento; documentarla en el informe.
    @param n_bins Cantidad de intervalos B por dimensión.
    @return HistogramDensities ajustado.
    """
    # TODO(punto 2.1): construir un histograma normalizado por dimensión (p. ej. torch.histogram).
    raise NotImplementedError("Punto 2.1: implementar fit_histogram_densities.")


def log_likelihood(model, X, epsilon=1e-12):
    """!
    @brief Calcula el puntaje de verosimilitud sumado sobre todas las dimensiones.
    @param model HistogramDensities ajustado.
    @param X Tensor de forma (n, D).
    @param epsilon Densidad mínima para evitar log(0) fuera del rango de los histogramas.
    @return Tensor de forma (n,) con la suma de log p_d(x_d).
    """
    # TODO(punto 2.1)
    raise NotImplementedError("Punto 2.1: implementar log_likelihood.")


def select_threshold(scores, quantile=0.05):
    """!
    @brief Define el umbral de verosimilitud a partir de puntajes de entrenamiento.
    @param scores Puntajes de `log_likelihood` sobre los datos de entrenamiento.
    @param quantile Fracción de datos de entrenamiento que se acepta marcar como anomalía.
    @return Umbral escalar (float).
    """
    # TODO(punto 2.1): justificar el criterio del umbral en el informe.
    raise NotImplementedError("Punto 2.1: implementar select_threshold.")


def predict_anomalies(scores, threshold):
    """!
    @brief Marca como anomalía (spam = 1) cada puntaje por debajo del umbral.
    @param scores Tensor de forma (n,) con puntajes de verosimilitud.
    @param threshold Umbral devuelto por `select_threshold`.
    @return Tensor `int64` de forma (n,) con etiquetas 0/1.
    """
    return (torch.as_tensor(scores) < threshold).to(torch.int64)


# ---------------------------------------------------------------------------
# 2.1: gráficos de las densidades y de los puntajes
# ---------------------------------------------------------------------------


def plot_densities(model, max_dimensions=6, title="Densidades estimadas por dimensión"):
    """!
    @brief Grafica los histogramas de las primeras dimensiones.
    @param model HistogramDensities ajustado.
    @param max_dimensions Cantidad máxima de dimensiones que se grafican.
    @param title Título de la figura (también define el nombre del PNG exportado).
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 2.1)
    raise NotImplementedError("Punto 2.1: implementar plot_densities.")


def plot_score_distribution(scores, y, threshold, title="Puntajes de verosimilitud por clase"):
    """!
    @brief Grafica la distribución de puntajes de ham y spam junto con el umbral.
    @param scores Tensor de forma (n,) con puntajes.
    @param y Etiquetas 0/1 de forma (n,).
    @param threshold Umbral de decisión.
    @param title Título de la figura.
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 2.1)
    raise NotImplementedError("Punto 2.1: implementar plot_score_distribution.")
