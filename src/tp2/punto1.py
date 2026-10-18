"""!
@file punto1.py
@brief Punto 1 (60 puntos): aprendizaje supervisado con regresión logística en PyTorch.

Contiene los cinco apartados del punto 1:
- 1.1 `forward` y 1.2 `train_logistic_regression` (más `predict_proba` y `predict`).
- 1.3 datos sintéticos linealmente separables y sus gráficos.
- 1.4 extracción de características con el codificador.
- 1.5 reducción de dimensionalidad con PCA y sus gráficos.

Contrato: `X` es un tensor `float32` de forma (n, D) e `y` contiene etiquetas
0/1 `int64` de forma (n,), con spam = 1. Este archivo no importa ningún otro
archivo del proyecto; el notebook le entrega los datos y combina los
resultados. Las dependencias opcionales del codificador se importan dentro de
`load_encoder`.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# 1.1 y 1.2: evaluación y entrenamiento de la regresión logística
# ---------------------------------------------------------------------------


@dataclass
class LogisticRegressionModel:
    """!
    @brief Parámetros entrenados de una regresión logística binaria.
    """

    ## Vector de pesos de forma (D,).
    weights: torch.Tensor
    ## Sesgo escalar (tensor de forma ()).
    bias: torch.Tensor
    ## Entropía cruzada media antes del entrenamiento y después de cada época
    ## (epochs + 1 valores).
    loss_history: list = field(default_factory=list)


def _sigmoid(z):
    """!
    @brief Calcula la sigmoide 1 / (1 + e^(-z)) sin desbordamientos.

    Se usa e^(-|z|), que siempre está en (0, 1]: para z >= 0 se evalúa
    1 / (1 + e^(-z)) y para z < 0 la forma equivalente e^z / (1 + e^z). Así
    ninguna rama produce infinitos, ni en el valor ni en su gradiente.

    @param z Tensor de logits.
    @return Tensor de la misma forma con valores en [0, 1].
    """
    exp_neg_abs = torch.exp(-z.abs())
    return torch.where(z >= 0, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))


def _cross_entropy(X, y, weights, bias):
    """!
    @brief Calcula la entropía cruzada binaria media a partir de los logits.

    Con z = Xw + b, la pérdida de cada muestra es log(1 + e^z) - y z, que es
    igual a -[y log p + (1 - y) log(1 - p)] con p = sigmoide(z), pero no se
    indefine cuando p vale exactamente 0 o 1. El término log(1 + e^z) se
    evalúa como max(z, 0) + log(1 + e^(-|z|)).

    @param X Tensor de forma (n, D).
    @param y Etiquetas 0/1 en punto flotante, forma (n,).
    @param weights Tensor de forma (D,).
    @param bias Tensor escalar.
    @return Pérdida media (float).
    """
    logits = X @ weights + bias
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    return float((softplus - y * logits).mean())


def forward(X, weights, bias):
    """!
    @brief Evalúa el modelo: p(y = 1 | x) = sigmoide(x · w + b).

    Implementación propia y vectorizada: un producto matriz-vector para los
    logits de las n muestras y la sigmoide estable `_sigmoid`.

    @param X Tensor de forma (n, D).
    @param weights Tensor de forma (D,).
    @param bias Tensor escalar.
    @return Tensor de forma (n,) con probabilidades en [0, 1].
    """
    return _sigmoid(X @ weights + bias)


def train_logistic_regression(X, y, learning_rate=0.1, epochs=1000, seed=None):
    """!
    @brief Entrena la regresión logística minimizando la entropía cruzada binaria.

    Implementación propia con descenso del gradiente por lotes completos. En
    cada época se evalúa el modelo con `forward` y se aplica el gradiente
    exacto de la entropía cruzada media:
    dL/dw = X^T (p - y) / n y dL/db = media(p - y).
    Los pesos parten de una normal con desviación 0.01 (generada con `seed`)
    y el sesgo, de cero.

    @param X Tensor de entrenamiento de forma (n, D).
    @param y Etiquetas 0/1 de forma (n,).
    @param learning_rate Tasa de aprendizaje del descenso del gradiente.
    @param epochs Cantidad de épocas (pasadas completas sobre los datos).
    @param seed Semilla para inicializar los parámetros; `None` no fija la semilla.
    @return LogisticRegressionModel con pesos, sesgo e historial de pérdida.
    """
    X = torch.as_tensor(X)
    if not X.is_floating_point():
        X = X.to(torch.float32)
    y = torch.as_tensor(y, device=X.device).to(X.dtype)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError(f"Se esperaba X de forma (n, D) e y de forma (n,); se recibió "
                         f"{tuple(X.shape)} y {tuple(y.shape)}.")
    if learning_rate <= 0 or epochs < 0:
        raise ValueError("learning_rate debe ser positiva y epochs no negativa.")

    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    n_samples, n_features = X.shape
    weights = (0.01 * torch.randn(n_features, generator=generator, dtype=X.dtype)).to(X.device)
    bias = torch.zeros((), dtype=X.dtype, device=X.device)

    loss_history = [_cross_entropy(X, y, weights, bias)]
    for _ in range(epochs):
        error = forward(X, weights, bias) - y
        weights = weights - learning_rate * (X.T @ error) / n_samples
        bias = bias - learning_rate * error.mean()
        loss_history.append(_cross_entropy(X, y, weights, bias))

    return LogisticRegressionModel(weights=weights, bias=bias, loss_history=loss_history)


def predict_proba(model, X):
    """!
    @brief Calcula la probabilidad de spam de cada fila de `X`.
    @param model LogisticRegressionModel entrenado.
    @param X Tensor de forma (n, D).
    @return Tensor de forma (n,) con probabilidades en [0, 1].
    """
    return forward(X, model.weights, model.bias)


def predict(model, X, threshold=0.5):
    """!
    @brief Clasifica cada fila de `X` como ham (0) o spam (1).
    @param model LogisticRegressionModel entrenado.
    @param X Tensor de forma (n, D).
    @param threshold Probabilidad mínima para predecir spam.
    @return Tensor `int64` de forma (n,) con etiquetas 0/1.
    """
    return (predict_proba(model, X) >= threshold).to(torch.int64)


# ---------------------------------------------------------------------------
# 1.3: datos sintéticos linealmente separables
# ---------------------------------------------------------------------------


def make_linearly_separable(n_samples=200, margin=1.0, noise=0.5, seed=None):
    """!
    @brief Genera dos clases separables por una recta en el plano.
    @param n_samples Cantidad total de puntos (aproximadamente la mitad por clase).
    @param margin Distancia mínima de cada punto a la frontera verdadera.
    @param noise Dispersión de los puntos alrededor de su centro.
    @param seed Semilla del generador.
    @return Tupla `(X, y)`: `X` de forma (n, 2) en `float32` e `y` de forma (n,) en `int64`.
    """
    # TODO(punto 1.3): generar los datos con un torch.Generator y garantizar la separabilidad.
    raise NotImplementedError("Punto 1.3: implementar make_linearly_separable.")


def plot_dataset(X, y, title="Datos sintéticos linealmente separables"):
    """!
    @brief Grafica los puntos coloreados por clase.
    @param X Tensor de forma (n, 2).
    @param y Etiquetas 0/1 de forma (n,).
    @param title Título de la figura (también define el nombre del PNG exportado).
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 1.3)
    raise NotImplementedError("Punto 1.3: implementar plot_dataset.")


def plot_decision_boundary(X, y, predict_proba_fn, title="Frontera de decisión", grid_points=200):
    """!
    @brief Grafica los datos junto con la frontera p = 0.5 del clasificador.
    @param X Tensor de forma (n, 2).
    @param y Etiquetas 0/1 de forma (n,).
    @param predict_proba_fn Función que recibe un tensor (m, 2) y devuelve probabilidades (m,).
    @param title Título de la figura.
    @param grid_points Resolución de la malla por eje.
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 1.3)
    raise NotImplementedError("Punto 1.3: implementar plot_decision_boundary.")


def plot_loss_history(model, title="Pérdida durante el entrenamiento"):
    """!
    @brief Grafica la curva de aprendizaje del entrenamiento.
    @param model LogisticRegressionModel con `loss_history`.
    @param title Título de la figura.
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 1.3): graficar la pérdida por época.
    raise NotImplementedError("Punto 1.3: implementar plot_loss_history.")


# ---------------------------------------------------------------------------
# 1.4: extracción de características con el codificador
# ---------------------------------------------------------------------------


def load_encoder(model_name, device="cpu"):
    """!
    @brief Carga el codificador preentrenado que convierte textos en vectores.
    @param model_name Identificador del modelo (lo define `ENCODER_NAME` en el notebook).
    @param device Dispositivo de PyTorch donde se ejecuta el codificador.
    @return Objeto codificador listo para `extract_features`.
    """
    # TODO(punto 1.4): elegir el codificador (p. ej. un modelo de sentence-transformers
    # o BERT de Hugging Face), importarlo aquí y registrar la dependencia en requirements.txt.
    raise NotImplementedError("Punto 1.4: implementar load_encoder.")


def extract_features(texts, encoder, batch_size=64):
    """!
    @brief Codifica cada mensaje en un vector de características.
    @param texts Lista de mensajes sin preprocesar.
    @param encoder Codificador devuelto por `load_encoder`.
    @param batch_size Cantidad de mensajes codificados por lote.
    @return Tensor `float32` de forma (n, D), en el mismo orden que `texts`.
    """
    # TODO(punto 1.4): codificar por lotes, sin gradientes (torch.no_grad).
    raise NotImplementedError("Punto 1.4: implementar extract_features.")


# ---------------------------------------------------------------------------
# 1.5: reducción de dimensionalidad con PCA
# ---------------------------------------------------------------------------


@dataclass
class PCAModel:
    """!
    @brief Resultado del ajuste de un PCA.
    """

    ## Media de los datos de ajuste, forma (D,).
    mean: torch.Tensor
    ## Componentes principales en columnas, forma (D, k).
    components: torch.Tensor
    ## Varianza explicada por cada componente, forma (k,).
    explained_variance: torch.Tensor
    ## Fracción de la varianza total explicada por cada componente, forma (k,).
    explained_variance_ratio: torch.Tensor


def components_for_fraction(n_features, fraction):
    """!
    @brief Traduce un porcentaje de dimensiones (50 %, 25 %) en una cantidad de componentes.
    @param n_features Dimensión original D.
    @param fraction Fracción de D que se conserva, en (0, 1].
    @return Cantidad de componentes, al menos 1.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction debe estar en el intervalo (0, 1].")
    return max(1, round(n_features * fraction))


def fit_pca(X, n_components):
    """!
    @brief Ajusta un PCA sobre `X`.

    En las corridas se ajusta solo con el entrenamiento y se aplica a la
    prueba, para no filtrar información del conjunto de prueba.

    @param X Tensor de forma (n, D).
    @param n_components Cantidad k de componentes que se conservan.
    @return PCAModel con media, componentes y varianza explicada.
    """
    # TODO(punto 1.5): centrar los datos y obtener los componentes (p. ej. torch.linalg.svd).
    raise NotImplementedError("Punto 1.5: implementar fit_pca.")


def transform_pca(model, X):
    """!
    @brief Proyecta `X` sobre los componentes del modelo.
    @param model PCAModel ajustado.
    @param X Tensor de forma (n, D).
    @return Tensor de forma (n, k).
    """
    # TODO(punto 1.5)
    raise NotImplementedError("Punto 1.5: implementar transform_pca.")


def plot_explained_variance(model, title="Varianza explicada por componente"):
    """!
    @brief Grafica la varianza explicada individual y acumulada.
    @param model PCAModel ajustado.
    @param title Título de la figura.
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 1.5)
    raise NotImplementedError("Punto 1.5: implementar plot_explained_variance.")


def plot_pca_projection(X_reduced, y, title="Proyección del dataset con PCA"):
    """!
    @brief Grafica el dataset proyectado (dos primeras componentes), coloreado por clase.
    @param X_reduced Tensor de forma (n, k) con k >= 2.
    @param y Etiquetas 0/1 de forma (n,).
    @param title Título de la figura.
    @return Figura de Matplotlib (sin mostrar).
    """
    # TODO(punto 1.5): decidir cómo graficar los casos de 50 % y 25 % (p. ej. pares de componentes).
    raise NotImplementedError("Punto 1.5: implementar plot_pca_projection.")
