"""!
@file punto1.py
@brief Punto 1 (60 puntos): aprendizaje supervisado con regresión logística en PyTorch.

Contiene los cinco apartados del punto 1:
- 1.1 `forward` y 1.2 `train_logistic_regression` (más `predict_proba` y `predict`).
- 1.3 datos sintéticos linealmente separables y sus gráficos (datos, curva de
  pérdida, evolución de la frontera y mapa de probabilidad).
- 1.4 extracción de características con el codificador, calibración de la tasa
  de aprendizaje y las épocas, y gráficos de las 30 corridas (métricas por
  corrida y matriz de confusión promedio).
- 1.5 reducción de dimensionalidad con PCA y sus gráficos.

Todas las figuras usan el mismo estilo: azul para la clase 0 (ham) y naranja
para la clase 1 (spam), con marcadores distintos por clase, ejes discretos y
un subtítulo que explica cómo leer la figura.

Contrato: `X` es un tensor `float32` de forma (n, D) e `y` contiene etiquetas
0/1 `int64` de forma (n,), con spam = 1. Este archivo no importa ningún otro
archivo del proyecto; el notebook le entrega los datos y combina los
resultados. Las dependencias opcionales del codificador se importan dentro de
`load_encoder`.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

## Frontera verdadera de los datos sintéticos (1.3): 0.6 x1 + 0.8 x2 - 0.5 = 0.
## El vector normal (0.6, 0.8) es unitario, así que w* · x + b* es la distancia con signo.
SYNTHETIC_TRUE_WEIGHTS = (0.6, 0.8)
SYNTHETIC_TRUE_BIAS = -0.5

# Estilo de las figuras: azul y naranja se distinguen también con daltonismo, y cada
# clase lleva además su propio marcador para no depender solo del color.
_CLASS_COLORS = ("#2a78d6", "#eb6834")
_CLASS_MARKERS = ("o", "^")
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#ffffff"
# Rampa ordinal de gris para mostrar el avance de las épocas (claro = temprano).
_EPOCH_RAMP = ("#c3c2b7", "#a3a29b", "#898781", "#52514e", "#0b0b0b")
# Rampa secuencial azul para magnitudes (porcentajes, puntajes).
_BLUE_RAMP = ("#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95")
# La frontera verdadera de 1.3 se dibuja discontinua, para distinguirla sin depender del color.
_TRUE_BOUNDARY_STYLE = {"color": _INK_SECONDARY, "linewidth": 1.5, "linestyle": "--"}


def _generator(seed):
    """!
    @brief Crea un generador de PyTorch reproducible.
    @param seed Semilla entera; `None` usa una semilla no determinista.
    @return torch.Generator listo para usar.
    """
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def _style_axis(axis, grid=True):
    """!
    @brief Aplica el estilo común: ejes y cuadrícula discretos, texto en tinta neutra.
    @param axis Eje de Matplotlib.
    @param grid Si es verdadero, dibuja una cuadrícula fina detrás de los datos.
    @return None.
    """
    axis.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_AXIS)
    axis.tick_params(colors=_INK_MUTED, labelcolor=_INK_SECONDARY, labelsize=9)
    axis.xaxis.label.set_color(_INK_SECONDARY)
    axis.yaxis.label.set_color(_INK_SECONDARY)
    if grid:
        axis.grid(color=_GRID, linewidth=0.8, linestyle="-")
        axis.set_axisbelow(True)


def _set_titles(figure, axis, title, subtitle=None):
    """!
    @brief Escribe el título de la figura y, encima del eje, un subtítulo explicativo.

    El título va como `suptitle`, que es lo que `utils.show_figure` usa para
    nombrar el PNG exportado.

    @param figure Figura de Matplotlib.
    @param axis Eje principal.
    @param title Título de la figura.
    @param subtitle Texto opcional que explica cómo leer la figura.
    @return None.
    """
    figure.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold", color=_INK)
    if subtitle:
        axis.set_title(subtitle, loc="left", fontsize=9, color=_INK_SECONDARY, pad=8)
    figure.patch.set_facecolor(_SURFACE)


def _points_as_numpy(X, y):
    """!
    @brief Convierte puntos y etiquetas a arreglos de NumPy para graficar.
    @param X Tensor de forma (n, 2).
    @param y Etiquetas 0/1 de forma (n,).
    @return Tupla `(X, y)` de arreglos de NumPy.
    """
    X = torch.as_tensor(X).detach().cpu().numpy()
    y = torch.as_tensor(y).detach().cpu().numpy().astype(int)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Las gráficas de 1.3 requieren puntos de forma (n, 2).")
    return X, y


def _padded_limits(X, padding=0.6):
    """!
    @brief Calcula límites cuadrados con margen alrededor de los puntos.
    @param X Arreglo de forma (n, 2).
    @param padding Margen agregado a cada lado.
    @return Tupla `((xmin, xmax), (ymin, ymax))`.
    """
    low = X.min(axis=0) - padding
    high = X.max(axis=0) + padding
    return (low[0], high[0]), (low[1], high[1])


def _scatter_classes(axis, X, y, class_names, alpha=1.0):
    """!
    @brief Dibuja los puntos de cada clase con su color y marcador, con un anillo blanco.
    @param axis Eje de Matplotlib.
    @param X Arreglo de forma (n, 2).
    @param y Etiquetas 0/1.
    @param class_names Nombres de las clases 0 y 1 para la leyenda.
    @param alpha Opacidad de los puntos.
    @return None.
    """
    for label in (0, 1):
        points = X[y == label]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=34,
            marker=_CLASS_MARKERS[label],
            color=_CLASS_COLORS[label],
            edgecolors=_SURFACE,
            linewidths=0.8,
            alpha=alpha,
            label=f"{class_names[label]} ({len(points)})",
            zorder=3,
        )


def _draw_boundary(axis, weights, bias, limits, **line_style):
    """!
    @brief Dibuja la recta w1 x1 + w2 x2 + b = 0 dentro de los límites dados.
    @param axis Eje de Matplotlib.
    @param weights Pesos (w1, w2).
    @param bias Sesgo b.
    @param limits Límites `((xmin, xmax), (ymin, ymax))`.
    @param line_style Argumentos de estilo de `axis.plot`.
    @return None.
    """
    w1, w2 = (float(value) for value in torch.as_tensor(weights).flatten())
    bias = float(bias)
    (xmin, xmax), (ymin, ymax) = limits
    if abs(w2) >= abs(w1):
        x1 = np.linspace(xmin, xmax, 2)
        x2 = -(w1 * x1 + bias) / w2
    else:
        x2 = np.linspace(ymin, ymax, 2)
        x1 = -(w2 * x2 + bias) / w1
    axis.plot(x1, x2, **line_style)


def _finish_plane(axis, limits):
    """!
    @brief Fija límites, proporción 1:1 y etiquetas de un gráfico en el plano.
    @param axis Eje de Matplotlib.
    @param limits Límites `((xmin, xmax), (ymin, ymax))`.
    @return None.
    """
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal")
    axis.set_xlabel("$x_1$")
    axis.set_ylabel("$x_2$")


def _legend(axis, handles=None, **options):
    """!
    @brief Dibuja una leyenda discreta con texto en tinta neutra.
    @param axis Eje de Matplotlib.
    @param handles Elementos opcionales de la leyenda.
    @param options Argumentos adicionales de `axis.legend`.
    @return Leyenda creada.
    """
    legend = axis.legend(handles=handles, frameon=True, fontsize=9, **options)
    legend.get_frame().set_edgecolor(_GRID)
    legend.get_frame().set_facecolor(_SURFACE)
    for text in legend.get_texts():
        text.set_color(_INK_SECONDARY)
    return legend


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

    n_samples, n_features = X.shape
    weights = 0.01 * torch.randn(n_features, generator=_generator(seed), dtype=X.dtype)
    weights = weights.to(X.device)
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

    La frontera verdadera es fija: w* = `SYNTHETIC_TRUE_WEIGHTS` (vector unitario)
    y b* = `SYNTHETIC_TRUE_BIAS`. Cada punto se coloca a una distancia con signo
    d de esa recta, con |d| = margin + noise * |z| y z normal estándar, y a una
    posición normal (desviación 2) a lo largo de ella. La clase 1 queda del lado
    d > 0 y la clase 0 del lado d < 0, por lo que entre las clases hay una franja
    vacía de ancho 2 * margin y los datos son separables por construcción.

    @param n_samples Cantidad total de puntos; la mitad (redondeada) por clase.
    @param margin Distancia mínima de cada punto a la frontera verdadera.
    @param noise Escala de la distancia adicional a la frontera, más allá del margen.
    @param seed Semilla del generador.
    @return Tupla `(X, y)`: `X` de forma (n, 2) en `float32` e `y` de forma (n,) en `int64`.
    """
    generator = _generator(seed)
    normal = torch.tensor(SYNTHETIC_TRUE_WEIGHTS)
    along = torch.stack((-normal[1], normal[0]))

    y = (torch.arange(n_samples) % 2)[torch.randperm(n_samples, generator=generator)]
    sign = 2.0 * y - 1.0
    distance = sign * (margin + noise * torch.randn(n_samples, generator=generator).abs())
    position = 2.0 * torch.randn(n_samples, generator=generator)
    # Con w* unitario, w* · X + b* = distance para cada punto.
    X = (distance - SYNTHETIC_TRUE_BIAS)[:, None] * normal + position[:, None] * along
    return X.to(torch.float32), y.to(torch.int64)


def describe_boundary(model, true_boundary):
    """!
    @brief Compara la frontera aprendida con la verdadera.

    La recta w · x + b = 0 no cambia si se multiplican w y b por una constante,
    así que se comparan normalizados por ||w||: el ángulo entre los vectores
    normales mide si la orientación es correcta y ||w|| mide qué tan segura es
    la transición de probabilidad alrededor de la frontera.

    @param model LogisticRegressionModel entrenado sobre datos en 2D.
    @param true_boundary Tupla `(weights, bias)` de la frontera verdadera.
    @return Diccionario con `w1`, `w2` y `b` normalizados, el ángulo en grados y `||w||`.
    """
    true_weights = torch.as_tensor(true_boundary[0], dtype=torch.float64).flatten()
    weights = model.weights.detach().to(torch.float64).flatten().cpu()
    norm = weights.norm()
    # atan2(|w x w*|, w · w*) es preciso también para ángulos muy pequeños, a diferencia de arccos.
    cross = weights[0] * true_weights[1] - weights[1] * true_weights[0]
    angle = torch.rad2deg(torch.atan2(cross.abs(), weights @ true_weights))
    return {
        "w1": float(weights[0] / norm),
        "w2": float(weights[1] / norm),
        "b": float(model.bias.detach().cpu() / norm),
        "ángulo con la verdadera (°)": float(angle),
        "||w||": float(norm),
    }


def plot_dataset(
    X,
    y,
    title="Datos sintéticos linealmente separables",
    true_boundary=None,
    class_names=("clase 0", "clase 1"),
):
    """!
    @brief Grafica los puntos de cada clase y, opcionalmente, la frontera verdadera.
    @param X Tensor de forma (n, 2).
    @param y Etiquetas 0/1 de forma (n,).
    @param title Título de la figura (también define el nombre del PNG exportado).
    @param true_boundary Tupla opcional `(weights, bias)` de la frontera verdadera.
    @param class_names Nombres de las clases 0 y 1 para la leyenda.
    @return Figura de Matplotlib (sin mostrar).
    """
    X, y = _points_as_numpy(X, y)
    limits = _padded_limits(X)
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    _style_axis(axis)
    _scatter_classes(axis, X, y, class_names)
    if true_boundary is not None:
        _draw_boundary(
            axis, *true_boundary, limits, label="frontera verdadera", zorder=2, **_TRUE_BOUNDARY_STYLE
        )
    _finish_plane(axis, limits)
    _set_titles(figure, axis, title, "Cada punto es una muestra; ninguna cae en la franja central.")
    _legend(axis, loc="upper right")
    figure.tight_layout()
    return figure


def plot_decision_boundary(
    X,
    y,
    predict_proba_fn,
    title="Frontera de decisión",
    grid_points=200,
    true_boundary=None,
    class_names=("clase 0", "clase 1"),
):
    """!
    @brief Grafica los datos sobre el mapa de probabilidad del clasificador y su frontera p = 0.5.

    El fondo va de azul (probabilidad baja de clase 1) a naranja (probabilidad
    alta), pasando por gris en p = 0.5; la línea negra es la frontera aprendida.

    @param X Tensor de forma (n, 2).
    @param y Etiquetas 0/1 de forma (n,).
    @param predict_proba_fn Función que recibe un tensor (m, 2) y devuelve probabilidades (m,).
    @param title Título de la figura.
    @param grid_points Resolución de la malla por eje.
    @param true_boundary Tupla opcional `(weights, bias)` de la frontera verdadera.
    @param class_names Nombres de las clases 0 y 1 para la leyenda.
    @return Figura de Matplotlib (sin mostrar).
    """
    X, y = _points_as_numpy(X, y)
    limits = _padded_limits(X)
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(*limits[0], grid_points), np.linspace(*limits[1], grid_points)
    )
    grid = torch.as_tensor(np.column_stack((grid_x1.ravel(), grid_x2.ravel())), dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.as_tensor(predict_proba_fn(grid)).cpu().numpy().reshape(grid_x1.shape)

    colormap = LinearSegmentedColormap.from_list(
        "probabilidad", [_CLASS_COLORS[0], "#cde2fb", "#f0efec", "#f8d2c2", _CLASS_COLORS[1]]
    )
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    _style_axis(axis, grid=False)
    background = axis.contourf(
        grid_x1, grid_x2, probabilities, levels=np.linspace(0, 1, 11), cmap=colormap, alpha=0.45
    )
    axis.contour(grid_x1, grid_x2, probabilities, levels=[0.5], colors=_INK, linewidths=2, zorder=2)
    if true_boundary is not None:
        _draw_boundary(axis, *true_boundary, limits, zorder=2, **_TRUE_BOUNDARY_STYLE)
    _scatter_classes(axis, X, y, class_names)
    _finish_plane(axis, limits)

    colorbar = figure.colorbar(background, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(f"probabilidad estimada de {class_names[1]}", color=_INK_SECONDARY)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.ax.tick_params(colors=_INK_MUTED, labelcolor=_INK_SECONDARY, labelsize=9)
    colorbar.outline.set_edgecolor(_AXIS)

    handles, _ = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], color=_INK, linewidth=2, label="frontera aprendida (p = 0.5)"))
    if true_boundary is not None:
        handles.append(Line2D([], [], label="frontera verdadera", **_TRUE_BOUNDARY_STYLE))
    subtitle = "Color de fondo: probabilidad que asigna el modelo a cada punto del plano."
    _set_titles(figure, axis, title, subtitle)
    _legend(axis, handles=handles, loc="upper right")
    figure.tight_layout()
    return figure


def plot_boundary_evolution(
    X,
    y,
    snapshots,
    title="Evolución de la frontera durante el entrenamiento",
    true_boundary=None,
    class_names=("clase 0", "clase 1"),
):
    """!
    @brief Grafica la frontera aprendida después de distintas cantidades de épocas.

    Las líneas van de gris claro (pocas épocas) a negro (entrenamiento completo),
    para mostrar cómo el descenso del gradiente rota y desplaza la frontera hasta
    separar las clases.

    @param X Tensor de forma (n, 2).
    @param y Etiquetas 0/1 de forma (n,).
    @param snapshots Diccionario `{épocas: LogisticRegressionModel}`, como mucho cinco entradas.
    @param title Título de la figura.
    @param true_boundary Tupla opcional `(weights, bias)` de la frontera verdadera.
    @param class_names Nombres de las clases 0 y 1 para la leyenda.
    @return Figura de Matplotlib (sin mostrar).
    """
    if not 1 <= len(snapshots) <= len(_EPOCH_RAMP):
        raise ValueError(f"Se admiten entre 1 y {len(_EPOCH_RAMP)} instantáneas.")
    X, y = _points_as_numpy(X, y)
    limits = _padded_limits(X)
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    _style_axis(axis)
    _scatter_classes(axis, X, y, class_names, alpha=0.35)

    ramp = _EPOCH_RAMP[len(_EPOCH_RAMP) - len(snapshots):]
    handles = []
    for color, (epochs, model) in zip(ramp, sorted(snapshots.items())):
        _draw_boundary(axis, model.weights, model.bias, limits, color=color, linewidth=2, zorder=4)
        handles.append(Line2D([], [], color=color, linewidth=2, label=f"época {epochs}"))
    if true_boundary is not None:
        _draw_boundary(axis, *true_boundary, limits, zorder=4, **_TRUE_BOUNDARY_STYLE)
        handles.append(Line2D([], [], label="frontera verdadera", **_TRUE_BOUNDARY_STYLE))
    _finish_plane(axis, limits)
    subtitle = "Cada línea es la frontera p = 0.5 tras esa cantidad de épocas."
    _set_titles(figure, axis, title, subtitle)
    _legend(axis, handles=handles, loc="upper right")
    figure.tight_layout()
    return figure


def plot_loss_history(model, title="Pérdida durante el entrenamiento"):
    """!
    @brief Grafica la curva de aprendizaje: entropía cruzada media en cada época.
    @param model LogisticRegressionModel con `loss_history`.
    @param title Título de la figura.
    @return Figura de Matplotlib (sin mostrar).
    """
    history = np.asarray(model.loss_history, dtype=float)
    epochs = np.arange(len(history))
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    _style_axis(axis)
    axis.plot(epochs, history, color=_CLASS_COLORS[0], linewidth=2, solid_capstyle="round")
    for index, label in ((0, "inicial"), (len(history) - 1, "final")):
        axis.scatter(
            epochs[index], history[index], s=40, color=_CLASS_COLORS[0], edgecolors=_SURFACE,
            linewidths=1.5, zorder=3,
        )
        axis.annotate(
            f"{label}: {history[index]:.3f}",
            (epochs[index], history[index]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=_INK,
            ha="left" if index == 0 else "right",
        )
    axis.set_xlim(0, max(1, epochs[-1]))
    axis.set_ylim(0, history.max() * 1.12)
    axis.set_xlabel("época")
    axis.set_ylabel("entropía cruzada media")
    subtitle = "Más baja es mejor: el modelo asigna más probabilidad a la clase correcta."
    _set_titles(figure, axis, title, subtitle)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 1.4: extracción de características con el codificador
# ---------------------------------------------------------------------------


def load_encoder(model_name, device="cpu"):
    """!
    @brief Carga un codificador de oraciones de sentence-transformers.

    Decisión del equipo (19 de octubre de 2026): `sentence-transformers/all-MiniLM-L6-v2`,
    que produce vectores de 384 dimensiones de norma 1 y corre rápido en CPU.
    La primera llamada descarga el modelo a la caché de Hugging Face.

    @param model_name Identificador del modelo (lo define `ENCODER_NAME` en el notebook).
    @param device Dispositivo de PyTorch donde se ejecuta el codificador.
    @return Objeto `SentenceTransformer` listo para `extract_features`.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def extract_features(texts, encoder, batch_size=64):
    """!
    @brief Codifica cada mensaje en un vector de características.
    @param texts Lista de mensajes sin preprocesar.
    @param encoder Codificador devuelto por `load_encoder` (cualquier objeto con el método
           `encode` de sentence-transformers).
    @param batch_size Cantidad de mensajes codificados por lote.
    @return Tensor `float32` de forma (n, D), en el mismo orden que `texts`.
    """
    with torch.no_grad():
        embeddings = encoder.encode(
            list(texts), batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
        )
    return torch.as_tensor(embeddings).detach().to(torch.float32).cpu()


def grid_search_logistic_regression(
    X_fit, y_fit, X_val, y_val, learning_rates, epoch_values, score_fn, seed=None
):
    """!
    @brief Calibra la tasa de aprendizaje y las épocas con un conjunto de validación.

    Entrena una regresión logística por cada combinación y la evalúa con
    `score_fn` sobre la validación. La validación debe salir solo de los datos
    de entrenamiento, nunca de la prueba. Como el F1 de una validación pequeña
    cambia a saltos (cada mensaje mal clasificado lo mueve un escalón), muchas
    combinaciones pueden empatar; el empate se resuelve con la entropía
    cruzada en la validación, que mide además qué tan bien calibradas están
    las probabilidades.

    @param X_fit Características de ajuste, forma (n, D).
    @param y_fit Etiquetas 0/1 de ajuste.
    @param X_val Características de validación, forma (m, D).
    @param y_val Etiquetas 0/1 de validación.
    @param learning_rates Tasas de aprendizaje que se prueban.
    @param epoch_values Cantidades de épocas que se prueban.
    @param score_fn Función `score_fn(y_true, y_pred) -> float`, mayor es mejor (p. ej. F1).
    @param seed Semilla de inicialización, la misma para todas las combinaciones.
    @return DataFrame con las columnas `learning_rate`, `epochs`, `score`,
            `validation_loss` y `train_loss`, ordenado de mejor a peor: mayor puntaje
            y, a igual puntaje, menor pérdida en validación.
    """
    X_val = torch.as_tensor(X_val)
    y_val_float = torch.as_tensor(y_val).to(X_val.dtype)
    rows = []
    for learning_rate in learning_rates:
        for epochs in epoch_values:
            model = train_logistic_regression(X_fit, y_fit, learning_rate, epochs, seed=seed)
            rows.append(
                {
                    "learning_rate": float(learning_rate),
                    "epochs": int(epochs),
                    "score": float(score_fn(y_val, predict(model, X_val))),
                    "validation_loss": _cross_entropy(X_val, y_val_float, model.weights, model.bias),
                    "train_loss": model.loss_history[-1],
                }
            )
    grid = pd.DataFrame(rows).sort_values(["score", "validation_loss"], ascending=[False, True])
    return grid.reset_index(drop=True)


def plot_calibration(
    grid,
    title="Calibración de la regresión logística",
    score_label="F1 en validación",
    color_span=0.05,
):
    """!
    @brief Grafica el puntaje de validación de cada combinación como un mapa de calor.

    Cada celda muestra el puntaje; la celda recuadrada es la combinación elegida
    (la primera fila de `grid`). La escala de color cubre solo los puntajes
    cercanos al mejor (hasta `color_span` por debajo), para que se distingan las
    diferencias que deciden la elección; los puntajes menores quedan en el tono
    más claro y la barra de color lo indica con una flecha.

    @param grid DataFrame devuelto por `grid_search_logistic_regression`.
    @param title Título de la figura.
    @param score_label Nombre del puntaje para la barra de color.
    @param color_span Ancho del rango de puntajes que cubre la escala de color.
    @return Figura de Matplotlib (sin mostrar).
    """
    table = grid.pivot(index="learning_rate", columns="epochs", values="score").sort_index()
    values = table.to_numpy()
    high = np.nanmax(values)
    low = max(np.nanmin(values), high - color_span)
    colormap = LinearSegmentedColormap.from_list("puntaje", _BLUE_RAMP)
    figure, axis = plt.subplots(figsize=(1.3 * table.shape[1] + 2.6, 0.62 * table.shape[0] + 1.9))
    _style_axis(axis, grid=False)
    image = axis.imshow(values, cmap=colormap, aspect="auto", vmin=low, vmax=high)

    # Texto blanco solo sobre los tonos oscuros de la rampa, para que se lea en todas las celdas.
    dark_from = low + 0.6 * (high - low)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=9,
                color=_SURFACE if values[row, column] >= dark_from else _INK,
            )
    best = grid.iloc[0]
    best_row = list(table.index).index(best["learning_rate"])
    best_column = list(table.columns).index(best["epochs"])
    # La mejor celda es la más oscura, así que su recuadro va en blanco.
    axis.add_patch(
        plt.Rectangle(
            (best_column - 0.46, best_row - 0.44), 0.92, 0.88, fill=False, edgecolor=_SURFACE,
            linewidth=3,
        )
    )

    axis.set_xticks(range(table.shape[1]), [str(value) for value in table.columns])
    axis.set_yticks(range(table.shape[0]), [f"{value:g}" for value in table.index])
    axis.set_xlabel("épocas")
    axis.set_ylabel("tasa de aprendizaje")
    for side in ("left", "bottom"):
        axis.spines[side].set_visible(False)
    colorbar = figure.colorbar(
        image, ax=axis, fraction=0.046, pad=0.03, extend="min" if low > np.nanmin(values) else "neither"
    )
    colorbar.set_label(score_label, color=_INK_SECONDARY)
    colorbar.ax.tick_params(colors=_INK_MUTED, labelcolor=_INK_SECONDARY, labelsize=9)
    colorbar.outline.set_edgecolor(_AXIS)
    subtitle = (
        f"Recuadro blanco: combinación elegida (tasa {best['learning_rate']:g}, "
        f"{int(best['epochs'])} épocas). El color distingue los puntajes más altos."
    )
    _set_titles(figure, axis, title, subtitle)
    figure.tight_layout()
    return figure


def plot_run_metrics(runs, title="Métricas en las 30 corridas", metrics=None):
    """!
    @brief Grafica el valor de cada métrica en cada corrida, con su media y desviación.

    Cada punto es una corrida (partición distinta). La barra negra marca la
    media y el texto muestra media ± desviación estándar.

    @param runs DataFrame de `utils.repeat_experiment` con una columna por métrica.
    @param title Título de la figura.
    @param metrics Lista opcional de pares `(columna, nombre visible)`; por omisión, la
           tasa de aciertos y el F1-score.
    @return Figura de Matplotlib (sin mostrar).
    """
    metrics = metrics or [("tasa_aciertos", "tasa de aciertos"), ("f1", "F1-score")]
    offsets = np.random.default_rng(0).uniform(-0.12, 0.12, len(runs))
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    _style_axis(axis)
    axis.grid(axis="x", visible=False)
    low, high = 1.0, 0.0
    for position, (column, _) in enumerate(metrics):
        values = runs[column].to_numpy(dtype=float)
        low, high = min(low, values.min()), max(high, values.max())
        axis.scatter(
            position + offsets, values, s=36, color=_CLASS_COLORS[0], alpha=0.8,
            edgecolors=_SURFACE, linewidths=1.2, zorder=3,
        )
        mean, std = values.mean(), values.std(ddof=1)
        axis.plot(
            [position - 0.25, position + 0.25], [mean, mean], color=_INK, linewidth=2.5, zorder=4
        )
        axis.annotate(
            f"media {mean:.3f}\n± {std:.3f}", (position + 0.27, mean), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=9, color=_INK,
        )
    span = max(high - low, 0.01)
    axis.set_ylim(low - 0.25 * span, min(high + 0.25 * span, 1.0 + 0.05 * span))
    axis.set_xlim(-0.5, len(metrics) - 0.2)
    axis.set_xticks(range(len(metrics)), [name for _, name in metrics])
    axis.set_ylabel("valor en la partición de prueba")
    subtitle = f"Cada punto es una de las {len(runs)} corridas; la barra negra es la media."
    _set_titles(figure, axis, title, subtitle)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(runs, title="Matriz de confusión promedio", class_names=("ham", "spam")):
    """!
    @brief Grafica la matriz de confusión promedio de varias corridas.

    Filas: clase real; columnas: clase predicha. Cada celda muestra la cantidad
    media de mensajes por corrida y el porcentaje de su fila; el color sigue a
    ese porcentaje, de modo que la diagonal oscura indica aciertos.

    @param runs DataFrame con las columnas `tn`, `fp`, `fn` y `tp` (una fila por corrida).
    @param title Título de la figura.
    @param class_names Nombres de las clases negativa y positiva.
    @return Figura de Matplotlib (sin mostrar).
    """
    counts = np.array(
        [[runs["tn"].mean(), runs["fp"].mean()], [runs["fn"].mean(), runs["tp"].mean()]]
    )
    percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
    colormap = LinearSegmentedColormap.from_list("porcentaje", _BLUE_RAMP)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    _style_axis(axis, grid=False)
    image = axis.imshow(percentages, cmap=colormap, vmin=0, vmax=100)
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, f"{counts[row, column]:.1f}\n({percentages[row, column]:.1f} %)",
                ha="center", va="center", fontsize=11,
                color=_SURFACE if percentages[row, column] > 55 else _INK,
            )
    axis.set_xticks([0, 1], [f"predicho: {name}" for name in class_names])
    axis.set_yticks([0, 1], [f"real: {name}" for name in class_names])
    for side in ("left", "bottom"):
        axis.spines[side].set_visible(False)
    axis.tick_params(length=0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("% de los mensajes de la clase real", color=_INK_SECONDARY)
    colorbar.ax.tick_params(colors=_INK_MUTED, labelcolor=_INK_SECONDARY, labelsize=9)
    colorbar.outline.set_edgecolor(_AXIS)
    subtitle = f"Mensajes por corrida en la prueba, promedio de {len(runs)} corridas."
    _set_titles(figure, axis, title, subtitle)
    figure.tight_layout()
    return figure


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
