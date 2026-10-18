"""!
@file utils.py
@brief Infraestructura compartida por los tres puntos: datos, métricas, corridas, figuras, tablas y pruebas.

No contiene lógica de ningún punto del enunciado y no importa `punto1`,
`punto2` ni `punto3`. Convención de etiquetas para todo el proyecto:
`ham = 0` (mensaje normal) y `spam = 1` (clase positiva y, en el punto 2,
la anomalía).
"""

import csv
import hashlib
import inspect
import io
import random
import re
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_FILENAME = "SMSSpamCollection"
LABELS = {"ham": 0, "spam": 1}
RUN_COLUMNS = ("corrida", "semilla")

_figures_directory = None
_tables_directory = None


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------


def set_global_seed(seed):
    """!
    @brief Fija las semillas de `random`, NumPy y PyTorch.
    @param seed Semilla entera compartida.
    @return None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Datos: carga del dataset, particiones y muestreos
# ---------------------------------------------------------------------------


def download_sms_spam(directory):
    """!
    @brief Descarga el dataset desde el repositorio UCI si todavía no existe localmente.
    @param directory Carpeta donde se guarda el archivo (por ejemplo `data/raw`).
    @return Ruta del archivo `SMSSpamCollection`.
    """
    directory = Path(directory)
    path = directory / DATASET_FILENAME
    if path.exists():
        return path

    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        path.write_bytes(archive.read(DATASET_FILENAME))
    return path


def load_sms_spam(path):
    """!
    @brief Lee el archivo SMSSpamCollection (una línea por mensaje: etiqueta, tabulador, texto).
    @param path Ruta del archivo `SMSSpamCollection`.
    @return DataFrame con las columnas `label` (texto), `text` (mensaje sin preprocesar)
            y `target` (0 = ham, 1 = spam).
    """
    messages = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        encoding_errors="replace",
        dtype=str,
    )
    messages["target"] = messages["label"].map(LABELS)
    if messages["target"].isna().any():
        unknown = sorted(messages.loc[messages["target"].isna(), "label"].unique())
        raise ValueError(f"Etiquetas desconocidas en el dataset: {unknown}")
    messages["target"] = messages["target"].astype("int64")
    return messages


def stratified_split(targets, train_fraction=0.7, seed=None):
    """!
    @brief Particiona índices en entrenamiento y prueba conservando la proporción de clases.
    @param targets Etiquetas (tensor o secuencia) de forma (n,).
    @param train_fraction Fracción de cada clase que va a entrenamiento, en (0, 1).
    @param seed Semilla de la partición; la misma semilla produce la misma partición.
    @return Tupla `(train_indices, test_indices)` de tensores `int64` disjuntos que
            cubren todos los índices.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction debe estar en el intervalo (0, 1).")

    targets = torch.as_tensor(targets)
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    train_parts, test_parts = [], []
    for label in torch.unique(targets):
        class_indices = torch.nonzero(targets == label, as_tuple=True)[0]
        class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)]
        n_train = round(train_fraction * len(class_indices))
        train_parts.append(class_indices[:n_train])
        test_parts.append(class_indices[n_train:])

    # Se mezclan las clases para que los lotes no queden ordenados por etiqueta.
    train_indices = torch.cat(train_parts)
    test_indices = torch.cat(test_parts)
    train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)]
    test_indices = test_indices[torch.randperm(len(test_indices), generator=generator)]
    return train_indices, test_indices


def sample_indices(targets, fraction=0.3, seed=None):
    """!
    @brief Toma una muestra estratificada sin reemplazo (punto 3: muestras del 30 %).
    @param targets Etiquetas (tensor o secuencia) de forma (n,).
    @param fraction Fracción del dataset que se muestrea, en (0, 1).
    @param seed Semilla del muestreo.
    @return Tensor `int64` con los índices muestreados.
    """
    sampled_indices, _ = stratified_split(targets, train_fraction=fraction, seed=seed)
    return sampled_indices


# ---------------------------------------------------------------------------
# Métricas: tasa de aciertos y F1-score (clase positiva: spam = 1)
# ---------------------------------------------------------------------------


def _as_label_tensors(y_true, y_pred):
    """!
    @brief Convierte etiquetas reales y predichas en tensores enteros planos.
    @param y_true Etiquetas reales (tensor o secuencia).
    @param y_pred Etiquetas predichas (tensor o secuencia).
    @return Tupla de tensores `int64` de forma (n,).
    """
    y_true = torch.as_tensor(y_true).flatten().to(torch.int64)
    y_pred = torch.as_tensor(y_pred).flatten().to(torch.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Formas distintas: {tuple(y_true.shape)} y {tuple(y_pred.shape)}.")
    if y_true.numel() == 0:
        raise ValueError("No hay etiquetas para evaluar.")
    return y_true, y_pred


def confusion_counts(y_true, y_pred, positive_label=1):
    """!
    @brief Cuenta verdaderos y falsos positivos y negativos.
    @param y_true Etiquetas reales.
    @param y_pred Etiquetas predichas.
    @param positive_label Etiqueta considerada positiva (spam).
    @return Diccionario con las llaves `tp`, `fp`, `fn` y `tn`.
    """
    y_true, y_pred = _as_label_tensors(y_true, y_pred)
    true_positive = y_true == positive_label
    predicted_positive = y_pred == positive_label
    return {
        "tp": int((true_positive & predicted_positive).sum()),
        "fp": int((~true_positive & predicted_positive).sum()),
        "fn": int((true_positive & ~predicted_positive).sum()),
        "tn": int((~true_positive & ~predicted_positive).sum()),
    }


def accuracy(y_true, y_pred):
    """!
    @brief Calcula la tasa de aciertos.
    @param y_true Etiquetas reales.
    @param y_pred Etiquetas predichas.
    @return Fracción de predicciones correctas, en [0, 1].
    """
    y_true, y_pred = _as_label_tensors(y_true, y_pred)
    return float((y_true == y_pred).to(torch.float64).mean())


def f1_score(y_true, y_pred, positive_label=1):
    """!
    @brief Calcula el F1-score de la clase positiva.
    @param y_true Etiquetas reales.
    @param y_pred Etiquetas predichas.
    @param positive_label Etiqueta considerada positiva (spam).
    @return Media armónica de precisión y exhaustividad; 0 si no hay verdaderos positivos.
    """
    counts = confusion_counts(y_true, y_pred, positive_label)
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denominator if denominator else 0.0


def classification_metrics(y_true, y_pred):
    """!
    @brief Reúne las métricas que pide el enunciado en un solo diccionario.
    @param y_true Etiquetas reales.
    @param y_pred Etiquetas predichas.
    @return Diccionario con `tasa_aciertos` y `f1`.
    """
    return {"tasa_aciertos": accuracy(y_true, y_pred), "f1": f1_score(y_true, y_pred)}


# ---------------------------------------------------------------------------
# Corridas: repetición de experimentos y resúmenes
# ---------------------------------------------------------------------------


def repeat_experiment(run, n_runs=30, base_seed=0):
    """!
    @brief Ejecuta un experimento varias veces con semillas consecutivas.

    No conoce ningún modelo: recibe una función `run(seed) -> dict` que arma el
    notebook, por lo que sirve igual para los tres puntos.

    @param run Función que recibe una semilla entera y devuelve un diccionario de métricas.
    @param n_runs Cantidad de corridas.
    @param base_seed Semilla de la primera corrida; la corrida k usa `base_seed + k - 1`.
    @return DataFrame con una fila por corrida y las columnas `corrida`, `semilla` y métricas.
    """
    rows = []
    for run_index in range(n_runs):
        seed = base_seed + run_index
        rows.append({"corrida": run_index + 1, "semilla": seed, **dict(run(seed))})
    return pd.DataFrame(rows)


def summarize_runs(runs):
    """!
    @brief Resume las métricas numéricas de varias corridas.
    @param runs DataFrame producido por `repeat_experiment`.
    @return DataFrame con una fila por métrica y las columnas media, desviación, mínimo y máximo.
    """
    metrics = runs.drop(columns=list(RUN_COLUMNS), errors="ignore").select_dtypes("number")
    summary = metrics.agg(["mean", "std", "min", "max"]).T
    summary.columns = ["media", "desviación", "mínimo", "máximo"]
    return summary


def compare_experiments(results):
    """!
    @brief Construye la tabla comparativa entre métodos.
    @param results Diccionario `{nombre del método: DataFrame de repeat_experiment}`.
    @return DataFrame con una fila por método y columnas `<métrica> media` y `<métrica> desviación`.
    """
    rows = {}
    for method, runs in results.items():
        summary = summarize_runs(runs)
        row = {}
        for metric in summary.index:
            row[f"{metric} media"] = summary.loc[metric, "media"]
            row[f"{metric} desviación"] = summary.loc[metric, "desviación"]
        rows[method] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_metric_distributions(results, metric, title=None):
    """!
    @brief Dibuja un diagrama de caja de una métrica para cada método.
    @param results Diccionario `{nombre del método: DataFrame de repeat_experiment}`.
    @param metric Columna que se grafica, por ejemplo `f1` o `tasa_aciertos`.
    @param title Título de la figura; por omisión se deriva de la métrica.
    @return Figura de Matplotlib (sin mostrar).
    """
    figure, axis = plt.subplots(figsize=(max(6, 1.6 * len(results)), 4.5))
    axis.boxplot([runs[metric] for runs in results.values()], tick_labels=list(results))
    axis.set_title(title or f"Distribución de {metric} en las corridas")
    axis.set_ylabel(metric)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Figuras y tablas: exportación para el informe
# ---------------------------------------------------------------------------


def _slugify(text):
    """!
    @brief Convierte un texto en un nombre de archivo estable.
    @param text Texto de origen, por ejemplo un título con LaTeX o tildes.
    @return Nombre en minúsculas, ASCII y separado por guiones.
    """
    text = text.splitlines()[0].replace("–", "-") if text else ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"[$_{}\\]", "", text.lower())).strip("-")


def configure_figure_export(directory):
    """!
    @brief Activa el guardado en PNG de cada figura mostrada con `show_figure`.
    @param directory Carpeta de destino; se crea si no existe y se eliminan sus PNG previos.
    @return Ruta absoluta de la carpeta de destino.
    """
    global _figures_directory
    # Se parte del estilo por omisión para que el tema del IDE no llegue a las figuras del informe.
    matplotlib.rcdefaults()
    _figures_directory = Path(directory).resolve()
    _figures_directory.mkdir(parents=True, exist_ok=True)
    for file in _figures_directory.glob("*.png"):
        file.unlink()
    return _figures_directory


def configure_table_export(directory):
    """!
    @brief Activa el guardado en CSV de cada tabla mostrada con `display_table`.
    @param directory Carpeta de destino; se crea si no existe y se eliminan sus CSV previos.
    @return Ruta absoluta de la carpeta de destino.
    """
    global _tables_directory
    _tables_directory = Path(directory).resolve()
    _tables_directory.mkdir(parents=True, exist_ok=True)
    for file in _tables_directory.glob("*.csv"):
        file.unlink()
    return _tables_directory


def _figure_title(figure):
    """!
    @brief Obtiene el título principal de una figura.
    @param figure Figura de Matplotlib.
    @return Título de la figura, del primer eje con título o "figura".
    """
    if figure._suptitle is not None and figure._suptitle.get_text():
        return figure._suptitle.get_text()
    return next((axis.get_title() for axis in figure.axes if axis.get_title()), "figura")


def show_figure(figure, name=None):
    """!
    @brief Guarda la figura (si la exportación está activa) y la muestra.

    Los archivos de cada punto devuelven figuras sin mostrarlas; el notebook
    las pasa por esta función para que todas se exporten con el mismo criterio.

    @param figure Figura de Matplotlib ya terminada.
    @param name Nombre opcional del archivo; por omisión se deriva del título.
    @return None.
    """
    if _figures_directory is not None:
        filename = _slugify(name or _figure_title(figure)) or "figura"
        figure.savefig(
            _figures_directory / f"{filename}.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.show()


def display_table(table, title_text, formats=None, na_rep="—"):
    """!
    @brief Muestra un DataFrame con título y formato consistente, y lo exporta a CSV.
    @param table DataFrame de pandas que se desea mostrar.
    @param title_text Título visible; también define el nombre del CSV exportado.
    @param formats Mapeo opcional de columnas a formatos de pandas Styler.
    @param na_rep Texto usado para representar valores ausentes.
    @return None.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("La tabla debe ser un DataFrame de pandas.")

    if _tables_directory is not None:
        table.to_csv(_tables_directory / f"{_slugify(title_text) or 'tabla'}.csv")

    table_id = hashlib.sha256(title_text.encode("utf-8")).hexdigest()[:12]
    style = table.style.set_uuid(table_id).set_caption(title_text).set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "1.1em"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("padding-bottom", "0.5em"),
                ],
            }
        ],
        overwrite=False,
    )
    style = style.format(formats, na_rep=na_rep)
    display(style)


# ---------------------------------------------------------------------------
# Pruebas unitarias dentro del notebook
# ---------------------------------------------------------------------------


def run_tests(*modules):
    """!
    @brief Ejecuta las funciones `test_*` de los módulos de prueba indicados.

    Una prueba que lanza `NotImplementedError` se reporta como pendiente: así
    el notebook muestra el avance del equipo mientras hay puntos sin terminar.

    @param modules Módulos de `tests/` ya importados.
    @return DataFrame con módulo, prueba, estado y detalle de cada prueba.
    @throws AssertionError Si alguna prueba implementada falla.
    """
    rows = []
    for module in modules:
        tests = [
            (name, function)
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_") and function.__module__ == module.__name__
        ]
        tests.sort(key=lambda item: inspect.getsourcelines(item[1])[1])
        for name, function in tests:
            try:
                function()
                status, detail = "superada", ""
            except NotImplementedError as error:
                status, detail = "pendiente", str(error)
            except Exception as error:
                status, detail = "fallida", f"{type(error).__name__}: {error}"
            rows.append(
                {"módulo": module.__name__, "prueba": name, "estado": status, "detalle": detail}
            )

    results = pd.DataFrame(rows, columns=["módulo", "prueba", "estado", "detalle"])
    display_table(results, "Resultado de las pruebas unitarias")

    counts = results["estado"].value_counts()
    print(
        f"{counts.get('superada', 0)} superadas, {counts.get('pendiente', 0)} pendientes, "
        f"{counts.get('fallida', 0)} fallidas."
    )
    if counts.get("fallida", 0):
        raise AssertionError("Hay pruebas unitarias fallidas; ver la tabla anterior.")
    return results
