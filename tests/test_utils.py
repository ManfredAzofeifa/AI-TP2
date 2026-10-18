"""!
@file test_utils.py
@brief Pruebas de la infraestructura compartida: datos, particiones, métricas y corridas.
"""

import math
import tempfile
from pathlib import Path

import torch

from tp2 import utils

TARGETS = torch.tensor([0] * 80 + [1] * 20)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def test_load_sms_spam_parses_labels_and_quotes():
    """!
    @brief Comprueba que se leen etiquetas y textos con comillas sin alterarlos.
    @return None.
    """
    lines = 'ham\tGo until jurong point, "crazy".. Available\nspam\tFree entry! Txt 87121 now\n'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / utils.DATASET_FILENAME
        path.write_text(lines, encoding="utf-8")
        messages = utils.load_sms_spam(path)

    assert messages["target"].tolist() == [0, 1]
    assert messages["text"].iloc[0] == 'Go until jurong point, "crazy".. Available'


def test_stratified_split_is_disjoint_and_complete():
    """!
    @brief Comprueba que entrenamiento y prueba no se traslapan y cubren todos los índices.
    @return None.
    """
    train_indices, test_indices = utils.stratified_split(TARGETS, train_fraction=0.7, seed=0)
    all_indices = torch.cat((train_indices, test_indices)).sort().values

    assert len(train_indices) == 70
    assert len(test_indices) == 30
    torch.testing.assert_close(all_indices, torch.arange(len(TARGETS)))


def test_stratified_split_preserves_class_proportions():
    """!
    @brief Comprueba que cada partición conserva la proporción 80/20 de las clases.
    @return None.
    """
    train_indices, test_indices = utils.stratified_split(TARGETS, train_fraction=0.7, seed=0)

    assert int(TARGETS[train_indices].sum()) == 14
    assert int(TARGETS[test_indices].sum()) == 6


def test_stratified_split_is_reproducible():
    """!
    @brief Comprueba que la misma semilla da la misma partición y otra semilla, una distinta.
    @return None.
    """
    first, _ = utils.stratified_split(TARGETS, seed=3)
    second, _ = utils.stratified_split(TARGETS, seed=3)
    other, _ = utils.stratified_split(TARGETS, seed=4)

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, other)


def test_sample_indices_size():
    """!
    @brief Comprueba que la muestra del 30 % tiene el tamaño y la estratificación esperados.
    @return None.
    """
    sampled = utils.sample_indices(TARGETS, fraction=0.3, seed=1)

    assert len(sampled) == 30
    assert len(torch.unique(sampled)) == 30
    assert int(TARGETS[sampled].sum()) == 6


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def test_accuracy_and_f1_known_values():
    """!
    @brief Compara contra valores calculados a mano (tp = 2, fp = 1, fn = 1, tn = 2).
    @return None.
    """
    y_true = torch.tensor([1, 1, 1, 0, 0, 0])
    y_pred = torch.tensor([1, 1, 0, 1, 0, 0])

    assert utils.confusion_counts(y_true, y_pred) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert math.isclose(utils.accuracy(y_true, y_pred), 4 / 6)
    assert math.isclose(utils.f1_score(y_true, y_pred), 2 / 3)


def test_f1_without_true_positives_is_zero():
    """!
    @brief Comprueba el caso degenerado en que nunca se predice spam.
    @return None.
    """
    y_true = [1, 0, 0, 1]
    y_pred = [0, 0, 0, 0]

    assert utils.f1_score(y_true, y_pred) == 0.0
    assert utils.classification_metrics(y_true, y_pred) == {"tasa_aciertos": 0.5, "f1": 0.0}


def test_mismatched_shapes_raise():
    """!
    @brief Comprueba que etiquetas de tamaños distintos producen un error explícito.
    @return None.
    """
    try:
        utils.accuracy([0, 1, 1], [0, 1])
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError con formas distintas.")


# ---------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------


def test_repeat_experiment_uses_consecutive_seeds():
    """!
    @brief Comprueba la cantidad de corridas, las semillas y las métricas registradas.
    @return None.
    """
    runs = utils.repeat_experiment(lambda seed: {"f1": seed / 10}, n_runs=3, base_seed=5)

    assert runs["corrida"].tolist() == [1, 2, 3]
    assert runs["semilla"].tolist() == [5, 6, 7]
    assert runs["f1"].tolist() == [0.5, 0.6, 0.7]


def test_summarize_and_compare_runs():
    """!
    @brief Comprueba media y desviación del resumen y de la tabla comparativa.
    @return None.
    """
    runs = utils.repeat_experiment(lambda seed: {"f1": float(seed)}, n_runs=3, base_seed=1)
    summary = utils.summarize_runs(runs)
    comparison = utils.compare_experiments({"modelo": runs})

    assert list(summary.index) == ["f1"]
    assert math.isclose(summary.loc["f1", "media"], 2.0)
    assert math.isclose(summary.loc["f1", "desviación"], 1.0)
    assert math.isclose(comparison.loc["modelo", "f1 media"], 2.0)
