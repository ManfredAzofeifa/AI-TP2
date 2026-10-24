"""!
@file test_experiment_protocol.py
@brief Verifica calibración interna, separación de prueba, caché y búsqueda vectorizada del notebook.
"""

from unittest.mock import patch

import pandas as pd
import torch

from notebook_loader import notebook


def _small_dataset():
    """!
    @brief Genera datos reproducibles con un identificador único en la primera coordenada.
    @return Tupla de características (100, 3) y etiquetas binarias (100,).
    """
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(100, 3, generator=generator)
    features[:, 0] = torch.arange(100) / 100
    targets = torch.tensor([0] * 80 + [1] * 20)
    return features, targets


def _winning_grid():
    """!
    @brief Define una cuadrícula pequeña cuyo primer candidato debe usarse automáticamente.
    @return DataFrame ordenado con tasa 0.5 y 4 épocas como mejor combinación.
    """
    return pd.DataFrame({
        "learning_rate": [0.5, 0.25], "epochs": [4, 3],
        "score": [0.9, 0.8], "validation_loss": [0.2, 0.3], "train_loss": [0.1, 0.2],
    })


def test_batched_grid_matches_independent_training():
    """!
    @brief Contrasta cada checkpoint y tasa vectorizados con entrenamientos independientes.
    @return None.
    """
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(100, 5, generator=generator)
    targets = (features[:, 0] + 0.3 * features[:, 1] > 0.2).to(torch.int64)
    rates, checkpoints = [0.1, 1.0, 10.0], [0, 1, 10, 40]
    grid = notebook.grid_search_logistic_regression(
        features[:70], targets[:70], features[70:], targets[70:],
        rates, checkpoints, notebook.f1_score, seed=2,
    ).set_index(["learning_rate", "epochs"])
    assert len(grid) == len(rates) * len(checkpoints)
    for rate in rates:
        for epochs in checkpoints:
            model = notebook.train_logistic_regression(
                features[:70], targets[:70], rate, epochs, seed=2,
            )
            row = grid.loc[(rate, epochs)]
            assert row["score"] == notebook.f1_score(
                targets[70:], notebook.predict(model, features[70:]),
            )
            expected_loss = notebook._cross_entropy(
                features[70:], targets[70:].to(torch.float32), model.weights, model.bias,
            )
            assert abs(row["validation_loss"] - expected_loss) < 2e-6
            assert abs(row["train_loss"] - model.loss_history[-1]) < 2e-6


def test_each_run_calibrates_with_its_own_training_and_uses_winner():
    """!
    @brief Comprueba en dos semillas que ajuste y validación excluyen la prueba y se usa el ganador.
    @return None.
    """
    features, targets = _small_dataset()
    real_train = notebook.train_logistic_regression
    for seed in (2026, 2027):
        train_indices, test_indices = notebook.stratified_split(targets, 0.7, seed)
        with patch.object(notebook, "grid_search_logistic_regression", return_value=_winning_grid()) as grid:
            with patch.object(notebook, "train_logistic_regression", wraps=real_train) as train:
                result = notebook.calibrated_logistic_regression_run(
                    features, targets, seed, [0.25, 0.5], [3, 4],
                )
        X_fit, _, X_validation, _ = grid.call_args.args[:4]
        fit_ids = torch.round(X_fit[:, 0] * 100).to(torch.int64)
        validation_ids = torch.round(X_validation[:, 0] * 100).to(torch.int64)
        assert not bool(torch.isin(fit_ids, test_indices).any())
        assert not bool(torch.isin(validation_ids, test_indices).any())
        assert not bool(torch.isin(fit_ids, validation_ids).any())
        assert torch.equal(torch.cat((fit_ids, validation_ids)).sort().values, train_indices.sort().values)
        assert grid.call_args.kwargs["seed"] == seed
        torch.testing.assert_close(train.call_args.args[0], features[train_indices])
        assert train.call_args.args[2:4] == (0.5, 4)
        assert result["learning_rate"] == 0.5 and result["epochs"] == 4
        assert result["tp"] + result["tn"] + result["fp"] + result["fn"] == len(test_indices)


def test_pca_calibration_excludes_validation_and_reuses_only_matching_fit():
    """!
    @brief Verifica PCA interno solo en ajuste, PCA final en entrenamiento y reutilización entre k.
    @return None.
    """
    features, targets = _small_dataset()
    seed = 2027
    train_indices, test_indices = notebook.stratified_split(targets, 0.7, seed)
    fit_positions, validation_positions = notebook.stratified_split(targets[train_indices], 0.8, seed)
    fit_indices = train_indices[fit_positions]
    validation_indices = train_indices[validation_positions]
    cache = {}
    with patch.object(notebook, "grid_search_logistic_regression", return_value=_winning_grid()):
        with patch.object(notebook, "fit_pca", wraps=notebook.fit_pca) as fit_pca:
            for n_components in (2, 1):
                notebook.calibrated_logistic_regression_run(
                    features, targets, seed, [0.25, 0.5], [3, 4],
                    n_components=n_components, pca_cache=cache,
                )
    assert fit_pca.call_count == 2
    torch.testing.assert_close(fit_pca.call_args_list[0].args[0], features[fit_indices])
    torch.testing.assert_close(fit_pca.call_args_list[1].args[0], features[train_indices])
    internal_ids = torch.round(fit_pca.call_args_list[0].args[0][:, 0] * 100).to(torch.int64)
    assert not bool(torch.isin(internal_ids, validation_indices).any())
    assert not bool(torch.isin(internal_ids, test_indices).any())


def test_feature_cache_checks_text_order_model_revision_shape_and_values():
    """!
    @brief Rechaza cachés antiguas o con textos, modelo, revisión, dimensiones o valores incompatibles.
    @return None.
    """
    metadata = notebook.feature_cache_metadata(["hola", "oferta"], "modelo", 768, "revision1")
    payload = {"metadata": metadata, "features": torch.ones(2, 768, dtype=torch.float32)}
    assert notebook.feature_cache_is_valid(payload, metadata)
    assert not notebook.feature_cache_is_valid(payload["features"], metadata)
    for expected in (
        notebook.feature_cache_metadata(["oferta", "hola"], "modelo", 768, "revision1"),
        notebook.feature_cache_metadata(["hola", "oferta"], "otro-modelo", 768, "revision1"),
        notebook.feature_cache_metadata(["hola", "oferta"], "modelo", 768, "revision2"),
        notebook.feature_cache_metadata(["hola", "oferta"], "modelo", 384, "revision1"),
    ):
        assert not notebook.feature_cache_is_valid(payload, expected)
    for invalid in (torch.ones(2, 384), torch.ones(2, 768, dtype=torch.float64), torch.full((2, 768), float("nan"))):
        assert not notebook.feature_cache_is_valid({"metadata": metadata, "features": invalid}, metadata)
