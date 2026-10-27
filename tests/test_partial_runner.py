"""!
@file test_partial_runner.py
@brief Verifica el ejecutor auxiliar de los puntos 1 y 2 sin ejecutar experimentos ni modelos.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import nbformat

from notebook_loader import NOTEBOOK_PATH, notebook


def _partial_runner():
    """!
    @brief Importa el ejecutor compartido sin ejecutar experimentos ni acceder a modelos.
    @return Módulo del script portátil.
    """
    path = NOTEBOOK_PATH.parents[1] / "scripts/run_puntos_1_2.py"
    spec = importlib.util.spec_from_file_location("tp2_partial_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_runner_keeps_settings_and_excludes_point3():
    """!
    @brief Verifica la selección parcial y que activar la auditoría no modifique la fuente.
    @return None.
    """
    runner = _partial_runner()
    source = nbformat.read(NOTEBOOK_PATH, as_version=4)
    original = copy.deepcopy(source)
    prepared = runner.prepare_notebook(source, include_audit=True)
    ids = [cell.id for cell in prepared.cells]
    assert runner.POINT3_CELL not in ids
    assert ids[-2:] == list(runner.TEST_CELLS)
    config = next(cell.source for cell in source.cells if cell.id == runner.CONFIG_CELL)
    assert next(cell.source for cell in prepared.cells if cell.id == runner.CONFIG_CELL) == config
    assert "RUN_METRIC_AUDIT = True" in next(c.source for c in prepared.cells if c.id == runner.AUDIT_CELL)
    assert source == original


def test_partial_runner_validates_cache_identity_without_touching_results():
    """!
    @brief Acepta solo la caché identificada por el notebook y conserva resultados al comprobarla.
    @return None.
    """
    import torch

    runner = _partial_runner()
    source = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'ENCODER_NAME = "test"\nENCODER_REVISION = "fixed"\nEXPECTED_EMBEDDING_DIMENSIONS = 2',
        id=runner.CONFIG_CELL,
    )])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "data/raw/SMSSpamCollection"
        raw.parent.mkdir(parents=True)
        raw.write_text("ham\thola\nspam\toferte\n", encoding="utf-8")
        results = root / "results/unchanged.txt"
        results.parent.mkdir()
        results.write_text("conservar", encoding="utf-8")
        path, valid = runner.valid_cache(root, source, notebook)
        assert not valid
        metadata = notebook.feature_cache_metadata(["hola", "oferte"], "test", 2, "fixed")
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
        assert path.name == f"features-{key}.pt"
        path.parent.mkdir(parents=True)
        torch.save({"metadata": metadata, "features": torch.zeros(2, 2)}, path)
        assert runner.valid_cache(root, source, notebook)[1]
        path.write_bytes(b"cache incompatible")
        assert not runner.valid_cache(root, source, notebook)[1]
        assert results.read_text(encoding="utf-8") == "conservar"


def test_partial_runner_starts_a_clean_kernel_with_current_python():
    """!
    @brief Ejecuta una celda mínima en un kernel nuevo del intérprete invocante, sin modelos.
    @return None.
    """
    runner = _partial_runner()
    document = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import sys\nprint(sys.executable)"
    )])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "run/result.ipynb"
        runner.execute_notebook(document, root, output)
        saved = nbformat.read(output, as_version=4)
        executable = saved.cells[0].outputs[0].text.strip()
        assert Path(executable) == Path(sys.executable)
        assert saved.cells[0].execution_count == 1
