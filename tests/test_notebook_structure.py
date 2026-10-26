"""!
@file test_notebook_structure.py
@brief Verifica que el cuaderno conserve sus definiciones y dependencias opcionales locales.
"""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import nbformat

from notebook_loader import NOTEBOOK_PATH, definition_cells, notebook


def _gpu_host_module():
    """!
    @brief Carga el diagnóstico de infraestructura sin consultar servicios al importar.
    @return Módulo de diagnóstico sin ejecutar su CLI.
    """
    path = NOTEBOOK_PATH.parents[1] / "scripts/gpu_host_status.py"
    spec = importlib.util.spec_from_file_location("tp2_gpu_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_status_cuda_unavailable_does_not_allocate(monkeypatch):
    """!
    @brief Verifica el diagnóstico CPU y la negativa explícita a probar CUDA inexistente.
    @param monkeypatch Fixture para simular PyTorch sin GPU ni modelos.
    @return None.
    """
    module = _gpu_host_module()
    fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None),
                           cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake)
    assert module.cuda_status()["smoke_test"] == "not_requested"
    assert module.cuda_status(True)["smoke_test"] == "cuda_unavailable"


def test_gpu_status_discards_failed_command_and_exception_details(monkeypatch):
    """!
    @brief Evita publicar stderr, stdout fallido o mensajes privados de importación.
    @param monkeypatch Fixture para simular errores con información privada.
    @return None.
    """
    module = _gpu_host_module()
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout="PRIVATE_TOKEN", stderr="PRIVATE_KEY"))
    assert module.command(["test"]) == (1, "")

    def failed_import(name):
        """!
        @brief Simula un error cuyo mensaje debe permanecer privado.
        @param name Nombre del módulo solicitado.
        @return No retorna; lanza un error controlado.
        """
        raise RuntimeError("PRIVATE_TOKEN")

    monkeypatch.setattr(module.importlib, "import_module", failed_import)
    report = module.cuda_status(True)
    assert report["error_type"] == "RuntimeError"
    assert report["smoke_test"] == "failed"
    assert "PRIVATE" not in json.dumps(report)


def test_gpu_status_reports_dirty_without_serializing_filenames(monkeypatch, tmp_path):
    """!
    @brief Conserva el estado Git y limita los servicios a valores públicos conocidos.
    @param monkeypatch Fixture para simular comandos sin consultar servicios reales.
    @param tmp_path Directorio de prueba sin credenciales ni repositorios privados.
    @return None.
    """
    module = _gpu_host_module()

    def fake_command(arguments, timeout=5):
        """!
        @brief Simula respuestas de diagnósticos, incluida una ruta privada de Git.
        @param arguments Argumentos solicitados por el diagnóstico.
        @param timeout Límite temporal sin efecto en esta simulación.
        @return Código y respuesta controlados.
        """
        if "--porcelain" in arguments:
            return 0, "?? PRIVATE_TOKEN.env"
        if "HEAD" in arguments:
            return 0, "abcdef"
        if "--show-current" in arguments:
            return 0, "main"
        return 0, "PRIVATE_TOKEN"

    monkeypatch.setattr(module, "command", fake_command)
    monkeypatch.setattr(module, "cuda_status", lambda smoke: {"available": False})
    report = module.host_status(tmp_path, jupyter_ports=())
    assert report["git"] == {"commit": "abcdef", "branch": "main", "dirty": True}
    assert report["nvidia_smi"]["devices"] == []
    assert "PRIVATE" not in json.dumps(report)


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


def test_notebook_is_valid_and_self_contained():
    """!
    @brief Valida el archivo guardado y que no dependa del antiguo paquete tp2.
    @return None.
    """
    document = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(document)
    assert len({cell.id for cell in document.cells}) == len(document.cells)
    assert definition_cells()
    for cell in document.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "tp2"
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "tp2" for alias in node.names)


def test_definitions_keep_optional_imports_inside_functions():
    """!
    @brief Evita que cargar las definiciones importe modelos o repita funciones.
    @return None.
    """
    optional = {"transformers", "sentence_transformers", "peft", "accelerate"}
    names = []
    for cell in definition_cells():
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in optional
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in optional for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            else:
                assert isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr))
    assert len(names) == len(set(names)), "Hay definiciones que se sobrescriben entre apartados."


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
