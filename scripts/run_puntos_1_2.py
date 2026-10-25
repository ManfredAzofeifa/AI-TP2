"""!
@file run_puntos_1_2.py
@brief Ejecuta las celdas del notebook de los puntos 1 y 2, sin duplicar su implementación.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

POINT3_CELL = "4f781ca7"
AUDIT_CELL = "273bdc0f5dff8d3e"
TEST_CELLS = ("721dc258", "eec906b3")
CONFIG_CELL = "5044730f"


def read_cache_settings(document):
    """!
    @brief Lee los literales de identidad de la caché sin ejecutar la configuración.
    @param document Notebook original.
    @return Diccionario con codificador, revisión y dimensión definidos en el notebook.
    """
    source = next(cell.source for cell in document.cells if cell.id == CONFIG_CELL)
    names = {"ENCODER_NAME", "ENCODER_REVISION", "EXPECTED_EMBEDDING_DIMENSIONS"}
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                values[name] = ast.literal_eval(node.value)
    if values.keys() != names:
        raise ValueError("La configuración no contiene la identidad completa de la caché.")
    return values


def valid_cache(root, document, definitions):
    """!
    @brief Comprueba la caché local antes de ejecutar celdas que limpian resultados.
    @param root Raíz del repositorio.
    @param document Notebook original, con su configuración vigente.
    @param definitions Definiciones cargadas por tests/notebook_loader.py.
    @return Tupla con ruta de la caché y validez según el contrato del notebook.
    """
    import torch

    settings = read_cache_settings(document)
    messages = definitions.load_sms_spam(root / "data/raw/SMSSpamCollection")
    metadata = definitions.feature_cache_metadata(
        messages.text.tolist(), settings["ENCODER_NAME"],
        settings["EXPECTED_EMBEDDING_DIMENSIONS"], settings["ENCODER_REVISION"],
    )
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = root / "data/processed" / f"features-{key}.pt"
    if not path.is_file():
        return path, False
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
        return path, False
    return path, definitions.feature_cache_is_valid(payload, metadata)


def prepare_notebook(document, include_audit=False):
    """!
    @brief Selecciona puntos 1 y 2 y pruebas, y activa la auditoría solo en una copia.
    @param document Notebook original; permanece sin modificaciones.
    @param include_audit Si es verdadero, activa la comparación adicional de métricas.
    @return Notebook ejecutable sin punto 3 ni comparación final pendiente.
    """
    cells = copy.deepcopy(document.cells)
    ids = [cell.id for cell in cells]
    if not {POINT3_CELL, AUDIT_CELL, *TEST_CELLS}.issubset(ids):
        raise ValueError("Cambió la estructura del notebook; revisar los IDs del ejecutor.")
    selected = cells[:ids.index(POINT3_CELL)]
    selected += [cell for cell in cells if cell.id in TEST_CELLS]
    for cell in selected:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
        if cell.id == AUDIT_CELL:
            original = "RUN_METRIC_AUDIT = False"
            if original not in cell.source:
                raise ValueError("La auditoría debe estar desactivada en el notebook canónico.")
            if include_audit:
                cell.source = cell.source.replace(original, "RUN_METRIC_AUDIT = True", 1)
    return nbformat.v4.new_notebook(cells=selected, metadata=copy.deepcopy(document.metadata))


def execute_notebook(document, root, output, allow_download=False):
    """!
    @brief Ejecuta una copia en un kernel limpio del mismo intérprete que invoca el script.
    @param document Notebook preparado para esta ejecución.
    @param root Directorio de trabajo del kernel.
    @param output Ruta del notebook ejecutado; también se guarda si ocurre un error.
    @param allow_download Autoriza el acceso a modelos de Hugging Face durante esta ejecución.
    @return None.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if not allow_download:
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="tp2-kernel-") as directory:
        kernel = Path(directory) / "tp2-local"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python local TP2", "language": "python",
        }), encoding="utf-8")
        manager = AsyncKernelManager(
            kernel_name="tp2-local",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory], ensure_native_kernel=False),
        )
        client = NotebookClient(
            document, km=manager, timeout=1800,
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute(cleanup_kc=True, env=environment)
        finally:
            nbformat.write(document, output)


def main():
    """!
    @brief Valida el entorno y ejecuta los puntos pedidos con autorización explícita en la CLI.
    @return Código de salida 0 si la validación o la ejecución terminan correctamente.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--check", action="store_true", help="Comprueba la caché sin ejecutar ni borrar resultados.")
    parser.add_argument("--audit", action="store_true", help="Incluye las 30 corridas por alternativa de la auditoría.")
    parser.add_argument("--allow-download", action="store_true", help="Permite regenerar MPNet si falta una caché válida.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    document = nbformat.read(root / "notebooks/tp2_detector_spam.ipynb", as_version=4)
    prepared = prepare_notebook(document, args.audit)
    sys.path.insert(0, str(root / "tests"))
    from notebook_loader import notebook

    path, cached = valid_cache(root, document, notebook)
    print(f"Intérprete: {sys.executable}")
    print(f"Caché: {path.relative_to(root)}; válida: {cached}")
    if args.check:
        return 0 if cached else 1
    if not cached and not args.allow_download:
        parser.error("Falta una caché válida. Para autorizar su regeneración, añadir --allow-download.")
    output = root / "build/notebook-run/puntos-1-2-ejecutado.ipynb"
    (root / "build").mkdir(exist_ok=True)
    if (root / "results").is_dir():
        backup = Path(tempfile.mkdtemp(prefix="results-before-", dir=root / "build"))
        shutil.copytree(root / "results", backup, dirs_exist_ok=True)
        print(f"Respaldo de resultados: {backup.relative_to(root)}", flush=True)
    print("Ejecutando puntos 1 y 2 y pruebas; puede tardar varios minutos.", flush=True)
    execute_notebook(prepared, root, output, args.allow_download)
    print(f"Notebook ejecutado: {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
