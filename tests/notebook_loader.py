"""!
@file notebook_loader.py
@brief Carga las definiciones del cuaderno sin descargar datos ni ejecutar experimentos.
"""

import json
import sys
from pathlib import Path
from types import ModuleType

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "tp2_detector_spam.ipynb"
DEFINITIONS_PREFIX = "# Definiciones:"


def definition_cells():
    """!
    @brief Lee únicamente las celdas identificadas como definiciones del cuaderno.
    @return Lista de celdas de código sin configuración, descargas ni experimentos.
    """
    document = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        cell for cell in document["cells"]
        if cell["cell_type"] == "code"
        and "".join(cell["source"]).startswith(DEFINITIONS_PREFIX)
    ]


def load_notebook_definitions():
    """!
    @brief Ejecuta las definiciones en un espacio propio compartido por las pruebas.
    @return Módulo con las funciones y constantes del notebook, sin sus resultados.
    """
    module_name = "tp2_notebook_definitions"
    module = ModuleType(module_name)
    module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module_name] = module
    cells = definition_cells()
    if not cells:
        raise ValueError("El notebook no contiene celdas de definiciones identificadas.")
    for cell in cells:
        source = "".join(cell["source"])
        exec(compile(source, f"{NOTEBOOK_PATH}#{cell['id']}", "exec"), module.__dict__)
    return module


notebook = load_notebook_definitions()
