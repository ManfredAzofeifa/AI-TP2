"""!
@file test_module_independence.py
@brief Verifica la regla de modularidad: ningún módulo de `tp2` importa a otro del paquete.
"""

import ast
import importlib
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "tp2"


def _package_modules():
    """!
    @brief Lista los módulos del paquete, sin `__init__.py`.
    @return Rutas de los archivos `.py` de `src/tp2`, ordenadas por nombre.
    """
    return sorted(path for path in PACKAGE_DIRECTORY.glob("*.py") if path.stem != "__init__")


def _internal_imports(path):
    """!
    @brief Encuentra los imports de un archivo que apuntan al propio paquete `tp2`.
    @param path Ruta del módulo que se analiza.
    @return Lista de imports internos encontrados (vacía si el módulo es independiente).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").split(".")[0] == "tp2":
                found.append(f"from {'.' * node.level}{node.module or ''} import ...")
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "tp2")
    return found


def test_modules_do_not_import_each_other():
    """!
    @brief Falla si algún módulo de `src/tp2` importa otro módulo del paquete.
    @return None.
    """
    violations = {path.name: _internal_imports(path) for path in _package_modules()}
    violations = {name: imports for name, imports in violations.items() if imports}
    assert not violations, f"Los módulos deben ser independientes; integrar en el notebook: {violations}"


def test_modules_import_without_optional_dependencies():
    """!
    @brief Falla si algún módulo no se puede importar con las dependencias base.

    Las dependencias opcionales (`transformers`, `peft`, ...) deben importarse
    dentro de las funciones que las usan, no al inicio del módulo.

    @return None.
    """
    for path in _package_modules():
        importlib.import_module(f"tp2.{path.stem}")
