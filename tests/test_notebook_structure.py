"""!
@file test_notebook_structure.py
@brief Verifica que el cuaderno conserve sus definiciones y dependencias opcionales locales.
"""

import ast
import nbformat

from notebook_loader import NOTEBOOK_PATH, definition_cells


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
