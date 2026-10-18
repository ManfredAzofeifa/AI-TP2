"""!
@file conftest.py
@brief Configuración de pytest: agrega `src/` al path y reporta como pendientes las pruebas sin implementar.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """!
    @brief Convierte un `NotImplementedError` en una prueba omitida con el motivo "Pendiente".

    Así `pytest` distingue el trabajo todavía no hecho de un error real, igual
    que `utils.run_tests` en el notebook.

    @param item Prueba que se está ejecutando.
    @param call Información de la fase ejecutada.
    @return Reporte de pytest, posiblemente marcado como omitido.
    """
    report = yield
    if call.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] + 1, f"Pendiente: {call.excinfo.value}")
    return report
