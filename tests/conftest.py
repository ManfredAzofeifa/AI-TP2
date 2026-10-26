"""!
@file conftest.py
@brief Configuración de pytest y reporte de pruebas todavía pendientes.
"""

import matplotlib
import pytest

# Las pruebas solo construyen figuras: un backend sin ventanas evita depender de Tk.
matplotlib.use("Agg")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """!
    @brief Convierte un `NotImplementedError` en una prueba omitida con el motivo "Pendiente".

    Así `pytest` distingue el trabajo todavía no hecho de un error real, igual
    que `run_tests` en el notebook.

    @param item Prueba que se está ejecutando.
    @param call Información de la fase ejecutada.
    @return Reporte de pytest, posiblemente marcado como omitido.
    """
    report = yield
    if call.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] + 1, f"Pendiente: {call.excinfo.value}")
    return report
