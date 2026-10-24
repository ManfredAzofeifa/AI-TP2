"""!
@file test_punto3.py
@brief Pruebas del punto 3: clasificación con modelos de lenguaje.

Las pruebas no deben descargar ni ejecutar modelos grandes: se prueban las
partes deterministas, como la interpretación de las respuestas.
"""

from notebook_loader import notebook as punto3


def test_parse_label_handles_responses():
    """!
    @brief (Recomendada, 3.1) Respuestas como "spam", " Ham." o "SPAM\n" deben traducirse
           a 1 o 0, y las ambiguas deben seguir la regla documentada en `parse_label`.
    @return None.
    """
    raise NotImplementedError("Punto 3.1: diseñar la prueba de parse_label.")
