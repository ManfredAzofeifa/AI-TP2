# Notebook

`tp2_detector_spam.ipynb` es la fuente canónica de toda la implementación:
incluye funciones compartidas, modelos, experimentos y análisis. Presenta
resultados, figuras y pruebas. Debe contener, en
orden, configuración reproducible, carga del dataset, una sección por punto
del enunciado, comparación de resultados, pruebas unitarias y conclusiones.

Las implementaciones se escriben en celdas de su apartado, con funciones
documentadas antes de los experimentos que las usan. Las pruebas externas
de `tests/` cargan estas definiciones mediante `tests/notebook_loader.py`,
sin duplicar su lógica ni ejecutar experimentos completos. No crear copias
personales dentro de esta carpeta. Las
pruebas temporales deben realizarse fuera del repositorio o en ramas de
trabajo y descartarse antes de integrar cambios.
