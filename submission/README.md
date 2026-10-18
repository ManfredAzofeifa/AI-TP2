# Entrega

Esta carpeta se reserva para las copias finales y verificadas de los archivos
solicitados por TEC Digital. No trabajar directamente sobre esas copias.

El enunciado pide un archivo `.zip` con el código fuente LaTeX, el PDF y el
notebook. Como el notebook importa `punto1.py`, `punto2.py`, `punto3.py` y
`utils.py`, el `.zip` debe incluir (ver también la checklist de `TODO.md`):

- `notebooks/tp2_detector_spam.ipynb`, ejecutado de principio a fin.
- `src/tp2/` y `tests/`, de los que depende el notebook.
- `requirements.txt`.
- `docs/report/` (fuentes LaTeX) y `results/figures/`, que el informe usa.
- `tp2_detector_spam.pdf`, generado con `./build_pdf.sh`.

## Doxygen

La documentación interna de Doxygen se copia en la carpeta `doxygen`. Dentro
del folder `html` existe un archivo principal, `index.html`, que se usa para
navegar a través de la documentación interna.
