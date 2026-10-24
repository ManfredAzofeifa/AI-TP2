# Entrega

Esta carpeta se reserva para las copias finales y verificadas de los archivos
solicitados por TEC Digital. No trabajar directamente sobre esas copias.

El enunciado pide un archivo `.zip` con el código fuente LaTeX, el PDF y el
notebook. Toda la implementación está en el notebook; el `.zip` debe incluir
(ver también la checklist de `TODO.md`):

- `notebooks/tp2_detector_spam.ipynb`, ejecutado de principio a fin.
- `tests/`, con las pruebas que cargan las definiciones del notebook.
- `data/raw/SMSSpamCollection`, el dataset original de UCI sin editar, y su
  fuente y licencia en `data/README.md`.
- `requirements.txt`.
- `docs/report/` (fuentes LaTeX) y `results/figures/`, que el informe usa.
- `tp2_detector_spam.pdf`, generado con `./build_pdf.sh`.

## Doxygen

La documentación interna de Doxygen se copia en la carpeta `doxygen`. Dentro
del folder `html` existe un archivo principal, `index.html`, que se usa para
navegar a través de la documentación interna.
