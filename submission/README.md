# Entrega

Entregado el 26/10/2026 por TEC Digital. Esta carpeta se arma con
`./build_submission.sh` desde la raíz del proyecto; el `.zip` se crea
comprimiendo su contenido (sin este README):

- `tp2_detector_spam.pdf`: informe compilado (35 páginas).
- `informe/`: fuentes LaTeX (`main.tex`, `sections/`, `references.bib`) y las
  figuras que usa en `figuras/`. Compila sin el resto del proyecto con
  `latexmk -pdf main.tex`.
- `notebooks/tp2_detector_spam.ipynb`: toda la implementación, sin salidas.
- `tests/`: 74 pruebas unitarias que cargan las definiciones del notebook;
  desde esta carpeta, `python -m pytest tests`.

Solo el PDF y este README se versionan; `informe/`, `notebooks/` y `tests/`
son copias generadas. El `.zip` entregado tiene SHA-256
`11a520305d942c453f860b3387dcfe8de646ddd8e73a816c9c41a2633997a936`.
