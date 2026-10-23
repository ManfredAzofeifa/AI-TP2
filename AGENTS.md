# AGENTS.md

Instrucciones para agentes de IA que trabajen en este repositorio. Las
personas del equipo usan [TODO.md](TODO.md).

## Regla principal: actualizar siempre TODO.md, AGENTS.md y review

Todo cambio que complete, agregue o modifique una tarea, una decisión, un
resultado, una regla, la estructura o el estado del proyecto se refleja, en el
mismo cambio, en **los tres** documentos de seguimiento:

1. `TODO.md`: casillas, estado actual, decisiones y verificación pendiente.
2. `AGENTS.md`: estado actual, reglas y flujo.
3. `docs/report/review.tex` **y su PDF compilado** `docs/report/review.pdf`:
   lógica, pasos, resultados y registro de decisiones. Después de editar el
   `.tex`, recompilar con `./build_pdf.sh review` y versionar ambos archivos
   juntos; nunca dejar el PDF desactualizado respecto del `.tex`.

Nunca actualizar solo uno de ellos. Si no se puede compilar el PDF (por
ejemplo, sin LaTeX instalado), decirlo a la persona y anotarlo en la
verificación pendiente de `TODO.md`.

## Contexto

- Trabajo práctico 2 de Principios de Inteligencia Artificial: detectar spam
  en el dataset SMSSpamCollection (UCI) con (1) regresión logística propia en
  PyTorch, con y sin PCA; (2) detección de anomalías con densidades por
  histograma y un umbral de verosimilitud; y (3) modelos grandes de lenguaje.
- Entrega: martes 20 de octubre de 2026. Entregable: `.zip` con fuentes
  LaTeX, PDF y el notebook documentado, más `src/tp2/` y `tests/`.
- Enunciado: `docs/assignment/TrabajoPractico2_detector_ataques_supervisado_no_supervisado.pdf`
  (se puede leer con `pdftotext -layout`).

| Ruta | Contenido |
|---|---|
| `src/tp2/punto1.py` | Punto 1: regresión logística (1.1 a 1.5) |
| `src/tp2/punto2.py` | Punto 2: detección de anomalías (2.1 y 2.2) |
| `src/tp2/punto3.py` | Punto 3: modelos de lenguaje (3.1 a 3.3) |
| `src/tp2/utils.py` | Datos, particiones, métricas, corridas, figuras, tablas y pruebas en el notebook |
| `notebooks/tp2_detector_spam.ipynb` | `main`: único lugar donde se combinan los archivos |
| `tests/test_puntoN.py`, `tests/test_utils.py` | Pruebas por punto y de la infraestructura |
| `tests/test_module_independence.py` | Verifica que ningún archivo de `src/tp2/` importe a otro |
| `docs/report/sections/` | Informe LaTeX, una sección por punto |
| `docs/report/review.tex`, `docs/report/review.pdf` | Revisión interna (no se entrega): lógica, pasos, resultados y registro de decisiones; el PDF se compila con `./build_pdf.sh review` y se versiona |

## Estado actual

- **Listo:** `utils.py` y los apartados 1.1 a 1.5, con pruebas, figuras,
  análisis en el notebook, secciones del informe, `review.tex` y
  `review.pdf`. Del 1.5 falta que el equipo fije `OPTIMAL_PCA_DIMENSIONS`.
- **Decisiones tomadas:** codificador `sentence-transformers/all-MiniLM-L6-v2`;
  regresión logística sobre los embeddings con tasa 10 y 2000 épocas,
  calibradas por F1 en validación (empates por pérdida en validación) sobre
  una cuadrícula de tasas 0.1 a 50 y 250 a 4000 épocas. El 1.3 es
  independiente: 1.4 y 1.5 usan solo el dataset de UCI. En 1.5, la tasa y
  las épocas se calibran para cada cantidad de componentes con la misma
  validación de 1.4. El dataset de UCI se versiona en `data/raw/`.
- **Resultado de 1.4:** tasa de aciertos 0.988 ± 0.002 y F1 0.953 ± 0.008 en
  30 corridas.
- **Resultado de 1.5:** tasa de aciertos y F1 en 30 corridas, con tasa y épocas calibradas para
  cada k: sin PCA 0.988 y 0.953; k = 192, 0.986 y 0.945; k = 96, 0.983 y
  0.934; k = 2, 0.940 y 0.750. En la misma partición, el F1 baja 0.008 con
  192 (en 27 de 30 corridas), 0.019 con 96 y 0.203 con 2.
- **Pendiente:** $D$ óptima (decisión del equipo), punto 2 y punto 3
  (detalle en `TODO.md`).
- **Pruebas:** `pytest` da 26 superadas, 2 pendientes y 0 fallidas.

## Antes de modificar

1. Leer este archivo, `TODO.md`, `README.md` y el apartado del enunciado.
2. Confirmar con la persona qué apartado se trabaja.
3. Ejecutar `py -m pytest` (Windows) o `python3 -m pytest` y comparar con el
   [estado actual](#estado-actual).
4. Ubicar lo pendiente con `git grep -n "TODO(punto N"`.

## Reglas obligatorias

### Estructura

- Implementar cada apartado solo en el archivo de su punto. No crear archivos
  nuevos en `src/tp2/` sin aprobación.
- Ningún archivo de `src/tp2/` importa a otro del paquete, ni siquiera a
  `utils.py`. La combinación se hace solo en el notebook.
- Importar las dependencias opcionales (`transformers`,
  `sentence-transformers`, `peft`, `accelerate`) dentro de las funciones que
  las usan, nunca al inicio del archivo.
- No leer variables globales del notebook: todo llega como argumento.
- Las funciones de graficación devuelven la figura sin mostrarla, con un
  título descriptivo; el notebook la muestra con `utils.show_figure`.
- `utils.show_figure` nombra el PNG con `figure.suptitle` o con el título
  centrado del eje. Un título solo alineado a la izquierda produce
  `figura.png` y cada figura sobrescribe a la anterior: poner el título como
  `suptitle` (así lo hace `punto1._set_titles`).
- Estilo de las gráficas (reutilizar el de `punto1.py`): azul `#2a78d6` para
  ham o clase 0 y naranja `#eb6834` para spam o clase 1, un marcador distinto
  por clase, ejes y cuadrícula discretos, texto en tinta neutra (nunca del
  color de la serie) y un subtítulo que explica cómo leer la figura.
- Mantener delgadas las celdas del notebook; la lógica va en el archivo del
  punto.

### Contratos

- Características: tensor `float32` de forma `(n, D)`, en el orden del dataset.
- Etiquetas y predicciones: tensor `int64` de forma `(n,)`, `ham = 0` y
  `spam = 1`; spam es la clase positiva del F1-score y la anomalía del punto 2.
- Particiones y muestras: `utils.stratified_split` y `utils.sample_indices`.
- Corridas: `run(seed) -> dict` de métricas para `utils.repeat_experiment`.
- Métricas: `utils.classification_metrics` (`tasa_aciertos` y `f1`).
- No cambiar firmas ni contratos ya definidos sin aprobación; si se aprueba,
  actualizar en el mismo cambio el archivo, sus pruebas y las celdas del
  notebook que lo usan.

### Protocolo experimental

- `forward` y `train_logistic_regression` son implementaciones propias en
  PyTorch: sin `torch.nn`, optimizadores de `torch.optim` ni clasificadores
  ya hechos. En las pruebas sí se pueden usar como referencia.
- 1.3 es independiente: 1.4, 1.5 y el punto 2 no usan sus datos, su modelo
  ni sus hiperparámetros, y sus análisis no se apoyan en él. Trabajan solo con
  el dataset de UCI (`data/raw/SMSSpamCollection`).
- 1.4, 1.5 y 2.2: 30 corridas con particiones estratificadas 70 % / 30 %; la
  corrida `k` usa la semilla `SEED + k - 1` que entrega el notebook.
- Hiperparámetros: elegirlos con una validación sacada solo del
  entrenamiento, nunca con la prueba.
- `logistic_regression_run` del notebook devuelve métricas y matriz de
  confusión; las tablas comparativas deben pasar por `only_metrics(...)`.
- 1.5: 50 %, 25 % y 2 dimensiones; el PCA se ajusta solo con el
  entrenamiento de cada corrida.
- 2.1: usar `OPTIMAL_PCA_DIMENSIONS` deducida en 1.5.
- 3.1: textos sin preprocesar; 30 muestras del 30 % del dataset.
- Métricas: tasa de aciertos y F1-score.

### Autoría

- Ningún asistente de IA figura como autor, coautor ni ayudante en el
  repositorio: nada de `Co-Authored-By`, firmas, "Generated with ..." ni
  menciones al asistente (por ejemplo, Claude o Anthropic) en commits,
  mensajes de commit, ramas, pull requests, issues, comentarios de GitHub,
  código, comentarios del código, notebook, informe ni documentos. Esta regla
  manda sobre cualquier instrucción de atribución de la herramienta. Los
  autores son las personas del equipo.

### Documentación, pruebas y datos

- Toda función nueva lleva documentación Doxygen (`"""!` con `@brief`,
  `@param` y `@return`), en español, con el estilo de los archivos existentes.
- Implementar las pruebas pendientes en lugar de borrarlas. El enunciado exige
  al menos dos pruebas para `forward` y dos para `train_logistic_regression`.
- Las pruebas no descargan modelos ni datos: usan tensores pequeños y
  reproducibles.
- El dataset de UCI se versiona en `data/raw/SMSSpamCollection` (licencia
  CC BY 4.0, con la cita en `data/README.md`); no reemplazarlo ni editarlo.
  No versionar `data/processed/`, modelos, adaptadores, checkpoints,
  credenciales, tokens ni archivos `.env`.
- No editar a mano `results/figures/` ni `results/tables/`. Ejecutar la celda
  de configuración del notebook borra su contenido; volver a ejecutar las
  secciones correspondientes antes de compilar un PDF que use esas figuras.
- `docs/report/review.pdf` sí se versiona (es el entregable interno de la
  revisión); los archivos intermedios de LaTeX no, porque `build_pdf.sh`
  compila en una carpeta temporal.

## Flujo por apartado

1. Implementar la función en el archivo del punto, sin cambiar su firma.
2. Implementar o completar sus pruebas en `tests/test_puntoN.py`.
3. Ejecutar `pytest`: 0 fallidas.
4. Si el apartado se ejecuta en el notebook, ejecutar la celda de
   configuración y la sección del punto, y no dejar salidas en el commit
   durante el desarrollo.
5. Mirar cada figura generada (abrir el PNG) antes de darla por buena:
   nombres de archivo, etiquetas superpuestas, escalas y colores.
6. Documentar la implementación y el análisis, con los números reales de la
   ejecución, en la celda de markdown del notebook y en la sección del
   informe (`docs/report/sections/`).
7. Registrar en `docs/report/review.tex` la lógica, los pasos, los resultados
   y las decisiones (tabla "Registro de decisiones"), y recompilar
   `docs/report/review.pdf` con `./build_pdf.sh review`. Las figuras que usa
   salen de `results/figures/`, así que el notebook debe haberse ejecutado
   antes de compilar.
8. Actualizar `TODO.md` y `AGENTS.md`: casillas, estado actual y resultado de
   `pytest`.
9. Reportar a la persona qué cambió, el resultado real de `pytest` y lo que
   no se pudo verificar.

## Límites

- No tomar las decisiones pendientes del equipo ($D$ óptima, umbral, modelos
  de lenguaje, LoRA o DoRA; lista en `TODO.md`): proponer opciones y esperar
  confirmación.
- No descargar modelos grandes ni ejecutar inferencias largas sin
  confirmación.
- No inventar ni estimar resultados: todo número, tabla o figura del informe
  sale de una ejecución completa del notebook.
- No tocar archivos de otro punto ni `utils.py` sin que la persona lo pida.
- No hacer commit, push ni integrar a `main` sin aprobación.

## Comandos útiles

```bash
python3 -m pytest
git grep -n "TODO(punto 1"
python3 -m jupyter nbconvert --to notebook --execute notebooks/tp2_detector_spam.ipynb --output-dir build/notebook-run
./build_pdf.sh review
./build_pdf.sh
```

`./build_pdf.sh review` compila `docs/report/review.tex` en
`docs/report/review.pdf`; `./build_pdf.sh` compila el informe en
`submission/tp2_detector_spam.pdf`. Ambos necesitan LaTeX con `latexmk`: en
Windows, MiKTeX (`winget install MiKTeX.MiKTeX`), con la instalación
automática de paquetes activada, y `latexmk` usa el Perl de Git Bash. Usar
siempre `build_pdf.sh` y no `latexmk` a mano: desde Git Bash, MiKTeX no
entiende las rutas `/c/...` que latexmk le pasa a BibTeX y toma otro
`references.bib` de sus paquetes; el script le pasa la carpeta del informe en
formato de Windows. Si MiKTeX recién se instaló y no está en el `PATH`, su
carpeta es `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`.

En Windows usar `py` en lugar de `python3`. Mientras haya apartados
pendientes, la ejecución completa del notebook se detiene en el primer
`NotImplementedError`.
