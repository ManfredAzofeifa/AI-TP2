# TODO - TP2 Detector de spam

Guía de trabajo para las personas del equipo: qué falta, qué reglas seguir y
qué se entrega. Marcar las casillas a medida que se completan, en el mismo
pull request que hace el cambio. Los agentes de IA siguen
[AGENTS.md](AGENTS.md); si cambia el estado, una tarea o una regla, actualizar
también ese archivo y la revisión interna (`docs/report/review.tex` y su PDF).

- **Entrega:** martes 20 de octubre de 2026, por TEC Digital.
- **Modo de trabajo:** grupos de 3 personas.
- **Entregable:** un `.zip` con el código fuente LaTeX, el PDF y el notebook
  debidamente documentado (ver [Checklist de entrega](#checklist-de-entrega)).
- **Enunciado:** `docs/assignment/TrabajoPractico2_detector_ataques_supervisado_no_supervisado.pdf`.
- **Dataset:** SMSSpamCollection (UCI), mensajes SMS etiquetados `ham` o `spam`.

## Estado actual

- **Listo:** infraestructura compartida (`src/tp2/utils.py`) y puntos 1.1 a
  1.4 del punto 1, con sus pruebas, figuras, análisis en el notebook y
  secciones del informe.
- **Resultado de 1.4:** regresión logística sobre los embeddings de
  `all-MiniLM-L6-v2`, tasa 10 y 2000 épocas: tasa de aciertos 0.988 ± 0.002 y
  F1-score 0.953 ± 0.008 en 30 corridas.
- **Pendiente:** 1.5, punto 2 y punto 3.
- **Pruebas:** `pytest` da 23 superadas, 3 pendientes y 0 fallidas.
- **Revisión interna:** `docs/report/review.tex` y su PDF compilado,
  `docs/report/review.pdf`, explican la lógica, los pasos, los resultados y
  las decisiones de todo lo realizado.

## Cómo trabajar

### Antes de empezar

1. Leer el enunciado, `README.md`, este archivo y `CONTRIBUTING.md`.
2. Instalar el entorno (`README.md`, "Preparación del entorno").
3. Ejecutar `py -m pytest` (Windows) o `python3 -m pytest` y comparar con el
   [estado actual](#estado-actual).
4. Crear una rama corta para el apartado, por ejemplo `feat/punto1-pca`.
5. Buscar lo pendiente con `git grep -n "TODO(punto 1"` (cambiar el número
   según el punto).

### Al terminar cada función

1. Reemplazar el `NotImplementedError` por la implementación.
2. Implementar o completar su prueba en `tests/test_puntoN.py`.
3. Ejecutar `pytest`: no debe haber pruebas fallidas.
4. Ejecutar la celda de configuración y la sección del punto en el notebook.
5. Escribir la descripción y el análisis en el notebook y en la sección del
   informe.
6. Registrar en `docs/report/review.tex` la lógica, los pasos, los resultados
   y las decisiones del apartado, y recompilar el PDF con
   `./build_pdf.sh review` (el `.tex` y `review.pdf` se versionan juntos).
7. Marcar la casilla y actualizar el [estado actual](#estado-actual) en este
   archivo y en `AGENTS.md`, y abrir el pull request.

## Reglas del proyecto

### Estructura

- Cada punto vive solo en su archivo: `src/tp2/punto1.py`, `punto2.py` o
  `punto3.py`. Lo compartido (datos, particiones, métricas, corridas,
  figuras, tablas y pruebas en el notebook) está en `src/tp2/utils.py`.
- Ningún archivo de `src/tp2/` importa a otro, ni siquiera a `utils.py`.
  Todo se combina en el notebook. `tests/test_module_independence.py` falla
  si se rompe esta regla.
- Si dos puntos necesitan la misma lógica, se pasa como argumento desde el
  notebook o se acuerda en equipo moverla a `utils.py`.
- Las dependencias opcionales (`transformers`, `sentence-transformers`,
  `peft`, `accelerate`) se importan dentro de las funciones que las usan.
- Los archivos no leen variables globales del notebook: todo valor
  compartido llega como argumento.
- Las funciones de graficación devuelven la figura sin mostrarla y con un
  título descriptivo; el notebook la muestra con `utils.show_figure`, que
  nombra el PNG con el título de la figura (`figure.suptitle`) o con el título
  centrado del eje. Un título solo alineado a la izquierda produce
  `figura.png`.
- Las gráficas del punto 1 siguen un estilo común que conviene reutilizar:
  azul para ham o clase 0 y naranja para spam o clase 1 (se distinguen con
  daltonismo), un marcador distinto por clase, ejes discretos y un subtítulo
  que explica cómo leer la figura.
- Las celdas del notebook se mantienen delgadas: llamadas a los archivos de
  cada punto y presentación de resultados.

### Contratos entre archivos

| Dato | Formato |
|---|---|
| Características | tensor `float32` de forma `(n, D)`, en el orden del dataset |
| Etiquetas y predicciones | tensor `int64` de forma `(n,)`: `ham = 0`, `spam = 1` |
| Clase positiva del F1-score y anomalía del punto 2 | `spam = 1` |
| Particiones y muestras | índices de `utils.stratified_split` y `utils.sample_indices` |
| Una corrida | función `run(seed) -> dict` de métricas para `utils.repeat_experiment` |
| Métricas | `utils.classification_metrics`: `tasa_aciertos` y `f1` |

Las firmas de las funciones pendientes ya están definidas y documentadas. Si
un contrato cambia, avisar al equipo y actualizar en el mismo pull request el
archivo, sus pruebas y las celdas del notebook que lo usan.

### Protocolo experimental (del enunciado)

- La regresión logística (`forward` y `train_logistic_regression`) es una
  implementación propia en PyTorch, sin `torch.nn`, optimizadores ni
  clasificadores ya hechos.
- Puntos 1.4, 1.5 y 2.2: 30 corridas, cada una con una partición
  estratificada 70 % / 30 %. La corrida `k` usa la semilla `SEED + k - 1`
  que entrega el notebook; no usar semillas propias dentro de los archivos.
- Los hiperparámetros se eligen con una validación sacada solo del
  entrenamiento, nunca con la prueba (así se calibró 1.4).
- Punto 1.5: reducir a 50 %, 25 % y 2 dimensiones. El PCA de cada corrida se
  ajusta solo con el entrenamiento y se aplica a la prueba.
- Punto 2.1: usar la $D$ óptima deducida en 1.5 (`OPTIMAL_PCA_DIMENSIONS`).
- Punto 3.1: textos sin preprocesar; 30 muestras del 30 % del dataset.
- Métricas en todos los puntos: tasa de aciertos y F1-score.

### Documentación, pruebas y datos

- Toda función lleva documentación Doxygen (`"""!` con `@brief`, `@param` y
  `@return`).
- No borrar pruebas pendientes: implementarlas. Una prueba o función sin
  terminar lanza `NotImplementedError` y se reporta como pendiente.
- No versionar `data/raw/`, `data/processed/`, modelos, adaptadores ni
  checkpoints. Nunca incluir credenciales, tokens ni archivos `.env`.
- No editar a mano `results/figures/` ni `results/tables/`: el notebook borra
  y regenera su contenido en cada ejecución.
- Todo número, tabla o figura del informe debe provenir de una ejecución
  completa del notebook.

## Decisiones pendientes del equipo

Registrar cada decisión en la celda de configuración del notebook, en la
sección "Objetivo y protocolo" del notebook y en el informe.

- [ ] Confirmar los integrantes; se copiaron del TP1 (`README.md`,
      notebook y `docs/report/main.tex`).
- [x] `ENCODER_NAME` (1.4): `sentence-transformers/all-MiniLM-L6-v2`
      (19/10/2026), con `sentence-transformers==6.1.0` en `requirements.txt`.
- [x] `LR_LEARNING_RATE` y `LR_EPOCHS` (1.4): calibrados por F1 en una
      validación del entrenamiento, con empates resueltos por la pérdida en
      validación: tasa 10 y 2000 épocas.
- [ ] `OPTIMAL_PCA_DIMENSIONS` (fin de 1.5, se usa en 2.1).
- [ ] Datos con los que se construyen las densidades (2.1): solo ham del
      entrenamiento (lo que hace hoy el notebook) o todo el entrenamiento.
- [ ] Criterio del umbral (2.1) y `HISTOGRAM_BINS`.
- [ ] `LLM_MODELS` (3.1, al menos uno; 3.2 extra, al menos dos más), el
      prompt y qué hacer con respuestas ambiguas. Son 5 574 mensajes por
      modelo; preferir modelos que corran en las computadoras del equipo.
- [ ] `LORA_MODEL` y LoRA o DoRA (3.3, extra).

## Punto 1 - `src/tp2/punto1.py` (60 puntos)

Es la ruta crítica: el punto 2 necesita sus características (1.4) y su PCA
(1.5) para ejecutarse. Ya implementadas también: `predict_proba`, `predict` y
`components_for_fraction`.

- [x] **1.1 (10)** `forward`: logits $Xw + b$ y sigmoide estable propia.
  - [x] 1.1a: `test_forward_zero_parameters_returns_one_half` y
        `test_forward_matches_sigmoid_and_is_stable`.
- [x] **1.2 (10)** `train_logistic_regression`: descenso del gradiente por
      lotes completos sobre la entropía cruzada, con gradiente exacto,
      inicialización con `seed` e historial de pérdida (épocas + 1 valores).
  - [x] 1.2a: `test_train_follows_cross_entropy_gradient` y
        `test_train_separates_linearly_separable_data`.
- [x] **1.3 (10)** `make_linearly_separable` (frontera verdadera fija
      0.6 x1 + 0.8 x2 - 0.5 = 0, margen 1), `describe_boundary`,
      `plot_dataset`, `plot_loss_history`, `plot_boundary_evolution` y
      `plot_decision_boundary`. Cuatro figuras, dos tablas y análisis.
  - [x] `test_make_linearly_separable_shapes_and_reproducibility` y
        `test_boundary_plots_return_figures`.
- [x] **1.4 (10)** `load_encoder`, `extract_features`,
      `grid_search_logistic_regression`, `plot_calibration`,
      `plot_run_metrics` y `plot_confusion_matrix`. Calibración, 30 corridas
      70/30, tres figuras, tablas y análisis.
  - [x] `test_extract_features_keeps_order_and_contract`,
        `test_grid_search_covers_grid_and_orders_by_score` y
        `test_run_plots_return_figures`.
- [ ] **1.5 (20)** `fit_pca`, `transform_pca`, `plot_explained_variance` y
      `plot_pca_projection`. Reducir a 50 %, 25 % y 2 dimensiones, graficar
      todo el dataset, analizar la clasificación y deducir la $D$ óptima
      (`OPTIMAL_PCA_DIMENSIONS`).
  - [ ] Recomendada: `test_fit_pca_recovers_dominant_direction`.
- [ ] Informe: `01-punto1-regresion-logistica.tex` (1.1 a 1.4 y
      `04-unit-tests.tex` ya escritos; falta 1.5).
- Nota para 1.5: `logistic_regression_run` ya acepta `n_components` y guarda
  la matriz de confusión de cada corrida; las tablas comparativas usan
  `only_metrics(...)` para mostrar solo `tasa_aciertos` y `f1`.

## Punto 2 - `src/tp2/punto2.py` (30 puntos)

Ya implementada: `predict_anomalies` (puntaje menor que el umbral = spam). Se
puede programar y probar desde ya con tensores sintéticos; para ejecutarlo en
el notebook necesita las características (1.4), `fit_pca` y `transform_pca`
(1.5) y `OPTIMAL_PCA_DIMENSIONS`. Mientras tanto se puede fijar una $D$
provisional.

- [ ] **2.1 (20)** `fit_histogram_densities`, `log_likelihood`,
      `select_threshold`, `plot_densities` y `plot_score_distribution`.
      Graficar y comentar las densidades; definir el umbral sobre la suma de
      verosimilitudes de todas las densidades.
  - [ ] Recomendada: `test_histogram_densities_integrate_to_one`.
- [ ] **2.2 (10)** 30 corridas 70/30 (la celda ya está en el notebook);
      reportar y comentar frente a los resultados del punto 1.
- [ ] Informe: `02-punto2-deteccion-anomalias.tex`.

## Punto 3 - `src/tp2/punto3.py` (10 puntos y 20 extra)

Es independiente de los otros puntos: solo usa los textos sin preprocesar y
`utils.py`. El notebook clasifica cada mensaje una sola vez por modelo, guarda
las predicciones en `data/processed/` y las 30 muestras del 30 % las
reutilizan. Con un modelo fijo y decodificación determinista, el resultado es
el mismo que clasificar cada muestra, con 30 veces menos inferencias.

- [ ] **3.1 (10)** `load_llm`, `parse_label` y `classify_messages`; al menos
      un modelo en `LLM_MODELS`. Reportar y comentar F1-score y tasa de
      aciertos frente a los puntos 1 y 2.
  - [ ] Recomendada: `test_parse_label_handles_responses`.
- [ ] **3.2 (10 extra)** al menos dos modelos más en `LLM_MODELS`.
- [ ] **3.3 (10 extra)** `finetune_with_lora` y `classify_finetuned` con
      LoRA o DoRA; fijar `LORA_MODEL`; reportar y analizar. Si no se hace, la
      celda se omite sola y el notebook sigue.
- [ ] Informe: `03-punto3-modelos-lenguaje.tex`.

## Integración

- [ ] Completar las constantes de la celda de configuración del notebook.
- [ ] Analizar la sección "Comparación de resultados" del notebook.
- [ ] Informe: `00-introduction.tex`, `05-comparison.tex` y
      `06-conclusions.tex`.
- [ ] Mantener al día el "Estado" del encabezado del notebook y la lista
      final de pendientes.
- [ ] Generar Doxygen (`README.md`, "Documentación interna") y copiar
      `build/doxygen/html` a `submission/doxygen/html`.
- [ ] Ejecución final: **Kernel → Restart Kernel and Run All Cells…** y un
      commit deliberado con las salidas, figuras y tablas finales.

## Verificación pendiente

Lo siguiente no se pudo comprobar todavía y debe revisarse en la primera
ejecución:

- [x] Descarga y lectura del dataset real: 5 574 mensajes, 4 827 ham
      (86.6 %) y 747 spam (13.4 %), como indica UCI.
- [x] LaTeX instalado (MiKTeX, con `latexmk`) y `review.tex` compilado en
      `docs/report/review.pdf`.
- [x] El informe completo (`main.tex`) compila: 12 páginas, sin citas sin
      resolver ni líneas fuera del margen (19/10/2026).
- [ ] Generar `submission/tp2_detector_spam.pdf` con `./build_pdf.sh` en la
      entrega, con todos los puntos terminados.
- [ ] Generación de Doxygen (no estaba instalado).

## Checklist de entrega

- [ ] El notebook corre de principio a fin sin errores desde un kernel limpio.
- [ ] `pytest` no tiene pruebas fallidas y las cuatro pruebas obligatorias de
      la regresión logística siguen pasando.
- [ ] PDF generado con `./build_pdf.sh` en `submission/tp2_detector_spam.pdf`.
- [ ] Doxygen generado y copiado a `submission/doxygen/html`.
- [ ] El `.zip` incluye: `notebooks/tp2_detector_spam.ipynb` ejecutado,
      `src/tp2/` y `tests/` (el notebook depende de ellos),
      `requirements.txt`, `docs/report/`, `results/figures/` y el PDF.
- [ ] Subir el `.zip` a TEC Digital antes del martes 20 de octubre.
