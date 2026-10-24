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

- **Migración (24/10/2026):** utilidades y puntos 1 a 3 se concentran en
  `notebooks/tp2_detector_spam.ipynb`; las pruebas cargan sus definiciones.
- **1.1–1.2 confirmado:** logística básica con sesgo, `Xw+b`, sigmoide y
  entropía cruzada; mantener al menos dos pruebas por función.
- **1.3 aislado:** datos sintéticos, entrenamiento y parámetros independientes.
- **Codificador confirmado:** `sentence-transformers/all-mpnet-base-v2`, D=768;
  en 1.5 se comparan 384, 192 y 2 componentes con el caso sin PCA.
- **Dataset auditado:** idéntico al archivo oficial UCI; 5574 mensajes,
  4827 ham y 747 spam. Tiene etiqueta y texto, sin características originales.
  Hay 5171 textos únicos y 403 filas repetidas adicionales; la partición por
  filas puede compartir textos idénticos entre entrenamiento y prueba.
- **Corrección de evaluación:** calibración interna en cada una de las 30
  corridas y cada k; se usa el ganador automáticamente. Las cifras anteriores
  MiniLM/384 quedan históricas; ejecución MPNet/768 completa: 30 corridas
  por método. F1 sin PCA 0.9641 ± 0.0063; con 384 componentes 0.9637 ±
  0.0075; con 192, 0.9596 ± 0.0082; con 2, 0.7845 ± 0.0189.
  Tasa de aciertos sin PCA 0.9906 ± 0.0016.
- **D óptima confirmada (24/10/2026):** `OPTIMAL_PCA_DIMENSIONS = 384`;
  reduce a la mitad las 768 características y fija 384 histogramas para 2.1.
  Conserva 96.16 % de varianza descriptiva y F1 0.9637 frente a 0.9641 sin
  PCA, con diferencia media pareada −0.0004; no es prueba de equivalencia.
- **Pendiente:** punto 2 y punto 3.
- **Pruebas verificadas:** 30 superadas, 2 pendientes (2.1 y 3.1), 0 fallidas.
  Las 120 corridas UCI se completaron sin ejecutar previamente 1.3.
- **Revisión interna:** `docs/report/review.tex` y su PDF se actualizan juntos.

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

- Toda la implementación vive en el único notebook, organizada por apartados;
  se retiran los archivos por punto y las utilidades separadas.
- Separar definiciones (`# Definiciones:`), configuración, experimentos y
  análisis. `tests/notebook_loader.py` ejecuta solo definiciones para probarlas.
- Las dependencias opcionales (`transformers`, `sentence-transformers`,
  `peft`, `accelerate`) se importan dentro de las funciones que las usan.
- Las funciones reciben datos y parámetros como argumentos; no dependen de
  resultados globales de otras secciones.
- Las funciones de graficación devuelven la figura sin mostrarla y con un
  título descriptivo; el notebook la muestra con `show_figure`, que
  nombra el PNG con el título de la figura (`figure.suptitle`) o con el título
  centrado del eje. Un título solo alineado a la izquierda produce
  `figura.png`.
- Las gráficas del punto 1 siguen un estilo común que conviene reutilizar:
  azul para ham o clase 0 y naranja para spam o clase 1 (se distinguen con
  daltonismo), un marcador distinto por clase, ejes discretos y un subtítulo
  que explica cómo leer la figura.
- El notebook incluye la lógica y sus llamadas, en celdas separadas y con
  documentación Doxygen.

### Contratos del notebook

| Dato | Formato |
|---|---|
| Características | tensor `float32` de forma `(n, D)`, en el orden del dataset |
| Etiquetas y predicciones | tensor `int64` de forma `(n,)`: `ham = 0`, `spam = 1` |
| Clase positiva del F1-score y anomalía del punto 2 | `spam = 1` |
| Particiones y muestras | índices de `stratified_split` y `sample_indices` |
| Una corrida | función `run(seed) -> dict` de métricas para `repeat_experiment` |
| Métricas | `classification_metrics`: `tasa_aciertos` y `f1` |

Las firmas de las funciones pendientes ya están definidas y documentadas. Si
un contrato cambia, avisar al equipo y actualizar en el mismo pull request el
archivo, sus pruebas y las celdas del notebook que lo usan.

### Protocolo experimental (del enunciado)

- La regresión logística (`forward` y `train_logistic_regression`) es una
  implementación propia en PyTorch, sin `torch.nn`, optimizadores ni
  clasificadores ya hechos.
- Puntos 1.4, 1.5 y 2.2: 30 corridas, cada una con una partición
  estratificada 70 % / 30 %. La corrida `k` usa la semilla `SEED + k - 1`
  que entrega el notebook; no fijar semillas distintas dentro de las funciones.
- El 1.3 (datos sintéticos) es independiente: 1.4, 1.5 y el punto 2 trabajan
  solo con el dataset de UCI y no usan nada del 1.3.
- Calibrar hiperparámetros dentro del entrenamiento de cada corrida y para
  cada k; usar automáticamente el ganador, sin consultar su prueba.
- Punto 1.5: reducir a 50 %, 25 % y 2 dimensiones (384, 192 y 2 de 768).
  Ajustar PCA con ajuste interno durante la calibración y con el 70 % de
  entrenamiento al evaluar; usar las mismas particiones para comparar.
- Punto 2.1: usar la $D$ confirmada a partir de 1.5
  (`OPTIMAL_PCA_DIMENSIONS = 384`), con un histograma por componente.
- Punto 3.1: textos sin preprocesar; 30 muestras del 30 % del dataset.
- Métricas en todos los puntos: tasa de aciertos y F1-score.

### Autoría

- Ningún asistente de IA figura como autor, coautor ni ayudante en el
  repositorio: nada de `Co-Authored-By`, firmas ni "Generated with ..." en
  commits, pull requests, issues, código, notebook ni informe.

### Documentación, pruebas y datos

- Toda función lleva documentación Doxygen (`"""!` con `@brief`, `@param` y
  `@return`).
- No borrar pruebas pendientes: implementarlas. Una prueba o función sin
  terminar lanza `NotImplementedError` y se reporta como pendiente.
- El dataset de UCI sí se versiona (`data/raw/SMSSpamCollection`) para que
  todo el equipo use el mismo archivo. No versionar `data/processed/`,
  modelos, adaptadores ni checkpoints. Nunca incluir credenciales, tokens ni archivos `.env`.
- No editar a mano `results/figures/` ni `results/tables/`: el notebook borra
  y regenera su contenido en cada ejecución.
- Toda métrica, tabla o figura experimental del informe proviene de la
  ejecución completa de su sección; la ejecución final de todo el notebook
  espera a los puntos 2 y 3.

## Decisiones pendientes del equipo

Registrar cada decisión en la celda de configuración del notebook, en la
sección "Objetivo y protocolo" del notebook y en el informe.

- [ ] Confirmar los integrantes; se copiaron del TP1 (`README.md`,
      notebook y `docs/report/main.tex`).
- [x] `ENCODER_NAME`: `sentence-transformers/all-mpnet-base-v2`, D=768
      (24/10/2026). Sustituye MiniLM/384; descarga y ejecución autorizadas.
- [x] Corregir protocolo 1.4–1.5: elegir tasa y épocas por validación interna
      para cada corrida y k, con F1 y pérdida como desempate.
- [x] `OPTIMAL_PCA_DIMENSIONS = 384` (24/10/2026): decisión del equipo
      basada en MPNet/768; reduce a la mitad las características, conserva
      96.16 % de varianza descriptiva y F1 medio 0.9637 frente a 0.9641 sin
      PCA (diferencia pareada −0.0004). No implica equivalencia estadística.
- [ ] Datos con los que se construyen las densidades (2.1): solo ham del
      entrenamiento (lo que hace hoy el notebook) o todo el entrenamiento.
- [ ] Criterio del umbral (2.1) y `HISTOGRAM_BINS`.
- [ ] `LLM_MODELS` (3.1, al menos uno; 3.2 extra, al menos dos más), el
      prompt y qué hacer con respuestas ambiguas. Son 5 574 mensajes por
      modelo; preferir modelos que corran en las computadoras del equipo.
- [ ] `LORA_MODEL` y LoRA o DoRA (3.3, extra).

## Punto 1 - sección 1 del notebook (60 puntos)

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
      `plot_run_metrics` y `plot_confusion_matrix`. Calibración interna y 30 corridas
      70/30 con MPNet/768 verificadas, con figuras, tablas y análisis.
  - [x] `test_extract_features_keeps_order_and_contract`,
        `test_grid_search_covers_grid_and_orders_by_score` y
        `test_run_plots_return_figures`.
- [x] **1.5 (20)** `fit_pca` (SVD de los datos centrados, signo fijo),
      `transform_pca`, `plot_explained_variance` y `plot_pca_projection`.
      Reducción a 50 %, 25 % y 2 dimensiones, calibración de tasa y épocas
      para cada k y corrida; 30 corridas, figuras, tablas y análisis
      verificados con MPNet/768.
  - [x] `test_fit_pca_recovers_dominant_direction`,
        `test_transform_pca_matches_explained_variance` y
        `test_pca_plots_return_figures`.
  - [x] Fijar `OPTIMAL_PCA_DIMENSIONS = 384` (decisión del equipo,
        24/10/2026), con la justificación en el notebook y el informe.
- [x] Informe: 1.4–1.5 y `04-unit-tests.tex` actualizados con los nuevos
      resultados y la decisión confirmada de D = 384.
- Nota: `logistic_regression_run` enlaza los datos y la cuadrícula a
  `calibrated_logistic_regression_run`; recibe semilla y `n_components` y
  devuelve métricas, confusión y parámetros elegidos dentro de la corrida.
  Las tablas comparativas usan `only_metrics(...)`.

## Punto 2 - sección 2 del notebook (30 puntos)

Ya implementada: `predict_anomalies` (puntaje menor que el umbral = spam). Se
puede programar y probar desde ya con tensores sintéticos; para ejecutarlo en
el notebook necesita las características (1.4), `fit_pca` y `transform_pca`
(1.5) y `OPTIMAL_PCA_DIMENSIONS = 384`, ya confirmada. Se estimarán
384 histogramas, uno por componente; la implementación y evaluación del
punto 2 siguen pendientes.

- [ ] **2.1 (20)** `fit_histogram_densities`, `log_likelihood`,
      `select_threshold`, `plot_densities` y `plot_score_distribution`.
      Graficar y comentar las densidades; definir el umbral sobre la suma de
      verosimilitudes de todas las densidades.
  - [ ] Recomendada: `test_histogram_densities_integrate_to_one`.
- [ ] **2.2 (10)** 30 corridas 70/30 (la celda ya está en el notebook);
      reportar y comentar frente a los resultados del punto 1.
- [ ] Informe: `02-punto2-deteccion-anomalias.tex`.

## Punto 3 - sección 3 del notebook (10 puntos y 20 extra)

Es independiente de los otros puntos: solo usa los textos sin preprocesar y
las funciones compartidas del cuaderno. El notebook clasifica cada mensaje una sola vez por modelo, guarda
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

Registro de verificaciones realizadas y trabajo pendiente:

- [x] Descarga y lectura del dataset real: 5 574 mensajes, 4 827 ham
      (86.6 %) y 747 spam (13.4 %), como indica UCI. Se descarga en
      `data/raw/SMSSpamCollection`, que se versiona desde el 23/10/2026.
- [x] LaTeX instalado (MiKTeX, con `latexmk`) y `review.tex` compilado en
      `docs/report/review.pdf`.
- [x] El informe (`main.tex`) compila: 17 páginas, sin citas sin resolver
      ni líneas fuera del margen en la pasada final (24/10/2026); es una
      versión preliminar porque los puntos 2 y 3 siguen pendientes.
- [ ] Generar `submission/tp2_detector_spam.pdf` con `./build_pdf.sh` en la
      entrega, con todos los puntos terminados.
- [x] Migración del notebook: `nbformat`, IDs únicos y AST válidos;
      pruebas externas sobre las mismas definiciones, 30 superadas y 2 pendientes.
- [x] Celda de pruebas del notebook: 30 superadas, 2 pendientes y 0 fallidas.
      La inspección estática del IDE no resuelve algunos imports de tests y
      atributos del módulo dinámico; la ejecución real los verifica.
- [x] MPNet con forma (5574, 768), vectores finitos de norma 1; 30 corridas
      por método (sin PCA y k=384/192/2), semillas 2026 a 2055.
- [ ] Generación de Doxygen (no estaba instalado).

## Checklist de entrega

- [ ] El notebook corre de principio a fin sin errores desde un kernel limpio.
- [ ] `pytest` no tiene pruebas fallidas y las cuatro pruebas obligatorias de
      la regresión logística siguen pasando.
- [ ] PDF generado con `./build_pdf.sh` en `submission/tp2_detector_spam.pdf`.
- [ ] Doxygen generado y copiado a `submission/doxygen/html`.
- [ ] El `.zip` incluye: `notebooks/tp2_detector_spam.ipynb` ejecutado,
      `tests/` (verificación externa; el notebook concentra la implementación),
      `requirements.txt`, `docs/report/`, `results/figures/` y el PDF.
- [ ] Subir el `.zip` a TEC Digital antes del martes 20 de octubre.
