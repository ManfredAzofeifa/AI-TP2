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
- **D óptima confirmada (24/10/2026):** `OPTIMAL_PCA_DIMENSIONS = 384`
  para la regresión logística; reduce a la mitad las 768 características.
  Conserva 96.16 % de varianza descriptiva y F1 0.9637 frente a 0.9641 sin
  PCA, con diferencia media pareada −0.0004; no es prueba de equivalencia.
- **2.1 implementado (24/10/2026):** densidades solo con ham del
  entrenamiento, 30 intervalos, umbral en el cuantil 5 % de un 20 % de ham
  reservado. El detector usa `ANOMALY_PCA_DIMENSIONS = 2`: entre los casos
  de 1.5, en la validación interna de los 30 entrenamientos 2 componentes
  dan F1 0.7085 ± 0.0318 y ganan en las 30; con 384, F1 0.0401 (el spam
  resulta más verosímil que el ham en 330 de 384 componentes).
- **2.2 completo (24/10/2026):** en las mismas 30 particiones, el detector
  obtiene tasa de aciertos 0.9195 ± 0.0084 y F1 0.7043 ± 0.0247; su F1 es
  menor que el de la regresión sin PCA (−0.2597) y que el de la regresión con
  k = 2 (−0.0802) en las 30 corridas. Detecta casi el mismo spam que la
  regresión con k = 2, pero marca más ham (4.9 % contra 1.9 %).
- **Pendiente:** punto 3.
- **Pruebas verificadas:** 48 superadas, 1 pendiente (3.1), 0 fallidas.
  La ejecución original de las 120 corridas UCI no dependió de 1.3.
  La auditoría repitió 1 y 2 en orden desde un kernel nuevo y las pruebas
  del notebook también dieron 48/1/0.
- **Auditoría de 1 y 2 (24/10/2026):** corregida la derivada de `forward`
  con logit cero (debe ser 1/4), con prueba de regresión. No cambia el
  gradiente manual del entrenamiento. El punto 2 está implementado, pero
  su conformidad literal sigue abierta: el enunciado pide reutilizar D=384
  y el equipo usa D=2. La selección global de D agrega particiones que se
  solapan; no afirmar que todas las pruebas quedaron aisladas de esa decisión.
- **Métricas auditadas:** 30 corridas por variante adicional. La regresión
  ampliada no mejora F1; el detector calibrado obtiene accuracy 0.9321 y F1
  0.7173, pero eleva el spam escapado de 64.0 a 78.1 por prueba. Entre 145
  y 181 textos de prueba ya aparecen en entrenamiento (media 9.70 %).
  Es evidencia exploratoria; conservar las constantes vigentes.
- **Paquete para el equipo (25/10/2026):** `scripts/run_puntos_1_2.py`,
  `docs/reproducibility.md` y `requirements-audit.txt`. El notebook del commit
  queda sin salidas de desarrollo; CSV, PNG y los dos PDF revisados se
  comparten. El informe sigue siendo preliminar. La guía permite regenerar
  la caché y los resultados sin rutas ni herramientas privadas de esta máquina.
- **Revisión interna:** `docs/report/review.tex` y su PDF se actualizan juntos.

## Próximos pasos (24/10/2026)

Lo que falta, en orden. Quien tome el repositorio empieza por aquí.

1. **Resolver con el docente el alcance del 2.1.** La persona confirmó
   conservar la configuración vigente y dejar las alternativas como sugerencias.
   El detector vigente usa D=2; la referencia literal con D=384 ya se evaluó
   en 30 corridas (F1 0.0386). La calibración adicional mejora el F1 medio
   del detector a 0.7173, pero deja escapar más spam; no se adoptó.
   Ampliar la cuadrícula de regresión no mejoró F1 (0.9636 frente a 0.9641).
   Puntos 1 y 2 y pruebas ya se ejecutaron desde un kernel nuevo, con salidas
   conservadas en la copia local ejecutada, tabla 48/1/0 y las 18 figuras
   regeneradas e inspeccionadas.
   La ejecución final de principio a fin sigue pendiente del punto 3.
   La configuración borra las exportaciones, incluidas las tablas de auditoría;
   para regenerar estas últimas activar la celda opcional con autorización.
   Mantener `RUN_METRIC_AUDIT = False` cuando no se quieran repetir esas corridas.
   Sin caché, MPNet puede requerir descarga y más de 10 minutos de extracción:
   pedir confirmación antes de descargar o ejecutar corridas largas.
2. **Avisar a todo el equipo** que el detector del punto 2 usa
   `ANOMALY_PCA_DIMENSIONS = 2` y no 384; la regresión conserva
   `OPTIMAL_PCA_DIMENSIONS = 384`.
3. **Punto 3 (10 puntos + 20 extra).** Primero decidir en equipo
   `LLM_MODELS`, el prompt y el tratamiento de respuestas ambiguas (ver
   [Decisiones pendientes](#decisiones-pendientes-del-equipo)); después
   implementar 3.1 y, si hay tiempo, 3.2 y 3.3.
4. **Informe:** completar `05-comparison.tex` (todos los métodos) y
   `06-conclusions.tex`, y el resumen de conclusiones al final del notebook.
   Dependen del punto 3.
5. **Confirmar los integrantes** (se copiaron del TP1) y la **fecha de
   entrega**: el enunciado dice martes 20 de octubre y este registro es del
   24; si hay prórroga, corregirla aquí, en `AGENTS.md` y en `README.md`.
6. **Cierre:** generar Doxygen, el PDF final con `./build_pdf.sh` y el `.zip`
   ([Checklist de entrega](#checklist-de-entrega)).

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
- Punto 2.1: la lectura literal pide reutilizar la D de 1.5 (384). La
  configuración vigente del equipo usa `ANOMALY_PCA_DIMENSIONS = 2`;
  documentar ambos casos y validar esa variante con el docente. Los
  histogramas usan solo ham; seleccionar D por F1 sí usa spam de validación.
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
  espera al punto 3.

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
- [x] Datos de las densidades (2.1, 24/10/2026): solo ham del entrenamiento.
- [x] Umbral (2.1, 24/10/2026): cuantil 5 % (`ANOMALY_THRESHOLD_QUANTILE`) de
      los puntajes de un 20 % de ham reservado (`ANOMALY_THRESHOLD_FRACTION`);
      `HISTOGRAM_BINS = 30`.
- [x] Dimensión del detector (2.1, 24/10/2026): `ANOMALY_PCA_DIMENSIONS = 2`,
      el mejor de los casos de 1.5 en validación interna; la regresión
      conserva `OPTIMAL_PCA_DIMENSIONS = 384`. Avisar al resto del equipo.
- [ ] Validar con el docente si acepta D=2 como variante del 2.1; conservar
      también el resultado de D=384 para responder la lectura literal.
- [x] Conservar la configuración vigente, confirmado el 25/10/2026. Las
      alternativas quedan documentadas sin adoptarse; no cambian las constantes.
- [x] Compartir el ejecutor parcial y la guía de reproducción, con versiones
      directas observadas; excluir cachés, pesos de modelos y entornos locales.
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
  - [x] Prueba adicional del gradiente de `forward` en logit cero; reproduce
        el fallo anterior y verifica las derivadas de pesos y sesgo.
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

Usa las características (1.4), `fit_pca` y `transform_pca` (1.5). El
detector trabaja con `ANOMALY_PCA_DIMENSIONS = 2` componentes, elegidas entre
los casos de 1.5 con la validación interna de los 30 entrenamientos.

- [x] **2.1 (20)** `fit_histogram_densities`, `per_dimension_log_likelihood`,
      `log_likelihood`, `select_threshold`, `fit_anomaly_detector`,
      `detector_metrics`, `evaluate_detector_dimensions`,
      `component_log_likelihood_gap` y cuatro gráficas. Selección de D,
      densidades, puntajes, región normal y análisis (24/10/2026).
  - [x] 15 pruebas en `tests/test_punto2.py`, incluida
        `test_histogram_densities_integrate_to_one`.
- [x] **2.2 (10)** 30 corridas 70/30 con las particiones del punto 1;
      tablas de métricas, errores por clase, diferencias pareadas frente a la
      regresión sin PCA y con k = 2, dos diagramas de caja y análisis.
- [x] Informe: `02-punto2-deteccion-anomalias.tex` (2.1 y 2.2).

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
- [x] Informe: `00-introduction.tex`, con alcance auditado de 1 y 2.
- [ ] Informe: `05-comparison.tex` y `06-conclusions.tex`, después del punto 3.
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
      ni líneas fuera del margen en la pasada final (24/10/2026); con el
      punto 2 eran 23 páginas. La versión auditada y la revisión interna
      compilan y se revisaron visualmente. Es preliminar: falta el punto 3.
- [ ] Generar `submission/tp2_detector_spam.pdf` con `./build_pdf.sh` en la
      entrega, con todos los puntos terminados.
- [x] Migración del notebook: `nbformat`, IDs únicos y AST válidos;
      pruebas externas sobre las mismas definiciones, 30 superadas y 2 pendientes.
- [x] Celda de pruebas del notebook (antes del punto 2): 30 superadas, 2
      pendientes y 0 fallidas.
      La inspección estática del IDE no resuelve algunos imports de tests y
      atributos del módulo dinámico; la ejecución real los verifica.
- [x] MPNet con forma (5574, 768), vectores finitos de norma 1; 30 corridas
      por método (sin PCA y k=384/192/2), semillas 2026 a 2055.
- [x] 2.1 ejecutado con la caché MPNet (misma revisión) en otra máquina; sus
      figuras y la tabla de selección se exportaron sin reejecutar el punto 1.
- [x] 2.2 ejecutado junto con las 120 corridas del punto 1 en Linux: las
      tablas de métricas del punto 1 coinciden exactamente con las versionadas
      (solo cambian pérdidas de calibración en el séptimo decimal y un empate
      de épocas en la corrida 23 con k = 2, sin cambio de métricas). No se
      sobrescribieron los archivos del punto 1.
- [x] Auditoría: puntos 1 y 2 completos desde un kernel nuevo; 48/1/0 en
      la celda de pruebas, 18 figuras regeneradas e inspeccionadas, diagramas
      de caja con estilo común y medianas neutras. Sin descargas.
- [x] 30 corridas por alternativa adicional, con selección interna de cada
      entrenamiento y tablas exportadas desde la celda opcional del notebook.
- [ ] Ejecución final de principio a fin después de implementar el punto 3.
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
