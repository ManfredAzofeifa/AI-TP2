# TODO - TP2 Detector de spam

Guía de trabajo para las personas del equipo y para cualquier agente de IA
que trabaje en el repositorio. Leer completo antes de modificar archivos y
marcar las casillas a medida que se completan, en el mismo pull request que
hace el cambio.

- **Entrega:** martes 20 de octubre de 2026, por TEC Digital.
- **Modo de trabajo:** grupos de 3 personas.
- **Entregable:** un `.zip` con el código fuente LaTeX, el PDF y el notebook
  debidamente documentado (ver [Checklist de entrega](#checklist-de-entrega)).
- **Enunciado:** `docs/assignment/TrabajoPractico2_detector_ataques_supervisado_no_supervisado.pdf`.
- **Dataset:** SMSSpamCollection (UCI), mensajes SMS etiquetados `ham` o `spam`.

## Organización: un archivo por punto

| Punto | Valor | Archivo | Pruebas | Sección del notebook | Sección del informe | Responsable |
|---|---|---|---|---|---|---|
| 1. Regresión logística | 60 | `src/tp2/punto1.py` | `tests/test_punto1.py` | 1 | `01-punto1-regresion-logistica.tex`, `04-unit-tests.tex` | por asignar |
| 2. Detección de anomalías | 30 | `src/tp2/punto2.py` | `tests/test_punto2.py` | 2 | `02-punto2-deteccion-anomalias.tex` | por asignar |
| 3. Modelos de lenguaje | 10 + 20 extra | `src/tp2/punto3.py` | `tests/test_punto3.py` | 3 | `03-punto3-modelos-lenguaje.tex` | por asignar |
| Integración | — | `src/tp2/utils.py` (listo), notebook | `tests/test_utils.py` | Configuración, comparación y pruebas | `00`, `05`, `06` | por asignar |

Se eligió un archivo por punto (aunque se podría dividir más) para que con
tres personas la asignación sea directa: cada persona es dueña de su archivo,
sus pruebas, su sección del notebook y su sección del informe.

**Carga de trabajo.** El punto 1 vale 60 puntos y es la ruta crítica: el
punto 2 necesita sus características (1.4) y su PCA (1.5) para ejecutarse.
Sugerencia: quien tenga el punto 3, el más liviano, asume también la
integración y apoya el punto 1 si se atrasa.

## Restricciones (obligatorias para personas y agentes)

### Estructura

1. Cada punto vive solo en su archivo: `punto1.py`, `punto2.py` o
   `punto3.py`. La infraestructura compartida (datos, particiones, métricas,
   corridas, figuras, tablas y pruebas en el notebook) está en `utils.py`.
2. Ningún archivo de `src/tp2/` importa a otro del paquete, ni siquiera a
   `utils.py`. Todo se combina en el notebook.
   `tests/test_module_independence.py` falla si se rompe esta regla.
3. Si dos puntos necesitan la misma lógica, se pasa como argumento desde el
   notebook (una función o un tensor) o se acuerda en equipo moverla a
   `utils.py`.
4. Las dependencias opcionales (`transformers`, `sentence-transformers`,
   `peft`, `accelerate`) se importan dentro de las funciones que las usan,
   nunca al inicio del archivo.
5. Los archivos no leen variables globales del notebook: todo valor
   compartido llega como argumento.
6. Las funciones de graficación devuelven la figura sin mostrarla y con un
   título descriptivo; el notebook la muestra con `utils.show_figure`, que la
   exporta a `results/figures/`.
7. Las celdas del notebook se mantienen delgadas: llamadas a los archivos de
   cada punto y presentación de resultados. Si una celda crece, la lógica se
   mueve al archivo del punto.

### Contratos entre archivos

| Dato | Formato |
|---|---|
| Características | tensor `float32` de forma `(n, D)`, en el orden del dataset |
| Etiquetas y predicciones | tensor `int64` de forma `(n,)`: `ham = 0`, `spam = 1` |
| Clase positiva del F1-score y anomalía del punto 2 | `spam = 1` |
| Particiones y muestras | índices de `utils.stratified_split` y `utils.sample_indices` |
| Una corrida | función `run(seed) -> dict` de métricas para `utils.repeat_experiment` |
| Métricas | `utils.classification_metrics`: `tasa_aciertos` y `f1` |

Las firmas de las funciones pendientes ya están definidas y documentadas.
No cambiarlas sin avisar al equipo; si un contrato cambia, actualizar en el
mismo pull request el archivo, sus pruebas y las celdas del notebook que lo
usan.

### Protocolo experimental (del enunciado)

- La regresión logística (`forward` y `train_logistic_regression`) se
  implementa en PyTorch, sin clasificadores ya hechos.
- Puntos 1.4, 1.5 y 2.2: 30 corridas, cada una con una partición
  estratificada 70 % / 30 %. La corrida `k` usa la semilla `SEED + k - 1`
  que entrega el notebook; no usar semillas propias dentro de los archivos.
- Punto 1.5: reducir a 50 %, 25 % y 2 dimensiones. El PCA de cada corrida se
  ajusta solo con el entrenamiento y se aplica a la prueba.
- Punto 2.1: usar la $D$ óptima deducida en 1.5 (`OPTIMAL_PCA_DIMENSIONS`).
- Punto 3.1: textos sin preprocesar; 30 muestras del 30 % del dataset.
- Métricas en todos los puntos: tasa de aciertos y F1-score.

### Documentación, pruebas y datos

- Toda función lleva documentación Doxygen (`"""!` con `@brief`, `@param` y
  `@return`).
- El enunciado exige al menos dos pruebas para `forward` y dos para
  `train_logistic_regression`. No borrar pruebas pendientes: implementarlas.
- Una prueba o función sin terminar lanza `NotImplementedError`; `pytest` y
  el notebook la reportan como pendiente, no como fallida.
- No versionar `data/raw/`, `data/processed/`, modelos, adaptadores ni
  checkpoints. Nunca incluir credenciales, tokens (Hugging Face, APIs) ni
  archivos `.env`.
- No editar a mano `results/figures/` ni `results/tables/`: el notebook borra
  y regenera su contenido en cada ejecución.
- Todo número, tabla o figura del informe debe provenir de una ejecución
  completa del notebook. No inventar ni estimar resultados.

## Antes de empezar

1. Leer el enunciado, `README.md`, este archivo y `CONTRIBUTING.md`.
2. Instalar el entorno (`README.md`, "Preparación del entorno").
3. Ejecutar `py -m pytest` (Windows) o `python3 -m pytest`. En el estado
   inicial del esqueleto el resultado es **14 superadas y 8 pendientes**.
4. Crear una rama corta para el punto, por ejemplo `feat/punto1-pca`.
5. Buscar lo pendiente del punto con `git grep -n "TODO(punto 1"` (cambiar
   el número según el punto).

## Al terminar cada función

1. Reemplazar el `NotImplementedError` por la implementación.
2. Implementar o completar su prueba en `tests/test_puntoN.py`.
3. Ejecutar `pytest`: no debe haber pruebas fallidas.
4. Ejecutar la celda de configuración y la sección del punto en el notebook.
5. Escribir el análisis en el notebook (celdas **Análisis**) y en la sección
   del informe.
6. Marcar la casilla en este archivo y abrir el pull request.

## Decisiones pendientes del equipo

Registrar cada decisión en la celda de configuración del notebook, en la
sección "Objetivo y protocolo" del notebook y en el informe.

- [ ] Asignar responsables (tabla de arriba y `CONTRIBUTING.md`).
- [ ] Confirmar los integrantes; se copiaron del TP1 (`README.md`,
      notebook y `docs/report/main.tex`).
- [ ] `ENCODER_NAME` (1.4): codificador de los mensajes; registrar la
      dependencia y su versión en `requirements.txt`.
- [ ] `LR_LEARNING_RATE` y `LR_EPOCHS` (1.2): hiperparámetros de la
      regresión logística.
- [ ] `OPTIMAL_PCA_DIMENSIONS` (fin de 1.5, se usa en 2.1).
- [ ] Datos con los que se construyen las densidades (2.1): solo ham del
      entrenamiento (lo que hace hoy el notebook) o todo el entrenamiento.
- [ ] Criterio del umbral (2.1) y `HISTOGRAM_BINS`.
- [ ] `LLM_MODELS` (3.1, al menos uno; 3.2 extra, al menos dos más), el
      prompt y qué hacer con respuestas ambiguas. Considerar el costo: son
      5 574 mensajes por modelo; preferir modelos que corran en las
      computadoras del equipo.
- [ ] `LORA_MODEL` y LoRA o DoRA (3.3, extra).

## Punto 1 - `src/tp2/punto1.py` (60 puntos)

Responsable: por asignar. Ya implementadas: `predict_proba`, `predict` y
`components_for_fraction`.

- [ ] **1.1 (10)** `forward`: $p(y=1\mid x)=\sigma(x^\top w+b)$, vectorizado.
  - [ ] 1.1a: `test_forward_first_case` y `test_forward_second_case`.
- [ ] **1.2 (10)** `train_logistic_regression`: entropía cruzada binaria,
      inicialización con `seed`, historial de pérdida por época.
  - [ ] 1.2a: `test_train_first_case` y `test_train_second_case`.
- [ ] **1.3 (10)** `make_linearly_separable`, `plot_dataset`,
      `plot_decision_boundary` y `plot_loss_history`. Gráficos y tablas que
      ilustren el proceso; análisis.
  - [ ] Recomendada: `test_make_linearly_separable_shapes_and_reproducibility`.
- [ ] **1.4 (10)** `load_encoder` y `extract_features`; fijar
      `ENCODER_NAME`. 30 corridas 70/30; reportar y comentar tasa de aciertos
      y F1-score.
- [ ] **1.5 (20)** `fit_pca`, `transform_pca`, `plot_explained_variance` y
      `plot_pca_projection`. Reducir a 50 %, 25 % y 2 dimensiones, graficar
      todo el dataset, analizar la clasificación y deducir la $D$ óptima
      (`OPTIMAL_PCA_DIMENSIONS`).
  - [ ] Recomendada: `test_fit_pca_recovers_dominant_direction`.
- [ ] Informe: `01-punto1-regresion-logistica.tex` y `04-unit-tests.tex`.

## Punto 2 - `src/tp2/punto2.py` (30 puntos)

Responsable: por asignar. Ya implementada: `predict_anomalies` (puntaje
menor que el umbral = spam). Se puede programar y probar desde ya con
tensores sintéticos; para ejecutarlo en el notebook necesita las
características (1.4), `fit_pca` y `transform_pca` (1.5) y
`OPTIMAL_PCA_DIMENSIONS`. Mientras tanto se puede fijar una $D$ provisional.

- [ ] **2.1 (20)** `fit_histogram_densities`, `log_likelihood`,
      `select_threshold`, `plot_densities` y `plot_score_distribution`.
      Graficar y comentar las densidades; definir el umbral sobre la suma de
      verosimilitudes de todas las densidades.
  - [ ] Recomendada: `test_histogram_densities_integrate_to_one`.
- [ ] **2.2 (10)** 30 corridas 70/30 (la celda ya está en el notebook);
      reportar y comentar frente a los resultados del punto 1.
- [ ] Informe: `02-punto2-deteccion-anomalias.tex`.

## Punto 3 - `src/tp2/punto3.py` (10 puntos y 20 extra)

Responsable: por asignar. Es independiente de los otros puntos: solo usa los
textos sin preprocesar y `utils.py`. El notebook clasifica cada mensaje una
sola vez por modelo, guarda las predicciones en `data/processed/` y las 30
muestras del 30 % las reutilizan. Con un modelo fijo y decodificación
determinista, el resultado es el mismo que clasificar cada muestra, con
30 veces menos inferencias.

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
- [ ] Actualizar el "Estado" del encabezado del notebook y la lista final de
      pendientes.
- [ ] Generar Doxygen (`README.md`, "Documentación interna") y copiar
      `build/doxygen/html` a `submission/doxygen/html`.
- [ ] Ejecución final: **Kernel → Restart Kernel and Run All Cells…** y un
      commit deliberado con las salidas, figuras y tablas finales.

## Verificación pendiente del esqueleto

Lo siguiente no se pudo comprobar al crear el esqueleto y debe revisarse en
la primera ejecución:

- [ ] Descarga y lectura del dataset real. La URL de UCI responde y la
      lectura se probó con un archivo de ejemplo; según UCI, la tabla
      "Distribución de clases del dataset" debería sumar 5 574 mensajes.
- [ ] Compilación del informe con `./build_pdf.sh` (no había LaTeX instalado).
- [ ] Generación de Doxygen (no estaba instalado).

## Checklist de entrega

- [ ] El notebook corre de principio a fin sin errores desde un kernel limpio.
- [ ] `pytest` no tiene pruebas fallidas y las cuatro pruebas obligatorias de
      la regresión logística están implementadas.
- [ ] PDF generado con `./build_pdf.sh` en `submission/tp2_detector_spam.pdf`.
- [ ] Doxygen generado y copiado a `submission/doxygen/html`.
- [ ] El `.zip` incluye: `notebooks/tp2_detector_spam.ipynb` ejecutado,
      `src/tp2/` y `tests/` (el notebook depende de ellos),
      `requirements.txt`, `docs/report/`, `results/figures/` y el PDF.
- [ ] Subir el `.zip` a TEC Digital antes del martes 20 de octubre.

## Para agentes de IA

- Trabajar solo en los archivos del punto asignado; no tocar los de otro
  punto ni `utils.py` sin que una persona del equipo lo pida.
- No cambiar firmas, contratos, semillas ni el protocolo por iniciativa
  propia. Las decisiones pendientes las toma el equipo: proponer opciones y
  esperar confirmación antes de fijarlas.
- No descargar modelos grandes ni ejecutar inferencias largas sin
  confirmación del equipo.
- Antes de terminar, ejecutar `pytest` y reportar el resultado real,
  incluidas las pruebas fallidas o pendientes.
- No hacer commit de datos, cachés, modelos ni credenciales, y no hacer push
  ni integrar a `main` sin aprobación.
