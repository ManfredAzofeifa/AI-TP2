# AGENTS.md

Instrucciones para agentes de IA que trabajen en este repositorio. Las
personas del equipo usan [TODO.md](TODO.md).

## Regla principal: actualizar siempre TODO.md y AGENTS.md

Todo cambio que complete, agregue o modifique una tarea, una decisión, una
regla, la estructura o el estado del proyecto se refleja, en el mismo cambio,
en **ambos** archivos: `TODO.md` (casillas, estado actual, decisiones,
verificación pendiente) y `AGENTS.md` (estado actual, reglas, flujo). Nunca
actualizar solo uno de los dos.

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

## Estado actual

- **Listo:** `utils.py`, 1.1 (`forward`) y 1.2 (`train_logistic_regression`)
  con sus dos pruebas unitarias cada uno.
- **Pendiente:** 1.3 a 1.5, punto 2 y punto 3 (detalle en `TODO.md`).
- **Pruebas:** `pytest` da 18 superadas, 4 pendientes y 0 fallidas.

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
- 1.4, 1.5 y 2.2: 30 corridas con particiones estratificadas 70 % / 30 %; la
  corrida `k` usa la semilla `SEED + k - 1` que entrega el notebook.
- 1.5: 50 %, 25 % y 2 dimensiones; el PCA se ajusta solo con el
  entrenamiento de cada corrida.
- 2.1: usar `OPTIMAL_PCA_DIMENSIONS` deducida en 1.5.
- 3.1: textos sin preprocesar; 30 muestras del 30 % del dataset.
- Métricas: tasa de aciertos y F1-score.

### Documentación, pruebas y datos

- Toda función nueva lleva documentación Doxygen (`"""!` con `@brief`,
  `@param` y `@return`), en español, con el estilo de los archivos existentes.
- Implementar las pruebas pendientes en lugar de borrarlas. El enunciado exige
  al menos dos pruebas para `forward` y dos para `train_logistic_regression`.
- Las pruebas no descargan modelos ni datos: usan tensores pequeños y
  reproducibles.
- No versionar `data/raw/`, `data/processed/`, modelos, adaptadores,
  checkpoints, credenciales, tokens ni archivos `.env`.
- No editar a mano `results/figures/` ni `results/tables/`.

## Flujo por apartado

1. Implementar la función en el archivo del punto, sin cambiar su firma.
2. Implementar o completar sus pruebas en `tests/test_puntoN.py`.
3. Ejecutar `pytest`: 0 fallidas.
4. Si el apartado se ejecuta en el notebook, ejecutar la celda de
   configuración y la sección del punto, y no dejar salidas en el commit
   durante el desarrollo.
5. Documentar la implementación en la celda de markdown del notebook y en la
   sección del informe (`docs/report/sections/`).
6. Actualizar `TODO.md` y `AGENTS.md`: casillas, estado actual y resultado de
   `pytest`.
7. Reportar a la persona qué cambió, el resultado real de `pytest` y lo que
   no se pudo verificar.

## Límites

- No tomar las decisiones pendientes del equipo (codificador, $D$ óptima,
  umbral, modelos de lenguaje, LoRA o DoRA; lista en `TODO.md`): proponer
  opciones y esperar confirmación.
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
```

En Windows usar `py` en lugar de `python3`. Mientras haya apartados
pendientes, la ejecución completa del notebook se detiene en el primer
`NotImplementedError`.
