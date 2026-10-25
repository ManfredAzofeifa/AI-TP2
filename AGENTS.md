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
  LaTeX, PDF y el notebook documentado, más `tests/` para verificación externa.
- Enunciado: `docs/assignment/TrabajoPractico2_detector_ataques_supervisado_no_supervisado.pdf`
  (se puede leer con `pdftotext -layout`).

| Ruta | Contenido |
|---|---|
| `notebooks/tp2_detector_spam.ipynb` | Única implementación: utilidades, puntos 1 a 3, experimentos y análisis |
| `tests/test_puntoN.py`, `tests/test_utils.py` | Pruebas que cargan las definiciones del notebook con `tests/notebook_loader.py` |
| `tests/test_notebook_structure.py` | Validez del notebook autocontenido y dependencias opcionales locales |
| `docs/report/sections/` | Informe LaTeX, una sección por punto |
| `docs/report/review.tex`, `docs/report/review.pdf` | Revisión interna (no se entrega): lógica, pasos, resultados y registro de decisiones; el PDF se compila con `./build_pdf.sh review` y se versiona |

## Estado actual

- **Estructura (24/10/2026):** implementación autocontenida en el único notebook;
  se retira la distribución en archivos por punto y `utils.py`.
- **1.1 y 1.2:** regresión logística propia: score lineal básico con sesgo
  `X @ weights + bias`, sigmoide y entropía cruzada. Se entrenan pesos y sesgo.
- **1.3:** demostración sintética aislada; sus datos, modelos e hiperparámetros
  no se reutilizan en 1.4, 1.5 ni 2.
- **Codificador confirmado:** `sentence-transformers/all-mpnet-base-v2`,
  características de 768 dimensiones. El dataset UCI contiene texto y etiqueta;
  la dimensión proviene del codificador, no del archivo de datos.
- **Evaluación corregida:** validación interna en cada corrida y para cada k;
  selección automática por F1, con pérdida como desempate. PCA se ajusta solo
  en ajuste interno al calibrar y en todo el entrenamiento al evaluar.
- **Resultados:** las cifras anteriores de MiniLM/384 son históricas y no
  representan MPNet/768 ni el protocolo corregido. Ejecución MPNet/768 verificada en 30 corridas por método.
  F1: sin PCA 0.9641 ± 0.0063; con 384 componentes 0.9637 ± 0.0075; con
  192, 0.9596 ± 0.0082; con 2, 0.7845 ± 0.0189.
  Tasa de aciertos sin PCA: 0.9906 ± 0.0016.
- **D óptima confirmada (24/10/2026):** `OPTIMAL_PCA_DIMENSIONS = 384`,
  la mitad de las 768 características. Conserva 96.16 % de varianza en el
  PCA descriptivo y F1 medio 0.9637 frente a 0.9641 sin PCA (diferencia
  pareada −0.0004); esta cercanía no prueba equivalencia estadística.
  Es la D de la regresión logística.
- **2.1 implementado (24/10/2026):** densidades solo con ham del
  entrenamiento, 30 intervalos y umbral en el cuantil 5 % de un 20 % de ham
  reservado (`fit_anomaly_detector`). El detector usa
  `ANOMALY_PCA_DIMENSIONS = 2`, decisión del equipo: entre los casos de 1.5,
  gana en la validación interna de los 30 entrenamientos (F1 0.7085 frente
  a 0.0401 con 384, donde el spam resulta más verosímil que el ham).
- **2.2 completo (24/10/2026):** 30 corridas con las particiones del punto 1;
  tasa de aciertos 0.9195 ± 0.0084 y F1 0.7043 ± 0.0247, menor que la
  regresión sin PCA (F1 −0.2597) y con k = 2 (−0.0802) en las 30 corridas.
- **Pendiente:** punto 3.
- **Pruebas verificadas:** 48 superadas, 1 pendiente (3.1), 0 fallidas.
  Puntos 1 y 2 y pruebas se repitieron desde un kernel nuevo con caché local;
  la tabla de pruebas también da 48/1/0. Las figuras regeneradas de esa
  auditoría están en `codex/audit-puntos-1-2`; main conserva sus PNG previos.
  La ejecución final completa sigue pendiente del punto 3. Informe preliminar
  y revisión interna compilados y revisados visualmente.
- **Preparación para el equipo (25/10/2026):** ejecutor portable en
  `scripts/run_puntos_1_2.py` y guía en `docs/reproducibility.md`. El notebook
  se versiona sin salidas de desarrollo. Main incorpora los fuentes,
  pruebas, dependencias, PDF y la tabla de pruebas; los CSV experimentales
  adicionales y PNG regenerados permanecen en `codex/audit-puntos-1-2`. `requirements-audit.txt` registra las versiones directas
  observadas. No incluir cachés, pesos de modelos ni entornos locales.
- **Auditoría de 1 y 2 (24/10/2026):** `forward` conserva ahora derivada
  1/4 con logit cero; una prueba reproduce el fallo de la implementación
  con `abs`. El entrenamiento ya usa el gradiente analítico y no depende
  de Autograd. La conformidad literal de 2.1 queda abierta: reutilizar
  D=384 frente a la variante D=2 del equipo. Documentar ambos casos y
  validar esa variante con el docente. Agregar validaciones de particiones
  solapadas no aísla globalmente sus pruebas de la selección de D.

- **Selección mínima para main (25/10/2026):** corrección de `forward`,
  notebook, pruebas, ejecutor portátil, dependencias y documentación necesaria
  para continuar el punto 3. Las sugerencias siguen sin adoptarse. Los PDF
  se recompilan con los PNG previos de main, sin reejecutar experimentos.

## Próximos pasos

Lista completa y notas de ejecución en
[TODO.md](TODO.md#próximos-pasos-24102026). Resumen para agentes:

1. Resolver con el docente el alcance de 2.1 frente al enunciado.
   La persona confirmó el 25/10/2026 conservar la configuración vigente;
   las alternativas permanecen documentadas sin adoptarse. La referencia
   D=384 ya tiene 30
   corridas (F1 0.0386). El detector calibrado mejora F1 a 0.7173, pero
   deja escapar más spam. La regresión ampliada no mejora. Mantener las
   constantes vigentes mientras no se acuerde otra configuración.
   La celda opcional de auditoría queda desactivada; activarla repite corridas
   largas. La configuración limpia también sus tablas. La autorización
   recibida cubre esta auditoría con caché local, sin descargar modelos.
   Después del punto 3, ejecutar todo el notebook para la entrega.
2. Recordar a la persona que avise al equipo: el detector usa
   `ANOMALY_PCA_DIMENSIONS = 2`; la regresión, `OPTIMAL_PCA_DIMENSIONS = 384`.
3. Punto 3: no elegir `LLM_MODELS`, prompt ni LoRA/DoRA sin confirmación del
   equipo; proponer opciones.
4. Informe: `05-comparison.tex`, `06-conclusions.tex` y resumen final del
   notebook, después del punto 3.
5. Confirmar integrantes y fecha de entrega (el enunciado dice 20/10/2026 y
   el registro es del 24/10/2026); después, Doxygen, PDF final y `.zip`.

## Antes de modificar

1. Leer este archivo (incluidos los [próximos pasos](#próximos-pasos)),
   `TODO.md`, `README.md` y el apartado del enunciado.
2. Respetar el alcance indicado por la persona; aclarar solo lo que quede ambiguo.
3. Ejecutar `py -m pytest` (Windows) o `python3 -m pytest` y comparar con el
   [estado actual](#estado-actual).
4. Ubicar lo pendiente con `git grep -n "TODO(punto N"`.

## Reglas obligatorias

### Estructura

- Implementar toda la lógica en `notebooks/tp2_detector_spam.ipynb`, organizada
  por apartados. No mantener copias de la implementación en archivos `.py`.
- Las celdas de definiciones empiezan por `# Definiciones:` y solo contienen
  imports base, constantes, clases y funciones. `tests/notebook_loader.py` las
  carga sin ejecutar configuración, descargas ni experimentos.
- Importar las dependencias opcionales (`transformers`,
  `sentence-transformers`, `peft`, `accelerate`) dentro de las funciones que
  las usan, nunca al inicio del archivo.
- No leer variables globales del notebook: todo llega como argumento.
- Las funciones de graficación devuelven la figura sin mostrarla, con un
  título descriptivo; el notebook la muestra con `show_figure`.
- `show_figure` nombra el PNG con `figure.suptitle` o con el título
  centrado del eje. Un título solo alineado a la izquierda produce
  `figura.png` y cada figura sobrescribe a la anterior: poner el título como
  `suptitle` (así lo hace `_set_titles`).
- Estilo de las gráficas (reutilizar el de las funciones del punto 1): azul `#2a78d6` para
  ham o clase 0 y naranja `#eb6834` para spam o clase 1, un marcador distinto
  por clase, ejes y cuadrícula discretos, texto en tinta neutra (nunca del
  color de la serie) y un subtítulo que explica cómo leer la figura.
- Separar celdas de definiciones, configuración, ejecución y análisis. Las
  funciones reciben sus datos y parámetros explícitamente.

### Contratos

- Características: tensor `float32` de forma `(n, D)`, en el orden del dataset.
- Etiquetas y predicciones: tensor `int64` de forma `(n,)`, `ham = 0` y
  `spam = 1`; spam es la clase positiva del F1-score y la anomalía del punto 2.
- Particiones y muestras: `stratified_split` y `sample_indices`.
- Corridas: `run(seed) -> dict` de métricas para `repeat_experiment`.
- Métricas: `classification_metrics` (`tasa_aciertos` y `f1`).
- No cambiar firmas ni contratos ya definidos sin aprobación; si se aprueba,
  actualizar en el mismo cambio la definición, sus pruebas y las celdas del
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
- Hiperparámetros: calibrarlos por separado en cada corrida con una validación
  sacada solo de su entrenamiento; aplicar automáticamente el ganador. No
  reutilizar la calibración de una partición en las otras 29.
- `logistic_regression_run` del notebook devuelve métricas y matriz de
  confusión; las tablas comparativas deben pasar por `only_metrics(...)`.
- 1.5: 50 %, 25 % y 2 dimensiones (384, 192 y 2 con D=768); PCA se
  ajusta solo con ajuste interno al calibrar y con entrenamiento al evaluar.
  Comparar con el caso sin PCA usando las mismas particiones y semillas.
- 2.1 y 2.2: el detector usa `ANOMALY_PCA_DIMENSIONS = 2`, elegida por el
  equipo entre los casos de 1.5 con validación interna; ajustar el PCA solo
  con el entrenamiento de cada corrida, las densidades solo con sus ham y el
  umbral con ham reservado de ese entrenamiento. La regresión conserva
  `OPTIMAL_PCA_DIMENSIONS = 384`.
- D=2 es una variante respecto de la lectura literal de 2.1; registrar
  esa diferencia al informar cumplimiento. La selección por F1 usa spam
  de validación, aunque las densidades se ajusten solo con ham. El cuantil
  empírico fija una fracción objetivo en calibración y no garantiza la misma
  tasa de falsos positivos en datos futuros.
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
- Las pruebas cargan las definiciones del notebook y usan tensores pequeños
  reproducibles; no descargan modelos ni datos.
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

1. Implementar la función en una celda de definiciones de su apartado.
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

- No tomar las decisiones pendientes del equipo (modelos
  de lenguaje, LoRA o DoRA; lista en `TODO.md`): proponer opciones y esperar
  confirmación.
- No descargar modelos grandes ni ejecutar inferencias largas sin
  confirmación.
- No inventar ni estimar resultados: toda métrica, tabla o figura experimental
  del informe sale de la ejecución completa de su sección. La ejecución final
  completa queda pendiente mientras el punto 3 no esté implementado.
- Las cachés identifican dataset, codificador y configuración; validar forma
  `(5574, 768)` y finitud antes de usar características. El archivo UCI se
  conserva íntegro; sus textos repetidos son una limitación de la partición por filas.
- Mantener el alcance solicitado dentro del notebook; coordinar ediciones por
  celdas cuando varias personas trabajen a la vez.
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
