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
- Entrega realizada el 26/10/2026 (el enunciado fijaba el 20/10/2026):
  `.zip` con PDF, fuentes LaTeX (`informe/`), notebook y `tests/`, armado con
  `./build_submission.sh`. El proyecto está cerrado; ver el estado actual.
- Enunciado: `docs/assignment/TrabajoPractico2_detector_ataques_supervisado_no_supervisado.pdf`
  (se puede leer con `pdftotext -layout`).

| Ruta | Contenido |
|---|---|
| `notebooks/tp2_detector_spam.ipynb` | Única implementación: utilidades, puntos 1 a 3, experimentos y análisis |
| `tests/test_puntoN.py`, `tests/test_utils.py` | Pruebas que cargan las definiciones del notebook con `tests/notebook_loader.py` |
| `tests/test_notebook_structure.py` | Validez del notebook autocontenido y dependencias opcionales locales |
| `tests/test_partial_runner.py` | Ejecutor auxiliar `scripts/run_puntos_1_2.py`; no se entrega |
| `build_submission.sh`, `submission/` | Arma la entrega: PDF versionado y copias generadas (`informe/`, `notebooks/`, `tests/`) sin versionar |
| `docs/report/sections/` | Informe LaTeX, una sección por punto |
| `docs/report/review.tex`, `docs/report/review.pdf` | Revisión interna (no se entrega): lógica, pasos, resultados y registro de decisiones; el PDF se compila con `./build_pdf.sh review` y se versiona |

## Estado actual

- **Ejecución completa con salidas (27/10/2026):** kernel WSL nuevo, las 46
  celdas de código en orden (contadores 1–46), sin errores. Reutilizó las
  cachés de MPNet, de los tres LLM y de 3.3: no cargó modelos ni entrenó.
  Duró 37.5 minutos; las corridas de 1.4 y 1.5 ocuparon 19.4 y 16.5, con un
  hilo de CPU (`torch.set_num_threads(1)`); sus resúmenes coinciden con el informe
  (F1 0.9641, ΔF1 −0.0004/−0.0045/−0.1796, detector −0.2597/−0.0802), igual
  que 3.1–3.3 y la comparación global, que se ejecutó por primera vez.
  Celda de pruebas: 74/0/0. Segunda entrega, con el notebook ejecutado:
  `tp2_detector_spam_entrega_con_salidas.zip` (SHA-256 `702ea61f…aa0e2c5e`),
  armada en `build/entrega-con-salidas/`; respaldo en
  `build/notebook-backups/tp2_detector_spam.ejecutado-20261027.ipynb`.
  El notebook del repositorio sigue sin salidas. En la figura de comparación
  se superponen los nombres largos de los LLM en el eje x (cosmético).
- **Entrega y cierre (26/10/2026):** `.zip` entregado (SHA-256
  `11a520305d942c453f860b3387dcfe8de646ddd8e73a816c9c41a2633997a936`), con
  PDF de 35 páginas, fuentes LaTeX y sus 25 figuras, notebook sin salidas y
  74 pruebas. El informe usa redacción impersonal ("se eligió", "se usó"),
  sin mencionar Git, ramas, rutas del repositorio, WSL como transferencia ni
  decisiones "del equipo" o "de la persona"; incluye comparación global y
  conclusiones. Compilar `informe/` y ejecutar `tests/` desde una copia
  aislada: sin referencias indefinidas y 74/0/0. El proyecto da 77/0/0.
  El notebook toma como raíz la carpeta con `tests/notebook_loader.py` o, sin
  ella, el directorio actual. No se hicieron: ejecución completa con salidas,
  Doxygen ni limpieza de la redacción interna del notebook.
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
- **3.1 ejecutado (26/10/2026):** elección de la persona:
  `LLM_MODELS = ["Qwen/Qwen2.5-3B-Instruct"]`, prompt sin ejemplos en inglés,
  decodificación voraz, respuestas ambiguas como ham. La persona autorizó
  el 26/10/2026 descargar y evaluar este modelo en la GPU WSL conectada a
  PyCharm; no cargarlo en el Mac. Respuestas en `data/processed/llm/`.
  RTX 3080 Ti, bfloat16, lote 16: 5574 mensajes en 84.2 s, sin incluir carga
  ni descarga. Revisión `aa8e72537993ba99e69dfaafa59ed015b17504d1` fijada en
  tokenizador y pesos. En 30 muestras: accuracy 0.7853 ± 0.0073 y F1
  0.5452 ± 0.0090. Sin respuestas ambiguas; 1168 falsos positivos y 30 falsos
  negativos en todo el dataset. Figuras, CSV y análisis de 3.1 completados.
  La nueva ejecución reproduce exactamente la caché anterior y libera el modelo.
- **Pendiente:** nada obligatorio. 3.1, 3.2 y 3.3 están en `origin/main`
  (`a9654d6`); el cierre de entrega requiere nueva autorización para commit
  y push. Las autorizaciones anteriores no cubren publicaciones posteriores.
- **Pruebas verificadas (26/10/2026):** 77 superadas, 0 pendientes, 0 fallidas,
  en Mac y en la celda WSL (64/0/0 al cerrar 3.2; 48/1/0 antes de 3.1). `tests/conftest.py` fija el backend `Agg`.
  Puntos 1 y 2 y pruebas se repitieron desde un kernel nuevo con caché local;
  la tabla de pruebas también da 48/1/0. Las figuras regeneradas de esa
  auditoría están en `codex/audit-puntos-1-2`; main conserva sus PNG previos.
  La ejecución final completa sigue pendiente del cierre. Informe preliminar
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
- **Continuación en main (26/10/2026):** los clones del Mac y de WSL volvieron
  a `main`, base `d520690`, conservando el notebook local modificado por la
  persona y los archivos auxiliares de PyCharm en WSL. La conexión vigente es
  Jupyter externo por Cloudflare Quick Tunnel sin cuenta ni dominio, con token
  Jupyter; no se requiere SSH/Tailscale. El equipo comparte ahora ese host,
  de a una persona o un trabajo GPU a la vez, sin kernels dedicados garantizados.
  Una consulta por MCP confirmó kernel Linux/WSL2, PyTorch `2.14.0+cu130`,
  CUDA disponible y RTX 3080 Ti. Esto verifica identidad y disponibilidad,
  no inferencia LLM ni entrenamiento. La infraestructura anterior permanece
  en `codex/wsl-gpu-host`; no se integra al volver a main. Pruebas de main:
  48 superadas, 1 pendiente (3.1), 0 fallidas.
- **Integración de 3.1 (26/10/2026):** main incorpora `db5db2e`, conservando
  Qwen2.5-3B-Instruct, prompt, reglas de respuestas y las ocho pruebas nuevas.
  La celda local de diagnóstico CUDA se retiró con respaldo. Los conflictos
  de contexto conservan ambas decisiones; el PDF interno se regenera desde
  el TeX reconciliado. Verificación tras incorporar el commit: 56/0/0.
  La suspensión accidental interrumpió la disponibilidad: conservar estado
  requiere mantener vivos Windows/WSL/Jupyter; no prometer persistencia absoluta.
  Dos pruebas nuevas verifican la revisión sin depender de `_commit_hash` y
  el ejecutor del notebook con fixtures pytest. La celda remota también da
  58/0/0. Al usar un kernel persistente, sincronizar el archivo guardado de
  WSL y recargar los módulos de pruebas; ejecutar celdas del editor no
  actualiza por sí solo el archivo que consume `notebook_loader.py`.
- **Cierre de 3.1 (26/10/2026):** evaluación repetida desde la caché, sin
  cargar el modelo ni repetir inferencia; métricas idénticas y 58/0/0 en
  WSL y en la suite local. Las dos figuras describen muestras del 30 %,
  se transfieren con SHA-256 y se revisan visualmente. Las 65 celdas
  guardadas de WSL coinciden en fuente con el notebook local; la
  configuración y los algoritmos de 1 y 2 siguen sin cambios. El notebook
  de desarrollo queda sin salidas; los PDF se recompilan con las figuras
  corregidas. Ese cierre se publicó después en `2534503` por autorización.
- **Respaldo autorizado (26/10/2026):** la persona solicita hacer commit y
  push del cierre de 3.1 a `origin/main` antes de trabajar en 3.2. Esta
  autorización sustituye el límite anterior para ese respaldo; no cubre
  elegir los modelos pendientes ni realizar el extra 3.3.
- **3.2 completo (26/10/2026):** 3.1 respaldado en `origin/main`, commit
  `2534503`. Después, la persona confirmó `mistralai/Mistral-7B-Instruct-v0.3`
  y `Qwen/Qwen2.5-14B-Instruct`, ambos NF4 con doble cuantización y cálculo
  bfloat16, y autorizó descargar y evaluar uno a la vez en WSL. Revisiones:
  `c170c708c41dac9275d15a8fff4eca08d52bab71` y
  `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, respectivamente.
  `load_llm` conserva su firma; `load_llm_nf4` usa una GPU completa y no
  permite descargar capas a CPU/disco. Las cachés NF4 incluyen revisión,
  lote, precisión y versiones; la caché de 3.1 conserva su identidad.
  WSL tiene `accelerate==1.15.0` y `bitsandbytes==0.50.2`, sin cambiar
  PyTorch ni Transformers. 5574 respuestas por modelo, mismas 30 muestras:
  Mistral: accuracy 0.7923 ± 0.0084, F1 0.5518 ± 0.0108; Qwen 14B:
  accuracy 0.8954 ± 0.0065, F1 0.7126 ± 0.0139. Tiempos de generación:
  168.6/536.8 s; lotes 16/4; picos asignados 4.625/9.774 GiB. Mistral
  tiene 12 respuestas ambiguas (todas ham); Qwen 14B, ninguna. Frente a
  Qwen 3B, Qwen 14B reduce falsos positivos de 1168 a 552 y falsos negativos
  de 30 a 26. Las diferencias son descriptivas; las muestras se solapan.
  Se exportaron e inspeccionaron figuras y se verificaron 22 artefactos por
  SHA-256; las 30 filas por modelo se reprodujeron en CPU con tolerancia
  1e-12. Suite local y celda WSL: 64 superadas. Modelos liberados de GPU.
  Las fuentes de las 69 celdas guardadas coinciden entre Mac y WSL; ambas
  copias quedan sin salidas de desarrollo, con respaldo ejecutado local.
  Investigación y fuentes: `docs/research/punto3-2-modelos-cuantizados.md`.
- **Acceso LAN (26/10/2026):** PyCharm llega al Jupyter existente por la IP
  LAN de Windows; reenvío a localhost y firewall limitado al Mac. Se
  verificaron HTTP 200, rechazo sin token y ejecución Linux/CUDA por MCP.
  El túnel de Jupyter quedó retirado tras comprobar la LAN.
  Configuración, DHCP y reversión: `docs/jupyter-lan.md`. El kernel antiguo
  ya no existía al retomar; su salida registra un error de ZeroMQ durante
  los widgets Xet. Se recuperaron los tres blobs de Mistral verificados por
  SHA-256 y se desactivó ese progreso en las cargas siguientes. Conservar
  las cachés; cambiar la conexión de red no exige reiniciar las sesiones.
- **3.3 completo (26/10/2026):** Qwen 14B con LoRA sobre NF4 (rango 8,
  alpha 16, tasa 2e-4, una época, lote 1, acumulación 16), PEFT 0.21.0 y
  bucle explícito; partición externa 70/30 fija con 20 % de validación del
  entrenamiento. Solo se supervisa ham/spam y EOS; base NF4/BF16 congelada.
  Época: 196 pasos en 3498.5 s, picos 10.19/10.37 GiB; validación
  0.5163 → 0.0446. Prueba de 1672 filas: F1 0.7120 → 0.9708, accuracy
  0.8935 → 0.9922, FP 174 → 5, FN 4 → 8; en 1545 filas sin texto de ajuste,
  F1 0.7024 → 0.9688. Un solo entrenamiento: comparación descriptiva.
  Run `qwen14b-dda32081338adef6` en `data/processed/lora/` de WSL; al Mac
  solo pasaron JSON, CSV y PNG pequeños (SHA-256) y las métricas se
  recalcularon en CPU. Adaptador y checkpoints no salen de WSL ni de Git.
  `RUN_LORA_TRAINING = False`; si faltan respuestas, la celda de ajuste
  reutiliza `training.json` y el adaptador. No repetir el entrenamiento.

## Próximos pasos

El trabajo está entregado. Mejoras opcionales, detalladas en
[TODO.md](TODO.md#próximos-pasos-24102026):

1. Ejecución completa con salidas: hecha el 27/10/2026. Repetirla tarda
   unos 40 minutos con las cachés de WSL y regenera `results/` en WSL.
2. Retirar del notebook la redacción de proceso interno (equipo, repositorio,
   WSL/PyCharm), con el mismo criterio que el informe.
3. Doxygen. 4. Opinión del docente sobre D=2 frente a D=384 en 2.1.

No reiniciar ni matar la sesión remota existente; las cachés de WSL son las
únicas con adaptadores y checkpoints de 3.3.

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
- Punto 3: toda celda que carga o usa un modelo de lenguaje llama a
  `select_llm_device(require_gpu=True)` o reutiliza la caché de respuestas;
  carga, prueba rápida, clasificación y liberación van en celdas separadas de
  la evaluación, para cambiar de modelo con solo `LLM_MODEL_NAME`.
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

- El informe entregado (`docs/report/main.tex` y `sections/`) describe solo lo
  que contiene la entrega: redacción impersonal en español ("se decidió",
  "se incorporó"), sin "el equipo" ni "la persona", y sin Git, ramas,
  commits, rutas internas del repositorio ni procesos de sincronización.
  Esas notas van en `review.tex`, `TODO.md` y este archivo.
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
  La autorización del 26/10/2026 cubre la descarga y evaluación del modelo
  ya confirmado para 3.1 en WSL. La autorización posterior de 3.2 y sus
  modelos figura en el estado actual. La selección y el piloto de 3.3
  también están confirmados arriba; otro modelo o técnica requieren consulta.
- HF: autenticar interactivamente en WSL con un token dedicado de lectura,
  preferiblemente de alcance limitado. No escribir tokens en celdas, argumentos,
  salidas, Git ni chat. El login de HF se guarda fuera del repositorio; quienes
  compartan la cuenta Linux pueden usarlo. Verificar solo presencia/autenticación,
  nunca imprimir credenciales. Un token mejora límites de acceso/descarga,
  no la velocidad de inferencia local.
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
./build_submission.sh
```

`./build_pdf.sh review` compila `docs/report/review.tex` en
`docs/report/review.pdf`; `./build_pdf.sh` compila el informe en
`submission/tp2_detector_spam.pdf`; `./build_submission.sh` además copia
fuentes LaTeX, figuras usadas, notebook y pruebas a `submission/`. Ambos necesitan LaTeX con `latexmk`: en
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
