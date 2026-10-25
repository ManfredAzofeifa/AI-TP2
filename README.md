# TP2 - Detector de spam supervisado y no supervisado

Repositorio del Trabajo Práctico 2 del curso de Principios de Inteligencia
Artificial. El objetivo es detectar mensajes spam del dataset
[SMSSpamCollection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
con regresión logística en PyTorch (con y sin PCA), detección de anomalías por
densidades de histograma y modelos grandes de lenguaje.

Toda la implementación vive en `notebooks/tp2_detector_spam.ipynb`: carga de
datos, funciones compartidas, modelos y experimentos. Las funciones se definen
en celdas de código junto a la explicación del apartado correspondiente. El
notebook exporta figuras y tablas a `results/`, que el informe LaTeX de
`docs/report/` usa para responder el enunciado. Las pruebas externas de
`tests/` comprueban las definiciones del mismo notebook, sin duplicar la
implementación.

**Las tareas pendientes, las reglas del proyecto y los pasos para trabajar
están en [TODO.md](TODO.md) (personas) y [AGENTS.md](AGENTS.md) (agentes de
IA).**

## Entregables

- Notebook principal: `notebooks/tp2_detector_spam.ipynb`, con toda la
  implementación y sus explicaciones; `tests/` conserva las pruebas externas.
- Informe externo escrito en LaTeX: `docs/report/`.
- PDF final del informe, generado desde LaTeX.
- Documentación interna con formato Doxygen, generada desde una exportación
  temporal del notebook y las pruebas.

La fecha de entrega es el martes 20 de octubre de 2026; la lista completa está
en [TODO.md](TODO.md#checklist-de-entrega).

## Estructura

```text
AI-TP2/
├── AGENTS.md                   instrucciones para agentes de IA
├── TODO.md                     tareas, reglas y checklist de entrega (personas)
├── data/                       dataset original versionado y cachés excluidas de Git
├── docs/
│   ├── assignment/             enunciado original
│   ├── doxygen/                portada de Doxygen
│   └── report/                 informe LaTeX (main.tex) y revisión interna (review.tex y review.pdf)
├── notebooks/
│   └── tp2_detector_spam.ipynb implementación, experimentos y análisis
├── results/
│   ├── figures/                PNG exportados por el notebook
│   └── tables/                 CSV exportados por el notebook
├── submission/                 copias finales para la entrega
└── tests/                      pruebas y carga de definiciones del notebook
```

## Mapa de secciones

| Punto | Sección del notebook | Pruebas | Sección del informe | Estado |
|---|---|---|---|---|
| 1. Regresión logística (60) | 1.1 a 1.5 | `tests/test_punto1.py` | `01-punto1-regresion-logistica.tex` | 1.1–1.5 verificados con MPNet 768; D = 384 confirmada |
| 2. Detección de anomalías (30) | 2.1 y 2.2 | `tests/test_punto2.py` | `02-punto2-deteccion-anomalias.tex` | 2.1 y 2.2 listos; detector con D = 2 |
| 3. Modelos de lenguaje (10 + 20 extra) | 3.1 a 3.3 | `tests/test_punto3.py` | `03-punto3-modelos-lenguaje.tex` | pendiente |
| Funciones compartidas | Configuración, datos, métricas, corridas, figuras y tablas | `tests/test_utils.py` | — | implementado |

Las funciones compartidas del notebook incluyen la carga del dataset, las particiones estratificadas
70/30, las muestras del 30 %, la tasa de aciertos, el F1-score, la repetición
de 30 corridas con tablas comparativas, la exportación de figuras y tablas, y
la ejecución de pruebas dentro del notebook. En las celdas de cada punto,
las funciones pendientes tienen su firma, su contrato documentado y un
`NotImplementedError` que indica el apartado que les corresponde.

## Organización y dependencias

El notebook es la única fuente de implementación. Cada persona trabaja las
celdas de su apartado, sus pruebas externas y su sección del informe; los
cambios de orden o configuración se coordinan para evitar conflictos.

- Los apartados 1.1 y 1.2 implementan un modelo lineal con sesgo,
  `X @ weights + bias`, seguido de sigmoide, y su entrenamiento por entropía
  cruzada: la regresión logística que pide el enunciado.
- El apartado 1.3 usa únicamente datos sintéticos. Sus datos, modelo e
  hiperparámetros no se reutilizan en 1.4, 1.5 ni el punto 2.
- El 1.4 usa `sentence-transformers/all-mpnet-base-v2`, con 768 dimensiones.
  El 1.5 compara esas características con PCA a 384 (50 %), 192 (25 %) y
  2 dimensiones. En cada una de las 30 corridas, los hiperparámetros se
  eligen mediante una validación interna extraída solo del entrenamiento;
  el PCA de esa validación se ajusta únicamente con su subconjunto de ajuste.
  Para la evaluación final se reajustan PCA y regresor con el entrenamiento
  completo, conservando la prueba fuera de toda selección.

- El equipo confirmó `OPTIMAL_PCA_DIMENSIONS = 384` el 24/10/2026. Reduce
  a la mitad las 768 características y conserva 96.16 % de varianza en el
  PCA descriptivo. El F1 medio es 0.9637 frente a 0.9641 sin PCA, con una
  diferencia pareada de −0.0004; esto no constituye una prueba de equivalencia
  estadística. Es la D de la regresión logística.
- El detector de anomalías (2.1) usa 2 componentes (`ANOMALY_PCA_DIMENSIONS`),
  el mejor de los casos de 1.5 en una validación interna de los 30
  entrenamientos: con 384, el spam resulta más verosímil que el ham. Las
  densidades se construyen solo con ham y el umbral sale de ham reservado.
  En 30 corridas (2.2) obtiene F1 0.7043, frente a 0.9641 de la regresión
  sin PCA.
- El punto 3 solo usa los textos originales y las funciones compartidas: es independiente.
- El punto 1 vale 60 puntos y es la ruta crítica.

La [rama de auditoría](https://github.com/ManfredAzofeifa/AI-TP2/tree/codex/audit-puntos-1-2) conserva los CSV experimentales
adicionales y las figuras regeneradas. `main` incorpora los cambios de
fuentes, pruebas, dependencias y PDF necesarios para continuar el proyecto;
sus figuras previas permanecen versionadas.

## Preparación del entorno

Para reproducir la auditoría desde un clon nuevo, seguir
[Reproducir la auditoría en otra computadora](docs/reproducibility.md):
entorno virtual, versiones observadas, caché MPNet y ejecución parcial sin
depender de PyCharm. El ejecutor compartido es `scripts/run_puntos_1_2.py`;
`--check` valida la caché sin borrar resultados y `--audit` activa únicamente
las alternativas documentadas, conservando las constantes del notebook.

Se necesita Python 3.11 o superior; las versiones de `requirements.txt` son
las validadas en el TP1 con Python 3.14. Los comandos usan el Python
instalado en el sistema: `py` en Windows y `python3` en macOS y Linux. Abrir
una terminal en la raíz del repositorio; si al inicio de la línea aparece
`(.venv)`, ejecutar primero `deactivate`.

### macOS y Linux

```bash
python3 -m pip install -r requirements.txt
python3 -c "import torch, pandas, matplotlib, pytest; print('ok')"
```

Si `python3` es anterior a 3.11, instalar una versión reciente desde
python.org y usar ese ejecutable (por ejemplo, `python3.14`). Si `pip`
responde `externally-managed-environment`, crear un entorno virtual con
`python3 -m venv .venv`, instalar con
`.venv/bin/python -m pip install -r requirements.txt` y usar
`.venv/bin/python` en lugar de `python3`.

### Windows PowerShell

```powershell
py -m pip install -r requirements.txt
py -c "import torch, pandas, matplotlib, pytest; print('ok')"
```

`requirements.txt` fija pandas 3.0.3 porque, con el Control inteligente de
aplicaciones de Windows 11, pandas 3.0.6 queda bloqueado al importarse.

En ambos casos, el último comando debe imprimir `ok`. `requirements.txt`
incluye `sentence-transformers`, que usa el codificador del punto 1.4:
`sentence-transformers/all-mpnet-base-v2` (`ENCODER_NAME`). Su primera carga puede descargarlo
a la caché de Hugging Face. No descargar modelos grandes ni ejecutar
inferencias largas sin confirmación. Las dependencias del punto 3 (`peft`, ...) están
comentadas hasta que el equipo elija los modelos.

## Ejecución

Iniciar JupyterLab desde la raíz del repositorio con `py -m jupyter lab`
(Windows) o `python3 -m jupyter lab` (macOS y Linux), abrir
`notebooks/tp2_detector_spam.ipynb` con el kernel **Python 3 (ipykernel)** y
elegir **Kernel → Restart Kernel and Run All Cells…**.

- El dataset está versionado en `data/raw/SMSSpamCollection`; si falta, el
  notebook lo descarga de UCI. Las características
  del codificador y las predicciones de los modelos de lenguaje se guardan en
  `data/processed/` para no recalcularlas.
- El dataset original contiene texto y etiqueta, no vectores de 768
  características. La dimensión inicial viene de MPNet-base-v2. Los
  resultados anteriores con MiniLM-L6-v2 (384 dimensiones) quedan invalidados
  para el nuevo protocolo. Ya se regeneraron características, calibración y
  30 corridas por método: F1 sin PCA 0.9641; con 384 componentes 0.9637;
  con 192, 0.9596; con 2, 0.7845.
- El archivo original de UCI tiene 5574 mensajes (4827 ham y 747 spam),
  incluidos 403 duplicados adicionales. Se conserva sin editar; las
  particiones por filas pueden compartir textos idénticos entre entrenamiento
  y prueba, limitación que debe considerarse al interpretar las métricas.
- Mientras haya puntos pendientes, la ejecución se detiene en el primer
  `NotImplementedError` e indica qué apartado falta. El punto 3.3 (extra) se
  omite solo si `LORA_MODEL` no está definido.
- La sección **Pruebas unitarias** necesita las definiciones y la celda de
  configuración; no requiere ejecutar los experimentos para ver el avance.

## Pruebas unitarias

Las pruebas también se ejecutan fuera del notebook:

```bash
python3 -m pytest
```

```powershell
py -m pytest
```

El resultado esperado en cada momento está en el "Estado actual" de
[TODO.md](TODO.md#estado-actual). Las pruebas sin implementar aparecen como
omitidas con el motivo `Pendiente`. `tests/notebook_loader.py` carga las
definiciones del notebook para probarlas sin ejecutar experimentos,
descargar datos ni cargar modelos. Las comprobaciones estructurales también
verifican la organización del notebook y las dependencias opcionales.

## Documentación interna

Doxygen no consume el notebook directamente. Su fuente documentable se genera
de forma temporal y nunca se edita manualmente; `tests/` se incluye
directamente:

```bash
jupyter nbconvert --to python notebooks/tp2_detector_spam.ipynb --output tp2_detector_spam --output-dir build/doxygen-source
doxygen Doxyfile
```

La salida HTML queda en `build/doxygen/html/`. Tanto la exportación temporal
como la documentación generada se excluyen de Git.

## Informe externo

Con una distribución de LaTeX y `latexmk` instalados (en Windows, MiKTeX:
`winget install MiKTeX.MiKTeX`, con la instalación automática de paquetes
activada), ejecutar desde la raíz:

```bash
./build_pdf.sh
```

El script descarta los archivos intermedios y deja únicamente el PDF generado
en `submission/tp2_detector_spam.pdf`. (En el TP1 el script creaba
`submissions/` pero copiaba a `submission/`; aquí ambas rutas coinciden).

## Revisión interna del proyecto

`docs/report/review.tex` es un documento aparte, que no se entrega, para
entender todo lo realizado: la lógica de cada apartado, los pasos seguidos,
los resultados y el registro de decisiones. Se actualiza con cada avance y
se versiona junto con su PDF compilado, `docs/report/review.pdf`. Para
recompilarlo (sin dejar archivos intermedios en `docs/report/`):

```bash
./build_pdf.sh review
```

Las figuras que usa salen de `results/figures/`, así que antes hay que
ejecutar las secciones correspondientes del notebook.

## Estado de verificación

El estado vigente de las pruebas, las ejecuciones y los PDF se registra en
[TODO.md](TODO.md#estado-actual) y en la revisión interna. La auditoría del
24/10/2026 confirmó que el dataset versionado es idéntico al archivo oficial
de UCI. El cambio confirmado a MPNet 768 y a validación interna por corrida
se verificó con 30 corridas por método en 1.4 y 1.5, antes de ejecutar 1.3.
Las pruebas actuales dan 48 superadas, 1 pendiente y ninguna fallida.
La auditoría de 1 y 2 añadió la verificación del gradiente de `forward` en
logit cero y distinguió la variante D=2 del detector de la D=384 solicitada
por la lectura literal de 2.1. Esa interpretación debe validarse con el docente.

## Equipo

- Alejandro Quesada
- Manfred Azofeifa
- Jose Adrián Herrera
