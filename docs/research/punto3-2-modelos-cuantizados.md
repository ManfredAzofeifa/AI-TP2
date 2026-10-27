# Opciones para 3.2: modelos cuantizados de más de 4B

Consulta de fuentes primarias: 26/10/2026. La persona confirmó la pareja
Qwen2.5-14B-Instruct + Mistral-7B-Instruct-v0.3, ambos NF4, y autorizó su
descarga y evaluación secuencial después del respaldo de 3.1 en main
(`2534503`). **La consulta documental no produjo resultados experimentales**:
durante esta investigación no se descargaron pesos, no se instalaron
paquetes y no se ejecutó inferencia. La calificación GPU se registra aparte.

## Selección confirmada y alternativa

La pareja confirmada es **`Qwen/Qwen2.5-14B-Instruct` y
`mistralai/Mistral-7B-Instruct-v0.3`, ambos cargados con NF4 de 4 bits**.
Combina dos familias y dos tamaños superiores a 4B; ambos son modelos
Instruct con arquitectura causal que permite conservar el cargador de
Transformers. El 14B requiere comprobar la memoria real antes de evaluar
todo el dataset. La alternativa **Qwen2.5-7B-Instruct + Qwen2.5-14B-Instruct,
ambos NF4**, facilitaría estudiar escala dentro de la misma familia y se
conserva como alternativa sin adoptar. La selección no permite anticipar
que algún modelo vaya a mejorar el F1 del TP2.

| Candidato | Tamaño publicado o metadato oficial | Acceso consultado y licencia | Uso en la comparación |
|---|---:|---|---|
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 14.7B | Sin aprobación manual; Apache 2.0 | Mayor tamaño y continuidad con la línea base Qwen3B |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.25B en el metadato Safetensors | Sin aprobación manual; Apache 2.0 | Otra familia; menor presión de memoria |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.61B | Sin aprobación manual; Apache 2.0 | Alternativa para comparar 3B/7B/14B dentro de Qwen2.5 |

Las fichas de Qwen documentan `AutoModelForCausalLM`, `AutoTokenizer` y
`apply_chat_template`; Mistral también proporciona ejemplos con Transformers.
Usar la plantilla de cada tokenizador, conservando el mismo texto del prompt
de clasificación. El estado de acceso se verificó sin autenticación en las
API oficiales de [Qwen7B](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct),
[Qwen14B](https://huggingface.co/api/models/Qwen/Qwen2.5-14B-Instruct) y
[Mistral7B](https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.3):
`gated=false`. No exige aceptar condiciones manualmente como paso previo.

## Ruta de cuantización compatible con el proyecto

**NF4 mediante bitsandbytes** permite cargar pesos del repositorio original
y cuantizar las capas lineales durante la carga; no necesita calibrar la
cuantización con el dataset SMS. La integración oficial usa
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` y admite
`bnb_4bit_use_double_quant=True` y cómputo `bfloat16`. Cuantizar para
inferencia no entrena adaptadores y no ejecuta el extra 3.3. La descarga del
checkpoint original sigue siendo de tamaño BF16; cuantizar al cargar no
reduce por sí solo los bytes descargados. [Integración de bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

En la observación inicial, el entorno del trabajo principal era Linux/WSL, Python 3.14.4,
PyTorch 2.14.0+cu130 y Transformers 5.20.2; carecía de `accelerate` y
`bitsandbytes`. La guía de instalación exige Python >=3.10 y PyTorch >=2.4;
sus binarios Linux x86-64 incluyen CUDA 13.0–13.2 y `sm86`.
Es compatibilidad declarada, todavía no una validación ejecutada en este
host. [Instalación oficial](https://huggingface.co/docs/bitsandbytes/installation).

PyPI publica `bitsandbytes==0.50.2`, con rueda
`bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl` del 27/08/2026,
Python >=3.10 y dependencia `torch>=2.4,<3`. La etiqueta `py3-none` no
restringe la rueda a una ABI de CPython anterior a 3.14; no demuestra que
todas las rutas de ejecución funcionen en 3.14. El extra de pruebas de
ese paquete declara `transformers<5`; ese límite no pertenece a las
dependencias de ejecución, pero refuerza la necesidad de probar el entorno
real sin actualizar o degradar Transformers a ciegas.
[Versión y archivos oficiales de bitsandbytes](https://pypi.org/project/bitsandbytes/0.50.2/),
[metadatos de la versión](https://pypi.org/pypi/bitsandbytes/0.50.2/json).

También existe `accelerate==1.15.0`, con Python >=3.10 y rueda
`accelerate-1.15.0-py3-none-any.whl` del 09/09/2026. Estas versiones son
candidatas verificadas en el índice; todavía no se instalaron ni validaron
en el host. [PyPI de Accelerate](https://pypi.org/project/accelerate/1.15.0/),
[metadatos](https://pypi.org/pypi/accelerate/1.15.0/json).

## Memoria: cálculo, referencia y medición pendiente

La cantidad de parámetros no basta para afirmar que el modelo cabe en 12
GiB. El siguiente cálculo es **solo un límite aritmético de almacenamiento**:
`n_parámetros × 4 / 8 / 2**30`; excluye escalas, capas sin cuantizar,
activaciones, caché KV, contexto CUDA y memoria de otros procesos.

| Modelo | Límite de todos sus parámetros a 4 bits | Bytes de sus pesos BF16 publicados |
|---|---:|---:|
| Qwen2.5-7B | 3.55 GiB | 15,231,233,024 |
| Qwen2.5-14B | 6.88 GiB | 29,540,067,328 |
| Mistral-7B-v0.3 | 3.38 GiB | 14,496,047,104 |

Los tamaños BF16 proceden de los índices oficiales de
[Qwen7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/model.safetensors.index.json),
[Qwen14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/model.safetensors.index.json)
y [Mistral7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/c170c708c41dac9275d15a8fff4eca08d52bab71/model.safetensors.index.json).
Qwen14B usa vocabulario de 152064, ancho de 5120 y embeddings de entrada y
salida no compartidos: conservar algunas matrices en BF16 deja una huella
mayor que el límite anterior. [Configuración oficial](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/config.json).

Como referencia externa, Qwen midió 9.79 GB con AWQ y 9.94 GB con GPTQ-Int4
para 14B y entrada de un token; con entrada de 6144 tokens, 11.22 y 11.36 GB.
Su plataforma fue A100 80 GB, CUDA 12.1 y versiones antiguas de PyTorch y
Transformers. Es evidencia de esas configuraciones, **no una medición NF4
en la RTX 3080 Ti ni una garantía para lote 16**. [Benchmark oficial de Qwen](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html).

La calificación pendiente debe empezar con lote 1, medir memoria libre y
pico asignado/reservado, probar la longitud máxima real de los SMS sin
truncarlos silenciosamente y ampliar el lote solo si hay margen. Si 14B
no cabe con el protocolo intacto, registrar el fallo y proponer una
alternativa; no trasladar capas a CPU sin acordarlo.

## Por qué no empezar con otros formatos

- **AWQ:** Qwen distribuye una [versión oficial de 14B AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ).
  Sin embargo, AutoAWQ quedó archivado el 11/05/2025 y su última combinación
  probada fue Torch 2.6.0/Transformers 4.51.3. La documentación de HF además
  advierte que instalar AutoAWQ puede degradar Transformers a 4.47.1.
  No conviene cambiar el entorno compartido que ya funciona para acomodarlo.
  [Repositorio de AutoAWQ](https://github.com/casper-hansen/AutoAWQ),
  [integración AWQ](https://huggingface.co/docs/transformers/quantization/awq).
- **GPTQ:** también existe [Qwen14B GPTQ-Int4 oficial](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4).
  La integración actual utiliza GPTQModel; AutoGPTQ ya no está soportado y
  existen limitaciones de compatibilidad con checkpoints antiguos. Implica
  validar otro backend compilado y más dependencias. [Guía GPTQ](https://huggingface.co/docs/transformers/quantization/gptq).
- **GGUF:** es adecuado para motores como llama.cpp. En Transformers
  5.20.2, la ruta comprimida documentada se limita a Qwen3.5/Qwen3.5-MoE;
  Qwen2 y Mistral siguen la ruta que descomprime al cargar. Por tanto, pasar
  un GGUF de Qwen2.5 al cargador actual no garantiza conservar su ahorro de
  memoria. Usar llama.cpp sería otra implementación que requiere acordar y
  verificar su equivalencia de prompt/tokenización. [Guía GGUF 5.20.2](https://huggingface.co/docs/transformers/v5.20.2/quantization/gguf).

## Otros candidatos considerados

- [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407):
  12B, Apache 2.0 y `MistralForCausalLM`; más presión de memoria que Mistral7B.
  Puede ser una alternativa si se quiere una segunda familia más grande,
  con su calificación de memoria pendiente.
- [Ministral 3 8B](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
  y [14B Instruct](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512):
  Apache 2.0 y acceso abierto, pero los checkpoints del propietario son FP8,
  multimodales y usan `Mistral3ForConditionalGeneration` con tokenizador
  `mistral-common`. No son reemplazos directos de `AutoModelForCausalLM` con
  NF4. Las variantes Instruct y Reasoning son distintas; no confundirlas.
- [Gemma3-12B-it](https://huggingface.co/google/gemma-3-12b-it)
  y [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct):
  sus API oficiales indican acceso con aprobación manual y licencias propias
  ([Gemma](https://huggingface.co/api/models/google/gemma-3-12b-it),
  [Llama](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct)).
  Son opciones, pero añaden un paso del titular de la cuenta que Qwen y
  Mistral7B no requieren.

## Protocolo y registro antes de ejecutar

Conservar textos originales, prompt sin ejemplos en inglés, generación
voraz, máximo de ocho tokens, regla de ambigüedad como ham y las mismas
30 muestras con semillas 2026–2055. Una plantilla de chat distinta forma
parte del modelo; debe registrarse. Los candidatos Instruct propuestos no
exigen pedir una cadena de razonamiento; la prueba rápida debe comprobar
respuestas utilizables bajo el límite de ocho tokens.

La caché y los detalles deben distinguir modelo, revisión inmutable de
pesos y tokenizador, cuantización (`nf4`, 4 bits, doble cuantización),
dtype de cómputo, biblioteca/versión y protocolo de generación. No se debe
reutilizar una respuesta BF16 como si hubiera sido generada con NF4.
Registrar también lote, memoria pico y tiempos de carga/generación por
separado. Estos son requisitos de reproducibilidad de la comparación,
no resultados obtenidos.

Comparar nuevos modelos NF4 con Qwen3B BF16 mezcla cambios de tamaño,
familia y precisión. Permite comparar configuraciones prácticas, pero no
atribuir toda diferencia a los parámetros o a cuantizar. Una ejecución
adicional de Qwen3B NF4 sería un control opcional; no sustituiría uno de los
dos modelos adicionales que exige 3.2 y necesitaría autorización propia.

## Verificación de preparación en el host

Después de la investigación y de la selección confirmada, se instalaron
Accelerate 1.15.0 y bitsandbytes 0.50.2 mediante una celda del notebook en
WSL. El plan de pip solo añadió esos dos paquetes; no cambió PyTorch ni
Transformers. Una consulta del kernel confirmó ambas versiones importadas.
Las 64 pruebas locales pasan sin descargar modelos. La memoria libre
observada antes de cargar Mistral fue 10.81 GiB; esa consulta describe el
host antes de la carga, no el consumo de un modelo.

## Capacidad observada en la RTX 3080 Ti

Los dos modelos cargaron completos en GPU, sin capas en CPU o disco:

| Modelo NF4 | Pesos y buffers | Lote seleccionado | Prompt máximo | Pico asignado en prueba rápida |
|---|---:|---:|---:|---:|
| Mistral 7B | 3.75 GiB | 16 | 307 tokens | 4.75 GiB |
| Qwen 14B | 9.05 GiB | 4 | 312 tokens | 9.77 GiB |

Son observaciones de esta ejecución, no estimaciones basadas solo en el
número de parámetros. El pico corresponde al asignador de PyTorch y no
incluye la memoria de otros procesos o del controlador. La prueba rápida
usa doce mensajes, incluido el prompt más largo, sin truncamiento. Los
lotes se eligieron por capacidad antes de la clasificación completa,
sin modificar el prompt ni consultar las métricas para elegirlos.

La primera descarga de Mistral terminó sus tres archivos, pero el kernel
registró un error de ZeroMQ durante widgets de progreso Xet. Los tres
SHA-256 se verificaron contra los metadatos oficiales y los archivos se
protegieron mediante enlaces duros antes de recuperar sus blobs. La carga
posterior reutilizó los 14 496 080 928 bytes. Las siguientes cargas desactivan
Xet y las barras de Hugging Face, usando HTTP. La migración a LAN y su
verificación están documentadas en [la guía local](../jupyter-lan.md).

La clasificación completa terminó sin errores: Mistral tardó 168.6 s y
Qwen 14B, 536.8 s, excluyendo carga y descarga. Sus picos de memoria
asignada fueron 4.625 y 9.774 GiB. En las mismas 30 muestras, F1 fue
0.5518 ± 0.0108 y 0.7126 ± 0.0139; la referencia Qwen 3B BF16 conserva
0.5452 ± 0.0090. Los modelos se liberaron. Las respuestas y los CSV se
transfirieron con SHA-256 y sus métricas se reprodujeron en CPU con las
funciones del notebook. Las 64 pruebas pasan localmente y en WSL.
