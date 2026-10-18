"""!
@file punto3.py
@brief Punto 3 (10 puntos y 20 extra): clasificación con modelos grandes de lenguaje.

Contiene los tres apartados del punto 3:
- 3.1 y 3.2 (extra): clasificación de los mensajes sin preprocesar con uno o
  varios modelos de lenguaje.
- 3.3 (extra): ajuste fino eficiente con LoRA o DoRA.

Contrato: recibe textos sin preprocesar y devuelve etiquetas `int64` 0/1 con
spam = 1. Como la clasificación con un modelo fijo no depende de la muestra,
el notebook clasifica cada mensaje una sola vez y las 30 muestras del 30 %
reutilizan esas predicciones. Este archivo no importa ningún otro archivo del
proyecto; `transformers` y `peft` se importan dentro de las funciones.
"""

import torch

DEFAULT_PROMPT = (
    "Classify the following SMS message as 'spam' or 'ham'. "
    "Answer with a single word.\n\nMessage: {message}\nAnswer:"
)


# ---------------------------------------------------------------------------
# 3.1 y 3.2: clasificación sin ajuste
# ---------------------------------------------------------------------------


def load_llm(model_name, device="cpu"):
    """!
    @brief Carga un modelo grande de lenguaje y su tokenizador.
    @param model_name Identificador del modelo (lista `LLM_MODELS` del notebook).
    @param device Dispositivo de PyTorch donde se ejecuta el modelo.
    @return Objeto listo para `classify_messages` (p. ej. un pipeline de Hugging Face).
    """
    # TODO(punto 3.1): elegir el modelo y la forma de cargarlo; registrar la dependencia.
    raise NotImplementedError("Punto 3.1: implementar load_llm.")


def parse_label(response):
    """!
    @brief Traduce la respuesta textual del modelo a una etiqueta.
    @param response Texto generado por el modelo.
    @return 1 si la respuesta indica spam y 0 si indica ham.
    """
    # TODO(punto 3.1): definir y documentar qué hacer con respuestas ambiguas.
    raise NotImplementedError("Punto 3.1: implementar parse_label.")


def classify_messages(texts, llm, prompt_template=DEFAULT_PROMPT, batch_size=8):
    """!
    @brief Clasifica cada mensaje con el modelo de lenguaje.
    @param texts Lista de mensajes sin preprocesar.
    @param llm Modelo devuelto por `load_llm`.
    @param prompt_template Plantilla con el marcador `{message}`.
    @param batch_size Cantidad de mensajes por lote.
    @return Tensor `int64` de forma (n,) con etiquetas 0/1, en el orden de `texts`.
    """
    # TODO(punto 3.1): generar con decodificación determinista y usar parse_label.
    raise NotImplementedError("Punto 3.1: implementar classify_messages.")


# ---------------------------------------------------------------------------
# 3.3 (extra): ajuste fino eficiente con LoRA o DoRA
# ---------------------------------------------------------------------------


def finetune_with_lora(
    model_name,
    train_texts,
    train_labels,
    rank=8,
    alpha=16,
    use_dora=False,
    epochs=1,
    learning_rate=2e-4,
    seed=None,
):
    """!
    @brief Ajusta un modelo de lenguaje para clasificar spam con adaptadores de bajo rango.
    @param model_name Identificador del modelo base.
    @param train_texts Lista de mensajes de entrenamiento.
    @param train_labels Etiquetas 0/1 de entrenamiento.
    @param rank Rango r de las matrices de adaptación.
    @param alpha Factor de escala de LoRA.
    @param use_dora Si es verdadero, usa DoRA en lugar de LoRA.
    @param epochs Cantidad de épocas de ajuste.
    @param learning_rate Tasa de aprendizaje.
    @param seed Semilla de inicialización y orden de los datos.
    @return Modelo ajustado listo para `classify_finetuned`.
    """
    # TODO(punto 3.3): configurar peft.LoraConfig (use_dora=use_dora) y entrenar.
    raise NotImplementedError("Punto 3.3: implementar finetune_with_lora.")


def classify_finetuned(model, texts, batch_size=16):
    """!
    @brief Clasifica mensajes con el modelo ajustado.
    @param model Modelo devuelto por `finetune_with_lora`.
    @param texts Lista de mensajes.
    @param batch_size Cantidad de mensajes por lote.
    @return Tensor `int64` de forma (n,) con etiquetas 0/1.
    """
    # TODO(punto 3.3)
    raise NotImplementedError("Punto 3.3: implementar classify_finetuned.")
