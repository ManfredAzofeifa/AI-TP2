"""!
@file test_punto3.py
@brief Pruebas del punto 3: clasificación con modelos de lenguaje.

Las pruebas no deben descargar ni ejecutar modelos grandes: se prueban las
partes deterministas, como la interpretación de las respuestas, con un
tokenizador y un modelo simulados.
"""

import pytest
import torch
import sys
from types import SimpleNamespace

from notebook_loader import notebook as punto3


def test_load_llm_pins_tokenizer_and_weights_without_config_commit(monkeypatch, tmp_path):
    """!
    @brief Registra la revisión real aunque Transformers no conserve config._commit_hash.
    @param monkeypatch Fixture que sustituye las bibliotecas por objetos sin red ni pesos.
    @param tmp_path Caché simulada de Hugging Face.
    @return None.
    """
    revision = "a" * 40
    config = tmp_path / revision / "config.json"
    config.parent.mkdir()
    config.write_text("{}")
    calls = {}

    def tokenizer_from_pretrained(name, **kwargs):
        """!
        @brief Simula un tokenizador sin token de relleno.
        @param name Identificador solicitado.
        @param kwargs Opciones de carga que se verificarán.
        @return Tokenizador simulado.
        """
        calls["tokenizer"] = (name, kwargs)
        return SimpleNamespace(pad_token=None, eos_token="EOS")

    def model_from_pretrained(name, **kwargs):
        """!
        @brief Simula pesos con config sin atributo de revisión.
        @param name Identificador solicitado.
        @param kwargs Opciones de carga que se verificarán.
        @return Modelo simulado.
        """
        calls["model"] = (name, kwargs)
        return SimpleNamespace(config=SimpleNamespace(), to=lambda device: None, eval=lambda: None)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(hf_hub_download=lambda *a, **k: str(config)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_from_pretrained),
    ))
    llm = punto3.load_llm("org/model", device="cpu")
    assert llm.revision == revision
    assert calls["tokenizer"][1]["revision"] == calls["model"][1]["revision"] == revision
    assert llm.tokenizer.pad_token == "EOS"


class _FakeTokenizer:
    """!
    @brief Tokenizador simulado: un token por carácter, relleno a la izquierda con 0.
    """

    chat_template = "simulada"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        """!
        @brief Envuelve el turno del usuario como lo haría una plantilla de chat.
        @param messages Lista con un único mensaje del usuario.
        @param tokenize Debe ser falso: se espera texto.
        @param add_generation_prompt Debe ser verdadero: se abre el turno del asistente.
        @return Prompt con marcadores de usuario y asistente.
        """
        assert not tokenize and add_generation_prompt
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, prompts, return_tensors, padding, add_special_tokens):
        """!
        @brief Codifica un lote con relleno a la izquierda.
        @param prompts Lista de prompts del lote.
        @param return_tensors Formato pedido (`"pt"`).
        @param padding Debe ser verdadero.
        @param add_special_tokens Debe ser falso con plantilla de chat.
        @return Diccionario con `input_ids` y `attention_mask`.
        """
        assert return_tensors == "pt" and padding and not add_special_tokens
        width = max(len(prompt) for prompt in prompts)
        ids = torch.tensor([[0] * (width - len(p)) + [ord(c) for c in p] for p in prompts])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    def batch_decode(self, tokens, skip_special_tokens):
        """!
        @brief Decodifica cada fila ignorando el relleno.
        @param tokens Tensor de forma (lote, tokens).
        @param skip_special_tokens Se ignora.
        @return Lista de textos.
        """
        return ["".join(chr(t) for t in row.tolist() if t) for row in tokens]


class _FakeModel:
    """!
    @brief Modelo simulado: responde " Spam." si el prompt contiene "WIN" y "ham" si no.
    """

    def __init__(self):
        """!
        @brief Inicializa el registro de tamaños de lote.
        @return None.
        """
        self.batch_sizes = []

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id):
        """!
        @brief Agrega la respuesta simulada después del prompt, como `generate` de Hugging Face.
        @param input_ids Prompts codificados.
        @param attention_mask Máscara de relleno.
        @param max_new_tokens Máximo de tokens nuevos.
        @param do_sample Debe ser falso (decodificación voraz).
        @param pad_token_id Token de relleno.
        @return Tensor con el prompt seguido de los tokens nuevos.
        """
        assert do_sample is False
        self.batch_sizes.append(input_ids.shape[0])
        new_rows = []
        for row in input_ids:
            prompt = "".join(chr(t) for t in row.tolist() if t)
            answer = " Spam." if "WIN" in prompt else "ham"
            codes = [ord(c) for c in answer][:max_new_tokens]
            new_rows.append(codes + [pad_token_id] * (max_new_tokens - len(codes)))
        return torch.cat([input_ids, torch.tensor(new_rows)], dim=1)


def _fake_llm():
    """!
    @brief Construye un `LanguageModel` con el tokenizador y el modelo simulados.
    @return Modelo listo para `generate_responses`.
    """
    return punto3.LanguageModel("simulado", _FakeTokenizer(), _FakeModel(), "cpu")


def test_parse_label_handles_responses():
    """!
    @brief (Recomendada, 3.1) Respuestas como "spam", " Ham." o "SPAM\n" deben traducirse
           a 1 o 0, y las ambiguas deben seguir la regla documentada en `parse_label`.
    @return None.
    """
    assert punto3.parse_label("spam") == 1
    assert punto3.parse_label("SPAM\n") == 1
    assert punto3.parse_label(" Spam.") == 1
    assert punto3.parse_label("Answer: spam") == 1
    assert punto3.parse_label(" Ham.") == 0
    assert punto3.parse_label("ham") == 0
    assert punto3.parse_label("This is ham, not spam") == 0
    # Regla documentada: sin "spam" ni "ham", la respuesta cuenta como ham.
    assert punto3.parse_label("") == 0
    assert punto3.parse_label("I cannot tell") == 0
    assert punto3.parse_label("hamburger") == 0


def test_parse_label_negations_and_ambiguity():
    """!
    @brief Las negaciones de spam cuentan como ham y solo se marcan ambiguas las respuestas
           sin ninguna de las dos palabras.
    @return None.
    """
    for response in ("Not spam", "it is not a spam message", "non-spam", "isn't spam"):
        assert punto3.parse_label(response) == 0
        assert not punto3.is_ambiguous_response(response)
    assert punto3.is_ambiguous_response("I cannot tell")
    assert punto3.is_ambiguous_response("spammy")
    assert not punto3.is_ambiguous_response("HAM")
    labels = punto3.labels_from_responses(["spam", "ham", "???"])
    assert labels.dtype == torch.int64 and labels.tolist() == [1, 0, 0]


def test_select_llm_device_requires_gpu(monkeypatch):
    """!
    @brief Sin GPU CUDA, la selección del dispositivo se detiene salvo que se permita la CPU.
    @param monkeypatch Fixture de pytest para simular la ausencia de GPU.
    @return None.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        punto3.select_llm_device(require_gpu=True)
    assert punto3.select_llm_device(require_gpu=False) == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert punto3.select_llm_device(require_gpu=True) == "cuda"


def test_generate_responses_keeps_order_and_decodes_only_new_tokens():
    """!
    @brief Las respuestas vuelven en el orden original aunque los lotes se agrupen por longitud.
    @return None.
    """
    texts = ["WIN a prize now!!!", "ok", "see you at lunch tomorrow", "WIN", "hi there"]
    llm = _fake_llm()
    responses = punto3.generate_responses(
        texts, llm, "Message: {message}", batch_size=2, max_new_tokens=8
    )
    assert responses == [" Spam.", "ham", "ham", " Spam.", "ham"]
    assert llm.model.batch_sizes == [2, 2, 1]
    prompts = punto3.build_prompts(["{x}"], llm.tokenizer, "Message: {message}")
    assert prompts == ["<user>Message: {x}<assistant>"]


def test_classify_messages_returns_label_tensor():
    """!
    @brief `classify_messages` cumple el contrato: tensor `int64` (n,) en el orden de los textos.
    @return None.
    """
    labels = punto3.classify_messages(["hello", "WIN cash", "call me"], _fake_llm(), batch_size=2)
    assert labels.dtype == torch.int64
    assert labels.tolist() == [0, 1, 0]


def test_llm_cache_roundtrip_and_validation(tmp_path):
    """!
    @brief La caché de respuestas se recupera solo con los mismos textos y configuración.
    @param tmp_path Carpeta temporal de pytest.
    @return None.
    """
    texts = ["hola", "WIN ahora"]
    metadata = punto3.llm_cache_metadata(texts, "org/modelo", "P: {message}", 8)
    path = punto3.llm_responses_path(tmp_path, metadata)
    assert path.name.startswith("llm-org_modelo-") and path.suffix == ".json"
    punto3.save_llm_responses(path, metadata, ["ham", "spam"], details={"device": "prueba"})
    payload = punto3.load_llm_responses(path, metadata)
    assert payload["responses"] == ["ham", "spam"] and payload["details"]["device"] == "prueba"

    other = punto3.llm_cache_metadata(texts, "org/modelo", "Otro: {message}", 8)
    assert punto3.llm_responses_path(tmp_path, other) != path
    with pytest.raises(ValueError):
        punto3.load_llm_responses(path, other)
    with pytest.raises(ValueError):
        punto3.save_llm_responses(path, metadata, ["ham"])


def test_llm_sample_metrics_uses_stratified_sample():
    """!
    @brief Cada corrida evalúa la muestra estratificada de su semilla e incluye la confusión.
    @return None.
    """
    targets = torch.tensor([0] * 70 + [1] * 30)
    predictions = targets.clone()
    predictions[:5] = 1
    result = punto3.llm_sample_metrics(7, targets, predictions, sample_fraction=0.3)
    sampled = punto3.sample_indices(targets, 0.3, 7)
    expected = punto3.classification_metrics(targets[sampled], predictions[sampled])
    assert result["tasa_aciertos"] == expected["tasa_aciertos"]
    assert result["f1"] == expected["f1"]
    assert result["tp"] + result["fp"] + result["fn"] + result["tn"] == len(sampled)
    assert result["tp"] == int(targets[sampled].sum())


def test_summarize_responses_counts_normalized_answers():
    """!
    @brief Las respuestas se agrupan sin distinguir mayúsculas ni espacios.
    @return None.
    """
    summary = punto3.summarize_responses(["Spam", " spam ", "ham", "???"])
    rows = {row.respuesta: row for row in summary.itertuples()}
    assert rows["spam"].mensajes == 2 and rows["spam"].etiqueta == "spam"
    assert rows["???"].ambigua and rows["???"].etiqueta == "ham"
