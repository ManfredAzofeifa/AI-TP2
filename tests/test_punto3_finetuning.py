"""!
@file test_punto3_finetuning.py
@brief Pruebas pequeñas del ajuste LoRA: máscaras, pérdida causal y gradientes.
"""

from types import SimpleNamespace
import sys
import json

import pytest
import torch

from notebook_loader import notebook as punto3


class TinyTokenizer:
    """! @brief Tokenizador mínimo que comparte EOS y padding para detectar enmascarados incorrectos. """
    chat_template = None
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        """!
        @brief Distingue prompt y etiquetas sin descargar tokenizadores.
        @param text Texto solicitado.
        @param add_special_tokens Indica si se agregan tokens especiales.
        @return Lista pequeña de identificadores.
        """
        return [1] if text == 'ham' else [2] if text == 'spam' else [3, 4, 3]


class TinyCausalModel(torch.nn.Module):
    """! @brief Modelo causal pequeño con base congelada y una actualización entrenable. """
    def __init__(self):
        """! @brief Inicializa pesos reproducibles para comparar gradientes. @return None. """
        super().__init__()
        self.base = torch.nn.Parameter(torch.zeros(5, 5), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.arange(25, dtype=torch.float32).reshape(5, 5) / 100)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kwargs):
        """!
        @brief Devuelve logits completos o posiciones seleccionadas del mismo cálculo.
        @param input_ids Identificadores de tokens.
        @param attention_mask Máscara del lote.
        @param logits_to_keep Índices de posiciones que se conservan.
        @param kwargs Opciones de la interfaz causal.
        @return Objeto con logits diferenciables.
        """
        logits = self.base[input_ids] + self.lora_A[input_ids]
        if not isinstance(logits_to_keep, int):
            logits = logits[:, logits_to_keep, :]
        return SimpleNamespace(logits=logits)


def _tiny_llm():
    """! @brief Construye un modelo pequeño exclusivamente CPU. @return LanguageModel de prueba. """
    return punto3.LanguageModel('tiny', TinyTokenizer(), TinyCausalModel(), 'cpu', 'a' * 40)


def test_adapter_examples_mask_prompt_and_keep_real_eos():
    """! @brief Verifica que solo la respuesta y EOS sean supervisados. @return None. """
    examples = punto3.prepare_adapter_examples(TinyTokenizer(), ['Hola\nsin cambiar', 'WIN'], [0, 1])
    assert examples[0] == {'input_ids': [3, 4, 3, 1, 0], 'labels': [-100, -100, -100, 1, 0]}
    shorter = {'input_ids': [3, 2, 0], 'labels': [-100, 2, 0]}
    batch = punto3.collate_adapter_examples([examples[0], shorter], pad_token_id=0)
    assert batch['labels'][1].tolist() == [-100, 2, 0, -100, -100]
    assert batch['attention_mask'][1].tolist() == [1, 1, 1, 0, 0]


@pytest.mark.parametrize('labels,max_length', [([2], 512), ([0, 1], 512), ([0], 4)])
def test_adapter_examples_reject_invalid_labels_and_truncation(labels, max_length):
    """!
    @brief Rechaza etiquetas incompatibles y mensajes que no caben, sin truncarlos.
    @param labels Etiquetas del caso.
    @param max_length Límite del caso.
    @return None.
    """
    with pytest.raises(ValueError):
        punto3.prepare_adapter_examples(TinyTokenizer(), ['SMS'], labels, max_length)


def test_sparse_completion_loss_matches_full_masked_loss_and_gradient():
    """! @brief Compara pérdida y gradientes con la entropía cruzada causal completa. @return None. """
    llm = _tiny_llm()
    examples = punto3.prepare_adapter_examples(llm.tokenizer, ['A', 'B'], [0, 1])
    batch = punto3.collate_adapter_examples(examples, 0)
    sparse = punto3.adapter_completion_loss(llm.model, batch)
    sparse_gradient = torch.autograd.grad(sparse, llm.model.lora_A)[0]
    logits = llm.model(batch['input_ids']).logits
    full = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 5),
                                             batch['labels'][:, 1:].reshape(-1), ignore_index=-100)
    full_gradient = torch.autograd.grad(full, llm.model.lora_A)[0]
    torch.testing.assert_close(sparse, full)
    torch.testing.assert_close(sparse_gradient, full_gradient)
    assert llm.model.base.grad is None


def test_capacity_probe_restores_weights_rng_and_mode():
    """! @brief La prueba física no aporta entrenamiento ni consume aleatoriedad del ajuste. @return None. """
    llm = _tiny_llm()
    llm.model.eval()
    initial = llm.model.lora_A.detach().clone()
    rng = torch.get_rng_state().clone()
    example = punto3.prepare_adapter_examples(llm.tokenizer, ['A'], [1])[0]
    result = punto3.probe_adapter_training(llm, example)
    torch.testing.assert_close(llm.model.lora_A, initial, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not llm.model.training
    assert llm.model.lora_A.grad is None
    assert result['loss'] > 0 and result['tokens'] == 5


def test_accumulation_normalizes_last_partial_group_and_keeps_base_fixed():
    """! @brief Compara acumulación con dos pasos de referencia, incluido el grupo incompleto. @return None. """
    llm = _tiny_llm()
    reference = _tiny_llm()
    examples = punto3.prepare_adapter_examples(llm.tokenizer, ['A', 'B', 'C'], [0, 1, 1])
    options = punto3.adapter_training_config({'gradient_accumulation_steps': 2, 'max_grad_norm': 100.0})
    optimizer = torch.optim.SGD([llm.model.lora_A], lr=0.1)
    reference_optimizer = torch.optim.SGD([reference.model.lora_A], lr=0.1)
    order = torch.randperm(3, generator=torch.Generator().manual_seed(7)).tolist()
    for indices in (order[:2], order[2:]):
        reference_optimizer.zero_grad()
        losses = [punto3.adapter_completion_loss(reference.model,
                  punto3.collate_adapter_examples([examples[index]], 0)) for index in indices]
        torch.stack(losses).mean().backward()
        reference_optimizer.step()
    initial_base = llm.model.base.detach().clone()
    rows = punto3.train_adapter_epoch(llm, examples, optimizer, options, seed=7)
    torch.testing.assert_close(llm.model.lora_A, reference.model.lora_A)
    torch.testing.assert_close(llm.model.base, initial_base, rtol=0, atol=0)
    assert [row['examples_seen'] for row in rows] == [2, 3]


def test_adapter_partitions_exclude_test_from_fit_and_validation():
    """! @brief Verifica separación por filas y reproducibilidad de la partición anidada. @return None. """
    targets = torch.tensor([0] * 80 + [1] * 20)
    parts = punto3.adapter_partition_indices(targets, seed=2026)
    repeated = punto3.adapter_partition_indices(targets, seed=2026)
    sets = {key: set(value.tolist()) for key, value in parts.items()}
    assert sets['fit'].isdisjoint(sets['validation'])
    assert sets['fit'] | sets['validation'] == sets['train']
    assert sets['train'].isdisjoint(sets['test'])
    assert sets['train'] | sets['test'] == set(range(100))
    for key in parts:
        assert torch.equal(parts[key], repeated[key])


def test_prepare_lora_freezes_base_without_upcasting_embeddings(monkeypatch):
    """!
    @brief La preparación congela la base BF16 y entrena solo adaptadores.
    @param monkeypatch Simulación de CUDA y PEFT, sin descargas.
    @return None.
    """
    base = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    base.is_loaded_in_4bit = True
    base.config = SimpleNamespace(use_cache=True)
    calls = {}
    base.gradient_checkpointing_enable = lambda **kwargs: calls.update(kwargs)

    def attach(model, config, **kwargs):
        """!
        @brief Agrega una matriz de adaptación simulada sobre la base congelada.
        @param model Modelo pequeño.
        @param config Configuración registrada.
        @param kwargs Opciones de PEFT.
        @return El mismo modelo con un adaptador entrenable.
        """
        assert kwargs['autocast_adapter_dtype'] is True
        model.register_parameter('lora_A', torch.nn.Parameter(torch.zeros(2, 1)))
        return model

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(LoraConfig=lambda **kw: kw,
                                                            get_peft_model=attach))
    llm = punto3.LanguageModel('tiny', TinyTokenizer(), base, 'cuda', 'a' * 40)
    punto3.prepare_lora_model(llm)
    assert base.weight.dtype == torch.bfloat16
    assert not base.weight.requires_grad and not base.bias.requires_grad
    assert base.lora_A.requires_grad
    assert calls == {'gradient_checkpointing_kwargs': {'use_reentrant': False}}


def test_classify_finetuned_preserves_generation_contract(monkeypatch):
    """!
    @brief La clasificación ajustada reutiliza el clasificador y el orden de 3.1.
    @param monkeypatch Sustituye la generación por una función sin modelos grandes.
    @return None.
    """
    llm = _tiny_llm()
    expected = torch.tensor([1, 0], dtype=torch.int64)
    calls = {}

    def classify(texts, model, batch_size):
        """!
        @brief Registra los argumentos de clasificación.
        @param texts Mensajes recibidos.
        @param model Modelo recibido.
        @param batch_size Lote recibido.
        @return Predicciones de referencia.
        """
        calls.update(texts=texts, model=model, batch_size=batch_size)
        return expected

    monkeypatch.setattr(punto3, 'classify_messages', classify)
    assert torch.equal(punto3.classify_finetuned(llm, ['WIN', 'Hola'], 2), expected)
    assert calls == {'texts': ['WIN', 'Hola'], 'model': llm, 'batch_size': 2}


def test_training_history_figure_has_descriptive_export_title():
    """! @brief La figura representa registros reales y devuelve título exportable sin mostrarla. @return None. """
    import matplotlib.pyplot as plt
    figure = punto3.plot_adapter_training_history(
        [{'step': 1, 'loss': 0.9}, {'step': 2, 'loss': 0.5}],
        [{'epoch': 0, 'validation_loss': 1.0}, {'epoch': 1, 'validation_loss': 0.6}],
        title='Historial de prueba',
    )
    assert figure._suptitle.get_text() == 'Historial de prueba'
    assert len(figure.axes) == 2
    plt.close(figure)


@pytest.mark.parametrize('changed_adapter', [False, True])
def test_adapter_cache_rejects_history_from_other_weights(tmp_path, changed_adapter):
    """!
    @brief La evaluación no combina respuestas e historial de adaptadores diferentes.
    @param tmp_path Carpeta temporal del experimento.
    @param changed_adapter Si se simula reemplazar el adaptador después de generar respuestas.
    @return None.
    """
    metadata = punto3.llm_cache_metadata(['A'], 'tiny+LoRA', punto3.DEFAULT_PROMPT, 8)
    details = {'adapter_sha256': {'adapter_model.safetensors': 'a' * 64}, 'training': {'epochs': 1}}
    punto3.save_llm_responses(tmp_path / 'responses.json', metadata, ['ham'], details)
    training = {'request': metadata, **details}
    if changed_adapter:
        training = {**training, 'adapter_sha256': {'adapter_model.safetensors': 'b' * 64}}
    (tmp_path / 'training.json').write_text(json.dumps(training))
    if changed_adapter:
        with pytest.raises(ValueError):
            punto3.load_adapter_experiment(tmp_path, metadata)
    else:
        response, record = punto3.load_adapter_experiment(tmp_path, metadata)
        assert response['responses'] == ['ham']
        assert record == training
