"""No downloads, network, torch installation or model loading in these tests."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent/'qlora_training/llama_training'


def module():
    spec = importlib.util.spec_from_file_location('llama_experiment_test', HERE/'llama_experiment.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_model_identity_and_scope():
    config = json.loads((HERE/'config.json').read_text())
    assert config['model_id'] == 'unsloth/Llama-3.2-1B-Instruct'
    assert config['requires_huggingface_access'] is False
    assert len(config['revision']) == 40
    assert config['batch_size']*config['gradient_accumulation_steps'] == 8
    assert 'not_blind_test' in config['benchmark_scope']


def test_fixed_date_and_label_mask():
    m = module()
    class Tokenizer:
        eos_token_id = 999
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['date_string'] == '01 Jan 2025'
            assert 'enable_thinking' not in kwargs
            return [1000]
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]
        def __call__(self, text, **kwargs):
            return {'input_ids': self.encode(text), 'offset_mapping': [(i, i+1) for i in range(len(text))]}
    record = m.encode_llama(Tokenizer(), 'request', 'security', domain_only=True)
    assert ''.join(chr(v) for v in record['labels'] if v != -100) == 'security'
    assert record['labels'][0] == record['labels'][-1] == -100


def test_hardware_gate_before_downloads(monkeypatch):
    m = module()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match='No Mac/CPU'):
        m.cuda_check()


def test_small_gpu_rejected(monkeypatch):
    m = module()
    props = SimpleNamespace(total_memory=8*1024**3, major=7)
    cuda = SimpleNamespace(is_available=lambda: True, get_device_properties=lambda _: props)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match='10 GiB'):
        m.cuda_check()


def test_anonymous_loader_never_uses_saved_token(monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'cuda_check', lambda: {})
    monkeypatch.setenv('HF_TOKEN', 'invalid-existing-token-must-not-be-used')
    calls = []
    class Tokenizer:
        eos_token = 'eos'
    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == 'unsloth/Llama-3.2-1B-Instruct'
            assert kwargs['token'] is False
            assert kwargs['trust_remote_code'] is False
            calls.append(kwargs)
            return Tokenizer() if len(calls) == 1 else SimpleNamespace(
                is_loaded_in_4bit=True, config=SimpleNamespace(model_type='llama'))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: False), float16='fp16')
    fake_transformers = SimpleNamespace(AutoTokenizer=Factory, AutoModelForCausalLM=Factory,
                                        BitsAndBytesConfig=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    engine = m.configure_engine()
    engine.load_qwen()
    assert len(calls) == 2
    assert calls[1]['use_safetensors'] is True


def test_unknown_model_not_silently_loaded():
    engine = module().configure_engine()
    with pytest.raises(ValueError, match='Unknown Llama'):
        engine.make_predictor('qwen', Path('/tmp'))


def test_prediction_row_alignment():
    m = module()
    with pytest.raises(ValueError, match='row order'):
        m.scored([{'id': 'expected', 'domain': 'it'}], {'ids': ['different']}, {})


def test_notebook_python_and_secret_handling():
    notebook = json.loads((HERE/'Llama_QLoRA_NoToken.ipynb').read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    for cell in cells:
        ast.parse(''.join(cell['source']))
    source = '\n'.join(''.join(c['source']) for c in cells)
    assert 'userdata' not in source
    assert 'token=False' in source
    assert 'getpass(' not in source
    assert 'print(access_token' not in source
    assert 'access_token' not in source
    assert 'hf_hub_download' in source
