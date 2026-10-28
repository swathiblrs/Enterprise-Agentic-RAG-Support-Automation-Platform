"""Local-only tests: no GPU, network, models or extra packages."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent/'qlora_training/llama_retraining'
sys.path.insert(0, str(HERE))
from label_space import label_space


def retraining():
    spec = importlib.util.spec_from_file_location('retraining_test', HERE/'retrain.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    eos_token_id = 128009
    labels = ['cloud', 'data', 'it', 'other', 'production', 'security']
    def encode(self, text, **kwargs):
        label = json.loads(text)['domain']
        return [5018, 12482, 3332, 100+ self.labels.index(label), 9388]
    def __call__(self, text, **kwargs):
        domain = json.loads(text)['domain']
        end = 11+len(domain)
        return {'input_ids': self.encode(text), 'offset_mapping': [(0, 2), (2, 8), (8, 11), (11, end), (end, end+2)]}


def test_single_token_label_space():
    result = label_space(Tokenizer())
    assert result['token_ids'] == list(range(100, 106))
    assert result['completion_prefix'] == [5018, 12482, 3332]
    assert result['completion_suffix'] == [9388, 128009]


def test_multi_token_labels_fail_instead_of_changing_scorer():
    class CharacterTokenizer:
        eos_token_id = 999
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]
        def __call__(self, text, **kwargs):
            return {'input_ids': self.encode(text), 'offset_mapping': [(i, i+1) for i in range(len(text))]}
    with pytest.raises(ValueError, match='exactly one'):
        label_space(CharacterTokenizer())


def test_training_answer_is_not_in_input():
    module = retraining()
    tokenizer = Tokenizer()
    spec = label_space(tokenizer)
    def encode(tokenizer, text, domain, *args, **kwargs):
        value = 100+tokenizer.labels.index(domain)
        return {'input_ids': [77, 78, 5018, 12482, 3332, value, 9388, 128009],
                'attention_mask': [1]*8, 'labels': [-100]*5+[value, -100, -100]}
    support = SimpleNamespace(encode_llama=encode)
    first = module.prefix_features(support, tokenizer, spec, 'request', 'it')
    second = module.prefix_features(support, tokenizer, spec, 'request', 'security')
    assert first['input_ids'] == second['input_ids'] == [77, 78, 5018, 12482, 3332]
    assert first['labels'] == 2 and second['labels'] == 5
    assert not set(spec['token_ids']) & set(first['input_ids'])


def test_non_batch_one_is_rejected_before_forward():
    with pytest.raises(ValueError, match='batch size one'):
        retraining().restricted_logits(None, SimpleNamespace(shape=(2, 10)), None, [])


def test_no_token_and_macro_f1_checkpoint_configuration():
    config = json.loads((HERE/'config.json').read_text())
    assert config['requires_huggingface_access'] is False
    assert config['learning_rate'] == .00005
    assert config['checkpoint_metric'] == 'macro_f1'
    assert config['max_steps'] == 300
    source = (HERE/'retrain.py').read_text()
    assert "metric_for_best_model='macro_f1'" in source
    assert 'self.model_accepts_loss_kwargs = False' in source


def test_notebook_has_no_credentials_and_uses_new_output():
    notebook = json.loads((HERE/'Llama_Retraining_NoToken.ipynb').read_text())
    source = ''
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            text = ''.join(cell['source'])
            ast.parse(text)
            source += text
    assert 'userdata' not in source and 'access_token' not in source
    assert 'token=False' in source
    assert 'llama_retraining_bundle.zip' in source
    assert '/content/llama_retrained_results.zip' in source
