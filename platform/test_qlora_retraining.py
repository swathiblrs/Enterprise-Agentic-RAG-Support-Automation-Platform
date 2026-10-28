"""No GPU, network or package installation required for these checks."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'qlora_training'))
from protocol import encode_example


class CharacterTokenizer:
    eos_token_id = 1000
    def apply_chat_template(self, *args, **kwargs):
        return [1001, 1002]
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [(i, i+1) for i in range(len(text))]}


@pytest.mark.parametrize('domain', ['cloud', 'data', 'it', 'other', 'production', 'security'])
def test_domain_only_masks_json_and_eos(domain):
    result = encode_example(CharacterTokenizer(), 'request', domain, domain_only=True)
    target = [v for v in result['labels'] if v != -100]
    assert ''.join(chr(c) for c in target) == domain
    assert result['labels'][-1] == -100
    assert result['input_ids'][-1] == 1000


def test_legacy_objective_is_unchanged():
    result = encode_example(CharacterTokenizer(), 'request', 'security')
    assert [v for v in result['labels'] if v != -100] == CharacterTokenizer().encode('{"domain":"security"}')+[1000]


def test_new_dataset_has_no_original_test_in_training():
    path = ROOT/'qlora_training/retraining_v2/build_retraining.py'
    spec = importlib.util.spec_from_file_location('build_retraining_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train, valid, test, audit = module.materialize()
    assert audit['sizes'] == [612, 210, 120]
    assert audit['groups'][2] == 60
    assert all(not r['id'].startswith('challenge-') for r in train)
    assert sum(r['id'].startswith('challenge-') for r in valid) == 120
    assert all(r['id'].startswith('v2-test') for r in test)


def test_retraining_notebook_is_valid_python():
    notebook = json.loads((ROOT/'qlora_training/retraining_v2/QLoRA_Retraining_v2.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))


def test_reference_adapter_is_in_common_comparison():
    config = json.loads((ROOT/'qlora_training/retraining_v2/config.json').read_text())
    assert 'qlora_v1' in config['comparison_models']
    assert config['domain_only_loss'] is True
    assert config['learning_rate'] == .0001
    source = (ROOT/'qlora_training/experiment.py').read_text()
    assert "ROOT/'reference_adapter' if name == 'qlora_v1'" in source
