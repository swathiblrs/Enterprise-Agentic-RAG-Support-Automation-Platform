"""CPU-only checks: these do not prove CUDA training or model accuracy."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'qlora_training'))
from protocol import (LABELS, audit_splits, calibrate, encode_example, metrics,
                      pad_examples, read_rows, softmax, verify_bundle)


class Tokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert messages[1]['role'] == 'user'
        return [1, 2, 3]
    def encode(self, text, **kwargs):
        assert json.loads(text)['domain'] in LABELS
        return [4, 5]


def test_completion_only_and_padding():
    example = encode_example(Tokenizer(), 'VPN broken', 'it')
    assert example['labels'] == [-100, -100, -100, 4, 5, 9]
    smaller = {k: v[:-1] for k, v in example.items()}
    batch = pad_examples([example, smaller], 9)
    assert batch['labels'][1][-1] == -100
    assert batch['attention_mask'][1][-1] == 0


def test_long_inputs_fail_without_silent_truncation():
    with pytest.raises(ValueError, match='token limit'):
        encode_example(Tokenizer(), 'request', 'it', max_length=5)


def test_metrics_known_confusion():
    rows = [{'id': str(i), 'domain': label} for i, label in enumerate(LABELS)]
    probs = [[float(i == j) for j in range(6)] for i in range(6)]
    result = metrics(rows, probs)
    assert result['macro_f1'] == result['accuracy'] == 1
    assert result['per_domain_recall']['security'] == 1
    assert result['auto_route_coverage'] == 5/6
    probs[0] = probs[1]
    assert metrics(rows, probs)['macro_f1'] == pytest.approx((0+2/3+4)/6)


def test_calibration_validation_only():
    rows = [{'id': str(i), 'domain': label} for i, label in enumerate(LABELS)]
    scores = [[10.*(i == j) for j in range(6)] for i in range(6)]
    selection = calibrate(rows, scores)
    assert selection['selected_using'] == 'validation_only'
    assert sum(softmax([10000, 9999])) == pytest.approx(1.)


@pytest.mark.parametrize('field,value', [('group_id', 'train-group'), ('text', 'same request')])
def test_overlap_rejected(field, value):
    train = [{'id': 'a', 'text': 'same request', 'domain': 'it', 'group_id': 'train-group'}]
    valid = [{'id': 'b', 'text': 'distinct', 'domain': 'it', 'group_id': 'valid-group'}]
    valid[0][field] = value
    with pytest.raises(ValueError, match='leakage'):
        audit_splits(train, valid, [])


def test_real_splits_are_disjoint():
    train, valid = [read_rows(HERE/'routing_training/data'/n) for n in ('train.jsonl', 'validation.jsonl')]
    challenge = read_rows(HERE/'qlora_training/challenge.jsonl')
    result = audit_splits(train, valid, challenge)
    assert result['sizes'] == [1620, 90, 120]
    assert result['groups'][2] == 60
    assert all(sum(r['domain'] == label for r in challenge) == 20 for label in LABELS)


def test_notebook_code_and_no_local_install():
    notebook = json.loads((HERE/'qlora_training/QLoRA_Comparison.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    source = (HERE/'qlora_training/experiment.py').read_text()
    assert "cuda_check()  # Must precede" in source
    assert 'trust_remote_code=False' in source
    assert 'use_safetensors=True' in source


def test_cuda_gate_precedes_downloads(monkeypatch):
    spec = importlib.util.spec_from_file_location('qlora_experiment', HERE/'qlora_training/experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class FakeCuda:
        @staticmethod
        def is_available():
            return False
    class FakeTorch:
        cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch())
    with pytest.raises(RuntimeError, match='No CPU/Mac fallback'):
        module.load_qwen()


def test_bundle_integrity_and_path_guard(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'files': {'../escape': 'bad'}}))
    with pytest.raises(ValueError, match='unsafe'):
        verify_bundle(tmp_path)


def test_challenge_requires_frozen_selection(tmp_path):
    spec = importlib.util.spec_from_file_location('qlora_experiment', HERE/'qlora_training/experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match='frozen validation selection'):
        module.worker('tfidf', 'challenge', tmp_path)


def test_result_export_excludes_optimizer_pickle(tmp_path):
    import zipfile
    from export_results import export
    source = tmp_path/'results'
    (source/'checkpoints').mkdir(parents=True)
    (source/'adapter').mkdir()
    (source/'comparison.json').write_text('{}')
    (source/'adapter/adapter_model.safetensors').write_bytes(b'test fixture, not real weights')
    (source/'checkpoints/optimizer.pt').write_bytes(b'not for export')
    target = tmp_path/'results.zip'
    export(source, target)
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {'comparison.json', 'adapter/adapter_model.safetensors'}


def test_incomplete_result_cannot_be_exported(tmp_path):
    from export_results import export
    with pytest.raises(ValueError, match='not completed'):
        export(tmp_path, tmp_path/'partial.zip')
