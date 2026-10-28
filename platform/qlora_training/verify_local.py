"""Offline validation-path smoke test with existing weights; never runs QLoRA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parent


def baseline(name, out):
    sys.path.insert(0, str(PLATFORM))
    import domain_classifier
    sys.modules['portable_tfidf'] = domain_classifier
    import experiment
    with tempfile.TemporaryDirectory(prefix='qlora-baseline-smoke-') as temporary:
        root = Path(temporary)
        (root/'data').symlink_to(PLATFORM/'routing_training/data', target_is_directory=True)
        (root/'baselines').symlink_to(PLATFORM/'routing_models/grouped-minilm-v1', target_is_directory=True)
        experiment.ROOT = root
        experiment.worker(name, 'validation', out)


def run(out):
    from protocol import audit_splits, calibrate, metrics, read_rows, softmax
    out.mkdir(parents=True, exist_ok=False)
    audit = audit_splits(read_rows(PLATFORM/'routing_training/data/train.jsonl'),
                         read_rows(PLATFORM/'routing_training/data/validation.jsonl'),
                         read_rows(HERE/'challenge.jsonl'))
    rows = read_rows(PLATFORM/'routing_training/data/validation.jsonl')
    previous = json.loads((PLATFORM/'routing_training/reports/grouped-minilm-v1/comparison.json').read_text())
    results = {}
    for name in ('tfidf', 'frozen', 'setfit'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model', name, '--out', str(out)], check=True)
        record = json.loads((out/f'{name}_validation.json').read_text())
        if record['ids'] != [row['id'] for row in rows]:
            raise RuntimeError('Validation row alignment failed')
        choice = calibrate(rows, record['scores'])
        score = metrics(rows, [softmax(s, choice['temperature']) for s in record.pop('scores')],
                        choice['threshold'], choice['margin'])
        expected = previous['selection']['candidates'][name]['validation']['cases']
        if [r['predicted'] for r in score['cases']] != [r['predicted'] for r in expected]:
            raise RuntimeError('Portable baseline changed predictions: '+name)
        results[name] = {'validation': score, 'engineering': record, 'original_predictions_match': True}
    report = {'status': 'local_validation_only', 'network': 'HF offline modes; existing model files only',
              'qlora_training_executed': False, 'challenge_predictions_opened': False,
              'dataset_audit': audit, 'results': results}
    (out/'local_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'dataset_audit': audit,
                      'baseline_matches': {name: r['original_predictions_match'] for name, r in results.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tfidf', 'frozen', 'setfit'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.model:
        baseline(args.model, args.out)
    else:
        run(args.out)
