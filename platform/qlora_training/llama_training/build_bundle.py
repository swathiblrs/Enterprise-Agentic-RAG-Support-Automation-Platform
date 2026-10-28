"""Prepare a small Llama bundle with cached references; never download models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
sys.path.insert(0, str(SHARED))
from protocol import audit_splits, calibrate, digest, metrics, read_rows, softmax


def build(results, destination):
    data = SHARED/'retraining_v2/data'
    train, valid, benchmark = [read_rows(data/n) for n in ('train.jsonl', 'validation.jsonl', 'challenge.jsonl')]
    audit = audit_splits(train, valid, benchmark)
    with zipfile.ZipFile(results) as archive:
        report = json.loads(archive.read('comparison.json'))
        evidence = json.loads(archive.read('training.json'))
        if archive.read('challenge_opened.lock').decode() != hashlib.sha256(archive.read('selection.json')).hexdigest():
            raise ValueError('Qwen reference selection lock is invalid')
        if any(digest(data/n) != evidence[key] for n, key in
               (('train.jsonl', 'train_sha256'), ('validation.jsonl', 'validation_sha256'))):
            raise ValueError('Reference training data differs')
        if evidence['optimizer_steps'] <= 0 or evidence['changed_adapter_tensors'] <= 0:
            raise ValueError('Reference adapter has no verified training evidence')
        references = {'source_archive_sha256': digest(results), 'hardware': report['hardware'], 'candidates': {}}
        for source, name in (('tfidf', 'tfidf'), ('frozen', 'frozen_minilm'), ('setfit', 'setfit'),
                             ('base_qwen', 'qwen3_base'), ('qlora', 'qwen3_qlora')):
            validation = json.loads(archive.read(source+'_validation.json'))
            challenge = json.loads(archive.read(source+'_challenge.json'))
            if validation['ids'] != [r['id'] for r in valid] or challenge['ids'] != [r['id'] for r in benchmark]:
                raise ValueError('Reference case order differs')
            choice = calibrate(valid, validation['scores'])
            actual = metrics(benchmark, [softmax(s, choice['temperature']) for s in challenge['scores']],
                             choice['threshold'], choice['margin'])
            if actual['confusion_matrix'] != report['test'][source]['metrics']['confusion_matrix']:
                raise ValueError('Reference metrics cannot be reproduced')
            references['candidates'][name] = {'validation': validation, 'challenge': challenge}
        saved = HERE/'qwen_v2_reference'
        saved.mkdir(exist_ok=True)
        for name in ('training.json', 'comparison.json', 'RESULTS.md'):
            (saved/name).write_bytes(archive.read(name))
    members = {n: (HERE/n).read_bytes() for n in ('llama_experiment.py', 'config.json', 'README.md')}
    for name in ('protocol.py', 'export_results.py', 'requirements-colab.txt'):
        members[name] = (SHARED/name).read_bytes()
    members['engine.py'] = (SHARED/'experiment.py').read_bytes()
    for name in ('train.jsonl', 'validation.jsonl', 'challenge.jsonl'):
        members['data/'+name] = (data/name).read_bytes()
    members['references.json'] = json.dumps(references).encode()
    manifest = {'schema_version': 1, 'provenance': {
        'scope': 'Previously inspected 120-case synthetic benchmark; not a fresh held-out test',
        'audit': audit, 'model': 'unsloth/Llama-3.2-1B-Instruct',
        'upstream_model': 'meta-llama/Llama-3.2-1B-Instruct', 'authentication': 'anonymous token=False',
        'training': 'Same 612 examples as Qwen v2; rank16, domain-only loss, effective batch8',
        'validation': 'Same 210 development cases as Qwen v2',
        'references': 'Verified prior-session predictions, not newly executed baselines',
        'qwen_results_sha256': digest(results),
        'safety': 'No model weights, credentials, private documents, or databases in this bundle'},
        'files': {n: hashlib.sha256(v).hexdigest() for n, v in members.items()}}
    members['manifest.json'] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    print(json.dumps({'bundle': str(destination), 'mib': destination.stat().st_size/1024**2,
                      'weights_included': False, 'model_access': 'Public Unsloth distribution, no token; Meta Llama license still applies'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path.home()/'Downloads/qlora_v2_results.zip')
    parser.add_argument('--output', type=Path, default=HERE/'llama_no_token_bundle.zip')
    args = parser.parse_args()
    build(args.results, args.output)
