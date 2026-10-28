"""Package existing artifacts for a second Llama run, without network access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
sys.path.insert(0, str(SHARED))
from protocol import audit_splits, digest, metrics, read_rows, softmax


def build(previous, destination):
    data = SHARED/'retraining_v2/data'
    train, valid, benchmark = [read_rows(data/name) for name in
                               ('train.jsonl', 'validation.jsonl', 'challenge.jsonl')]
    audit = audit_splits(train, valid, benchmark)
    config = json.loads((HERE/'config.json').read_text())
    with zipfile.ZipFile(previous) as archive:
        if sum(info.file_size for info in archive.infolist()) > 100*1024**2:
            raise ValueError('Unexpected prior archive size')
        report = json.loads(archive.read('comparison.json'))
        training = json.loads(archive.read('training.json'))
        if archive.read('challenge_opened.lock').decode() != hashlib.sha256(archive.read('selection.json')).hexdigest():
            raise ValueError('Prior selection hash mismatch')
        if any(training['config'][key] != config[key] for key in ('model_id', 'revision')):
            raise ValueError('Prior adapter is not the same pinned Llama checkpoint')
        for filename, key in (('train.jsonl', 'train_sha256'), ('validation.jsonl', 'validation_sha256')):
            if digest(data/filename) != training[key]:
                raise ValueError('Prior training data mismatch')
        if training['optimizer_steps'] <= 0 or training['changed_adapter_tensors'] <= 0:
            raise ValueError('Prior training is not verified')
        prediction = json.loads(archive.read('llama_qlora_challenge.json'))
        if prediction['ids'] != [r['id'] for r in benchmark]:
            raise ValueError('Prior benchmark case order mismatch')
        choice = report['selection']['candidates']['llama_qlora']
        actual = metrics(benchmark, [softmax(s, choice['temperature']) for s in prediction['scores']],
                         choice['threshold'], choice['margin'])
        if actual['confusion_matrix'] != report['test']['llama_qlora']['metrics']['confusion_matrix']:
            raise ValueError('Prior benchmark metrics mismatch')
        members = {name: (HERE/name).read_bytes() for name in
                   ('retrain.py', 'config.json', 'label_space.py', 'README.md')}
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            members['previous_adapter/'+filename] = archive.read('adapter/'+filename)
        adapter = json.loads(members['previous_adapter/adapter_config.json'])
        if adapter['base_model_name_or_path'] != config['model_id']:
            raise ValueError('Adapter identity mismatch')
        snapshot = HERE/'previous_run'
        snapshot.mkdir(exist_ok=True)
        for filename in ('training.json', 'comparison.json', 'RESULTS.md'):
            (snapshot/filename).write_bytes(archive.read(filename))
    references = {'source_archive_sha256': digest(previous), 'historical_only': True, 'models': {}}
    for name, result in report['test'].items():
        references['models'][name] = {
            'metrics': {key: value for key, value in result['metrics'].items() if key != 'cases'},
            'engineering': result['engineering'], 'scoring': 'previous_six_pass_json_likelihood',
            'execution': 'historical_not_reexecuted'}
    members['references.json'] = json.dumps(references).encode()
    for name in ('protocol.py', 'export_results.py', 'requirements-colab.txt'):
        members[name] = (SHARED/name).read_bytes()
    members['engine.py'] = (SHARED/'experiment.py').read_bytes()
    members['llama_support.py'] = (SHARED/'llama_training/llama_experiment.py').read_bytes()
    for name in ('train.jsonl', 'validation.jsonl', 'challenge.jsonl'):
        members['data/'+name] = (data/name).read_bytes()
    manifest = {'schema_version': 1, 'provenance': {
        'dataset_audit': audit, 'scope': 'Known synthetic development benchmark, not blind or real-world validation',
        'training_data_changed': False, 'training_examples': 612, 'development_examples': 210,
        'benchmark_examples': 120, 'method': 'six native domain-token classifier; old and new adapters rerun identically',
        'previous_results_sha256': digest(previous), 'model': config['model_id'],
        'authentication': 'public checkpoint, token=False, no secrets',
        'safety': 'Only the already-downloaded prior adapter is included; no new model downloads on Mac'},
        'files': {name: hashlib.sha256(value).hexdigest() for name, value in members.items()}}
    members['manifest.json'] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    print(json.dumps({'bundle': str(destination), 'mib': destination.stat().st_size/1024**2,
                      'new_weights_downloaded': False, 'prior_adapter_included': True, 'dataset_audit': audit}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, default=Path.home()/'Downloads/llama_qlora_results.zip')
    parser.add_argument('--output', type=Path, default=HERE/'llama_retraining_bundle.zip')
    args = parser.parse_args()
    build(args.previous, args.output)
