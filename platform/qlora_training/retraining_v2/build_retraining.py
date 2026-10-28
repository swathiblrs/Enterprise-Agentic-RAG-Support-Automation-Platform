"""Build a second Colab experiment without downloads, installs or local training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
PLATFORM = SHARED.parent
sys.path.insert(0, str(SHARED))
from protocol import LABELS, audit_splits, digest, read_rows


def materialize():
    specification = json.loads((HERE/'scenarios.json').read_text())
    original = read_rows(PLATFORM/'routing_training/data/train.jsonl')
    train = [r for r in original if r['variant'] == 0]
    valid = read_rows(PLATFORM/'routing_training/data/validation.jsonl') + read_rows(SHARED/'challenge.jsonl')
    added, test = [], []
    for split, destination in (('train', added), ('test', test)):
        for domain in LABELS:
            for group, texts in enumerate(specification[split][domain]):
                for index, text in enumerate(texts):
                    destination.append({'id': f'v2-{split}-{domain}-{group}-{index}', 'text': text,
                                        'domain': domain, 'group_id': f'v2/{split}/{domain}/{group}',
                                        'origin': 'project_authored_synthetic',
                                        'review_status': 'not_independently_reviewed'})
    train += added
    audit = audit_splits(train, valid, test)
    for rows in (train, valid, test):
        counts = Counter(r['domain'] for r in rows)
        if len(set(counts.values())) != 1:
            raise ValueError('Unbalanced classes')
    audit['class_counts'] = [dict(Counter(r['domain'] for r in rows)) for rows in (train, valid, test)]
    audit['training_changes'] = 'Removed 1080 formatting variants; retained 540 scenarios; added 72 boundary examples.'
    audit['validation_notice'] = '210 development examples include the 120 previously inspected test cases.'
    audit['held_out_notice'] = '120 new synthetic cases in 60 paraphrase groups. Not human-reviewed or real tickets.'
    return train, valid, test, audit


def build(results, destination):
    train, valid, test, audit = materialize()
    data_dir = HERE/'data'
    data_dir.mkdir(exist_ok=True)
    for name, rows in (('train.jsonl', train), ('validation.jsonl', valid), ('challenge.jsonl', test)):
        (data_dir/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (HERE/'data_audit.json').write_text(json.dumps(audit, indent=2))
    with zipfile.ZipFile(results) as previous:
        if sum(i.file_size for i in previous.infolist()) > 100*1024**2:
            raise ValueError('Unexpected result archive size')
        selection_bytes = previous.read('selection.json')
        if previous.read('challenge_opened.lock').decode() != hashlib.sha256(selection_bytes).hexdigest():
            raise ValueError('Prior validation lock mismatch')
        training = json.loads(previous.read('training.json'))
        if training['optimizer_steps'] <= 0 or training['changed_adapter_tensors'] <= 0:
            raise ValueError('Prior adapter is not verified as trained')
        comparison = json.loads(previous.read('comparison.json'))
        if digest(PLATFORM/'routing_training/data/train.jsonl') != training['train_sha256']:
            raise ValueError('Original training data mismatch')
        members = {}
        # Reuse existing baseline weights without fetching any models.
        with zipfile.ZipFile(SHARED/'colab_bundle.zip') as base:
            if hashlib.sha256(base.read('manifest.json')).hexdigest() != comparison['bundle_manifest_sha256']:
                raise ValueError('Prior run did not use the expected original bundle')
            for name in base.namelist():
                if name.startswith('baselines/') or name == 'portable_tfidf.py':
                    members[name] = base.read(name)
        for name in ('protocol.py', 'experiment.py', 'export_results.py', 'requirements-colab.txt'):
            members[name] = (SHARED/name).read_bytes()
        for name in ('train.jsonl', 'validation.jsonl', 'challenge.jsonl'):
            members['data/'+name] = (data_dir/name).read_bytes()
        members['data_audit.json'] = (HERE/'data_audit.json').read_bytes()
        members['README.md'] = (HERE/'README.md').read_bytes()
        members['config.json'] = (HERE/'config.json').read_bytes()
        for name in previous.namelist():
            if name.startswith('adapter/'):
                relative = Path(name).relative_to('adapter')
                if '..' in relative.parts or relative.suffix not in ('.json', '.txt', '.safetensors'):
                    raise ValueError('Unsafe prior adapter member')
                members['reference_adapter/'+relative.as_posix()] = previous.read(name)
        if 'reference_adapter/adapter_model.safetensors' not in members:
            raise ValueError('Prior adapter missing')
        # Preserve readable evidence; do not execute or load any downloaded weights locally.
        prior_dir = HERE/'prior_run'
        prior_dir.mkdir(exist_ok=True)
        for name in ('training.json', 'comparison.json', 'selection.json', 'RESULTS.md'):
            (prior_dir/name).write_bytes(previous.read(name))
    provenance = {'version': 2, 'dataset_audit': audit,
                  'previous_results_zip_sha256': digest(results),
                  'previous_test_role': 'Development/validation only; not a fresh test',
                  'baseline_scope': 'Original frozen baseline artifacts; not retrained on QLoRA-specific augmentation',
                  'comparison': 'QLoRA v1 and v2 use identical inference prompts, scoring and new test rows',
                  'review': 'Synthetic and not independently human-reviewed; no production claim'}
    manifest = {'schema_version': 1, 'provenance': provenance,
                'files': {name: hashlib.sha256(value).hexdigest() for name, value in members.items()}}
    members['manifest.json'] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    print(json.dumps({'bundle': str(destination), 'mib': destination.stat().st_size/1024**2,
                      'audit': audit}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path.home()/'Downloads/qlora_results.zip')
    parser.add_argument('--output', type=Path, default=HERE/'colab_retraining_v2.zip')
    args = parser.parse_args()
    build(args.results, args.output)
