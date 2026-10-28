"""Package existing local artifacts only. Never downloads or installs anything."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parent


def build(destination):
    members = {}
    for name in ('protocol.py', 'experiment.py', 'export_results.py', 'config.json', 'requirements-colab.txt', 'README.md'):
        members[name] = (HERE/name).read_bytes()
    for name in ('train.jsonl', 'validation.jsonl'):
        members['data/'+name] = (PLATFORM/'routing_training/data'/name).read_bytes()
    members['data/challenge.jsonl'] = (HERE/'challenge.jsonl').read_bytes()
    members['portable_tfidf.py'] = (PLATFORM/'domain_classifier.py').read_bytes()
    artifacts = PLATFORM/'routing_models/grouped-minilm-v1'
    for name in ('tfidf.json', 'frozen_head.json', 'setfit_head.json'):
        members['baselines/'+name] = (artifacts/name).read_bytes()
    allowed = {'.json', '.txt', '.md', '.safetensors'}
    for name in ('frozen_encoder', 'setfit_encoder'):
        for path in sorted((artifacts/name).rglob('*')):
            if path.is_file():
                if path.is_symlink() or path.suffix not in allowed:
                    raise ValueError('Unexpected encoder member: '+str(path))
                members['baselines/'+name+'/'+path.relative_to(artifacts/name).as_posix()] = path.read_bytes()
    provenance = {
        'train': 'Existing 1,620 synthetic rows / 540 base scenarios / 108 topic families',
        'validation': 'Existing 90 synthetic scenarios / 18 families; previously used for baseline selection',
        'test': 'New 120 project-authored synthetic scenarios / 60 contrastive pairs; 20 per domain',
        'review': 'No independent human review. Not evidence of real-world accuracy.',
        'original_90_test_rows': 'Excluded: already inspected in previous experiment',
        'baseline_run': 'grouped-minilm-v1; trained artifacts are bundled without retraining',
        'base_minilm_revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
    }
    manifest = {'schema_version': 1, 'provenance': provenance,
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}}
    members['manifest.json'] = json.dumps(manifest, indent=2).encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            if '..' in Path(name).parts or Path(name).is_absolute():
                raise ValueError('Unsafe archive path')
            archive.writestr(name, data)
    print(json.dumps({'bundle': str(destination), 'mib': destination.stat().st_size/1024**2,
                      'members': len(members), 'provenance': provenance}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'colab_bundle.zip')
    build(parser.parse_args().output)
