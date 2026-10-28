"""Export metrics and safetensors adapters, excluding pickled optimizer checkpoints."""
import argparse
from pathlib import Path
import zipfile


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if not (source/'comparison.json').is_file():
        raise ValueError('Comparison has not completed; do not export a partial result as final')
    members = [p for p in source.iterdir() if p.is_file() and p.suffix in ('.json', '.md', '.lock')]
    members += [p for p in (source/'adapter').rglob('*') if p.is_file()
                and p.suffix in ('.json', '.txt', '.jinja', '.safetensors')]
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(members):
            if path.is_symlink():
                raise ValueError('Unexpected export symlink')
            archive.write(path, path.relative_to(source).as_posix())
    return str(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(export(args.source, args.destination))
