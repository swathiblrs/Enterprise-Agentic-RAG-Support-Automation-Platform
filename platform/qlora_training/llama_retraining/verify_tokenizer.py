"""Inspect only the tokenizer already in Downloads; no weights or network access."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from label_space import label_space


def verify(archive_path):
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory(prefix='llama-label-check-') as temp:
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                path = Path(name)
                if path.parent.as_posix() == 'adapter' and path.suffix in ('.json', '.txt', '.jinja'):
                    (Path(temp)/path.name).write_bytes(archive.read(name))
        tokenizer = AutoTokenizer.from_pretrained(temp, local_files_only=True, trust_remote_code=False)
        result = label_space(tokenizer)
    print(json.dumps({'status': 'verified_existing_tokenizer_only', 'label_space': result,
                      'model_weights_loaded': False, 'network_downloads_enabled': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path.home()/'Downloads/llama_qlora_results.zip')
    verify(parser.parse_args().results)
