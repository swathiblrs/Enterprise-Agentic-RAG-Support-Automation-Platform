"""Check token masks using the already-downloaded tokenizer, never model weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from protocol import completion_tokens, read_rows


def verify(results):
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory(prefix='qlora-tokenizer-check-') as folder:
        with zipfile.ZipFile(results) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                # Extract only direct tokenizer text/JSON files. Never extract/load adapter weights.
                if path.parent.as_posix() == 'adapter' and path.suffix in ('.json', '.txt'):
                    (Path(folder)/path.name).write_bytes(archive.read(member))
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        counts = {}
        maximum = 0
        for name in ('train.jsonl', 'validation.jsonl', 'challenge.jsonl'):
            rows = read_rows(HERE/'data'/name)
            for row in rows:
                completion, targets = completion_tokens(tokenizer, row['domain'], domain_only=True)
                active = [v for v in targets if v != -100]
                if row['domain'] not in tokenizer.decode(active):
                    raise ValueError('Domain tokens do not decode to their label')
                maximum = max(maximum, len(completion))
            counts[name] = len(rows)
        report = {'status': 'verified_existing_tokenizer_only', 'rows_checked': counts,
                  'maximum_completion_tokens': maximum, 'full_prompt_length_checked': False,
                  'note': 'The v1 export omitted chat_template.jinja; full prompts are checked during Colab tokenization.',
                  'model_weights_loaded': False,
                  'network_downloads_enabled': False, 'gpu_training_executed': False}
    (HERE/'tokenization_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path.home()/'Downloads/qlora_results.zip')
    verify(parser.parse_args().results)
