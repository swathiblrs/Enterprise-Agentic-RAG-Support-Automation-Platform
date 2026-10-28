"""Validate that native Llama labels share one prefix and one token per class."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE/'protocol.py').exists() else HERE.parent))
from protocol import LABELS, completion_tokens


def label_space(tokenizer):
    prefix = suffix = None
    token_ids = []
    for domain in LABELS:
        tokens, targets = completion_tokens(tokenizer, domain, domain_only=True)
        active = [i for i, value in enumerate(targets) if value != -100]
        if len(active) != 1:
            raise ValueError('Expected exactly one native token per domain; do not silently change scoring')
        index = active[0]
        before, after = tokens[:index], tokens[index+1:]
        if prefix is not None and (before != prefix or after != suffix):
            raise ValueError('Domain completions do not share the same surrounding tokens')
        prefix, suffix = before, after
        token_ids.append(tokens[index])
    if len(set(token_ids)) != len(LABELS):
        raise ValueError('Class tokens are not unique')
    return {'labels': LABELS, 'token_ids': token_ids, 'completion_prefix': prefix,
            'completion_suffix': suffix}
