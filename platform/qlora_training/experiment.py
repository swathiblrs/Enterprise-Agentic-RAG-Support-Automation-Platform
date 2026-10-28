"""Colab CUDA-only training and isolated-process model comparison. No API calls."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from protocol import (LABELS, audit_splits, calibrate, digest, encode_example, metrics,
                      pad_examples, read_rows, softmax, verify_bundle)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text())
os.environ.update(TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true', HF_HUB_DISABLE_TELEMETRY='1')


def cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. Stop here; select a free Colab GPU runtime. No CPU/Mac fallback.')
    props = torch.cuda.get_device_properties(0)
    if props.major < 6 or props.total_memory < 6 * 1024**3:
        raise RuntimeError('This experiment requires CUDA compute capability >=6 and at least 6 GiB VRAM.')
    return {'gpu': props.name, 'vram_gib': props.total_memory/1024**3, 'torch': torch.__version__}


def load_qwen(adapter=None):
    cuda_check()  # Must precede model/tokenizer downloads.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'], revision=CONFIG['revision'], trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_id'], revision=CONFIG['revision'], trust_remote_code=False, use_safetensors=True,
        device_map={'': 0}, torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    if not getattr(model, 'is_loaded_in_4bit', False):
        raise RuntimeError('Base model was not loaded in 4-bit')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, tokenizer


class Rows:
    def __init__(self, values):
        self.values = values
    def __len__(self):
        return len(self.values)
    def __getitem__(self, index):
        return self.values[index]


def train(out):
    hardware = cuda_check()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    set_seed(CONFIG['seed'])
    out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_qwen()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type='CAUSAL_LM', r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'], target_modules=CONFIG['target_modules'], bias='none'))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Unexpected trainable base parameters')
    before = {n: p.detach().cpu().clone() for n, p in trainable}
    datasets = [Rows([encode_example(tokenizer, r['text'], r['domain'], CONFIG['max_length'],
                                     domain_only=CONFIG.get('domain_only_loss', False))
                     for r in read_rows(ROOT / 'data' / name)])
                for name in ('train.jsonl', 'validation.jsonl')]
    def collate(examples):
        return {k: torch.tensor(v, dtype=torch.long) for k, v in pad_examples(examples, tokenizer.pad_token_id).items()}
    arguments = TrainingArguments(
        output_dir=str(out / 'checkpoints'), max_steps=CONFIG['max_steps'],
        per_device_train_batch_size=CONFIG['batch_size'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        learning_rate=CONFIG['learning_rate'], weight_decay=.01, warmup_ratio=.1,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        eval_strategy='steps', save_strategy='steps', eval_steps=50, save_steps=50,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, prediction_loss_only=True, report_to=[], push_to_hub=False,
        optim='adamw_torch', logging_steps=10, seed=CONFIG['seed'], data_seed=CONFIG['seed'])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = Trainer(model=model, args=arguments, train_dataset=datasets[0], eval_dataset=datasets[1],
                      data_collator=collate, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    changed = [n for n, p in model.named_parameters() if n in before and not torch.equal(before[n], p.detach().cpu())]
    if not changed:
        raise RuntimeError('No adapter weights changed; training is not verified')
    adapter = out / 'adapter'
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    report = {'status': 'trained', 'hardware': hardware, 'config': CONFIG,
              'optimizer_steps': trainer.state.global_step, 'seconds': time.perf_counter()-start,
              'trainable_parameters': sum(p.numel() for _, p in trainable),
              'changed_adapter_tensors': len(changed), 'best_validation_loss': trainer.state.best_metric,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
              'train_sha256': digest(ROOT/'data/train.jsonl'),
              'validation_sha256': digest(ROOT/'data/validation.jsonl')}
    (out/'training.json').write_text(json.dumps(report, indent=2))


def make_predictor(name, out):
    import numpy as np
    if name == 'tfidf':
        from portable_tfidf import probabilities
        head = json.loads((ROOT/'baselines/tfidf.json').read_text())
        def predict(text):
            values = probabilities(text, head) or [1/6]*6
            return [math.log(max(1e-15, values[head['classes'].index(label)])) for label in LABELS]
        return predict, 'cpu'
    if name in ('frozen', 'setfit'):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        encoder = SentenceTransformer(str(ROOT/'baselines'/f'{name}_encoder'), device='cpu', local_files_only=True)
        encoder.max_seq_length = 64
        head = json.loads((ROOT/'baselines'/f'{name}_head.json').read_text())
        weights, bias = np.array(head['coefficients']), np.array(head['intercept'])
        def predict(text):
            features = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = weights @ features + bias
            return [float(scores[head['classes'].index(label)]) for label in LABELS]
        return predict, 'cpu'
    import torch
    adapter = out/'adapter' if name == 'qlora' else ROOT/'reference_adapter' if name == 'qlora_v1' else None
    model, tokenizer = load_qwen(adapter)
    model.eval()
    def predict(text):
        scores = []
        # One candidate at a time bounds vocabulary-logit memory on a free T4.
        with torch.inference_mode():
            for label in LABELS:
                example = encode_example(tokenizer, text, label, CONFIG['max_length'])
                ids = torch.tensor([example['input_ids']], device=model.device)
                target = torch.tensor(example['labels'], device=model.device)
                length = int((target != -100).sum().item())
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False, logits_to_keep=length+1).logits[0, :-1].float()
                completion = target[-length:]
                score = torch.log_softmax(logits, dim=-1).gather(1, completion[:, None]).mean()
                scores.append(float(score.item()))
        return scores
    return predict, 'cuda'


def worker(name, split, out):
    if split == 'challenge':
        lock = out/'challenge_opened.lock'
        if not lock.is_file() or lock.read_text() != digest(out/'selection.json'):
            raise RuntimeError('Challenge requires an unchanged, frozen validation selection')
        if (out/f'{name}_challenge.json').exists():
            raise RuntimeError('This model has already scored the challenge in this run')
    import psutil
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        start = time.perf_counter()
        predict, device = make_predictor(name, out)
        load_seconds = time.perf_counter()-start
        rows = read_rows(ROOT/'data'/('validation.jsonl' if split == 'validation' else 'challenge.jsonl'))
        for row in rows[:3]:
            predict(row['text'])
        if device == 'cuda':
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        scores, elapsed = [], []
        for row in rows:
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            scores.append(predict(row['text']))
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
        elapsed.sort()
        result = {'ids': [r['id'] for r in rows], 'scores': scores, 'device': device,
                  'load_seconds': load_seconds, 'warm_p50_ms': elapsed[math.ceil(.5*len(elapsed))-1],
                  'warm_p95_ms': elapsed[math.ceil(.95*len(elapsed))-1],
                  'latency_samples': len(rows), 'batch_size': 1,
                  'peak_process_rss_mib': max(peak[0], process.memory_info().rss)/1024**2,
                  'memory_method': '10ms RSS sampling, isolated process including imports/model; not incremental model memory',
                  'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2 if device == 'cuda' else None,
                  'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/1024**2 if device == 'cuda' else None}
        (out/f'{name}_{split}.json').write_text(json.dumps(result, indent=2))
    finally:
        stop.set()
        thread.join()


def run(out):
    hardware = cuda_check()
    manifest = verify_bundle(ROOT)
    audit = audit_splits(*(read_rows(ROOT/'data'/name) for name in
                           ('train.jsonl', 'validation.jsonl', 'challenge.jsonl')))
    def child(*args):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *args, '--out', str(out)], check=True)
    # A fresh run directory prevents overwriting results or re-opening the same test run.
    child('train')
    names = tuple(CONFIG.get('comparison_models', ['tfidf', 'frozen', 'setfit', 'base_qwen', 'qlora']))
    valid = read_rows(ROOT/'data/validation.jsonl')
    selections = {}
    for name in names:
        child('worker', '--model', name, '--split', 'validation')
        record = json.loads((out/f'{name}_validation.json').read_text())
        assert record['ids'] == [r['id'] for r in valid]
        choice = calibrate(valid, record['scores'])
        probs = [softmax(s, choice['temperature']) for s in record['scores']]
        choice['metrics'] = metrics(valid, probs, choice['threshold'], choice['margin'])
        selections[name] = choice
    selection = {'candidates': selections, 'basis': 'validation macro-F1; ties favor current frozen model',
                 'winner': max(names, key=lambda n: (selections[n]['metrics']['macro_f1'], n == 'frozen')),
                 'test_used_for_selection': False, 'active_router_changed': False}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    with (out/'challenge_opened.lock').open('x') as lock:
        lock.write(digest(out/'selection.json'))
    test = read_rows(ROOT/'data/challenge.jsonl')
    results = {}
    for name in names:
        child('worker', '--model', name, '--split', 'challenge')
        record = json.loads((out/f'{name}_challenge.json').read_text())
        assert record['ids'] == [r['id'] for r in test]
        choice = selections[name]
        probs = [softmax(s, choice['temperature']) for s in record.pop('scores')]
        results[name] = {'metrics': metrics(test, probs, choice['threshold'], choice['margin']),
                         'engineering': record}
    report = {'status': 'completed', 'hardware': hardware, 'dataset_audit': audit,
              'bundle_manifest_sha256': digest(ROOT/'manifest.json'), 'dataset_provenance': manifest['provenance'],
              'selection': selection, 'test': results,
              'packages': {n: importlib.metadata.version(n) for n in
                           ('torch', 'transformers', 'peft', 'bitsandbytes', 'sentence-transformers', 'numpy')},
              'limitations': ['Synthetic held-out challenge; not independent human-reviewed real tickets.',
                             'Original validation set has been used in previous experiments.',
                             'CPU baselines versus CUDA Qwen timings are hardware-specific, not architecture-only speedups.',
                             'Qwen uses six length-normalized candidate likelihoods, not free-generation JSON validity.',
                             'RSS is sampled total process memory; GPU allocated/reserved are separate, not additive.',
                             'Model latency excludes API, queue, retrieval and RAG generation.',
                             'No automatic deployment or claim that fine-tuning improves the model.'] + CONFIG.get('limitations_extra', [])}
    (out/'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# QLoRA Routing Comparison', '', '| Model | Macro-F1 | Accuracy | p95 ms | RSS MiB | GPU allocated MiB |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        m, e = result['metrics'], result['engineering']
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['accuracy']:.2%} | {e['warm_p95_ms']:.2f} | {e['peak_process_rss_mib']:.1f} | {e['peak_gpu_allocated_mib']} |")
    lines += ['', '## Per-Domain Recall', '', '| Model | ' + ' | '.join(LABELS) + ' |',
              '|---|' + '---:|' * len(LABELS)]
    for name, result in results.items():
        recalls = result['metrics']['per_domain_recall']
        lines.append('| ' + name + ' | ' + ' | '.join(f'{recalls[label]:.2%}' for label in LABELS) + ' |')
    lines += ['', 'Per-case predictions and coverage: comparison.json.', '', *report['limitations']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'train', 'worker', 'run'])
    parser.add_argument('--out', type=Path, default=ROOT/'results')
    parser.add_argument('--model', choices=['tfidf', 'frozen', 'setfit', 'base_qwen', 'qlora', 'qlora_v1'])
    parser.add_argument('--split', choices=['validation', 'challenge'], default='validation')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(cuda_check(), indent=2))
    elif args.command == 'train':
        train(args.out)
    elif args.command == 'worker':
        if not args.model:
            parser.error('--model is required')
        worker(args.model, args.split, args.out)
    else:
        run(args.out)
