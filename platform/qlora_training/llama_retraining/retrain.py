"""Bounded no-token Llama experiment with a direct six-token classification head."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE/'protocol.py').exists() else HERE.parent))
from protocol import LABELS, audit_splits, calibrate, digest, metrics, read_rows, softmax, verify_bundle
from label_space import label_space

CONFIG = json.loads((HERE/'config.json').read_text())


def support_engine():
    path = HERE/'llama_support.py'
    if not path.exists():
        path = HERE.parent/'llama_training/llama_experiment.py'
    spec = importlib.util.spec_from_file_location('llama_retraining_support', path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    support.HERE, support.CONFIG = HERE, CONFIG
    return support, support.configure_engine()


def prefix_features(support, tokenizer, spec, text, domain=None):
    # Always build with the same placeholder, never the answer's domain.
    record = support.encode_llama(tokenizer, text, LABELS[0], CONFIG['max_length'], domain_only=True)
    positions = [i for i, token in enumerate(record['labels']) if token != -100]
    if len(positions) != 1 or record['input_ids'][positions[0]] != spec['token_ids'][0]:
        raise ValueError('Unexpected label-token layout')
    end = positions[0]
    features = {'input_ids': record['input_ids'][:end], 'attention_mask': record['attention_mask'][:end]}
    if domain is not None:
        features['labels'] = LABELS.index(domain)
    return features


def restricted_logits(model, input_ids, attention_mask, token_ids):
    if input_ids.shape[0] != 1:
        raise ValueError('The memory-bounded prefix classifier requires batch size one')
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
    return output.logits[:, -1, token_ids].float()


def make_trainer_class(parent, token_ids, smoothing):
    import torch.nn.functional as functional
    class DomainTrainer(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logits = restricted_logits(model, inputs['input_ids'], inputs['attention_mask'], token_ids)
            loss = functional.cross_entropy(logits, inputs['labels'], label_smoothing=smoothing)
            return (loss, {'logits': logits}) if return_outputs else loss

    return DomainTrainer


def classification_metrics(prediction):
    rows = [{'id': str(i), 'domain': LABELS[int(label)]} for i, label in enumerate(prediction.label_ids)]
    score = metrics(rows, [softmax(row.tolist()) for row in prediction.predictions])
    return {'macro_f1': score['macro_f1'], 'accuracy': score['accuracy'],
            'security_recall': score['per_domain_recall']['security']}


def train(out):
    support, engine = support_engine()
    hardware = support.cuda_check()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    if CONFIG['batch_size'] != 1:
        raise ValueError('Batch size is fixed at one; use gradient accumulation for effective batch eight')
    set_seed(CONFIG['seed'])
    out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = engine.load_qwen()
    spec = label_space(tokenizer)
    train_rows, valid_rows = [read_rows(HERE/'data'/name) for name in ('train.jsonl', 'validation.jsonl')]
    datasets = [engine.Rows([prefix_features(support, tokenizer, spec, r['text'], r['domain']) for r in rows])
                for rows in (train_rows, valid_rows)]
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'], lora_dropout=CONFIG['lora_dropout'],
        target_modules=CONFIG['target_modules'], bias='none'))
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise RuntimeError('Only adapter weights may train')
    before = {name: value.detach().cpu().clone() for name, value in trainable}
    def collate(examples):
        if len(examples) != 1:
            raise ValueError('Batch-one collator received multiple examples')
        item = examples[0]
        return {key: torch.tensor([value], dtype=torch.long) for key, value in item.items()}
    DomainTrainer = make_trainer_class(Trainer, spec['token_ids'], CONFIG['label_smoothing'])
    arguments = TrainingArguments(output_dir=str(out/'checkpoints'), max_steps=CONFIG['max_steps'],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'], learning_rate=CONFIG['learning_rate'],
        weight_decay=.01, warmup_ratio=.1, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), eval_strategy='steps', save_strategy='steps',
        eval_steps=50, save_steps=50, save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='macro_f1', greater_is_better=True, label_names=['labels'],
        remove_unused_columns=False, report_to=[], push_to_hub=False, optim='adamw_torch',
        logging_steps=10, seed=CONFIG['seed'], data_seed=CONFIG['seed'])
    trainer = DomainTrainer(model=model, args=arguments, train_dataset=datasets[0], eval_dataset=datasets[1],
        data_collator=collate, compute_metrics=classification_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    changed = [name for name, value in model.named_parameters()
               if name in before and not torch.equal(before[name], value.detach().cpu())]
    if not changed:
        raise RuntimeError('No adapter weights changed')
    model.save_pretrained(out/'adapter', safe_serialization=True)
    tokenizer.save_pretrained(out/'adapter')
    report = {'status': 'trained', 'hardware': hardware, 'config': CONFIG, 'label_space': spec,
              'optimizer_steps': trainer.state.global_step, 'seconds': time.perf_counter()-start,
              'trainable_parameters': sum(value.numel() for _, value in trainable),
              'changed_adapter_tensors': len(changed), 'best_validation_macro_f1': trainer.state.best_metric,
              'best_checkpoint': Path(trainer.state.best_model_checkpoint).name,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
              'train_sha256': digest(HERE/'data/train.jsonl'),
              'validation_sha256': digest(HERE/'data/validation.jsonl')}
    (out/'training.json').write_text(json.dumps(report, indent=2))
    (out/'training_history.json').write_text(json.dumps(trainer.state.log_history, indent=2))


def predictor(name, out, support, engine):
    import torch
    adapters = {'base_llama': None, 'previous_llama': HERE/'previous_adapter', 'retrained_llama': out/'adapter'}
    if name not in adapters:
        raise ValueError('Unknown Llama candidate')
    model, tokenizer = engine.load_qwen(adapters[name])
    spec = label_space(tokenizer)
    model.eval()
    def predict(text):
        features = prefix_features(support, tokenizer, spec, text)
        with torch.inference_mode():
            values = restricted_logits(model,
                torch.tensor([features['input_ids']], device=model.device),
                torch.tensor([features['attention_mask']], device=model.device), spec['token_ids'])
        return values[0].tolist()
    return predict, 'cuda'


def worker(name, split, out):
    support, engine = support_engine()
    engine.make_predictor = lambda candidate, folder: predictor(candidate, folder, support, engine)
    engine.worker(name, split, out)


def run(out):
    support, _ = support_engine()
    hardware = support.cuda_check()
    manifest = verify_bundle(HERE)
    if out.exists():
        raise RuntimeError('Use a fresh output folder; never overwrite a completed run')
    train_rows, valid, benchmark = [read_rows(HERE/'data'/name) for name in
                                   ('train.jsonl', 'validation.jsonl', 'challenge.jsonl')]
    audit = audit_splits(train_rows, valid, benchmark)
    def child(*args):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *args, '--out', str(out)], check=True)
    child('train')
    names = ('base_llama', 'previous_llama', 'retrained_llama')
    selections = {}
    for name in names:
        child('worker', '--model', name, '--split', 'validation')
        record = json.loads((out/f'{name}_validation.json').read_text())
        choice = calibrate(valid, record['scores'])
        choice['metrics'] = support.scored(valid, record, choice)
        selections[name] = choice
    selection = {'candidates': selections, 'basis': 'development macro-F1; benchmark not used to select checkpoint',
                 'winner': max(names, key=lambda name: selections[name]['metrics']['macro_f1']),
                 'active_router_changed': False, 'benchmark_scope': CONFIG['benchmark_scope']}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    with (out/'challenge_opened.lock').open('x') as lock:
        lock.write(digest(out/'selection.json'))
    results = {}
    for name in names:
        child('worker', '--model', name, '--split', 'challenge')
        record = json.loads((out/f'{name}_challenge.json').read_text())
        results[name] = {'metrics': support.scored(benchmark, record, selections[name]),
                         'engineering': {key: value for key, value in record.items() if key not in ('scores', 'ids')},
                         'execution': 'new_session_same_single_pass_scorer'}
    references = json.loads((HERE/'references.json').read_text())
    report = {'status': 'completed', 'hardware': hardware, 'model': CONFIG, 'dataset_audit': audit,
              'bundle_manifest_sha256': digest(HERE/'manifest.json'), 'dataset_provenance': manifest['provenance'],
              'selection': selection, 'test': results, 'historical_references': references,
              'limitations': ['Previously inspected synthetic development benchmark, not an independent blind test.',
                             'Both old and new Llama adapters are re-evaluated under the same direct token scorer.',
                             'Old six-pass JSON-likelihood accuracy/latency is a different method, not a training-only baseline.',
                             'Qwen and lightweight reference metrics are historical; their methods/hardware differ.',
                             'No production generalization, end-to-end RAG latency, or automatic promotion is claimed.']}
    (out/'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# Llama Retraining: Same-Scorer Comparison', '',
             '| Model | Accuracy | Macro-F1 | Security recall | p95 ms |', '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        m, e = result['metrics'], result['engineering']
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['per_domain_recall']['security']:.2%} | {e['warm_p95_ms']:.2f} |")
    lines += ['', *report['limitations']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'train', 'worker', 'run'])
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('--model', choices=['base_llama', 'previous_llama', 'retrained_llama'])
    parser.add_argument('--split', choices=['validation', 'challenge'], default='validation')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(support_engine()[0].cuda_check(), indent=2))
    elif args.command == 'train':
        train(args.out)
    elif args.command == 'worker':
        if not args.model:
            parser.error('--model is required')
        worker(args.model, args.split, args.out)
    else:
        run(args.out)
