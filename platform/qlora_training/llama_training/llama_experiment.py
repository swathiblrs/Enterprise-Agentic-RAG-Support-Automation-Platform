"""Public Llama checkpoint experiment; explicit anonymous downloads in Colab only."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE/'protocol.py').exists() else HERE.parent))
from protocol import (SYSTEM, audit_splits, calibrate, completion_tokens, digest,
                      metrics, read_rows, softmax, verify_bundle)

CONFIG = json.loads((HERE/'config.json').read_text())


def cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Use a hosted Colab CUDA GPU. No Mac/CPU or paid fallback is configured.')
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < CONFIG['minimum_vram_gib']*1024**3 or props.major < 6:
        raise RuntimeError('A CUDA GPU with at least 10 GiB VRAM is required. Stop if a free GPU is unavailable.')
    return {'gpu': props.name, 'vram_gib': props.total_memory/1024**3, 'torch': torch.__version__}


def encode_llama(tokenizer, text, domain, max_length=512, domain_only=False):
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
        tokenize=True, add_generation_prompt=True, date_string=CONFIG['chat_date'])
    completion, target = completion_tokens(tokenizer, domain, domain_only)
    ids = prompt + completion
    if len(ids) > max_length:
        raise ValueError('Input exceeds the fixed token limit; no silent truncation is allowed')
    return {'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': [-100]*len(prompt)+target}


def configure_engine():
    # Reuse the tested training/measurement engine without modifying the Qwen experiments.
    path = HERE/'engine.py'
    if not path.exists():
        path = HERE.parent/'experiment.py'
    spec = importlib.util.spec_from_file_location('llama_training_engine', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.ROOT, engine.CONFIG = HERE, CONFIG
    engine.cuda_check, engine.encode_example = cuda_check, encode_llama
    original_predictor = engine.make_predictor
    def anonymous_loader(adapter=None):
        cuda_check()  # Refuse unsupported hardware before model downloads.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['model_id'], revision=CONFIG['revision'], token=False, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG['model_id'], revision=CONFIG['revision'], token=False, trust_remote_code=False,
            use_safetensors=True, device_map={'': 0}, torch_dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        if not getattr(model, 'is_loaded_in_4bit', False) or model.config.model_type != 'llama':
            raise RuntimeError('Expected a 4-bit Llama model; no substitute model is permitted')
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        return model, tokenizer
    def predictor(name, out):
        aliases = {'base_llama': 'base_qwen', 'llama_qlora': 'qlora'}
        if name not in aliases:
            raise ValueError('Unknown Llama model candidate')
        return original_predictor(aliases[name], out)
    # The old loader name is a compatibility hook; model_id/revision select only Meta Llama.
    engine.load_qwen, engine.make_predictor = anonymous_loader, predictor
    return engine


def scored(rows, record, choice):
    if record['ids'] != [r['id'] for r in rows]:
        raise ValueError('Predictions do not match the benchmark row order')
    return metrics(rows, [softmax(s, choice['temperature']) for s in record['scores']],
                   choice['threshold'], choice['margin'])


def run(out):
    hardware = cuda_check()
    manifest = verify_bundle(HERE)
    if out.exists():
        raise RuntimeError('Use a fresh output folder; existing experiments are never overwritten')
    train, valid, benchmark = [read_rows(HERE/'data'/n) for n in
                               ('train.jsonl', 'validation.jsonl', 'challenge.jsonl')]
    audit = audit_splits(train, valid, benchmark)
    def child(*args):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *args, '--out', str(out)], check=True)
    child('train')
    names = ('base_llama', 'llama_qlora')
    references = json.loads((HERE/'references.json').read_text())
    selections = {}
    for name in names:
        child('worker', '--model', name, '--split', 'validation')
        record = json.loads((out/f'{name}_validation.json').read_text())
        choice = calibrate(valid, record['scores'])
        choice['metrics'] = scored(valid, record, choice)
        selections[name] = choice
    for name, reference in references['candidates'].items():
        record = reference['validation']
        choice = calibrate(valid, record['scores'])
        choice['metrics'] = scored(valid, record, choice)
        selections[name] = choice
    selection = {'candidates': selections, 'basis': 'development macro-F1; no automatic deployment',
                 'winner': max(selections, key=lambda n: selections[n]['metrics']['macro_f1']),
                 'active_router_changed': False,
                 'benchmark_scope': CONFIG['benchmark_scope'],
                 'note': 'The benchmark was previously inspected; this is not a new blind test.'}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    with (out/'challenge_opened.lock').open('x') as lock:
        lock.write(digest(out/'selection.json'))
    results = {}
    for name in names:
        child('worker', '--model', name, '--split', 'challenge')
        record = json.loads((out/f'{name}_challenge.json').read_text())
        result = scored(benchmark, record, selections[name])
        results[name] = {'metrics': result,
                         'engineering': {k: v for k, v in record.items() if k not in ('scores', 'ids')},
                         'execution': 'new_gpu_session'}
    for name, reference in references['candidates'].items():
        record = reference['challenge']
        results[name] = {'metrics': scored(benchmark, record, selections[name]),
                         'engineering': {k: v for k, v in record.items() if k not in ('scores', 'ids')},
                         'execution': 'cached_previous_session_not_rerun'}
    report = {'status': 'completed', 'model': CONFIG, 'hardware': hardware, 'dataset_audit': audit,
              'bundle_manifest_sha256': digest(HERE/'manifest.json'),
              'dataset_provenance': manifest['provenance'], 'selection': selection, 'test': results,
              'benchmark_scope': CONFIG['benchmark_scope'],
              'reference_archive_sha256': references['source_archive_sha256'],
              'packages': {n: importlib.metadata.version(n) for n in
                           ('torch', 'transformers', 'peft', 'bitsandbytes')},
              'limitations': ['The 120 synthetic benchmark cases were already inspected during Qwen development.',
                             'No independently reviewed real tickets or fresh blind test are included.',
                             'References reuse verified old predictions; their latency/memory were not rerun in this session.',
                             'Fixed TF-IDF/MiniLM/SetFit artifacts were trained before the QLoRA boundary augmentation.',
                             'CPU reference and GPU LLM timings are hardware-specific, not architecture-only speedups.',
                             'Six-candidate classification timing excludes queue, API, retrieval and answer generation.',
                             'Constrained likelihood classification does not establish free-generation JSON quality.',
                             'An adapter cannot be transferred between Qwen and Llama. No active router was changed.']}
    (out/'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# Llama 3.2 QLoRA: Historical Synthetic Benchmark', '',
             '| Model | Execution | Accuracy | Macro-F1 | p95 ms |', '|---|---|---:|---:|---:|']
    for name, result in results.items():
        m, e = result['metrics'], result['engineering']
        lines.append(f"| {name} | {result['execution']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {e['warm_p95_ms']:.2f} |")
    lines += ['', 'Per-domain recall, coverage, confusion matrices and memory: comparison.json.', '', *report['limitations']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'train', 'worker', 'run'])
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('--model', choices=['base_llama', 'llama_qlora'])
    parser.add_argument('--split', choices=['validation', 'challenge'], default='validation')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(cuda_check(), indent=2))
    elif args.command == 'run':
        run(args.out)
    elif args.command == 'train':
        configure_engine().train(args.out)
    else:
        if not args.model:
            parser.error('--model is required')
        configure_engine().worker(args.model, args.split, args.out)
