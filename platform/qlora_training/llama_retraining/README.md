# Llama Retraining: Direct Domain Classification

**Status: prepared, not trained.** The first verified Llama run achieved 91.67% accuracy and 91.70% macro-F1 on the known 120-case synthetic benchmark. This second run has no results yet. It does not change the live router or overwrite the original adapter.

## Why Change the Method

The first run used domain-token loss normalized over the entire vocabulary and selected checkpoints on validation loss. It stopped at 200 optimizer steps. Its inference scored six complete JSON answers with six forward passes.

The existing tokenizer was verified offline: each of our six domains has exactly one distinct native Llama token following the same JSON prefix. This permits an explicit six-class cross-entropy objective and a single forward pass at inference. The native model is still a causal Transformer trained using 4-bit QLoRA; the vocabulary logits are restricted to the six domain tokens, not a newly invented model family.

## Fixed Changes

- Same pinned public `unsloth/Llama-3.2-1B-Instruct` checkpoint, no token, no Unsloth package, no paid API.
- Same 612 training examples, 210 development examples and 120 previously inspected benchmark cases. No benchmark sentences or new hand-labeled fixes are added to training.
- Rank16/alpha32/dropout0.10 attention adapters, learning rate0.00005, label smoothing0.05, up to300 steps, effective batch8 with batch1/accumulation8.
- Select the best checkpoint using development **macro-F1**, with early stopping. Record complete training history and the selected checkpoint.
- Classification inputs stop **before** the domain token; the target label is not fed to the model. Padding or future JSON tokens cannot leak the answer into the prediction.
- Evaluate untuned Llama, the original adapter and the new adapter using the **same direct domain-token scorer in the same GPU session**. This separates scoring changes from retraining changes.

The prior 91.67% result used a different scorer. Do not describe any improvement over it as a training-only gain. Compare `previous_llama` with `retrained_llama` in the new report instead. Old Qwen and lightweight-model figures are retained separately as historical references, not reexecuted benchmarks.

## Run in Free Colab

1. Upload `Llama_Retraining_NoToken.ipynb` at https://colab.research.google.com/.
2. Select an available free T4 GPU. No local runtime, secrets, paid compute units or paid fallback.
3. Run cells in order; upload `llama_retraining_bundle.zip` when prompted. It contains the adapter already downloaded in your first result, not a newly downloaded base model.
4. Download `llama_retrained_results.zip` after training and evaluation finish.

No model weights or packages are downloaded to the Mac during preparation. The GPU run happens only after you start it in your own Colab session. There is no guarantee of improvement, GPU availability, or free-session completion.

## What the Report Means

Report macro-F1, accuracy, per-domain recall, confusion matrices, accepted accuracy/coverage, isolated-process RSS, CUDA peak allocated/reserved memory, and synchronized batch-one warm p50/p95 latency. Timings are classification-only, not full RAG/API/queue latency. A reduced number of forward passes is an implementation change, not a measured speedup until the run completes.

The 120 cases are an inspected synthetic development benchmark, **not a new blind test**. Human-reviewed or externally sourced evaluation is still needed. Thresholds selected for at least90% development accepted accuracy do not establish production-safe calibration. Model outputs are constrained domain choices, not evidence of free-form reasoning or JSON generation quality.

Built with Llama. Meta's license and redistribution/naming/attribution requirements still apply to the [public checkpoint](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). No adapters are published by this workflow.
