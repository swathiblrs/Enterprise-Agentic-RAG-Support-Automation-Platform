# Llama 3.2 Support Domain Classifier: No Token

**Status: the first Colab GPU run completed and was verified.** It achieved 91.67% accuracy and 91.70% macro-F1 on the known synthetic benchmark. [Verified results](../llama_retraining/previous_run/RESULTS.md). [The second Llama experiment](../llama_retraining/README.md) is prepared but has not trained yet. Uses the public `unsloth/Llama-3.2-1B-Instruct` community distribution of Llama 3.2, not Qwen and not the original Llama-3-8B. No Hugging Face account, token, Secrets configuration or Unsloth package is needed for this notebook.

Meta's own repository remains gated. This workflow instead uses an openly distributed checkpoint on Unsloth's public repository, subject to the Meta Llama 3.2 license. We verified public repository metadata and anonymous safetensors access without downloading model weights to the Mac. We do not claim byte-for-byte equivalence to Meta's repository.

## Run

1. Upload `Llama_QLoRA_NoToken.ipynb` at https://colab.research.google.com/.
2. Select a free hosted T4 GPU. Do not use a local runtime, paid compute units, or an inference API.
3. Run cells in order; upload **llama_no_token_bundle.zip** when prompted. Replace the old token-requiring notebook; do not run its access cell.
4. The anonymous-access cell checks the public checkpoint configuration using `token=False`. If the source becomes private/gated or inaccessible, stop; there is no authentication bypass or paid fallback.
5. Train and evaluate, then download `llama_qlora_results.zip`. No automatic deployment or hub upload occurs.

No `HF_TOKEN` is read, requested, or used. Downloads explicitly pass `token=False` even if an old token exists in the session. Model weights and GPU dependencies download only inside Google's disposable hosted runtime, not onto the Mac during preparation.

## Fixed Protocol

- Public model pinned at revision `5a8abab4a5d6f164389b1079fb721cfab8d7126c`; remote code disabled, safetensors weights, model-type check requires Llama.
- Standard Transformers + PEFT + bitsandbytes; no Unsloth installation or custom remote model code.
- NF4 double quantization; rank 16, alpha 32, dropout 0.10 on q/k/v/o projections.
- Domain-token-only loss, learning rate 0.0001, up to 300 steps, batch 1/accumulation 8, gradient checkpointing and validation-loss early stopping.
- Requires a CUDA GPU with at least 10 GiB VRAM; this guard does not guarantee the run cannot exhaust memory. FP16 on T4, BF16 where supported.
- Same 612 balanced training examples and 210 development cases as Qwen v2. Chat-template date is fixed. Confidence calibration and model choice use development predictions.
- Untuned and trained Llama run now. Qwen/TF-IDF/MiniLM/SetFit use verified cached predictions, explicitly marked as previous-session references; their timings are not remeasured in this session.

## Evaluation Limits

The 120 synthetic benchmark cases were previously inspected in Qwen development; they are **not a fresh blind test**. They are not independently human-reviewed real tickets. A separately sourced evaluation set is needed before generalization claims.

Report macro-F1, accuracy, per-domain recall, confusion matrices, accepted accuracy/coverage, sampled total process RSS, CUDA allocated/reserved memory, and batch-one warm p50/p95. Six-candidate classification timings exclude API/queue/retrieval/generation. Cached reference timings are historical and hardware-specific. Lightweight baselines did not receive QLoRA's training augmentation.

The live router and verified Qwen adapters remain unchanged. Only the first-run Llama results are established. No second-run result, arbiter, production endpoint, or quality improvement is claimed.

## License

Built with Llama. No token does not mean no license conditions. Comply with Meta's Llama 3.2 license and acceptable-use terms, including redistribution/naming/attribution requirements before publishing adapters. This workflow does not publish weights or accept terms on your behalf.

[Public checkpoint and license](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) | [Upstream model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization) | [Colab limits](https://research.google.com/colaboratory/faq.html)
