# QLoRA Routing Experiment

**Status: the first Colab training/evaluation run completed.** QLoRA achieved 94.17% accuracy and 94.18% macro-F1 on its 120 synthetic test cases. [Verified results](retraining_v2/prior_run/RESULTS.md). The active router is unchanged. [The second targeted experiment](retraining_v2/README.md) also completed, achieving 95.83% accuracy and 95.82% macro-F1 on a different 120-case benchmark; compare adapters on common rows, not percentages across different sets. [The Llama experiment](llama_training/README.md) is prepared but has not trained yet.

Local verification passed all 13 new protocol/export checks (85 total platform tests). The three baseline models reproduced their earlier predictions on the 90 validation rows with downloads disabled. [Local verification](local_verification.json) is **not** a QLoRA test result.

No LLM arbiter is implemented. Qwen is a domain-classifier challenger, not a judge.
The active application router is unchanged.

## Mac Safety and Cost

- Local preparation uses Python's standard library and existing project files.
- No model weights, GPU drivers, CUDA, bitsandbytes, or packages are downloaded to the Mac.
- The notebook installs dependencies and downloads the public Qwen model only inside Google's disposable hosted GPU runtime.
- No paid inference API, subscription, credit-card setup, automatic paid fallback, or cloud deployment is used.
- Choose a free hosted GPU runtime, not a local runtime, Colab Enterprise, or paid compute units. If unavailable, stop and retry another time.
- Upload only the supplied bundle: synthetic examples, baseline models and experiment code. No private tickets, credentials, production database or company documents.
- Results download is an explicit final notebook action; adapters use safetensors, not pickle.

Free GPU availability and session lifetime are not guaranteed: [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Run

1. Open `QLoRA_Comparison.ipynb` using File > Upload notebook at https://colab.research.google.com/.
2. Select Runtime > Change runtime type > a free GPU, such as T4.
3. Run the cells in order, uploading `colab_bundle.zip` when prompted.
4. The hardware check runs before package installation and again before model downloads. It refuses a non-CUDA runtime.
5. Download `qlora_results.zip` after the run. `comparison.json`, `training.json` and `RESULTS.md` contain actual measurements.

No notebook execution can take place until you open your Colab account and start its GPU runtime.
Local packaging command, using an existing Python installation:

```sh
python3 platform/qlora_training/build_bundle.py
```

## Fixed Experiment

| Component | Protocol |
|---|---|
| Qwen base | Qwen3-0.6B, public Apache-2.0, pinned revision in config.json |
| Quantization | bitsandbytes 4-bit NF4, double quantization; BF16 when supported, otherwise FP16 |
| Adapter | Rank 16, alpha 32, dropout 0.05; q/k/v/o attention projections |
| Objective | Completion-only causal cross-entropy for one JSON domain label; user and system tokens masked |
| Training | Up to 200 steps, AdamW, batch 2, accumulation 4; validation loss every 50 steps, early stopping |
| Comparators | Existing char-TF-IDF logistic regression, frozen MiniLM + LR, SetFit MiniLM + LR, and untuned 4-bit Qwen |
| Output | cloud, data, it, other, production, security |
| Dataset | 1,620 train rows, 90 validation scenarios, 120 new held-out synthetic challenge cases |

Related paraphrases remain in the original grouped splits. The new challenge has 60 contrastive pairs, 20 examples per class, with lexical-duplicate screening against train/validation.
The old 90-row test is excluded because its results were already inspected.
This is a new held-out synthetic test, **not independently annotated real-world data**.
Labels and examples need human review before stronger claims are made.

All five models use identical validation and test rows. Adapter checkpoint choice uses validation loss;
probability temperature and acceptance thresholds use validation only. The selection file is hashed and locked
before scoring the challenge. New run folders cannot overwrite prior results. Testing does not promote any model.
The original validation set has already influenced baseline development: a limitation, not a fresh blind validation set.

## Measurements and Interpretation

- **Macro-F1:** calculate F1 separately for all six classes, then average equally, avoiding majority-class dominance.
- **Per-domain recall:** correctly classified requests / actual requests in that domain, revealing missed security or IT cases.
- **Accuracy and confusion matrix:** total correct fraction and which classes are confused.
- **Accepted accuracy / coverage:** correctness and fraction remaining after validation-selected confidence and margin gates.
- **Auto-route coverage:** accepted fraction excluding `other`; accepting an out-of-scope classification is not routing it.
- **Latency:** batch-one warm p50/p95; 3 warm-ups, every validation/test request timed; CUDA synchronized before/after each call.
- **Memory:** sampled total process RSS every 10 ms in a separate process per model; Qwen additionally reports PyTorch CUDA peak allocated/reserved memory. These are different measures and are not summed.
- **Training evidence:** optimizer steps, changed adapter tensors, trainable parameter count, validation loss, training time, peak CUDA memory, and input file hashes.

Qwen classification scores all six canonical JSON completions using mean completion-token log-likelihood.
This is constrained classification, **not** proof of free-generation JSON validity, open-ended reasoning, or calibrated confidence.
Baselines run on CPU; Qwen runs on GPU. Timings describe that deployment configuration, not a same-device architecture speedup.
Model timings exclude API/queue/retrieval/RAG generation and cannot replace earlier end-to-end benchmarks.
The challenge is small and contrastive: do not interpret a few points as statistical significance.

Training may lose to a smaller model. Publish that outcome honestly; QLoRA is not automatically faster or more accurate.
The first Colab artifacts now exist. Any resume claim must name the synthetic evaluation scope; the second retraining run has verified results, while Llama training remains pending.

Implementation references: [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B),
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).
