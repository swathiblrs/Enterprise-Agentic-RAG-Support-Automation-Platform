# Routing Model Training and Comparison

This experiment compares a character TF-IDF classifier, a frozen MiniLM encoder with a trained classifier, and a SetFit-fine-tuned MiniLM encoder with a trained classifier. The scope is six-way domain routing: cloud, data, IT, other, production, and security. The separate category/priority model and retrieval embeddings are unchanged.

## Data and Review Status

The generated dataset has 1,800 rows from 720 base scenarios and 144 topic families. All data is project-authored synthetic material, not real support records. It has been structurally audited, but it is not independently human-reviewed.

| Split | Rows | Base scenarios | Topic families |
| --- | --- | --- | --- |
| Train | 1,620 | 540 | 108 |
| Validation | 90 | 90 | 18 |
| Test | 90 | 90 | 18 |

Training includes two formatting variants per base scenario. These variants do not add independent incidents. Validation and test use only unaugmented text. Topic families, scenarios, and all their variants stay in one split; per-class group allocation is deterministic.

The audit checks IDs, labels, normalized duplicate text, family/scenario overlap, and cross-split character-shingle similarity. It found no exact duplicates or near matches at the 0.85 Jaccard threshold; maximum observed train/held-out similarity was 0.431. This does not rule out semantic leakage, shortcuts from the writing style, or ambiguous labels.

Human review remains pending. data/review_queue.jsonl contains all 720 base scenarios with empty reviewer, reviewed_label, and notes fields. The original 120 routing examples and 80 category/priority examples are not silently relabeled or mixed into this experiment.

## Training Protocol

All three candidates use the same training and validation rows. Logistic Regression C is chosen from 0.1, 1, and 10 using validation macro-F1 and accepted coverage. Acceptance score/margin thresholds are selected on validation only, seeking maximum coverage with at least 90% accepted accuracy and at least six accepted examples.

SetFit trains the MiniLM body with contrastive pairs from training labels: 80 optimizer steps, batch size 16, two pair-sampling iterations, and seed 42. The encoder weight hashes differ before and after training, confirming that fine-tuning updated parameters. An exported JSON classifier head is fitted on its resulting embeddings.

The first hardware probe inside the sandbox reported CPU-only capability. The actual training process outside the sandbox auto-selected Apple MPS. The initial runner hard-coded CPU in report metadata; the record was corrected using the training warning and a same-environment device check. Saved-model latency was subsequently measured on CPU for all three candidates. Future runs explicitly move the model to CPU for comparison after recording its training device.

The winner is frozen before test evaluation, using validation macro-F1, then accepted coverage, then warm latency. selection.json is hashed into test_opened.lock before the test file is opened. There is no test-score-based model selection or threshold adjustment.

## Results

| Candidate | Validation macro-F1 | Test accuracy | Test macro-F1 | Test accepted coverage | Accepted test accuracy | CPU warm p95 |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 97.77% | 96.67% | 96.67% | 90.00% | 100.00% (81/81) | 0.16 ms |
| Frozen MiniLM + Logistic Regression | 100.00% | 96.67% | 96.57% | 96.67% | 97.70% (85/87) | 8.36 ms |
| SetFit MiniLM + Logistic Regression | 98.89% | 95.56% | 95.48% | 100.00% | 95.56% (86/90) | 7.49 ms |

Frozen MiniLM was selected from validation. TF-IDF has a slightly higher test macro-F1; that did not change the preselected winner. Fine-tuning did not improve this run, and no improvement over the baseline is claimed for SetFit.

The selected model's test recall is 100% for cloud, data, other, production, and security, and 80% for IT (12/15). This IT weakness needs review. Per-domain recall/F1, confusion matrices, acceptance decisions, and error cases are in reports/grouped-minilm-v1/comparison.json.

Latency uses 50 validation texts, batch size one, after warm-up, and excludes IPC and request processing. The selected isolated runtime reproduced all 90 validation predictions and acceptance decisions; cold startup was approximately 3.98 seconds and mean warm inference including IPC was 5.99 ms in that separate check.

These are synthetic benchmark results with only 15 final test examples per class. The test set has now been inspected; future tuning requires a new untouched final holdout. The prior 24-case routing benchmark is a different dataset and should not be used for a direct improvement claim.

## Routing Integration

domain_router.py retains explicit domain selection and higher-priority service/security policies. Only requests without policy matches reach the selected ML fallback. Other-class or low-confidence predictions request clarification.

The active_model.json manifest points to the validation-selected artifact. The API runs its encoder in an isolated persistent subprocess using the existing .venv-support environment for the selected frozen encoder, keeping ML libraries out of the API environment. The small JSON head is packaged with the reports and the pretrained encoder revision is pinned. Fine-tuned encoder artifacts remain separate under routing_models/. The runtime uses local encoder files and JSON head parameters, not an untrusted pickle.

If no usable artifact manifest is available, the original portable TF-IDF model remains available with an explicit legacy_fallback marker. If a selected model's worker fails during inference, the request abstains rather than guessing. Restart the API and workers after changing the active manifest.

Application regression checks also exposed missing service knowledge: explicit VPN wording now recognizes 'virtual private network', HTTP server errors have an explicit production signal, and vague requests without a service or evidence ask for clarification. These policy fixes do not change classifier weights, thresholds, or the reported classifier test scores.

The retrieval model remains the original all-MiniLM-L6-v2 used by ChromaDB. The classifier's saved encoder is a separate copy. No knowledge-base reindexing is part of this experiment.

## Reproduce

Use a separate training environment with the versions in requirements.txt and the resolved package versions recorded in comparison.json. The pretrained MiniLM model must be downloaded into the Hugging Face cache before the offline training command.

```sh
python3 -m venv .venv-training
.venv-training/bin/python -m pip install -r routing_training/requirements.txt
.venv-training/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")'
.venv-training/bin/python routing_training/build_dataset.py
.venv-training/bin/python routing_training/train_compare.py --run-id YOUR_NEW_RUN_ID --steps 80
.venv-training/bin/python routing_training/standardize_latency.py YOUR_NEW_RUN_ID
.venv-api/bin/python routing_training/verify_deployment.py YOUR_NEW_RUN_ID
```

The run ID must be new; the runner refuses to overwrite an experiment. Compare selection.json first and perform independent review before activating a later model. Large model files under routing_models/ are ignored by Git; reports, training code, and the selected JSON head are retained. The current frozen winner can run from the pinned pretrained model cache without the training environment. Copy a reviewed run's deployment.json to routing_training/active_model.json and restart the service to activate it.

## Next Data Work

Independently review the base labels, especially IT/cloud and IT/production boundaries. Collect authorized, de-identified real requests and ambiguous no-service questions. Keep source conversations and related incidents grouped. Create new held-out groups before another tuning cycle. Do not describe the current 1,800 rows as reviewed real tickets.

## References

- [SetFit training overview](https://huggingface.co/docs/setfit/quickstart)
- [SetFit training arguments](https://huggingface.co/docs/setfit/reference/trainer)
- [SentenceTransformer training](https://www.sbert.net/docs/sentence_transformer/training_overview.html)

The existing retrieval collection was verified separately: all 30 stored 384-dimensional vectors match the original pretrained retrieval encoder (maximum absolute difference 2.54e-7). Chroma SQLite metadata changes on access, so this is a content/vector check rather than byte-for-byte database identity. The 36 historical development routing cases also pass; they are regression checks, not new held-out evidence.
