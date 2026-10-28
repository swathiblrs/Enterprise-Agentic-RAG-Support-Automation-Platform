# QLoRA Retraining v2

**Status: the second Colab GPU run completed and its artifacts were verified.** It achieved 95.83% accuracy and 95.82% macro-F1 on the new 120-case synthetic benchmark; both adapters were evaluated on identical rows. [Results](../llama_training/qwen_v2_reference/RESULTS.md). The original Colab run completed with 94.17% accuracy and 94.18% macro-F1 on its original 120 synthetic cases. Its seven mistakes informed this experiment, so those cases are no longer a held-out test.

## Changes

- Retain 540 distinct original training scenarios and remove 1,080 repetitive formatting variants.
- Add 72 project-authored synthetic boundary examples covering subtle security symptoms, infrastructure configuration versus compromise, application versus data failures, and non-support intent.
- Use 612 balanced training examples. No original test sentence is copied into training.
- Use the original 90 validation cases plus the previously inspected 120-case challenge as 210 development cases.
- Reserve 120 new synthetic cases in 60 paraphrase groups for one final evaluation. Grouping and lexical screening do not establish independent human annotation or absence of semantic overlap.
- Train on domain-value token loss, masking JSON punctuation, the common key, EOS, system and user tokens. Inference remains the original six-candidate JSON likelihood scoring for an apples-to-apples comparison with the saved v1 adapter.
- Retrain from the pinned base model with rank 16, learning rate 0.0001 (previously 0.0002), dropout 0.10 (previously 0.05), up to 300 steps, and validation-loss early stopping.
- Evaluate the saved v1 adapter and v2 adapter on the **same new test**. Also report untuned Qwen and the original fixed TF-IDF/MiniLM/SetFit baselines. Those baselines do not receive the new training augmentation; this is an iteration comparison, not a controlled equal-training-data algorithm study.

## Run on Colab

1. Open `QLoRA_Retraining_v2.ipynb` in hosted Google Colab.
2. Select a free T4 GPU, not a local runtime or a paid fallback.
3. Run cells in order and upload `colab_retraining_v2.zip` when prompted.
4. Download `qlora_v2_results.zip`. The `qlora_v1` and `qlora` rows compare both adapters on the new test.

The previous adapter is bundled only as an evaluation reference; this is a fresh training run, not resumption of an optimizer checkpoint. No models or packages are downloaded or installed on the Mac. No paid API, private tickets, production database, automatic deployment or arbiter is involved. The result export excludes optimizer pickle checkpoints.

## Success Criteria

Select using development data before opening the new test. Compare macro-F1, security recall, other per-domain recalls, accepted accuracy/coverage, memory and p95 latency on the common new test. Report regressions as well as gains. There is no guarantee that training longer or adding examples improves results. Do not tune again on the same test and claim it remains untouched.

The new cases are project-authored synthetic examples, not independently reviewed real support tickets. Different cases and paired paraphrases mean the effective number of independent scenarios is smaller than the row count. Real-world quality still needs external annotation and a separately sourced evaluation set.
