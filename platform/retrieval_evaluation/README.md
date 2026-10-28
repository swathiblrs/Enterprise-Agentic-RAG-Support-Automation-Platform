# Expanded Retrieval Evaluation

The primary retrieval benchmark now has 100 answerable questions across 80 scenario groups, plus 20 unsupported-question controls. It covers all 15 support documents. The prior 20-question suite remains a historical cohort.

## Composition

| Cohort | Questions | Purpose |
| --- | --- | --- |
| Original controlled cases | 20 | Preserve the earlier retrieval measurement |
| Existing public FAQ cases | 40 | Preserve service-specific development coverage |
| New questions | 40 | Add procedural, scope, ownership, priority, and multi-source cases |
| Unsupported controls | 20 | Observe behavior on questions outside the indexed knowledge |

All labels are project-authored and not independently human-reviewed. Related formulations have a shared scenario_group; the 100 answerable questions represent 80 groups. Reused development questions do not constitute a new untouched external test.

## Measured Results at K=3

| Metric | Result on 100 answerable questions |
| --- | --- |
| Expected-source Recall@3 | 93.00% |
| Hit@3 | 94.00% |
| Fixed-slot expected-source Precision@3 | 34.33% |
| Precision over returned unique sources | 86.67% |
| Top-1 expected-source accuracy | 93.00% |
| Mean reciprocal rank | 0.9350 |
| nDCG@3 | 0.9286 |
| Scenario-group macro Recall@3 | 93.23% |

The 20-question original cohort reproduces 100% Recall@3 and 46.67% fixed-slot Precision@3. The 40 existing FAQ cases have 100% recall; the 40 new questions have 82.5% recall.

The broader result should replace the old 20-question number when describing the current evaluation. It does not mean the retriever was changed or regressed: the question set, scope, and relevance-label density changed.

## Why Precision@3 Is Lower

The answerable set contains 89 questions with one expected source and 11 with two. With a fixed denominator of three slots per question, even finding every expected source yields at most 111/300 = 37% expected-source Precision@3. The run matched 103 expected sources across those slots.

Returning one expected document scores 1/3 fixed-slot precision and 100% recall for a question with one expected document. Precision over returned sources is reported separately to make that distinction visible. Expected-source lists designate the annotated references; they are not exhaustive human judgments of every potentially useful document.

## Formulas

For each answerable question, use the sources of the top three returned chunks, deduplicate source IDs, and do not fill missing slots from later results.

- Precision@3 = unique matched expected sources / 3.
- Recall@3 = unique matched expected sources / number of expected sources.
- Hit@3 = 1 if at least one expected source appears, otherwise 0.
- Top-1 = 1 if the first returned source is expected.
- MRR = reciprocal rank of the first expected source, or 0 when absent.
- nDCG@3 uses binary relevance and an ideal ranking containing up to three expected sources.

Question-macro metrics average these values across the 100 answerable questions. Group-macro metrics first average questions within each scenario group, then average the 80 groups.

## Gaps Exposed

Six new questions did not retrieve their expected institution-specific source when phrased with generic campus/university wording. Two questions found only one of their two expected documents. The primary patterns are generic-vs-specific source selection and suppression of supplementary sources.

These are incomplete expected-source matches, not automatically eight incorrect answers; the benchmark does not generate or semantically judge answers. The retriever, its intent rules, and the index were left unchanged during this evaluation.

All 20 unsupported controls still returned some retrieval context. They are excluded from precision/recall, and this diagnostic does not establish whether the full application would answer them: domain routing, clarification, and answer generation are separate stages.

Next improvements should address scope/tenant context and multi-source selection, followed by fresh independently labeled questions. Do not tune labels merely to match the existing output.

## Run

From platform/:

```sh
.venv-support/bin/python retrieval_benchmark.py
```

Or from platform/support/:

```sh
../.venv-support/bin/python -m src.evaluator
```

The evaluator defaults to the expanded retrieval suite. The historical full support-workflow evaluation remains available using --legacy-workflow; its old heuristic answer metrics are not semantic faithfulness measurements.

The question file is support/tests/retrieval_eval_questions.json. results_100.json records the dataset, corpus, and retriever hashes, per-query outcomes, strata, grouping, and evaluation mode. The runner calls fresh retrieval without the public FAQ response cache or an LLM. It does not change the corpus or fit a model.
