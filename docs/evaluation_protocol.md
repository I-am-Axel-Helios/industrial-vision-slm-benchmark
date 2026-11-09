# Evaluation Protocol — Industrial Vision SLM Benchmark

[English](evaluation_protocol.md) | [Français](evaluation_protocol.fr.md)

## 1. Objective

Evaluate the technical reliability of Qwen3-1.7B
on image processing, OpenCV and industrial vision.

The same evaluation set will be used to compare
the baseline, RAG, SFT and optional DPO systems.

## 2. Evaluation dataset

Dataset: `data/evaluation/questions.jsonl`

Each question contains:
- id
- split (dev or test)
- topic
- question
- reference_answer
- key_points
- source_url

Development questions are used to debug and validate
the evaluation pipeline.

Test questions remain held out from model training,
prompt optimization and hyperparameter selection.

## 3. Evaluation criteria

### 3.1. Factual correctness (0–3)

- 3: Technically correct.
- 2: Correct overall, with minor inaccuracies.
- 1: Partially correct, with major errors or gaps.
- 0: Incorrect, irrelevant or unanswered.

### 3.2. Completeness (0–2)

- 2: All essential key points are covered.
- 1: Some essential points are missing.
- 0: Essential points are absent.

### 3.3. Hallucination

Boolean indicator:
- true: At least one demonstrably false or invented
  technical claim.
- false: No such claim identified.

A missing detail alone is not a hallucination.

### 3.4. Context faithfulness (0–2)

Only applicable to RAG systems.

- 2: Important claims are supported by retrieved context.
- 1: Some important claims lack sufficient support.
- 0: Main claims are unsupported or contradicted.

For non-RAG systems: N/A.

## 4. Aggregated metrics

Quality score = factual correctness + completeness.

Maximum quality score = 5.

Report:
- Mean quality score.
- Mean factual correctness.
- Mean completeness.
- Hallucination rate.
- Mean context faithfulness for RAG.
- Citation correctness for RAG.
- Median inference latency.
- Generation throughput.
- Peak GPU memory.

## 5. Evaluation rules

- Use the same held-out test questions.
- Keep generation settings consistent when possible.
- Do not train on test questions or reference answers.
- Do not use test results to select prompts or models.
- Verify reference answers against documented sources.
- Do not assume semantic similarity implies correctness.
- Record uncertain manual judgments.
- Report limitations and evaluation sample sizes.

## 6. Experimental status

Protocol defined.

Full evaluation dataset and benchmark pending.