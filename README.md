# Industrial Vision SLM Benchmark

[English](README.md) | [Français](README.fr.md)

## Description

Personal and academic project focused on specializing
a Small Language Model (SLM) for image processing
and industrial computer vision.

The objective is to investigate and compare different
approaches to adapting a language model to a
specialized technical domain.

## Objectives

- Build a specialized computer vision document corpus.
- Evaluate a pretrained Small Language Model.
- Develop a RAG pipeline with vector retrieval.
- Perform supervised fine-tuning using QLoRA.
- Compare the performance of different approaches.
- Explore preference optimization (DPO), if feasible.

## Planned Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PEFT / LoRA / QLoRA
- Google Colab (NVIDIA T4)
- Streamlit

## Base Model

Qwen3-1.7B (`Qwen/Qwen3-1.7B`)

## Project Structure

- `configs/`: experiment configurations.
- `notebooks/`: experimental notebooks.
- `src/`: reusable Python source code.
- `data/`: datasets and document corpora.
- `outputs/`: results, metrics and trained adapters.
- `tests/`: automated tests.
- `docs/`: technical documentation and protocols.

## Evaluation Dataset

Section 9 produced a versioned OpenCV evaluation dataset with 50 technical questions across five topics:
- Image filtering and denoising
- Thresholding and segmentation
- Mathematical morphology
- Edge detection and contour analysis
- Industrial visual inspection

### Dataset Structure

| Split | Questions | Purpose |
|---|---:|---|
| Development | 10 | Prompt and evaluation pipeline development |
| Test | 40 | Held-out model evaluation |
| Total | 50 | Cross-domain technical benchmark |

Each question contains:
- A unique identifier
- A development or test split
- A technical topic
- A question
- A reference answer
- Expected key points
- A documentation source URL

References are aligned with OpenCV 4.13.0.

### Validation and Reproducibility

The evaluation dataset includes:

- JSONL schema validation
- Six automated unit tests
- Topic and split distribution checks
- Textual similarity screening
- SHA-256 fingerprints for the complete and test datasets

The dataset is frozen using a versioned manifest:

`data/evaluation/freeze_manifest.json`

Run the following commands from the project root:

```bash
python src/validate_evaluation_dataset.py
python src/audit_evaluation_dataset.py
python -m unittest discover -s tests -p "test_*.py" -v
python src/freeze_evaluation_dataset.py verify
```

The test split must not be used for model training,
prompt optimization, or hyperparameter selection.

The checks ensure structural consistency and detect
changes to the frozen dataset. They do not independently
guarantee the correctness of every reference answer.

## Project Status

**Current milestone: Baseline evaluation completed**

### Completed
- Google Colab GPU environment setup.
- Qwen3-1.7B tokenizer and model initialization.
- Initial inference and GPU performance measurements.
- Construction and validation of a 50-question evaluation dataset.
- Frozen development/test split: 10 development and 40 test questions.
- Definition of the manual evaluation protocol.
- Comparison of two prompting strategies on the development set.
- Selection and locking of prompt V1 before test evaluation.
- Baseline inference on all 40 test questions.
- Manual quality assessment and performance analysis.
- Validation and persistence of final benchmark results.


### Next Steps
1. Build a specialized computer vision document corpus.
2. Implement and evaluate a Retrieval-Augmented Generation (RAG) pipeline.
3. Fine-tune the base model using LoRA/QLoRA.
4. Compare baseline, RAG, and fine-tuned configurations.
5. Explore preference optimization (DPO), if feasible.

## Baseline Evaluation

The baseline was evaluated using the pretrained
`Qwen/Qwen3-1.7B` model without RAG or fine-tuning.

**Experimental configuration:**

- Model: Qwen3-1.7B.
- Precision: FP16.
- Hardware: NVIDIA Tesla T4.
- Prompt: V1, selected using development questions only.
- Generation: deterministic, with thinking mode disabled.
- Maximum generation length: 512 new tokens.
- Evaluation dataset: 50 questions (10 development, 40 test).
- Domains: filtering, segmentation, morphology, edges, and industrial inspection.

Each test response was manually assessed using:

- Technical accuracy: 0–3 points.
- Completeness: 0–2 points.
- Factual error: independent boolean indicator.

The maximum test score is 200 points.

## Results

### Overall Test Performance

| Metric | Result |
|---|---:|
| Evaluated questions | 40/40 |
| Technical quality | 113/200 (56.5%) |
| Responses flagged for factual errors | 31/40 (77.5%) |
| Generations reaching the token limit | 9/40 (22.5%) |
| Mean generation latency | 16.932 s |
| Global generation throughput | 24.10 tokens/s |
| Maximum observed PyTorch GPU memory | 3.31 GiB |

### Quality by Domain

| Domain | Score | Quality |
|---|---:|---:|
| Edges | 27/40 | 67.5% |
| Filtering | 26/40 | 65.0% |
| Industrial inspection | 25/40 | 62.5% |
| Segmentation | 18/40 | 45.0% |
| Morphology | 17/40 | 42.5% |

### Key Findings

- The pretrained model handles several general image-processing concepts correctly.
- Important errors remain in OpenCV API usage, mathematical definitions, and morphological operations.
- Morphology and segmentation achieved the lowest scores in this evaluation.
- Long and fluent explanations do not necessarily imply technical correctness.
- Reaching the generation limit is a potential source of incomplete answers.

These findings are specific to the frozen 40-question test set and should not be interpreted as general estimates of model performance.

## Reproducibility

The evaluation resources include:

- `data/evaluation/questions.jsonl`: evaluation questions and reference answers.
- `data/evaluation/freeze_manifest.json`: frozen dataset metadata and checksums.
- `notebooks/01_baseline_evaluation.ipynb`: baseline experiments and evaluation workflow.
- `docs/evaluation_protocol.md`: evaluation methodology.
- `docs/evaluation_protocol.fr.md`: French evaluation methodology.

Experimental responses, metrics, and manual assessments were saved persistently to Google Drive under:

`industrial-vision-slm-benchmark/outputs/baseline/`

The finalized quality results are stored in:

- `baseline_test_v1_quality_scores_final.json`
- `baseline_test_v1_quality_summary.json`

The locked configuration and test set will be preserved for future RAG and fine-tuning comparisons.

**Limitations:** the test set is relatively small, quality scores rely on manual assessment, and runtime performance depends on the execution environment.

## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)
- [Baseline Evaluation Notebook](notebooks/01_baseline_evaluation.ipynb)