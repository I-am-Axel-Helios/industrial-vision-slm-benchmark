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

In development.

### Completed
- Project repository and development environment
- Google Colab GPU setup
- Qwen3-1.7B initial inference experiments
- Initial latency and GPU memory measurements
- Evaluation protocol and bilingual documentation
- 50-question OpenCV evaluation dataset
- Dataset validator, unit tests, and audit
- Dataset freezing with SHA-256 verification

### Current Phase
Section 10 — Baseline evaluation of Qwen3-1.7B.

### Next Steps
1. Run baseline inference on the development questions.
2. Implement reproducible inference and metric logging.
3. Evaluate the frozen test split.
4. Develop the RAG pipeline.
5. Explore QLoRA fine-tuning and optional DPO.
6. Compare quality, latency, and GPU memory usage.

## Results

The first inference experiments have been completed.
The full comparative benchmark is pending.

## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)