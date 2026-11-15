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

**Current milestone:** Baseline evaluation and document corpus preparation completed.

**Next phase:** Embeddings, vector indexing, and RAG retrieval evaluation.

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
- Construction and validation of the OpenCV document corpus.
- Generation of 62 chunks and selection of 55 retrieval chunks.
- Corpus integrity checks and reproducibility report.


### Next Steps
1. Generate multilingual embeddings for the retrieval corpus.
2. Build and evaluate a vector search index.
3. Integrate retrieved context with Qwen3-1.7B.
4. Compare baseline and RAG performance.
5. Fine-tune the base model using LoRA/QLoRA.
6. Compare baseline, RAG, and fine-tuned configurations.
7. Explore preference optimization (DPO), if feasible.

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

## Document Corpus Preparation

The project includes a specialized technical corpus built from the official OpenCV 4.13.0 documentation.

The corpus was designed to provide external knowledge for the upcoming Retrieval-Augmented Generation (RAG) experiments.

### Data Sources

Seven OpenCV documentation pages were collected, covering five technical domains:

- Image filtering
- Mathematical morphology
- Thresholding and segmentation
- Edge detection and contour analysis
- Industrial visual inspection

The source registry records document identifiers, original URLs, processing paths, and SHA-256 checksums.

### Processing Pipeline

The corpus preparation follows this workflow:

1. Download and verify the original HTML documents.
2. Extract technical content using Beautiful Soup.
3. Convert HTML into structured Markdown.
4. Preserve headings, mathematical expressions,
   and Python code examples.
5. Apply hierarchical chunking based on Markdown sections.
6. Validate chunk integrity and content preservation.
7. Filter navigation, exercises, and reference-only passages.

Missing illustrations are explicitly identified rather
than replaced with invented descriptions.

### Corpus Statistics

| Metric | Result |
|---|---:|
| Source documents | 7 |
| Technical domains | 5 |
| Extracted characters | 42,335 |
| Preserved Python code blocks | 26 |
| Original chunks | 62 |
| Preserved atomic blocks | 228/228 |
| Excluded chunks | 7 |
| Final retrieval chunks | 55 |

### Chunking Configuration

- Strategy: hierarchical section-based chunking.
- Maximum chunk size: 1,500 characters.
- Overlap: 0 characters.
- Code blocks and mathematical expressions preserved.
- Section headings repeated in each chunk.

### Retrieval Corpus Distribution

| Domain | Chunks |
|---|---:|
| Edges | 23 |
| Filtering | 7 |
| Industrial inspection | 6 |
| Morphology | 10 |
| Segmentation | 9 |
| **Total** | **55** |

The selected chunks have a mean length of 757.5
characters and a median length of 808 characters.

### Reproducibility

The corpus pipeline is implemented in:

`notebooks/02_document_corpus.ipynb`

The generated metadata and manifests include:

- `data/corpus/source_registry.json`
- `data/corpus/chunk_manifest.json`
- `data/corpus/retrieval_manifest.json`
- `data/corpus/corpus_report.json`

The pipeline also generates:

- `data/corpus/chunks/chunks.jsonl`
- `data/corpus/chunks/retrieval_chunks.jsonl`

Generated document content is intended to remain local
when excluded by `.gitignore`. The notebook and metadata
document the preparation procedure.

SHA-256 checks verify the integrity of source documents,
processed documents, and the resulting corpora.

### Limitations

- The corpus contains only seven OpenCV documentation pages.
- Domain coverage is not uniform.
- Illustrations are not included in the text corpus.
- Chunk sizes are measured in characters rather than tokens.
- The retrieval filtering policy uses simple structural rules.
- Retrieval relevance has not yet been evaluated.
- No RAG inference results are available yet.

The next phase will focus on embeddings, vector indexing,
retrieval evaluation, and integration with Qwen3-1.7B.

## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)
- [Baseline Evaluation Notebook](notebooks/01_baseline_evaluation.ipynb)
- [Document Corpus Preparation Notebook](notebooks/02_document_corpus.ipynb)