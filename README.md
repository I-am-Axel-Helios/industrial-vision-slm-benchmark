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

**Current milestone:** Baseline evaluation of Qwen3-1.7B, OpenCV document corpus preparation, E5 + FAISS semantic retrieval evaluation, and RAG pipeline development completed. Notebook 04 was validated on Google Colab by reusing existing experimental results.

**Next phase:** Fine-tune Qwen3-1.7B using supervised LoRA/QLoRA adaptation, then prepare a controlled comparison between the base model, the RAG pipeline, and the adapted configurations.


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
- Multilingual E5 embeddings generated and saved for the 55 retrieval chunks (384 dimensions).
- Two overlength chunks handled with documented token-window embedding aggregation.
- Exact FAISS `IndexFlatIP` search validated against NumPy and saved with integrity checks.
- Configurable Top-k retrieval and OpenCV source tracing implemented.
- Development evaluation split loaded (10 questions); test split kept for final assessment (40 questions).
- Completed relevance annotation for 10 development questions across 55 chunks, totaling 550 judgments.
- Evaluated retrieval using Precision@k, Recall@k, and MRR for k = 1, 3, and 5.
- Analyzed ranking errors and identified document coverage limitations.
- Built a RAG pipeline combining Qwen3-1.7B with the E5 + FAISS retrieval engine.
- Integrated structured documentary context with `[S1]`, `[S2]`, etc. source citations.
- Defined and tokenized RAG prompts with thinking mode disabled.
- Completed 20 experiments on 10 development questions using Top-3 and Top-5 configurations.
- Implemented persistent generation storage and a resume mechanism to avoid unnecessary regeneration.
- Manually evaluated 20 responses for factual accuracy and 17 citation-bearing responses for documentary support.
- Consolidated evaluation results and performed a paired Top-3 versus Top-5 comparison.
- Verified 80 documentary evidence passages associated with the evaluations.
- Validated Notebook 04 on Google Colab: all 34 Python cells executed without errors, reusing the 20 existing generations.



### Next Steps

1. Prepare supervised fine-tuning of Qwen3-1.7B using LoRA/QLoRA.
2. Build and evaluate the adapted model configurations.
3. Prepare a controlled comparison of the baseline, RAG pipeline, and fine-tuned models using comparable questions and evaluation criteria.
4. Study trade-offs between response quality, latency, memory consumption, and inference cost.
5. Explore preference optimization (DPO), if resources and previous experimental results allow.


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

The locked configuration and baseline results on the 40-question test set are preserved as a reference for future comparisons.

Notebook 04 RAG results are based exclusively on the ten development questions and use a different evaluation rubric. They therefore do not yet constitute a direct comparison with the baseline test results.

**Limitations:** the test set is relatively small, quality scores rely on manual assessment, and runtime performance depends on the execution environment.

## Document Corpus Preparation

The project includes a specialized technical corpus built from the official OpenCV 4.13.0 documentation.

The corpus has been embedded, indexed, and evaluated on the development split. Results demonstrate effective retrieval of directly relevant passages when available, while highlighting ranking errors and incomplete document coverage.

The corpus has been integrated with Qwen3-1.7B in the Notebook 04 RAG pipeline, which was evaluated on the development split.


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
4. Preserve headings, mathematical expressions, and Python code examples.
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

The selected chunks have a mean length of 757.5 characters and a median length of 808 characters.

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

Generated document content is intended to remain local when excluded by `.gitignore`. The notebook and metadata document the preparation procedure.

SHA-256 checks verify the integrity of source documents,
processed documents, and the resulting corpora.

### Limitations

- The corpus contains only seven OpenCV documentation pages.
- Domain coverage is not uniform.
- Illustrations are not included in the text corpus.
- Chunk sizes are measured in characters rather than tokens.
- The retrieval filtering policy uses simple structural rules.
- Retrieval has been evaluated on ten development questions, but its performance on the held-out test split has not yet been measured.
- RAG generation has been evaluated on the ten development questions; evaluation on the 40-question held-out test split remains future work.

## Semantic Retrieval Engine (Development Evaluation Completed)

The retrieval pipeline is implemented in `notebooks/03_retrieval_evaluation.ipynb`.

It searches the **55 selected OpenCV chunks** using a multilingual text encoder and an exact vector index.

### Implemented Pipeline

1. Encode French queries and English OpenCV passages with `intfloat/multilingual-e5-small`.
2. Use the required `query: ` and `passage: ` prefixes; normalize embeddings to unit length.
3. Generate **55 embeddings of 384 dimensions** on CPU.
4. Handle two passages exceeding the 512-token input limit through token windows and length-weighted embedding aggregation, without changing the original corpus.
5. Build an exact FAISS `IndexFlatIP` index; verify its Top-3 ranking and scores against an exhaustive NumPy search.
6. Save and reload embeddings, chunk identifiers, and the FAISS index with SHA-256 integrity checks and idempotent writes.
7. Implement reusable Top-k retrieval and attach source titles, sections, official OpenCV URLs, and `[S1]`-style citations.

**Exploratory checks:** Top-1, Top-3, and Top-5 retrieval was verified for three technical questions (nine searches total).

Rankings were stable across k values. These checks confirm implementation consistency, document relevance was subsequently evaluated quantitatively on the development split.

### Observed Ranking Limitations

Some first-ranked passages were only indirectly related to the question:

| Exploratory question | First-ranked section | Similarity |
|---|---|---:|
| Morphological erosion | Morphology — Theory | 0.8508 |
| Gaussian filtering before Canny | Filtering — Bilateral Filtering | 0.8457 |
| Template matching | Contours Hierarchy — RETR_CCOMP | 0.8281 |

Similarity scores are **not relevance probabilities**.


### Quantitative Retrieval Evaluation

The frozen evaluation dataset contains **50 questions**: **10 development** and **40 held-out test** questions.

All ten development questions were annotated across the 55 document chunks, producing **550 relevance judgments** stored in `data/evaluation/retrieval_qrels_dev.json`.

Two relevance conventions were evaluated:

- **Main:** grades 1 and 2 are relevant.
- **Strict:** only grade 2 is relevant.

Metrics were computed at k = 1, 3, and 5.

| Metric | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Overall Precision — main | 0.5000 | 0.3000 | 0.2400 |
| Conditional macro Recall — main | 0.3021 | 0.4583 | 0.6250 |
| Conditional MRR — main | 0.6250 | 0.7292 | 0.7604 |
| Overall Precision — strict | 0.4000 | 0.2000 | 0.1400 |
| Conditional macro Recall — strict | 0.5714 | 0.8571 | 1.0000 |
| Conditional MRR — strict | 0.5714 | 0.6905 | 0.7262 |

**Document coverage:**

- Main convention: 8 of 10 questions covered.
- Strict convention: 7 of 10 questions covered.
- CV008: only partially relevant passages available.
- CV009 and CV010: no relevant passages identified.

At k = 5, all **7 directly relevant passages** were retrieved for questions with directly relevant evidence.

Under the main convention, **12 of 21 relevant question-chunk pairs** were retrieved within the Top-5, corresponding to a micro Recall of 57.14%.

Conditional Recall and MRR exclude questions with no relevant passage. Overall Precision includes all ten development questions.

### Ranking Analysis and Limitations

Qualitative evaluation identified several ranking issues:

- CV001: the directly relevant passage ranked second.
- CV002: the directly relevant passage ranked third.
- CV007: the Canny hysteresis explanation ranked fourth, behind less relevant passages.

These findings highlight the trade-off between document recall and irrelevant context as retrieval depth increases.

**Limitations:**

- Evaluation based on only ten questions.
- Human relevance judgments without independent inter-annotator validation.
- Incomplete document coverage.
- No isolated evaluation of long-passage embedding aggregation.
- RAG answer quality was evaluated in Notebook 04 on the development split only; performance on the held-out test split remains to be assessed.

Similarity scores are not relevance probabilities.

The **40 held-out test questions** were not used
to tune retrieval.

### Reproducibility

Development relevance judgments are versioned in `data/evaluation/retrieval_qrels_dev.json`.

Generated embeddings, chunk identifiers, and the FAISS index remain local in the Git-ignored `outputs/retrieval/` directory.

The notebook documents SHA-256 integrity checks, idempotent saves, retrieval experiments, and evaluation metrics.

**Completed integration:** the E5 + FAISS retrieval engine is used by the Notebook 04 RAG pipeline. Top-3 and Top-5 configurations were evaluated on the development split. A direct comparison against the baseline and future adapted models still requires a common experimental protocol.


## RAG Pipeline — Development and Evaluation Completed

The RAG pipeline is implemented in `notebooks/04_rag_evaluation.ipynb`.

It combines the Qwen3-1.7B language model with the E5 + FAISS semantic retrieval engine to generate technical answers grounded in the OpenCV documentation.

### System Architecture

The pipeline follows these steps:

1. Receive a technical question in French.
2. Encode the question using `intfloat/multilingual-e5-small`.
3. Retrieve relevant passages from the FAISS index.
4. Build documentary context with `[S1]`, `[S2]`, etc. source references.
5. Construct and tokenize the RAG prompt.
6. Generate an answer using `Qwen/Qwen3-1.7B`, with thinking mode disabled.
7. Persist the answer, experimental parameters, and documentary references.
8. Manually evaluate factual accuracy and documentary support.

The system reuses the 55 OpenCV retrieval chunks, their 384-dimensional embeddings, and the previously validated `IndexFlatIP` index without modifying the documentary corpus.

### Experimental Protocol

The experiments use ten development questions, each evaluated with two retrieval configurations:

- **Top-3:** three retrieved documentary passages.
- **Top-5:** five retrieved documentary passages.

A total of 20 RAG responses are available.

Manual assessment uses two criteria:

- **Factual accuracy:** score from 0 to 5.
- **Documentary support:** score from 0 to 2, applied only to responses containing citations.

Factual accuracy was evaluated for all 20 responses. Documentary support was evaluated for the 17 responses containing at least one citation.

The 80 experiment-to-passage associations were preserved to support source verification.

The 40 held-out test questions were not used to tune the RAG pipeline.

### Reference Experiment Results

| Metric | Top-3 | Top-5 |
|---|---:|---:|
| Evaluated responses | 10 | 10 |
| Total factual score | 32/50 | 38/50 |
| Mean factual accuracy | 3.20/5 | 3.80/5 |
| Responses containing citations | 8/10 | 9/10 |
| Mean documentary support | 1.00/2 | 1.44/2 |
| Insufficient support (0/2) | 1 | 0 |
| Partial support (1/2) | 6 | 5 |
| Complete support (2/2) | 1 | 4 |

Documentary support averages are calculated only over responses containing citations. The three responses without citations are not automatically assigned a score of 0/2.

Across the ten evaluated questions, Top-5 improved factual scores for five questions and tied Top-3 on the remaining five.

Mean factual accuracy increased from 3.20/5 to 3.80/5. Mean documentary support increased from 1.00/2 to 1.44/2.

These findings are limited to the reference experiment and do not establish statistically generalizable superiority of Top-5.

### Persistence and Reproducibility

The notebook supports three execution modes:

- `auto`: reuse all 20 responses when available; otherwise generate missing responses.
- `reuse`: use existing generations only.
- `generate_missing`: preserve existing results and generate missing responses.

Generating new responses requires a compatible CUDA GPU. Reusing existing results does not require loading the generative model weights onto a GPU.

Generated responses are stored in:

`outputs/rag/dev_runs/`

Evaluation artifacts are stored in:

`outputs/rag/analysis/`

The main annotation files are:

- `rag_human_review_dev.csv`: persistent manual evaluation sheet.
- `rag_review_mapping_dev.json`: mapping between review identifiers and experiments.
- `rag_source_evidence_dev.csv`: documentary evidence associated with responses.

Human scores are never assigned automatically. Evaluation statistics are calculated only when the required annotations are available.

Existing generation and annotation files are preserved during compatible resumed executions.

Notebook 04 was validated on Google Colab in reuse mode:

- All 34 Python cells executed without errors.
- All 20 RAG generations were reused.
- No new generation was required.
- All 20 factual assessments were verified.
- All 17 documentary assessments were verified.
- The final validation in section 5 succeeded.

A complete from-scratch execution on an independent installation has not yet been validated.

The nine required documentary artifacts must be available before execution. They can be prepared using the previous notebooks or provided through the `rag_colab_assets.zip` archive.

Generated results and annotations are stored in persistent output directories. Reproducing the reference evaluation on another environment requires access to these artifacts, which are not necessarily tracked by Git.

### Limitations and Future Work

- Evaluation limited to ten development questions.
- Documentary corpus restricted to seven OpenCV pages.
- Human judgments without independent inter-annotator validation.
- Citations sometimes provide only partial or insufficient support.
- No direct comparison against the baseline using the same questions and scoring criteria.
- Inference cost and memory consumption were not systematically compared between Top-3 and Top-5.
- RAG performance has not yet been evaluated on the 40-question held-out test split.

The next phase will focus on adapting Qwen3-1.7B with LoRA/QLoRA, followed by a controlled comparison against the reference configurations.

## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)
- [Baseline Evaluation Notebook](notebooks/01_baseline_evaluation.ipynb)
- [Document Corpus Preparation Notebook](notebooks/02_document_corpus.ipynb)
- [Semantic Retrieval and Evaluation Notebook](notebooks/03_retrieval_evaluation.ipynb)
- [RAG Pipeline Construction and Evaluation Notebook](notebooks/04_rag_evaluation.ipynb)