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

## Project Status

In development.

Completed:
- Colab GPU environment setup.
- Qwen3-1.7B tokenizer and model loading.
- Initial inference and GPU performance measurements.

In progress:
- Evaluation dataset and scoring protocol.

Planned:
- Document corpus construction.
- RAG implementation and evaluation.
- QLoRA fine-tuning.
- Preference optimization, if feasible.

## Results

The first inference experiments have been completed.
The full comparative benchmark is pending.

## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)