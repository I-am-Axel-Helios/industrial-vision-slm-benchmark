# Industrial Vision SLM Benchmark

[English](README.md) | [Français](README.fr.md)

## Description

Projet personnel consacré à la spécialisation d'un Small Language Model (SLM) pour le traitement d'images et la vision industrielle.

L'objectif est d'étudier et de comparer différentes méthodes d'adaptation d'un modèle de langage à un domaine technique.

## Objectifs

- Construire un corpus documentaire spécialisé en Computer Vision.
- Évaluer un Small Language Model préentraîné.
- Développer un pipeline RAG avec recherche vectorielle.
- Réaliser un fine-tuning supervisé avec QLoRA.
- Comparer les performances des différentes approches.
- Étudier l'optimisation par préférences (DPO), si réalisable.

## Technologies prévues

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PEFT / LoRA / QLoRA
- Google Colab (NVIDIA T4)
- Streamlit

## Modèle de départ

Qwen3-1.7B (`Qwen/Qwen3-1.7B`)

## Organisation

- `configs/` : configurations des expériences.
- `notebooks/` : notebooks d'expérimentation.
- `src/` : code Python du projet.
- `data/` : données et corpus documentaires.
- `outputs/` : résultats, métriques et modèles entraînés.
- `tests/` : tests automatisés.

## État du projet

Terminé :
- Configuration de l'environnement GPU Colab.
- Chargement du tokenizer et du modèle Qwen3-1.7B.
- Inférence initiale et mesures de performance GPU.

En cours :
- Ensemble de données d'évaluation et protocole de notation.

Prévu :
- Construction du corpus de documents.
- Mise en œuvre et évaluation de RAG.
- Fine-tuning QLoRA.
- Optimisation des préférences, si possible.

## Résultats

Les résultats expérimentaux seront ajoutés après les évaluations.