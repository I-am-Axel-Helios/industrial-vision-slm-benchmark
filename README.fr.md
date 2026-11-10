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
- `docs/` : documentation technique et protocoles.

## Dataset d'évaluation
La section 9 a permis de constituer un jeu d'évaluation OpenCV versionné de 50 questions techniques couvrant cinq domaines :
- Filtrage et débruitage
- Seuillage et segmentation
- Morphologie mathématique
- Détection de contours et analyse de formes
- Inspection visuelle industrielle

### Organisation du dataset

| Ensemble | Questions | Utilisation |
|---|---:|---|
| Développement | 10 | Développement des prompts et du pipeline d'évaluation |
| Test | 40 | Évaluation réservée des modèles |
| Total | 50 | Benchmark technique multidomaine |

Chaque question contient :
- Un identifiant unique
- Un ensemble dev ou test
- Un domaine technique
- Une question
- Une réponse de référence
- Des points clés attendus
- Une URL documentaire

Les références sont harmonisées vers OpenCV 4.13.0.

### Validation et reproductibilité
Le dataset dispose des contrôles suivants :
- Validation de la structure JSONL
- Six tests unitaires automatisés
- Vérification des domaines et de la répartition
- Détection de similarités textuelles
- Empreintes SHA-256 du dataset complet et du jeu test

Un manifeste versionné conserve les empreintes :
`data/evaluation/freeze_manifest.json`

Commandes à exécuter depuis la racine du projet :

```bash
python src/validate_evaluation_dataset.py
python src/audit_evaluation_dataset.py
python -m unittest discover -s tests -p "test_*.py" -v
python src/freeze_evaluation_dataset.py verify
```

Les questions de test ne doivent pas être utilisées
pour entraîner le modèle, optimiser les prompts ou
sélectionner les hyperparamètres.

Ces contrôles garantissent la cohérence structurelle
et permettent de détecter les modifications du dataset
figé. Ils ne garantissent pas, à eux seuls, l'exactitude
scientifique de chaque réponse de référence.

## État du projet

### Terminé
- Création du dépôt et de l'environnement de développement
- Configuration de Google Colab avec GPU
- Premières inférences de Qwen3-1.7B
- Premières mesures de latence et de mémoire GPU
- Protocole d'évaluation et documentation bilingue
- Dataset OpenCV de 50 questions
- Validateur, tests unitaires et audit
- Gel du dataset avec vérification SHA-256

### Phase actuelle

Section 10 — Évaluation initiale de Qwen3-1.7B.

### Prochaines étapes
1. Exécuter les premières évaluations sur le jeu dev.
2. Implémenter une inférence reproductible et les mesures.
3. Évaluer le jeu de test figé.
4. Développer le pipeline RAG.
5. Étudier le fine-tuning QLoRA et éventuellement DPO.
6. Comparer qualité, latence et consommation GPU.

## Résultats

Les résultats expérimentaux seront ajoutés après les évaluations.