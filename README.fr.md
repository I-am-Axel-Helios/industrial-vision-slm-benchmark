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

**Étape actuelle : benchmark initial terminé.**

### Terminé
- Configuration de l'environnement GPU Google Colab.
- Chargement du tokenizer et du modèle Qwen3-1.7B.
- Première inférence et mesures de performance GPU.
- Construction et validation d'un dataset de 50 questions.
- Gel de la répartition : 10 questions de développement et 40 questions de test.
- Définition du protocole de notation manuelle.
- Comparaison de deux stratégies de prompting sur le jeu de développement.
- Sélection et verrouillage du prompt V1 avant l'évaluation finale.
- Génération des réponses aux 40 questions du jeu de test.
- Évaluation manuelle de la qualité et analyse des performances.
- Validation et sauvegarde des résultats définitifs.

### Prochaines étapes
1. Construire un corpus documentaire spécialisé en Computer Vision.
2. Développer et évaluer un pipeline RAG
3. Adapter le modèle initial avec LoRA/QLoRA.
4. Comparer la baseline, le RAG et le modèle fine-tuné.
5. Étudier l'optimisation par préférences (DPO), si réalisable.

## Évaluation du modèle initial

La baseline a été évaluée avec le modèle préentraîné
`Qwen/Qwen3-1.7B`, sans RAG ni fine-tuning.

**Configuration expérimentale :**

- Modèle : Qwen3-1.7B.
- Précision : FP16.
- Matériel : NVIDIA Tesla T4.
- Prompt : V1, sélectionné uniquement sur le jeu de développement.
- Génération : déterministe, sans mode thinking.
- Longueur maximale : 512 nouveaux tokens.
- Dataset : 50 questions (10 développement, 40 test).
- Domaines : filtrage, segmentation, morphologie, contours et inspection industrielle.

Chaque réponse du jeu de test a été évaluée selon :

- Exactitude technique : 0 à 3 points.
- Complétude : 0 à 2 points.
- Erreur factuelle : indicateur booléen indépendant.

Le score maximal est de 200 points.

## Résultats

### Performances globales sur le jeu de test

| Métrique | Résultat |
|---|---:|
| Questions évaluées | 40/40 |
| Qualité technique | 113/200 (56,5 %) |
| Réponses avec erreur factuelle signalée | 31/40 (77,5 %) |
| Générations atteignant la limite de tokens | 9/40 (22,5 %) |
| Latence moyenne de génération | 16,932 s |
| Débit global de génération | 24,10 tokens/s |
| Pic mémoire GPU PyTorch observé | 3,31 Gio |

### Qualité par domaine

| Domaine | Score | Qualité |
|---|---:|---:|
| Contours | 27/40 | 67,5 % |
| Filtrage | 26/40 | 65,0 % |
| Inspection industrielle | 25/40 | 62,5 % |
| Segmentation | 18/40 | 45,0 % |
| Morphologie | 17/40 | 42,5 % |

### Principales observations

- Le modèle préentraîné maîtrise plusieurs notions générales de traitement d'images.
- Des erreurs importantes persistent dans les fonctions OpenCV, les définitions mathématiques et les opérations morphologiques.
- La morphologie et la segmentation obtiennent les scores les plus faibles sur ce jeu de test.
- Une explication longue et fluide ne garantit pas son exactitude technique.
- Les générations atteignant la limite peuvent présenter des réponses incomplètes.

Ces observations concernent uniquement le jeu de test figé de 40 questions et ne constituent pas une estimation générale des performances du modèle.

## Reproductibilité

Les ressources d'évaluation comprennent :

- `data/evaluation/questions.jsonl` : questions et réponses de référence.
- `data/evaluation/freeze_manifest.json` : métadonnées et empreintes du dataset figé.
- `notebooks/01_baseline_evaluation.ipynb` : expériences et protocole d'évaluation de la baseline.
- `docs/evaluation_protocol.md` : protocole d'évaluation en anglais.
- `docs/evaluation_protocol.fr.md` : protocole d'évaluation en français.

Les réponses générées, les métriques et les notations manuelles ont été sauvegardées sur Google Drive dans :

`industrial-vision-slm-benchmark/outputs/baseline/`

Les résultats définitifs sont enregistrés dans :

- `baseline_test_v1_quality_scores_final.json`
- `baseline_test_v1_quality_summary.json`

La configuration verrouillée et le jeu de test seront conservés pour comparer les futures approches RAG et fine-tuning.

**Limites :** le jeu de test est relativement petit, les notes reposent sur une évaluation manuelle et les performances d'exécution dépendent de l'environnement matériel.


## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)
- [Notebook d'évaluation de la baseline](notebooks/01_baseline_evaluation.ipynb)