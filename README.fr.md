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

Les questions de test ne doivent pas être utilisées pour entraîner le modèle, optimiser les prompts ou sélectionner les hyperparamètres.

Ces contrôles garantissent la cohérence structurelle et permettent de détecter les modifications du dataset figé. Ils ne garantissent pas, à eux seuls, l'exactitude scientifique de chaque réponse de référence.

## État du projet

**Étape actuelle :** évaluation initiale, construction du corpus documentaire et évaluation du moteur de recherche sémantique terminées.

**Prochaine phase :** intégrer les passages retrouvés et leurs citations à Qwen3-1.7B pour construire et évaluer le pipeline RAG.

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
- Construction et validation du corpus documentaire OpenCV.
- Génération de 62 chunks et sélection de 55 chunks pour le RAG.
- Contrôles d'intégrité et rapport de reproductibilité du corpus.
- Génération et sauvegarde de 55 embeddings multilingues E5 de 384 dimensions.
- Traitement documenté de deux chunks trop longs par fenêtres de tokens et agrégation des embeddings.
- Recherche exacte `IndexFlatIP` avec FAISS, comparée à NumPy et sauvegardée avec contrôles d'intégrité.
- Recherche Top-k paramétrable et traçabilité des sources OpenCV.
- Chargement des 10 questions de développement ; jeu test de 40 questions réservé à l'évaluation finale.
- Annotation de 10 questions de développement sur 55 chunks, soit 550 jugements de pertinence.
- Évaluation quantitative du retrieval avec Precision@k, Recall@k et MRR pour k = 1, 3 et 5.
- Analyse des erreurs de classement et identification des limites de couverture documentaire.

### Prochaines étapes
1. Intégrer le retrieval et les citations documentaires à Qwen3-1.7B.
2. Évaluer le RAG et le comparer à la baseline figée.
3. Adapter le modèle avec LoRA/QLoRA.
4. Comparer la baseline, le RAG et les configurations adaptées.
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


## Préparation du corpus documentaire

Le projet comprend un corpus documentaire technique construit à partir de la documentation officielle OpenCV 4.13.0.

Ce corpus a été conçu pour fournir des connaissances externes au futur système RAG
(Retrieval-Augmented Generation).

### Sources documentaires

Sept pages de documentation OpenCV ont été collectées, couvrant cinq domaines techniques :

- Filtrage d'images
- Morphologie mathématique
- Seuillage et segmentation
- Détection et analyse de contours
- Inspection visuelle industrielle

Le registre documentaire conserve les identifiants, les URLs originales, les chemins des fichiers traités et leurs empreintes SHA-256.

### Pipeline de préparation

Le traitement documentaire comprend :

1. Téléchargement et vérification des documents HTML.
2. Extraction du contenu technique avec Beautiful Soup.
3. Conversion du HTML en Markdown structuré.
4. Préservation des titres, formules mathématiques et exemples de code Python.
5. Découpage hiérarchique basé sur les sections Markdown.
6. Contrôle d'intégrité et de conservation du contenu.
7. Filtrage des éléments de navigation, exercices et références bibliographiques isolées.

Les illustrations absentes sont explicitement signalées,
sans générer de descriptions inventées.

### Statistiques du corpus

| Indicateur | Résultat |
|---|---:|
| Documents sources | 7 |
| Domaines techniques | 5 |
| Caractères extraits | 42 335 |
| Blocs de code Python conservés | 26 |
| Chunks originaux | 62 |
| Blocs élémentaires conservés | 228/228 |
| Chunks exclus | 7 |
| Chunks retenus pour le RAG | 55 |

### Configuration du chunking

- Stratégie : découpage hiérarchique par sections.
- Taille maximale : 1 500 caractères.
- Chevauchement : 0 caractère.
- Préservation du code et des expressions mathématiques.
- Répétition du titre de section dans chaque chunk.

### Répartition du corpus RAG

| Domaine | Chunks |
|---|---:|
| Contours | 23 |
| Filtrage | 7 |
| Inspection industrielle | 6 |
| Morphologie | 10 |
| Segmentation | 9 |
| **Total** | **55** |

Les chunks retenus ont une longueur moyenne de 757,5 caractères et une longueur médiane de 808 caractères.

### Reproductibilité

Le pipeline documentaire est implémenté dans :

`notebooks/02_document_corpus.ipynb`

Les principaux fichiers de métadonnées sont :

- `data/corpus/source_registry.json`
- `data/corpus/chunk_manifest.json`
- `data/corpus/retrieval_manifest.json`
- `data/corpus/corpus_report.json`

Le pipeline génère également :

- `data/corpus/chunks/chunks.jsonl`
- `data/corpus/chunks/retrieval_chunks.jsonl`

Les contenus documentaires générés restent locaux lorsqu'ils sont exclus par `.gitignore`. Le notebook et les métadonnées décrivent leur préparation.

Les contrôles SHA-256 permettent de vérifier l'intégrité
des documents sources, des documents traités et des
corpus produits.

### Limites

- Le corpus contient seulement sept pages OpenCV.
- La couverture documentaire varie selon les domaines.
- Les illustrations ne sont pas intégrées au corpus textuel.
- La taille des chunks est mesurée en caractères, et non en tokens.
- Le filtrage repose sur des règles structurelles simples.
- Le retrieval a été évalué sur dix questions de développement, mais ses performances sur le jeu de test réservé ne sont pas encore mesurées.
- Aucun résultat d'inférence RAG n'est encore disponible.

Le corpus a été encodé, indexé et évalué sur le jeu de développement. Les expériences montrent une bonne récupération des passages directement pertinents lorsqu'ils existent, mais révèlent également des limites de classement et de couverture documentaire.

L'intégration générative avec Qwen3-1.7B reste à réaliser.


## Moteur de recherche sémantique (évaluation de développement terminée)

Le pipeline de recherche est développé dans `notebooks/03_retrieval_evaluation.ipynb`.

Il interroge les **55 chunks OpenCV sélectionnés**, à l'aide d'un encodeur multilingue et d'un index vectoriel exact.

### Pipeline implémenté

1. Encoder les questions françaises et les passages OpenCV anglais avec `intfloat/multilingual-e5-small`.
2. Appliquer les préfixes `query: ` et `passage: ` requis, puis normaliser les vecteurs.
3. Générer **55 embeddings de 384 dimensions** sur CPU.
4. Traiter deux passages dépassant 512 tokens par fenêtres et moyenne pondérée de leurs embeddings, sans modifier le corpus d'origine.
5. Construire un index FAISS exact `IndexFlatIP` et vérifier les scores et le classement Top-3 par comparaison avec NumPy.
6. Sauvegarder puis recharger les embeddings, identifiants et l'index avec des contrôles SHA-256 et des écritures idempotentes.
7. Implémenter une recherche Top-k réutilisable, avec titres, sections, URLs OpenCV officielles et citations de type `[S1]`.

**Vérifications exploratoires :** les recherches Top-1, Top-3 et Top-5 ont été testées sur trois questions techniques (neuf recherches au total). Les classements sont stables entre les valeurs de k.

Ces contrôles exploratoires valident la cohérence technique de la recherche. La pertinence documentaire a ensuite été mesurée quantitativement sur le jeu de développement.

### Limites de classement observées

Certains passages classés premiers ne répondent qu'indirectement à la question :

| Question exploratoire | Première section retrouvée | Similarité |
|---|---|---:|
| Érosion morphologique | Morphology — Theory | 0,8508 |
| Filtrage gaussien avant Canny | Filtering — Bilateral Filtering | 0,8457 |
| Template matching | Contours Hierarchy — RETR_CCOMP | 0,8281 |

Les scores de similarité **ne sont pas des probabilités de pertinence**.


### Évaluation quantitative du retrieval

Le jeu d'évaluation figé contient **50 questions** : **10 de développement** et **40 de test réservé**.

Les dix questions de développement ont été annotées sur les 55 chunks du corpus, soit **550 jugements** de pertinence, enregistrés dans `data/evaluation/retrieval_qrels_dev.json`.

Deux conventions sont utilisées :

- **Principale :** notes 1 et 2 pertinentes.
- **Stricte :** seule la note 2 est pertinente.

Les métriques sont évaluées pour k = 1, 3 et 5.

| Indicateur | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Precision globale — principale | 0,5000 | 0,3000 | 0,2400 |
| Recall macro conditionnel — principal | 0,3021 | 0,4583 | 0,6250 |
| MRR conditionnel — principal | 0,6250 | 0,7292 | 0,7604 |
| Precision globale — stricte | 0,4000 | 0,2000 | 0,1400 |
| Recall macro conditionnel — strict | 0,5714 | 0,8571 | 1,0000 |
| MRR conditionnel — strict | 0,5714 | 0,6905 | 0,7262 |

**Couverture documentaire :**

- Convention principale : 8 questions couvertes sur 10.
- Convention stricte : 7 questions couvertes sur 10.
- CV008 : passages partiellement pertinents uniquement.
- CV009 et CV010 : aucun passage pertinent identifié.

À k = 5, les **7 passages directement pertinents** sont retrouvés pour les questions qui en disposent.

Selon la convention principale, **12 des 21 couples question-chunk pertinents** sont retrouvés dans les Top-5, soit un rappel micro de 57,14 %.

Les moyennes conditionnelles de Recall et de MRR excluent les questions sans passage pertinent. La précision globale inclut les dix questions.

### Analyse des erreurs et limites

L'analyse qualitative montre plusieurs erreurs de classement :

- CV001 : la réponse directement pertinente apparaît au rang 2.
- CV002 : la réponse directement pertinente apparaît au rang 3.
- CV007 : l'explication de l'hystérésis de Canny apparaît au rang 4, derrière des passages moins pertinents.

Ces résultats soulignent le compromis entre le rappel documentaire et le bruit introduit lorsque le nombre de passages récupérés augmente.

**Limites :**

- Évaluation réalisée sur seulement dix questions.
- Jugements humains sans validation inter-annotateurs indépendante.
- Couverture incomplète du corpus.
- Effet de l'agrégation des embeddings longs non étudié séparément.
- Aucune qualité de réponse RAG encore mesurée.

Les scores de similarité ne sont pas des probabilités de pertinence.

Les **40 questions de test réservées** n'ont pas été utilisées pour ajuster le retrieval.

### Reproductibilité

Les annotations de développement sont versionnées dans `data/evaluation/retrieval_qrels_dev.json`.

Les fichiers dérivés d'embeddings, les identifiants et l'index FAISS sont conservés localement dans `outputs/retrieval/`, exclu du versionnement Git.

Le notebook documente les contrôles SHA-256, les sauvegardes idempotentes, les recherches documentaires et les métriques d'évaluation.

**Prochaine étape :** intégrer ce moteur à Qwen3-1.7B, puis comparer les réponses générées avec et sans contexte documentaire.


## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)
- [Notebook d'évaluation de la baseline](notebooks/01_baseline_evaluation.ipynb)
- [Notebook de préparation du corpus documentaire](notebooks/02_document_corpus.ipynb)
- [Notebook de recherche sémantique et évaluation](notebooks/03_retrieval_evaluation.ipynb)
