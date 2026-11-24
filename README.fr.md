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

**Étape actuelle :** évaluation initiale de Qwen3-1.7B, préparation du corpus documentaire OpenCV, évaluation du moteur de recherche E5 + FAISS et construction du pipeline RAG terminées. Le Notebook 04 a été validé sur Google Colab en réutilisant les résultats expérimentaux existants.

**Prochaine phase :** adapter Qwen3-1.7B par fine-tuning supervisé avec LoRA/QLoRA, puis préparer une comparaison contrôlée entre le modèle initial, le RAG et les configurations adaptées.

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
- Construction du pipeline RAG associant Qwen3-1.7B au moteur E5 + FAISS.
- Intégration d'un contexte documentaire structuré avec citations `[S1]`, `[S2]`, etc.
- Définition et tokenisation des prompts RAG avec le mode thinking désactivé.
- Réalisation de 20 expériences sur les 10 questions de développement, en configurations Top-3 et Top-5.
- Sauvegarde persistante des générations et mécanisme de reprise sans régénération inutile.
- Évaluation humaine de 20 réponses pour l'exactitude factuelle et de 17 réponses contenant des citations pour la justification documentaire.
- Consolidation des résultats et comparaison appariée des configurations Top-3 et Top-5.
- Vérification des 80 passages documentaires associés aux évaluations.
- Validation du Notebook 04 sur Google Colab : 34 cellules Python exécutées sans erreur, avec réutilisation des 20 générations existantes.


### Prochaines étapes

1. Préparer le fine-tuning supervisé de Qwen3-1.7B avec LoRA/QLoRA.
2. Construire et évaluer les configurations adaptées du modèle.
3. Préparer une comparaison contrôlée entre la baseline, le RAG et les modèles adaptés, avec des questions et des critères d'évaluation comparables.
4. Étudier les compromis entre qualité des réponses, latence, consommation mémoire et coût d'inférence.
5. Explorer l'optimisation par préférences (DPO), si les ressources et les résultats précédents le permettent.


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

La configuration verrouillée et les résultats de la baseline sur les 40 questions de test sont conservés comme référence pour les comparaisons futures.

Les résultats RAG du Notebook 04 portent uniquement sur les dix questions de développement et reposent sur une grille d'évaluation distincte. Ils ne constituent donc pas encore une comparaison directe avec les résultats de la baseline sur le jeu de test.


**Limites :** le jeu de test est relativement petit, les notes reposent sur une évaluation manuelle et les performances d'exécution dépendent de l'environnement matériel.


## Préparation du corpus documentaire

Le projet comprend un corpus documentaire technique construit à partir de la documentation officielle OpenCV 4.13.0.

Ce corpus fournit les connaissances documentaires externes au pipeline RAG (Retrieval-Augmented Generation) implémenté dans le Notebook 04.

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
- L'évaluation générative RAG a été réalisée sur les dix questions de développement ; son évaluation sur les 40 questions de test reste à effectuer.

Le corpus a été encodé, indexé et évalué sur le jeu de développement. Les expériences montrent une bonne récupération des passages directement pertinents lorsqu'ils existent, mais révèlent également des limites de classement et de couverture documentaire.

L'intégration générative avec Qwen3-1.7B a été réalisée et évaluée dans le Notebook 04. Les résultats sont présentés dans la section consacrée au pipeline RAG.


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
- La qualité des réponses RAG a été évaluée dans le Notebook 04 uniquement sur le jeu de développement ; son évaluation sur le jeu de test reste à effectuer.

Les scores de similarité ne sont pas des probabilités de pertinence.

Les **40 questions de test réservées** n'ont pas été utilisées pour ajuster le retrieval.

### Reproductibilité

Les annotations de développement sont versionnées dans `data/evaluation/retrieval_qrels_dev.json`.

Les fichiers dérivés d'embeddings, les identifiants et l'index FAISS sont conservés localement dans `outputs/retrieval/`, exclu du versionnement Git.

Le notebook documente les contrôles SHA-256, les sauvegardes idempotentes, les recherches documentaires et les métriques d'évaluation.

**Intégration réalisée :** le moteur E5 + FAISS est utilisé par le pipeline RAG du Notebook 04. Les configurations Top-3 et Top-5 ont été évaluées sur le jeu de développement. Une comparaison directe avec la baseline et les futurs modèles adaptés reste à réaliser selon un protocole commun.


## Pipeline RAG — Construction et évaluation terminées

Le pipeline RAG est développé dans `notebooks/04_rag_evaluation.ipynb`.

Il combine le modèle génératif Qwen3-1.7B et le moteur de recherche sémantique E5 + FAISS pour produire des réponses techniques appuyées sur la documentation OpenCV.

### Architecture du système

Le pipeline comprend :

1. Réception d'une question technique en français.
2. Encodage de la question avec `intfloat/multilingual-e5-small`.
3. Recherche des passages pertinents dans l'index FAISS.
4. Construction d'un contexte documentaire avec références `[S1]`, `[S2]`, etc.
5. Construction et tokenisation du prompt RAG.
6. Génération de la réponse avec `Qwen/Qwen3-1.7B`, mode thinking désactivé.
7. Sauvegarde de la réponse, des paramètres et des références documentaires.
8. Évaluation humaine de l'exactitude factuelle et des justifications documentaires.

Le système réutilise les 55 chunks OpenCV, les embeddings de 384 dimensions et l'index `IndexFlatIP` préparés précédemment, sans modifier le corpus documentaire.

### Protocole expérimental

Les expériences utilisent les dix questions du jeu de développement, chacune évaluée avec deux configurations :

- **Top-3 :** trois passages documentaires récupérés.
- **Top-5 :** cinq passages documentaires récupérés.

Au total, 20 réponses RAG sont disponibles.

La notation humaine repose sur deux critères :

- **Exactitude factuelle :** note de 0 à 5.
- **Justification documentaire :** note de 0 à 2, uniquement pour les réponses contenant des citations.

Les évaluations factuelles sont réalisées sur les 20 réponses. Les évaluations documentaires concernent les 17 réponses contenant au moins une citation.

Les 80 associations entre expériences et passages récupérés sont conservées pour permettre la vérification des sources.

Les 40 questions du jeu de test n'ont pas été utilisées pour ajuster ce pipeline RAG.

### Résultats de l'expérience de référence

| Indicateur | Top-3 | Top-5 |
|---|---:|---:|
| Réponses évaluées | 10 | 10 |
| Score factuel total | 32/50 | 38/50 |
| Exactitude factuelle moyenne | 3,20/5 | 3,80/5 |
| Réponses contenant des citations | 8/10 | 9/10 |
| Justification documentaire moyenne | 1,00/2 | 1,44/2 |
| Justifications insuffisantes (0/2) | 1 | 0 |
| Justifications partielles (1/2) | 6 | 5 |
| Justifications complètes (2/2) | 1 | 4 |

Les moyennes documentaires sont calculées uniquement sur les réponses contenant des citations. Les trois réponses sans citation ne reçoivent pas automatiquement la note 0/2.

Sur les dix questions étudiées, le Top-5 améliore la note factuelle pour cinq questions et obtient une note identique sur les cinq autres.

La moyenne factuelle passe de 3,20/5 à 3,80/5. La moyenne documentaire passe de 1,00/2 à 1,44/2.

Ces observations concernent exclusivement l'expérience de référence et ne démontrent pas une supériorité statistiquement généralisable du Top-5.

### Sauvegarde et reproductibilité

Le notebook utilise trois modes d'exécution :

- `auto` : réutilise les vingt réponses si elles sont disponibles ; sinon, génère les réponses manquantes.
- `reuse` : exploite uniquement les générations existantes.
- `generate_missing` : conserve les résultats existants et génère les réponses absentes.

Une génération nécessite un GPU CUDA compatible. La réutilisation des résultats existants ne nécessite pas de charger les poids du modèle génératif sur GPU.

Les résultats sont sauvegardés dans :

`outputs/rag/dev_runs/`

Les fichiers d'évaluation sont conservés dans :

`outputs/rag/analysis/`

Les principaux fichiers d'annotation sont :

- `rag_human_review_dev.csv` : grille persistante d'évaluation humaine.
- `rag_review_mapping_dev.json` : correspondances entre évaluations et expériences.
- `rag_source_evidence_dev.csv` : passages documentaires associés aux réponses.

Les annotations humaines ne sont jamais attribuées automatiquement. Les statistiques correspondantes sont calculées uniquement lorsque les évaluations requises sont disponibles.

Les fichiers de génération et d'annotation existants sont préservés lors des reprises compatibles.

Le Notebook 04 a été validé sur Google Colab en mode reprise :

- 34 cellules Python exécutées sans erreur.
- 20 générations RAG réutilisées.
- Aucune nouvelle génération nécessaire.
- 20 évaluations factuelles vérifiées.
- 17 évaluations documentaires vérifiées.
- Contrôle final de la section 5 réussi.

Une exécution complète depuis zéro sur une installation indépendante n'a pas encore été validée.

Les neuf artefacts documentaires nécessaires doivent être disponibles avant l'exécution. Ils peuvent être préparés à partir des notebooks précédents ou fournis au moyen de l'archive `rag_colab_assets.zip`.

Les résultats générés et les annotations sont conservés dans les répertoires de sortie persistants. Leur disponibilité sur un autre environnement dépend de la récupération de ces fichiers, qui ne sont pas nécessairement versionnés dans Git.

### Limites et perspectives

- Évaluation limitée à dix questions de développement.
- Corpus documentaire restreint à sept pages OpenCV.
- Notation humaine sans validation inter-annotateurs indépendante.
- Qualité des citations parfois insuffisante ou partielle.
- Absence de comparaison directe avec la baseline sur les mêmes questions et selon les mêmes critères.
- Coût d'inférence et consommation mémoire non comparés systématiquement entre Top-3 et Top-5.
- Aucune évaluation RAG réalisée sur les 40 questions du jeu de test.

La prochaine phase consistera à adapter Qwen3-1.7B avec LoRA/QLoRA, puis à préparer une comparaison contrôlée avec les configurations de référence.

## Documentation

- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Protocole d'évaluation](docs/evaluation_protocol.fr.md)
- [Notebook d'évaluation de la baseline](notebooks/01_baseline_evaluation.ipynb)
- [Notebook de préparation du corpus documentaire](notebooks/02_document_corpus.ipynb)
- [Notebook de recherche sémantique et évaluation](notebooks/03_retrieval_evaluation.ipynb)
- [Notebook de construction et d'évaluation du pipeline RAG](notebooks/04_rag_evaluation.ipynb)
