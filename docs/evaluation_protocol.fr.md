# Protocole d'évaluation — Industrial Vision SLM Benchmark

[English](evaluation_protocol.md) | [Français](evaluation_protocol.fr.md)

## 1. Objectif

Évaluer la fiabilité technique de Qwen3-1.7B dans
les domaines du traitement d'images, d'OpenCV
et de la vision industrielle.

Le même jeu d'évaluation sera utilisé pour comparer
le modèle initial, le RAG, le SFT et éventuellement
les systèmes optimisés par DPO.

## 2. Jeu de données d'évaluation

Dataset : `data/evaluation/questions.jsonl`

Chaque question contient :
- id
- split (dev ou test)
- topic
- question
- reference_answer
- key_points
- source_url

Les questions de développement servent à tester
et valider le pipeline d'évaluation.

Les questions de test sont exclues de l'entraînement,
de l'optimisation des prompts et de la sélection
des hyperparamètres.

## 3. Critères d'évaluation

### 3.1. Exactitude factuelle (0–3)

- 3 : Réponse techniquement correcte.
- 2 : Réponse globalement correcte, avec des imprécisions mineures.
- 1 : Réponse partiellement correcte, avec des erreurs
  ou des lacunes importantes.
- 0 : Réponse incorrecte, hors sujet ou absente.

### 3.2. Complétude (0–2)

- 2 : Tous les éléments essentiels sont couverts.
- 1 : Certains éléments essentiels manquent.
- 0 : Les éléments essentiels sont absents.

### 3.3. Hallucination

Indicateur booléen :
- true : Au moins une affirmation technique fausse
  ou inventée est identifiée.
- false : Aucune affirmation de ce type n'est identifiée.

Une information manquante ne constitue pas,
à elle seule, une hallucination.

### 3.4. Fidélité au contexte (0–2)

Critère applicable uniquement aux systèmes RAG.

- 2 : Les affirmations importantes sont soutenues
  par le contexte documentaire récupéré.
- 1 : Certaines affirmations importantes ne sont
  pas suffisamment soutenues.
- 0 : Les affirmations principales ne sont pas
  étayées ou sont contredites par le contexte.

Pour les systèmes sans RAG : N/A.

## 4. Métriques agrégées

Score de qualité = exactitude factuelle + complétude.

Score maximal de qualité = 5.

Métriques à présenter :
- Score moyen de qualité.
- Exactitude factuelle moyenne.
- Complétude moyenne.
- Taux d'hallucination.
- Fidélité moyenne au contexte pour le RAG.
- Exactitude des citations pour le RAG.
- Latence médiane d'inférence.
- Débit de génération.
- Pic de mémoire GPU.

## 5. Règles expérimentales

- Utiliser le même jeu de test réservé.
- Conserver des paramètres de génération cohérents
  entre les différentes configurations.
- Ne pas entraîner les modèles sur les questions
  de test ou leurs réponses de référence.
- Ne pas utiliser les résultats du test pour
  sélectionner les prompts ou les modèles.
- Vérifier les réponses de référence à partir
  de sources documentaires.
- Ne pas assimiler automatiquement la similarité
  sémantique à l'exactitude factuelle.
- Documenter les évaluations manuelles incertaines.
- Présenter les limites et la taille des échantillons.

## 6. État expérimental

Protocole défini.

Construction complète du jeu d'évaluation
et benchmark comparatif à venir.
