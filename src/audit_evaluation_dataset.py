"""
Audit du dataset d'évaluation OpenCV.

Contrôle :
- le nombre de questions ;
- la couverture des cinq domaines ;
- la répartition dev/test ;
- les identifiants attendus ;
- l'origine des références ;
- les ressemblances potentielles entre questions dev et test.
"""

import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "evaluation" / "questions.jsonl"

EXPECTED_TOPICS = {"filtering", "segmentation", "morphology", "edges", "industrial_inspection",}
EXPECTED_IDS = {
    f"CV{i:03d}" for i in range(1, 51)
}

def main():
    # Chargement des questions JSONL
    with DATASET.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line) for line in file if line.strip()
        ]

    errors = []
    warnings = []

    # Vérifier le volume et les identifiants
    ids = {record["id"] for record in records}

    if len(records) != 50:
        errors.append("Le dataset doit contenir 50 questions.")

    if ids != EXPECTED_IDS:
        errors.append("Les ids CV001-CV-050 sont incomplets.")

    # Compter les questions par domaine et split
    counts = Counter(
        (record["topic"], record["split"]) for record in records
    )

    topics = {record["topic"] for record in records}

    if topics != EXPECTED_TOPICS:
        errors.append("Liste des domaines incorrecte.")

    print("=== Audit du dataset OpenCV ===\n")
    for topic in sorted(EXPECTED_TOPICS):
        dev_count = counts[topic, "dev"]
        test_count = counts[topic, "test"]

        print(f"{topic:<24} dev={dev_count} | test={test_count}")

        if dev_count != 2 or test_count != 8:
            errors.append(f"{topic} : répartition dev/test incorrecte.")

    # Examiner les références documentaires
    versions = Counter()

    for record in records:
        url = urlparse(record["source_url"])

        if url.scheme != "https" or url.hostname != "docs.opencv.org":
            errors.append(f"{record["id"]} : source non officielle.")
            continue

        parts = url.path.strip("/").split("/")
        version = parts[0] if parts else "inconnue"

        versions[version] += 1

    print("\nVersions des sources :")
    for version, count in sorted(versions.items()):
        print(f"    {version} : {count} questions")

    if len(versions) > 1:
        warnings.append("Plusieurs versions documentaires sont utilisées.")

    # Repérer les formulations très proches entre dev et test.
    dev_records = [r for r in records if r["split"] == "dev"]
    test_records = [r for r in records if r["split"] == "test"]

    similar_pairs = []

    for dev in dev_records:
        for test in test_records:
            score = SequenceMatcher(None, dev["question"].lower(), test["question"].lower()).ratio()

            if score >= 0.80:
                similar_pairs.append(
                    (dev["id"], test["id"], score)
                )

    print("\nQuestions potentiellement similaires :")
    if similar_pairs:
        for dev_id, test_id, score in similar_pairs:
            print(f"    {dev_id} / {test_id} : {score:.2f}")

        warnings.append("Des questions proches nécessitent une revue manuelle.")
    else:
        print(" Aucune paire détectée au seuil de 0.80.")

    # Bilan de l'audit
    print("\n=== Résultat ===")

    for warning in warnings:
        print(f"[AVERTISSEMENT] {warning}")

    for error in errors:
        print(f"[ERREUR] {error}")

    if errors:
        print("\n[ECHEC] Audit non validé.")
        return 1

    print("[OK] Contrôles structurels et de répartition réussis.")
    print("Revue scientifique des réponses encore nécessaire.\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())