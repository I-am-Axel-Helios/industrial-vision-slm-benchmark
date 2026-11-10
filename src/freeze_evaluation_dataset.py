"""
Figer et vérifier le contenu du dataset d'évaluation OpenCV.

Modes :
    freeze : créer le manifeste de référence.
    verify : comparer le dataset avec sa version figée.
"""

import argparse
import hashlib
import json
import sys

from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "evaluation" / "questions.jsonl"
MANIFEST = ROOT / "data" / "evaluation" / "freeze_manifest.json"

TOPICS = {"filtering", "segmentation", "morphology", "edges", "industrial_inspection",}

def content_sha256(records):
    """Calculer une empreinte stable du contunu JSON"""

    # Tri des questions par identifiant
    records = sorted(records, key=lambda record: record["id"])

    # Normalisation du JSON pour éviter des différences liées aux espaces,
    # à l'ordre des clés ou à l'OS 
    normalized = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",",":"))
        for record in records
    ) + "\n"

    # Calcul de l'empreinte cryptographique
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def build_manifest():
    """Contrôler le dataset & construire son manifeste."""

    records = [
        json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    ids = [record["id"] for record in records]

    # Vérif des identifiants
    assert len(records) == 50, "50 questions attendues."
    assert len(set(ids)) == 50, "Identifiants dupliqués."

    # Vérif de la répartition
    counts = Counter(
        (record["topic"], record["split"]) for record in records
    )

    assert {record["topic"] for record in records} == TOPICS, "Domaines incorrects."

    for topic in TOPICS:
        assert counts[(topic, "dev")] == 2
        assert counts[(topic, "test")] == 8

    # Vérif de l'harmonisation des références
    assert all(
        record["source_url"].startswith("https://docs.opencv.org/4.13.0/")
        for record in records
    ), "Des références ne sont pas en OpenCV 4.13.0."

    # Extraction des questions réservées au test
    tests = [record for record in records if record["split"] == "test"]

    # Construction des métadonnées reproductibles
    return {
        "schema_version": 1,
        "dataset": "data/evaluation/questions.jsonl",
        "refernce_documentation": "OpenCV 4.13.0",
        "question_count": len(records),
        "dev_count": len(records) - len(tests),
        "test_count": len(tests),
        "dataset_content_sha256": content_sha256(records),
        "test_content_sha256": content_sha256(tests),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("freeze", "verify"),
        help="Créer ou vérifier le manifeste."
    )

    args = parser.parse_args()

    try:
        current = build_manifest()

        if args.mode == "freeze":
            # Empêcher un remplacement accidentel du manifeste
            if MANIFEST.exists():
                print("[ERREUR] Le manifeste existe déjà. Utilise le mode verify.")
                return 1

            MANIFEST.write_text(
                json.dumps(current, ensure_ascii=False, indent=2) + "\n", 
                encoding="utf-8"
            )

            print("[OK] Manifeste créé :", MANIFEST)

        else:
            if not MANIFEST.exists():
                print("[ERREUR] Aucun manifeste à vérifier")
                return 1

            frozen = json.loads(
                MANIFEST.read_text(encoding="utf-8")
            )

            if frozen != current:
                print("[ERREUR] Le dataset diffère de la version figée.")
                return 1

            print("[OK] Dataset identique à la version figée.")

        print(
            "Dataset SHA-256 :",
            current["dataset_content_sha256"]
        )

        print(
            "Test SHA-256    :",
            current["test_content_sha256"]
        )

        return 0

    except(AssertionError, KeyError, OSError, ValueError) as error:
        print("[ERREUR]", error)

if __name__ == "__main__":
    sys.exit(main())
    