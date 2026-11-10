import json
import re
import sys

from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

# Répertoire racine du projet, déduit de ce fichier.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Chemin du dataset.
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"

# Champs obligatoires de chaque question.
REQUIRED_FIELDS = {
    "id", "split", "topic",
    "question", "reference_answer",
    "key_points", "source_url",
}

VALID_SPLITS = {"dev", "test"}


# ------------------------------------------------------------
# 2. Validation d'une question
# ------------------------------------------------------------
def validate_question(record, line_number):
    """Vérifier les champs et valeurs d'une question."""

    errors = []

    if not isinstance(record, dict):
        return [f"Ligne {line_number}: objet JSON attendu."]

    # Vérifier les champs obligatoires.
    missing = REQUIRED_FIELDS - record.keys()

    if missing:
        errors.append(f"Ligne {line_number}: champs manquants : {sorted(missing)}")
        return errors

    # Vérifier les champs textuels non vides.
    text_fields = [
        "id", "topic",
        "question", "reference_answer", "source_url",
    ]

    for field in text_fields:
        value = record[field]

        if not isinstance(value, str) or not value.strip():
            errors.append(f"Ligne {line_number}: '{field}' invalide.")

    # Vérifier le format des identifiants.
    if isinstance(record["id"], str):
        if not re.fullmatch(r"CV\d{3}", record["id"]):
            errors.append(f"Ligne {line_number}: ID attendu : CV001.")

    # Vérifier le split.
    if record["split"] not in ("dev", "test"):
        errors.append(f"Ligne {line_number}: split invalide.")

    # Vérifier les éléments techniques attendus.
    points = record["key_points"]

    if (not isinstance(points, list) or len(points) == 0 or any(not isinstance(p, str) or not p.strip() for p in points)):
        errors.append(f"Ligne {line_number}: key_points invalide.")

    # Vérifier la structure de l'URL.
    if isinstance(record["source_url"], str):
        url = urlparse(record["source_url"])

        if url.scheme != "https" or not url.netloc:
            errors.append(f"Ligne {line_number}: URL HTTPS invalide.")

    return errors


# ------------------------------------------------------------
# 3. Validation du fichier JSONL
# ------------------------------------------------------------
def main():

    print("=== Validation du dataset OpenCV ===\n")

    if not DATASET_PATH.is_file():
        print(f"[ERREUR] Fichier absent : {DATASET_PATH}")
        return 1

    errors = []
    records = []

    # Lire le fichier ligne par ligne.
    with DATASET_PATH.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Ligne {line_number}: JSON invalide : {exc}")
                continue

            # Vérifier le contenu.
            errors.extend(validate_question(record, line_number))

            if isinstance(record, dict):
                records.append(record)

    # Vérifier les identifiants en double.
    ids = [r["id"] for r in records if isinstance(r.get("id"), str)]

    for identifier, count in Counter(ids).items():
        if count > 1:
            errors.append(f"ID dupliqué : {identifier}")

    # Vérifier les questions identiques.
    questions = [r["question"].strip().casefold() for r in records if isinstance(r.get("question"), str)]

    if len(questions) != len(set(questions)):
        errors.append("Questions identiques détectées.")

    # Afficher les résultats.
    if errors:
        print("[ERREUR] Dataset invalide.\n")

        for error in errors:
            print("-", error)

        return 1

    splits = Counter(r["split"] for r in records)

    print(f"[OK] Questions valides : {len(records)}")
    print(f"[OK] Questions dev : {splits['dev']}")
    print(f"[OK] Questions test : {splits['test']}")
    print("[OK] Aucun ID dupliqué.")
    print("[OK] Aucune question identique.")

    print("\n[OK] Validation terminée avec succès.\n")

    return 0


# ------------------------------------------------------------
# 4. Point d'entrée du programme
# ------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())