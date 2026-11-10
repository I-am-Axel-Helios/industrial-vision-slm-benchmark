import unittest

from src.validate_evaluation_dataset import validate_question

class TestEvaluationVaidator(unittest.TestCase):
    """Tester la validation des questions d'évaluation"""

    def setUp(self):
        # Créer une question valide avant chaque test
        self.record = {
            "id": "CV001",
            "split": "dev",
            "topic": "morphology",
            "question": "Qu'est-ce qu'une ouverture morphologique ?",
            "reference_answer": "Une érosion suivie d'une dilatation.",
            "key_points": ["érosion", "dilatation"],
            "source_url": "https://docs.opencv.org/"
        }

    # Test 1 - Une question valide doit être acceptée
    def test_valid_question(self):
        errors = validate_question(self.record, 1)
        self.assertEqual(errors, [])

    # Test 2 - Un champ obligatoire manquant doit être détecté
    def test_missing_field(self):
        del self.record["question"]

        errors = validate_question(self.record, 1)
        self.assertTrue(errors)

    # Test 3 - Un identifiant incorrect doit être rejeté
    def test_invalid_identifier(self):
        self.record["id"] = "INVALID"

        errors = validate_question(self.record, 1)
        self.assertTrue(errors)

    # Test 4 - Une valeur de split incorrecte doit être rejetée
    def test_invalid_split(self):
        self.record["split"] = "training"

        errors = validate_question(self.record, 1)
        self.assertTrue(errors)

    # Test 5 - Une liste key_points vide doit être rejetée
    def test_empty_key_points(self):
        self.record["key_points"] = []

        errors = validate_question(self.record, 1)
        self.assertTrue(errors)

    # Test 6 - Une URL invalide doit être rejetée
    def test_invalid_source_url(self):
        self.record["source_url"] = "not-a-valid-url"

        errors = validate_question(self.record, 3)
        self.assertTrue(errors)

if __name__ == "__main__":
    unittest.main()