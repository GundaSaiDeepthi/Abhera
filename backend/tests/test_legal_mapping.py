"""
Unit and Integration Tests for Legal Mapping Engine.

Verifies correct database retrieval of legal provisions from PostgreSQL, multi-label mapping,
deduplication, value preservation, and strict anti-hallucination fallback rules.
"""

import unittest
from typing import List

from app.database import SessionLocal
from app.models.law import Law
from app.services.legal_mapping import LegalMappingService, map_legal_provisions, NO_MATCH_MESSAGE


class TestLegalMappingEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()
        cls.service = LegalMappingService(db=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # 1. Known matching provision
    def test_01_matching_provision(self):
        res = self.service.get_legal_provisions(predicted_labels=["Domestic Violence"])
        self.assertTrue(res["matched"])
        self.assertGreater(len(res["legal_information"]), 0)
        
        first = res["legal_information"][0]
        self.assertIn("act", first)
        self.assertIn("section", first)
        self.assertIn("description", first)
        self.assertEqual(first["source"], "database")
        self.assertEqual(first["applicable_label"], "DV")

    # 2. Multiple matching provisions for a single category
    def test_02_multiple_matching_provisions(self):
        # Domestic Violence has 8 matching law records in data/laws.csv (excluding DOWRY_ACT which has DOWRY label)
        res = self.service.get_legal_provisions(predicted_labels=["DV"])
        self.assertTrue(res["matched"])
        self.assertEqual(len(res["legal_information"]), 8)

        # DV + DOWRY together return 10 matching law records
        res_multi = self.service.get_legal_provisions(predicted_labels=["DV", "DOWRY"])
        self.assertTrue(res_multi["matched"])
        self.assertEqual(len(res_multi["legal_information"]), 10)

    # 3. Multiple predicted labels
    def test_03_multiple_predicted_labels(self):
        res = self.service.get_legal_provisions(predicted_labels=["Domestic Violence", "Workplace Harassment"])
        self.assertTrue(res["matched"])
        
        labels_retrieved = set(item["applicable_label"] for item in res["legal_information"])
        self.assertIn("DV", labels_retrieved)
        self.assertIn("WH", labels_retrieved)

    # 4. No matching provision fallback
    def test_04_no_matching_provision(self):
        res = self.service.get_legal_provisions(predicted_labels=["NON_EXISTENT_INCIDENT"])
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["legal_information"]), 0)
        self.assertEqual(res["message"], NO_MATCH_MESSAGE)
        self.assertEqual(res["message"], "Information not available in the provided knowledge base.")

    # 5. Invalid label types
    def test_05_invalid_label(self):
        res1 = self.service.get_legal_provisions(predicted_labels=None)
        self.assertFalse(res1["matched"])
        self.assertEqual(res1["message"], "Information not available in the provided knowledge base.")

        res2 = self.service.get_legal_provisions(predicted_labels=[12345, None, ""])
        self.assertFalse(res2["matched"])
        self.assertEqual(res2["message"], "Information not available in the provided knowledge base.")

    # 6. Empty labels list
    def test_06_empty_labels(self):
        res = self.service.get_legal_provisions(predicted_labels=[])
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["legal_information"]), 0)
        self.assertEqual(res["message"], "Information not available in the provided knowledge base.")

    # 7. Duplicate predicted labels
    def test_07_duplicate_labels(self):
        res_single = self.service.get_legal_provisions(predicted_labels=["Domestic Violence"])
        res_dup = self.service.get_legal_provisions(predicted_labels=["DV", "Domestic Violence", "DV"])
        
        self.assertEqual(len(res_single["legal_information"]), len(res_dup["legal_information"]))

    # 8. Database values are strictly preserved
    def test_08_database_values_are_preserved(self):
        res = map_legal_provisions(predicted_labels=["Sexual Harassment"], db=self.db)
        self.assertTrue(res["matched"])
        
        # Query database directly to verify exact value preservation
        db_records = self.db.query(Law).filter(Law.applicable_label == "SH").order_by(Law.id.asc()).all()
        self.assertEqual(len(res["legal_information"]), len(db_records))

        for ret_item, db_item in zip(res["legal_information"], db_records):
            self.assertEqual(ret_item["act"], db_item.act_name)
            self.assertEqual(ret_item["section"], str(db_item.section_number))
            self.assertEqual(ret_item["description"], db_item.section_text)

    # 9. EXPLICIT ANTI-HALLUCINATION TEST
    def test_09_anti_hallucination(self):
        """
        Anti-Hallucination Rule Verification:
        Must NOT return guessed Acts, Sections, laws, or descriptions for unmapped categories.
        MUST return exact fallback message.
        """
        res = map_legal_provisions(predicted_labels=["UNKNOWN_CATEGORY"], db=self.db)
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["legal_information"]), 0)
        self.assertNotIn("act", res)
        self.assertNotIn("section", res)
        self.assertEqual(res["message"], "Information not available in the provided knowledge base.")


if __name__ == "__main__":
    unittest.main()
