"""
Unit and Integration Tests for Anti-Hallucination Safety Layer.

Verifies strict gating and rejection of fake/hallucinated legal and support service items,
exact fallback message behavior, value preservation, and explicit hallucination attack resistance.
"""

import unittest

from app.database import SessionLocal
from app.services.anti_hallucination import AntiHallucinationService, validate_payload
from app.services.legal_mapping import LegalMappingService, NO_MATCH_MESSAGE as LEGAL_NO_MATCH_MESSAGE
from app.services.support_mapping import SupportMappingService, NO_MATCH_MESSAGE as SUPPORT_NO_MATCH_MESSAGE


class TestAntiHallucinationLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()
        cls.service = AntiHallucinationService(db=cls.db)
        cls.legal_service = LegalMappingService(db=cls.db)
        cls.support_service = SupportMappingService(db=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # 1. Valid legal record is accepted
    def test_01_valid_legal_record_is_accepted(self):
        legal_res = self.legal_service.get_legal_provisions(predicted_labels=["Domestic Violence"])
        valid_item = legal_res["legal_information"][0]
        
        is_valid, reason = self.service.validate_legal_record(valid_item, self.db)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    # 2. Valid support record is accepted
    def test_02_valid_support_record_is_accepted(self):
        support_res = self.support_service.get_support_services(predicted_labels=["Domestic Violence"])
        valid_item = support_res["support_services"][0]
        
        is_valid, reason = self.service.validate_support_record(valid_item, self.db)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    # 3. Fabricated legal record is rejected
    def test_03_fabricated_legal_record_is_rejected(self):
        fake_legal_item = {
            "act": "Fake Women's Safety Act",
            "section": "999",
            "description": "This is a fabricated legal provision.",
            "applicable_label": "DV",
            "source": "model_guess",
        }
        is_valid, reason = self.service.validate_legal_record(fake_legal_item, self.db)
        self.assertFalse(is_valid)
        self.assertIsNotNone(reason)
        self.assertIn("Fabricated legal record rejected", reason)

    # 4. Fabricated support record is rejected
    def test_04_fabricated_support_record_is_rejected(self):
        fake_support_item = {
            "service_name": "Fake Women's Helpline",
            "service_type": "Helpline",
            "contact_number": "9999999999",
            "state": "Delhi",
            "district": "New Delhi",
            "applicable_label": "DV",
            "source": "model_guess",
        }
        is_valid, reason = self.service.validate_support_record(fake_support_item, self.db)
        self.assertFalse(is_valid)
        self.assertIsNotNone(reason)
        self.assertIn("Fabricated support service rejected", reason)

    # 5. Unknown category does not generate law
    def test_05_unknown_category_does_not_generate_law(self):
        res = self.service.validate_and_gate_results(
            predicted_labels=["UNKNOWN_CATEGORY"],
            db_session=self.db,
        )
        self.assertFalse(res["legal_matched"])
        self.assertEqual(len(res["legal_information"]), 0)
        self.assertEqual(res["legal_message"], LEGAL_NO_MATCH_MESSAGE)
        self.assertEqual(res["legal_message"], "Information not available in the provided knowledge base.")

    # 6. Unknown category does not generate support
    def test_06_unknown_category_does_not_generate_support(self):
        res = self.service.validate_and_gate_results(
            predicted_labels=["UNKNOWN_CATEGORY"],
            db_session=self.db,
        )
        self.assertFalse(res["support_matched"])
        self.assertEqual(len(res["support_services"]), 0)
        self.assertEqual(res["support_message"], SUPPORT_NO_MATCH_MESSAGE)
        self.assertEqual(res["support_message"], "No matching support service was found in the available support-services database.")

    # 7. Missing legal data is not filled
    def test_07_missing_legal_data_is_not_filled(self):
        incomplete_item = {"act": "BNS", "section": "85", "description": ""}
        is_valid, reason = self.service.validate_legal_record(incomplete_item, self.db)
        self.assertFalse(is_valid)
        self.assertIn("Missing or empty", reason)

    # 8. Missing support contact is not filled
    def test_08_missing_support_contact_is_not_filled(self):
        incomplete_item = {
            "service_name": "National Commission for Women Helpline",
            "service_type": "Helpline",
            "contact_number": "",
            "state": "National",
            "district": "All India",
        }
        is_valid, reason = self.service.validate_support_record(incomplete_item, self.db)
        self.assertFalse(is_valid)
        self.assertIn("Missing or empty", reason)

    # 9. Multiple labels independently validated
    def test_09_multiple_labels(self):
        res = validate_payload(predicted_labels=["DV", "WH"], db=self.db)
        self.assertTrue(res["legal_matched"])
        self.assertTrue(res["support_matched"])
        
        self.assertEqual(res["audit"]["legal"]["rejected"], 0)
        self.assertEqual(res["audit"]["support"]["rejected"], 0)

    # 10. Database values are strictly preserved
    def test_10_database_values_preserved(self):
        res = validate_payload(predicted_labels=["Sexual Harassment"], db=self.db)
        self.assertTrue(res["legal_matched"])
        for item in res["legal_information"]:
            self.assertIn("act", item)
            self.assertIn("section", item)
            self.assertIn("description", item)

    # 11. No external source dependency
    def test_11_no_external_source_dependency(self):
        res = validate_payload(predicted_labels=["Stalking"], db=self.db)
        self.assertIn("audit", res)
        self.assertIn("legal", res["audit"])
        self.assertIn("support", res["audit"])

    # 12. Empty results handled safely
    def test_12_empty_results(self):
        res = validate_payload(predicted_labels=[], db=self.db)
        self.assertFalse(res["legal_matched"])
        self.assertFalse(res["support_matched"])
        self.assertEqual(res["legal_message"], "Information not available in the provided knowledge base.")
        self.assertEqual(res["support_message"], "No matching support service was found in the available support-services database.")

    # 13. EXPLICIT HALLUCINATION ATTACK TEST
    def test_13_explicit_hallucination_attack(self):
        """
        Hallucination Attack Test:
        Simulates an unsafe model response containing fake laws and fake support contacts.
        Must REJECT 100% of fake items and return safe exact fallbacks.
        """
        fake_legal_input = {
            "matched": True,
            "legal_information": [
                {
                    "act": "Fake Women's Safety Act",
                    "section": "999",
                    "description": "This is a fabricated legal provision.",
                    "applicable_label": "DV",
                    "source": "model_guess",
                }
            ],
        }

        fake_support_input = {
            "matched": True,
            "support_services": [
                {
                    "service_name": "Fake Women's Helpline",
                    "service_type": "Helpline",
                    "contact_number": "9999999999",
                    "state": "Delhi",
                    "district": "New Delhi",
                    "applicable_label": "DV",
                    "source": "model_guess",
                }
            ],
        }

        res = validate_payload(
            predicted_labels=["DV"],
            legal_input=fake_legal_input,
            support_input=fake_support_input,
            db=self.db,
        )

        # Verify legal fake item was rejected
        self.assertFalse(res["legal_matched"])
        self.assertEqual(len(res["legal_information"]), 0)
        self.assertEqual(res["legal_message"], LEGAL_NO_MATCH_MESSAGE)
        self.assertEqual(res["audit"]["legal"]["rejected"], 1)

        # Verify support fake item was rejected
        self.assertFalse(res["support_matched"])
        self.assertEqual(len(res["support_services"]), 0)
        self.assertEqual(res["support_message"], SUPPORT_NO_MATCH_MESSAGE)
        self.assertEqual(res["audit"]["support"]["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
