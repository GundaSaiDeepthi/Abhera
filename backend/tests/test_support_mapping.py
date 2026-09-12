"""
Unit and Integration Tests for Support Service Mapping Engine.

Verifies correct database retrieval of support services from PostgreSQL, location-based ranking,
multi-label combination, deduplication, value preservation, and strict anti-hallucination rules.
"""

import unittest

from app.database import SessionLocal
from app.models.support_service import SupportService
from app.services.support_mapping import SupportMappingService, map_support_services, NO_MATCH_MESSAGE


class TestSupportMappingEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()
        cls.service = SupportMappingService(db=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # 1. Category match
    def test_01_category_match(self):
        res = self.service.get_support_services(predicted_labels=["Domestic Violence"])
        self.assertTrue(res["matched"])
        self.assertGreater(len(res["support_services"]), 0)
        
        first = res["support_services"][0]
        self.assertIn("service_name", first)
        self.assertIn("service_type", first)
        self.assertIn("contact_number", first)
        self.assertIn("state", first)
        self.assertIn("district", first)
        self.assertEqual(first["source"], "database")

    # 2. State match
    def test_02_state_match(self):
        res = self.service.get_support_services(predicted_labels=["Domestic Violence"], state="Delhi")
        self.assertTrue(res["matched"])
        
        # Verify Delhi state record is present
        states = [s["state"] for s in res["support_services"]]
        self.assertIn("Delhi", states)

    # 3. District match
    def test_03_district_match(self):
        res = self.service.get_support_services(predicted_labels=["Domestic Violence"], district="New Delhi")
        self.assertTrue(res["matched"])
        
        districts = [s["district"] for s in res["support_services"]]
        self.assertIn("New Delhi", districts)

    # 4. Category + State + District match
    def test_04_category_state_district_match(self):
        res = self.service.get_support_services(
            predicted_labels=["Domestic Violence"],
            state="Delhi",
            district="New Delhi",
        )
        self.assertTrue(res["matched"])
        
        # Top ranked services must be the local Delhi/New Delhi services
        top_service = res["support_services"][0]
        self.assertEqual(top_service["state"], "Delhi")
        self.assertEqual(top_service["district"], "New Delhi")

    # 5. Multiple predicted labels
    def test_05_multiple_predicted_labels(self):
        res = self.service.get_support_services(predicted_labels=["DV", "CA"])
        self.assertTrue(res["matched"])
        self.assertGreater(len(res["support_services"]), 0)

    # 6. No match fallback
    def test_06_no_match(self):
        res = self.service.get_support_services(
            predicted_labels=["UNKNOWN_INCIDENT_XYZ"],
            state="NonExistentState",
            district="NonExistentDistrict",
        )
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["support_services"]), 0)
        self.assertEqual(res["message"], NO_MATCH_MESSAGE)
        self.assertEqual(
            res["message"],
            "No matching support service was found in the available support-services database."
        )

    # 7. Missing location parameters handled safely
    def test_07_missing_location(self):
        res = self.service.get_support_services(predicted_labels=["Stalking"], state=None, district=None)
        self.assertTrue(res["matched"])
        
        # Must return National helplines
        for s in res["support_services"]:
            self.assertIn(s["state"].lower(), ["national", "all india"])

    # 8. Empty labels list
    def test_08_empty_labels(self):
        res = self.service.get_support_services(predicted_labels=[])
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["support_services"]), 0)
        self.assertEqual(res["message"], NO_MATCH_MESSAGE)

    # 9. Duplicate predicted labels
    def test_09_duplicate_labels(self):
        res_single = self.service.get_support_services(predicted_labels=["Domestic Violence"])
        res_dup = self.service.get_support_services(predicted_labels=["DV", "Domestic Violence", "DV"])
        self.assertEqual(len(res_single["support_services"]), len(res_dup["support_services"]))

    # 10. Database values strictly preserved
    def test_10_database_values_preserved(self):
        res = map_support_services(predicted_labels=["Stalking"], db=self.db)
        self.assertTrue(res["matched"])
        
        # Verify first item matches exact DB record
        top_item = res["support_services"][0]
        db_rec = self.db.query(SupportService).filter(SupportService.name == top_item["service_name"]).first()
        self.assertIsNotNone(db_rec)
        self.assertEqual(top_item["service_type"], db_rec.service_type)
        self.assertEqual(top_item["contact_number"], db_rec.contact_number)
        self.assertEqual(top_item["state"], db_rec.state)
        self.assertEqual(top_item["district"], db_rec.district)

    # 11. No fabricated contact information
    def test_11_no_fabricated_contact_information(self):
        res = map_support_services(predicted_labels=["NON_EXISTENT_LABEL"], db=self.db)
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["support_services"]), 0)
        self.assertNotIn("service_name", res)
        self.assertNotIn("contact_number", res)

    # 12. EXPLICIT ANTI-HALLUCINATION TEST
    def test_12_anti_hallucination(self):
        """
        Anti-Hallucination Rule Verification:
        Must NOT return guessed organization, phone number, helpline, email, or address.
        MUST return exact fallback message.
        """
        res = map_support_services(
            predicted_labels=["UNKNOWN_CATEGORY"],
            state="UnknownState",
            district="UnknownDistrict",
            db=self.db,
        )
        self.assertFalse(res["matched"])
        self.assertEqual(len(res["support_services"]), 0)
        self.assertNotIn("service_name", res)
        self.assertNotIn("contact_number", res)
        self.assertEqual(
            res["message"],
            "No matching support service was found in the available support-services database."
        )


if __name__ == "__main__":
    unittest.main()
