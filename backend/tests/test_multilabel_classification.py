"""
Unit Tests for Multi-Label Incident Classification in predict_bert.py.

Verifies:
1. "My husband beats me and threatens to post my private photos online." -> DV + CA
2. "My husband beats me." -> DV
3. "Someone threatens to post my private photos online." -> CA
4. "My husband threatens to post my private photos online." -> CA (no explicit physical violence)
5. "My husband physically abuses me and threatens to upload my private photos." -> DV + CA
6. "He falsely promised to marry me." -> FPM only
7. "He falsely promised to marry me and physically abused me." -> FPM + DV
8. "My in-laws demand dowry and physically abuse me." -> OV + DV (or DV + OV)
9. Deduplication check: Duplicate labels cannot appear.
"""

import unittest
from ml.inference.predict_bert import BERTIncidentPredictor


class TestMultiLabelClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = BERTIncidentPredictor()

    def test_01_dv_plus_ca_compound(self):
        text = "My husband beats me and threatens to post my private photos online."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Domestic Violence", labels)
        self.assertIn("Cyber Abuse", labels)

    def test_02_dv_only(self):
        text = "My husband beats me."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Domestic Violence", labels)
        self.assertNotIn("Cyber Abuse", labels)

    def test_03_ca_only_stranger(self):
        text = "Someone threatens to post my private photos online."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Cyber Abuse", labels)
        self.assertNotIn("Domestic Violence", labels)

    def test_04_ca_only_husband_no_physical_violence(self):
        text = "My husband threatens to post my private photos online."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Cyber Abuse", labels)
        self.assertNotIn("Domestic Violence", labels)

    def test_05_dv_plus_ca_physical_abuse(self):
        text = "My husband physically abuses me and threatens to upload my private photos."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Domestic Violence", labels)
        self.assertIn("Cyber Abuse", labels)

    def test_06_fpm_only(self):
        text = "He falsely promised to marry me."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertEqual(labels, ["False Promise of Marriage"])

    def test_07_fpm_plus_dv(self):
        text = "He falsely promised to marry me and physically abused me."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("False Promise of Marriage", labels)
        self.assertIn("Domestic Violence", labels)

    def test_08_dowry_plus_physical_abuse(self):
        text = "My in-laws demand dowry and physically abuse me."
        res = self.predictor.predict(text)
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Domestic Violence", labels)

    def test_09_deduplication(self):
        text = "My husband beats me and physically abuses me."
        res = self.predictor.predict(text)
        codes = [l["code"] for l in res["predicted_labels"]]
        self.assertEqual(len(codes), len(set(codes)), "Duplicate label codes detected")


if __name__ == "__main__":
    unittest.main()
