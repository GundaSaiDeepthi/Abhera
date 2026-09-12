"""
Unit and Integration Tests for Dynamic Question Engine.

Tests cover all 12 required scenarios including entity-dependent question suppression,
multi-label requirement combination, state persistence, and completion criteria.
"""

import unittest
from typing import List
from dataclasses import dataclass

from app.question_engine.rules import get_required_information, normalize_category
from app.question_engine.question_selector import (
    get_known_information,
    get_missing_information,
    select_next_question,
)
from app.question_engine.engine import process_submission


@dataclass
class MockQuestion:
    question_id: str
    category: str
    question_text: str
    required_entity: str
    priority: int
    active: bool = True


def get_mock_questions_bank() -> List[MockQuestion]:
    return [
        MockQuestion("Q_SAFETY_01", "General", "Are you in immediate danger?", "SAFETY_STATUS", 100),
        MockQuestion("Q_PERP_01", "General", "What is your relationship to the person?", "PERP_REL", 90),
        MockQuestion("Q_LOC_01", "Sexual Harassment", "Where did this happen?", "LOCATION", 85),
        MockQuestion("Q_TIME_01", "Stalking", "How often does this occur?", "TIME_FREQ", 85),
        MockQuestion("Q_PLATFORM_01", "Cyber Harassment", "Which platform was involved?", "PLATFORM", 85),
        MockQuestion("Q_DETAILS_01", "General", "Describe what happened.", "INCIDENT_DETAILS", 70),
    ]


class TestDynamicQuestionEngine(unittest.TestCase):

    def setUp(self):
        self.bank = get_mock_questions_bank()

    # 1. New submission with no information
    def test_01_new_submission_no_info(self):
        res = process_submission(
            submission_id="SUB-001",
            narrative_text="",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            previous_answers={},
            questions_bank_override=self.bank,
        )
        self.assertFalse(res["completion_status"])
        self.assertIn("SAFETY_STATUS", res["missing_information"])
        self.assertIn("PERP_REL", res["missing_information"])
        self.assertEqual(res["question_id"], "Q_SAFETY_01")
        self.assertIsNotNone(res["next_question"])

    # 2. Narrative already containing PERP_REL (via NER)
    def test_02_narrative_with_perp_rel(self):
        ner_entities = [{"label": "PERP_REL", "text": "former colleague"}]
        known = get_known_information(narrative_text="", ner_entities=ner_entities)
        self.assertIn("PERP_REL", known)
        self.assertEqual(known["PERP_REL"], "former colleague")

        res = process_submission(
            submission_id="SUB-002",
            narrative_text="",
            bert_results={"labels": ["Workplace Harassment"]},
            ner_entities=ner_entities,
            questions_bank_override=self.bank,
        )
        self.assertNotIn("PERP_REL", res["missing_information"])
        self.assertNotEqual(res["question_id"], "Q_PERP_01")

    # 3. Narrative already containing LOCATION (via NER)
    def test_03_narrative_with_location(self):
        ner_entities = [{"label": "LOCATION", "text": "office cafeteria"}]
        res = process_submission(
            submission_id="SUB-003",
            narrative_text="",
            bert_results={"labels": ["Sexual Harassment"]},
            ner_entities=ner_entities,
            questions_bank_override=self.bank,
        )
        self.assertNotIn("LOCATION", res["missing_information"])
        self.assertNotEqual(res["question_id"], "Q_LOC_01")

    # 4. Narrative already containing TIME_FREQ (via NER)
    def test_04_narrative_with_time_freq(self):
        ner_entities = [{"label": "TIME_FREQ", "text": "every evening"}]
        res = process_submission(
            submission_id="SUB-004",
            narrative_text="",
            bert_results={"labels": ["Stalking"]},
            ner_entities=ner_entities,
            questions_bank_override=self.bank,
        )
        self.assertNotIn("TIME_FREQ", res["missing_information"])
        self.assertNotEqual(res["question_id"], "Q_TIME_01")

    # 5. Narrative already containing PLATFORM (via NER)
    def test_05_narrative_with_platform(self):
        ner_entities = [{"label": "PLATFORM", "text": "Instagram"}]
        res = process_submission(
            submission_id="SUB-005",
            narrative_text="",
            bert_results={"labels": ["Cyber Harassment"]},
            ner_entities=ner_entities,
            questions_bank_override=self.bank,
        )
        self.assertNotIn("PLATFORM", res["missing_information"])
        self.assertNotEqual(res["question_id"], "Q_PLATFORM_01")

    # 6. Previously answered question
    def test_06_previously_answered_question(self):
        prev_answers = {"Q_SAFETY_01": "No, I am safe in a shelter currently."}
        res = process_submission(
            submission_id="SUB-006",
            narrative_text="",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            previous_answers=prev_answers,
            questions_bank_override=self.bank,
        )
        self.assertNotEqual(res["question_id"], "Q_SAFETY_01")
        self.assertEqual(res["question_id"], "Q_PERP_01")

    # 7. Multiple incident categories (Multi-label combination)
    def test_07_multiple_incident_categories(self):
        reqs = get_required_information(["DV", "ST"])
        self.assertIn("PERP_REL", reqs)
        self.assertIn("SAFETY_STATUS", reqs)
        self.assertIn("TIME_FREQ", reqs)
        self.assertIn("PLATFORM", reqs)
        self.assertIn("INCIDENT_DETAILS", reqs)

    # 8. Completion when all required information exists
    def test_08_completion_when_all_info_exists(self):
        ner_entities = [
            {"label": "PERP_REL", "text": "husband"},
            {"label": "LOCATION", "text": "home"},
        ]
        prev_answers = {
            "Q_SAFETY_01": "Safe now",
            "Q_DETAILS_01": "Pushed me during argument",
        }
        res = process_submission(
            submission_id="SUB-008",
            narrative_text="My husband pushed me at home yesterday.",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=ner_entities,
            previous_answers=prev_answers,
            questions_bank_override=self.bank,
        )
        self.assertTrue(res["completion_status"])
        self.assertEqual(len(res["missing_information"]), 0)
        self.assertIsNone(res["next_question"])
        self.assertIsNone(res["question_id"])

    # 9. No duplicate questions
    def test_09_no_duplicate_questions(self):
        asked_ids = {"Q_SAFETY_01", "Q_PERP_01"}
        next_q = select_next_question(
            missing_info=["SAFETY_STATUS", "PERP_REL", "INCIDENT_DETAILS"],
            active_categories=["General"],
            asked_question_ids=asked_ids,
            questions_bank=self.bank,
        )
        self.assertIsNotNone(next_q)
        self.assertNotIn(next_q.question_id, asked_ids)

    # 10. Unknown / unsupported incident category
    def test_10_unknown_category_fallback(self):
        norm = normalize_category("UNKNOWN_CATEGORY_123")
        self.assertEqual(norm, "Unknown_Category_123")

        res = process_submission(
            submission_id="SUB-010",
            narrative_text="",
            bert_results={"labels": ["UNKNOWN_CATEGORY_123"]},
            ner_entities=[],
            questions_bank_override=self.bank,
        )
        self.assertFalse(res["completion_status"])
        self.assertIn("SAFETY_STATUS", res["missing_information"])

    # 11. Resuming an existing submission
    def test_11_resuming_existing_submission(self):
        # Step 1: User asks first question
        step1 = process_submission(
            submission_id="SUB-RESUME",
            narrative_text="",
            bert_results={"labels": ["Stalking"]},
            ner_entities=[],
            previous_answers={},
            questions_bank_override=self.bank,
        )
        q1_id = step1["question_id"]

        # Step 2: User returns later with answer to Q1
        step2 = process_submission(
            submission_id="SUB-RESUME",
            narrative_text="",
            bert_results={"labels": ["Stalking"]},
            ner_entities=[],
            previous_answers={q1_id: "I am safe right now."},
            questions_bank_override=self.bank,
        )
        self.assertNotEqual(step2["question_id"], q1_id)
        self.assertFalse(step2["completion_status"])

    # 12. Entity-dependent question suppression
    def test_12_entity_dependent_question_suppression(self):
        ner_entities = [
            {"label": "PERP_REL", "text": "ex-husband"},
            {"label": "TIME_FREQ", "text": "every day for a month"},
            {"label": "PLATFORM", "text": "WhatsApp"},
        ]
        known = get_known_information(narrative_text="", ner_entities=ner_entities)
        reqs = get_required_information(["Stalking"])
        missing = get_missing_information(reqs, known)

        self.assertNotIn("PERP_REL", missing)
        self.assertNotIn("TIME_FREQ", missing)
        self.assertNotIn("PLATFORM", missing)
        self.assertIn("SAFETY_STATUS", missing)


class TestConditionalQuestioning(unittest.TestCase):

    def setUp(self):
        self.bank = get_mock_questions_bank()

    def test_01_sufficiently_detailed_description_no_questions(self):
        res = process_submission(
            submission_id="SUB-COND-01",
            narrative_text="My husband beats me every day and I am currently safe.",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            questions_bank_override=self.bank,
        )
        self.assertTrue(res["completion_status"])
        self.assertEqual(len(res["missing_information"]), 0)
        self.assertIsNone(res["next_question"])

    def test_02_missing_required_field_asks_only_that_question(self):
        res = process_submission(
            submission_id="SUB-COND-02",
            narrative_text="My husband beats me every day.",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            questions_bank_override=self.bank,
        )
        self.assertFalse(res["completion_status"])
        self.assertEqual(res["missing_information"], ["SAFETY_STATUS"])
        self.assertEqual(res["question_id"], "Q_SAFETY_01")

    def test_03_already_provided_information_prevents_corresponding_question(self):
        res = process_submission(
            submission_id="SUB-COND-03",
            narrative_text="A falsely promised to marry me. This happened last week and I feel unsafe.",
            bert_results={"labels": ["False Promise of Marriage"]},
            ner_entities=[{"label": "PERP_REL", "text": "A"}],
            questions_bank_override=self.bank,
        )
        self.assertTrue(res["completion_status"])
        self.assertEqual(len(res["missing_information"]), 0)
        self.assertIsNone(res["next_question"])

    def test_04_irrelevant_category_questions_never_asked(self):
        res = process_submission(
            submission_id="SUB-COND-04",
            narrative_text="My husband beats me.",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            questions_bank_override=self.bank,
        )
        self.assertNotIn("LOCATION", res["missing_information"])
        self.assertNotIn("PLATFORM", res["missing_information"])
        self.assertNotEqual(res["question_id"], "Q_LOC_01")
        self.assertNotEqual(res["question_id"], "Q_PLATFORM_01")

    def test_05_follow_up_answer_completes_questioning(self):
        # Turn 1: Missing safety
        res1 = process_submission(
            submission_id="SUB-COND-05",
            narrative_text="My husband beats me every day.",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            previous_answers={},
            questions_bank_override=self.bank,
        )
        self.assertFalse(res1["completion_status"])
        self.assertEqual(res1["question_id"], "Q_SAFETY_01")

        # Turn 2: User answers safety question
        res2 = process_submission(
            submission_id="SUB-COND-05",
            narrative_text="My husband beats me every day.\nI am safe now.",
            bert_results={"labels": ["Domestic Violence"]},
            ner_entities=[],
            previous_answers={"Q_SAFETY_01": "I am safe now."},
            questions_bank_override=self.bank,
        )
        self.assertTrue(res2["completion_status"])
        self.assertIsNone(res2["next_question"])

    def test_06_fpm_dv_disambiguation_preserved(self):
        from ml.inference.predict_bert import BERTIncidentPredictor
        predictor = BERTIncidentPredictor()
        
        # FPM only
        res_fpm = predictor.predict("A falsely promised to marry B.")
        labels_fpm = [l["name"] for l in res_fpm["predicted_labels"]]
        self.assertEqual(labels_fpm, ["False Promise of Marriage"])

        # FPM + DV
        res_mixed = predictor.predict("He falsely promised to marry me and physically abused me.")
        labels_mixed = [l["name"] for l in res_mixed["predicted_labels"]]
        self.assertIn("False Promise of Marriage", labels_mixed)
        self.assertIn("Domestic Violence", labels_mixed)


if __name__ == "__main__":
    unittest.main()
