"""
Unit Tests for Dynamic Question Generation Architecture.

Validates all 10 required test scenarios:
1. No questions asked when all required information exists.
2. Missing information causes exactly one question.
3. Question text refers to actual incident narrative/context.
4. Information already present is never requested again.
5. Different descriptions produce different question wording when the same field is missing.
6. Different incident categories produce contextually appropriate questions.
7. Previous answers are considered.
8. Previously asked questions are not repeated.
9. Questioning stops once all required information is gathered.
10. FPM/DV disambiguation remains intact and fully functional.
"""

import unittest
from app.models.question import Question
from app.question_engine.engine import process_submission
from app.question_engine.dynamic_generator import generate_dynamic_question
from ml.inference.predict_bert import BERTIncidentPredictor


class TestDynamicQuestionGeneration(unittest.TestCase):
    def setUp(self):
        # Controlled mock questions bank
        self.questions_bank = [
            Question(
                question_id="Q_DV_01",
                category="Domestic Violence",
                required_entity="SAFETY_STATUS",
                priority=100,
                active=True,
                question_text="Are you currently in immediate danger?",
            ),
            Question(
                question_id="Q_DV_02",
                category="Domestic Violence",
                required_entity="PERP_REL",
                priority=90,
                active=True,
                question_text="What is your relationship to the person?",
            ),
            Question(
                question_id="Q_ST_01",
                category="Stalking",
                required_entity="PLATFORM",
                priority=80,
                active=True,
                question_text="Which platform was used?",
            ),
            Question(
                question_id="Q_CA_01",
                category="Cyber Harassment",
                required_entity="TIME_FREQ",
                priority=70,
                active=True,
                question_text="How often does this happen?",
            ),
        ]

    def test_01_all_info_present_no_questions_asked(self):
        """Scenario 1 & Example 2: When all required information exists in narrative, NO question is asked."""
        narrative = "My husband beats me every day at home and I am currently safe."
        bert_res = ["Domestic Violence"]

        res = process_submission(
            submission_id="SUB-TEST-01",
            narrative_text=narrative,
            bert_results=bert_res,
            questions_bank_override=self.questions_bank,
        )

        self.assertTrue(res["completion_status"])
        self.assertIsNone(res["next_question"])
        self.assertEqual(len(res["missing_information"]), 0)

    def test_02_missing_info_causes_exactly_one_question(self):
        """Scenario 2 & Example 1: Missing SAFETY_STATUS causes exactly one question."""
        narrative = "My husband beats me every day at home."
        bert_res = ["Domestic Violence"]

        res = process_submission(
            submission_id="SUB-TEST-02",
            narrative_text=narrative,
            bert_results=bert_res,
            questions_bank_override=self.questions_bank,
        )

        self.assertFalse(res["completion_status"])
        self.assertIsNotNone(res["next_question"])
        self.assertEqual(res["question_id"], "Q_DV_01")
        self.assertIn("SAFETY_STATUS", res["missing_information"])

    def test_03_question_refers_to_actual_narrative_context(self):
        """Scenario 3: Question text refers to the actual incident description."""
        narrative = "My husband beats me every day at home."
        bert_res = ["Domestic Violence"]

        res = process_submission(
            submission_id="SUB-TEST-03",
            narrative_text=narrative,
            bert_results=bert_res,
            questions_bank_override=self.questions_bank,
        )

        q_text = res["next_question"]
        # Question text must reference narrative context (e.g. husband beats you every day at home)
        self.assertTrue(
            "husband" in q_text.lower() or "beats" in q_text.lower(),
            f"Expected context reference in question text: {q_text}",
        )
        self.assertIn("immediate danger", q_text.lower())

    def test_04_information_already_present_never_requested_again(self):
        """Scenario 4: Information already present (husband, every day, home) is not asked."""
        narrative = "My husband beats me every day at home."
        bert_res = ["Domestic Violence"]

        res = process_submission(
            submission_id="SUB-TEST-04",
            narrative_text=narrative,
            bert_results=bert_res,
            questions_bank_override=self.questions_bank,
        )

        # Missing should only contain SAFETY_STATUS, NOT PERP_REL or INCIDENT_DETAILS
        self.assertNotIn("PERP_REL", res["missing_information"])
        self.assertNotIn("INCIDENT_DETAILS", res["missing_information"])

    def test_05_different_descriptions_produce_different_question_wording(self):
        """Scenario 5: Different incident descriptions produce different question phrasing for the same missing field."""
        q1 = generate_dynamic_question(
            required_entity="SAFETY_STATUS",
            active_categories=["Domestic Violence"],
            narrative_text="My husband beats me every day at home.",
            known_info={"PERP_REL": "husband", "INCIDENT_DETAILS": "beats me every day at home"},
        )

        q2 = generate_dynamic_question(
            required_entity="SAFETY_STATUS",
            active_categories=["Stalking"],
            narrative_text="Someone keeps following me after college.",
            known_info={"INCIDENT_DETAILS": "someone keeps following me after college"},
        )

        self.assertNotEqual(q1, q2)
        self.assertIn("husband", q1.lower())
        self.assertIn("following", q2.lower())

    def test_06_different_categories_produce_contextually_appropriate_questions(self):
        """Scenario 6: Different incident categories produce category-appropriate questions."""
        q_ca = generate_dynamic_question(
            required_entity="PLATFORM",
            active_categories=["Cyber Harassment"],
            narrative_text="I receive abusive messages online.",
            known_info={"INCIDENT_DETAILS": "abusive messages online"},
        )

        self.assertIn("platform", q_ca.lower())
        self.assertIn("instagram", q_ca.lower())

    def test_07_previous_answers_are_considered(self):
        """Scenario 7: Previous answers satisfy missing information requirement."""
        narrative = "My husband beats me every day at home."
        bert_res = ["Domestic Violence"]
        prev_ans = {"Q_DV_01": "I am currently safe and at a shelter."}

        res = process_submission(
            submission_id="SUB-TEST-07",
            narrative_text=narrative,
            bert_results=bert_res,
            previous_answers=prev_ans,
            questions_bank_override=self.questions_bank,
        )

        self.assertTrue(res["completion_status"])
        self.assertIsNone(res["next_question"])

    def test_08_previously_asked_questions_not_repeated(self):
        """Scenario 8: Previously asked question IDs are excluded."""
        narrative = "I am being harassed."
        bert_res = ["Domestic Violence"]
        prev_ans = {"Q_DV_01": "I don't know."}

        res = process_submission(
            submission_id="SUB-TEST-08",
            narrative_text=narrative,
            bert_results=bert_res,
            previous_answers=prev_ans,
            questions_bank_override=self.questions_bank,
        )

        # Should NOT ask Q_DV_01 again
        self.assertNotEqual(res.get("question_id"), "Q_DV_01")

    def test_09_questioning_stops_once_all_info_available(self):
        """Scenario 9 & Example 4: FPM narrative with all info completes immediately."""
        narrative = "A falsely promised to marry me. This happened last week and I feel unsafe."
        bert_res = ["False Promise of Marriage"]

        res = process_submission(
            submission_id="SUB-TEST-09",
            narrative_text=narrative,
            bert_results=bert_res,
            questions_bank_override=self.questions_bank,
        )

        self.assertTrue(res["completion_status"])
        self.assertIsNone(res["next_question"])

    def test_10_fpm_dv_disambiguation_intact(self):
        """Scenario 10: FPM/DV post-classification disambiguation layer remains functional."""
        predictor = BERTIncidentPredictor()

        # Pure FPM sentence should predict FPM only
        res_fpm = predictor.predict("A falsely promised to marry B.")
        labels_fpm = [l["name"] for l in res_fpm.get("predicted_labels", [])]
        self.assertIn("False Promise of Marriage", labels_fpm)
        self.assertNotIn("Domestic Violence", labels_fpm)

        # Explicit DV sentence should predict DV
        res_dv = predictor.predict("My husband beats me and physically abuses me every day.")
        labels_dv = [l["name"] for l in res_dv.get("predicted_labels", [])]
        self.assertIn("Domestic Violence", labels_dv)


if __name__ == "__main__":
    unittest.main()
