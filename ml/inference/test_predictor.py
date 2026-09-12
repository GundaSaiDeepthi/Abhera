import json
from pathlib import Path
import sys

# Add ml/ directory to Python path if necessary
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.inference.predict_bert import BERTIncidentPredictor

TEST_NARRATIVES = [
    {
        "id": 1,
        "description": "Workplace Harassment & Sexual Harassment Example",
        "text": "My senior manager at the office repeatedly sends unsolicited personal messages after hours, touches my shoulder inappropriately during meetings, and implied my promotion depends on agreeing to go out with him.",
    },
    {
        "id": 2,
        "description": "Domestic Violence Example",
        "text": "My husband physically assaulted me at home, slapped me during an argument, locked me inside the room for hours, and constantly threatens physical violence against me.",
    },
    {
        "id": 3,
        "description": "Cyber Abuse & Online Harassment Example",
        "text": "An anonymous person created fake social media accounts using my photos, uploaded edited images online without my consent, and keeps sending abusive threats via email.",
    },
    {
        "id": 4,
        "description": "Stalking Example",
        "text": "A man follows me every evening on my walk home from college, waits outside my apartment gate, and repeatedly calls me from different unknown numbers despite being told to stop.",
    },
    {
        "id": 5,
        "description": "Other Violence & Street Abuse Example",
        "text": "A group of strangers blocked my path at the bus stand, shouted obscene verbal insults, and pushed me aggressively when I tried to leave.",
    },
]


def run_predictor_tests():
    print("==================================================")
    print("BERT MULTI-LABEL INFERENCE PREDICTOR TEST SUITE")
    print("==================================================")
    print("Note: These 5 representative test examples demonstrate model inference behavior,")
    print("input processing, and per-label thresholding. They are qualitative test cases,")
    print("NOT statistical dataset accuracy measurements.")
    print("==================================================\n")

    predictor = BERTIncidentPredictor()

    for item in TEST_NARRATIVES:
        narrative_id = item["id"]
        description = item["description"]
        text = item["text"]

        print(f"--------------------------------------------------")
        print(f"TEST EXAMPLE #{narrative_id}: [{description}]")
        print(f"--------------------------------------------------")
        print(f"1. Input Narrative:\n   \"{text}\"\n")

        result = predictor.predict(text)

        predicted_labels = result["predicted_labels"]
        all_scores = result["all_scores"]
        thresholds = result["thresholds"]

        pred_codes = [p["code"] for p in predicted_labels]
        pred_names = [p["name"] for p in predicted_labels]

        print(f"2. Predicted Label Codes: {pred_codes if pred_codes else ['NONE']}")
        print(f"3. Predicted Label Names: {pred_names if pred_names else ['None (Unclassified)']}\n")

        print("4 & 5. Label Probabilities & Optimal Thresholds Used:")
        for code, prob in all_scores.items():
            thresh = thresholds.get(code, 0.50)
            is_pos = "POSITIVE [SELECTED]" if prob >= thresh else "Negative"
            print(f"   - {code:<3} ({predictor.label_mapping.get('id2label', {}).get(code, code)}): Prob = {prob:.4f} | Thresh = {thresh:.2f} -> {is_pos}")

        print("\nStructured Predictor Output Payload:")
        print(json.dumps(result, indent=2))
        print("\n")

    print("==================================================")
    print("PREDICTOR TEST SUITE COMPLETED SUCCESSFULLY")
    print("==================================================")


if __name__ == "__main__":
    run_predictor_tests()
