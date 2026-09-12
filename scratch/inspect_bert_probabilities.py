import sys
import json
from pathlib import Path

sys.path.insert(0, r"d:\Abhera(Mini)")
sys.path.insert(0, r"d:\Abhera(Mini)\backend")

import torch
from ml.inference.predict_bert import BERTIncidentPredictor, LABEL_NAME_MAP

def inspect_scenarios():
    predictor = BERTIncidentPredictor()
    
    print("CONFIGURED THRESHOLDS:")
    print(json.dumps(predictor.optimal_thresholds, indent=2))
    print(f"LABEL COLUMNS: {predictor.label_columns}")
    
    scenarios = {
        "Scenario A (DV)": "My husband beats me regularly and threatens me.",
        "Scenario B (Stalking / Cyber)": "Someone keeps following me after college and sends threatening messages on Instagram.",
        "Scenario C (Workplace)": "My colleague keeps sending inappropriate messages and touching me at work.",
        "Scenario D (Dowry / Domestic)": "My in-laws are demanding money and threatening me because of dowry.",
    }
    
    for name, text in scenarios.items():
        print("\n==================================================")
        print(f"{name}: '{text}'")
        print("==================================================")
        
        result = predictor.predict(text)
        all_scores = result["all_scores"]
        thresholds = result["thresholds"]
        predicted = [p["name"] for p in result["predicted_labels"]]
        
        print("RAW PROBABILITIES PER LABEL:")
        for label_code in predictor.label_columns:
            full_name = LABEL_NAME_MAP.get(label_code, label_code)
            prob = all_scores[label_code]
            thresh = thresholds.get(label_code, 0.50)
            passes = prob >= thresh
            print(f"  - {label_code} ({full_name:<20}): Prob = {prob:.4f} | Thresh = {thresh:.2f} | Passed = {passes}")
            
        print(f"\nFINAL PASSED LABELS FOR {name}: {predicted}")

if __name__ == "__main__":
    inspect_scenarios()
