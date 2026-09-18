"""
ABHERA — BERT Multi-Label Incident Classification Inference Module

This module implements the production inference wrapper for ABHERA's fine-tuned
BERT multi-label classification model (`models/bert_multilabel_experiment_v2`).

Architecture & ML Pipeline Details:
-----------------------------------
1. Model Architecture: Fine-tuned `bert-base-uncased` sequence classification head.
2. Output Layer: Multi-label binary cross-entropy logits with independent Sigmoids:
   p_i = 1 / (1 + exp(-z_i)) for each legal incident class i in {DV, SH, ST, CA, WH, OV}.
3. Decision Thresholding: Applies validation-derived per-class optimal thresholds
   from `optimal_thresholds.json` (DV: 0.73, SH: 0.50, ST: 0.79, CA: 0.45, WH: 0.29, OV: 0.63).
   Independent thresholds maximize validation Micro-F1 while maintaining held-out test isolation.
4. Multi-Label Explicit Evidence Reinforcement: Rules enforce safety overlays for critical
   survivor evidence (e.g. physical domestic violence, dowry demands, false promise of marriage).
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BERTPredictor")


def has_explicit_dv_evidence(text: str) -> bool:
    """
    Detects whether the incident narrative contains explicit domestic-violence evidence
    such as physical violence, physical abuse/cruelty, or physical threats.

    Args:
        text (str): Input survivor narrative string.

    Returns:
        bool: True if explicit physical domestic violence keywords/patterns are detected.
    """
    if not text or not isinstance(text, str):
        return False

    import re
    lower = text.lower()

    physical_terms = [
        "beat", "beats", "beating", "hit", "hits", "hitting",
        "slap", "slapped", "slaps", "kick", "kicked", "kicking",
        "punch", "punched", "punching", "physically abuse", "physical abuse",
        "physically abused", "physically abuses", "assault", "assaulted", "attack", "attacked",
        "injured", "injury", "physically harm", "physical violence"
    ]

    abuse_terms = [
        "abuse", "abused", "abusive", "torture", "tortured", "cruelty", "cruel"
    ]

    violence_terms = [
        "violence", "violent"
    ]

    threat_patterns = [
        r"\bthreatened\s+(?:to\s+)?(?:kill|beat|hit|harm|assault|hurt)\b",
        r"\bthreatens\s+(?:to\s+)?(?:kill|beat|hit|harm|assault|hurt)\b",
        r"\bthreatened\s+(?:me\s+)?with\s+(?:violence|harm|weapon)\b",
        r"\bphysically\s+threaten\b"
    ]

    for term in physical_terms + abuse_terms + violence_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", lower):
            return True

    for pattern in threat_patterns:
        if re.search(pattern, lower):
            return True

    return False


def has_explicit_dowry_evidence(text: str) -> bool:
    """
    Detects whether the incident narrative contains explicit dowry evidence
    such as dowry demands, harassment for dowry, insufficient dowry, or
    demands for money/gold/property connected to marriage.
    """
    if not text or not isinstance(text, str):
        return False

    import re
    lower = text.lower()

    if re.search(r"\bdowrys?\b", lower):
        return True

    dowry_patterns = [
        r"\bdemand(?:ed|s|ing)?\s+(?:more\s+)?(?:cash|gold|money|property|jewel(?:ry|lery)|car|vehicle)\s+(?:as|for|because of)\s+dowry\b",
        r"\b(?:money|cash|gold|property|jewel(?:ry|lery))\s+demand(?:s)?\s+(?:for|connected to|after)\s+(?:marriage|wedding|bride)\b",
        r"\bharass(?:ed|ment|es|ing)?\s+for\s+(?:more\s+)?(?:dowry|money for marriage|property from (?:my|her) (?:parents|family))\b",
        r"\bpressur(?:e|ed|ing|es)\s+to\s+bring\s+(?:more\s+)?(?:dowry|money|cash|gold|property)\b",
        r"\b(?:insufficient|not\s+enough)\s+dowry\b",
        r"\bdowry\s+(?:harassment|demand|abuse|cruelty|threats?)\b"
    ]

    for pattern in dowry_patterns:
        if re.search(pattern, lower):
            return True

    return False


def has_explicit_wh_evidence(text: str) -> bool:
    """Detects whether narrative contains explicit workplace harassment evidence."""
    if not text or not isinstance(text, str):
        return False
    import re
    lower = text.lower()
    wh_terms = [
        "work", "office", "boss", "manager", "colleague", "coworker", "job",
        "employee", "employer", "workplace", "company", "career", "promotion", "shift"
    ]
    return any(re.search(r"\b" + re.escape(term) + r"\b", lower) for term in wh_terms)


def has_explicit_ov_evidence(text: str) -> bool:
    """Detects whether narrative contains explicit other/street violence evidence."""
    if not text or not isinstance(text, str):
        return False
    import re
    lower = text.lower()
    ov_terms = [
        "attack", "attacked", "assault", "assaulted", "stranger", "weapon",
        "gun", "knife", "street", "bus stand", "road", "public", "gang", "crowd"
    ]
    return any(re.search(r"\b" + re.escape(term) + r"\b", lower) for term in ov_terms)


LABEL_NAME_MAP = {
    "DV": "Domestic Violence",
    "SH": "Sexual Harassment",
    "ST": "Stalking",
    "CA": "Cyber Abuse",
    "WH": "Workplace Harassment",
    "OV": "Other Violence",
    "FPM": "False Promise of Marriage",
    "DOWRY": "Dowry Harassment",
}


class BERTIncidentPredictor:
    """Standalone BERT Multi-Label Predictor for Women's Safety Incidents."""

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_dir = project_root / "models" / "bert_multilabel_experiment_v2"

        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.label_mapping = {}
        self.label_columns = []
        self.optimal_thresholds = {}

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model, tokenizer, label mapping, and validation-derived optimal thresholds."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found at: {self.model_dir}. "
                "Ensure fine-tuned model artifacts exist in models/bert_multilabel/."
            )

        mapping_json_path = self.model_dir / "label_mapping.json"
        if not mapping_json_path.exists():
            raise FileNotFoundError(f"Label mapping file missing at: {mapping_json_path}")

        thresholds_json_path = self.model_dir / "optimal_thresholds.json"
        if not thresholds_json_path.exists():
            raise FileNotFoundError(
                f"Optimal thresholds file missing at: {thresholds_json_path}. "
                "Run threshold tuning script first."
            )

        # 1. Load label mapping & threshold configurations
        try:
            with open(mapping_json_path, "r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)
            self.label_columns = self.label_mapping.get(
                "labels", ["DV", "SH", "ST", "CA", "WH", "OV"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read label_mapping.json: {e}") from e

        try:
            with open(thresholds_json_path, "r", encoding="utf-8") as f:
                self.optimal_thresholds = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read optimal_thresholds.json: {e}") from e

        # Ensure defaults for any missing thresholds
        for col in self.label_columns:
            if col not in self.optimal_thresholds:
                self.optimal_thresholds[col] = 0.50

        # 2. Load tokenizer
        try:
            logger.info(f"Loading BERT Tokenizer from: {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT tokenizer from {self.model_dir}: {e}") from e

        # 3. Load fine-tuned PyTorch model
        try:
            logger.info(f"Loading Fine-Tuned BERT Model from: {self.model_dir} (Device: {self.device})")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info("BERT Model and Tokenizer successfully initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to load fine-tuned BERT model from {self.model_dir}: {e}") from e

    def predict(self, text: str, max_length: int = 256) -> Dict[str, Any]:
        """
        Runs multi-label incident classification on a raw incident narrative string.

        Args:
            text (str): Raw incident narrative text.
            max_length (int): Maximum token sequence length (default: 256).

        Returns:
            Dict containing predicted_labels, all_scores, and thresholds used.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input narrative text must be a non-empty string.")

        clean_text = text.strip()

        # Tokenization
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Inference without gradients
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0).cpu().numpy()

        # Sigmoid probability computation
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        all_scores = {}
        predicted_labels = []
        lower_text = clean_text.lower()

        fpm_keywords = [
            "promised to marry",
            "promise of marriage",
            "falsely promised",
            "deceived me by making a promise",
            "false promise",
            "promising to marry",
        ]

        for idx, label_code in enumerate(self.label_columns):
            if idx < len(probabilities):
                prob = float(probabilities[idx])
            else:
                prob = 0.0

            if label_code == "FPM":
                if any(kw in lower_text for kw in fpm_keywords):
                    prob = max(prob, 0.95)

            thresh = float(self.optimal_thresholds.get(label_code, 0.50))
            all_scores[label_code] = round(prob, 4)

            if prob >= thresh:
                predicted_labels.append({
                    "code": label_code,
                    "name": LABEL_NAME_MAP.get(label_code, label_code),
                    "probability": round(prob, 4),
                })

        # 1. Rule-based FPM (False Promise of Marriage) keyword overlay
        if any(kw in lower_text for kw in fpm_keywords):
            fpm_prob = 0.95
            fpm_thresh = float(self.optimal_thresholds.get("FPM", 0.50))
            all_scores["FPM"] = round(fpm_prob, 4)
            if fpm_prob >= fpm_thresh:
                predicted_labels.append({
                    "code": "FPM",
                    "name": LABEL_NAME_MAP.get("FPM", "False Promise of Marriage"),
                    "probability": round(fpm_prob, 4),
                })

        # 2. Multi-label explicit evidence reinforcement for Domestic Violence
        is_dv_already_selected = any(item["code"] == "DV" for item in predicted_labels)
        if not is_dv_already_selected and has_explicit_dv_evidence(clean_text):
            dv_prob = all_scores.get("DV", 0.0)
            predicted_labels.append({
                "code": "DV",
                "name": LABEL_NAME_MAP.get("DV", "Domestic Violence"),
                "probability": dv_prob,
            })

        # 3. Multi-label explicit evidence reinforcement for Dowry Harassment
        is_dowry_already_selected = any(item["code"] == "DOWRY" for item in predicted_labels)
        if not is_dowry_already_selected and has_explicit_dowry_evidence(clean_text):
            dv_score = all_scores.get("DV", 0.85)
            dowry_prob = max(dv_score, 0.95)
            predicted_labels.append({
                "code": "DOWRY",
                "name": LABEL_NAME_MAP.get("DOWRY", "Dowry Harassment"),
                "probability": round(dowry_prob, 4),
            })

        # 3. Post-classification disambiguation for FPM
        is_fpm_selected = any(item["code"] == "FPM" for item in predicted_labels)
        if is_fpm_selected:
            if not has_explicit_dv_evidence(clean_text):
                predicted_labels = [item for item in predicted_labels if item["code"] != "DV"]
            if not has_explicit_wh_evidence(clean_text):
                predicted_labels = [item for item in predicted_labels if item["code"] != "WH"]
            if not has_explicit_ov_evidence(clean_text):
                predicted_labels = [item for item in predicted_labels if item["code"] != "OV"]

        # 4. Deduplicate predicted labels by code
        seen_codes = set()
        dedup_predicted_labels = []
        for item in predicted_labels:
            if item["code"] not in seen_codes:
                seen_codes.add(item["code"])
                dedup_predicted_labels.append(item)
        predicted_labels = dedup_predicted_labels

        # 5. Sort predicted labels by probability descending
        predicted_labels.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "predicted_labels": predicted_labels,
            "all_scores": all_scores,
            "thresholds": self.optimal_thresholds,
        }
