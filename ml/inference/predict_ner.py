"""
ABHERA — Named Entity Recognition (NER) Inference Module

This module implements the production inference wrapper for ABHERA's fine-tuned
BERT Token Classification model (`models/ner`).

Architecture & NER Pipeline Details:
------------------------------------
1. Model Architecture: Fine-tuned `bert-base-uncased` token-level classification head.
2. Labeling Scheme: BIO Tagging (Begin, Inside, Outside) for legal entities:
   - PERP_REL (Perpetrator Relationship, e.g. "husband", "colleague")
   - LOCATION (Incident Location, e.g. "office", "bus stop")
   - DATE_TIME (Time/Frequency details, e.g. "last night", "for 2 weeks")
   - CONTACT_INFO (Phone/Handle/Email, e.g. "WhatsApp", "@anon_user")
   - EVIDENCE (Physical/Digital Evidence, e.g. "screenshots", "audio recording")
3. Token Alignment: Offsets mapping aligns WordPiece sub-tokens back to original
   character character spans in raw narrative text.
4. Functional Distinction: NER identifies structured evidentiary entities;
   BERT classification (`predict_bert.py`) categorizes legal incident types.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NERPredictor")


def extract_spans_from_bio(tags: List[str]) -> List[tuple]:
    """
    Extracts contiguous entity spans (start_token_idx, end_token_idx, label)
    from a sequence of BIO token tags.

    Args:
        tags (List[str]): Sequence of predicted BIO tags (e.g. ['B-PERP_REL', 'I-PERP_REL', 'O']).

    Returns:
        List[tuple]: Extracted spans as (start_idx, end_idx, entity_type).
    """
    spans = []
    current_label = None
    start_idx = -1

    for idx, tag in enumerate(tags):
        if tag == "O" or tag == "-100":
            if current_label is not None:
                spans.append((start_idx, idx - 1, current_label))
                current_label = None
                start_idx = -1
        elif tag.startswith("B-"):
            if current_label is not None:
                spans.append((start_idx, idx - 1, current_label))
            current_label = tag[2:]
            start_idx = idx
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if current_label == ent_type and current_label is not None:
                pass
            else:
                if current_label is not None:
                    spans.append((start_idx, idx - 1, current_label))
                current_label = ent_type
                start_idx = idx

    if current_label is not None:
        spans.append((start_idx, len(tags) - 1, current_label))

    return spans


class NEREntityPredictor:
    """Standalone BERT Token Classification Predictor for Named Entity Recognition."""

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_dir = project_root / "models" / "ner"

        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads tokenizer, model, and id/label mapping files."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"NER model directory missing at: {self.model_dir}")

        label_map_path = self.model_dir / "label_mapping.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"NER label mapping file missing at: {label_map_path}")

        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)

            self.label2id = {k: int(v) for k, v in mapping_data["label2id"].items()}
            self.id2label = {int(k): v for k, v in mapping_data["id2label"].items()}
        except Exception as e:
            raise RuntimeError(f"Failed to read NER label mapping: {e}") from e

        try:
            logger.info(f"Loading NER Tokenizer from: {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            logger.info(f"Loading NER Model from: {self.model_dir} (Device: {self.device})")
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info("NER Model and Tokenizer successfully initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to load fine-tuned NER model: {e}") from e

    def predict(self, text: str, max_length: int = 256) -> List[Dict[str, Any]]:
        """
        Runs NER entity extraction on input text.

        Returns list of dicts: [{"label": "PERP_REL", "text": "ex-husband"}, ...]
        """
        if not isinstance(text, str) or not text.strip():
            return []

        clean_text = text.strip()
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        tags = [self.id2label.get(p, "O") for p in preds]
        spans = extract_spans_from_bio(tags)

        extracted_entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for start_idx, end_idx, label in spans:
            s_idx = start_idx
            while s_idx > 0 and tokens[s_idx].startswith("##"):
                s_idx -= 1

            e_idx = end_idx
            while e_idx + 1 < len(tokens) and tokens[e_idx + 1].startswith("##"):
                e_idx += 1

            start_char = int(offset_mapping[s_idx][0])
            end_char = int(offset_mapping[e_idx][1])
            if start_char == 0 and end_char == 0:
                continue

            entity_text = clean_text[start_char:end_char].strip()
            if entity_text:
                extracted_entities.append({
                    "label": label,
                    "text": entity_text,
                    "start_char": start_char,
                    "end_char": end_char,
                })

        # Post-processing heuristic for perpetrator subject in promise of marriage contexts
        has_perp = any(e["label"] == "PERP_REL" for e in extracted_entities)
        if not has_perp:
            import re
            m = re.search(r"\b([A-Za-z0-9_]+)\s+(?:falsely\s+promised|promised\s+to\s+marry|deceived)\b", clean_text, re.IGNORECASE)
            if m:
                subj = m.group(1).strip()
                if subj and not any(e["label"] == "PERP_REL" and e["text"].lower() == subj.lower() for e in extracted_entities):
                    extracted_entities.append({
                        "label": "PERP_REL",
                        "text": subj,
                        "start_char": m.start(1),
                        "end_char": m.end(1),
                    })

        return extracted_entities
