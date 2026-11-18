"""
Module for Chapter 9: Named Entity Recognition
"""

def get_chapter_09():
    return r'''# CHAPTER 9: Named Entity Recognition (NER)

---

## 9.1 What is Named Entity Recognition (NER)?

**Named Entity Recognition (NER)** is a specialized token-level sequence classification task in Natural Language Processing. While sentence classification looks at an entire paragraph and assigns a single high-level label (e.g., "This text describes Stalking"), NER examines every single word or sub-word token individually and determines whether it belongs to a pre-defined informational category.

In ABHERA, the distinction between classification and NER is fundamental:
- **BERT Multi-Label Classification**: Answers *"What crimes occurred?"* (e.g., Stalking + Cyber Abuse).
- **BERT Named Entity Recognition**: Answers *"Who, where, when, and through what means?"* (e.g., "ex-boyfriend", "Instagram", "every evening for 2 weeks").

```
SENTENCE CLASSIFICATION (One Label per Narrative):
"My manager touched me inappropriately in the office elevator yesterday."
  |
  +---> [ BERT Multi-Label Classifier ] ---> Predicted: Sexual Harassment (SH), Workplace Harassment (WH)

NAMED ENTITY RECOGNITION (Word-by-Word Sequence Tagging):
" My     manager      touched me inappropriately in the  office elevator   yesterday ."
  |         |            |     |        |         |  |         |             |       |
  O    B-PERP_REL        O     O        O         O  O     B-LOCATION    B-TIME_FREQ O
```

---

## 9.2 Why NER is Needed in Legal Informatics

When a survivor visits a police station or files an official complaint with an Internal Complaints Committee (ICC), the authority cannot register a First Information Report (FIR) based purely on an abstract category. An FIR legally requires **factual, evidentiary particulars**:

1. **Identity / Relationship of the Accused**: Is the perpetrator a family member (triggering Domestic Violence Act / Section 85 BNS), a workplace supervisor (triggering POSH Act), or an unknown stranger?
2. **Jurisdiction / Locus Delicti (Location)**: Which police station or court has territorial jurisdiction over the crime?
3. **Temporal Duration / Frequency**: Did this happen once, or is it a persistent pattern of harassment over months? Under BNS Section 78 (Stalking), establishing repeated or continuous behavior is essential for criminal prosecution.
4. **Medium / Instrument of Crime**: Was an electronic device, WhatsApp, or Instagram used? This dictates whether Information Technology Act sections (Section 66E, 67) must be invoked alongside the penal code.

If ABHERA only classified the category, the system would have to blindly ask 10+ tedious follow-up questions for every disclosure. By running NER first, ABHERA automatically extracts facts the survivor has already provided, asking follow-up questions only for what is genuinely missing.

---

## 9.3 The 6 Entity Types Used in ABHERA

ABHERA defines an ontology of **6 legal entity classes**:

| Entity Tag | Entity Description | Real-World Examples (Fictionalized) | Legal Importance |
| :--- | :--- | :--- | :--- |
| **`PERP_REL`** | Perpetrator identity or relationship to survivor | `"husband"`, `"manager"`, `"colleague"`, `"landlord"`, `"ex-boyfriend"`, `"stranger"` | Dictates whether PWDVA (domestic), POSH (workplace), or BNS applies. |
| **`LOCATION`** | Physical scene of incident | `"office canteen"`, `"hostel gate"`, `"metro station"`, `"elevator"`, `"in-laws house"` | Establishes police station jurisdiction and workplace boundary definitions. |
| **`TIME_FREQ`** | Temporal duration or frequency | `"every evening"`, `"for the past 3 weeks"`, `"last Friday night"`, `"multiple times a day"` | Proves continuity of stalking or cruelty; establishes statute of limitations. |
| **`PLATFORM`** | Communication channel or app | `"WhatsApp"`, `"Instagram"`, `"Telegram"`, `"official company email"`, `"phone call"` | Triggers Section 66E / 67 / 67A IT Act 2000 and digital evidence preservation. |
| **`EVIDENCE`** | Mentions of physical/digital proof | `"screenshots"`, `"CCTV footage"`, `"audio recording"`, `"medical report"`, `"chat backup"` | Forensically cataloged in Section 8 of the generated legal dossier. |
| **`LAW_SEC`** | Explicit statutory references | `"Section 78"`, `"Section 354A"`, `"POSH Act"`, `"Domestic Violence Act"` | Mapped directly into the formal legal complaint framework. |

---

## 9.4 The 13-Tag BIO Annotation Ontology

In Natural Language Processing, entities often span multiple consecutive words (e.g., *"office conference room"* or *"for the past two months"*). If we simply tagged each word as `LOCATION`, the computer would not know if `"office"` and `"conference room"` were two separate locations or one continuous location!

To solve this, ABHERA uses the standardized **BIO (Begin, Inside, Outside)** tagging scheme:
- **`B-<ENTITY>`**: Assigned to the first word/token of an entity span.
- **`I-<ENTITY>`**: Assigned to any subsequent tokens belonging to that same continuous span.
- **`O`**: Assigned to all non-entity tokens (outside).

With 6 entity types plus the outside tag, ABHERA's label space contains exactly **13 discrete BIO tags**:

```json
{
  "0": "O",
  "1": "B-PERP_REL",   "2": "I-PERP_REL",
  "3": "B-LOCATION",   "4": "I-LOCATION",
  "5": "B-TIME_FREQ",  "6": "I-TIME_FREQ",
  "7": "B-PLATFORM",   "8": "I-PLATFORM",
  "9": "B-EVIDENCE",   "10": "I-EVIDENCE",
  "11": "B-LAW_SEC",   "12": "I-LAW_SEC"
}
```

---

## 9.5 WordPiece Sub-Word Offset Alignment

A major technical challenge in BERT token classification arises from **sub-word tokenization**. 

When a word like `"Instagram"` is tokenized by BERT's fast tokenizer, it might be split into sub-word tokens: `['insta', '##gram']`. If the model tags `insta` as `B-PLATFORM` and `##gram` as `I-PLATFORM`, how do we map this back to the exact slice of text in the user's original message?

ABHERA solves this in `ml/inference/predict_ner.py` using **character offset mapping**:
1. When `tokenizer(text, return_offsets_mapping=True)` is called, the tokenizer produces a tuple of `(start_character_index, end_character_index)` for every sub-token.
2. During prediction, the model outputs a BIO tag for each token.
3. The span extractor groups contiguous `B-` and `I-` tags and looks up the start character of the `B-` token and the end character of the final `I-` token.
4. It then slices the survivor's original string: `original_text[start_char : end_char]`. This guarantees 100% spelling preservation, capitalization preservation, and punctuation accuracy.

---

## 9.6 Model Architecture and Training Details

- **Base Model**: `bert-base-uncased` with `AutoModelForTokenClassification`.
- **Classification Head**: Dropout ($p=0.1$) followed by a linear projection layer (`torch.nn.Linear(768, 13)`).
- **Optimizer**: AdamW with learning rate $3 \times 10^{-5}$ and weight decay $0.01$.
- **Training Epochs**: 12 epochs over annotated incident datasets (`models/ner/training_config.json`).
- **Batch Size**: 16 with maximum sequence length 256 tokens.
- **Weights Artifact**: `models/ner/model.safetensors` (415.4 MB).

---

## 9.7 Empirical NER Evaluation Benchmarks

The NER model was evaluated on held-out test records (`models/ner/test_metrics.json`), achieving near-perfect entity extraction:

```
+===================================================================================================+
|                                    ABHERA NER TEST BENCHMARKS                                     |
+---------------------+-------------------+-------------------+-------------------+-----------------+
| Entity Dimension    | True Spans (Gold) | True Positives    | False Negatives   | Recall (%)      |
+---------------------+-------------------+-------------------+-------------------+-----------------+
| PERP_REL            | 24                | 24                | 0                 | 100.0%          |
| TIME_FREQ           | 17                | 17                | 0                 | 100.0%          |
| PLATFORM            | 12                | 12                | 0                 | 100.0%          |
| LOCATION            | 13                | 12                | 1                 | 92.31%          |
+---------------------+-------------------+-------------------+-------------------+-----------------+
| OVERALL MICRO       | 66 Gold Spans     | 65 Detected (TP)  | 1 Missed (FN)     | 98.48% Recall   |
| OVERALL PRECISION   | 65 Detections     | 65 Valid (0 FP)   | 0 False Positives | 100.0% Precision|
| OVERALL MICRO-F1    |                   |                   |                   | 99.24% F1-SCORE |
+===================================================================================================+
```

*Note on Error Analysis*: The single missed location span occurred in an unusually phrased compound sentence where the location was described metaphorically.

---

## 9.8 Fictional Walkthrough Example of NER Extraction

Consider this fictional survivor narrative submitted to ABHERA:

> *"My ex-husband has been calling my phone every single night for the past three weeks and threatening to come to my office in Hyderabad."*

### Step 1: WordPiece Tokenization & BIO Tagging
```
Token:        [CLS]   my      ex       -       husband    has     been    calling
Tag:            O      O   B-PERP_REL I-PERP_REL I-PERP_REL  O        O        O

Token:        my     phone   every    single   night     for     the     past     three   weeks
Tag:          O    B-PLATFORM B-TIME_FREQ I-TIME_FREQ I-TIME_FREQ I-TIME_FREQ I-TIME_FREQ I-TIME_FREQ I-TIME_FREQ I-TIME_FREQ

Token:       and   threatening  to     come     to      my     office      in    hyderabad   [SEP]
Tag:          O         O        O       O       O       O   B-LOCATION    O    B-LOCATION     O
```

### Step 2: Extracted Entity Dictionary Returned by `predict_ner.py`:
```json
{
  "PERP_REL": ["ex-husband"],
  "PLATFORM": ["phone"],
  "TIME_FREQ": ["every single night for the past three weeks"],
  "LOCATION": ["office", "Hyderabad"]
}
```

### Step 3: How the Backend Uses This Output:
1. **The Dynamic Question Engine** checks required slots for Domestic Violence (`DV`) and Stalking (`ST`):
   - `PERP_REL`: Found ("ex-husband"). **Slot fulfilled!** Do not ask.
   - `PLATFORM`: Found ("phone"). **Slot fulfilled!** Do not ask.
   - `TIME_FREQ`: Found ("every single night..."). **Slot fulfilled!** Do not ask.
   - `LOCATION`: Found ("office", "Hyderabad"). **Slot fulfilled!** Do not ask.
   - `SAFETY_STATUS`: **Missing!**
2. The Question Engine asks **only one follow-up question**:
   > *"Are you currently in immediate physical danger, or do you need emergency support right now?"*
3. The survivor is spared from answering four redundant questions about facts they had already explained!
'''
