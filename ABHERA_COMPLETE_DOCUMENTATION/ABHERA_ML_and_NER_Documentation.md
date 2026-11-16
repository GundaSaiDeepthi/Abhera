# ABHERA Machine Learning & Named Entity Recognition Specification
**Project:** ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance  
**Model Architecture:** Bidirectional Encoder Representations from Transformers (BERT)  
**Base Pre-trained Weights:** `bert-base-uncased` (110 Million Parameters)  
**Primary Tasks:**  
1. Multi-Label Sequence Classification Head (`models/bert_multilabel_experiment_v2`)  
2. Token-Level Named Entity Recognition Head (`models/ner`)  

---

## 1. Deep Learning & Transformer Foundations

ABHERA rejects bag-of-words keyword matching in favor of **bidirectional self-attention transformer representations**. Harassment narratives are characterized by polysemy, syntactic negation, and subtle power dynamics (e.g., *"he said he loves me so much he follows me everywhere"* is legal stalking, not affection).

### 1.1 The Multi-Head Self-Attention Mechanism
The transformer encoder projects input tokens into Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices:
$$	ext{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = 	ext{softmax}\left( rac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} ight) \mathbf{V}$$
Across 12 attention heads ($d_k = 64$), BERT attends to distant tokens simultaneously, capturing syntactic dependencies regardless of word distance.

---

## 2. Multi-Label Classification Formulation

An incident disclosure frequently spans multiple legal infractions simultaneously (e.g., non-consensual sharing of intimate images by a spouse constitutes both **Domestic Violence [DV]** and **Cyber Abuse [CA]**).

### 2.1 Multi-Label Sigmoid Activation
Unlike multi-class softmax which enforces a mutually exclusive probability simplex ($\sum p_i = 1$), ABHERA computes independent binary posterior probabilities:
$$\hat{p}_c = \sigma(z_c) = rac{1}{1 + e^{-z_c}} \quad orall c \in \{	ext{DV, SH, ST, CA, WH, OV}\}$$

### 2.2 Positive-Weighted Binary Cross-Entropy Loss
In natural legal datasets, positive harassment instances are significantly rarer than negative non-occurrences. Standard unweighted BCE produces severe false negatives. ABHERA implements `BCEWithLogitsLoss` parameterized with positive class weights ($pos\_weight_c$):
$$\mathcal{L} = - rac{1}{C} \sum_{c=1}^C \left[ w_c \cdot y_c \log \sigma(z_c) + (1 - y_c) \log(1 - \sigma(z_c)) ight]$$
where $w_c = rac{N - N_c}{N_c}$.

---

## 3. Model Evolution: BERT v1 vs. Promoted BERT v2

```
       +-----------------------------------------------------------------------------------------+
       |                           BERT v1 vs. BERT v2 COMPARATIVE AUDIT                         |
       +-----------------------------------------------------------------------------------------+
       |  Hyperparameter / Metric      | BERT v1 (Legacy Pilot) | BERT v2 (Promoted Production)  |
       +-------------------------------+------------------------+--------------------------------+
       |  Base Architecture            | bert-base-uncased      | bert-base-uncased              |
       |  Training Epochs              | 2 Epochs (Underfit)    | 6 Epochs (Fully Converged)     |
       |  Max Sequence Length          | 64 Tokens (Truncated)  | 128 Tokens (Full Narrative)    |
       |  Batch Size                   | 32 Samples             | 32 Samples                     |
       |  Optimizer                    | AdamW (lr=3e-5)        | AdamW (lr=3e-5, weight_decay=0.01)
       |  Loss Function                | Unweighted BCE         | BCEWithLogitsLoss + pos_weight |
       |  Decision Threshold Strategy  | Fixed 0.50 Threshold   | Validation-Calibrated Grid Search
       |  Exact Match Accuracy (Test)  | 83.54% (0.8354)        | 95.73% (0.9573)  [+12.19 pts]  |
       |  Micro-Averaged F1 (Test)     | 91.33% (0.9133)        | 98.09% (0.9809)  [+6.76 pts]   |
       |  Micro-Averaged Recall (Test) | 87.29% (0.8729)        | 99.45% (0.9945)  [+12.16 pts]  |
       |  Multi-Label Exact Match      | 0.00% (0 / 17 exact)   | 94.12% (16 / 17 exact)         |
       |  Total False Negatives (FN)   | 23 Misses              | 1 Single Miss (TEST_0050)      |
       |  Total True Positives (TP)    | 158 Correct Hits       | 180 Correct Hits (+22 crimes)  |
       +-------------------------------+------------------------+--------------------------------+
```

---

## 4. Calibrated Decision Thresholds (`optimal_thresholds.json`)

To prevent arbitrary threshold bias, decision thresholds were calibrated using a 100-step grid search ($[0.10, 0.90]$) on held-out validation loss:

| Class Code | Full Legal Description | Calibrated Optimal Threshold ($	au_c$) | Rationale & Behavioral Effect |
| :---: | :--- | :---: | :--- |
| **DV** | Domestic Violence | **0.73** | High threshold suppresses ambiguous family disputes while capturing physical abuse. |
| **SH** | Sexual Harassment | **0.50** | Standard neutral threshold for balanced verbal and physical harassment. |
| **ST** | Stalking | **0.79** | High threshold prevents confusing benign casual encounters with continuous stalking. |
| **CA** | Cyber Abuse | **0.45** | Sensitive threshold capturing subtle online harassment and digital impersonation. |
| **WH** | Workplace Harassment | **0.29** | Ultra-sensitive threshold ensuring subtle corporate coercion and hostile work environments are detected. |
| **OV** | Other Violence | **0.63** | Moderated threshold for miscellaneous public assaults and weapon threats. |

---

## 5. Detailed Forensic Error Analysis: The Single False Negative (`TEST_0050`)

Out of 164 held-out test records evaluated in `ml/evaluation/FINAL_EVALUATION_RESULTS.json`, BERT v2 achieved 180 True Positives, 6 False Positives, and **exactly 1 False Negative**:

- **Instance ID:** `TEST_0050`
- **Narrative Text:** *"A man I met once at a conference keeps sending messages to my personal phone asking to meet."*
- **Ground Truth Labels:** `ST` (Stalking: 1), `SH` (Sexual Harassment: 1).
- **Model Predictions:**
  - `p_ST = 0.8412 \ge 0.79 \implies \mathbf{POSITIVE}$ (Stalking correctly detected).
  - `p_SH = 0.4549 < 0.50 \implies \mathbf{NEGATIVE}$ (Missed Sexual Harassment by 0.0451).
- **Forensic Diagnosis:** The narrative contains zero overtly vulgar or sexually explicit keywords; the harassment is implied by persistent unwanted romantic advances. The model assigned probability 0.4549, falling just 4.51 percentage points below the 0.50 threshold.
- **Production Safety Net:** In the live application, the Dynamic Question Engine triggers: *"Did these messages contain unwanted sexual advances, explicit remarks, or threats?"* When the user answers affirmatively, `SH` is immediately appended to the active session context.

---

## 6. Classical ML Baseline Comparisons

| Model Architecture | Feature Representation | Micro-F1 | Exact Match Accuracy | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | Word & Char N-Grams (5000 max features) | 0.9779 | 0.9573 | 0.9735 |
| **TF-IDF + Linear SVM** | Word & Char N-Grams (5000 max features) | 0.9727 | 0.9512 | 0.9669 |
| **BERT v1 (Legacy Pilot)** | Transformer Embeddings (64 tokens) | 0.9133 | 0.8354 | 0.9037 |
| **BERT v2 (Production Promoted)**| Bidirectional Self-Attention (128 tokens) | **0.9809** | **0.9573** | **0.9758** |

*Note on Supervised Feature Selection:* Empirical inspection confirmed that ABHERA utilizes **zero supervised feature selection** (`SelectKBest`, `chi2`, `RFE`). Vocabulary is constrained solely through unsupervised frequency pruning in `TfidfVectorizer(max_features=5000, min_df=2)`.

---

## 7. Named Entity Recognition (NER) Architecture & Benchmarks

ABHERA implements token-level sequence classification using `BertForTokenClassification` (`models/ner`).

### 7.1 The 13-Tag BIO Ontology
- `O`: Outside any legal entity.
- `B-PERP_REL` / `I-PERP_REL`: Perpetrator relationship (*"husband"*, *"colleague"*, *"ex-boyfriend"*).
- `B-LOCATION` / `I-LOCATION`: Physical location (*"Rajiv Chowk metro"*, *"office cafeteria"*).
- `B-TIME_FREQ` / `I-TIME_FREQ`: Temporal frequency (*"every evening"*, *"for 3 weeks"*).
- `B-PLATFORM` / `I-PLATFORM`: Digital channel (*"WhatsApp"*, *"Instagram DM"*, *"work email"*).
- `B-EVIDENCE` / `I-EVIDENCE`: Physical/digital proof (*"screenshots"*, *"audio recording"*).
- `B-LAW_SEC` / `I-LAW_SEC`: Statutory references (*"Section 354"*, *"BNS 69"*).

### 7.2 Subword Token Alignment & Offset Mapping
HuggingFace fast tokenizers split words into sub-tokens (e.g., *"WhatsApp"* $	o$ `['what', '##sap', '##p']`). `predict_ner.py` leverages token `offset_mapping` tuples `(start_char, end_char)` to reconstruct exact character spans in raw narrative strings:
```python
# Reconstructing character spans from WordPiece tokens
for token, (start, end), tag in zip(tokens, offsets, bio_tags):
    if tag.startswith("B-"):
        current_span = {"start": start, "end": end, "type": tag[2:]}
    elif tag.startswith("I-") and current_span:
        current_span["end"] = end
```

### 7.3 Empirical NER Test Set Evaluation (`models/ner/test_metrics.json`)
Evaluated across 60 annotated survivor narratives containing 66 gold-standard entity spans:

| Entity Category | Gold Spans | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PERP_REL`** | 24 | 24 | 0 | 0 | 100.0% | 100.0% | **1.0000** |
| **`LOCATION`** | 13 | 12 | 0 | 1 | 100.0% | 92.31% | **0.9600** |
| **`TIME_FREQ`** | 17 | 17 | 0 | 0 | 100.0% | 100.0% | **1.0000** |
| **`PLATFORM`** | 12 | 12 | 0 | 0 | 100.0% | 100.0% | **1.0000** |
| **OVERALL MICRO**| **66** | **65** | **0** | **1** | **100.0%** | **98.48%** | **0.9924** |
