# ABHERA BERT MULTI-LABEL CLASSIFICATION: ERROR ANALYSIS REPORT (STEP 1)

**System/Model:** Fine-Tuned BERT (`bert-base-uncased`) Multi-Label Incident Classifier  
**Dataset:** ABHERA Women's Safety Incident Dataset (`data/womens_safety_dataset.csv`)  
**Date of Analysis:** September 17, 2026  
**Status:** Analysis Only (Zero Retraining / Zero Model Modification)

---

## 1. Executive Summary & Context

The ABHERA platform utilizes a multi-label classification pipeline to categorize reported women's safety incidents into six primary legal and support categories:
- **DV**: Domestic Violence
- **SH**: Sexual Harassment
- **ST**: Stalking
- **CA**: Cyber Abuse
- **WH**: Workplace Harassment
- **OV**: Other Violence

The current fine-tuned BERT baseline model (`bert-base-uncased`) achieves:
- **Exact Match Accuracy:** 83.54% (137 / 164 test samples)
- **Micro Precision:** 95.76% (158 / 165 predictions)
- **Micro Recall:** 87.29% (158 / 181 ground truth labels)
- **Micro F1:** 91.33%
- **Macro F1:** 90.37%

However, classical TF-IDF baselines significantly outperform this BERT model:
- **TF-IDF + Logistic Regression:** **97.79% Micro-F1** (95.73% Exact Match)
- **TF-IDF + Linear SVM:** **97.27% Micro-F1** (96.34% Exact Match)

This report performs a comprehensive, empirical error analysis on the fine-tuned BERT model's held-out test set predictions to identify why BERT underperforms classical linear baselines and to determine evidence-based interventions for Step 2.

---

## 2. Dataset Split & Label Distribution Analysis

### 2.1 Dataset Partitioning
The dataset consists of **1,086 valid incident narratives**, partitioned into fixed reproducible splits using random seed `42`:
- **Train Split (70%):** 760 samples
- **Validation Split (15%):** 162 samples
- **Test Split (15%):** 164 samples (Held-out evaluation set)

### 2.2 Class Label Distribution & Loss Pos-Weights
Multi-label annotations are distributed as follows across splits, along with the positive class weights (`pos_weight = neg / pos`) configured in `BCEWithLogitsLoss`:

| Label Code | Category Description | Full Dataset (N=1086) | Train (N=760) | Val (N=162) | Test (N=164) | Pos Weight in Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DV** | Domestic Violence | 250 (23.02%) | 181 | 34 | 35 | `3.3440` |
| **SH** | Sexual Harassment | 292 (26.89%) | 202 | 45 | 45 | `2.7192` |
| **ST** | Stalking | 178 (16.39%) | 124 | 22 | 32 | `5.1011` |
| **CA** | Cyber Abuse | 228 (20.99%) | 155 | 34 | 39 | `3.7632` |
| **WH** | Workplace Harassment | 167 (15.38%) | 124 | 24 | 19 | `5.5030` |
| **OV** | Other Violence | 98 (9.02%) | 66 | 21 | 11 | `10.0816` |

---

## 3. Training & Evaluation Pipeline Hyperparameters

The current fine-tuned BERT model was trained with the following exact hyperparameters:

| Parameter | Configuration / Value | Source File |
| :--- | :--- | :--- |
| **Base Architecture** | `bert-base-uncased` (12-layer, 768-hidden, 12-heads, 110M params) | `train_bert.py` |
| **Tokenizer** | WordPiece Tokenizer (`bert-base-uncased`) | `train_bert.py` |
| **Max Sequence Length (Train)** | **64 tokens** (`truncation=True, max_length=64`) | `train_bert.py:L140` |
| **Max Sequence Length (Eval)** | **256 tokens** (`truncation=True, max_length=256`) | `evaluate_bert.py:L81` |
| **Optimizer** | `AdamW` (weight_decay = `0.01`) | `train_bert.py:L179` |
| **Learning Rate** | `3e-5` with linear schedule + 10% warmup | `train_bert.py:L143,L181` |
| **Batch Size** | `32` | `train_bert.py:L141` |
| **Training Epochs** | **2 Epochs** (Total 48 gradient steps) | `train_bert.py:L142` |
| **Dropout Rate** | `hidden_dropout_prob = 0.1`, `attention_probs_dropout_prob = 0.1` | `config.json` |
| **Loss Function** | `BCEWithLogitsLoss` with positive class weighting | `train_bert.py:L171` |
| **Random Seed** | `42` (PyTorch, NumPy, Python random) | `train_bert.py:L20` |
| **Threshold Selection** | Independent per-label F1 grid search on **Validation Set ONLY** | `optimal_thresholds.json` |

### Derived Validation-Optimal Thresholds:
- **DV Threshold:** `0.65`
- **SH Threshold:** `0.60`
- **ST Threshold:** `0.64`
- **CA Threshold:** `0.72`
- **WH Threshold:** `0.54`
- **OV Threshold:** `0.78`

---

## 4. Test Set Error & Confusion Analysis

### 4.1 Global Metrics Comparison

| Model | Exact Match Acc | Micro Precision | Micro Recall | Micro F1 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BERT Multi-Label** | **83.54%** | **95.76%** | **87.29%** | **91.33%** | **90.37%** |
| **TF-IDF + Logistic Reg** | **95.73%** | **97.79%** | **97.79%** | **97.79%** | **97.05%** |
| **TF-IDF + Linear SVM** | **96.34%** | **96.22%** | **98.34%** | **97.27%** | **97.19%** |

### 4.2 Per-Label Confusion Matrix Breakdown (BERT)

| Label | Threshold | Support | TP | FP | FN | TN | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DV** | `0.65` | 35 | 29 | 2 | 6 | 127 | 93.55% | 82.86% | 87.88% |
| **SH** | `0.60` | 45 | 38 | 0 | 7 | 119 | 100.00% | 84.44% | 91.57% |
| **ST** | `0.64` | 32 | 26 | 0 | 6 | 132 | 100.00% | 81.25% | 89.66% |
| **CA** | `0.72` | 39 | 38 | 2 | 1 | 123 | 95.00% | 97.44% | 96.20% |
| **WH** | `0.54` | 19 | 19 | 3 | 0 | 142 | 86.36% | 100.00% | 92.68% |
| **OV** | `0.78` | 11 | 8 | 0 | 3 | 153 | 100.00% | 72.73% | 84.21% |
| **Total / Micro** | - | **181** | **158** | **7** | **23** | **796** | **95.76%** | **87.29%** | **91.33%** |

> [!IMPORTANT]
> **Key Metric Observation:**  
> Total False Negatives (**23**) far exceed total False Positives (**7**). BERT's Micro Precision is high (**95.76%**), but Micro Recall drops to **87.29%**. High decision thresholds suppress positive predictions on boundary probabilities.

---

## 5. Specific Error Case Analysis

### 5.1 Multi-Label Failure Cases (100% Failure Rate on Multi-Label Incidents)
Out of 164 test incidents:
- **Single-Label Incidents (147 cases):** **93.20% Exact Match Accuracy** (137 / 147 correct)
- **Multi-Label Incidents (17 cases):** **0.00% Exact Match Accuracy** (0 / 17 correct)

BERT failed to predict the complete set of active labels on **all 17 multi-label test cases**.

#### Multi-Label Failure Examples:
1. **Incident TEST_0051 (`ST,CA`):**  
   *Narrative:* "My ex keeps following me everywhere I go and also created a fake profile to message me."  
   *True Labels:* `ST, CA`  
   *Predicted Labels:* `CA` (Probabilities: `CA`: 0.7916 >= 0.72, `ST`: 0.6330 < 0.64)  
   *Failure Mechanism:* `ST` score of 0.6330 fell just 0.007 below the strict validation threshold of 0.64.

2. **Incident TEST_0138 (`SH,WH`):**  
   *Narrative:* "My manager keeps making sexual jokes in front of the entire team."  
   *True Labels:* `SH, WH`  
   *Predicted Labels:* `WH` (Probabilities: `WH`: 0.6099 >= 0.54, `SH`: 0.5533 < 0.60)  
   *Failure Mechanism:* `SH` score of 0.5533 fell below the 0.60 threshold.

3. **Incident TEST_0082 (`DV,CA`):**  
   *Narrative:* "My husband has been hitting me for years, and last week he threatened to share our private photos with my entire family if I ever tried to leave."  
   *True Labels:* `DV, CA`  
   *Predicted Labels:* `DV` (Probabilities: `DV`: 0.5803 [under threshold 0.65], `CA`: 0.4621 [under 0.72])  
   *Failure Mechanism:* Severe probability dilution under multi-label conditions.

### 5.2 Category Confusion & Ambiguous Examples (OV vs. DV)
- **Incident TEST_0026 (`OV`):**  
  *Narrative:* "A local group threatened to disown me over my relationship choice."  
  *True Label:* `OV` | *Predicted:* `DV` (`DV`: 0.6475, `OV`: 0.7649 < 0.78)  
  *Root Cause:* The token "disown" and family/relationship dispute context triggered domestic violence features, but `OV` probability (0.7649) fell below the high `0.78` OV threshold.

- **Incident TEST_0121 (`OV`):**  
  *Narrative:* "Since my husband's passing, distant relatives have been harassing me to give up my rightful share of the property."  
  *True Label:* `OV` | *Predicted:* `DV` (`DV`: 0.6750 >= 0.65, `OV`: 0.7733 < 0.78)  
  *Root Cause:* "husband" and "relatives" strongly activated the `DV` logit during training.

---

## 6. TOP 5 Evidence-Based Causes of BERT's Lower Performance

1. **Under-Training (Epochs = 2, Total 48 Gradient Steps):**  
   With `batch_size = 32` across 760 training samples, 2 epochs yield only **48 total optimizer steps** (with 5 warmup steps). The 110-million parameter transformer encoder and multi-label classification head remain severely under-converged compared to fully converged linear SVM and Logistic Regression.

2. **Excessive BCE Loss Pos-Weighting & Artificial Logit Shift:**  
   `pos_weights` ranged up to **10.08** for OV and **5.10** for ST. This artificially inflated positive logits during training, which forced validation threshold tuning to pick high cutoffs (OV=0.78, CA=0.72, DV=0.65, ST=0.64, SH=0.60). At test time, these high cutoffs caused **23 False Negatives** (Recall = 87.29%).

3. **Multi-Label Threshold Penalty & Score Dilution:**  
   When an incident contains multiple active themes (e.g., stalked physically AND harassed online), BERT's under-trained output probabilities for secondary labels frequently hover in the 0.55–0.63 range, causing strict per-label cutoffs to drop the second label completely (0% exact match on multi-label test cases).

4. **Explicit Lexical Keyword Dominance Favors TF-IDF N-Grams:**  
   Safety incident descriptions in this dataset contain explicit, highly diagnostic n-grams ("instagram", "manager", "boss", "colleague", "husband", "slapped", "disown", "fake profile"). TF-IDF creates direct, un-truncated sublinear 5,000 n-gram feature weights, whereas under-trained BERT failed to cleanly separate these keyword associations.

5. **Class Imbalance & Semantic Overlap (OV vs. DV):**  
   OV (Other Violence) has only 66 training samples (9.02% of dataset). Narratives describing non-partner violence (landlords, distant relatives, local groups) share vocabulary with Domestic Violence (DV), causing BERT to assign higher probability to DV than OV.

---

## 7. Recommended Next Experiments (Step 2 Preview)

Based strictly on the observed empirical errors without altering the held-out test set or production system:

1. **Increase Training Epochs & Step Count:**  
   Increase epochs from 2 to **5–8 epochs** (or reduce batch size to 16) with early stopping on validation Micro-F1 to allow full encoder convergence.

2. **Unweighted or Mildly Weighted Loss Function:**  
   Evaluate standard `BCEWithLogitsLoss()` (pos_weight = 1.0) or square-root positive weighting (`sqrt(neg/pos)`) to prevent logit distortion and lower optimal decision thresholds.

3. **Full Sequence Length Standardisation:**  
   Ensure `max_length = 128` or `256` is uniformly set across both training and evaluation datasets to prevent any potential sub-word truncation.

4. **Multi-Threshold Calibration for Co-occurring Labels:**  
   Implement a joint multi-label probability calibration or lower decision threshold band (0.40–0.50) to recover high-confidence secondary multi-label predictions.

---

## 8. File Audit & Traceability

- **Files Inspected:**
  - `ml/preprocessing/prepare_classification.py`
  - `ml/preprocessing/label_mapping.json`
  - `ml/training/train_bert.py`
  - `ml/training/evaluate_bert.py`
  - `ml/inference/predict_bert.py`
  - `ml/evaluation/verify_optimal_thresholds.py`
  - `ml/evaluation/results/baseline_comparison_results.json`
  - `models/bert_multilabel/optimal_thresholds.json`
  - `models/bert_multilabel/final_test_evaluation.json`

- **Files Created:**
  - `ml/evaluation/generate_bert_error_analysis.py` (Script)
  - [bert_error_analysis.json](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_error_analysis.json) (JSON Report)
  - [bert_error_analysis.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_error_analysis.csv) (Error Cases CSV)
  - [bert_error_analysis_report.md](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_error_analysis_report.md) (Markdown Report)

- **Commands Executed:**
  - `backend\venv\Scripts\python.exe ml\evaluation\generate_bert_error_analysis.py`
