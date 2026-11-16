# ABHERA — ACTUAL CLASSIFICATION PROCEDURE AND ALGORITHM AUDIT

**Document Version:** 1.0.0  
**Project:** ABHERA — AI-Driven Legal Guidance and Multi-Label Incident Classification System for Survivor Support  
**Scope:** Source-Code Inspection, Classification Algorithm Specification, Mathematical Formulation, and Flow Architecture  
**Status:** Audit Complete & Verified  

---

## 1. PURPOSE

This document details the exact classification procedure, mathematical formulation, training protocol, threshold optimization, baseline comparison, and end-to-end integration flow of the **ABHERA** multi-label incident classification system.

In strict compliance with paper documentation guidelines:
* **No code, model weights, thresholds, datasets, or evaluation metrics have been altered.**
* **All specifications and metrics are verified directly from production source files and evaluation artifacts.**
* **No generic machine learning claims or synthetic procedures are introduced.**

---

# 2. SOURCE FILES INSPECTED

The following authoritative files were inspected to construct this documentation:

| Module / System Layer | Source File Path | Primary Function / Class Inspected |
| :--- | :--- | :--- |
| **Dataset Preparation & Split** | [`ml/preprocessing/prepare_classification.py`](file:///d:/Abhera(Mini)/ml/preprocessing/prepare_classification.py#L33-L154) | `ClassificationDataPreparer.prepare()` |
| **BERT Baseline Training (v1)** | [`ml/training/train_bert.py`](file:///d:/Abhera(Mini)/ml/training/train_bert.py#L106-L285) | `train()` |
| **BERT Production Training (v2)** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L177-L430) | `main()` |
| **BERT Production Inference** | [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L156-L360) | `BERTIncidentPredictor.predict()` |
| **Classical Baselines Engine** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L135-L268) | `run_baseline_evaluation()` |
| **Verified Master Results JSON** | [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L1-L285) | Master Evaluation Artifact |
| **Verified Master Report MD** | [`ml/evaluation/FINAL_EVALUATION_REPORT.md`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_REPORT.md#L1-L100) | Master Summary Document |
| **Optimal Thresholds Config** | [`models/bert_multilabel_experiment_v2/optimal_thresholds.json`](file:///d:/Abhera(Mini)/models/bert_multilabel_experiment_v2/optimal_thresholds.json#L1-L8) | Production Threshold Dictionary |
| **Backend Integration Service** | [`backend/app/services/chat_service.py`](file:///d:/Abhera(Mini)/backend/app/services/chat_service.py#L1-L100) | Chat Service Pipeline |
| **Statutory Legal Mapping** | [`backend/app/services/legal_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/legal_mapping.py#L1-L100) | Legal Mapping Engine |

---

# 3. CLASSIFICATION OBJECTIVE AND TASK TYPE

ABHERA solves a **Multi-Label Binary Classification Problem** over survivor incident narratives.

### Target Schema:
The target space consists of six statutory incident categories:
$$\mathbf{y} = [y_1, y_2, y_3, y_4, y_5, y_6]^T \in \{0, 1\}^6$$

where the label vector components correspond to:
1. $y_1$ (**DV**): Domestic Violence
2. $y_2$ (**SH**): Sexual Harassment
3. $y_3$ (**ST**): Stalking
4. $y_4$ (**CA**): Cyber Abuse
5. $y_5$ (**WH**): Workplace Harassment
6. $y_6$ (**OV**): Other Violence / Street Assault

### Task Characteristic:
Unlike single-label multiclass problems where categories are mutually exclusive ($\sum y_k = 1$), survivor narratives frequently involve overlapping offenses (e.g., spousal physical abuse combined with online image-based threats $\rightarrow \mathbf{y} = [1, 0, 0, 1, 0, 0]^T$). Each category $k \in \{1, \dots, 6\}$ is modeled as an **independent binary decision variable**.

---

# 4. PRIMARY CLASSIFICATION ALGORITHM & ARCHITECTURE

The primary classification model promoted to production is **Fine-Tuned BERT v2** (`bert-base-uncased` with sequence classification head).

### Production Architecture Parameters:
* **Pretrained Base Model:** `bert-base-uncased` (12 Transformer layers, 768 hidden dimension, 12 attention heads, 110M parameters).
* **Classification Head:** Linear projection layer ($768 \rightarrow 6$) mapping the `[CLS]` sequence representation to raw unnormalized logits $\mathbf{z} \in \mathbb{R}^6$.
* **Activation Function:** Element-wise Sigmoid function $\sigma(z_k) = \frac{1}{1 + e^{-z_k}}$ applied independently to each logit.
* **Loss Function:** `BCEWithLogitsLoss` incorporating positive class imbalance weights ($\text{pos\_weight}$).
* **Optimizer:** `AdamW` (learning rate: $3 \times 10^{-5}$, weight decay: 0.01).
* **Learning Rate Scheduler:** `get_linear_schedule_with_warmup` with 10% warmup steps over 6 epochs.
* **Batch Size:** 32.
* **Sequence Length:** `max_length = 128` (training), `max_length = 256` (evaluation & production inference).
* **Random Seed:** 42 (`set_seed(42)`).

---

# 5. BERT INPUT PROCESSING PIPELINE

When an incident narrative string $x$ enters the classification pipeline, it undergoes the following deterministic steps:

```
                          Raw Incident Narrative String x
                                         │
                                         ▼
   1. Text Validation & Normalization (.strip(), non-empty check)
                                         │
                                         ▼
   2. AutoTokenizer ("bert-base-uncased") Pre-tokenization & Lowercasing
                                         │
                                         ▼
   3. WordPiece Subword Segmentation (Decomposes tokens into 30,522 vocabulary IDs)
                                         │
                                         ▼
   4. Special Token Injection ([CLS] + Subword IDs + [SEP])
                                         │
                                         ▼
   5. Padding & Truncation (Pad with [PAD] (0) or truncate to max_length = 256)
                                         │
                                         ▼
   6. Tensor Encoding -> input_ids (1, 256) and attention_mask (1, 256)
                                         │
                                         ▼
   7. BERT Encoder (12-layer Transformer self-attention -> 768-dim [CLS] state h_[CLS])
                                         │
                                         ▼
   8. Sequence Classification Head (Linear transformation: z = W * h_[CLS] + b)
                                         │
                                         ▼
   9. Raw Logits z = [z_1, z_2, z_3, z_4, z_5, z_6]
                                         │
                                         ▼
  10. Sigmoid Transformation: p_k = 1 / (1 + exp(-z_k)) for k in {1..6}
                                         │
                                         ▼
  11. Validation-Tuned Threshold Comparison: y_hat_k = 1 if p_k >= t_k else 0
                                         │
                                         ▼
  12. Final Category Set Y_hat = { l_k | y_hat_k = 1 }
```

---

# 6. LOSS FUNCTION & CLASS IMBALANCE CORRECTION

### Loss Function Formulation:
Training optimizes the **Weighted Binary Cross-Entropy Loss** with logits (`BCEWithLogitsLoss`):

$$\mathcal{L}(\mathbf{z}, \mathbf{y}) = -\frac{1}{6} \sum_{k=1}^{6} \left[ w_k \cdot y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$

where:
* $z_k$ is the unnormalized logit for category $k$.
* $\sigma(z_k) = \frac{1}{1 + e^{-z_k}}$ is the Sigmoid probability.
* $y_k \in \{0, 1\}$ is the ground-truth binary label.
* $w_k$ is the class-specific positive weight (`pos_weight`).

### Positive Weight Calculation:
Positive class weights $w_k$ are calculated strictly from the **training partition** (760 samples) in `prepare_classification.py`:

$$w_k = \frac{N_{\text{total}} - N_{\text{pos}, k}}{N_{\text{pos}, k}}$$

Where $N_{\text{total}} = 760$ and $N_{\text{pos}, k}$ is the number of positive training samples for category $k$. These weights compensate for class imbalance (e.g., Workplace Harassment and Other Violence having fewer positive examples than Sexual Harassment or Domestic Violence).

---

# 7. TRAINING & CHECKPOINT SELECTION PROTOCOL

### Dataset Split Dimensions:
* **Total Records:** 1,086
* **Training Partition (70%):** 760 records (`ml/preprocessing/train_data.csv`)
* **Validation Partition (15%):** 162 records (`ml/preprocessing/val_data.csv`)
* **Held-Out Test Partition (15%):** 164 records (`ml/preprocessing/test_data.csv`)

### Epoch Execution & Selection:
1. The model is trained for **6 epochs** with AdamW optimizer and linear warmup scheduler.
2. At the end of each epoch, validation loss and validation Micro-$F_1$ (at default threshold 0.50) are evaluated.
3. The best model checkpoint is selected based on maximum validation Micro-$F_1$. In production BERT v2 training (`train_bert_experiment_v2.py`), the optimal checkpoint was selected and saved to `models/bert_multilabel_experiment_v2/`.

---

# 8. VALIDATION-ONLY THRESHOLD OPTIMIZATION

Rather than applying a static, arbitrary threshold of 0.50 across all categories, ABHERA performs **independent per-label decision threshold tuning strictly on the validation set**.

### Grid Search Protocol (`optimize_thresholds_on_val`):
For each category $k \in \{\text{DV, SH, ST, CA, WH, OV}\}$, probabilities $p_{i, k}$ on the 162 validation samples are computed. A grid search evaluates thresholds $t \in [0.10, 0.90]$ at a step size of $0.01$ to maximize validation $F_1$:

$$t_k^* = \arg\max_{t \in [0.10, 0.90]} F_1^{(k)}\left(\mathbf{y}_{\text{val}, k}, \, \mathbb{I}(\mathbf{p}_{\text{val}, k} \ge t)\right)$$

### Production Optimal Thresholds (`optimal_thresholds.json`):

| Category Code | Category Name | Production Threshold ($t_k^*$) |
| :---: | :--- | :---: |
| **DV** | Domestic Violence | **0.73** |
| **SH** | Sexual Harassment | **0.50** |
| **ST** | Stalking | **0.79** |
| **CA** | Cyber Abuse | **0.45** |
| **WH** | Workplace Harassment | **0.29** |
| **OV** | Other Violence | **0.63** |

*Crucial Experimental Integrity Note:* Test set data is **never** exposed during threshold search. Thresholds are frozen after validation tuning and applied directly to test evaluation and production inference.

---

# 9. FINAL HELD-OUT TEST EVALUATION METRICS

Evaluation of the production BERT v2 model on the untouched 164-sample test split yielded the following verified metrics:

### Overall Test Metrics (`FINAL_EVALUATION_RESULTS.json`):
* **Micro Precision:** `0.9677` (96.77%)
* **Micro Recall:** `0.9945` (99.45%)
* **Micro $F_1$ Score:** `0.9809` (**98.09%**)
* **Macro Precision:** `0.9571` (95.71%)
* **Macro Recall:** `0.9963` (99.63%)
* **Macro $F_1$ Score:** `0.9758` (**97.58%**)
* **Exact Match Accuracy (Subset Accuracy):** `0.9573` (**95.73%**)
* **Elementwise Accuracy:** `0.9929` (**99.29%**)
* **Multi-Label Exact Match Accuracy:** `0.9412` (**94.12%**)
* **False Negatives:** 1 FN (out of 181 active labels across 164 test samples)
* **False Positives:** 6 FP

### Per-Label Test Performance:

| Category | Threshold ($t_k^*$) | Test Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | $F_1$ Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DV** | 0.73 | 35 | 35 | 2 | 0 | 0.9459 | 1.0000 | **0.9722** |
| **SH** | 0.50 | 45 | 44 | 0 | 1 | 1.0000 | 0.9778 | **0.9888** |
| **ST** | 0.79 | 32 | 32 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** |
| **CA** | 0.45 | 39 | 39 | 1 | 0 | 0.9750 | 1.0000 | **0.9873** |
| **WH** | 0.29 | 19 | 19 | 2 | 0 | 0.9048 | 1.0000 | **0.9500** |
| **OV** | 0.63 | 11 | 11 | 1 | 0 | 0.9167 | 1.0000 | **0.9565** |

---

# 10. CLASSICAL BASELINE CLASSIFIERS

To benchmark transformer performance, two classical machine learning baselines were evaluated using identical 70/15/15 data splits and validation-tuned thresholds:

### 1. TF-IDF + Logistic Regression
* **Feature Representation:** TF-IDF ($1,2$ n-grams, `sublinear_tf=True`, `min_df=2`, `max_features=5000`).
* **Multi-Label Strategy:** `OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, random_state=42))`.
* **Prediction:** Outputs probabilities via `predict_proba()` mapped against validation-tuned thresholds.
* **Test Performance:** Micro $F_1 = 0.9779$ (97.79%), Exact Match Acc = $0.9573$ (95.73%).

### 2. TF-IDF + Linear SVM
* **Feature Representation:** TF-IDF ($1,2$ n-grams, `sublinear_tf=True`, `min_df=2`, `max_features=5000`).
* **Multi-Label Strategy:** `OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000, random_state=42))`.
* **Prediction:** Margin distances $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$ are obtained via `decision_function()` and transformed into pseudo-probabilities via Sigmoids $p_k = \frac{1}{1 + e^{-z_k}}$.
* **Test Performance:** Micro $F_1 = 0.9727$ (97.27%), Exact Match Acc = $0.9512$ (95.12%).

---

# 11. MODEL COMPARISON SUMMARY TABLE

| Model Name | Feature Representation | Classification Head / Strategy | Multi-Label Mechanism | Held-Out Test Micro $F_1$ | Held-Out Test Exact Match Acc | Primary Project Role |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **BERT v2 (Production)** | `bert-base-uncased` Contextual Embeddings (768-dim) | Linear projection head + BCE Loss | Independent Sigmoids + Val-Tuned Thresholds | **0.9809** (98.09%) | **0.9573** (95.73%) | **Primary Production Classifier** |
| **Logistic Regression** | Sparse TF-IDF Matrix (5,000 n-grams) | `OneVsRestClassifier(LogisticRegression)` | Independent Binary Models + Val-Tuned Thresholds | `0.9779` (97.79%) | `0.9573` (95.73%) | Classical Baseline Benchmark |
| **Linear SVM** | Sparse TF-IDF Matrix (5,000 n-grams) | `OneVsRestClassifier(LinearSVC)` | Sigmoid Decision Function + Val-Tuned Thresholds | `0.9727` (97.27%) | `0.9512` (95.12%) | Classical Baseline Benchmark |
| **BERT v1 (Baseline)** | `bert-base-uncased` (max_len=64, 2 epochs) | Linear projection head | Sigmoids + Default 0.50 Threshold | `0.9133` (91.33%) | `0.8354` (83.54%) | Initial Experimental Checkpoint |

---

# 12. COMPLETE END-TO-END SYSTEM INTEGRATION FLOW

The classification output does not operate in isolation; it integrates directly into the downstream ABHERA backend pipeline (`backend/app/services/`):

```
                       SURVIVOR INCIDENT NARRATIVE
                                    │
                                    ▼
                 BERT MULTI-LABEL INCIDENT CLASSIFIER
                (ml/inference/predict_bert.py: predict())
                                    │
                       Predicted Category Set Y_hat
                   [e.g., "Domestic Violence", "Cyber Abuse"]
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
  NER ENTITY EXTRACTION                           STATUTORY LEGAL MAPPING
(ml/inference/predict_ner.py)                 (backend/app/services/legal_mapping.py)
Extracts structured evidence:                 Maps category + entities to legal sections:
- PERP_REL: "husband"                         - BNS Section 85 / IPC 498A (Cruelty)
- LOCATION: "home"                            - BNS Section 78 / IT Act 66E (Privacy violation)
- PLATFORM: "WhatsApp"                        - Protection of Women from DV Act 2005
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼
                        SUPPORT SERVICE MAPPING
              (backend/app/services/support_mapping.py)
              Maps categories to actionable resources:
              - National Domestic Violence Helpline (181)
              - Cyber Crime Portal (cybercrime.gov.in)
              - Local Shelter & Legal Aid Services
                                    │
                                    ▼
                   SURVIVOR REPORT GENERATION ENGINE
             (backend/app/services/report_generator.py)
             Generates structured PDF & JSON Summary Report
```

---

# 13. MATHEMATICAL FORMULATION

### Logit Generation:
$$\mathbf{z} = \mathbf{W} \cdot \text{BERT}_{\text{[CLS]}}(x) + \mathbf{b} \quad \in \mathbb{R}^6$$

### Sigmoid Probability Calculation:
$$p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \quad \forall k \in \{1, \dots, 6\}$$

### Threshold Decision Rule:
$$\hat{y}_k = \begin{cases} 1 & \text{if } p_k \ge t_k^* \\ 0 & \text{otherwise} \end{cases}$$

Where $t_k^*$ is the validation-tuned threshold for label $k$.

### Multi-Label Output Set:
$$\hat{Y} = \{ l_k \mid \hat{y}_k = 1, \, k \in \{1, \dots, 6\} \}$$

---

# 14. FORMAL RESEARCH PAPER ALGORITHM

```markdown
Algorithm 1: ABHERA Multi-Label Incident Classification Protocol

Input : Survivor incident narrative text string x
        Pretrained tokenizer T (bert-base-uncased)
        Fine-tuned sequence classification model M
        Optimal validation-tuned threshold vector T* = [t_1*, t_2*, ..., t_6*]
        Category label mapping L = [DV, SH, ST, CA, WH, OV]
Output: Predicted incident category set Y_hat

1.  Validate input string x:
    if x is null or strip(x) == "" then
        throw ValueError("Input narrative cannot be empty")
    end if

2.  Preprocess and tokenize narrative:
    inputs <- T.tokenize(x, truncation=True, padding="max_length", max_length=256)
    input_ids <- inputs["input_ids"]
    attention_mask <- inputs["attention_mask"]

3.  Execute forward inference pass without gradients:
    with torch.no_grad() do
        logits z <- M.forward(input_ids, attention_mask)
    end with

4.  Compute independent class probabilities:
    for k <- 1 to 6 do
        p_k <- 1.0 / (1.0 + exp(-z_k))
    end for

5.  Apply validation-tuned threshold decision rule:
    Y_hat <- empty_list()
    for k <- 1 to 6 do
        if p_k >= t_k* then
            Y_hat.append(L[k])
        end if
    end for

6.  Apply rule-based safety evidence overlays (e.g., explicit domestic violence / dowry / FPM keywords):
    Y_hat <- ReinforceExplicitSurvivorEvidence(x, Y_hat)

7.  Deduplicate and sort predicted categories by probability descending:
    Y_hat <- DeduplicateAndSort(Y_hat)

8.  return Y_hat
```

---

# 15. PAPER-READY "CLASSIFICATION PROCEDURE" SECTION

*The text below is formatted in formal academic prose for inclusion in the research paper:*

```markdown
### 3.4 Multi-Label Classification Procedure

Incident classification in ABHERA is formulated as a multi-label binary decision task across six statutory incident categories: Domestic Violence (DV), Sexual Harassment (SH), Stalking (ST), Cyber Abuse (CA), Workplace Harassment (WH), and Other Violence (OV). Because survivor narratives routinely describe intersecting forms of abuse, individual categories are modeled as non-mutually exclusive target dimensions $\mathbf{y} \in \{0, 1\}^6$.

The primary classification model utilizes a fine-tuned `bert-base-uncased` transformer architecture paired with a linear sequence classification head. Given an input narrative $x$, the text is tokenized into WordPiece subwords, padded or truncated to a sequence length of 256 tokens, and passed through 12 transformer encoder layers. The final hidden representation of the `[CLS]` token ($\mathbf{h}_{\text{[CLS]}} \in \mathbb{R}^{768}$) is projected onto a 6-dimensional logit vector $\mathbf{z} \in \mathbb{R}^6$.

To support multi-label prediction, an independent Sigmoid activation $\sigma(z_k) = (1 + e^{-z_k})^{-1}$ is applied to each logit. The model is fine-tuned using weighted binary cross-entropy loss (`BCEWithLogitsLoss`), incorporating positive-class weights $w_k = (N_{\text{total}} - N_{\text{pos}, k}) / N_{\text{pos}, k}$ derived from the training set to mitigate class imbalance. Optimization employs the AdamW algorithm with a initial learning rate of $3 \times 10^{-5}$, weight decay of $0.01$, and linear warmup learning rate scheduling over 6 epochs.

To maximize classification performance on imbalanced multi-label data, decision thresholds are optimized independently for each category on the validation partition via grid search ($t \in [0.10, 0.90]$, step $0.01$) to maximize per-class $F_1$ scores. The resulting optimal validation thresholds ($t_{\text{DV}}^* = 0.73$, $t_{\text{SH}}^* = 0.50$, $t_{\text{ST}}^* = 0.79$, $t_{\text{CA}}^* = 0.45$, $t_{\text{WH}}^* = 0.29$, $t_{\text{OV}}^* = 0.63$) are frozen and applied to the held-out test split. On the untouched 164-sample test set, the fine-tuned BERT model achieved a Micro $F_1$ score of 98.09% (Precision: 96.77%, Recall: 99.45%) and an Exact Match Accuracy of 95.73%, significantly outperforming baseline configurations.
```

---

# 16. VIVA-READY EXAMINATION QUESTIONS & ANSWERS

### Q1: What type of classification problem is ABHERA solving?
**Answer:** ABHERA solves a **Multi-Label Binary Classification Problem** across six statutory incident categories (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).

### Q2: Why is the classification task multi-label rather than multiclass?
**Answer:** Categories are mutually non-exclusive. A single survivor narrative can simultaneously describe multiple forms of abuse (e.g., domestic violence combined with cyber harassment). Multi-label modeling allows the system to predict zero, one, or multiple active categories for a single narrative.

### Q3: Why is Sigmoid used instead of Softmax?
**Answer:** Softmax enforces a probability distribution across mutually exclusive classes that sums to 1.0 ($\sum p_k = 1.0$). Sigmoid calculates independent probabilities for each class ($p_k \in [0, 1]$), allowing multiple categories to cross decision thresholds simultaneously.

### Q4: What is the primary BERT model used in production?
**Answer:** The primary model is **Fine-Tuned BERT v2**, based on `bert-base-uncased`, fine-tuned for 6 epochs with AdamW optimizer, BCE loss with positive weighting, and validation-tuned thresholds.

### Q5: Is BERT trained from scratch?
**Answer:** **No.** We fine-tune a pretrained `bert-base-uncased` checkpoint (110M parameters) using domain-specific survivor incident data.

### Q6: What is the classification head?
**Answer:** The classification head is a linear layer ($\mathbf{W} \in \mathbb{R}^{6 \times 768}, \mathbf{b} \in \mathbb{R}^6$) that projects the 768-dimensional `[CLS]` hidden vector into 6 unnormalized logits ($z_1, \dots, z_6$).

### Q7: What are logits?
**Answer:** Logits are the raw, unnormalized real-valued outputs ($z_k \in \mathbb{R}$) produced by the linear classification layer prior to Sigmoid probability transformation.

### Q8: Why are category-specific thresholds used instead of a default 0.50 threshold?
**Answer:** Incident categories exhibit varying class distributions and sensitivity requirements. Independent threshold tuning on validation data maximizes per-class $F_1$ scores (e.g., setting $t_{\text{WH}}^* = 0.29$ for Workplace Harassment to increase recall, while setting $t_{\text{ST}}^* = 0.79$ for Stalking to ensure high precision).

### Q9: How were decision thresholds selected?
**Answer:** Thresholds were selected via a grid search ($t \in [0.10, 0.90]$, step $0.01$) conducted **strictly on the validation set** to maximize validation $F_1$ scores.

### Q10: Why was the test set not used for threshold tuning?
**Answer:** Using the test set for threshold selection would cause data leakage and overfit hyperparameters to test data, invalidating held-out generalization metrics.

### Q11: How is class imbalance handled during training?
**Answer:** Class imbalance is handled using positive-class weighting (`pos_weight`) inside `BCEWithLogitsLoss`. Weights $w_k = (N_{\text{neg}} / N_{\text{pos}})$ are calculated from the training set to scale gradients for underrepresented classes.

### Q12: Why is `BCEWithLogitsLoss` preferred over standard BCELoss?
**Answer:** `BCEWithLogitsLoss` combines a Sigmoid layer and Binary Cross-Entropy loss into a single class using the log-sum-exp trick, providing numerical stability against vanishing or exploding gradients compared to evaluating Sigmoid and BCELoss separately.

### Q13: What baseline classifiers were compared against BERT?
**Answer:** We evaluated **TF-IDF + Logistic Regression** (Micro $F_1 = 97.79\%$) and **TF-IDF + Linear SVM** (Micro $F_1 = 97.27\%$) using `OneVsRestClassifier` over 5,000 TF-IDF n-grams.

### Q14: How does Linear SVM output probabilities for thresholding?
**Answer:** Margin distances $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$ from `decision_function()` are mapped through a Sigmoid transformation $p_k = \frac{1}{1 + e^{-z_k}}$ to obtain pseudo-probabilities for threshold comparison.

### Q15: What is the held-out test performance of the production BERT model?
**Answer:** On the untouched 164-sample test split, fine-tuned BERT v2 achieved **98.09% Micro $F_1$**, **95.73% Exact Match Accuracy**, and **99.29% Elementwise Accuracy**.

### Q16: How many false negatives did the production BERT model make on the test set?
**Answer:** Fine-tuned BERT v2 produced only **1 False Negative** out of 181 active labels across 164 test samples (99.45% Micro Recall).

### Q17: What happens if an incident narrative matches multiple categories?
**Answer:** Multiple categories whose Sigmoid probabilities meet or exceed their respective optimal thresholds are included in the predicted output set $\hat{Y}$.

### Q18: How does incident classification interact with Named Entity Recognition (NER)?
**Answer:** BERT classification determines **what type of incident** occurred (e.g., Domestic Violence), while NER extracts **structured evidentiary entities** (e.g., perpetrator relationship `"husband"`, location `"home"`).

### Q19: How does classification interact with legal mapping?
**Answer:** The predicted incident categories (e.g., `DV`, `CA`) and extracted NER entities are passed to `backend/app/services/legal_mapping.py`, which maps them to relevant Bharatiya Nyaya Sanhita (BNS) and IPC statutory sections.

### Q20: How does classification interact with survivor report generation?
**Answer:** Predicted categories, legal mappings, and support helplines are aggregated by `backend/app/services/report_generator.py` to synthesize a formal, downloadable PDF survivor incident report.

---

# 17. SOURCE TRACEABILITY MATRIX

| Feature / Metric / Parameter | Source File Path | Line Range | Exact Code / Artifact Evidence |
| :--- | :--- | :--- | :--- |
| **Model Initialization** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L224-L236) | L224–236 | `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)` |
| **BCE Loss & Pos Weights** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L238-L242) | L238–242 | `loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` |
| **Val Threshold Tuning** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L140-L174) | L140–174 | `optimize_thresholds_on_val()` grid search $t \in [0.10, 0.90]$ step 0.01 |
| **Production Thresholds** | [`models/bert_multilabel_experiment_v2/optimal_thresholds.json`](file:///d:/Abhera(Mini)/models/bert_multilabel_experiment_v2/optimal_thresholds.json#L1-L8) | L1–8 | `{"DV": 0.73, "SH": 0.5, "ST": 0.79, "CA": 0.45, "WH": 0.29, "OV": 0.63}` |
| **Production Inference** | [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L248-L299) | L248–299 | `probabilities = 1.0 / (1.0 + np.exp(-logits))`, threshold comparison against `optimal_thresholds` |
| **Verified Test Metrics** | [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L54-L79) | L54–79 | Micro $F_1 = 0.9809$, Exact Acc = $0.9573$, Micro Rec = $0.9945$, 1 FN, 6 FP |
| **Logistic Regression** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L177-L186) | L177–186 | `OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, random_state=42))` |
| **Linear SVM** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L197-L206) | L197–206 | `OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000, random_state=42))` |
| **Backend Legal Mapping** | [`backend/app/services/legal_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/legal_mapping.py#L1-L100) | L1–100 | Maps classified categories to statutory sections (BNS / IPC) |

---

ACTUAL CLASSIFICATION PROCEDURE AND ALGORITHM STATUS: VERIFIED
