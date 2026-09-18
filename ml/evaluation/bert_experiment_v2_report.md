# ABHERA — BERT PERFORMANCE IMPROVEMENT: STEP 2 CONTROLLED TRAINING EXPERIMENT REPORT

**Experiment Name:** `BERT_MultiLabel_Experiment_v2`  
**Model Architecture:** `bert-base-uncased` (Multi-Label Classification)  
**Dataset:** ABHERA Incident Dataset (`data/womens_safety_dataset.csv`)  
**Date of Execution:** September 18, 2026  
**Status:** Experiment Completed & Verified  

---

## 1. Executive Summary & Headline Results

In Step 2, we conducted a controlled training experiment on a new, isolated experimental model checkpoint (`models/bert_multilabel_experiment_v2/`) without modifying the held-out test split, production model, or existing baseline artifacts.

### Headline Results:
- **Micro-F1 Score:** Increased from **91.33%** to **98.09%** (**+6.76% gain**).
- **Exact Match Accuracy:** Increased from **83.54%** to **95.73%** (**+12.19% gain**).
- **Micro Recall:** Increased from **87.29%** to **99.45%** (**+12.16% gain** — False Negatives reduced from 23 down to 1!).
- **Micro Precision:** Increased from **95.76%** to **96.77%** (**+1.01% gain**).
- **Macro-F1 Score:** Increased from **90.37%** to **97.58%** (**+7.21% gain**).
- **Multi-Label Exact Match:** Increased from **0.00%** (0/17) to **94.12%** (16/17) (**+94.12% gain**).

> [!IMPORTANT]
> **Key Benchmark Outcome:**  
> Experimental BERT v2 (**98.09% Micro-F1**) now **outperforms BOTH classical baselines**:
> - TF-IDF + Logistic Regression: `97.79%` Micro-F1
> - TF-IDF + Linear SVM: `97.27%` Micro-F1
> - **Experimental BERT v2:** **`98.09%` Micro-F1**

---

## 2. Experimental Training Configuration

| Parameter / Setting | Existing Baseline (BERT v1) | Experimental BERT v2 | Rationale & Change |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | `bert-base-uncased` | `bert-base-uncased` | Unchanged |
| **Random Seed** | `42` | `42` | Unchanged for 100% reproducibility |
| **Train / Val / Test Split** | 760 / 162 / 164 | 760 / 162 / 164 | Exact same split (`seed=42`) |
| **Training Sequence Length** | `max_length = 64` | **`max_length = 128`** | Increased to prevent narrative truncation |
| **Training Epochs** | `2` Epochs (48 steps) | **`6` Epochs (144 steps)** | Extended to allow transformer convergence |
| **Batch Size** | `32` | `32` | Unchanged |
| **Learning Rate** | `3e-5` (Linear Warmup) | `3e-5` (Linear Warmup) | Unchanged |
| **Weight Decay** | `0.01` (`AdamW`) | `0.01` (`AdamW`) | Unchanged |
| **Loss Function** | `BCEWithLogitsLoss` | `BCEWithLogitsLoss` | Unchanged (using original pos-weights) |
| **Model Selection Protocol** | Epoch 2 Checkpoint | **Val Micro-F1 Checkpoint Selection** | Saved best epoch based on Val Micro-F1 |
| **Threshold Tuning** | Val Grid Search | **Val Grid Search on Best Checkpoint** | Tuned on Validation Split ONLY |

---

## 3. Training & Validation Progress Across Epochs

Evaluation was performed after every epoch on the Validation Set (`val_data.csv`) using the default decision threshold ($t = 0.50$) to select the best checkpoint without looking at the test set.

| Epoch | Train Loss | Val Loss | Val Exact Match Acc | Val Micro Prec | Val Micro Rec | Val Micro F1 | Val Macro F1 | Best Checkpoint? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.0643 | 0.9330 | 1.23% | 35.15% | 93.33% | 51.06% | 51.33% | Yes |
| **2** | 0.7894 | 0.6344 | 16.05% | 51.04% | 95.00% | 66.41% | 65.62% | Yes |
| **3** | 0.5162 | 0.4188 | 51.85% | 69.53% | 98.89% | 81.65% | 82.51% | Yes |
| **4** | 0.3494 | 0.3136 | 76.54% | 82.71% | 98.33% | 89.85% | 91.04% | Yes |
| **5** | 0.2722 | 0.2695 | 82.10% | 86.41% | 98.89% | 92.23% | 93.12% | Yes |
| **6** | **0.2409** | **0.2575** | **82.72%** | **86.83%** | **98.89%** | **92.47%** | **93.30%** | **Selected Best (Epoch 6)** |

> **Best Epoch Selection:** Epoch 6 was selected as the best checkpoint with a Validation Micro-F1 of **92.47%** (Val Loss: 0.2575).

---

## 4. Validation Set Threshold Optimization

After selecting the Epoch 6 model checkpoint, independent per-label decision threshold optimization was executed on the **Validation Set ONLY** across grid $t \in [0.10, 0.90]$ (step 0.01) to maximize per-label F1:

| Label Code | Category Description | Existing Baseline Threshold | Experimental Optimized Threshold | Validation Max F1 |
| :---: | :--- | :---: | :---: | :---: |
| **DV** | Domestic Violence | 0.65 | **0.73** | 1.0000 |
| **SH** | Sexual Harassment | 0.60 | **0.50** | 0.9890 |
| **ST** | Stalking | 0.64 | **0.79** | 0.9778 |
| **CA** | Cyber Abuse | 0.72 | **0.45** | 0.9855 |
| **WH** | Workplace Harassment | 0.54 | **0.29** | 1.0000 |
| **OV** | Other Violence | 0.78 | **0.63** | 0.9756 |

The newly derived thresholds were saved to `models/bert_multilabel_experiment_v2/optimal_thresholds.json`.

---

## 5. Final Held-Out Test Set Results (Evaluated ONCE)

Only after model checkpoint selection and validation threshold tuning were complete, the experimental model was evaluated **ONCE** on the untouched held-out test dataset (`test_data.csv`, N=164).

### 5.1 Global Metrics Comparison Table

| Metric | Existing Baseline BERT | Experimental BERT v2 | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | 83.54% | **95.73%** | **+12.19%** |
| **Elementwise Binary Acc** | 96.95% | **99.29%** | **+2.34%** |
| **Micro Precision** | 95.76% | **96.77%** | **+1.01%** |
| **Micro Recall** | 87.29% | **99.45%** | **+12.16%** |
| **Micro F1 Score** | **91.33%** | **98.09%** | **+6.76%** |
| **Macro Precision** | 95.82% | **95.71%** | -0.11% |
| **Macro Recall** | 86.45% | **99.63%** | **+13.18%** |
| **Macro F1 Score** | 90.37% | **97.58%** | **+7.21%** |
| **Single-Label Exact Match** | 93.20% | **95.92%** | **+2.72%** |
| **Multi-Label Exact Match** | 0.00% (0/17) | **94.12% (16/17)** | **+94.12%** |

### 5.2 Per-Label Detailed Performance & Confusion Matrix (Experimental BERT v2)

| Label | Threshold | Support | TP | FP | FN | TN | Precision | Recall | F1 Score | Baseline F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DV** | 0.73 | 35 | 35 | 2 | 0 | 127 | 94.59% | 100.00% | **97.22%** | 87.88% |
| **SH** | 0.50 | 45 | 44 | 0 | 1 | 119 | 100.00% | 97.78% | **98.88%** | 91.57% |
| **ST** | 0.79 | 32 | 32 | 0 | 0 | 132 | 100.00% | 100.00% | **100.00%** | 89.66% |
| **CA** | 0.45 | 39 | 39 | 1 | 0 | 124 | 97.50% | 100.00% | **98.73%** | 96.20% |
| **WH** | 0.29 | 19 | 19 | 2 | 0 | 143 | 90.48% | 100.00% | **95.00%** | 92.68% |
| **OV** | 0.63 | 11 | 11 | 1 | 0 | 152 | 91.67% | 100.00% | **95.65%** | 84.21% |
| **Total / Micro** | - | **181** | **180** | **6** | **1** | **797** | **96.77%** | **99.45%** | **98.09%** | **91.33%** |

---

## 6. Answers to Explicit Requirements (Questions H & L)

1. **Exact Files Created:**
   - [models/bert_multilabel_experiment_v2/](file:///d:/Abhera%28Mini%29/models/bert_multilabel_experiment_v2/) (Model checkpoint, tokenizer, config)
   - [models/bert_multilabel_experiment_v2/optimal_thresholds.json](file:///d:/Abhera%28Mini%29/models/bert_multilabel_experiment_v2/optimal_thresholds.json)
   - [models/bert_multilabel_experiment_v2/training_config.json](file:///d:/Abhera%28Mini%29/models/bert_multilabel_experiment_v2/training_config.json)
   - [ml/evaluation/bert_experiment_v2_results.json](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_experiment_v2_results.json)
   - [ml/evaluation/bert_experiment_v2_history.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_experiment_v2_history.csv)
   - [ml/evaluation/bert_experiment_v2_comparison.json](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_experiment_v2_comparison.json)
   - [ml/evaluation/bert_experiment_v2_report.md](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_experiment_v2_report.md)

2. **Exact Training Configuration:**
   - Architecture: `bert-base-uncased`
   - Epochs: `6` | Max Length: `128` | Batch Size: `32`
   - Learning Rate: `3e-5` | Weight Decay: `0.01`
   - Loss Function: `BCEWithLogitsLoss` with original positive class weights (`pos_weights`: DV=3.344, SH=2.719, ST=5.101, CA=3.763, WH=5.503, OV=10.082)
   - Random Seed: `42`

3. **Validation Metrics for Every Epoch:**
   - Epoch 1: Val Micro-F1 = 51.06%, Val Exact Acc = 1.23%
   - Epoch 2: Val Micro-F1 = 66.41%, Val Exact Acc = 16.05%
   - Epoch 3: Val Micro-F1 = 81.65%, Val Exact Acc = 51.85%
   - Epoch 4: Val Micro-F1 = 89.85%, Val Exact Acc = 76.54%
   - Epoch 5: Val Micro-F1 = 92.23%, Val Exact Acc = 82.10%
   - Epoch 6: Val Micro-F1 = **92.47%**, Val Exact Acc = 82.72%

4. **Best Epoch Selected:** **Epoch 6** (Val Micro-F1: 92.47%)

5. **New Validation Thresholds:**
   - DV: `0.73` | SH: `0.50` | ST: `0.79` | CA: `0.45` | WH: `0.29` | OV: `0.63`

6. **Final Held-Out Test Metrics:**
   - Micro-F1: **98.09%** | Exact Match Acc: **95.73%**
   - Micro Precision: **96.77%** | Micro Recall: **99.45%** | Macro-F1: **97.58%**

7. **Comparison Against Existing BERT:**
   - Micro-F1: **98.09%** vs **91.33%** (**+6.76%**)
   - Exact Match Acc: **95.73%** vs **83.54%** (**+12.19%**)

8. **Micro-F1 Performance Status:** **Significantly Improved** (+6.76% gain).
9. **Micro Recall Status:** **Significantly Improved** (from 87.29% to **99.45%**; False Negatives dropped from 23 to 1!).
10. **Multi-Label Performance Status:** **Dramatically Improved** (from 0.00% to **94.12%** exact match accuracy on multi-label test cases).
11. **Training Warnings / Errors:** None. Standard HuggingFace transformer load notice regarding newly initialized sequence classification head (`classifier.weight` / `classifier.bias`), which is standard when initializing sequence classification heads from pre-trained masked language models.
12. **Commands Executed:**
   - `backend\venv\Scripts\python.exe ml\training\train_bert_experiment_v2.py`
