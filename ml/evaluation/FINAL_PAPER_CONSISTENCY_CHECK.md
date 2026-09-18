# ABHERA — Final Paper Consistency Audit Report

> **Project:** ABHERA (AI-Powered Legal & Support Assistant for Women's Safety)  
> **Audit Type:** Step 7 Final Paper Consistency & Cross-Artifact Audit  
> **Date:** September 18, 2026  
> **Auditor:** Independent Automated Consistency Engine  
> **Audit Outcome:** PASS  

---

## 1. Files Checked

The following 10 authoritative evaluation artifacts were cross-audited for absolute mathematical, textual, structural, and numerical consistency:

1. [ml/evaluation/FINAL_EVALUATION_RESULTS.json](file:///d:/Abhera%28Mini%29/ml/evaluation/FINAL_EVALUATION_RESULTS.json) — Primary consolidated JSON metrics database
2. [ml/evaluation/FINAL_EVALUATION_REPORT.md](file:///d:/Abhera%28Mini%29/ml/evaluation/FINAL_EVALUATION_REPORT.md) — Comprehensive 16-section system evaluation report
3. [ml/evaluation/PAPER_RESULTS_SECTION.md](file:///d:/Abhera%28Mini%29/ml/evaluation/PAPER_RESULTS_SECTION.md) — Publication-ready academic paper results section
4. [ml/evaluation/final_model_comparison.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/final_model_comparison.csv) — Comparative metrics table (CSV)
5. [ml/evaluation/final_bert_per_label_results.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/final_bert_per_label_results.csv) — Per-label breakdown & decision threshold table (CSV)
6. [ml/evaluation/final_multilabel_results.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/final_multilabel_results.csv) — Multi-label vs single-label exact match table (CSV)
7. [ml/evaluation/final_ner_results.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/final_ner_results.csv) — Named entity extraction performance table (CSV)
8. [ml/evaluation/final_e2e_validation.csv](file:///d:/Abhera%28Mini%29/ml/evaluation/final_e2e_validation.csv) — System integration & latency benchmark table (CSV)
9. [ml/evaluation/bert_v2_validation_audit.json](file:///d:/Abhera%28Mini%29/ml/evaluation/bert_v2_validation_audit.json) — Independent Step 3 audit log
10. [ml/evaluation/abhera_end_to_end_validation.json](file:///d:/Abhera%28Mini%29/ml/evaluation/abhera_end_to_end_validation.json) — Step 5 end-to-end integration log

---

## 2. Metric Consistency Verification

All reported BERT v2 evaluation metrics were verified across all 10 files. Zero numerical discrepancies were detected:

- **Exact Match Accuracy:** **95.73%** (0.9573; 157 / 164 samples correct) — *100% Consistent*
- **Elementwise Binary Accuracy:** **99.29%** (0.9929; 977 / 984 binary decisions correct) — *100% Consistent*
- **Micro Precision:** **96.77%** (0.9677; TP=180, FP=6) — *100% Consistent*
- **Micro Recall:** **99.45%** (0.9945; TP=180, FN=1) — *100% Consistent*
- **Micro F1-Score:** **98.09%** (0.9809) — *100% Consistent*
- **Macro Precision:** **95.71%** (0.9571) — *100% Consistent*
- **Macro Recall:** **99.63%** (0.9963) — *100% Consistent*
- **Macro F1-Score:** **97.58%** (0.9758) — *100% Consistent*
- **Global Confusion Matrix Aggregates:** `TP=180`, `FP=6`, `FN=1`, `TN=797` — *100% Consistent*

---

## 3. Dataset Consistency Verification

The corpus partitioning and dataset statistics were audited across all JSON, Markdown, and CSV documentation:

- **Total Corpus Size:** 1,086 narrative records (*100% Consistent*)
- **Training Set:** 760 records (70.0%) (*100% Consistent*)
- **Validation Set:** 162 records (15.0%) (*100% Consistent*)
- **Held-Out Test Set:** 164 records (15.0%) (*100% Consistent*)
- **Random Seed:** `42` (*100% Consistent*)
- **Data Leakage Verification:** Confirmed 0 duplicate record IDs and 0 narrative text overlaps across train, validation, and test splits (*100% Consistent*).

---

## 4. BERT Consistency Verification

BERT v1 baseline and BERT v2 promoted model parameters and performance benchmarks were cross-referenced:

- **BERT v1 Baseline Metrics:** Exact Match = **83.54%**, Micro Precision = **95.76%**, Micro Recall = **87.29%**, Micro F1 = **91.33%**, Macro F1 = **90.37%**, Multi-Label Exact Match = **0.00%**, False Negatives = **23** (*100% Consistent*).
- **BERT v2 Promoted Metrics:** Exact Match = **95.73%**, Micro Precision = **96.77%**, Micro Recall = **99.45%**, Micro F1 = **98.09%**, Macro F1 = **97.58%**, Multi-Label Exact Match = **94.12%**, False Negatives = **1** (*100% Consistent*).
- **Absolute Percentage-Point Improvements:**
  - Micro F1: **+6.76 percentage points** (91.33% $\rightarrow$ 98.09%) (*100% Consistent*)
  - Exact Match Accuracy: **+12.19 percentage points** (83.54% $\rightarrow$ 95.73%) (*100% Consistent*)
  - Macro F1: **+7.21 percentage points** (90.37% $\rightarrow$ 97.58%) (*100% Consistent*)
  - Micro Recall: **+12.16 percentage points** (87.29% $\rightarrow$ 99.45%) (*100% Consistent*)
  - Multi-Label Exact Match: **+94.12 percentage points** (0.00% $\rightarrow$ 94.12%) (*100% Consistent*)
- **Per-Label Thresholds & Confusion Matrices:**
  - **DV:** Threshold = 0.73, Support = 35, TP = 35, FP = 2, FN = 0, TN = 127, Precision = 94.59%, Recall = 100.00%, F1 = 97.22% (*100% Consistent*)
  - **SH:** Threshold = 0.50, Support = 45, TP = 44, FP = 0, FN = 1, TN = 119, Precision = 100.00%, Recall = 97.78%, F1 = 98.88% (*100% Consistent*)
  - **ST:** Threshold = 0.79, Support = 32, TP = 32, FP = 0, FN = 0, TN = 132, Precision = 100.00%, Recall = 100.00%, F1 = 100.00% (*100% Consistent*)
  - **CA:** Threshold = 0.45, Support = 39, TP = 39, FP = 1, FN = 0, TN = 124, Precision = 97.50%, Recall = 100.00%, F1 = 98.73% (*100% Consistent*)
  - **WH:** Threshold = 0.29, Support = 19, TP = 19, FP = 2, FN = 0, TN = 143, Precision = 90.48%, Recall = 100.00%, F1 = 95.00% (*100% Consistent*)
  - **OV:** Threshold = 0.63, Support = 11, TP = 11, FP = 1, FN = 0, TN = 152, Precision = 91.67%, Recall = 100.00%, F1 = 95.65% (*100% Consistent*)

---

## 5. NER Consistency Verification

The Named Entity Recognition pipeline evaluation metrics were checked across all reporting files:

- **Held-out Test Records:** 60 records (*100% Consistent*)
- **Gold Entity Spans:** 66 annotated spans (*100% Consistent*)
- **Entity-Level Micro Precision:** **100.00%** (*100% Consistent*)
- **Entity-Level Micro Recall:** **98.48%** (*100% Consistent*)
- **Entity-Level Micro-F1:** **99.24%** (*100% Consistent*)
- **Active-Class Macro-F1:** **99.00%** (*100% Consistent*)

---

## 6. Baseline Consistency Verification

Classical machine learning baseline benchmarks were verified across `FINAL_EVALUATION_RESULTS.json`, `FINAL_EVALUATION_REPORT.md`, `PAPER_RESULTS_SECTION.md`, and `final_model_comparison.csv`:

- **TF-IDF + Logistic Regression:** Exact Match = **95.73%**, Micro Precision = **97.79%**, Micro Recall = **97.79%**, Micro F1 = **97.79%**, Macro F1 = **97.35%**, Multi-Label Exact Match = **88.24%** (*100% Consistent*).
- **TF-IDF + Linear SVM:** Exact Match = **95.12%**, Micro Precision = **97.83%**, Micro Recall = **96.72%**, Micro F1 = **97.27%**, Macro F1 = **96.69%**, Multi-Label Exact Match = **82.35%** (*100% Consistent*).

---

## 7. Multi-Label Consistency Verification

Multi-label vs. single-label performance disaggregation was verified:

- **Single-Label Incident Reports:** $N=147$, Exact Match = 141, Accuracy = **95.92%** (*100% Consistent*)
- **Multi-Label Incident Reports:** $N=17$, Exact Match = 16, Accuracy = **94.12%** (*100% Consistent*)
- **Combined Test Set:** $N=164$, Exact Match = 157, Accuracy = **95.73%** (*100% Consistent*)
- **Remaining False Negative Instance (`TEST_0050`):** Ground truth `[SH, CA]`, predicted `[CA]`, Sexual Harassment probability $p=0.4549$ vs. decision threshold $\tau=0.50$ (*100% Consistent*).

---

## 8. End-to-End Consistency Verification

System integration test results and regression benchmarks were cross-verified:

- **End-to-End Integration Status:** **PASS** across all 11 sub-systems (BERT Inference API, NER Coexistence, Dynamic Question Engine, Legal Knowledge Base, Support Service Mapper, Executive Report Assembly, PDF Renderer, Database Consistency, FastAPI Routes, Anti-Hallucination Guard, Regression Suite) (*100% Consistent*).
- **Backend Regression Suite:** **154 / 154 Passed (100.0% Pass Rate)** (*100% Consistent*).

---

## 9. Terminology Check

The terminology across all markdown and manuscript documents was audited for strict scientific accuracy:

1. **Micro-F1 Nomenclature:** BERT classification performance is consistently specified as **Micro-F1** (or Micro F1-Score), avoiding ambiguous metric references (*Verified*).
2. **Entity-Level NER Nomenclature:** The 99.24% NER metric is explicitly titled **entity-level Micro-F1** across all documents (*Verified*).
3. **Exact Match Nomenclature:** Exact match multi-label accuracy is explicitly titled **Exact Match** or **Exact Match Accuracy** (*Verified*).
4. **Percentage-Point Improvements:** Improvements from BERT v1 to BERT v2 are strictly described as **absolute percentage-point improvements** (e.g., +6.76 percentage points for Micro-F1, +12.19 percentage points for Exact Match), ensuring zero confusion with relative percentage gains (*Verified*).
5. **Zero Test-Set Contamination Claims:** Audits explicitly confirm that decision thresholds were computed solely on the validation set (*Verified*).

---

## 10. Any Discrepancies Found

**NONE.**

An exhaustive line-by-line and value-by-value programmatic audit across all 10 target files revealed **0 numerical discrepancies**, **0 terminology violations**, **0 missing metrics**, and **0 invalid data claims**.

---

## 11. Final Status

The final paper consistency check has successfully completed. All evaluation artifacts, CSV packages, JSON schemas, system audit logs, and publication-ready markdown files are 100% mathematically, textually, and structurally aligned.

---

FINAL PAPER CONSISTENCY STATUS: PASS
