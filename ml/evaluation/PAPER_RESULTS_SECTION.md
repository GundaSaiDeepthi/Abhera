# 4. Experimental Results and Evaluation

This section presents the empirical evaluation of the **ABHERA** multi-label legal incident classification architecture and Named Entity Recognition (NER) pipeline. We evaluate classical machine learning baselines against fine-tuned bidirectional transformer representations (BERT) across held-out benchmark datasets. Furthermore, we analyze per-label decision thresholds, multi-label subset classification efficacy, entity-level extraction performance, and end-to-end system integration metrics.

---

## 4.1 Dataset and Experimental Setup

The empirical evaluation is conducted on the ABHERA Women's Safety Incident Corpus, which contains 1,086 manually verified and anonymized narrative reports covering six primary incident types: Domestic Violence (DV), Sexual Harassment (SH), Stalking (ST), Cyber Harassment / Abuse (CA), Workplace Harassment (WH), and Other Violence / Intimidation (OV). 

To prevent data leakage, the dataset was partitioned into stratified training (70%), validation (15%), and held-out test (15%) subsets using a fixed random seed ($S=42$). Table 1 summarizes the partition statistics of the benchmark dataset.

##### Table 1: ABHERA Benchmark Dataset Split Statistics
| Dataset Partition | Sample Count ($N$) | Percentage (%) | Functional Role in Experimentation |
| :--- | :---: | :---: | :--- |
| **Training Set** | 760 | 70.0% | Model parameter gradient optimization |
| **Validation Set** | 162 | 15.0% | Epoch termination & per-label threshold tuning |
| **Held-Out Test Set** | 164 | 15.0% | Independent final model benchmarking |
| **Total Corpus** | **1,086** | **100.0%** | Complete legal incident dataset |

All neural models were implemented using PyTorch and HuggingFace Transformers, utilizing the `bert-base-uncased` backbone architecture (110M parameters). Training was executed using the AdamW optimizer with a learning rate of $3 \times 10^{-5}$, weight decay of $0.01$, and binary cross-entropy with logits loss (`BCEWithLogitsLoss`) incorporating positive-class weighting.

---

## 4.2 Baseline Model Evaluation & Comparative Performance

We compare the optimized BERT architecture (BERT v2) against three reference models: classical Term Frequency-Inverse Document Frequency (TF-IDF) features paired with Logistic Regression, TF-IDF paired with Linear Support Vector Machines (SVM), and an initial neural baseline fine-tuned for 2 training epochs with a sequence length of 64 tokens (BERT v1).

Table 2 presents the quantitative results on the held-out test set ($N=164$).

##### Table 2: Comparative Classification Performance Across Baseline & Fine-Tuned Models
| Model Architecture | Exact Match Accuracy | Micro Precision | Micro Recall | Micro F1-Score | Macro F1-Score | Multi-Label Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 95.73% | 97.79% | 97.79% | 97.79% | 97.35% | 88.24% |
| **TF-IDF + Linear SVM** | 95.12% | 97.83% | 96.72% | 97.27% | 96.69% | 82.35% |
| **BERT v1 (2 Epochs, $L=64$)** | 83.54% | 95.76% | 87.29% | 91.33% | 90.37% | 0.00% |
| **BERT v2 (6 Epochs, $L=128$, Promoted)** | **95.73%** | **96.77%** | **99.45%** | **98.09%** | **97.58%** | **94.12%** |

As demonstrated in Table 2, classical TF-IDF baselines achieved strong performance on single-label incident narratives, with Logistic Regression reaching a Micro F1-score of 97.79%. However, BERT v1 exhibited clear under-fitting due to inadequate training epochs ($E=2$, 48 optimizer steps) and sequence truncation ($L=64$), resulting in a lower Micro F1-score of 91.33% and complete failure on multi-label instances (0.00% multi-label exact match).

BERT v2 successfully resolves these limitations. Extending training to 6 epochs and the maximum sequence length to 128 tokens, combined with validation-derived decision thresholds, yields a Micro F1-score of **98.09%** and a Macro F1-score of **97.58%**, outperforming all baseline models.

---

## 4.3 Fine-Tuned BERT v2 Improvement Analysis

Comparing the fine-tuned BERT v2 model directly to the BERT v1 baseline demonstrates substantial empirical gains across all evaluation metrics:

- **Micro F1-Score:** Increased from 91.33% to **98.09%**, representing an **absolute percentage-point improvement of +6.76 percentage points**.
- **Exact Match Accuracy:** Increased from 83.54% to **95.73%**, representing an **absolute percentage-point improvement of +12.19 percentage points**.
- **Macro F1-Score:** Increased from 90.37% to **97.58%**, representing an **absolute percentage-point improvement of +7.21 percentage points**.
- **Micro Recall:** Increased from 87.29% to **99.45%**, representing an **absolute percentage-point improvement of +12.16 percentage points**.
- **Multi-Label Subset Exact Match:** Increased from 0.00% to **94.12%**, representing an **absolute percentage-point improvement of +94.12 percentage points**.

Crucially for legal assistance systems where omitting valid allegations carries significant risk, BERT v2 reduced false negative instances from 23 in BERT v1 down to just **1** across the entire 164-sample test set—a **95.65% reduction in false-negative classification errors**.

---

## 4.4 Per-Class Decision Thresholds & Multi-Label Efficacy

To optimize multi-label classification boundaries without exposing held-out test data, per-class decision thresholds were calibrated using grid search ($[0.10, 0.90]$, step 0.01) exclusively on the validation set ($N=162$). Table 3 details the per-label performance on the held-out test set under these fixed validation-derived thresholds.

##### Table 3: Fine-Tuned BERT v2 Per-Label Evaluation & Validation Decision Thresholds
| Class Code | Legal Incident Category | Support | Threshold ($\tau$) | TP | FP | FN | TN | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DV** | Domestic Violence | 35 | 0.73 | 35 | 2 | 0 | 127 | 94.59% | 100.00% | 97.22% |
| **SH** | Sexual Harassment | 45 | 0.50 | 44 | 0 | 1 | 119 | 100.00% | 97.78% | 98.88% |
| **ST** | Stalking | 32 | 0.79 | 32 | 0 | 0 | 132 | 100.00% | 100.00% | 100.00% |
| **CA** | Cyber Harassment | 39 | 0.45 | 39 | 1 | 0 | 124 | 97.50% | 100.00% | 98.73% |
| **WH** | Workplace Harassment | 19 | 0.29 | 19 | 2 | 0 | 143 | 90.48% | 100.00% | 95.00% |
| **OV** | Other Violence | 11 | 0.63 | 11 | 1 | 0 | 152 | 91.67% | 100.00% | 95.65% |
| **OVERALL**| **Micro-Averaged Summary** | **181** | — | **180**| **6** | **1** | **797**| **96.77%**| **99.45%**| **98.09%** |

The effectiveness of BERT v2 across single-label versus complex multi-label survivor statements was further disaggregated. Table 4 summarizes the classification performance broken down by narrative complexity.

##### Table 4: Multi-Label vs. Single-Label Incident Classification Performance
| Incident Complexity Category | Sample Count ($N$) | Exact Match Count | Exact Match Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Single-Label Incident Reports** | 147 | 141 | 95.92% |
| **Multi-Label Incident Reports** | 17 | 16 | **94.12%** |
| **Total Held-Out Test Corpus** | **164** | **157** | **95.73%** |

Analysis of the single false-negative occurrence (`TEST_0050`) revealed a narrative combining cyber harassment and sexual harassment where the Sexual Harassment posterior probability ($p=0.4549$) fell slightly below its decision threshold ($\tau=0.50$), while Cyber Harassment was correctly identified ($p=0.9498$).

---

## 4.5 Named Entity Recognition (NER) Pipeline Evaluation

To complement intent classification, ABHERA deploys a fine-tuned token-classification NER model to extract critical evidentiary entities (Perpetrator, Location, Date/Time, Contact Details, Evidence) from victim statements. The NER pipeline was evaluated on an independent held-out dataset of 60 incident narratives containing 66 gold-annotated entity spans. Table 5 presents the entity-level performance metrics.

##### Table 5: Entity-Level Named Entity Recognition Evaluation
| Evaluation Metric | Quantitative Metric Value | Description |
| :--- | :---: | :--- |
| **Test Set Volume** | 60 records | Independent NER held-out evaluation corpus |
| **Gold Entity Spans** | 66 entities | Ground-truth annotated entity tokens |
| **Entity-Level Micro Precision** | **100.00%** | Zero false-positive entity extractions |
| **Entity-Level Micro Recall** | **98.48%** | 65 out of 66 gold entities extracted |
| **Entity-Level Micro-F1** | **99.24%** | Primary span extraction performance metric |
| **Active-Class Macro-F1** | **99.00%** | Unweighted mean across active entity categories |

As shown in Table 5, the NER extraction pipeline achieved an **entity-level Micro-F1 of 99.24%**, providing highly accurate structured entity metadata to downstream legal section mapping and report generation engines.

---

## 4.6 End-to-End System Integration & Latency Benchmarks

Following the successful evaluation of individual ML modules, the promoted BERT v2 model was integrated into the full ABHERA production backend. End-to-end integration tests were executed across all 11 core functional subsystems, backed by a 154-test regression suite. Table 6 provides latency benchmarks and validation results across the integrated architecture.

##### Table 6: End-to-End Subsystem Integration & Latency Benchmarks
| Subsystem Component | Functional Test Case | Status | Latency / Pass Metric |
| :--- | :--- | :---: | :---: |
| **BERT Multi-Label Classifier** | Intent classification & logit thresholding | **PASS** | $14.2 \text{ ms / sample}$ (GPU) |
| **NER Extraction Engine** | Entity extraction from raw narrative | **PASS** | $14.3 \text{ ms / sample}$ (GPU) |
| **Joint Intent + NER Pipeline** | Combined classification & entity parsing | **PASS** | $28.5 \text{ ms total}$ |
| **Dynamic Question Engine** | Contextual follow-up question synthesis | **PASS** | $100\% \text{ schema pass}$ |
| **Legal Knowledge Base Engine**| Statutory IPC / BNS mapping lookup | **PASS** | $100\% \text{ accuracy}$ |
| **Support Service Routing** | Emergency helpline & shelter matching | **PASS** | $100\% \text{ accuracy}$ |
| **Markdown Report Generator** | Incident summary compile engine | **PASS** | PASS |
| **PDF Rendering Pipeline** | Legal documentation export | **PASS** | PASS |
| **Database State Manager** | Session storage & relational sync | **PASS** | PASS |
| **REST API Server** | FastAPI endpoint handling | **PASS** | PASS |
| **Backend Regression Suite** | Automated integration test suite | **PASS** | **154 / 154 Passed (100%)** |

The integration evaluation confirms that BERT v2 achieves optimal classification accuracy while adding negligible runtime latency ($14.2\text{ ms}$), enabling real-time, responsive legal triage for survivor support.
