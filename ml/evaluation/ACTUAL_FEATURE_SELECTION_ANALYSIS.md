# ABHERA — ACTUAL FEATURE SELECTION IMPLEMENTATION AUDIT

**Document Version:** 1.0.0  
**Project:** ABHERA — AI-Driven Legal Guidance and Multi-Label Incident Classification System for Survivor Support  
**Scope:** Strict Source-Code Inspection and Documentation of Feature Selection, Feature Representation, and Feature Space Boundaries  
**Status:** Audit Complete & Verified  

---

## 1. PURPOSE

The purpose of this document is to provide a rigorous, code-authoritative audit of feature selection and feature representation in the ABHERA project.

In strict accordance with academic integrity and inspection guidelines:
* **No code, models, datasets, or evaluation results have been modified.**
* **No synthetic or assumed feature selection algorithms are introduced.**
* The distinction between **data preprocessing**, **feature extraction/representation**, **feature-space limitation**, **dimensionality reduction**, and **supervised feature selection** is explicitly maintained.
* Where a feature selection algorithm is absent from the codebase, it is explicitly declared as **"Not used in the current implementation."**

---

# 2. SOURCE FILES INSPECTED

A complete scan of the ABHERA repository was conducted across all classification, preprocessing, baseline evaluation, transformer training, inference, and NER scripts:

| Module / Pipeline Stage | Source File Path | Key Functions / Classes Inspected | Feature Processing Behavior |
| :--- | :--- | :--- | :--- |
| **Dataset Preparation** | [`ml/preprocessing/prepare_classification.py`](file:///d:/Abhera(Mini)/ml/preprocessing/prepare_classification.py#L33-L154) | `ClassificationDataPreparer.prepare()` | Data cleaning, valid row filtering, multi-hot target label formatting, 70/15/15 dataset split. |
| **Dataset Analysis** | [`ml/analysis/dataset_analysis.py`](file:///d:/Abhera(Mini)/ml/analysis/dataset_analysis.py#L24-L160) | `run_dataset_analysis()` | Label distribution, duplicate narrative checks, co-occurrence analysis. |
| **Baseline Feature Extraction & Evaluation** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L163-L206) | `run_baseline_evaluation()` | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=5000)` fitting and transformation. |
| **Logistic Regression Evaluator** | [`ml/evaluation/baseline_logistic_regression.py`](file:///d:/Abhera(Mini)/ml/evaluation/baseline_logistic_regression.py#L1-L8) | Main entry script | Delegates directly to `run_baseline_comparison.py`. |
| **Linear SVM Evaluator** | [`ml/evaluation/baseline_linear_svm.py`](file:///d:/Abhera(Mini)/ml/evaluation/baseline_linear_svm.py#L1-L8) | Main entry script | Delegates directly to `run_baseline_comparison.py`. |
| **BERT Training (v1)** | [`ml/training/train_bert.py`](file:///d:/Abhera(Mini)/ml/training/train_bert.py#L29-L56) | `WomensSafetyDataset.__getitem__()` | `bert-base-uncased` tokenization with `max_length=64`. |
| **BERT Training (v2 Production)** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L29-L56) | `WomensSafetyDataset.__getitem__()` | `bert-base-uncased` tokenization with `max_length=128/256`. |
| **BERT Production Inference** | [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L248-L267) | `BERTIncidentPredictor.predict()` | Tokenization with `max_length=256`, forward pass without gradients, Sigmoid computation. |
| **NER Data Preparation** | [`ml/ner/prepare_training_data.py`](file:///d:/Abhera(Mini)/ml/ner/prepare_training_data.py#L45-L102) | `align_spans_to_bio_tokens()` | Subword alignment via `return_offsets_mapping=True`, 13-tag BIO token tag assignment. |
| **NER Training** | [`ml/ner/train_ner.py`](file:///d:/Abhera(Mini)/ml/ner/train_ner.py#L86-L124) | `NERDataset.__getitem__()` | Token classification dataset prep, `-100` label masking for special tokens. |
| **NER Production Inference** | [`ml/inference/predict_ner.py`](file:///d:/Abhera(Mini)/ml/inference/predict_ner.py#L137-L160) | `NEREntityPredictor.predict()` | Subword argmax decoding and character offset span re-alignment. |
| **Leakage & Similarity Audit** | [`ml/evaluation/leakage_audit/run_leakage_audit.py`](file:///d:/Abhera(Mini)/ml/evaluation/leakage_audit/run_leakage_audit.py#L153-L175) | `run_full_audit()` | TF-IDF n-gram vectorizer fitting across full corpus for cross-split cosine similarity audit. |

---

# 3. CONCEPTUAL DISTINCTIONS IN THE ML PIPELINE

To prevent academic ambiguity, the ABHERA machine learning workflow is demarcated into four distinct operational concepts:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPROCESSING                                                                  │
│    Operations: Input validation, missing narrative filtering, binary multi-hot target  │
│    matrix generation, 70/15/15 train/val/test split partition.                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. FEATURE REPRESENTATION / FEATURE EXTRACTION (USED)                                  │
│    - Classical Models: TF-IDF vectorization over unigrams/bigrams (5,000 dimensions).  │
│    - Deep Learning: BERT subword tokenization & 768-dim contextual Transformer states. │
│    - NER: WordPiece token-level BIO tag sequence representation.                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. FEATURE-SPACE LIMITATION / BOUNDING (USED)                                          │
│    - TF-IDF min_df=2: Ignores terms appearing in fewer than 2 documents (unsupervised).│
│    - TF-IDF max_features=5000: Bounds vocabulary to top 5,000 terms by corpus-wide    │
│      term frequency (unsupervised, non-target-conditioned).                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. SUPERVISED FEATURE SELECTION (NOT USED IN CURRENT IMPLEMENTATION)                  │
│    Statistical target-conditioned feature subset selection (e.g., SelectKBest, Chi2,  │
│    Mutual Information, ANOVA F-test, RFE) is NOT implemented in the codebase.          │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. DIMENSIONALITY REDUCTION (NOT USED IN CURRENT IMPLEMENTATION)                      │
│    Subspace projection algorithms (e.g., PCA, TruncatedSVD) are NOT implemented.      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 4. CLASSICAL BASELINE FEATURE SPACE ANALYSIS

In [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L165-L175), the TF-IDF feature space for Logistic Regression and Linear SVM is configured as follows:

```python
# Exact implementation in ml/evaluation/run_baseline_comparison.py
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_features=5000,
)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_val_vec   = vectorizer.transform(X_val_raw)
X_test_vec  = vectorizer.transform(X_test_raw)
```

### Empirical Verification of Parameters:
* **`ngram_range=(1, 2)`:** Extracts both individual words (unigrams) and contiguous word pairs (bigrams).
* **`sublinear_tf=True`:** Applies logarithmic scaling $TF = 1 + \ln(tf)$ for $tf > 0$ to subdue repetitive term frequency noise.
* **`min_df=2`:** Frequency filtering that excludes tokens appearing in fewer than 2 documents across the corpus. This filters single-occurrence typos based purely on document frequency ($df$).
* **`max_features=5000`:** Vocabulary bounding that limits the matrix column dimension to 5,000 features based on highest term frequency ($tf$) across the training set.
* **Train-Only Fitting:** `vectorizer.fit_transform(X_train_raw)` is executed **strictly on training data**. Validation and test matrices are transformed using `vectorizer.transform()`, preventing feature vocabulary leakage.
* **Post-Extraction Selection:** **None.** All 5,000 extracted TF-IDF features are passed directly into `OneVsRestClassifier(LogisticRegression(...))` and `OneVsRestClassifier(LinearSVC(...))` without any target-conditioned feature pruning or statistical ranking.

---

# 5. BERT FEATURE REPRESENTATION ANALYSIS

In [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L224-L250) and [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L248-L267), BERT feature representation operates as follows:

1. **Tokenization:** Text is tokenized using `AutoTokenizer.from_pretrained("bert-base-uncased")`. Words are segmented into subwords using WordPiece tokenization with a 30,522 token vocabulary.
2. **Input Preparation:** Prepares integer `input_ids` and binary `attention_mask` tensors padded or truncated to `max_length=128` (training) or `max_length=256` (test/inference).
3. **Deep Contextual State Representation:** The fine-tuned transformer encoder maps input sequences to 768-dimensional contextual hidden representations via multi-head self-attention.
4. **Manual Feature Selection:** **None.** No manual token filtering, manual feature pruning, dense dimensionality reduction (PCA/SVD), or handcrafted feature concatenation is performed. The final `[CLS]` sequence representation is fed directly into the binary classification head.

---

# 6. NER FEATURE REPRESENTATION ANALYSIS

In [`ml/ner/prepare_training_data.py`](file:///d:/Abhera(Mini)/ml/ner/prepare_training_data.py#L45-L102) and [`ml/ner/train_ner.py`](file:///d:/Abhera(Mini)/ml/ner/train_ner.py#L86-L124), NER feature representation operates as follows:

1. **Subword Alignment:** Uses HuggingFace tokenizer offset mappings (`return_offsets_mapping=True`) to align character-level entity spans (`start`, `end`, `label`) with WordPiece subword tokens.
2. **BIO Tag Assignment:** Subwords matching entity character bounds are assigned `B-<LABEL>` (first subword) or `I-<LABEL>` (subsequent subwords). Unmatched tokens receive `O`. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are assigned label `-100` to mask them from loss calculations.
3. **Feature Selection:** **None.** Token-level predictions are generated directly from the 768-dimensional token hidden states of `AutoModelForTokenClassification` without explicit feature selection or handcrafted feature engineering.

---

# 7. FEATURE SELECTION ALGORITHM REPOSITORY AUDIT

A repository-wide search was executed to check for any imports or invocations of feature-selection or dimensionality-reduction algorithms across all Python source files:

```bash
# Grep search query across repository Python files:
SelectKBest | chi2 | mutual_info | f_classif | RFE | RFECV | VarianceThreshold | PCA | TruncatedSVD | feature_selection
```

**Audit Result:** Zero occurrences found in Python source code.

### Detailed Audit Summary Table:

| Technique | Status | Implemented Code Pattern | Source File Path | Technical Classification & Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **`SelectKBest`** | **NOT USED** | None | N/A | Supervised univariate feature selection is absent. |
| **Chi-Square ($\chi^2$) Test** | **NOT USED** | None | N/A | Target-conditioned $\chi^2$ feature ranking is absent. |
| **Mutual Information** | **NOT USED** | None | N/A | Information gain / mutual information filtering is absent. |
| **ANOVA F-Test** | **NOT USED** | None | N/A | `f_classif` feature selection is absent. |
| **RFE / RFECV** | **NOT USED** | None | N/A | Recursive feature elimination is absent. |
| **Variance Threshold** | **NOT USED** | None | N/A | Unsupervised low-variance feature pruning is absent. |
| **Principal Component Analysis (PCA)** | **NOT USED** | None | N/A | Linear dense dimensionality reduction is absent. |
| **Truncated SVD** | **NOT USED** | None | N/A | LSA / TruncatedSVD matrix reduction is absent. |
| **L1-Lasso Feature Selection** | **NOT USED** | None | N/A | L1-regularized sparse feature selection is absent. |
| **TF-IDF Vectorization** | **USED** | `TfidfVectorizer(...)` | `ml/evaluation/run_baseline_comparison.py` | **FEATURE EXTRACTION / REPRESENTATION:** Converts raw text into a 5,000-dim sparse matrix. |
| **`max_features=5000`** | **USED** | `max_features=5000` | `ml/evaluation/run_baseline_comparison.py` | **FEATURE-SPACE LIMITATION:** Bounds term vocabulary to top 5,000 terms by corpus frequency (unsupervised). |
| **`min_df=2`** | **USED** | `min_df=2` | `ml/evaluation/run_baseline_comparison.py` | **FEATURE-SPACE LIMITATION:** Ignores terms appearing in fewer than 2 documents (unsupervised frequency filtering). |
| **BERT WordPiece** | **USED** | `AutoTokenizer` | `ml/training/train_bert_experiment_v2.py` | **MODEL-BUILT-IN TOKENIZATION:** Subword token representation over 30,522 vocabulary items. |
| **NER Token Representation** | **USED** | `return_offsets_mapping` | `ml/ner/prepare_training_data.py` | **MODEL-BUILT-IN TOKEN REPRESENTATION:** Aligns subwords to 13-tag BIO schema. |

---

# 8. EXPLICIT DETERMINATION OF SUPERVISED FEATURE SELECTION

### Question:
> **Does the ABHERA project implement a separate supervised feature-selection algorithm?**

### Official Audit Answer:
> **NO.** The current implementation of the ABHERA project does **NOT** use a separate supervised feature-selection algorithm (such as `SelectKBest`, Chi-Square $\chi^2$, Mutual Information, ANOVA F-test, Recursive Feature Elimination, or L1-regularized selection).

### Technical Justification:
1. **Classical Baselines (TF-IDF + Logistic Regression / Linear SVM):** The parameter `max_features=5000` in `TfidfVectorizer` limits the vocabulary dimension to 5,000 terms based on overall corpus term frequency ($tf$), independent of target label correlations $\mathbf{y}$. Similarly, `min_df=2` removes terms occurring in fewer than 2 documents. These parameters represent **unsupervised feature-space bounding / vocabulary limitation**, not supervised feature selection.
2. **Deep Learning Models (BERT & NER):** Transformers utilize learned 768-dimensional contextual embeddings generated via multi-head self-attention mechanisms over the complete WordPiece token sequence. No manual feature subset selection or dimensionality reduction is applied.

---

# 9. PAPER-READY "FEATURE SELECTION" SECTION

*The text below is formatted in formal academic prose and can be included directly in the research paper:*

```markdown
### 3.3 Feature Representation and Feature Space Bounding

Rather than employing a separate supervised feature-selection algorithm (such as Chi-Square filtering, Mutual Information ranking, or Recursive Feature Elimination), the ABHERA framework relies on structured feature extraction and bounded feature-space representation tailored to each model architecture.

For traditional linear baselines (Logistic Regression and Linear SVM), text narratives are transformed into numerical vector spaces using a sublinear TF-IDF representation over unigrams and bigrams (`ngram_range=(1, 2)`). To prevent sparse matrix inflation while capturing salient domain phraseology, the vectorizer enforces an unsupervised feature-space bound (`max_features=5000`) and a minimum document frequency threshold (`min_df=2`). This configuration truncates the term vocabulary to the 5,000 most frequent n-grams across the training partition while eliminating single-occurrence typographical noise. All 5,000 extracted n-gram features are retained and passed directly to the classification heads without target-conditioned feature pruning.

For deep learning architectures (fine-tuned BERT classification and token-level NER), explicit feature selection is omitted in favor of end-to-end contextual representation learning. The `bert-base-uncased` encoder maps subword sequences into 768-dimensional dense contextual embeddings via multi-head self-attention. The model dynamically learns class-discriminative feature representations during backpropagation, rendering manual feature selection unnecessary.
```

---

# 10. VIVA-READY EXAMINATION QUESTIONS & ANSWERS

### Q1: Did you use feature selection in the project?
**Answer:** **No, explicit supervised feature selection was not used.** We did not apply algorithms like `SelectKBest`, Chi-Square ($\chi^2$), Mutual Information, or Recursive Feature Elimination. Instead, classical baselines used an unsupervised bounded TF-IDF feature space (`max_features=5000`), while BERT relied on end-to-end learned contextual embeddings.

### Q2: What is the difference between TF-IDF feature extraction and feature selection?
**Answer:** **Feature extraction** (e.g., TF-IDF) transforms unstructured text strings into a numerical vector matrix. **Feature selection** evaluates those extracted numerical features against target labels $\mathbf{y}$ (using statistical tests like $\chi^2$ or ANOVA) to select a discriminative subset. In ABHERA, TF-IDF performs feature extraction, but no subsequent target-conditioned feature selection is applied.

### Q3: Did you use SelectKBest?
**Answer:** **No, SelectKBest was not used.** All 5,000 extracted TF-IDF features were supplied directly to the baseline classifiers.

### Q4: Did you use Chi-square feature selection?
**Answer:** **No, Chi-square ($\chi^2$) feature selection was not used** in any script within the codebase.

### Q5: Did you use PCA or SVD for dimensionality reduction?
**Answer:** **No, dense dimensionality reduction algorithms such as PCA or TruncatedSVD were not used.** Classical models operated directly on sparse TF-IDF matrices, and BERT operated on native 768-dimensional contextual hidden states.

### Q6: What does `max_features=5000` mean in your TF-IDF implementation?
**Answer:** `max_features=5000` in `TfidfVectorizer` limits the size of the vocabulary matrix to the top 5,000 terms ordered by overall term frequency ($tf$) across the training corpus.

### Q7: Why is `max_features` not the same as supervised feature selection?
**Answer:** `max_features=5000` selects terms based purely on unsupervised term frequency ($tf$) across text narratives. It does **not** evaluate how well a term correlates with or predicts specific incident target labels $\mathbf{y}$. Supervised feature selection explicitly ranks features based on target label association.

### Q8: Does BERT use manually selected features?
**Answer:** **No.** BERT processes the complete tokenized sequence and automatically learns 768-dimensional contextual feature representations through its multi-head self-attention layers during fine-tuning.

### Q9: Does NER use explicit feature selection?
**Answer:** **No.** NER relies on token-level contextual representations generated by `AutoModelForTokenClassification` mapped directly to BIO entity labels via offset mappings without manual feature selection.

### Q10: Why did you use the implemented feature representation instead of manual feature selection?
**Answer:** For classical models, a 5,000-dimensional TF-IDF n-gram space cleanly captures domain keyphrases (e.g., `"husband beat"`, `"demanded dowry"`) without the risk of discarding weak but informative terms via aggressive feature selection. For BERT, transformer self-attention natively learns optimal feature representations directly from raw text during end-to-end training.

---

# 11. SOURCE TRACEABILITY MATRIX

| Conclusion / Finding | Source File Path | Line Range | Exact Code snippet / Evidence |
| :--- | :--- | :--- | :--- |
| **TF-IDF Configuration** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L165-L170) | L165–170 | `vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=5000)` |
| **Train-Only Fitting** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L172-L174) | L172–174 | `X_train_vec = vectorizer.fit_transform(X_train_raw)`<br>`X_val_vec = vectorizer.transform(X_val_raw)`<br>`X_test_vec = vectorizer.transform(X_test_raw)` |
| **LogReg Classifier** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L180-L183) | L180–183 | `OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, random_state=42))` |
| **Linear SVM Classifier** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L200-L203) | L200–203 | `OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000, random_state=42))` |
| **BERT Tokenizer Prep** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L46-L52) | L46–52 | `self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")` |
| **NER Token Offset Alignment** | [`ml/ner/prepare_training_data.py`](file:///d:/Abhera(Mini)/ml/ner/prepare_training_data.py#L57-L66) | L57–66 | `tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256)` |
| **Absence of Feature Selection** | Entire Repository (`ml/`, `backend/`) | Whole Codebase | 0 occurrences of `SelectKBest`, `chi2`, `mutual_info`, `RFE`, `PCA`, `TruncatedSVD`. |

---

ACTUAL FEATURE SELECTION STATUS: VERIFIED
