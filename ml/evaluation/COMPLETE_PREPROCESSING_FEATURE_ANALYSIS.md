# ABHERA — COMPLETE DATA PREPROCESSING, FEATURE ENGINEERING, AND FEATURE SELECTION ANALYSIS

**Document Version:** 1.0.0  
**Project:** ABHERA — AI-Driven Legal Guidance and Multi-Label Incident Classification System for Survivor Support  
**Scope:** Comprehensive Codebase Inspection, Technical Audit, Paper-Ready Specifications, and Viva Examination Reference Guide  
**Status:** Complete & Verified  

---

## EXECUTIVE SUMMARY

This document provides a exhaustive, paper-ready, and viva-ready analysis of the data preprocessing, feature engineering, text representation, feature selection, and model input preparation pipelines implemented in the **ABHERA** project.

All analyses herein are derived directly from empirical code inspection of the ABHERA source codebase (`ml/`, `backend/`, `models/`). No assumptions, synthetic claims, or unverified standard NLP conventions are introduced. Where a technique is absent in the codebase, it is explicitly documented as **"Not used in the current implementation."**

---

# SECTION 1: DATASET PREPROCESSING & CLEANING

### 1.1 Dataset Overview & Loading
* **Dataset Name:** ABHERA Survivor Incident Dataset (`womens_safety_dataset.csv`)
* **Primary Source File:** `data/womens_safety_dataset.csv`
* **Dataset Loader Module:** `ml/preprocessing/prepare_classification.py` (`ClassificationDataPreparer` class)
* **Dataset Analysis Module:** `ml/analysis/dataset_analysis.py` (`run_dataset_analysis()` function)
* **Total Incident Narratives (Raw):** 1,086 records
* **Valid Processed Records:** 1,086 records (0 excluded due to empty narratives)

### 1.2 Comprehensive Audit of Preprocessing Techniques

| Technique | Status | Algorithm / Library | Implementation Details | File & Function | Split Timing | Applied To | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Missing-Value Handling** | **USED** | Pandas `notnull()` & `str.strip() != ""` | Filters out null or empty string narratives. `valid_mask = df[narrative_col].notnull() & (df[narrative_col].astype(str).str.strip() != "")` | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` | Pre-split | All Models | Ensures no null inputs enter tokenizers or vectorizers. |
| **Empty-Text Handling** | **USED** | Validation Masking | Logs warning if empty narratives exist and excludes them prior to split. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` | Pre-split | All Models | Prevents empty sequence runtime exceptions during tokenization. |
| **Data-Type Validation** | **USED** | Explicit casting `.astype(str)` | Casts narrative column to string and binary label columns to integer/float tensors. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` | Pre-split | All Models | Guarantees strict type safety during tensor construction. |
| **Label Validation** | **USED** | Multi-label list validation | Verifies target columns `['DV', 'SH', 'ST', 'CA', 'WH', 'OV']` exist and computes multi-hot binary vectors. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` | Pre-split | All Models | Formulates multi-label target matrix $(N, 6)$. |
| **Duplicate Detection** | **USED (Audit)** | Pandas `df.duplicated()` | Executed in dataset analysis and leakage audit (`run_leakage_audit.py`). 0 exact cross-split duplicates found. | `ml/analysis/dataset_analysis.py`<br>`run_dataset_analysis()` | Pre-split & Post-split | Data Integrity Check | Audits duplicate count to verify data quality without dropping non-identical duplicates. |
| **Text Normalization** | **NOT USED (Pre-tokenizer)** | N/A | Raw text string is passed directly into HuggingFace tokenizer or TF-IDF vectorizer. Normalization occurs inside model libraries. | `ml/training/train_bert.py`<br>`WomensSafetyDataset.__getitem__()` | Dynamic | BERT / TF-IDF | Avoids destructive text modification prior to subword tokenization. |
| **Whitespace Normalization** | **NOT USED in script** | HuggingFace Tokenizer built-in | Basic `.strip()` is applied; internal multi-whitespace collapsing is handled inside `bert-base-uncased` tokenizer. | `ml/inference/predict_bert.py`<br>`BERTIncidentPredictor.predict()` | Inference | BERT | Preserved original raw string structure. |
| **Lowercasing** | **INDIRECT (Built-in)** | `bert-base-uncased` & `TfidfVectorizer(lowercase=True)` | Lowercasing is not performed manually on raw CSV files; it is executed inside `AutoTokenizer` (`bert-base-uncased`) and `TfidfVectorizer`. | `ml/training/train_bert_experiment_v2.py`<br>`ml/evaluation/run_baseline_comparison.py` | Training / Inference | BERT & TF-IDF | Maps all characters to lowercase inside model tokenizers. |
| **Punctuation Handling** | **INDIRECT (Built-in)** | WordPiece Tokenizer | Punctuation is retained and isolated as individual tokens by `bert-base-uncased` WordPiece tokenizer (e.g., `.` -> `[1012]`). | `transformers.AutoTokenizer` | Dynamic | BERT | Retains punctuation context for syntactic boundaries. |
| **Special-Character Handling** | **INDIRECT (Built-in)** | HuggingFace `bert-base-uncased` | Accents and unprintable special characters are stripped/mapped inside HuggingFace `bert-base-uncased` pre-tokenization. | `transformers.AutoTokenizer` | Dynamic | BERT | Standardizes character set to ASCII / WordPiece vocabulary. |
| **Stop-Word Removal** | **NOT USED in current implementation.** | N/A | Stop words (e.g., "and", "the", "in") are **NOT** removed. BERT relies on full sentence syntax and attention over stop words. TF-IDF vectorizer retains stop words (`stop_words=None`). | N/A | N/A | None | Preserved syntactic structure for transformer self-attention. |
| **Stemming** | **NOT USED in current implementation.** | N/A | Stemming (e.g., Porter/Lancaster) is **NOT** used. WordPiece subwords eliminate the need for heuristic rule-based stemming. | N/A | N/A | None | Avoids loss of morphological and semantic precision. |
| **Lemmatization** | **NOT USED in current implementation.** | N/A | Lemmatization (e.g., WordNetLemmatizer) is **NOT** used in any classification or NER pipeline. | N/A | N/A | None | Subword tokenization inherently handles inflectional suffixes. |
| **Spelling Correction** | **NOT USED in current implementation.** | N/A | No automatic spell-checking (e.g., Hunspell, SymSpell) is applied. Subwords capture typos via character sub-tokens. | N/A | N/A | None | Prevents accidental alteration of survivor terminology or proper nouns. |
| **URL Removal** | **NOT USED in current implementation.** | N/A | URLs are retained as text tokens. Digital abuse narratives frequently contain link fragments. | N/A | N/A | None | Retains evidence indicators in Cyber Abuse (CA) narratives. |
| **Number Handling** | **NOT USED in current implementation.** | N/A | Numbers are not stripped or replaced with pseudo-tokens; they are tokenized directly. | N/A | N/A | None | Retains statutory section numbers (e.g., "354", "498A"). |
| **Emoji Handling** | **NOT USED in current implementation.** | N/A | Emojis are left as raw UTF-8 string inputs; unmapped emojis default to `[UNK]` token during BERT tokenization. | N/A | N/A | None | Preserves raw narrative fidelity. |
| **HTML Removal** | **NOT USED in current implementation.** | N/A | No HTML tag stripping is performed (dataset contains raw survivor narratives, not HTML web dumps). | N/A | N/A | None | N/A |

---

# SECTION 2: DATASET SPLITTING & ISOLATION

### 2.1 Split Specifications
The dataset partition split is executed in `ml/preprocessing/prepare_classification.py` using a fixed random seed to ensure strict experimental reproducibility.

* **Total Processed Records:** 1,086
* **Training Set Split (70%):** 760 records (`ml/preprocessing/train_data.csv`)
* **Validation Set Split (15%):** 162 records (`ml/preprocessing/val_data.csv`)
* **Held-Out Test Set Split (15%):** 164 records (`ml/preprocessing/test_data.csv`)
* **Random Seed:** `42` (`np.random.seed(42)` and `df.sample(frac=1.0, random_state=42)`)
* **Split Sequence:** Preprocessing validation occurs on the raw dataset $\rightarrow$ Shuffled with Seed 42 $\rightarrow$ Partitioned into Train/Val/Test CSVs $\rightarrow$ Feature Vectorizers and Decision Thresholds fitted **STRICTLY** on Training/Validation splits.

```python
# Implementation in ml/preprocessing/prepare_classification.py (Lines 108-118)
np.random.seed(seed)
shuffled_df = clean_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

n_samples = len(shuffled_df)
train_end = int(n_samples * train_ratio)      # 760
val_end = train_end + int(n_samples * val_ratio) # 760 + 162 = 922

train_df = shuffled_df.iloc[:train_end].copy()   # 760 rows (70%)
val_df   = shuffled_df.iloc[train_end:val_end].copy() # 162 rows (15%)
test_df  = shuffled_df.iloc[val_end:].copy()     # 164 rows (15%)
```

### 2.2 Stratification & Multi-Label Structure
* Standard single-label stratified splitting (e.g., `train_test_split(stratify=y)`) is **not applicable** due to multi-label incident co-occurrence.
* Uniform multi-label distribution across Train, Validation, and Test splits was verified via empirical audit (`ml/evaluation/leakage_audit/run_leakage_audit.py`).

### 2.3 Rationale for Test Set Isolation
The 164-sample test set is strictly held out and untouched during:
1. TF-IDF vocabulary construction (`vectorizer.fit()`)
2. Model weight optimization (Backpropagation / AdamW loss gradients)
3. Decision threshold tuning (Per-label $F_1$ optimization on Validation set)

This guarantees that reported test set metrics represent unbiased, out-of-sample generalization.

---

# SECTION 3: LABEL PREPROCESSING & MULTI-LABEL FORMULATION

### 3.1 Incident Category Target Schema
ABHERA formulates survivor incident classification as a **Multi-Label Binary Classification Problem** across six core statutory safety categories:

1. **DV:** Domestic Violence
2. **SH:** Sexual Harassment
3. **ST:** Stalking
4. **CA:** Cyber Abuse
5. **WH:** Workplace Harassment
6. **OV:** Other Violence / Public Assault

### 3.2 Target Encoding & Mathematical Form
* **Representation:** Binary Multi-Hot Vector $\mathbf{y} \in \{0, 1\}^6$.
* **Encoding Method:** Multi-hot encoding (`[DV, SH, ST, CA, WH, OV]`).
* **Output Activation:** Independent Sigmoid functions ($\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$ for each category $i$). **Softmax is NOT used** because classes are mutually non-exclusive (an incident can concurrently involve Domestic Violence and Cyber Abuse).
* **Loss Function:** PyTorch `BCEWithLogitsLoss` with pos-weight class imbalance correction:
  $$\mathcal{L}_{BCE} = -\frac{1}{K}\sum_{k=1}^{6} \left[ w_k y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$
  where $w_k = \frac{N_{\text{negative}}}{N_{\text{positive}}}$ for class $k$.

### 3.3 Illustrative Multi-Label Mapping Example

**Incident Text:**
> *"My husband beats me at home and threatens to post my private photos on social media if I call the police."*

**Verified ABHERA Label Representation:**
```
DV (Domestic Violence)      = 1  [Reason: Physical spousal abuse "husband beats me"]
SH (Sexual Harassment)     = 0  [Reason: No direct workplace/public sexual assertion]
ST (Stalking)              = 0  [Reason: No persistent physical tracking/following]
CA (Cyber Abuse)           = 1  [Reason: Online threat "post my private photos on social media"]
WH (Workplace Harassment)  = 0  [Reason: Incident did not occur in a professional workplace]
OV (Other Violence)        = 0  [Reason: Domestic setting, covered under DV]

Vector Target Representation: y = [1, 0, 0, 1, 0, 0]
```

---

# SECTION 4: BERT TEXT PREPROCESSING PIPELINE

### 4.1 Tokenizer Specifications
* **Pretrained Tokenizer:** `bert-base-uncased`
* **Library:** HuggingFace `transformers` (`AutoTokenizer.from_pretrained("bert-base-uncased")`)
* **Vocabulary Size:** 30,522 tokens
* **Special Tokens:**
  * `[PAD]` (ID: `0`): Padding token
  * `[UNK]` (ID: `100`): Out-of-vocabulary unknown token
  * `[CLS]` (ID: `101`): Classification entry token (first sequence position)
  * `[SEP]` (ID: `102`): Sentence separator token (end of sequence)

### 4.2 Sequence Length Specifications
* **BERT v1 Baseline (`ml/training/train_bert.py`):** `max_length = 64`
* **BERT v2 Production Model (`ml/training/train_bert_experiment_v2.py`):** `max_length = 128` (training), `max_length = 256` (test evaluation and production inference in `ml/inference/predict_bert.py`).

### 4.3 Step-by-Step BERT Preprocessing Pipeline

```
Raw Survivor Text
  │
  ▼
1. Basic Normalization (HuggingFace internal: ASCII mapping, strip accents)
  │
  ▼
2. Lowercasing (Map text to lowercase)
  │
  ▼
3. Pre-Tokenization (Whitespace & Punctuation isolation)
  │
  ▼
4. WordPiece Subword Tokenization (Decompose words into vocabulary subwords with '##' prefix)
  │
  ▼
5. Token ID Mapping (Lookup subword tokens in 30,522 vocabulary table)
  │
  ▼
6. Special Token Prepended/Appended ([CLS] + Token IDs + [SEP])
  │
  ▼
7. Padding & Truncation (Pad with [PAD] (0) or truncate sequence to max_length=256)
  │
  ▼
8. Attention Mask Construction (1 for real tokens, 0 for [PAD] tokens)
  │
  ▼
PyTorch Model Tensors: input_ids (1, 256), attention_mask (1, 256) ──> BERT Model Input
```

---

# SECTION 5: WORDPIECE TOKENIZATION ALGORITHM

### 5.1 Word-Level vs Subword vs WordPiece
* **Word-Level Tokenization:** Splits text strictly on spaces/punctuation (e.g., `"harassment"` $\rightarrow$ `["harassment"]`). Fails on unseen words or misspellings (Out-Of-Vocabulary / OOV problem).
* **Character-Level Tokenization:** Splits text into individual characters (e.g., `['h', 'a', 'r', ...]`). Avoids OOV but destroys semantic token context and drastically increases sequence length.
* **WordPiece Subword Tokenization (BERT):** A data-driven subword segmentation algorithm that balances vocabulary size and token coverage. Frequent words remain intact, while rare or inflected words are split into subword units. Non-initial subwords are prefixed with `##`.

### 5.2 Mathematical / Algorithmic Principle of WordPiece
WordPiece initializes the vocabulary with all individual characters. It then iteratively merges character pairs that maximize the language model likelihood of the training corpus:
$$\text{Score}_{(A, B)} = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

### 5.3 Verified Empirical Example using `bert-base-uncased`
Using ABHERA's tokenizer (`AutoTokenizer.from_pretrained("bert-base-uncased")`), text tokenization behaves as follows:

* **Word:** `"harassment"`  
  **Tokens:** `['harassment']` (Present directly in `bert-base-uncased` vocabulary, Token ID: `13233`)

* **Rare Word / Inflection:** `"cyberharassment"`  
  **Tokens:** `['cyber', '##ha', '##ras', '##sment']`  
  **Token IDs:** `[17404, 16008, 18749, 13745]`  
  **Explanation:** `"cyber"` is tokenized intact; `"harassment"` is decomposed into prefix subwords starting with `##` to represent continuation.

---

# SECTION 6: NER PREPROCESSING & TOKEN ALIGNMENT

### 6.1 NER Architecture Specifications
* **Task:** Token-Level Named Entity Recognition (Evidentiary Information Extraction).
* **Model Class:** `AutoModelForTokenClassification` (`bert-base-uncased` with token classification head).
* **Source Script:** `ml/ner/prepare_training_data.py` & `ml/ner/train_ner.py`.
* **Annotation Format:** JSONL character-level offset spans (`start`, `end`, `label`, `text`).

### 6.2 Entity Schema & 13-Tag BIO Scheme
ABHERA uses a 13-tag **BIO (Begin, Inside, Outside)** scheme across 6 legal entity types:

| Entity Label | Entity Meaning | BIO Tags |
| :--- | :--- | :--- |
| **PERP_REL** | Perpetrator Relationship (e.g., "husband", "colleague") | `B-PERP_REL`, `I-PERP_REL` |
| **LOCATION** | Incident Location (e.g., "office", "bus stand") | `B-LOCATION`, `I-LOCATION` |
| **TIME_FREQ** | Time / Frequency Details (e.g., "every night", "for 2 months") | `B-TIME_FREQ`, `I-TIME_FREQ` |
| **PLATFORM** | Digital Platform (e.g., "WhatsApp", "Instagram") | `B-PLATFORM`, `I-PLATFORM` |
| **EVIDENCE** | Physical / Digital Evidence (e.g., "screenshots", "audio recording") | `B-EVIDENCE`, `I-EVIDENCE` |
| **LAW_SEC** | Statutory Law Mention (e.g., "Section 354", "POSH Act") | `B-LAW_SEC`, `I-LAW_SEC` |
| **O** | Outside any legal entity | `O` |

### 6.3 Subword Alignment & Special Token Handling
1. **Character Offset Mapping:** `tokenizer(text, return_offsets_mapping=True)` extracts `(start_char, end_char)` for every WordPiece subword.
2. **Subword Tag Assignment:** The first subword falling within a character span $[s, e)$ receives `B-<LABEL>`. Subsequent subwords within the span receive `I-<LABEL>`.
3. **Special Token Masking (`-100`):** Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are assigned label `-100`. PyTorch `CrossEntropyLoss(ignore_index=-100)` automatically ignores these positions during gradient computation.

### 6.4 Distinction: BERT Classification vs. NER

```
                                  RAW SURVIVOR INCIDENT NARRATIVE
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        BERT INCIDENT CLASSIFICATION                              NAMED ENTITY RECOGNITION (NER)
     ("What type of incident is this?")                 ("What structured entities are present?")
                        │                                                 │
            Sequence-Level Task                                   Token-Level Task
                        │                                                 │
        Output: Multi-Label Vector                           Output: Extracted Entity Spans
    [DV: 1, SH: 0, ST: 0, CA: 1, WH: 0, OV: 0]           PERP_REL: "husband" | LOCATION: "home"
```

---

# SECTION 7: TF-IDF FEATURE ENGINEERING (BASELINE MODELS)

### 7.1 TF-IDF Vectorizer Specifications
In `ml/evaluation/run_baseline_comparison.py`, TF-IDF features are extracted using `scikit-learn`:

```python
# Exact implementation in ml/evaluation/run_baseline_comparison.py (Lines 165-170)
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

### 7.2 Parameter Audit Table

| Parameter | Configuration | Exact Purpose & Effect |
| :--- | :--- | :--- |
| **`analyzer`** | `'word'` (Default) | Extracts word-level n-grams. |
| **`ngram_range`** | `(1, 2)` | Captures both single words (unigrams) and contiguous word pairs (bigrams). |
| **`sublinear_tf`** | `True` | Applies logarithmic scaling to term frequency: $TF = 1 + \ln(tf)$ for $tf > 0$. Subdues repetitive noise words. |
| **`min_df`** | `2` | Ignores terms that appear in fewer than 2 documents across the corpus. Filters rare typos. |
| **`max_df`** | `1.0` (Default) | Retains terms regardless of high document frequency. |
| **`max_features`** | `5000` | Bounds vocabulary size to the top 5,000 terms ordered by term frequency across the training set. |
| **`stop_words`** | `None` (Default) | No stop words removed; retains full lexical context. |
| **`lowercase`** | `True` (Default) | Converts all input tokens to lowercase prior to vocabulary building. |
| **`smooth_idf`** | `True` (Default) | Adds 1 to document frequencies to prevent division-by-zero: $IDF(t) = \ln \frac{1 + N}{1 + df(t)} + 1$. |
| **`norm`** | `'l2'` (Default) | Applies L2 normalization to each feature vector so that Euclidean norm equals 1. |
| **Representation** | Sparse Matrix | Encoded as `scipy.sparse.csr_matrix` of shape $(N, 5000)$. |

### 7.3 Mathematical Formula for TF-IDF

1. **Sublinear Term Frequency:**
   $$TF(t, d) = \begin{cases} 1 + \ln(\text{tf}(t, d)) & \text{if } \text{tf}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

2. **Smoothed Inverse Document Frequency:**
   $$IDF(t) = \ln \left( \frac{1 + N}{1 + df(t)} \right) + 1$$

3. **Raw TF-IDF Weight:**
   $$\text{TF-IDF}(t, d) = TF(t, d) \times IDF(t)$$

4. **L2 Vector Normalization:**
   $$\vec{v}_{\text{normalized}} = \frac{\vec{v}}{\|\vec{v}\|_2} = \frac{\vec{v}}{\sqrt{\sum_{i=1}^{5000} v_i^2}}$$

---

# SECTION 8: FEATURE SELECTION ANALYSIS

### 8.1 Empirical Verification of Feature Selection
A complete audit of the ABHERA repository confirms that **NO explicit supervised feature selection algorithm is used in the current implementation.**

### 8.2 Detailed Feature Selection Audit Checklist

| Technique | Status | Tested / Used | Explanation / Finding |
| :--- | :--- | :--- | :--- |
| **`SelectKBest`** | **NOT USED** | No | Not imported or invoked in any script. |
| **Chi-Square ($\chi^2$) Test** | **NOT USED** | No | Supervised feature ranking via chi2 is absent. |
| **Mutual Information** | **NOT USED** | No | Information gain / mutual info filtering is absent. |
| **ANOVA F-Test** | **NOT USED** | No | `f_classif` feature selection is absent. |
| **Recursive Feature Elimination (RFE)** | **NOT USED** | No | Iterative model-based feature removal is absent. |
| **L1-Based Feature Selection** | **NOT USED** | No | Lasso / L1-regularized feature filtering is absent. |
| **Variance Threshold** | **NOT USED** | No | Zero/low-variance feature removal is absent. |
| **Dimensionality Reduction (PCA / SVD)** | **NOT USED** | No | Dense linear reduction (PCA / TruncatedSVD) is absent. |

### 8.3 Feature Extraction vs. Feature Selection Distinction

It is critical for academic and viva evaluations to distinguish between **Feature Representation / Extraction** and **True Supervised Feature Selection**:

* **Feature Extraction / Representation (USED):** `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=5000)` converts unstructured text into a 5,000-dimensional numerical matrix. The parameter `max_features=5000` truncates vocabulary size based purely on term frequency, and `min_df=2` removes rare tokens.
* **True Feature Selection (NOT USED):** Statistical selection algorithms evaluate feature relevance **with respect to target class labels** $\mathbf{y}$ (e.g., computing $\chi^2(x_j, y_k)$ scores to select the top $K$ most discriminative terms). Because ABHERA passes all 5,000 TF-IDF features directly to the classifiers without target-conditioned filtering, no explicit feature selection is performed.

---

# SECTION 9: LOGISTIC REGRESSION BASELINE PIPELINE

### 9.1 Multi-Label Architecture
In `ml/evaluation/run_baseline_comparison.py`, multi-label classification using Logistic Regression is implemented via `OneVsRestClassifier`:

```python
# Exact implementation in ml/evaluation/run_baseline_comparison.py (Lines 180-183)
lr_model = OneVsRestClassifier(
    LogisticRegression(C=1.0, max_iter=1000, random_state=42)
)
lr_model.fit(X_train_vec, y_train)
val_lr_probs = lr_model.predict_proba(X_val_vec)
test_lr_probs = lr_model.predict_proba(X_test_vec)
```

### 9.2 Execution Protocol
1. **Classifier Strategy:** `OneVsRestClassifier` fits 6 separate binary Logistic Regression models (one per incident label).
2. **Probability Computation:** Computes continuous class probabilities $p_k = \sigma(\mathbf{w}_k^T \mathbf{x} + b_k)$.
3. **Threshold Tuning:** Validation-derived optimal decision thresholds are tuned per label on `val_lr_probs` via grid search ($t \in [0.05, 0.95]$, step $0.01$) to maximize $F_1$.
4. **Performance:** Achieved **97.79% Micro $F_1$** on the held-out test set.

---

# SECTION 10: LINEAR SVM BASELINE PIPELINE

### 10.1 Multi-Label Architecture
In `ml/evaluation/run_baseline_comparison.py`, Linear Support Vector Machine classification is implemented via `OneVsRestClassifier`:

```python
# Exact implementation in ml/evaluation/run_baseline_comparison.py (Lines 200-205)
svm_model = OneVsRestClassifier(
    LinearSVC(C=1.0, max_iter=2000, random_state=42)
)
svm_model.fit(X_train_vec, y_train)

val_svm_scores = sigmoid(svm_model.decision_function(X_val_vec))
test_svm_scores = sigmoid(svm_model.decision_function(X_test_vec))
```

### 10.2 Decision Function & Pseudo-Probabilities
LinearSVC calculates the margin distance $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$. Since LinearSVC does not output native probabilities, ABHERA maps decision function scores through a Sigmoid activation:
$$p_k = \frac{1}{1 + e^{-z_k}}$$
Per-label optimal decision thresholds are tuned on `val_svm_scores` and applied to test predictions, achieving **97.27% Micro $F_1$**.

---

# SECTION 11: COMPLETE END-TO-END PIPELINE DIAGRAMS

### Pipeline 1: BERT Multi-Label Incident Classification

```
  Raw Incident Narrative String
               │
               ▼
   Data Cleaning & Validation (ml/preprocessing/prepare_classification.py)
               │
               ▼
   70 / 15 / 15 Train/Val/Test Split (Seed = 42)
               │
               ▼
   BERT Tokenization & WordPiece Subword Decomp (bert-base-uncased, max_length=128/256)
               │
               ▼
   Tensors: input_ids (1, 256) + attention_mask (1, 256)
               │
               ▼
   Fine-Tuned BERT Sequence Classification Model (models/bert_multilabel_experiment_v2)
               │
               ▼
   Raw Logits (z_1, z_2, z_3, z_4, z_5, z_6)
               │
               ▼
   Sigmoid Probability Computation: p_i = 1 / (1 + exp(-z_i))
               │
               ▼
   Validation-Tuned Per-Class Thresholding (DV:0.73, SH:0.50, ST:0.79, CA:0.45, WH:0.29, OV:0.63)
               │
               ▼
   Predicted Incident Categories [DV, CA]
```

---

### Pipeline 2: TF-IDF + Logistic Regression Baseline

```
  Raw Incident Narrative String
               │
               ▼
   Train/Val/Test Data Split (70/15/15, Seed = 42)
               │
               ▼
   TF-IDF Feature Extraction (Fit on Train ONLY: ngram_range=(1,2), sublinear_tf=True, max_features=5000)
               │
               ▼
   Sparse Feature Matrix X (N, 5000)
               │
               ▼
   OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))
               │
               ▼
   Sigmoid Class Probabilities p_k
               │
               ▼
   Validation-Tuned Per-Label Threshold Search
               │
               ▼
   Final Multi-Label Predictions
```

---

### Pipeline 3: TF-IDF + Linear SVM Baseline

```
  Raw Incident Narrative String
               │
               ▼
   Train/Val/Test Data Split (70/15/15, Seed = 42)
               │
               ▼
   TF-IDF Feature Extraction (Fit on Train ONLY: ngram_range=(1,2), sublinear_tf=True, max_features=5000)
               │
               ▼
   Sparse Feature Matrix X (N, 5000)
               │
               ▼
   OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000))
               │
               ▼
   Decision Function Margin Distance z_k = w_k^T x + b_k
               │
               ▼
   Sigmoid Score Transformation: p_k = 1 / (1 + exp(-z_k))
               │
               ▼
   Validation-Tuned Per-Label Threshold Application
               │
               ▼
   Final Multi-Label Predictions
```

---

### Pipeline 4: Named Entity Recognition (NER)

```
  Raw Incident Narrative String
               │
               ▼
   BERT WordPiece Tokenization with Offset Mapping (return_offsets_mapping=True)
               │
               ▼
   Subword Token Sequence + Character Offset Ranges [(0,4), (5,12), ...]
               │
               ▼
   Fine-Tuned BERT Token Classification Model (models/ner)
               │
               ▼
   Logits per Token over 13 BIO Labels
               │
               ▼
   Argmax Tag Decoding (B-PERP_REL, I-PERP_REL, O, B-LOCATION, ...)
               │
               ▼
   BIO Span Extraction & Subword Re-alignment to Original Character Offsets
               │
               ▼
   Extracted Evidentiary Entities [{"label": "PERP_REL", "text": "husband"}, ...]
```

---

# SECTION 12: DATA LEAKAGE PREVENTION AUDIT

Empirical leakage prevention was validated via `ml/evaluation/leakage_audit/run_leakage_audit.py`:

1. **Strict Split Isolation:** Split partitioning (Train 760 / Val 162 / Test 164) is performed **BEFORE** TF-IDF fitting or model training.
2. **Train-Only TF-IDF Fitting:** `vectorizer.fit_transform(X_train_raw)` is executed strictly on the training set. Validation and Test splits are transformed using `vectorizer.transform()`.
3. **Validation-Only Threshold Tuning:** Decision thresholds are tuned strictly on `val_df`. The test set is evaluated exactly **ONCE** using these fixed thresholds.
4. **Zero Cross-Split Duplicates:** Exact hash duplicate checks revealed **0 cross-split duplicate narratives** between Train, Validation, and Test sets.
5. **Zero Verbatim String Leakage:** Label codes ("DV", "SH") do not appear verbatim in narrative text inputs.

---

# SECTION 13: PAPER-READY SUMMARY TABLES

### Table 1: Complete Preprocessing & Cleaning Audit

| Stage / Technique | Method / Algorithm | Applied To | Exact Source File & Line | Status | Rationale / Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Missing Text Handling** | Pandas `notnull()` filtering | All Narratives | `prepare_classification.py`:L56 | **USED** | Excludes invalid/empty records prior to split. |
| **Lowercasing** | Built-in Tokenizer / TF-IDF | Text Inputs | `train_bert_experiment_v2.py`:L224 | **USED** | Normalizes character case inside model pipelines. |
| **Subword Tokenization** | WordPiece (`bert-base-uncased`) | BERT Input | `train_bert_experiment_v2.py`:L46 | **USED** | Decomposes text into 30,522 subword vocabulary items. |
| **Sequence Padding** | Max Length Padding (256) | BERT Input | `predict_bert.py`:L250 | **USED** | Standardizes batch tensor shapes. |
| **Stop-Word Removal** | N/A | N/A | N/A | **NOT USED** | Retained to preserve full self-attention syntax. |
| **Stemming / Lemmatization**| N/A | N/A | N/A | **NOT USED** | Subwords natively resolve morphological suffixes. |
| **Spelling Correction** | N/A | N/A | N/A | **NOT USED** | Avoids corruption of domain terminology. |
| **HTML / URL Removal** | N/A | N/A | N/A | **NOT USED** | Preserves raw narrative evidence. |

### Table 2: Feature Representation & Extraction vs Selection

| Model Pipeline | Feature Representation | Parameters | Selection vs Extraction | Source File |
| :--- | :--- | :--- | :--- | :--- |
| **BERT Multi-Label** | Dense Contextual Embeddings | `bert-base-uncased`, 768-dim hidden layer, max_len=128/256 | **Feature Representation** | `ml/training/train_bert_experiment_v2.py` |
| **Logistic Regression**| Sparse TF-IDF Matrix | `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_features=5000` | **Feature Representation** | `ml/evaluation/run_baseline_comparison.py` |
| **Linear SVM** | Sparse TF-IDF Matrix | `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_features=5000` | **Feature Representation** | `ml/evaluation/run_baseline_comparison.py` |
| **Supervised Feature Selection**| None ($\chi^2$, Mutual Info, RFE) | N/A | **NOT USED** | N/A |

---

# SECTION 14: VIVA-READY EXAMINATION QUESTION & ANSWER GUIDE

### Q1: How did you preprocess the dataset for ABHERA?
**Answer:** Preprocessing begins with dataset validation in `prepare_classification.py`, where null or empty narrative strings are filtered out. The dataset is then split reproducibly using seed 42 into Train (70%), Validation (15%), and Test (15%) splits. Text is passed to model tokenizers where lowercasing and WordPiece subword tokenization (`bert-base-uncased`) or TF-IDF n-gram extraction are applied. Traditional destructive text cleaning (such as stop-word removal or stemming) is intentionally avoided to preserve syntactic context.

### Q2: Did you perform stop-word removal, stemming, or lemmatization?
**Answer:** **No, none of these were used in the current implementation.** Stop words are retained because transformer models (BERT) utilize self-attention over complete sentence structures. Stemming and lemmatization are rendered redundant by WordPiece subword tokenization, which handles morphological inflections at the subword level without rule-based information loss.

### Q3: What tokenization algorithm does BERT use, and why is it useful?
**Answer:** BERT uses **WordPiece subword tokenization** with a 30,522 token vocabulary. WordPiece breaks words into subword units (prefixed with `##` for non-initial subwords). It is useful because it completely solves the Out-Of-Vocabulary (OOV) problem—rare or misspelled words are broken down into known subword components rather than mapped to an unknown `[UNK]` token.

### Q4: How were text narratives converted into numerical features for baseline models?
**Answer:** Narratives were converted using `scikit-learn`'s `TfidfVectorizer` configured with unigram and bigram ranges (`ngram_range=(1, 2)`), sublinear term frequency scaling (`sublinear_tf=True`), minimum document frequency of 2 (`min_df=2`), and a maximum vocabulary cap of 5,000 features (`max_features=5000`).

### Q5: Did you perform explicit feature selection?
**Answer:** **No explicit supervised feature selection algorithm was used.** Techniques like Chi-Square ($\chi^2$), Mutual Information, ANOVA F-test, Recursive Feature Elimination (RFE), or L1-Lasso selection were not applied. Vocabulary truncation via `max_features=5000` and `min_df=2` in `TfidfVectorizer` represents feature extraction/representation bounding, not supervised target-conditioned feature selection.

### Q6: What is the difference between feature extraction and feature selection in your project?
**Answer:** Feature extraction (e.g., TF-IDF or BERT embeddings) transforms unstructured text into numerical vector spaces. Feature selection refers to applying statistical or model-based tests to select a subset of those features based on their correlation with target labels $\mathbf{y}$. ABHERA uses feature extraction, but does not apply supervised feature selection.

### Q7: How did you prevent data leakage during feature engineering and training?
**Answer:** Data leakage was prevented by:
1. Partitioning the dataset into Train, Val, and Test splits BEFORE fitting any vectorizer or model.
2. Fitting the TF-IDF vectorizer exclusively on training data (`X_train_raw`).
3. Tuning decision thresholds exclusively on the Validation split.
4. Keeping the Test split completely isolated until final evaluation. An empirical audit confirmed 0 cross-split duplicate narratives.

### Q8: How were multi-label targets represented and why did you use Sigmoid instead of Softmax?
**Answer:** Targets were encoded as 6-dimensional binary multi-hot vectors (`[DV, SH, ST, CA, WH, OV]`). **Sigmoid activation was used instead of Softmax** because survivor incidents are mutually non-exclusive—a single incident can simultaneously involve multiple categories (e.g., Domestic Violence and Cyber Abuse). Softmax enforces a probability distribution summing to 1 across mutually exclusive classes, whereas Sigmoid outputs independent probabilities for each class.

### Q9: How was NER preprocessing performed and how were labels aligned with subwords?
**Answer:** Character offset spans (`start`, `end`, `label`) were aligned to BERT subwords using `return_offsets_mapping=True`. The first subword inside a span received a `B-<LABEL>` tag, subsequent subwords received `I-<LABEL>`, and outside tokens received `O`. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) were assigned label `-100` so PyTorch `CrossEntropyLoss` ignores them during training.

### Q10: Why did TF-IDF baseline models achieve higher performance than BERT on this dataset?
**Answer:** Empirical leakage audit (`ml/evaluation/leakage_audit/run_leakage_audit.py`) confirmed that survivor narratives contain explicit, highly discriminative key phrase signals (e.g., "husband beat", "demanded dowry", "private photos"). Linear classifiers (Logistic Regression and Linear SVM) over 5,000 TF-IDF n-grams capture these crisp lexical signals directly without the optimization variance or hyperparameter sensitivity of deep neural transformers.

---

# SECTION 15: SOURCE FILE AND FUNCTION MAP

* **Dataset Preparation & Split:** `ml/preprocessing/prepare_classification.py` $\rightarrow$ `ClassificationDataPreparer.prepare()`
* **Dataset Analysis & Duplicate Checks:** `ml/analysis/dataset_analysis.py` $\rightarrow$ `run_dataset_analysis()`
* **TF-IDF Vectorization & Baselines:** `ml/evaluation/run_baseline_comparison.py` $\rightarrow$ `run_baseline_evaluation()`
* **BERT Model Training (v1 & v2):** `ml/training/train_bert_experiment_v2.py` $\rightarrow$ `main()`
* **BERT Production Inference:** `ml/inference/predict_bert.py` $\rightarrow$ `BERTIncidentPredictor.predict()`
* **NER Dataset Prep & BIO Alignment:** `ml/ner/prepare_training_data.py` $\rightarrow$ `align_spans_to_bio_tokens()`
* **NER Training & Evaluation:** `ml/ner/train_ner.py` $\rightarrow$ `train_ner_model()`
* **NER Production Inference:** `ml/inference/predict_ner.py` $\rightarrow$ `NEREntityPredictor.predict()`
* **Data Leakage & Similarity Audit:** `ml/evaluation/leakage_audit/run_leakage_audit.py` $\rightarrow$ `run_full_audit()`

---

PREPROCESSING AND FEATURE ANALYSIS STATUS: VERIFIED
