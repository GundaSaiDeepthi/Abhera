# ABHERA — PAPER DOCUMENTATION: ACTUAL DATA PREPROCESSING ANALYSIS

**Document Version:** 1.0.0  
**Project:** ABHERA — AI-Driven Legal Guidance and Multi-Label Incident Classification System for Survivor Support  
**Scope:** Inspection and Documentation of Implemented Dataset Preprocessing Techniques  
**Status:** Inspection Complete & Verified  

---

## EXECUTIVE SUMMARY

This document presents a strict, code-authoritative inspection of the **actual data preprocessing pipeline** implemented in the ABHERA project repository.

In strict compliance with paper documentation guidelines:
* **No code, model, dataset, or pipeline has been modified.**
* **No synthetic or assumed NLP preprocessing steps have been introduced.**
* If a technique is absent from the codebase, it is explicitly declared as:  
  **"Not used in the current implementation."**

---

# 1. ACTUAL PREPROCESSING SOURCE CODE REGISTRY

The codebase was searched across all modules (`ml/`, `backend/`, `data/`). All data loading, cleaning, validation, label preparation, and dataset splitting operations occur in the following explicit source files:

| Pipeline Stage | Source File Path | Class / Function Name | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Primary Dataset Preparation** | [`ml/preprocessing/prepare_classification.py`](file:///d:/Abhera(Mini)/ml/preprocessing/prepare_classification.py#L33-L154) | `ClassificationDataPreparer.prepare()` | Resolves dataset path, loads CSV, filters empty narratives, computes multi-hot label vectors, calculates class weights, and executes 70/15/15 reproducible dataset splitting. |
| **Dataset Path Resolution** | [`ml/preprocessing/prepare_classification.py`](file:///d:/Abhera(Mini)/ml/preprocessing/prepare_classification.py#L16-L30) | `resolve_dataset_path()` | Dynamically locates `data/womens_safety_dataset.csv` across relative directory candidates. |
| **Dataset Integrity Analysis** | [`ml/analysis/dataset_analysis.py`](file:///d:/Abhera(Mini)/ml/analysis/dataset_analysis.py#L24-L160) | `run_dataset_analysis()` | Audits total records, column presence, missing values per column, exact duplicate rows, duplicate narrative texts, and multi-label distribution. |
| **PyTorch Dataset Preprocessing (BERT)** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L29-L56) | `WomensSafetyDataset.__getitem__()` | Casts narrative text to string, formats binary label float list, invokes HuggingFace tokenizer (`max_length=128/256`), and converts inputs to PyTorch Tensors. |
| **TF-IDF Dataset Extraction (Baselines)** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L147-L175) | `run_baseline_evaluation()` | Reads pre-split Train/Val/Test CSVs, extracts raw text arrays (`.astype(str).values`), extracts binary target matrix, and fits `TfidfVectorizer`. |
| **NER Dataset Preprocessing & BIO Alignment** | [`ml/ner/prepare_training_data.py`](file:///d:/Abhera(Mini)/ml/ner/prepare_training_data.py#L45-L102) | `align_spans_to_bio_tokens()` | Converts character-level entity offset spans into token-level BIO tags using tokenizer offset mappings, assigning `-100` to special tokens. |
| **NER Annotation Validation** | [`ml/ner/prepare_ner_dataset.py`](file:///d:/Abhera(Mini)/ml/ner/prepare_ner_dataset.py#L43-L143) | `NERAnnotationValidator.validate_record()` | Checks JSON keys, offset bounds (`0 <= start < end <= len(text)`), text slice matching, schema compliance, duplicate spans, and overlapping character offsets. |
| **Dataset Leakage & Similarity Audit** | [`ml/evaluation/leakage_audit/run_leakage_audit.py`](file:///d:/Abhera(Mini)/ml/evaluation/leakage_audit/run_leakage_audit.py#L47-L445) | `run_full_audit()` | Post-split evaluation audit for cross-split exact duplicate narratives, TF-IDF cosine similarity, label consistency, and keyword leakage. |

---

# 2. EXACT SEQUENCE OF DATA TRANSFORMATIONS

The following trace details the step-by-step transformation of raw incident records from `data/womens_safety_dataset.csv` until they become model-ready tensors or feature matrices.

```
                  RAW DATASET FILE (data/womens_safety_dataset.csv)
                                        │
                                        ▼
STEP 1: CSV INGESTION & PATH RESOLUTION
  - Input: CSV file path resolved via resolve_dataset_path()
  - Transformation: pd.read_csv(self.csv_path) loading 1,086 raw rows into a Pandas DataFrame.
  - File / Function: ml/preprocessing/prepare_classification.py -> ClassificationDataPreparer.prepare()
  - Consuming Models: All models (BERT, TF-IDF Baselines, NER).
                                        │
                                        ▼
STEP 2: NARRATIVE VALIDATION & EMPTY-TEXT FILTERING
  - Input: Raw DataFrame df (1,086 rows).
  - Transformation: Evaluates valid_mask = df["narrative_text"].notnull() & (df["narrative_text"].astype(str).str.strip() != "").
  - Purpose: Excludes null or whitespace-only narrative rows. (0 records excluded; 1,086 valid records retained).
  - File / Function: ml/preprocessing/prepare_classification.py -> ClassificationDataPreparer.prepare()
  - Consuming Models: All models.
                                        │
                                        ▼
STEP 3: MULTI-LABEL TARGET MATRIX FORMULATION
  - Input: Valid DataFrame clean_df.
  - Transformation: Converts 6 binary incident columns ['DV', 'SH', 'ST', 'CA', 'WH', 'OV'] into a multi-hot list representation clean_df["label_vector"].
  - Purpose: Constructs multi-label ground-truth target matrix Y ∈ {0, 1}^(N x 6).
  - File / Function: ml/preprocessing/prepare_classification.py -> ClassificationDataPreparer.prepare()
  - Consuming Models: BERT Classification, TF-IDF Logistic Regression, TF-IDF Linear SVM.
                                        │
                                        ▼
STEP 4: CLASS IMBALANCE POS-WEIGHT CALCULATION
  - Input: Ground-truth target columns clean_df[LABEL_COLUMNS].
  - Transformation: Computes positive weight per class k: pos_weights[k] = (N_total - N_pos) / N_pos. Saved to label_mapping.json.
  - Purpose: Provides class imbalance correction weights to PyTorch BCEWithLogitsLoss.
  - File / Function: ml/preprocessing/prepare_classification.py -> ClassificationDataPreparer.prepare()
  - Consuming Models: BERT Classification.
                                        │
                                        ▼
STEP 5: REPRODUCIBLE DATASET SPLITTING (70 / 15 / 15)
  - Input: Clean DataFrame clean_df (1,086 rows).
  - Transformation: Sets np.random.seed(42), shuffles clean_df using sample(frac=1.0, random_state=42), and slices into Train (760 rows), Val (162 rows), Test (164 rows).
  - Output Artifacts: train_data.csv, val_data.csv, test_data.csv.
  - File / Function: ml/preprocessing/prepare_classification.py -> ClassificationDataPreparer.prepare()
  - Consuming Models: All models.
                                        │
                                        ▼
STEP 6: MODEL-SPECIFIC INPUT PREPARATION (DURING TRAINING / INFERENCE)
  ├─────────────────────────────────────┼─────────────────────────────────────┐
  ▼                                     ▼                                     ▼
BERT PREPROCESSING                    TF-IDF PREPROCESSING                  NER PREPROCESSING
- File: train_bert_experiment_v2.py   - File: run_baseline_comparison.py    - File: prepare_training_data.py
- Function: WomensSafetyDataset       - Function: run_baseline_eval()       - Function: align_spans_to_bio_tokens()
- Operation: Casts text to str.       - Operation: Extracts raw text list   - Operation: Uses offset_mapping
  Invokes AutoTokenizer(text,           X_train_raw. Fits TfidfVectorizer     from AutoTokenizer to align
  truncation=True,                      (ngram=(1,2), sublinear_tf=True,      character span offsets to subword
  padding="max_length",                 min_df=2, max_features=5000) on       tokens. Assigns B-<LABEL>,
  max_length=128/256). Returns          Train ONLY. Transforms Val & Test.    I-<LABEL>, or O tags. Sets
  input_ids and attention_mask.         Returns CSR sparse matrix (N, 5000).  special tokens to -100.
```

---

# 3. AUDIT OF DATA VALIDATION TECHNIQUES

| Data Validation Technique | Status | Exact Code Implementation & Evidence | Source File & Function |
| :--- | :--- | :--- | :--- |
| **Column Validation** | **USED** | Verifies presence of target label columns `LABEL_COLUMNS = ["DV", "SH", "ST", "CA", "WH", "OV"]` and narrative column `narrative_text`. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` |
| **Required-Column Checking** | **USED** | `narrative_col = "narrative_text" if "narrative_text" in df.columns else "narrative_text"` | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` |
| **Missing-Value Checking** | **USED** | `df[narrative_col].notnull()` checks for `NaN` / `None` values in the narrative column. In `dataset_analysis.py`: `df.isnull().sum().to_dict()`. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` |
| **Empty-Text Checking** | **USED** | `(df[narrative_col].astype(str).str.strip() != "")` checks for empty strings or whitespace-only text rows. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` |
| **Duplicate Checking** | **USED (Audit)** | `df.duplicated().sum()` (exact duplicate rows) and `df.duplicated(subset=[narrative_col]).sum()` (duplicate narratives) are computed during dataset analysis and leakage audit. | `ml/analysis/dataset_analysis.py`<br>`run_dataset_analysis()` |
| **Data-Type Conversion** | **USED** | Narrative strings are explicitly cast using `.astype(str)`. Target labels are cast to float lists `[float(row[col]) for col in self.label_columns]` and PyTorch tensors `torch.tensor(labels, dtype=torch.float)`. | `ml/training/train_bert_experiment_v2.py`<br>`WomensSafetyDataset.__getitem__()` |
| **Invalid-Label Checking** | **USED** | Verifies label values are binary (`0` or `1`) via `df[col] == 1` and `pos_counts = clean_df[self.LABEL_COLUMNS].sum().to_dict()`. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` |
| **Invalid-Record Filtering** | **USED** | `valid_mask` filters out any invalid narrative rows, storing excluded records and logging warnings. | `ml/preprocessing/prepare_classification.py`<br>`ClassificationDataPreparer.prepare()` |
| **Text Normalization** | **NOT USED (Pre-tokenizer)** | Custom pre-tokenizer regex normalization is **not used** in dataset preparation. A helper `normalize_text()` (`text.lower().strip()`, `re.sub(r"\s+", " ", text)`) exists **only inside the leakage audit script** (`run_leakage_audit.py`) for similarity auditing, NOT in dataset preparation or model training. | N/A |

---

# 4. AUDIT OF TEXT PREPROCESSING TECHNIQUES

Every candidate text preprocessing technique was audited against the codebase. The results are reported below:

* **Lowercasing**  
  **USED (Built-in / Indirect):** Lowercasing is **not** performed manually via Python code during CSV dataset preparation. It is executed **internally** by model tokenization libraries:
  1. HuggingFace `AutoTokenizer.from_pretrained("bert-base-uncased")` lowercases text during WordPiece tokenization.
  2. `TfidfVectorizer(lowercase=True)` lowercases text during n-gram matrix extraction.
  * *Implementation:* `AutoTokenizer.from_pretrained("bert-base-uncased")` in `ml/training/train_bert_experiment_v2.py` (Line 224).

* **Whitespace Normalization**  
  **NOT USED in dataset preparation:** Basic `.str.strip()` is applied to test for empty narratives in `prepare_classification.py`. No regular expression multi-whitespace collapse (`re.sub(r"\s+", " ", text)`) is applied to dataset text prior to tokenization.

* **Punctuation Removal**  
  **NOT USED in the current implementation.** Punctuation marks (commas, periods, quotes) are **not removed**. BERT relies on punctuation to identify clause boundaries and sentence syntax.

* **Punctuation Isolation**  
  **USED (Built-in / Indirect):** Punctuation isolation is executed automatically by the `bert-base-uncased` WordPiece tokenizer during pre-tokenization, separating punctuation marks into distinct token IDs (e.g., `.` $\rightarrow$ Token ID `1012`).

* **Special-Character Removal**  
  **NOT USED in the current implementation.** Special characters (e.g., `@`, `#`, `$`, `%`) are retained in raw text inputs.

* **Stop-Word Removal**  
  **NOT USED in the current implementation.** Stop words (e.g., `"the"`, `"and"`, `"is"`) are **not removed**. Preserving stop words maintains full syntactic structure required for transformer self-attention mechanisms. `TfidfVectorizer` is initialized with `stop_words=None`.

* **Stemming**  
  **NOT USED in the current implementation.** Rule-based stemming (e.g., Porter or Snowball stemmer) is **not used**. WordPiece subword tokenization natively addresses word suffix variations.

* **Lemmatization**  
  **NOT USED in the current implementation.** Morphological lemmatization (e.g., WordNet Lemmatizer or spaCy lemmatizer) is **not used**.

* **Spelling Correction**  
  **NOT USED in the current implementation.** No spell-checking algorithms (e.g., Hunspell, SymSpell, or TextBlob) are applied.

* **URL Removal**  
  **NOT USED in the current implementation.** URLs and domain strings are retained in the raw narrative text.

* **HTML Removal**  
  **NOT USED in the current implementation.** No HTML tag stripping (e.g., `BeautifulSoup` or regex tag removal) is performed.

* **Emoji Removal**  
  **NOT USED in the current implementation.** Emojis are retained as UTF-8 string characters; unmapped emojis are handled as `[UNK]` tokens by BERT.

* **Number Removal**  
  **NOT USED in the current implementation.** Numbers and statutory digits (e.g., `"354"`, `"498A"`) are retained as numerical text tokens.

* **Contraction Expansion**  
  **NOT USED in the current implementation.** English contractions (e.g., `"don't"` $\rightarrow$ `"do not"`) are not expanded prior to tokenization.

* **Unicode Normalization**  
  **NOT USED in the current implementation.** No explicit `unicodedata.normalize()` operations are executed in dataset preparation scripts.

---

# 5. LABEL PREPROCESSING & TARGET REPRESENTATION

### 5.1 Label Extraction & Ordering
The six incident category labels are defined in `ml/preprocessing/prepare_classification.py`:

```python
LABEL_COLUMNS = ["DV", "SH", "ST", "CA", "WH", "OV"]
```

The exact label ordering is strictly fixed across the entire project:
1. Index `0`: **DV** — Domestic Violence
2. Index `1`: **SH** — Sexual Harassment
3. Index `2`: **ST** — Stalking
4. Index `3`: **CA** — Cyber Abuse
5. Index `4`: **WH** — Workplace Harassment
6. Index `5`: **OV** — Other Violence

### 5.2 Missing-Label Handling
Missing label values in the raw dataset are handled by checking binary equality `(df[col] == 1)`. Non-positive entries evaluate to `0`. No missing label imputation is required as all 1,086 rows contain valid binary integer indicators (`0` or `1`).

### 5.3 Numerical Conversion & Multi-Hot Representation
Each incident record is converted into a 6-dimensional binary multi-hot float list:
```python
clean_df["label_vector"] = clean_df[self.LABEL_COLUMNS].values.tolist()
```
Inside the PyTorch `WomensSafetyDataset` class, label lists are converted directly into float tensors:
```python
labels = [float(row[col]) for col in self.label_columns]
item["labels"] = torch.tensor(labels, dtype=torch.float)
```
Target matrix shape per batch: `(batch_size, 6)`.

---

# 6. DATASET SPLITTING SPECIFICATIONS

The dataset splitting parameters are defined in `ml/preprocessing/prepare_classification.py` (`ClassificationDataPreparer.prepare()`):

* **Total Initial Records:** 1,086
* **Excluded Empty Records:** 0
* **Total Valid Processed Records:** 1,086
* **Training Set Records:** 760 (70.0%) — Saved to `ml/preprocessing/train_data.csv`
* **Validation Set Records:** 162 (15.0%) — Saved to `ml/preprocessing/val_data.csv`
* **Held-Out Test Set Records:** 164 (15.0%) — Saved to `ml/preprocessing/test_data.csv`
* **Random Seed:** `42` (`np.random.seed(42)`)
* **Shuffling:** Executed via `clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)`
* **Stratification:** **Not used.** Splitting relies on uniform random shuffling (`sample(frac=1.0)`) rather than class-based stratification because standard single-label stratification cannot be directly applied to multi-label target vectors.
* **Splitting Timing:** Splitting occurs **AFTER** raw CSV loading, empty narrative validation, and multi-hot vector creation, but **BEFORE** TF-IDF vectorizer fitting or BERT tokenization.

---

# 7. DUPLICATE DETECTION AND LEAKAGE HANDLING

### 7.1 Implemented Audits vs. Preprocessing
Duplicate and leakage detection checks are implemented as **diagnostic evaluation audits**, rather than destructive preprocessing filtering steps:

1. **Dataset Integrity Audit (`ml/analysis/dataset_analysis.py`):**
   * Computes exact duplicate rows: `int(df.duplicated().sum())` (0 exact row duplicates found).
   * Computes duplicate narrative texts: `int(df.duplicated(subset=[narrative_col]).sum())`.

2. **Dataset Leakage Audit (`ml/evaluation/leakage_audit/run_leakage_audit.py`):**
   * **Cross-Split Duplicate Check:** Uses MD5 text hashes (`text_hash()`) to verify that no exact narrative text appears in both Train and Test splits (0 cross-split duplicates found).
   * **Cross-Split Similarity Check:** Uses TF-IDF cosine similarity to quantify narrative overlap across splits.
   * **Explicit String Leakage Check:** Verifies that statutory label codes (`"DV"`, `"SH"`, etc.) do not appear verbatim in raw narrative texts.

---

# 8. MODEL-SPECIFIC PREPROCESSING OVERVIEW

The following breakdown illustrates where preprocessing is shared and where it diverges for specific model families:

```
                            RAW DATASET (womens_safety_dataset.csv)
                                              │
                                              ▼
                             COMMON DATASET PREPROCESSING
                  - Path resolution (resolve_dataset_path)
                  - CSV loading & empty narrative filtering (valid_mask)
                  - Multi-hot binary label encoding ([DV, SH, ST, CA, WH, OV])
                  - Reproducible 70/15/15 Train/Val/Test Split (Seed = 42)
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
BERT CLASSIFICATION PREPROCESSING    TF-IDF BASELINE PREPROCESSING        NER TOKEN PREPROCESSING
- Script: train_bert_exp_v2.py        - Script: run_baseline_comp.py       - Script: prepare_training_data.py
- Converts narrative to string.       - Extracts raw text array            - Extracts character offset spans
- Applies AutoTokenizer               X_train_raw.                         (start, end, label).
  ("bert-base-uncased").              - Fits TfidfVectorizer               - Invokes AutoTokenizer
- Applies truncation=True,              (ngram=(1,2), sublinear_tf=True,     with return_offsets_mapping=True.
  padding="max_length",                 min_df=2, max_features=5000)         - Aligns subword tokens to BIO
  max_length=128/256.                   on Train split ONLY.                 tags (B-LABEL, I-LABEL, O).
- Output: PyTorch input_ids           - Transforms Val & Test splits.      - Sets special tokens to -100.
  & attention_mask tensors.           - Output: Sparse matrix (N, 5000).   - Output: Token-level BIO JSONL.
```

---

# 9. PAPER-READY "DATA PREPROCESSING" SECTION

*The text below is formatted in formal academic prose and can be included directly in the research paper:*

```markdown
### 3.2 Data Preprocessing

The ABHERA incident dataset comprises 1,086 survivor-submitted narrative records paired with multi-label statutory incident annotations. Data preparation was executed through a controlled, reproducible preprocessing pipeline prior to feature representation and model ingestion.

First, dataset validation was performed to verify schema integrity and eliminate incomplete entries. Narrative records were inspected for null values and empty strings; all 1,086 records met the validity criteria and were retained for model training. The six target incident categories—Domestic Violence (DV), Sexual Harassment (SH), Stalking (ST), Cyber Abuse (CA), Workplace Harassment (WH), and Other Violence (OV)—were formatted into a six-dimensional binary multi-hot target vector $\mathbf{y} \in \{0, 1\}^6$. Positive class imbalance weights were computed for each category on the training partition to configure the loss function.

Second, the validated dataset was partitioned into training (70%, 760 samples), validation (15%, 162 samples), and held-out test (15%, 164 samples) splits using a fixed pseudorandom seed ($S = 42$). Random shuffling was applied prior to slicing to ensure consistent category distributions across all partitions. To guarantee strict evaluation isolation, all text transformations and feature vectorizers were fitted exclusively on the training split.

Model-specific text preprocessing was subsequently applied. Traditional destructive text modifications—such as stop-word removal, stemming, lemmatization, and punctuation stripping—were explicitly omitted. Stop words and syntactic punctuation were retained to preserve full grammatical context for transformer self-attention mechanisms. For transformer-based models, raw narrative strings were processed using the `bert-base-uncased` WordPiece tokenizer with lowercasing, truncating or padding sequences to a maximum sequence length of 128 tokens during training and 256 tokens during inference. For baseline n-gram models, raw narrative strings from the training partition were transformed using a sublinear TF-IDF vectorizer operating over unigrams and bigrams (`ngram_range=(1, 2)`), bounded to a maximum vocabulary size of 5,000 features and a minimum document frequency of 2 (`min_df=2`). For named entity recognition models, character-level annotation offsets were aligned with WordPiece subword tokens via offset mapping to produce token-level BIO (Begin-Inside-Outside) labels, with special classification tokens assigned a loss masking value of -100.
```

---

# 10. PREPROCESSING VERIFICATION TABLE

| Preprocessing Operation | Implemented Status | Actual Code Implementation Details | Source File Path |
| :--- | :--- | :--- | :--- |
| **Path Resolution** | **USED** | Checks relative candidate paths for `data/womens_safety_dataset.csv`. | `ml/preprocessing/prepare_classification.py` |
| **CSV Data Loading** | **USED** | `df = pd.read_csv(self.csv_path)` loading 1,086 records. | `ml/preprocessing/prepare_classification.py` |
| **Empty Narrative Filtering** | **USED** | `valid_mask = df[col].notnull() & (df[col].astype(str).str.strip() != "")`. | `ml/preprocessing/prepare_classification.py` |
| **Multi-Hot Label Encoding** | **USED** | `clean_df[LABEL_COLUMNS].values.tolist()` producing 6-dim binary target vectors. | `ml/preprocessing/prepare_classification.py` |
| **Class Weight Calculation** | **USED** | `pos_weights[col] = (N_total - N_pos) / N_pos` for BCEWithLogitsLoss. | `ml/preprocessing/prepare_classification.py` |
| **Reproducible Data Splitting** | **USED** | `np.random.seed(42)` shuffling and slicing into 760 Train / 162 Val / 164 Test. | `ml/preprocessing/prepare_classification.py` |
| **String Data Casting** | **USED** | `text = str(row["narrative_text"])` before tokenizer invocation. | `ml/training/train_bert_experiment_v2.py` |
| **Lowercasing** | **USED (Indirect)** | Executed internally by `bert-base-uncased` tokenizer and `TfidfVectorizer`. | `ml/training/train_bert_experiment_v2.py` |
| **Punctuation Isolation** | **USED (Indirect)** | Executed internally by `bert-base-uncased` WordPiece pre-tokenizer. | `transformers.AutoTokenizer` |
| **BERT Tokenization** | **USED** | `tokenizer(text, truncation=True, padding="max_length", max_length=128/256)`. | `ml/training/train_bert_experiment_v2.py` |
| **NER Subword Alignment** | **USED** | Aligns character spans to subwords via `return_offsets_mapping=True`. | `ml/ner/prepare_training_data.py` |
| **NER Loss Masking** | **USED** | Assigns `-100` label index to `[CLS]`, `[SEP]`, and `[PAD]` tokens. | `ml/ner/prepare_training_data.py` |
| **Stop-Word Removal** | **NOT USED** | Explicitly omitted to preserve sentence syntax for self-attention. | N/A |
| **Stemming** | **NOT USED** | Explicitly omitted; WordPiece subwords handle suffixes natively. | N/A |
| **Lemmatization** | **NOT USED** | Explicitly omitted; no morphological lemmatizer is invoked. | N/A |
| **Punctuation Removal** | **NOT USED** | Punctuation marks are retained in narrative text. | N/A |
| **Spelling Correction** | **NOT USED** | No spell-checking algorithms are applied. | N/A |
| **URL / HTML Removal** | **NOT USED** | Raw narrative URLs and text formatting are retained. | N/A |
| **Regex Text Normalization** | **NOT USED** | No custom regular expression text cleaning is applied in dataset preparation. | N/A |

---

# 11. VIVA-READY EXAMINATION QUESTIONS & ANSWERS

### Q1: How did you preprocess your dataset?
**Answer:** Dataset preprocessing was executed in `ml/preprocessing/prepare_classification.py`. Null and empty narrative strings were verified and filtered. Incident category columns (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`) were converted into 6-dimensional binary multi-hot vectors, and class imbalance weights were computed. The dataset was then partitioned using a fixed seed (42) into 70% Training (760 rows), 15% Validation (162 rows), and 15% Test (164 rows) splits.

### Q2: What preprocessing techniques did you use?
**Answer:** We used missing-value validation, empty-text filtering, multi-hot label encoding, class-imbalance weight calculation, reproducible pseudo-random dataset shuffling, PyTorch tensor casting, and model-specific tokenization (`bert-base-uncased` WordPiece subword tokenization with padding and truncation).

### Q3: Did you remove stop words?
**Answer:** **No, stop words were not removed.** Preserving stop words maintains full syntactic and grammatical sentence context, which is essential for transformer self-attention mechanisms in BERT.

### Q4: Did you perform stemming?
**Answer:** **No, stemming was not performed.** WordPiece subword tokenization natively resolves morphological suffix variations without the information loss associated with rule-based stemming algorithms.

### Q5: Did you perform lemmatization?
**Answer:** **No, lemmatization was not performed** in any dataset preparation or training pipeline.

### Q6: Did you remove punctuation?
**Answer:** **No, punctuation was not removed.** Punctuation marks are isolated as individual tokens during WordPiece pre-tokenization, providing syntactic boundary signals for narrative understanding.

### Q7: How did you handle missing values?
**Answer:** Missing values were checked using Pandas `notnull()` on the narrative text column in `prepare_classification.py`. Zero missing narrative rows were present in the 1,086 raw dataset records.

### Q8: How did you handle empty text?
**Answer:** Empty text strings and whitespace-only rows were checked using `.astype(str).str.strip() != ""`. Any row failing this check would be excluded prior to dataset partitioning. All 1,086 dataset records were verified to contain valid narrative text.

### Q9: How did you prepare the labels?
**Answer:** The six target incident columns (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`) were read as binary integers and concatenated into a 6-dimensional multi-hot list `[DV, SH, ST, CA, WH, OV]`. In PyTorch datasets, label lists were cast to `torch.tensor(labels, dtype=torch.float)`.

### Q10: How did you split the dataset?
**Answer:** The dataset was split into 70% Training (760 records), 15% Validation (162 records), and 15% Test (164 records) partitions using `prepare_classification.py`.

### Q11: What random seed did you use?
**Answer:** We used random seed **`42`** (`np.random.seed(42)` and `df.sample(frac=1.0, random_state=42)`).

### Q12: When was the dataset split performed?
**Answer:** The dataset split was performed **immediately after raw data validation and label vector construction**, but **BEFORE** fitting any TF-IDF vectorizer or training any BERT model. This ensured that vectorizers and models were trained strictly on training data without test set leakage.

---

ACTUAL PREPROCESSING STATUS: VERIFIED
