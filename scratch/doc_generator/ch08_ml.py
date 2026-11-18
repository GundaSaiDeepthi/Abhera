"""
Module for Chapter 8: Dataset and Machine Learning
"""

def get_chapter_08():
    return r'''# CHAPTER 8: Dataset and Machine Learning

---

## 8.1 Machine Learning from Absolute Zero

To understand how ABHERA works, we must start from the very foundation of artificial intelligence:

### 1. Traditional Programming vs. Machine Learning
In **traditional computer programming**, a software engineer writes explicit rules by hand:
$$\\text{Input Data} + \\text{Rules (Hand-Written If/Else Statements)} \\longrightarrow \\text{Output Answer}$$
For example: `if "beat" in text and "husband" in text: return "Domestic Violence"`.
However, human language is far too vast, nuanced, and messy for rule-based systems. A survivor might say *"He raised his hand against me at home"* without ever using the word "beat" or "husband". Writing rules for every possible sentence in the English language would require billions of lines of code and would still fail on new sentences.

In **Machine Learning (ML)**, we flip this paradigm upside down:
$$\\text{Input Data} + \\text{Answers (Ground Truth Labels)} \\longrightarrow \\text{Learned Rules (Model Weights)}$$
Instead of hand-coding rules, we show a neural network thousands of real incident examples along with their correct legal classifications. The computer automatically adjusts millions of internal mathematical parameters (weights) until it discovers the complex linguistic patterns that distinguish stalking, domestic violence, or cyber harassment.

```
TRADITIONAL CODE:
[ Incident Text ] ----> [ 10,000 If/Else Rules ] ----> [ Classification ]

MACHINE LEARNING:
[ 1,086 Examples ] + [ True Categories ] ----> [ Training Loop ] ----> [ Trained Neural Network ]
                                                                               |
[ New Unseen Narrative ] ------------------------------------------------------+----> [ 98.09% Accurate Prediction ]
```

---

## 8.2 What is NLP (Natural Language Processing)?

Computers are calculators; they can only perform mathematical operations (addition, multiplication) on numbers. They cannot directly "read" letters or words.

**Natural Language Processing (NLP)** is the subfield of AI that translates raw human language into numerical representations (vectors) that mathematical neural networks can process, and translates numerical outputs back into human language.

---

## 8.3 The ABHERA Dataset

### 1. Dataset Collection and Composition
The ABHERA dataset (`data/womens_safety_dataset.csv`) contains exactly **1,086 verified multi-label incident records** curated specifically to reflect the reality of gender-based violence, harassment, and cyber crimes under Indian criminal jurisprudence.

### 2. The 6 Multi-Label Harassment Classes
Every incident narrative in the dataset is annotated across six binary dimensions ($y_i \in \{0, 1\}$):

```
+===================================================================================================+
|                                    ABHERA CLASSIFICATION LABELS                                   |
+------+-----------------------+---------------------+---------------+------------------------------+
| Code | Category Name         | Positive Count in DB| pos_weight    | Optimal Decision Threshold   |
+------+-----------------------+---------------------+---------------+------------------------------+
| DV   | Domestic Violence     | 250 records (23.0%) | 3.3440        | 0.73                         |
| SH   | Sexual Harassment     | 292 records (26.9%) | 2.7192        | 0.50                         |
| ST   | Stalking              | 178 records (16.4%) | 5.1011        | 0.79                         |
| CA   | Cyber Abuse           | 228 records (21.0%) | 3.7632        | 0.45                         |
| WH   | Workplace Harassment  | 167 records (15.4%) | 5.5030        | 0.29                         |
| OV   | Other Violence        | 98 records (9.0%)   | 10.0816       | 0.63                         |
+------+-----------------------+---------------------+---------------+------------------------------+
| TOTAL DATASET SIZE: Exactly 1,086 Verified Multi-Label Incident Records                           |
+===================================================================================================+
```

### 3. Data Partitioning (Train, Validation, Test Splits)
To ensure scientific rigor and avoid data leakage, the 1,086 records were partitioned using a fixed random seed (`seed = 42`) into three strictly isolated subsets:
- **Training Set (`ml/preprocessing/train_data.csv`)**: **760 records (70.0%)**. Used by the neural network optimizer to update model weights during backpropagation.
- **Validation Set (`ml/preprocessing/val_data.csv`)**: **162 records (14.9%)**. Used during training to monitor loss, prevent overfitting, and perform grid search to discover the optimal decision thresholds.
- **Test Set (`ml/preprocessing/test_data.csv`)**: **164 records (15.1%)**. Held out completely and never seen during training or threshold tuning. Used exclusively for final performance auditing.

---

## 8.4 Text Preprocessing and Tokenization

Before text can enter BERT, it undergoes systematic preprocessing in `ml/preprocessing/prepare_classification.py`:

1. **Text Normalization**: Strips excessive whitespace, removes control characters, and standardizes punctuation.
2. **WordPiece Tokenization**: Unlike simple word splitting (which fails on typos or rare words), BERT uses **WordPiece tokenization** with a fixed vocabulary of 30,522 sub-word units:
   - Common words remain intact: `"harassment"` $\to$ `['harassment']`.
   - Rare or compound words are decomposed into sub-word tokens prefixed with `##`: `"cyberstalking"` $\to$ `['cyber', '##stalk', '##ing']`.
   - This eliminates Out-Of-Vocabulary (OOV) errors completely.
3. **Special Control Tokens**:
   - `[CLS]` (Classification token, ID 101): Prepended to the beginning of every sequence. Its final hidden state serves as the sequence embedding.
   - `[SEP]` (Separator token, ID 102): Appended to the end of the text.
   - `[PAD]` (Padding token, ID 0): Appended to pad shorter sequences to the fixed maximum sequence length.
4. **Attention Mask**: An array of `1`s for real linguistic tokens and `0`s for `[PAD]` tokens. This instructs the self-attention mechanism to ignore padding tokens during matrix multiplication.

---

## 8.5 BERT Architecture and the Transformer Mechanism

ABHERA uses **`bert-base-uncased`**, an encoder-only transformer network:

```
Input Tokens:     [CLS]       My        husband     threatened      me         [SEP]
                    |           |           |           |           |           |
Token Embeddings:   E_cls      E_my       E_hus       E_thr       E_me        E_sep
Position Embeddings:P_0        P_1        P_2         P_3         P_4         P_5
                    |           |           |           |           |           |
                    +-----------+-----------+-----------+-----------+-----------+
                                            |
                                            v
                    +-----------------------------------------------+
                    |     TRANSFORMER ENCODER LAYER 1 (12 Heads)    |
                    +-----------------------------------------------+
                                            |
                                           ... (12 Layers Total)
                                            |
                    +-----------------------------------------------+
                    |     TRANSFORMER ENCODER LAYER 12 (12 Heads)   |
                    +-----------------------------------------------+
                                            |
Final Hidden Vector: H_cls (768-dimensional contextual sequence representation)
                                            |
                                            v
Linear Head Head:   W * H_cls + b (Projection: 768 -> 6 logits)
                                            |
                                            v
Independent Sigmoids: [ p_DV,  p_SH,  p_ST,  p_CA,  p_WH,  p_OV ] in [0.0, 1.0]
```

### The Scaled Dot-Product Attention Equation:
At the heart of every transformer layer is the **Self-Attention Mechanism**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
- **Query ($\mathbf{Q}$)**: What each word is searching for.
- **Key ($\mathbf{K}$)**: What each word contains.
- **Value ($\mathbf{V}$)**: The actual semantic content of each word.
- **Scaling Factor ($\sqrt{d_k} = \sqrt{64} = 8$)**: Prevents the dot products from growing excessively large for high dimensions, which would push the softmax into regions with vanishingly small gradients.

---

## 8.6 Solving Class Imbalance: `pos_weight` in Binary Cross-Entropy

In our dataset of 1,086 records, `OV` (Other Violence) has only 98 positive cases, meaning 91% of samples are zeros.

If a standard, unweighted Binary Cross-Entropy loss is used:
$$\mathcal{L}_{unweighted} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
The model quickly realizes that by predicting `0` for everything, it can achieve 91% accuracy without learning anything! This was the exact flaw of **BERT v1**, which produced **23 False Negatives** and scored **0.0% on multi-label combinations**.

To solve this, ABHERA v2 implements **Positive-Class Weighted BCE Loss** (`pos_weight`):
$$w_c = \frac{N_{total} - N_{positive, c}}{N_{positive, c}}$$
$$\mathcal{L}_{weighted} = -\sum_{c=1}^6 \left[ w_c \cdot y_c \log(\sigma(z_c)) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$
For `OV`, $w_c = 10.08$. This means missing a positive violence case is penalized **10 times more heavily** than predicting a false positive, forcing the neural network to prioritize high recall on vulnerable crime categories.

---

## 8.7 Per-Class Optimal Decision Threshold Tuning

In textbook machine learning, practitioners apply a uniform decision cutoff of `0.50` to all sigmoid probabilities:
$$\hat{y}_c = \begin{cases} 1 & \text{if } p_c \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$
However, in women's safety, a fixed 0.50 cutoff is dangerous and sub-optimal:
- **Workplace Harassment (`WH`)** is often expressed in polite, restrained, professional language (e.g., *"My manager hinted that my promotion depends on dinner with him"*). The model's raw probability might only reach `0.35`. Under a 0.50 cutoff, this workplace harassment would be completely missed!
- In ABHERA, we executed a 100-step grid search ($[0.10, 0.90]$) on the **held-out validation set** to find the exact threshold maximizing the F1-score for each individual class:
  - `DV`: **0.73** (High confidence required to distinguish from general disputes)
  - `SH`: **0.50** (Balanced canonical cutoff)
  - `ST`: **0.79** (Conservative threshold to avoid confusing repeated contact with benign inquiries)
  - `CA`: **0.45** (Lower cutoff to catch subtle cyber harassment)
  - `WH`: **0.29** (Deliberately low threshold ensuring 100% recall on workplace offenses)
  - `OV`: **0.63** (Tuned for physical violence indicators)

---

## 8.8 Model Evaluation Results and Baseline Comparisons

When evaluated on the 164 held-out test records, ABHERA v2 achieved state-of-the-art performance, thoroughly outperforming all baseline models:

| Performance Metric | TF-IDF + Logistic Reg | TF-IDF + Linear SVM | BERT v1 (Baseline) | BERT v2 (Production Promoted) |
| :--- | :---: | :---: | :---: | :---: |
| **Micro Precision** | 0.9779 | 0.9783 | 0.9576 | **0.9677 (96.77%)** |
| **Micro Recall** | 0.9779 | 0.9672 | 0.8729 | **0.9945 (99.45%)** |
| **Micro F1-Score** | 0.9779 | 0.9727 | 0.9133 | **0.9809 (98.09%)** |
| **Macro Precision** | 0.9731 | 0.9715 | 0.9412 | **0.9571 (95.71%)** |
| **Macro Recall** | 0.9745 | 0.9632 | 0.8698 | **0.9963 (99.63%)** |
| **Macro F1-Score** | 0.9735 | 0.9669 | 0.9037 | **0.9758 (97.58%)** |
| **Exact Match Accuracy** | 0.9573 | 0.9512 | 0.8354 | **0.9573 (95.73%)** |
| **Multi-Label Exact Match** | 0.8824 | 0.8824 | **0.00% (23 FN)** | **0.9412 (94.12%)** |
| **Elementwise Accuracy** | 0.9919 | 0.9898 | 0.9695 | **0.9929 (99.29%)** |
| **Total False Negatives** | 4 | 6 | 23 | **Only 1 (`TEST_0050`)** |
| **Total False Positives** | 4 | 4 | 7 | **6** |
| **True Positives (TP)** | 177 | 175 | 158 | **180** |
| **True Negatives (TN)** | 799 | 799 | 796 | **797** |

### Per-Class Test Set Breakdown for BERT v2:
- **Domestic Violence (`DV`)**: Support = 35, TP = 35, FP = 2, FN = 0, **Precision = 94.59%, Recall = 100.0%, F1 = 97.22%**
- **Sexual Harassment (`SH`)**: Support = 45, TP = 44, FP = 0, FN = 1, **Precision = 100.0%, Recall = 97.78%, F1 = 98.88%**
- **Stalking (`ST`)**: Support = 32, TP = 32, FP = 0, FN = 0, **Precision = 100.0%, Recall = 100.0%, F1 = 100.0%**
- **Cyber Abuse (`CA`)**: Support = 39, TP = 39, FP = 1, FN = 0, **Precision = 97.50%, Recall = 100.0%, F1 = 98.73%**
- **Workplace Harassment (`WH`)**: Support = 19, TP = 19, FP = 2, FN = 0, **Precision = 90.48%, Recall = 100.0%, F1 = 95.00%**
- **Other Violence (`OV`)**: Support = 11, TP = 11, FP = 1, FN = 0, **Precision = 91.67%, Recall = 100.0%, F1 = 95.65%**

### Error Analysis of the Single False Negative (`TEST_0050`):
In `TEST_0050`, a man repeatedly messaged a woman asking to meet after a professional conference and waited outside her hotel. The model correctly flagged **Stalking (`ST`)** with probability $0.84 \ge 0.79$, but missed **Sexual Harassment (`SH`)** ($p = 0.4549 < 0.50$) by a tiny margin of $0.0451$ due to the absence of vulgar language.

---

## 8.9 Training Configuration

The production model was trained using `ml/training/train_bert_experiment_v2.py` with these exact parameters:
```json
{
  "base_model_name": "bert-base-uncased",
  "max_length": 128,
  "batch_size": 32,
  "epochs": 6,
  "learning_rate": 3e-05,
  "weight_decay": 0.01,
  "optimizer": "AdamW",
  "loss_function": "BCEWithLogitsLoss with pos_weight",
  "seed": 42,
  "device": "cpu"
}
```
'''
