"""
Module for Chapter 16: Comprehensive Glossary and Quick Revision Sheet
"""

def get_chapter_16():
    return r'''# CHAPTER 16: Comprehensive Glossary & Quick Revision Sheet

---

## 16.1 Alphabetical Technical Glossary

Below is an exhaustive alphabetical glossary of all technical, legal, and machine learning terms used throughout the ABHERA project documentation, explained in crystal-clear language:

### A
- **ACID**: Atomicity, Consistency, Isolation, Durability. A set of four fundamental properties that guarantee database transactions are processed reliably and data is never corrupted.
- **Activation Function**: A mathematical formula (like Sigmoid or Softmax) applied at the output of an artificial neural network neuron to determine whether it should fire and what output value it should produce.
- **AdamW**: An advanced optimization algorithm used to train neural networks that adjusts learning rates dynamically and applies weight decay correctly to prevent overfitting.
- **Anti-Hallucination Layer**: A software security gating system in ABHERA that checks every legal section and helpline number against an audited PostgreSQL database before returning it to the user, mathematically preventing AI from inventing fake laws.
- **ASGI**: Asynchronous Server Gateway Interface. The modern standard interface between Python asynchronous web servers (like Uvicorn) and web frameworks (like FastAPI).
- **Attention Mask**: An array of `1`s and `0`s sent to a transformer model indicating which tokens are real words (`1`) and which are empty padding (`0`), preventing padding from skewing calculations.
- **Attention Mechanism**: A mathematical algorithm in transformers that calculates how much focus a neural network should place on every other word in a sentence when interpreting a specific word.
- **Auto-resizing Textarea**: A web user interface input box that dynamically expands downward as the user types more lines of text, preventing awkward inner scrollbars.
- **Autoregressive**: A class of language models (like GPT) that generate text sequentially by predicting one word at a time based on the words that came before it.

### B
- **Backpropagation**: The foundational mathematical process in neural network training where calculation errors (loss) are propagated backward through layers using the calculus chain rule to update internal weights.
- **BCEWithLogitsLoss**: A PyTorch loss function combining an unscaled linear layer with independent Sigmoid functions and Binary Cross-Entropy, delivering superior numerical stability.
- **BERT**: Bidirectional Encoder Representations from Transformers. A revolutionary deep learning model developed by Google that reads sentences in both directions simultaneously to understand contextual meaning.
- **Bharatiya Nyaya Sanhita (BNS 2023)**: The official criminal penal code of the Republic of India that replaced the historic Indian Penal Code (IPC 1860) on July 1, 2024.
- **Bidirectional**: The capability of a neural network to look at words appearing both before and after a target word at the same time, capturing full sentence context.
- **Binary Cross-Entropy (BCE)**: A mathematical formula used to measure error in yes-or-no classification tasks.
- **BIO Tagging**: A sequence labeling format where `B-` marks the beginning of an entity, `I-` marks continuing words inside the entity, and `O` marks words outside the entity.

### C
- **Character Offset Mapping**: A feature in fast tokenizers that tracks the exact starting and ending character index of every sub-word token relative to the original raw text string.
- **Class Imbalance**: A dataset condition where some categories have many examples (e.g., 292 cases of Sexual Harassment) while others have very few (e.g., 98 cases of Other Violence).
- **Client-Side**: Processing that occurs entirely inside the user's web browser on their computer or smartphone, without sending raw data to a remote server.
- **Cloudflare Tunnel**: An encrypted networking tool that securely exposes a local workstation server to the public internet without opening router ports or exposing home IP addresses.
- **Context API**: A built-in feature in React that allows state data (like session IDs and chat messages) to be shared globally across all components without manual prop drilling.
- **CORS**: Cross-Origin Resource Sharing. A browser security rule that controls which external websites are allowed to make HTTP API calls to a backend server.
- **CPU (Central Processing Unit)**: The primary general-purpose microprocessor of a computer that executes software instructions.

### D
- **Data Leakage**: A critical mistake in machine learning where information from the testing set accidentally leaks into the training set, causing deceptively high but invalid accuracy scores.
- **Decoupled Architecture**: A software design where the frontend user interface and the backend server operate as separate, independent systems communicating only over network APIs.
- **Discriminative Model**: A machine learning model that learns the boundary between different classes (like classifying crimes), as opposed to generating new synthetic text.
- **DOM (Document Object Model)**: The tree-like programming interface that web browsers use to represent and manipulate HTML elements on a screen.
- **Dowry Prohibition Act, 1961**: An Indian statutory act prohibiting the giving, taking, or demanding of dowry in marriage.
- **Dynamic Question Engine**: ABHERA's slot-filling state machine that identifies what forensic facts are missing from a survivor's narrative and selects targeted follow-up inquiries.

### E
- **Elementwise Accuracy**: The percentage of individual binary decisions (across all classes and samples) that a multi-label model gets right.
- **Embedding**: A multi-dimensional list of numbers (vector) representing the semantic meaning of a word, token, or sentence in geometric space.
- **Encoder**: A transformer neural network architecture that processes an entire input sequence simultaneously to produce rich contextual vector representations.
- **Entity**: A specific, structured piece of real-world information extracted from text, such as a person, place, time, or digital app.
- **Epoch**: One complete pass of the entire training dataset through the neural network during the training loop.
- **Exact Match Accuracy**: A strict multi-label evaluation metric where a prediction scores 1 only if every single class label matches ground truth simultaneously.

### F
- **FastAPI**: A high-performance asynchronous Python web framework used to build modern RESTful API microservices with automatic Pydantic validation.
- **False Negative (FN)**: A critical error where a crime actually occurred, but the AI model failed to detect it.
- **False Positive (FP)**: A mistake where a crime did not occur, but the AI model incorrectly flagged it (false alarm).
- **F1-Score**: The harmonic mean of precision and recall, providing an accurate, balanced performance score on imbalanced datasets.
- **FIR (First Information Report)**: The official written document prepared by police in India upon receiving information about the commission of a cognizable criminal offense.
- **Foreign Key**: A database column in a child table that points to the primary key of a parent table to enforce relational connections.

### G
- **Generalization**: The ability of a trained machine learning model to make accurate predictions on brand new, unseen data that it was never trained on.
- **GIL (Global Interpreter Lock)**: A mechanism in standard CPython that prevents multiple native threads from executing Python bytecodes simultaneously.
- **Grid Search**: An algorithmic tuning technique that tests every possible threshold value systematically on validation data to discover the optimal operating point.

### H
- **Hallucination**: The dangerous phenomenon where a generative AI model invents false facts, non-existent laws, or fake numbers with high confidence.
- **Hot Module Replacement (HMR)**: A developer feature in Vite that updates edited code in the browser instantly without requiring a full page refresh.
- **HTTP (Hypertext Transfer Protocol)**: The universal network communication protocol of the World Wide Web.
- **Hugging Face**: The leading open-source platform providing pre-trained transformer models, tokenizers, and machine learning datasets.

### I
- **Inference**: The stage where a fully trained neural network is given new input text and calculates predictions without updating its internal weights.
- **Information Technology Act, 2000 (IT Act)**: The primary Indian statutory law governing cybercrimes, digital privacy, and electronic records.
- **Input IDs**: The numerical list of integer numbers that represent words or sub-words after being processed by a tokenizer.
- **IPC (Indian Penal Code 1860)**: The historic criminal penal code of India that was officially superseded by the BNS on July 1, 2024.

### J
- **JSON (JavaScript Object Notation)**: A lightweight, text-based data interchange format used to send structured data across REST APIs.
- **jsPDF**: A client-side JavaScript library that compiles and generates PDF documents directly inside browser memory.

### K
- **Key (in Self-Attention)**: A projected vector representation of a token that acts like a searchable index label for attention queries.

### L
- **Logits**: The raw, unscaled numerical outputs produced by the final linear layer of a neural network before applying Sigmoid or Softmax.
- **Loss Function**: A mathematical equation measuring the discrepancy between a model's prediction and the actual ground truth label.

### M
- **Macro-F1**: An evaluation metric calculated by finding the F1-score of each class independently and computing their unweighted average.
- **Max Length**: The maximum number of tokens allowed in an input sequence; sequences longer than this limit are truncated.
- **Micro-F1**: An evaluation metric calculated by summing all true positives, false positives, and false negatives globally across all classes.
- **Multi-Head Attention**: An attention design that splits token projections into multiple parallel subspaces to attend to different linguistic features simultaneously.
- **Multi-Label Classification**: Machine learning where an input can belong to zero, one, or several classes simultaneously.

### N
- **Named Entity Recognition (NER)**: An NLP task that locates and tags real-world entities (people, places, apps) inside continuous text.
- **NCRB**: National Crime Records Bureau of India. The government agency responsible for collecting and analyzing crime data.
- **Neuro-Symbolic AI**: An AI architecture combining neural networks (for statistical understanding) with symbolic rules/databases (for deterministic truth).
- **NLP (Natural Language Processing)**: The branch of AI focused on enabling computers to understand human language.

### O
- **Offset Mapping**: Tuples tracking the character start and end positions of sub-word tokens in the raw text string.
- **Optimal Decision Threshold**: A custom probability cutoff calibrated per class on validation data, replacing arbitrary 0.50 cutoffs.
- **ORM (Object-Relational Mapping)**: Software that lets developers interact with relational databases using object-oriented classes.
- **Out-of-Vocabulary (OOV)**: Words not present in a model's vocabulary list, solved by sub-word tokenizers like WordPiece.
- **Overfitting**: When an AI memorizes training data too closely and fails on new, unseen test data.

### P
- **Padding**: Adding special filler tokens (`[PAD]`) to shorter sentences so all sequences in a batch have equal length.
- **Pervasive Stalking**: Persistent, repeated following or digital monitoring covered under Section 78 of the BNS 2023.
- **POSH Act, 2013**: Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013.
- **Positive-Class Weight (`pos_weight`)**: A loss multiplier that penalizes errors on rare positive classes more heavily, fixing class imbalance.
- **PostgreSQL**: An open-source object-relational database known for ACID compliance, standards adherence, and JSON support.
- **Precision**: The fraction of positive predictions that were actually correct ($\frac{TP}{TP + FP}$).
- **Primary Key**: A column that uniquely identifies each individual record in a database table.
- **Pydantic**: A fast data parsing and validation library in Python based on standard type hints.
- **PWDVA, 2005**: Protection of Women from Domestic Violence Act, 2005. Providing civil protection, residence, and monetary orders.
- **PyTorch**: An open-source deep learning tensor computation and automatic differentiation framework developed by Meta.

### Q
- **Query (in Self-Attention)**: The vector representing what a specific token is searching for in other tokens.

### R
- **Recall**: The fraction of all real, actual positive cases that the model successfully caught ($\frac{TP}{TP + FN}$).
- **Referential Integrity**: A database condition ensuring that links between tables remain valid and child rows cannot exist without parent rows.
- **REST (Representational State Transfer)**: A standard architectural style for building networked web APIs using HTTP.

### S
- **Safetensors**: A modern, safe file format for saving neural network tensor weights without security risks.
- **Self-Attention**: The core mechanism of the transformer that calculates attention weights between all token pairs in a sentence.
- **Session**: A continuous sequence of conversational turns exchanged between a specific user and the backend assistant.
- **`sessionStorage`**: Browser memory that stores key-value pairs strictly for the duration of the current browser tab.
- **Sigmoid**: The mathematical activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ that bounds numbers between 0.0 and 1.0.
- **Single Page Application (SPA)**: A web app that loads once and updates dynamically via JavaScript without full browser reloads.
- **Softmax**: An activation function that normalizes a vector of numbers into a probability distribution summing to 1.0.
- **SQL (Structured Query Language)**: The standardized domain language used to query and manage relational databases.
- **SQLAlchemy**: The premier SQL toolkit and Object-Relational Mapper for Python.
- **Sub-word Tokenization**: Splitting words into smaller frequent chunks to eliminate out-of-vocabulary errors.

### T
- **Tailwind CSS**: A utility-first CSS framework providing pre-built styling classes for responsive web design.
- **Test Set**: An isolated dataset held out during training and threshold tuning, used only for final unbiased evaluation.
- **Token**: A sub-word unit or character sequence that represents the basic atomic input piece for a language model.
- **Token Classification**: An NLP task where an AI assigns a category label to every individual token in a sequence.
- **Transformer**: The dominant neural network architecture based on self-attention mechanisms, introduced in 2017.
- **True Positive (TP)**: A correct prediction where an actual crime was accurately detected by the AI.

### U
- **Uvicorn**: An ultra-fast ASGI web server implementation in Python used to run FastAPI applications.

### V
- **Validation Set**: A separate dataset used during training to monitor loss, prevent overfitting, and tune decision thresholds.
- **Vercel**: A cloud hosting platform that deploys web frontends across a worldwide Anycast Edge Content Delivery Network.
- **Virtual DOM**: An in-memory representation of real HTML DOM elements used by React to compute efficient screen updates.
- **Vite**: A modern, ultra-fast frontend build tool and local development server.

### W
- **Weight Decay**: A regularization technique that penalizes large neural network weights, preventing overfitting.
- **WordPiece**: The sub-word tokenization algorithm utilized by Google's BERT models.

---

## 16.2 Final Quick Revision Cheatsheet for Viva Day

Keep this one-page summary in mind before entering your examination room:

```
+===================================================================================================+
|                                    ABHERA QUICK REVISION CHEATSHEET                               |
+===================================================================================================+
| 1. CORE IDENTITY                                                                                  |
| - Full Form: Adaptive BERT-based Harassment Entity Recognition & Assistance                       |
| - Domain: AI, NLP, Women's Safety & Legal Informatics under Indian Law (BNS 2023)                 |
| - Architecture: Neuro-Symbolic (BERT Deep Learning + Deterministic PostgreSQL Knowledge Base)    |
+---------------------------------------------------------------------------------------------------+
| 2. VERIFIED ACCURACY BENCHMARKS (164 Held-Out Test Records)                                       |
| - Classification Micro-F1: 98.09% | Macro-F1: 97.58% | Exact Match Accuracy: 95.73%               |
| - Multi-Label Exact Match: 94.12% | Elementwise Accuracy: 99.29%                                  |
| - Total Decisions: 180 True Positives, 6 False Positives, ONLY 1 False Negative (TEST_0050)       |
| - NER Model Micro-F1: 99.24% | Precision: 100.0% | Recall: 98.48% (65 TP, 0 FP, 1 FN)            |
+---------------------------------------------------------------------------------------------------+
| 3. MODEL CONFIGURATION & PER-CLASS OPTIMAL THRESHOLDS                                             |
| - Model: bert-base-uncased (12 layers, 768 hidden dim, 12 heads, ~110M params)                   |
| - Loss: BCEWithLogitsLoss with pos_weight (OV: 10.08, WH: 5.50, ST: 5.10, CA: 3.76, DV: 3.34)     |
| - Optimal Thresholds: DV: 0.73 | SH: 0.50 | ST: 0.79 | CA: 0.45 | WH: 0.29 | OV: 0.63            |
| - Sequence Length: 128 (Classification), 256 (NER) | Batch Size: 32 | Optimizer: AdamW (lr: 3e-5) |
+---------------------------------------------------------------------------------------------------+
| 4. DATABASE & LEGAL REPOSITORY (PostgreSQL 16)                                                    |
| - Tables: Exactly 11 (users, sessions, incident_submissions, conversation_messages, predictions,  |
|            entities, questions, submission_question_answers, laws, support_services, reports)     |
| - Statutory Scope: 32 Audited Provisions across BNS 2023, IT Act 2000, POSH 2013, PWDVA 2005      |
| - Support Services: 12 Audited Services (Tier 1: 112, 181, 1090; Tier 2: Cybercrime; Tier 3: OSC) |
+---------------------------------------------------------------------------------------------------+
| 5. SYSTEM WORKFLOW & SAFEGUARDS                                                                   |
| - Flow: User Narrative -> BERT Classification -> NER Extraction -> Dynamic Questioning ->         |
|         Anti-Hallucination SQL Gating -> 10-Section Dossier Assembly -> Client-Side jsPDF Export  |
| - Testing: Exactly 154 Passing Automated Unit Tests across 11 Test Modules                        |
| - Deployment: Free Hybrid (React 18 on Vercel Edge + Local FastAPI/PyTorch via Cloudflare Tunnel) |
| - Privacy: Zero PII, no passwords/emails, transient sessionStorage, browser RAM vector PDF        |
+===================================================================================================+
```
'''
