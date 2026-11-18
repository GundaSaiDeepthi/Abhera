"""
Module for Chapter 14: Viva Preparation - Part 2 (Sections G through M, Questions 82 to 170)
"""

def get_viva_part2():
    return r'''## SECTION G: NLP and BERT Questions (Q82 – Q96)

### Q82: What is BERT and what does it stand for?
- **Beginner Answer**: BERT stands for **Bidirectional Encoder Representations from Transformers**. It is an advanced AI created by Google that reads sentences in both directions at once to deeply understand the true meaning of text.
- **Technical Explanation**: BERT is a multi-layer bidirectional transformer encoder model trained on self-supervised tasks (Masked Language Modeling and Next Sentence Prediction). Unlike older left-to-right architectures, BERT conditions on both left and right context across all layers simultaneously.
- **ABHERA Example**: We fine-tune `bert-base-uncased` for 6-class multi-label harassment classification and for 13-tag token sequence classification.

### Q83: What is the difference between an Encoder-only and a Decoder-only model?
- **Beginner Answer**: An Encoder (like BERT) reads and analyzes existing text to understand and categorize it. A Decoder (like GPT or ChatGPT) generates new text word-by-word.
- **Technical Explanation**: Encoder-only models use bidirectional self-attention where every token attends to every other token in the sequence, making them optimal for classification and sequence labeling. Decoder-only models use causal/masked self-attention where tokens only attend to preceding tokens, designed for autoregressive text generation.
- **ABHERA Example**: ABHERA deliberately uses an Encoder-only model (BERT) for factual understanding, avoiding Decoder models (GPT) to eliminate generative hallucination.

### Q84: What are the exact architectural specifications of `bert-base-uncased`?
- **Beginner Answer**: It has 12 layers of neural processors, a memory width of 768 units, 12 attention heads per layer, and roughly 110 million tunable dials (parameters).
- **Technical Explanation**:
  - $L = 12$ Transformer encoder blocks
  - $d_{model} = 768$ hidden dimension
  - $H = 12$ multi-head attention heads ($d_k = d_v = 768 / 12 = 64$)
  - $d_{ff} = 3072$ intermediate feed-forward dimension
  - Total parameters: $\approx 109.5 \text{ million}$
  - Vocabulary: 30,522 WordPiece tokens
- **ABHERA Example**: Loaded from `models/bert_multilabel_experiment_v2/config.json`.

### Q85: What is the self-attention mechanism in Transformers?
- **Beginner Answer**: It is how the computer looks at every word in a sentence and calculates which other words in that sentence are most closely related to it.
- **Technical Explanation**: A mechanism that computes dynamic contextual relationships between all token pairs in a sequence. It projects input vectors into Queries ($\mathbf{Q}$), Keys ($\mathbf{K}$), and Values ($\mathbf{V}$), calculating attention weights as:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
- **ABHERA Example**: In the sentence *"He followed me home"*, self-attention links *"followed"* strongly with *"home"*, signaling physical stalking.

### Q86: Why is the dot product scaled by $\sqrt{d_k}$ in self-attention?
- **Beginner Answer**: When you multiply large lists of numbers together, the result can become gigantic. Dividing by $\sqrt{d_k}$ shrinks the numbers back down so the math doesn't break.
- **Technical Explanation**: For large projection dimensions ($d_k = 64$), the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients ($\approx 0$). Dividing by $\sqrt{d_k} = \sqrt{64} = 8$ stabilizes variance to 1.0, preventing vanishing gradients during backpropagation.
- **ABHERA Example**: Standard formulation inside PyTorch's `MultiheadAttention` module.

### Q87: What is the purpose of the `[CLS]` token?
- **Beginner Answer**: It is a special marker placed at the very start of the sentence that collects the overall summary meaning of the entire sentence.
- **Technical Explanation**: The `[CLS]` (Classification) token is appended at index 0 of every input sequence. Because its representation is updated via bidirectional self-attention across all 12 transformer layers, its final hidden state $\mathbf{h}_{[CLS]} \in \mathbb{R}^{768}$ serves as the pooled semantic vector of the sequence, fed into linear classification heads.
- **ABHERA Example**: Sliced via `outputs.logits = linear_head(hidden_states[:, 0, :])`.

### Q88: What is the purpose of the `[SEP]` and `[PAD]` tokens?
- **Beginner Answer**: `[SEP]` is like a period marking the end of a sentence. `[PAD]` is empty filler added to short sentences so all sentences in a batch have the exact same length.
- **Technical Explanation**:
  - `[SEP]` (Separator, ID 102): Demarcates sequence boundaries.
  - `[PAD]` (Padding, ID 0): Equalizes sequence lengths across batches to enable efficient parallel tensor matrix multiplication on CPUs/GPUs.
- **ABHERA Example**: In `predict_bert.py`, sequences are padded to `max_length = 128`.

### Q89: What is WordPiece tokenization and how does it work?
- **Beginner Answer**: It breaks long or uncommon words into smaller puzzle pieces (sub-words) that the computer already knows, so it never gets stuck on unfamiliar words.
- **Technical Explanation**: A greedy sub-word segmentation algorithm initialized with individual characters and iteratively merging frequent pairs based on maximum likelihood. Sub-words that are not word-initial are prefixed with `##` (e.g., `"cybercrime"` $\to$ `['cyber', '##cr', '##ime']`).
- **ABHERA Example**: Enables BERT to understand domain-specific terms without an infinite vocabulary size.

### Q90: Why did ABHERA increase `max_length` from 64 in v1 to 128 in v2?
- **Beginner Answer**: 64 words was too short; it chopped off the ends of long stories where survivors explained the worst parts of their ordeal. 128 words captured almost all stories completely.
- **Technical Explanation**: In BERT v1, a 64-token limit truncated approximately 38% of survivor narratives, discarding critical evidentiary context that occurred at the end of the text. Setting `max_length = 128` preserved 99.4% of full narratives without truncation while keeping CPU inference latency under 300ms.
- **ABHERA Example**: Configured in `models/bert_multilabel_experiment_v2/training_config.json`.

### Q91: What are positional embeddings and why does BERT need them?
- **Beginner Answer**: Transformers look at all words at the same time, so without positional tags, they wouldn't know the difference between "dog bit man" and "man bit dog".
- **Technical Explanation**: Self-attention is permutation-invariant—it has no inherent awareness of word order. Positional embeddings $\mathbf{P} \in \mathbb{R}^{N \times d}$ are added to token embeddings ($\mathbf{E}_{input} = \mathbf{E}_{token} + \mathbf{E}_{pos} + \mathbf{E}_{segment}$) to inject spatial sequence ordering into the transformer.
- **ABHERA Example**: BERT uses learned absolute 1D positional embeddings up to 512 positions.

### Q92: What are Multi-Head Attention heads?
- **Beginner Answer**: Having 12 different pairs of eyes looking at the sentence simultaneously: one pair focuses on who did it, another focuses on where, and another focuses on how.
- **Technical Explanation**: Rather than computing a single attention distribution, Multi-Head Attention linearly projects $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into $H=12$ distinct lower-dimensional subspaces ($d_k = 64$), computes self-attention in parallel, and concatenates the outputs:
  $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\mathbf{W}^O$$
- **ABHERA Example**: In BERT Base, each of the 12 encoder layers uses 12 attention heads.

### Q93: What are the 6 optimal decision thresholds in ABHERA?
- **Beginner Answer**:
  - Domestic Violence (`DV`): **0.73**
  - Sexual Harassment (`SH`): **0.50**
  - Stalking (`ST`): **0.79**
  - Cyber Abuse (`CA`): **0.45**
  - Workplace Harassment (`WH`): **0.29**
  - Other Violence (`OV`): **0.63**
- **Technical Explanation**: Calibrated via 100-step grid search on the held-out validation set to maximize per-class F1-scores without touching the test set.
- **ABHERA Example**: Stored in `models/bert_multilabel_experiment_v2/optimal_thresholds.json`.

### Q94: Why is the threshold for Workplace Harassment (`WH`) set so low (0.29)?
- **Beginner Answer**: Victims of workplace harassment usually describe it very politely and quietly because they fear losing their jobs. A lower threshold ensures the AI doesn't miss these subtle cases.
- **Technical Explanation**: Workplace disclosures often feature formal, corporate language lacking overt violence words, producing lower raw transformer logits. The 0.29 threshold was determined mathematically on validation loss to guarantee 100% recall on workplace offenses.
- **ABHERA Example**: On the 164-sample test set, `WH` achieved 19 / 19 True Positives (100.0% recall).

### Q95: Why did BERT v1 fail completely on multi-label combinations (0.0% exact match)?
- **Beginner Answer**: It only trained for 2 quick rounds with no special weighting for rare crimes, so it got lazy and only predicted the most obvious single crime.
- **Technical Explanation**: BERT v1 was trained for only 2 epochs using unweighted BCE loss. The optimizer converged to local minima biased toward negative zeros, producing 23 False Negatives on multi-label records.
- **ABHERA Example**: Resolved in v2 by increasing epochs to 6, setting `pos_weight`, and tuning per-class thresholds, raising Multi-Label Exact Match to **94.12%**.

### Q96: What is the format of saved model weights in ABHERA (`.safetensors` vs `.bin`)?
- **Beginner Answer**: Safetensors is a safer, faster modern file format that stores the AI's learned weights without the risk of computer viruses.
- **Technical Explanation**: Traditional PyTorch `.bin` files use Python's `pickle` library, which allows arbitrary code execution vulnerabilities during deserialization. Hugging Face's `safetensors` format stores raw tensor buffers securely with zero-copy memory mapping (`mmap`), improving load speed and security.
- **ABHERA Example**: `models/bert_multilabel_experiment_v2/model.safetensors` (417.7 MB).

---

## SECTION H: NER Questions (Q97 – Q108)

### Q97: What is Named Entity Recognition (NER) and how does it differ from classification?
- **Beginner Answer**: Classification tells you what general crime happened; NER highlights specific details inside the sentence (like people's names, office locations, and phone apps).
- **Technical Explanation**: Classification is sequence-level categorization ($f: \text{Sequence} \to \mathcal{C}$); NER is dense token-level sequence labeling ($f: \text{Token}_i \to \mathcal{T}$), classifying each individual token into an entity tag ontology.
- **ABHERA Example**: In *"He messaged me on WhatsApp"*, classification predicts `CA`, while NER identifies `"WhatsApp"` as `PLATFORM`.

### Q98: What does BIO tagging mean?
- **Beginner Answer**: **B**egin (first word of a detail), **I**nside (middle/end words of that same detail), and **O**utside (regular words that are not details).
- **Technical Explanation**: The standard IOB2 sequence representation scheme:
  - `B-Tag`: Indicates the initial token of an entity span.
  - `I-Tag`: Indicates subsequent interior tokens of that same span.
  - `O`: Indicates tokens that do not belong to any entity span.
- **ABHERA Example**: `"office"` $\to$ `B-LOCATION`, `"canteen"` $\to$ `I-LOCATION`.

### Q99: How many BIO tags are used in ABHERA?
- **Beginner Answer**: Exactly 13 tags: Outside (`O`), plus Begin and Inside tags for 6 entity types.
- **Technical Explanation**: 13 discrete tags: `O`, `B-PERP_REL`, `I-PERP_REL`, `B-LOCATION`, `I-LOCATION`, `B-TIME_FREQ`, `I-TIME_FREQ`, `B-PLATFORM`, `I-PLATFORM`, `B-EVIDENCE`, `I-EVIDENCE`, `B-LAW_SEC`, `I-LAW_SEC`.
- **ABHERA Example**: Defined in `models/ner/label_mapping.json`.

### Q100: What is the entity-level Micro-F1 of ABHERA's NER model?
- **Beginner Answer**: 99.24%, meaning it extracts names, locations, and platforms with near-perfect precision.
- **Technical Explanation**: Micro Precision = 100.0%, Micro Recall = 98.48%, Micro F1-Score = **99.24%** across 66 true entity spans in 60 test narratives (65 True Positives, 0 False Positives, 1 False Negative).
- **ABHERA Example**: Empirical results verified in `models/ner/test_metrics.json`.

### Q101: What was the recall for each individual entity type in the test set?
- **Beginner Answer**:
  - Perpetrator Relationship (`PERP_REL`): **100.0%** (24 / 24)
  - Time & Frequency (`TIME_FREQ`): **100.0%** (17 / 17)
  - Platform (`PLATFORM`): **100.0%** (12 / 12)
  - Location (`LOCATION`): **92.31%** (12 / 13)
- **Technical Explanation**: Verified in `models/ner/test_metrics.json`. The single false negative occurred on a subtle metaphorical location.
- **ABHERA Example**: 100% precision across all classes (0 false positive hallucinations).

### Q102: How does ABHERA map sub-word tokens back to original character spans?
- **Beginner Answer**: The tokenizer saves the exact character start and end numbers for each piece of a word, so we can slice the exact original text without typos.
- **Technical Explanation**: Fast Rust-based tokenizers provide `offset_mapping` tuples: `(start_char, end_char)` for every token. ABHERA locates the `start_char` of the `B-` token and `end_char` of the final `I-` token, slicing `original_narrative[start_char:end_char]`.
- **ABHERA Example**: Implemented in `ml/inference/predict_ner.py`.

### Q103: How are extracted entities used by the Dynamic Question Engine?
- **Beginner Answer**: If the AI already found that the perpetrator is the "husband" and the app is "WhatsApp", it marks those questions as answered and never asks them again.
- **Technical Explanation**: Extracted entity keys populate the `known_information` dictionary. The slot evaluator removes these fulfilled dimensions from the required set, preventing duplicate inquiries.
- **ABHERA Example**: In `backend/app/question_engine/question_selector.py`: `get_known_information()`.

### Q104: What is the difference between `B-PERP_REL` and `I-PERP_REL`?
- **Beginner Answer**: `B-PERP_REL` marks the first word (like "ex"), while `I-PERP_REL` marks the continuing words (like "boyfriend").
- **Technical Explanation**: `B-` indicates the boundary onset of a new entity span. `I-` indicates continuation tokens belonging to that same span, enabling unambiguous separation of adjacent entities.
- **ABHERA Example**: In `"ex-boyfriend"`, `"ex"` is `B-PERP_REL`, and `"boyfriend"` is `I-PERP_REL`.

### Q105: What happens if an entity span contains special characters or emojis?
- **Beginner Answer**: The character offsets slice the exact original text string, so emojis and punctuation are preserved perfectly.
- **Technical Explanation**: Because offset mapping slices from the raw Python Unicode string using Python's native character indexing, emojis and symbols are preserved without corruption.
- **ABHERA Example**: Slicing `"@hacker_user123!"` extracts the exact handle.

### Q106: Why was `AutoModelForTokenClassification` used instead of `AutoModelForSequenceClassification`?
- **Beginner Answer**: Sequence classification outputs one answer for the whole paragraph; Token classification outputs an answer for every single word in the paragraph.
- **Technical Explanation**: `AutoModelForTokenClassification` attaches a linear projection layer `(d_model -> num_labels)` to all hidden vectors $\mathbf{h}_1, \dots, \mathbf{h}_N$, whereas sequence classification projects only the `[CLS]` token $\mathbf{h}_{[CLS]}$.
- **ABHERA Example**: In `ml/ner/train_ner.py`.

### Q107: What is the maximum sequence length for the NER model?
- **Beginner Answer**: 256 tokens.
- **Technical Explanation**: Set to 256 tokens to ensure that long narrative testimonies containing detailed contextual descriptions are evaluated without truncating entity mentions that appear late in the narrative.
- **ABHERA Example**: In `models/ner/training_config.json`: `"max_length": 256`.

### Q108: What loss function is used for training the NER model?
- **Beginner Answer**: Cross-Entropy Loss, while ignoring special padding tokens.
- **Technical Explanation**: Multi-class Cross-Entropy Loss evaluated across all active token positions. Sub-word continuation tokens or padding tokens are masked with label `-100` (`ignore_index = -100`) so they do not contribute gradients to the loss.
- **ABHERA Example**: Standard PyTorch CrossEntropyLoss implementation in Hugging Face token classification.

---

## SECTION I: Project Architecture Questions (Q109 – Q120)

### Q109: Explain the end-to-end request-response flow in ABHERA.
- **Beginner Answer**: User types story on React webpage $\to$ request sent over internet $\to$ FastAPI validates data $\to$ BERT models classify crime and extract details $\to$ Question Engine picks next question $\to$ PostgreSQL fetches laws $\to$ response sent back to screen.
- **Technical Explanation**: React Axios client dispatches `POST /api/chat/message` $\to$ Uvicorn/FastAPI validates `ChatMessageRequest` via Pydantic $\to$ `ChatService` orchestrates in-memory `BERTIncidentPredictor` and `NEREntityPredictor` $\to$ Dynamic Question Engine state machine evaluates missing forensic slots $\to$ `AntiHallucinationService` asserts verbatim SQL queries against `laws` and `support_services` tables $\to$ state persisted to PostgreSQL $\to$ `ChatMessageResponse` serialized to JSON and rendered on client.
- **ABHERA Example**: Documented in the sequence diagram in Chapter 3.

### Q110: Why is ABHERA described as a "Neuro-Symbolic" architecture?
- **Beginner Answer**: "Neuro" means neural networks (BERT) that understand human speech. "Symbolic" means strict rules and databases (PostgreSQL) that guarantee truthful facts. Combining both gives you the best of both worlds.
- **Technical Explanation**: Neuro-symbolic AI combines connectionist deep learning (statistical pattern recognition via transformer weights) with symbolic AI (deterministic relational algebra, logic rules, and finite state machines), achieving high perceptual flexibility with zero factual hallucination.
- **ABHERA Example**: BERT understands that *"he won't stop following me"* means Stalking (neural); PostgreSQL supplies the exact verbatim text of BNS Section 78 (symbolic).

### Q111: What are the 6 canonical forensic slots in the Dynamic Question Engine?
- **Beginner Answer**:
  1. Safety Status (Is the survivor safe right now?)
  2. Person Involved (Who did it?)
  3. Location (Where did it happen?)
  4. Time & Frequency (When and how often?)
  5. Platform (What app or device was used?)
  6. Incident Details (Specific details of the act)
- **Technical Explanation**: Defined in `backend/app/question_engine/rules.py` as `ENTITY_PRIORITY_ORDER`: `SAFETY_STATUS`, `PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`, `INCIDENT_DETAILS`.
- **ABHERA Example**: Stored in `CATEGORY_REQUIRED_ENTITIES`.

### Q112: Why is `SAFETY_STATUS` the highest priority question?
- **Beginner Answer**: If a survivor is in immediate physical danger right now, saving their life is more important than asking what app was used.
- **Technical Explanation**: In trauma-informed safety triage, emergency threat assessment takes precedence over forensic evidence gathering. If immediate danger is present, the system immediately surfaces emergency hotline 112.
- **ABHERA Example**: `Q_SAFETY_01` has priority `100` (highest in the system).

### Q113: What is the Anti-Hallucination Layer and how is it implemented?
- **Beginner Answer**: A security gate that checks every single law and helpline before showing it to the user. If a law doesn't match an actual row in our database, it is thrown out.
- **Technical Explanation**: A programmatic verification service (`backend/app/services/anti_hallucination.py`) that executes parameterized SQLAlchemy queries against `laws` and `support_services`, asserting that returned citations originate strictly from database primary keys.
- **ABHERA Example**: `validate_legal_record(item, db)` asserts database row existence.

### Q114: How does the system handle an incident that matches multiple categories?
- **Beginner Answer**: It combines the requirements of all matching crimes into one master checklist, removing duplicates so the user isn't asked the same question twice.
- **Technical Explanation**: The `get_required_information()` function computes the mathematical union of required slots across all predicted categories ($\bigcup_{c \in \mathcal{C}} \text{Slots}_c$) and sorts them by canonical priority order.
- **ABHERA Example**: For `DV` + `CA`, required slots are unioned: `["SAFETY_STATUS", "PERP_REL", "PLATFORM", "TIME_FREQ", "INCIDENT_DETAILS"]`.

### Q115: What is the difference between client-side and server-side PDF generation?
- **Beginner Answer**: Server-side creates the PDF on the company's computer and sends the file over the internet. Client-side builds the PDF directly inside your own web browser, so the file never touches any remote computer.
- **Technical Explanation**: Server-side PDF generation (e.g., Puppeteer, WeasyPrint) executes headless browser instances on the backend, creating disk I/O overhead and storing unencrypted PII on server hard disks. Client-side `jsPDF` compiles vector lines in browser RAM, ensuring zero server-side data retention.
- **ABHERA Example**: Generated in `frontend/src/pages/Report.jsx`.

### Q116: How does the system resolve missing locations from questionnaire answers?
- **Beginner Answer**: If the user didn't mention the location in their first message, but answered "at my college" when the bot asked, the report builder automatically extracts "college" and puts it into the location box in the report.
- **Technical Explanation**: In `backend/app/services/report_generator.py`, the `_extract_location_or_platform_from_answers()` helper scans user questionnaire responses for location and platform keywords, injecting normalized values into Section 3.
- **ABHERA Example**: Injects `"Location: Workplace / Office"` into `extracted_information`.

### Q117: What are the 10 sections of the final ABHERA report?
- **Beginner Answer**: 1. Incident Summary, 2. Identified Types, 3. Extracted Details, 4. Questionnaire Answers, 5. Legal Information, 6. Suggested Steps, 7. Support Services, 8. Evidence Notes, 9. AI Model Information, 10. Legal Disclaimer.
- **Technical Explanation**: Assembled in `ReportGeneratorService`: `incident_summary`, `identified_incident_type`, `extracted_information`, `user_provided_details`, `relevant_legal_information`, `suggested_next_steps`, `available_support_services`, `evidence_information_notes`, `model_prediction_information`, `disclaimer`.
- **ABHERA Example**: Displayed on `/report` and rendered into the PDF.

### Q118: How does the frontend track active sessions across browser refreshes?
- **Beginner Answer**: By saving the session ID in `sessionStorage`. If you hit refresh, the browser reads the ID back and reloads your chat history from the database.
- **Technical Explanation**: The `ConversationContext` initializes state from `sessionStorage.getItem('abhera_session_id')`. If present, it calls `GET /api/session/{session_id}` to restore message history, predicted labels, and current questions.
- **ABHERA Example**: In `frontend/src/context/ConversationContext.jsx`.

### Q119: What is the role of `data_importer.py` in system setup?
- **Beginner Answer**: It reads the CSV files with 32 laws and 12 helplines and inserts them safely into PostgreSQL when setting up the database.
- **Technical Explanation**: An automated ETL script that loads `data/laws.csv` and `data/support_services.csv`, validates columns, and populates PostgreSQL tables.
- **ABHERA Example**: `python -m app.services.data_importer`.

### Q120: How are CORS preflight requests handled?
- **Beginner Answer**: Before sending real data, the browser sends a quick test message (`OPTIONS`) asking "Are you allowed to talk to me?" FastAPI answers "Yes, you are on the approved list."
- **Technical Explanation**: The browser sends an HTTP `OPTIONS` request with `Origin` and `Access-Control-Request-Method` headers. FastAPI's `CORSMiddleware` intercepts the request and responds with `Access-Control-Allow-Origin: http://localhost:5173` and `Access-Control-Allow-Methods: *`.
- **ABHERA Example**: Configured in `backend/app/main.py`.

---

## SECTION J: Testing Questions (Q121 – Q132)

### Q121: How many automated unit tests exist in ABHERA?
- **Beginner Answer**: Exactly 154 unit tests across 11 test files, and every single test passes.
- **Technical Explanation**: Exactly 154 automated unit and integration tests distributed across 11 test modules in `backend/tests/`, achieving 100% pass rate.
- **ABHERA Example**: Verified via: `python -m unittest discover -s backend/tests`.

### Q122: What is the `TestClient` in FastAPI?
- **Beginner Answer**: A testing tool that lets you send fake web requests to your backend code inside Python without actually launching a real web server.
- **Technical Explanation**: A test harness built on HTTPX that simulates HTTP requests against a FastAPI ASGI application instance directly in memory, validating status codes, response headers, and JSON payloads.
- **ABHERA Example**: In `backend/tests/test_chat_api.py`: `self.client = TestClient(app)`.

### Q123: What does `test_hallucination.py` verify?
- **Beginner Answer**: It tries to trick the system with fake laws to prove that the system catches them and blocks them 100% of the time.
- **Technical Explanation**: Asserts that `AntiHallucinationService` rejects non-existent legal sections, empty payloads, and unauthorized modifications, ensuring zero hallucinations.
- **ABHERA Example**: Verifies that injecting `"IPC Section 999"` returns `False`.

### Q124: What is regression testing and why is it important?
- **Beginner Answer**: Running all your old tests every time you make a change to make sure your new code didn't break something that was already working.
- **Technical Explanation**: Re-running existing functional and non-functional tests to ensure that previously developed and tested software continues to perform as expected after changes, patches, or optimizations.
- **ABHERA Example**: When threshold tuning was introduced in BERT v2, the entire 154-test suite was re-run to confirm zero regressions in API endpoints.

### Q125: What is mocked testing and where is it used?
- **Beginner Answer**: Using fake pretend data (like a fake database or dummy model) so you can test one small piece of code quickly without needing the whole system running.
- **Technical Explanation**: Replacing external dependencies (such as network calls or database sockets) with mock objects that return pre-programmed responses (`unittest.mock`), allowing deterministic testing of edge cases.
- **ABHERA Example**: Used in testing question selector rules without querying PostgreSQL.

### Q126: How do you test that a 400 Bad Request is returned for empty messages?
- **Beginner Answer**: You send a message that has only spaces, and check that the server responds with error code 400.
- **Technical Explanation**: In `test_chat_api.py`:
  ```python
  response = self.client.post("/api/chat/message", json={"session_id": "SESS-1", "message": "   "})
  self.assertEqual(response.status_code, 400)
  ```
- **ABHERA Example**: Asserts validation logic in `backend/app/routes/chat.py`.

### Q127: How is the database cleaned up between tests?
- **Beginner Answer**: Each test deletes the temporary test sessions it created so the database stays clean for the next test.
- **Technical Explanation**: Test classes implement `tearDownClass` and `tearDown` methods that record created test session IDs and execute `DELETE FROM sessions WHERE session_id IN (...)`, relying on foreign key cascades to clean child rows.
- **ABHERA Example**: In `backend/tests/test_chat_api.py`: `cls.db.query(UserSession).filter(...).delete()`.

### Q128: How do you test that the question engine doesn't loop?
- **Beginner Answer**: You keep sending answers and verify that the system never asks the same question twice and eventually says "Completed".
- **Technical Explanation**: An automated simulation script sends 10 consecutive conversational turns and asserts that the set of returned question IDs has zero duplicate elements and reaches `status: COMPLETED` within $N \le 6$ turns.
- **ABHERA Example**: Tested in `backend/tests/test_question_engine.py`.

### Q129: What is the test coverage of the ABHERA codebase?
- **Beginner Answer**: All 11 core backend modules are covered by automated tests.
- **Technical Explanation**: Comprehensive coverage across the routes, services, schemas, question engine, models, and ML inference wrappers.
- **ABHERA Example**: 154 unit tests distributed across all 11 backend test files.

### Q130: How do you test multi-label classification accuracy?
- **Beginner Answer**: You compare the AI's predictions on 164 hidden test stories against the true labels agreed upon by human legal experts.
- **Technical Explanation**: Evaluated using scikit-learn metrics (`f1_score(y_true, y_pred, average='micro')`, `accuracy_score`) across the 164-sample test split.
- **ABHERA Example**: Documented in `ml/evaluation/FINAL_EVALUATION_RESULTS.json`.

### Q131: What is Elementwise Accuracy in multi-label classification?
- **Beginner Answer**: Checking each of the 6 category boxes individually. Out of 984 total boxes checked, how many were right?
- **Technical Explanation**: The total proportion of correct binary decisions across all classes and samples:
  $$\text{Elementwise Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{180 + 797}{984} = 99.29\%$$
- **ABHERA Example**: ABHERA achieved **99.29% Elementwise Accuracy**.

### Q132: What happens if PostgreSQL is not running when tests are executed?
- **Beginner Answer**: Tests that need to talk to the database will fail with a connection error.
- **Technical Explanation**: Integration tests that instantiate `SessionLocal()` will raise `psycopg2.OperationalError: could not connect to server: Connection refused`. Tests that do not touch the database (like rule parsing) still pass.
- **ABHERA Example**: Verified when testing offline workstation scenarios.

---

## SECTION K: Deployment Questions (Q133 – Q144)

### Q133: Explain the overall deployment architecture of ABHERA.
- **Beginner Answer**: The frontend runs in the cloud on Vercel so anyone can open the website. The heavy AI models and database run on our workstation computer, connected to Vercel through a secure Cloudflare Tunnel.
- **Technical Explanation**: A hybrid edge deployment: React static assets are compiled and served globally by Vercel CDN; the FastAPI backend, PyTorch models, and PostgreSQL DB run locally, exposed via an outbound encrypted Cloudflare Quick Tunnel (`cloudflared`).
- **ABHERA Example**: Detailed in Chapter 13.

### Q134: Why can't the entire backend and model run on Vercel or free cloud hosts?
- **Beginner Answer**: Vercel free functions and free cloud containers only give you 512 MB of memory. Our two BERT AI models require over 1.35 GB of memory, so free cloud servers crash instantly with an "Out of Memory" error.
- **Technical Explanation**: Free PaaS tiers enforce strict 512MB RAM limits. Hosting two 110M-parameter PyTorch BERT models plus Python runtime requires $\approx 1.35 \text{ GB}$ of host RAM. Loading the models on 512MB containers triggers an immediate Linux kernel `OOMKilled` signal.
- **ABHERA Example**: Model weights alone occupy 833.1 MB on disk (`.safetensors`).

### Q135: What is Cloudflare Tunnel and how does it work?
- **Beginner Answer**: A secure software tunnel that lets people on the internet visit a server running on your home computer without having to open dangerous ports in your home router.
- **Technical Explanation**: An outbound-only reverse proxy tunnel established by the `cloudflared` daemon over TLS to Cloudflare's nearest edge point of presence. Inbound user traffic hitting the Cloudflare URL is securely proxied over the tunnel to `localhost:8000`.
- **ABHERA Example**: Launched via: `cloudflared tunnel --url http://localhost:8000`.

### Q136: What command starts the FastAPI backend on Windows?
- **Beginner Answer**:
  ```powershell
  $env:PYTHONPATH="D:\Abhera(Mini)\backend;D:\Abhera(Mini)"
  & "D:\Abhera(Mini)\backend\venv\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Technical Explanation**: Sets the Python import path environment variable and launches the Uvicorn ASGI server bound to all local interfaces on TCP port 8000 with hot-reload enabled.
- **ABHERA Example**: Standard development and viva demonstration startup command.

### Q137: What is the purpose of `render.yaml`?
- **Beginner Answer**: A configuration blueprint that tells a paid cloud platform (like Render) how to automatically build and launch the backend and database.
- **Technical Explanation**: An Infrastructure-as-Code (IaC) manifest defining the build environment, Python version, start commands, managed PostgreSQL database, and environment variable bindings.
- **ABHERA Example**: Located at `D:\Abhera(Mini)\render.yaml`.

### Q138: What environment variables are required by the backend?
- **Beginner Answer**: Database user, password, host, port, database name, and allowed frontend website addresses.
- **Technical Explanation**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `DATABASE_URL`, and `CORS_ORIGINS`.
- **ABHERA Example**: Defined in `backend/.env.example`.

### Q139: What environment variable is required by the React frontend?
- **Beginner Answer**: `VITE_API_URL`, which tells the frontend the web address of the backend server.
- **Technical Explanation**: `VITE_API_URL` specifies the base URL prepended to all Axios HTTP requests. In production, it points to the Cloudflare Tunnel URL or cloud domain.
- **ABHERA Example**: In `frontend/.env.production`: `VITE_API_URL=https://my-tunnel.trycloudflare.com`.

### Q140: How do you build the React application for production?
- **Beginner Answer**: You run `npm run build`, which bundles and shrinks all your JavaScript and CSS files into a fast, tiny `dist/` folder.
- **Technical Explanation**: Vite invokes Rollup to perform tree-shaking, dead-code elimination, and asset minification, producing static HTML, JS, and CSS files in `frontend/dist/`.
- **ABHERA Example**: Executed via: `npm run build` inside `frontend/`.

### Q141: What happens if your local computer goes to sleep during a live demo?
- **Beginner Answer**: The tunnel closes, the backend stops answering, and the website on the examiner's phone will show a connection error.
- **Technical Explanation**: Sleep mode suspends CPU execution, severing the TCP socket between `cloudflared` and Cloudflare edge servers. The Cloudflare edge will return an HTTP `502 Bad Gateway` error to Vercel clients.
- **ABHERA Example**: Windows sleep mode must be disabled during viva presentations.

### Q142: What is the purpose of `.dockerignore` and `.gitignore`?
- **Beginner Answer**: They tell Git and Docker which files to ignore so you don't accidentally upload passwords, giant virtual environment folders, or database files to the public internet.
- **Technical Explanation**: Exclusion manifests that prevent sensitive files (`.env`), heavy binary caches (`venv/`, `node_modules/`), and local build artifacts from being committed to Git or copied into Docker build contexts.
- **ABHERA Example**: Located at `.gitignore` and `.dockerignore`.

### Q143: How do you start PostgreSQL service in Windows PowerShell?
- **Beginner Answer**: Run `net start postgresql-x64-16`.
- **Technical Explanation**: Starts the Windows background service for PostgreSQL 16 via the Service Control Manager.
- **ABHERA Example**: `net start postgresql-x64-16`.

### Q144: How does Vercel know how to deploy the React project?
- **Beginner Answer**: Vercel automatically detects the `frontend/` folder and `package.json`, runs `npm run build`, and puts the finished files on its worldwide edge servers.
- **Technical Explanation**: Vercel inspects the repository root, detects the Vite framework preset, installs Node dependencies (`npm install`), executes the build script (`npm run build`), and maps the `frontend/dist` directory to its edge routing engine.
- **ABHERA Example**: Configured in Vercel project settings with root directory `frontend`.

---

## SECTION L: Limitations and Future Scope (Q145 – Q154)

### Q145: What are the current limitations of the ABHERA system?
- **Beginner Answer**: It only understands English, takes about 300 milliseconds on CPU to run, requires the local computer to stay awake in the free setup, and provides educational legal help rather than acting as a courtroom lawyer.
- **Technical Explanation**:
  1. Language constraint (English only; no native Indic vernacular).
  2. CPU inference latency (~250-450ms per forward pass).
  3. Hybrid tunnel dependency in zero-cost deployment.
  4. Informational triage scope (no direct police CCTNS database integration).
- **ABHERA Example**: Acknowledged in Chapter 1.

### Q146: How will you add multilingual Indian language support in the future?
- **Beginner Answer**: By replacing the English BERT model with multilingual models like IndicBERT or mBERT, which were trained on Hindi, Telugu, Tamil, Bengali, and other Indian languages.
- **Technical Explanation**: Transition the backbone from `bert-base-uncased` to `IndicBERT` (AI4Bharat) or `XLM-RoBERTa`, which have vocabularies trained across 22 scheduled Indian languages, and fine-tune on translated and native vernacular harassment corpora.
- **ABHERA Example**: Planned Phase 17 roadmap.

### Q147: How can speech-to-text (voice intake) be integrated?
- **Beginner Answer**: By adding a microphone button where a survivor can just speak their story, and an AI speech tool (like OpenAI Whisper) turns their voice into text for ABHERA to analyze.
- **Technical Explanation**: Integrating an open-source acoustic model like `Whisper-v3` or `Bhashini Speech API` in the React frontend via the Web Audio API, streaming voice audio to the backend for real-time transcription before feeding into BERT.
- **ABHERA Example**: Beneficial for illiterate survivors or users in severe distress who cannot type.

### Q148: Can ABHERA automatically file an FIR with the police?
- **Beginner Answer**: Not right now. It prepares a complete report for the user to hand to the police. Filing an actual FIR automatically requires special government permission and security credentials.
- **Technical Explanation**: Automated FIR lodging requires certified API integration with state Crime and Criminal Tracking Network & Systems (CCTNS) databases and National Cyber Crime Reporting Portals, which are restricted to authorized law enforcement agencies.
- **ABHERA Example**: ABHERA acts as the pre-FIR evidentiary preparation dossier.

### Q149: How can image OCR be added to extract evidence from WhatsApp screenshots?
- **Beginner Answer**: By letting the user upload screenshots of threatening messages, using an OCR tool (like Tesseract) to read the text in the screenshot, and feeding that text into ABHERA.
- **Technical Explanation**: Adding an image upload pipeline that processes screenshots through an Optical Character Recognition (OCR) model (e.g., Tesseract or Donut), extracting sender handles, timestamps, and message text, and feeding them directly into the NER entity extractor.
- **ABHERA Example**: Fulfills the `EVIDENCE` and `PLATFORM` slots automatically.

### Q150: What is the hardware cost of running ABHERA in permanent cloud hosting?
- **Beginner Answer**: About \$15 to \$25 per month for a standard cloud server with 2 GB of RAM and a managed PostgreSQL database.
- **Technical Explanation**: A standard 2.0 GB RAM Linux container on Render or AWS EC2 costs $\approx \$15/\text{month}$, plus $\approx \$7/\text{month}$ for managed PostgreSQL storage, totaling $\approx \$22/\text{month}$.
- **ABHERA Example**: Sized in Chapter 13.

### Q151: Can ABHERA be deployed on edge devices (like smartphones directly)?
- **Beginner Answer**: Yes, if we shrink the BERT model using techniques like quantization or ONNX so it can run inside a mobile app without needing any internet connection.
- **Technical Explanation**: By applying Post-Training Quantization (INT8) or knowledge distillation (`DistilBERT` or `MobileBERT`) and converting the model to ONNX / TensorFlow Lite, the model footprint can be reduced from 417MB to ~60MB, enabling on-device inference without internet access.
- **ABHERA Example**: High potential for remote rural safety deployments.

### Q152: How can legal provisions be kept up to date when laws change?
- **Beginner Answer**: Because laws are stored in database tables instead of being baked into the AI code, a legal expert can simply edit the rows in PostgreSQL without retraining any AI models.
- **Technical Explanation**: Decoupled relational architecture: statutory amendments or new judicial precedents are updated via simple SQL `INSERT` or `UPDATE` queries on the `laws` table, requiring zero model retraining or application downtime.
- **ABHERA Example**: Adding a new BNS amendment takes one SQL query.

### Q153: How can ABHERA connect survivors to real-time human advocates?
- **Beginner Answer**: By adding a secure chat button that connects the survivor directly to an accredited NGO social worker or legal aid volunteer who can review the generated report with them.
- **Technical Explanation**: Adding a WebRTC or WebSocket-based escalation gateway that transfers the verified session state to an authenticated dashboard for certified Legal Aid Counsel (DLSA) or licensed counselors.
- **ABHERA Example**: Planned institutional portal roadmap.

### Q154: What is the long-term vision for ABHERA?
- **Beginner Answer**: To become India's most trusted, accessible, zero-cost digital safety companion for women, ensuring no survivor is ever denied justice due to fear, confusion, or legal illiteracy.
- **Technical Explanation**: A nationwide, open-access, neuro-symbolic legal triage infrastructure deployed across universities, corporate grievance systems, and public safety kiosks, bridging the gap between trauma and justice.
- **ABHERA Example**: Academic and research reference architecture.

---

## SECTION M: Difficult External Examiner & Defense Questions (Q155 – Q170)

### Q155: Why did you not use a modern LLM like LLaMA-3 or GPT-4 instead of BERT?
- **Beginner Answer**: Big LLMs frequently hallucinate and make up fake legal sections, which is dangerous in a safety project. Also, LLMs are huge, expensive, and leak private information to cloud servers. BERT is small, fast, completely predictable, and never hallucinates laws because our database provides them.
- **Technical Explanation**:
  1. **Hallucination Risk**: Autoregressive LLMs cannot guarantee 100% statutory factuality; hallucinating a penal code number invalidates an FIR.
  2. **Privacy**: Commercial LLM APIs transmit sensitive survivor trauma disclosures to third-party servers, violating confidentiality.
  3. **Computational Efficiency**: BERT Base has 110M parameters and runs in 250ms on CPU; LLaMA-3-8B has 8 billion parameters and requires dedicated 16GB+ VRAM GPUs.
  4. **Task Suitability**: Discriminative encoder models (BERT) are mathematically superior to generative decoders for sequence classification and token sequence labeling.
- **ABHERA Example**: In ABHERA, BERT is restricted strictly to perceptual understanding, while laws are retrieved with 100% provenance from PostgreSQL.

### Q156: Isn't TF-IDF + Logistic Regression almost as good as BERT on your benchmarks? Why add the complexity of BERT?
- **Beginner Answer**: TF-IDF only matches keywords. If a user uses synonyms, metaphors, or complex sentences, TF-IDF fails completely. BERT understands context, word order, and subtle threats that don't use obvious keywords. Furthermore, TF-IDF cannot perform token-level entity extraction (NER).
- **Technical Explanation**: While TF-IDF + Logistic Regression scored 97.79% Micro-F1 on in-domain keyword-rich benchmarks, it suffers from severe semantic brittleness: it cannot model polysemy, syntactic negation, or out-of-vocabulary expressions. Furthermore, TF-IDF cannot perform token classification for Named Entity Recognition; adopting BERT creates a unified, shared transformer representations architecture for both classification and NER.
- **ABHERA Example**: In sentences with indirect stalking (*"He waits near where I walk every evening"*), TF-IDF fails because neither the word "stalking" nor "harassment" appears. BERT captures the contextual threat.

### Q157: How do you mathematically justify your positive class weights (`pos_weight`)?
- **Beginner Answer**: We counted how many times each crime was missing in the dataset compared to how many times it was present, and set the penalty weight equal to that exact ratio.
- **Technical Explanation**: We compute the inverse class frequency ratio:
  $$w_c = \frac{N_{total} - N_{positive, c}}{N_{positive, c}}$$
  For `OV` (98 positive, 988 negative), $w_{OV} = \frac{1086 - 98}{98} = 10.0816$. This mathematically balances the gradient contribution of positive and negative examples during backpropagation, setting expected gradient magnitudes equal.
- **ABHERA Example**: Recorded in `models/bert_multilabel_experiment_v2/training_config.json`.

### Q158: Explain the single False Negative in your test set. Why did the model fail?
- **Beginner Answer**: In test case 50, a man repeatedly messaged a woman asking to meet after a conference. The model caught the Stalking part, but missed the Sexual Harassment part by just 4% because he didn't use any explicit vulgar words.
- **Technical Explanation**: In `TEST_0050`, the perpetrator engaged in persistent post-conference messaging. The model correctly flagged Stalking ($p = 0.84 \ge 0.79$). However, the Sexual Harassment logit evaluated to probability $p = 0.4549$, falling just short of the 0.50 threshold by 0.0451 due to the absence of overtly explicit or sexual lexicon.
- **ABHERA Example**: Documented in `ml/evaluation/FINAL_EVALUATION_RESULTS.json`.

### Q159: What is the computational complexity of the Self-Attention mechanism and how does it scale?
- **Beginner Answer**: It scales with the square of the sentence length ($O(N^2)$), because every word has to compare itself to every other word in the sentence.
- **Technical Explanation**: Computing $\mathbf{Q} \mathbf{K}^T$ requires multiplying an $(N \times d)$ matrix by a $(d \times N)$ matrix, yielding an $(N \times N)$ attention matrix with computational and memory complexity of $\mathcal{O}(N^2 \cdot d)$, where $N$ is sequence length and $d$ is hidden dimension.
- **ABHERA Example**: By constraining `max_length = 128` tokens for classification, memory consumption is capped at only $128^2 = 16,384$ attention calculations per head, running easily in CPU RAM.

### Q160: How do you prevent SQL Injection attacks in ABHERA?
- **Beginner Answer**: We never glue user text directly into SQL commands. We use SQLAlchemy, which automatically sanitizes all inputs using parameterized queries.
- **Technical Explanation**: All database queries are executed using SQLAlchemy's ORM or parameterized prepared statements (`bindparam`), where user inputs are transmitted as separate data parameters across the database wire protocol rather than concatenated into the SQL statement string.
- **ABHERA Example**: `db.query(Law).filter(Law.applicable_label == normalized_label)` parameterizes the query automatically.

### Q161: Why did you choose a 70/15/15 train/validation/test split instead of K-Fold Cross-Validation?
- **Beginner Answer**: Because deep learning models take hours to train. 70/15/15 gives plenty of training data while keeping an isolated validation set for threshold tuning and a pristine test set for final grading.
- **Technical Explanation**: For deep transformer models with 110 million parameters, running 5-Fold or 10-Fold cross-validation increases training compute by 5x to 10x without meaningful variance reduction. A stratified 70/15/15 split ($N_{train}=760, N_{val}=162, N_{test}=164$) ensures adequate support for minority classes while preserving strict test isolation.
- **ABHERA Example**: Fixed `random_state = 42` guarantees exact reproducibility.

### Q162: What is the difference between Token Classification and Sequence Classification?
- **Beginner Answer**: Sequence classification looks at the whole paragraph and gives one score. Token classification looks at every single word and gives a score to each word.
- **Technical Explanation**:
  - **Sequence Classification**: Produces a single tensor $\mathbf{y} \in \mathbb{R}^C$ derived from the pooled `[CLS]` token embedding.
  - **Token Classification**: Produces a 3D tensor $\mathbf{Y} \in \mathbb{R}^{B \times N \times T}$, assigning a probability distribution across $T$ tags to every token position $i \in \{1, \dots, N\}$.
- **ABHERA Example**: ABHERA uses Sequence Classification for harassment categories and Token Classification for BIO entity tags.

### Q163: How do you handle cases where the survivor is in immediate life-threatening danger?
- **Beginner Answer**: The very first question the system evaluates is whether the user is safe right now. If they are in danger, it immediately highlights emergency number 112 in bold red.
- **Technical Explanation**: The Dynamic Question Engine assigns priority `100` (highest) to `Q_SAFETY_01` (`SAFETY_STATUS`). If the survivor responds indicating active physical danger, the system surfaces immediate national emergency hotline 112 prominently before continuing any forensic questionnaire.
- **ABHERA Example**: In `seed_questions.py`: `"priority": 100, "required_entity": "SAFETY_STATUS"`.

### Q164: Could an adversary fool ABHERA by typing random gibberish or song lyrics?
- **Beginner Answer**: If someone types gibberish, none of the crime probabilities will reach the threshold, so the system will politely say that no incident could be determined.
- **Technical Explanation**: Gibberish or irrelevant text produces low, diffuse attention weights across the vocabulary, yielding negative logits and near-zero sigmoid probabilities ($p_c < 0.10$). Because all probabilities fall far below optimal thresholds, the system defaults to the safe fallback: *"Incident type could not be determined from the available model output."*
- **ABHERA Example**: Verified in `backend/tests/test_fpm_incident.py`.

### Q165: What legal liability does ABHERA bear if a user acts on its report?
- **Beginner Answer**: None, because every page and report clearly states that ABHERA is an educational and informational tool, not a licensed lawyer or emergency replacement.
- **Technical Explanation**: Legally governed by Section 10 of the dossier and terms of service: ABHERA operates strictly under the legal categorization of an informational and triage decision-support tool. It includes explicit statutory disclaimers stating that the system does not provide legal representation or formal legal advice.
- **ABHERA Example**: Section 10: *"This system provides informational assistance... It does not replace professional legal advice, emergency services, law enforcement, or other qualified support."*

### Q166: Why did you not use a Graph Database (like Neo4j) for legal mapping?
- **Beginner Answer**: Graph databases are great for complex networks of people, but our legal mapping is a simple 1-to-many relationship: one crime label maps to several law sections. PostgreSQL does this much faster and with less complexity.
- **Technical Explanation**: While statutory frameworks can be modeled as ontologies, ABHERA's operational query pattern is an indexed 1-to-N associative lookup (`applicable_label -> List[Law]`). A relational B-tree index in PostgreSQL executes this query in under 2ms with zero overhead, whereas a graph database would introduce unnecessary architectural complexity and RAM overhead.
- **ABHERA Example**: Fast index on `laws.applicable_label`.

### Q167: How do you prove that your model didn't just memorize the training dataset?
- **Beginner Answer**: We evaluated the model on 164 brand new test stories that were completely locked away and never seen by the model during training. It scored 98% on those new stories.
- **Technical Explanation**: By strictly enforcing a held-out test partition (`test_data.csv`, 164 records) that was neither utilized during AdamW gradient descent nor during threshold grid search. High exact match accuracy (95.73%) on unseen test data proves robust out-of-sample generalization.
- **ABHERA Example**: Verified in `ml/evaluation/FINAL_EVALUATION_RESULTS.json`.

### Q168: How does the system handle conflicting information between the initial narrative and follow-up answers?
- **Beginner Answer**: Newer answers take precedence over older statements, and both are clearly displayed in the final report so nothing is hidden.
- **Technical Explanation**: In `backend/app/services/chat_service.py`, user answers to follow-up questions are appended to the narrative context and stored in `submission_question_answers`. The `ReportGeneratorService` displays both the raw original narrative (Section 1) and explicit Q&A turns (Section 4), ensuring complete forensic transparency.
- **ABHERA Example**: Both original text and questionnaire answers appear in the report.

### Q169: What is the difference between Micro Precision and Macro Precision in ABHERA?
- **Beginner Answer**: Micro Precision was 96.77% across all 180 detections combined. Macro Precision was 95.71%, calculated by averaging the precision scores of each of the 6 categories individually.
- **Technical Explanation**:
  - $\text{Micro Precision} = \frac{\sum TP_c}{\sum TP_c + \sum FP_c} = \frac{180}{180 + 6} = 0.9677$
  - $\text{Macro Precision} = \frac{1}{6} \sum_{c=1}^6 \text{Precision}_c = 0.9571$
- **ABHERA Example**: Both exceed 95%, demonstrating balanced precision across all crime types.

### Q170: If you had one more month to work on this project, what single improvement would you prioritize?
- **Beginner Answer**: I would add voice input in Hindi and regional Indian languages so that a woman in severe distress can simply speak her story instead of having to type in English.
- **Technical Explanation**: I would implement an on-device Speech-to-Text pipeline using an open-source multilingual acoustic model (`Whisper-v3` / `IndicWav2Vec`), feeding transcribed vernacular tokens through an `IndicBERT` multi-label sequence classification head, dramatically expanding accessibility for underprivileged survivors across rural India.
- **ABHERA Example**: Direct evolution from Phase 16 to Phase 17 of the project roadmap.
'''
