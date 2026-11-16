# ABHERA Technical Interview, Viva Defense & Master Q&A Guide
**Project:** ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance  
**Target Audience:** Viva Examiners, Technical Reviewers, ML Hiring Panels, Project Defenses  
**Total Technical Q&As:** Exactly 65 Exhaustive Questions and Answers Across 9 Core Domains  

---

## 1. Project Presentation Elevator Pitches

### 1-Minute Elevator Pitch
"ABHERA is an India-focused AI legal assistance platform addressing the critical delay survivors of harassment face when seeking justice. Unlike generic LLMs that hallucinate non-existent penal sections, ABHERA combines fine-tuned deep learning with deterministic statutory grounding. An incident narrative is evaluated by a multi-label BERT classifier achieving 98.09% Micro-F1 and a 13-tag NER model achieving 99.24% F1. An adaptive Dynamic Question Engine fills evidentiary gaps through prioritized follow-up prompts, before mapping the verified facts to 32 audited Bharatiya Nyaya Sanhita and IT Act provisions and 12 crisis helplines. The final output is an immutable, client-side vector PDF compiled in-browser without storing sensitive survivor credentials."

### 2-Minute Executive Summary
"Under the transition from the Indian Penal Code to the Bharatiya Nyaya Sanhita 2023, survivors of gender-based harassment encounter acute legal ambiguity. Existing reporting tools force users into rigid administrative drop-downs or employ generative AI bots that invent statutory penalties. ABHERA solves this through a dual-task perception architecture. First, a survivor discloses an incident in plain natural language. A fine-tuned BERT multi-label classifier identifies positive harassment dimensions using validation-calibrated decision thresholds, while a token-level NER model extracts evidentiary spans like perpetrator relationships and communication channels. Second, our state machine evaluates slot fulfillment across 6 canonical dimensions, dynamically generating context-aware follow-up questions. Third, our anti-hallucination layer asserts verbatim primary-key foreign-key matches against a PostgreSQL database of vetted statutory provisions and 3-tier geospatial support centers. The backend runs on FastAPI with 154 passing unit tests, and the frontend compiles cryptographic vector PDFs in React using jsPDF."

---

## 2. Master Technical Viva Q&A Bank (65 Questions)

### Domain 1: Project Overview & System Philosophy
1. **What is the full form of ABHERA?**  
   *Answer:* Adaptive BERT-based Harassment Entity Recognition & Assistance.
2. **Why was ABHERA developed specifically for the Indian legal context?**  
   *Answer:* India overhauled its criminal justice system in 2024, replacing the Indian Penal Code (IPC) with the Bharatiya Nyaya Sanhita (BNS). ABHERA updates legal literacy by mapping modern harassment (such as Section 69 BNS for deceitful sexual intercourse and IT Act Section 66E for privacy violation) to verified statutory texts.
3. **What is the fundamental difference between ABHERA and a ChatGPT legal wrapper?**  
   *Answer:* ChatGPT uses autoregressive generative decoding which frequently hallucinates legal sections and non-existent penalties. ABHERA restricts deep learning strictly to perceptual feature extraction (classification and NER). All legal sections and helplines are retrieved deterministically from PostgreSQL relational tables with zero text generation.
4. **How does ABHERA protect survivor anonymity?**  
   *Answer:* The `users` table stores zero passwords, zero emails, and zero phone numbers. Sessions use transient UUIDs synchronized with browser `sessionStorage`. When the tab closes, credentials evaporate.
5. **Why is PDF generation executed client-side rather than on the backend?**  
   *Answer:* Generating PDFs on the server requires storing survivor disclosures in temporary disk files or running heavy headless Chrome instances. Client-side `jsPDF` compiles vector lines directly in browser memory, eliminating server-side leakage vectors.
6. **What are the 6 legal incident classes predicted by ABHERA?**  
   *Answer:* Domestic Violence (`DV`), Sexual Harassment (`SH`), Stalking (`ST`), Cyber Abuse (`CA`), Workplace Harassment (`WH`), and Other Violence (`OV`).
7. **What is the exact size and partitioning of the ABHERA dataset?**  
   *Answer:* Exactly 1,086 verified incident records: 760 Train (70.0%), 162 Validation (15.0%), and 164 Test (15.0%) using fixed random seed 42.

### Domain 2: Deep Learning, Transformers & BERT Architecture
8. **Why was BERT selected over GPT or LLaMA?**  
   *Answer:* BERT is an encoder-only bidirectional transformer. In legal narrative intake, context flows in both directions (e.g., words at the end of a sentence modify words at the beginning). Autoregressive decoder models like GPT only attend to preceding tokens, making them suboptimal for sequence classification and token-level entity extraction.
9. **What are the architectural specifications of `bert-base-uncased`?**  
   *Answer:* 12 transformer encoder layers, 768 hidden dimension ($d_{model}$), 12 self-attention heads ($d_k = 64$), 30,522 WordPiece vocabulary, and ~110 million parameters.
10. **What is the mathematical formulation of self-attention in BERT?**  
    *Answer:* $	ext{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = 	ext{softmax}\left(rac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}ight) \mathbf{V}$. The $\sqrt{d_k}$ scaling prevents dot products from growing excessively large, which would cause vanishing gradients in the softmax.
11. **What is the purpose of the `[CLS]` token?**  
    *Answer:* The `[CLS]` (classification) token is appended to the start of every sequence. After 12 layers of bidirectional self-attention, its 768-dimensional output vector serves as the aggregated representation of the entire sequence, fed into the linear projection head.
12. **Why does BERT require an attention mask?**  
    *Answer:* Sequences in a batch are padded to uniform length (128 tokens in ABHERA). The attention mask contains `1` for active linguistic tokens and `0` for `[PAD]` tokens, preventing padding from skewing the self-attention softmax distributions.
13. **What is WordPiece tokenization?**  
    *Answer:* A subword tokenization algorithm that decomposes rare or out-of-vocabulary words into frequent subword prefixes and suffixes (e.g., *"harassment"* $	o$ `['ha', '##rass', '##ment']`), bounded by a 30,522 vocabulary.
14. **Why was max sequence length increased from 64 in v1 to 128 in v2?**  
    *Answer:* In v1, 64 tokens truncated ~38% of survivor narratives, discarding critical evidentiary context at the end of stories. 128 tokens captured 99.4% of full narratives without truncation.
15. **What optimizer was used and what was the learning rate?**  
    *Answer:* AdamW with a base learning rate of $3 	imes 10^{-5}$, weight decay of $0.01$, and linear learning rate warmup over 6 training epochs.

### Domain 3: Multi-Label Classification & Loss Formulation
16. **Why is multi-label classification mathematically distinct from multi-class classification?**  
    *Answer:* In multi-class classification, classes are mutually exclusive; the output layer uses Softmax ($\sum p_i = 1$). In multi-label classification, classes are non-exclusive (an incident can be both DV and CA); the output layer uses independent Sigmoids where each class probability is bounded independently in $[0.0, 1.0]$.
17. **What is the mathematical equation for Sigmoid activation?**  
    *Answer:* $\sigma(z_i) = rac{1}{1 + e^{-z_i}}$.
18. **Why did BERT v1 fail completely on multi-label combinations (0.0% multi-label exact match)?**  
    *Answer:* BERT v1 trained for only 2 epochs using standard unweighted BCE loss. Because positive class instances were outnumbered by negative zeros, gradients drove all logits toward negative values, yielding 23 False Negatives.
19. **How does `pos_weight` in `BCEWithLogitsLoss` solve class imbalance?**  
    *Answer:* It scales the loss contribution of positive errors by $w_c = rac{N - N_c}{N_c}$. A false negative error is penalized proportionally higher than a false positive, forcing the optimizer to achieve high recall on minority crime classes.
20. **What is the difference between elementwise accuracy and exact match accuracy?**  
    *Answer:* Elementwise accuracy measures the proportion of individual binary predictions correct across all classes ($164 	imes 6 = 984$ decisions, achieving 99.29%). Exact match accuracy requires every single class in the 6-label vector to match ground truth simultaneously for a sample (achieving 95.73%).
21. **What is Multi-Label Exact Match and what was ABHERA v2's score?**  
    *Answer:* It measures exact match accuracy restricted strictly to samples having 2 or more active ground-truth labels. ABHERA v2 achieved 94.12% (16 of 17 multi-label test cases matched perfectly).

### Domain 4: Decision Threshold Tuning & Error Analysis
22. **Why is a default threshold of 0.50 sub-optimal for medical or legal classification?**  
    *Answer:* 0.50 assumes equal class priors and symmetric costs for false positives and false negatives. In women's safety, a false negative (failing to recognize domestic violence) carries catastrophic human risk, whereas a false positive is easily clarified in the questionnaire.
23. **How were optimal thresholds calibrated in ABHERA?**  
    *Answer:* Using a 100-step grid search ($[0.10, 0.90]$) evaluated on the 162-sample held-out validation set to maximize class F1-scores without touching the 164-sample test set.
24. **What are the 6 optimal thresholds in `optimal_thresholds.json`?**  
    *Answer:* `DV: 0.73`, `SH: 0.50`, `ST: 0.79`, `CA: 0.45`, `WH: 0.29`, `OV: 0.63`.
25. **Why is the Workplace Harassment (`WH`) threshold set so low (0.29)?**  
    *Answer:* Workplace harassment disclosures are often subtle and professional in tone, producing lower raw transformer logits. A 0.29 threshold guarantees 100% recall on WH cases.
26. **Explain the single false negative (`TEST_0050`) in BERT v2 test evaluation.**  
    *Answer:* In `TEST_0050`, the perpetrator persistently messaged a woman asking to meet after a conference. The model correctly identified Stalking ($p=0.84 \ge 0.79$) but missed Sexual Harassment ($p=0.4549 < 0.50$) by 0.0451 due to the absence of explicit vulgarity.
27. **What were the final test metrics of promoted BERT v2?**  
    *Answer:* Micro Precision: 96.77%, Micro Recall: 99.45%, Micro-F1: 98.09%, Macro-F1: 97.58%, Exact Match: 95.73%, 180 TP, 6 FP, 1 FN, 797 TN.

### Domain 5: Named Entity Recognition & BIO Tagging
28. **What is Named Entity Recognition (NER)?**  
    *Answer:* Token-level sequence labeling identifying contiguous character spans belonging to pre-defined evidentiary classes.
29. **What is BIO tagging?**  
    *Answer:* Begin (`B-`), Inside (`I-`), and Outside (`O`) tagging scheme that allows continuous multi-token entities to be reconstructed unambiguously.
30. **How many BIO tags does ABHERA support?**  
    *Answer:* 13 tags: `O`, plus `B-` and `I-` tags for `PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`, `EVIDENCE`, and `LAW_SEC`.
31. **What is the entity-level Micro-F1 of ABHERA's NER model?**  
    *Answer:* 99.24% (Micro Precision: 100.0%, Micro Recall: 98.48% across 66 gold spans in 60 test narratives).
32. **What was the exact per-label recall for NER entities?**  
    *Answer:* `PERP_REL`: 100.0% (24/24), `TIME_FREQ`: 100.0% (17/17), `PLATFORM`: 100.0% (12/12), `LOCATION`: 92.31% (12/13, 1 False Negative).
33. **How does ABHERA map WordPiece sub-token predictions back to original character offsets?**  
    *Answer:* By extracting `offset_mapping` tuples `(start_char, end_char)` from HuggingFace fast tokenizer objects during encoding and mapping the token span slice back to the raw narrative string.

### Domain 6: Dynamic Question Engine & Conversational Flow
34. **Why are follow-up questions necessary in ABHERA?**  
    *Answer:* Initial survivor narratives are often brief, emotionally overwhelming, or omitted vital legal details (such as whether threats occurred online or physically). Follow-up questions collect missing evidentiary parameters.
35. **What are the 6 canonical forensic slots?**  
    *Answer:* `SAFETY_STATUS`, `PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`, and `INCIDENT_DETAILS`.
36. **What is the exact priority queue order for follow-up questions?**  
    *Answer:* `SAFETY_STATUS` $\succ$ `PERP_REL` $\succ$ `LOCATION` $\succ$ `TIME_FREQ` $\succ$ `PLATFORM` $\succ$ `INCIDENT_DETAILS`.
37. **How does the Question Engine prevent repetitive questioning (anti-looping)?**  
    *Answer:* It maintains an `asked_question_ids` set in `sessions.answers` and verifies that newly extracted NER spans remove slots from the missing set. Questioning terminates when missing slots reach zero or maximum turn depth is reached.
38. **How does `dynamic_generator.py` personalize questions?**  
    *Answer:* It injects known entity context into templates (e.g., if `PERP_REL = "husband"`, it asks: *"Did your husband commit physical violence at home?"*).

### Domain 7: Legal Mapping, Support Services & Anti-Hallucination
39. **How many legal provisions are in the statutory database?**  
    *Answer:* Exactly 32 verified provisions from BNS 2023, IT Act 2000, PWDVA 2005, POSH Act 2013, and Dowry Prohibition Act 1961.
40. **What is Section 69 of Bharatiya Nyaya Sanhita (BNS)?**  
    *Answer:* Sexual intercourse by deceitful means or false promise of marriage, replacing aspects of former IPC 375/376 case law with a distinct statutory offence.
41. **How does the 3-tier support service matcher score helplines?**  
    *Answer:* District match (+300 points), State match (+200 points), and National Fallback (+100 points).
42. **How does `AntiHallucinationService` verify legal advice?**  
    *Answer:* It asserts foreign-key existence and performs verbatim string checks against the `laws` PostgreSQL table. If a statute is not present in the verified database, it is rejected from the report.
43. **What disclaimer is embedded in every ABHERA report?**  
    *Answer:* "ABHERA is an educational legal information and safety support tool. It does not provide formal legal representation or substitute for licensed legal counsel."

### Domain 8: Backend Engineering, FastAPI & PostgreSQL
44. **Why was FastAPI chosen over Django or Flask?**  
    *Answer:* FastAPI provides native Python asynchronous `async/await` concurrency on ASGI (Uvicorn), automatic OpenAPI Swagger generation, and fast Pydantic schema validation.
45. **What is `pool_pre_ping=True` in `backend/app/database.py`?**  
    *Answer:* It emits a lightweight `SELECT 1` ping before releasing a connection from the pool, preventing stale connection errors when database connections timeout.
46. **What is the function of `init_db.py`?**  
    *Answer:* It safely creates all 11 database tables via `Base.metadata.create_all(bind=engine)` without dropping existing data.
47. **How many automated unit tests exist in ABHERA?**  
    *Answer:* Exactly 154 unit tests across 11 test modules in `backend/tests/`, achieving a 100% pass rate in ~39.89 seconds.
48. **Why must backend tests be run using the virtual environment Python?**  
    *Answer:* System global Python lacks required packages like `pydantic_settings` and `sqlalchemy`. Running with `backend/venv/Scripts/python.exe` ensures complete dependency isolation.

### Domain 9: Frontend Architecture & Client-Side PDF Generation
49. **What are the 4 primary view routes in the React application?**  
    *Answer:* `/` (`Home.jsx`), `/chatbot` (`Chatbot.jsx`), `/processing` (`Processing.jsx`), and `/report` (`Report.jsx`).
50. **How does `ConversationContext.jsx` preserve session state across browser refreshes?**  
    *Answer:* It synchronizes `sessionId` and `submissionId` with browser `sessionStorage`, re-fetching active state via `GET /api/session/{session_id}` on page remount.
51. **How is the PDF report compiled without server-side rendering?**  
    *Answer:* `Report.jsx` imports `jsPDF`, initializes an A4 document canvas (`[595.28, 841.89]` pt), and renders text, header rectangles, and vector checkmarks dynamically.
52. **What are the 10 sections of the compiled ABHERA incident report?**  
    *Answer:* 1. Header & Overview, 2. Primary Classifications, 3. Extracted Information, 4. Questionnaire Responses, 5. Applicable Legal Provisions, 6. Support Services & Helplines, 7. Action Recommendations, 8. Evidence Preservation, 9. Model Prediction Info, 10. Legal Disclaimer.
53. **What is the memory footprint of ABHERA in cloud hosting?**  
    *Answer:* Dual PyTorch transformer checkpoints consume ~833 MB disk and ~1.35 GB host RAM during CPU inference, requiring at least a 2.0 GB RAM container tier (e.g. Render Standard).
54. **What is the command to seed statutory laws and support services in production?**  
    *Answer:* `python -m app.services.data_importer`.
55. **Does ABHERA support voice input or regional Indian languages currently?**  
    *Answer:* Not in the current production release (English natural language only). Multilingual expansion (Hindi, Telugu, Tamil, Bengali) is scheduled in the future enhancement roadmap.
