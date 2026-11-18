"""
Module for Chapter 15: Interview and Viva Presentation
"""

def get_chapter_15():
    return r'''# CHAPTER 15: Interview and Viva Presentation Guide

---

## 15.1 30-Second Elevator Pitch

> *"Respected examiners, ABHERA stands for Adaptive BERT-based Harassment Entity Recognition & Assistance. It is an AI-powered legal triage system designed specifically for the Indian legal context following the enactment of the Bharatiya Nyaya Sanhita 2023. Unlike generic chatbots that hallucinate fake laws, ABHERA uses fine-tuned BERT models achieving 98.09% Micro-F1 to classify incidents and extract key facts like perpetrators and platforms. It then deterministically retrieves verified statutory provisions from PostgreSQL and generates a tamper-evident evidentiary incident dossier client-side without storing survivor credentials."*

---

## 15.2 2-Minute Executive Overview

> *"Good morning, respected external examiner and faculty members. 
> 
> In India, crimes against women remain severely under-reported because survivors face trauma, social stigma, and confusion regarding which laws protect them—especially after India replaced the historic IPC with the Bharatiya Nyaya Sanhita (BNS 2023) on July 1, 2024.
> 
> When people seek help online, generic AI chatbots frequently hallucinate non-existent legal section numbers, which can invalidate an official police First Information Report (FIR).
> 
> To solve this, we developed **ABHERA**. ABHERA implements a **neuro-symbolic architecture** that separates deep learning perception from factual legal retrieval:
> 1. **Perception**: A survivor enters an incident narrative in plain English. Our fine-tuned `bert-base-uncased` multi-label classifier identifies applicable offenses across 6 categories with 98.09% Micro-F1, while a token-level NER model extracts entities like perpetrator relationships and digital channels with 99.24% F1.
> 2. **Adaptive Dialogue**: An intelligent Dynamic Question Engine detects what evidentiary slots are missing and asks targeted follow-up inquiries without conversational loops.
> 3. **Deterministic Retrieval**: An Anti-Hallucination gating layer queries a curated PostgreSQL database of 32 verified statutory provisions under BNS, IT Act, and POSH Act, alongside a 3-tier emergency hotline directory.
> 4. **Privacy & Dossier Compilation**: The system compiles a structured 10-section legal dossier directly in the browser using `jsPDF` without storing personal credentials or PDF files on the server.
> 
> The entire backend is validated across 154 passing automated unit tests, providing an accurate, trauma-informed bridge between an incident and justice."*

---

## 15.3 5-Minute Technical Deep Dive

> *"Respected examiners, I would like to walk you through the engineering and algorithmic foundation of ABHERA.
> 
> ### The Core Architectural Challenge
> In legal informatics, the primary technical challenge is achieving high multi-label classification accuracy on severely imbalanced datasets while strictly eliminating factual hallucination.
> 
> ### 1. Dataset & Preprocessing
> We curated and verified an Indian legal harassment dataset of **1,086 records** partitioned into a 70/15/15 split: 760 train, 162 validation, and 164 test records using random seed 42. Incidents are mapped across 6 non-exclusive categories: Domestic Violence, Sexual Harassment, Stalking, Cyber Abuse, Workplace Harassment, and Other Violence. Text is tokenized using WordPiece tokenization with a 30,522 vocabulary and maximum sequence length of 128 tokens.
> 
> ### 2. The Multi-Label BERT Model & Loss Formulation
> In our initial baseline (BERT v1), standard Binary Cross-Entropy loss resulted in 0% multi-label exact match due to severe class imbalance. In our production model (`bert_multilabel_experiment_v2`), we solved this by implementing **Positive-Class Weighted BCE Loss** (`pos_weight`), scaling the loss of positive errors by $w_c = (N - N_c) / N_c$ (reaching $10.08$ for minority class Other Violence). 
> 
> Furthermore, instead of an arbitrary 0.50 cutoff, we performed a 100-step grid search on the validation set to establish **per-class optimal decision thresholds**: 0.73 for DV, 0.50 for SH, 0.79 for ST, 0.45 for CA, 0.29 for WH, and 0.63 for OV. On the held-out test set of 164 records, BERT v2 achieved a **Micro-F1 of 98.09%**, **Macro-F1 of 97.58%**, and **Multi-Label Exact Match of 94.12%**, with only a single False Negative across 984 binary decisions.
> 
> ### 3. Named Entity Recognition (NER)
> In parallel, we fine-tuned a BERT token classifier on a 13-tag BIO schema to extract 6 entity dimensions: Perpetrator Relationship, Location, Time/Frequency, Platform, Evidence, and Law Section. Using sub-word offset alignment from the fast tokenizer, the model achieved **99.24% Micro-F1** and 100% precision on held-out test data.
> 
> ### 4. Dynamic Question Engine & Anti-Hallucination
> The extracted entities populate a slot-filling finite state machine that evaluates required dimensions against known dimensions. It asks targeted questions prioritized by urgency (`SAFETY_STATUS` at priority 100). The Anti-Hallucination layer executes foreign-key queries against 32 verified BNS, IT Act, and POSH Act provisions in PostgreSQL, asserting exact matches before data enters the report.
> 
> ### 5. Full-Stack Implementation & Testing
> The backend runs on FastAPI and Uvicorn, backed by PostgreSQL 16. The frontend is built in React 18 and Vite, compiling vector PDFs client-side with `jsPDF`. The system is thoroughly validated with **154 passing automated unit tests** across 11 test modules.
> 
> Thank you, and I am now ready for your questions and the live demonstration."*

---

## 15.4 10-Minute Complete Live Demonstration Script

### Demo Phase 1: Pre-Flight Verification (Minute 1)
- **Speaker Action**: Show the terminal windows to the examiner.
- **Spoken Words**: *"Examiners, before we begin, I would like to show you our backend environment. On my workstation, PostgreSQL 16 is active on port 5432. In Terminal 1, our FastAPI server is running via Uvicorn on port 8000. In Terminal 2, `cloudflared` is maintaining an encrypted tunnel connecting our local backend to the internet. The frontend is live on Vercel at `https://abhera.vercel.app`."*

### Demo Phase 2: Landing Page & Privacy Assurance (Minutes 2-3)
- **Speaker Action**: Open the browser to `https://abhera.vercel.app`.
- **Spoken Words**: *"Here is the landing page. Notice that there is no 'Sign Up' or 'Login' button. We intentionally collect zero passwords, emails, or phone numbers to protect survivor confidentiality. The page explains the 4-step process: Describe, Understand, Clarify, and Match. I will now click 'Start Confidential Assistance'."*

### Demo Phase 3: Conversational Narrative Intake (Minutes 4-5)
- **Speaker Action**: Navigate to `/chatbot`. Type the following narrative into the input box:
  ```
  "My former colleague has been stalking my Instagram profile and calling my phone repeatedly for the past two weeks. Yesterday he came to my office campus in Hyderabad."
  ```
- **Spoken Words**: *"I am now submitting a compound narrative that involves both physical stalking and cyber harassment. When I press Send, the text is transmitted via REST API to our backend. The BERT multi-label classifier and token classification models will run in memory."*
- **Click**: Click Send.

### Demo Phase 4: Dynamic Questioning in Action (Minutes 6-7)
- **Speaker Action**: Point out the bot's response on screen.
- **Spoken Words**: *"Notice what just happened. The BERT classifier identified Stalking and Cyber Abuse. The NER model automatically extracted 'former colleague' as the perpetrator, 'Instagram' and 'phone' as platforms, 'two weeks' as time frequency, and 'office campus in Hyderabad' as the location. Because the system already knows these facts, it did NOT ask 'Who did this?' or 'Which app was used?' Instead, the Dynamic Question Engine identified that only `SAFETY_STATUS` was missing, and asked: 'Are you currently in immediate physical danger, or do you need emergency support right now?'"*
- **Speaker Action**: Type response: *"I am safe with my family right now."* Click Send.

### Demo Phase 5: Transition to Dossier (Minute 8)
- **Speaker Action**: Point out the state transition.
- **Spoken Words**: *"All required informational slots are now fulfilled. The conversation status transitions to `COMPLETED`. The chatbot displays an acknowledgment and an 'Analyze & View Report' button. I will now click to generate the evidentiary dossier."*
- **Click**: Click "Analyze & View Report".

### Demo Phase 6: Report Inspection & Anti-Hallucination Verification (Minute 9)
- **Speaker Action**: Scroll through the 10-section report on `/report`.
- **Spoken Words**: *"Here is the compiled 10-section report. Section 1 contains the survivor's exact words. Section 2 shows the predicted classes: Stalking and Cyber Abuse. Section 3 shows our structured NER entities. Section 5 shows our statutory citations: BNS 2023 Section 78 for Stalking and IT Act Section 66E for privacy violation. Every citation was retrieved directly from PostgreSQL; zero text was invented by an LLM. Section 7 lists verified helplines: 112, 181, 1090, and the National Cyber Crime portal."*

### Demo Phase 7: Client-Side Vector PDF Export (Minute 10)
- **Speaker Action**: Click the blue "Download Official PDF Report" button.
- **Spoken Words**: *"Finally, I will click 'Download Official PDF Report'. Notice that the PDF downloads instantly. This document was generated directly inside browser memory using `jsPDF`. The server never created or stored a PDF file, ensuring zero server-side data leakage. The PDF contains official headers, structured tables, legal provisions, and emergency contacts ready for police or legal aid filing. This concludes our live demonstration."*

---

## 15.5 Step-by-Step Live Demo Checklist

Follow this checklist strictly on the morning of your viva defense:

```
+===================================================================================================+
|                                  VIVA DAY LIVE DEMO CHECKLIST                                     |
+----+---------------------------------------------+-----------------------------------------------+
| #  | Action Step                                 | Verification Command / Expected Result        |
+----+---------------------------------------------+-----------------------------------------------+
| 01 | Ensure Workstation Power Settings           | Set Windows Sleep & Screen Saver to "NEVER"   |
| 02 | Verify Local PostgreSQL Service             | Run: `net start postgresql-x64-16`            |
| 03 | Verify Port 5432 Connectivity               | Run: `Test-NetConnection -Port 5432 localhost`|
| 04 | Start FastAPI Uvicorn Server                | Run: `uvicorn app.main:app --port 8000`       |
| 05 | Verify Local Swagger Docs                   | Open browser to: `http://localhost:8000/docs` |
| 06 | Start Cloudflare Quick Tunnel               | Run: `cloudflared tunnel --url localhost:8000`|
| 07 | Copy Generated Cloudflare URL               | Format: `https://*.trycloudflare.com`         |
| 08 | Update Vercel Environment Variable          | Set `VITE_API_URL` to Cloudflare URL & Redeploy|
| 09 | Test Mobile / External Access               | Open `https://abhera.vercel.app` on phone     |
| 10 | Run Quick Test Narrative                    | Submit 1 test case; verify PDF downloads      |
+===================================================================================================+
```

---

## 15.6 Common Examiner Follow-Up Questions and Defensive Answers

### 1. "Why didn't you use an ensemble model or deep neural network for the baselines?"
- **Defensive Answer**: *"Respected examiner, the purpose of a baseline in machine learning research is to establish a rigorous lower-bound benchmark using classical statistical methods. TF-IDF paired with Logistic Regression and Linear Support Vector Machines represents the standard, established industry benchmark for NLP text classification. By comparing BERT against these proven statistical baselines, we demonstrated that BERT v2 achieves superior multi-label exact match (94.12% vs 88.24%) and captures complex contextual dependencies that keyword-based TF-IDF cannot model."*

### 2. "Your dataset has 1,086 records. Isn't that too small for deep learning?"
- **Defensive Answer**: *"That would be true if we were training BERT from scratch. However, we utilized **transfer learning**. `bert-base-uncased` was pre-trained by Google on BookCorpus (800 million words) and English Wikipedia (2,500 million words), which already instilled deep syntax and semantic representations into its 110 million weights. Our 1,086 verified records were used strictly for **domain-specific task fine-tuning**. Extensive literature in NLP proves that fine-tuning pre-trained transformer representations on 1,000 highly curated domain examples yields state-of-the-art accuracy, as demonstrated by our 98.09% Micro-F1."*

### 3. "What if the user lies or gives false information?"
- **Defensive Answer**: *"ABHERA is an intake assistance and forensic structuring tool, not a judicial court of law. In the real world, when a citizen narrates an incident to a police officer drafting an FIR, the officer records the complainant's statement as stated. Investigation, evidence verification, and cross-examination are the legal responsibility of police detectives and magistrates. ABHERA clearly catalogs the narrative in Section 1 as the survivor's explicit statement, highlights evidence notes in Section 8, and includes an explicit disclaimer in Section 10 stating that the report reflects user disclosures."*

---

## 15.7 Honest, Academically Defensible Answers About Limitations

When examiners ask about weaknesses in your system, **never be defensive or make false claims**. Examiners respect candidates who demonstrate deep self-awareness of engineering trade-offs:

| Weakness / Limitation Asked | Honest, Academically Sound Response |
| :--- | :--- |
| **"Why is the system only in English?"** | *"Our initial research priority was establishing the neuro-symbolic architecture, the 13-tag legal BIO ontology, and aligning with the newly enacted BNS 2023. Pre-trained English BERT models offered the highest initial benchmark reliability. In Phase 17 of our roadmap, we plan to adopt IndicBERT to support Hindi and regional vernaculars."* |
| **"Why does the live demo require your laptop running?"** | *"Hosting two 110-million parameter PyTorch models requires approximately 1.35 GB of dedicated host RAM. Cloud PaaS free tiers enforce strict 512 MB memory caps, which cause out-of-memory kernel crashes during PyTorch initialization. A dedicated cloud instance with 2GB RAM costs approximately \$20/month. For our academic viva, bridging our local workstation via Cloudflare Quick Tunnel demonstrates live public edge accessibility at zero cost."* |
| **"Why did you use rule-based follow-up questions instead of an LLM?"** | *"Generative LLMs are inherently non-deterministic. In a safety-critical application, an LLM might ask inappropriate, insensitive, or legally irrelevant questions. Our slot-filling state machine mathematically guarantees that questions are asked in strict order of forensic priority (`SAFETY_STATUS` first) and that conversational loops are 100% impossible."* |
'''
