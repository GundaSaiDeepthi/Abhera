"""
Module for Chapter 1: Introduction to ABHERA
"""

def get_chapter_01():
    return r'''# CHAPTER 1: Introduction to ABHERA

---

## 1.1 Meaning and Definition of ABHERA

**ABHERA** is an acronym that stands for **Adaptive BERT-based Harassment Entity Recognition & Assistance**.

To understand what ABHERA is from the ground up, let us dissect every word of its name:

```
+--------------------------------------------------------------------------------------------------+
|                                    THE ABHERA ACRONYM DECODED                                    |
+---+----------------+-----------------------------------------------------------------------------+
| A | Adaptive       | The system dynamically modifies its follow-up questions based on what       |
|   |                | information the survivor has already shared and what is legally missing.   |
+---+----------------+-----------------------------------------------------------------------------+
| B | BERT-based     | Powered by Bidirectional Encoder Representations from Transformers, a state-|
|   |                | of-the-art deep learning model developed by Google for text understanding.  |
+---+----------------+-----------------------------------------------------------------------------+
| H | Harassment     | Specifically engineered to identify and categorize 6 distinct harassment    |
|   |                | dimensions: Domestic Violence, Sexual Harassment, Stalking, Cyber Abuse,    |
|   |                | Workplace Harassment, and Other Violence.                                   |
+---+----------------+-----------------------------------------------------------------------------+
| E | Entity         | Identifies specific factual building blocks in the narrative, such as who  |
|   |                | the perpetrator is, where it occurred, and what platform was used.          |
+---+----------------+-----------------------------------------------------------------------------+
| R | Recognition    | Automatically classifies each word using Named Entity Recognition (NER)     |
|   |                | token classification with sub-word offset alignment.                        |
+---+----------------+-----------------------------------------------------------------------------+
| A | Assistance     | Provides zero-hallucination legal information under Indian law and connects |
|   |                | survivors with emergency helplines and generates an evidentiary report.     |
+---+----------------+-----------------------------------------------------------------------------+
```

In plain language: **ABHERA is an intelligent, trauma-informed software assistant designed to help women and survivors in India document harassment, understand which laws protect them, and receive immediate support without fear, confusion, or loss of privacy.**

---

## 1.2 Background of the Project

Harassment against women is one of the most widespread yet severely under-reported societal crises worldwide, and particularly across India:

1. **Alarming Crime Statistics**: According to the National Crime Records Bureau (NCRB) of India, hundreds of thousands of crimes against women are registered annually. These encompass domestic cruelty, cyber harassment, online stalking, workplace harassment, and physical violence. However, criminologists and social workers estimate that registered crimes represent only a fraction of actual incidents.
2. **The Trauma and Silence Cycle**: Experiencing harassment induces profound emotional shock, fear, shame, and anxiety. When a person is traumatized, cognitive science proves that their memory retrieval is fragmented. Survivors often cannot recall events chronologically or articulate their ordeal in the rigid, formal language demanded by police officers or legal institutions.
3. **The Stigma of Approaching Law Enforcement**: Entering a physical police station to file a First Information Report (FIR) is intimidating. Survivors fear social stigma, hostile questioning, victim-blaming, retaliation from perpetrators, and lack of confidentiality.
4. **The Critical Statutory Transition (BNS 2023)**: On July 1, 2024, the Republic of India officially enacted the **Bharatiya Nyaya Sanhita (BNS 2023)**, replacing the historic **Indian Penal Code (IPC 1860)**. While this modernization introduced crucial provisions (such as Section 69 punishing deceitful sexual intercourse on false promises of marriage, and updated stalking provisions under Section 78), it also created widespread confusion. Citizens, students, and even frontline social workers often do not know which new section applies to their situation.
5. **The Technological Gap**: Until now, a person seeking help had only two choices: wait on long telephone queues for crisis hotlines, or search the internet where legal jargon is dense and unhelpful. Recent general-purpose generative AI tools (such as standard chatbots) have emerged, but they introduce a dangerous risk: **hallucination**.

---

## 1.3 Problem Statement

When a survivor experiences harassment, they face a severe four-fold barrier:

```
               [ Traumatic Incident Occurs ]
                             |
         +-------------------+-------------------+
         |                                       |
[ Cognitive Fragmentation ]             [ Legal Jargon & Confusion ]
Survivor struggles to recall             Cannot identify applicable
dates, platforms, or facts              BNS 2023 / IT Act sections
         |                                       |
[ Fear & Hostility at Stations ]        [ LLM Generative Hallucination ]
Hesitation to approach police           Generic AI chatbots invent fake
due to stigma and retaliation           laws, misleading the survivor
         +-------------------+-------------------+
                             |
              [ CRISIS: Justice Delayed or Denied ]
```

Specifically:
- **Lack of Structured Triage**: When survivors attempt to write what happened, they often omit critical evidentiary anchors (such as dates, specific digital platforms, or frequency) required by law enforcement to take cognizance of an offense.
- **The LLM Hallucination Trap**: Large Language Models (LLMs) predict the next likely word probabilistically. When asked for legal advice, an LLM will confidently invent non-existent penal code numbers, fabricate jail sentences, or confuse civil dispute remedies with criminal offenses. In a legal context, citing a non-existent law on a police complaint causes immediate rejection.
- **Privacy and Data Exploitation**: Most online tools require phone numbers, email addresses, or social logins, creating a digital trail that can expose a vulnerable person to discovery by an abusive partner or family member.

---

## 1.4 Why ABHERA is Needed

ABHERA was created to solve these exact problems through a **neuro-symbolic, safety-critical architecture**:

- **Trauma-Informed Conversational Intake**: Instead of forcing the user to fill out cold, intimidating administrative forms, ABHERA allows them to type freely in everyday English.
- **Dual-Task Deep Learning Perception**: ABHERA uses two specialized BERT models. One model classifies the incident into legal categories (e.g., Stalking and Cyber Abuse), while the second model extracts specific factual entities (e.g., "ex-boyfriend", "Instagram", "past 3 weeks").
- **Adaptive Dynamic Questioning**: If the survivor mentions being stalked on Instagram but forgets to mention how often it happens or whether they feel physically safe, ABHERA gently asks targeted follow-up questions to fill only the missing details. It never repeats questions or loops endlessly.
- **Deterministic, Zero-Hallucination Legal Grounding**: ABHERA **never uses generative AI to write legal advice**. Instead, it looks up verified statutory sections (from BNS 2023, IT Act 2000, POSH Act 2013, and PWDVA 2005) directly from an audited PostgreSQL database. If a law is not verified in the database, ABHERA states that clearly.
- **Emergency Protection & Confidentiality**: Immediate emergency helplines (112, 181, 1090) are displayed instantly. The system does not require email addresses, passwords, or phone numbers to use, and all session credentials evaporate when the browser tab is closed.
- **Structured Evidentiary PDF Dossier**: The survivor can download an organized 10-section incident summary compiled directly in their web browser, which they can take to a lawyer, police station, or university grievance committee.

---

## 1.5 Existing Systems and Their Limitations

To appreciate ABHERA\\'s innovation, we must analyze the existing alternatives available to survivors:

| System / Approach | Working Mechanism | Critical Limitations |
| :--- | :--- | :--- |
| **Physical Police Station Reporting** | Survivor visits police station in person, narrates the story verbally to a duty officer who hand-writes an FIR. | High emotional barrier, victim-blaming, intense social stigma, fear of retaliation, and potential distortion of facts during transcription. |
| **Telephone Helplines (112, 181, 1090)** | Voice call to government emergency tele-operators. | Frequent busy signals, long hold times, operators lack the time to draft detailed evidentiary incident dossiers, verbal memory fades. |
| **Static Legal Web Portals** | Web pages containing raw text of Acts and legal articles (e.g., India Code, legal blogs). | Written in complex legal jargon, difficult for non-lawyers to navigate, requires knowing the exact legal terms in advance. |
| **Generic Commercial LLMs (ChatGPT, Claude, Gemini)** | Cloud-hosted autoregressive text generators trained on internet text. | **Severe hallucination risk**: fabricates repealed IPC sections, invents imaginary subsections, leaks sensitive user data to cloud training logs. |
| **Fixed Survey Forms / Dropdown Portals** | Web forms with 30+ mandatory dropdown fields and check-boxes. | Emotionally exhausting, rigid, forces traumatic narratives into ill-fitting checkboxes, high user abandonment rate. |

---

## 1.6 Proposed System: ABHERA

ABHERA introduces a hybrid **Neuro-Symbolic Architecture** that divides the task into two strictly separated stages:

```
+---------------------------------------------------------------------------------------------------+
|                                  THE ABHERA ARCHITECTURAL SEPARATION                              |
+---------------------------------------------------------------------------------------------------+
|  STAGE 1: NEURAL PERCEPTION (Deep Learning)                                                       |
|  - Fine-Tuned BERT Multi-Label Classifier: Categorizes free-form text into 6 legal classes.        |
|  - Fine-Tuned BERT Token Classifier (NER): Extracts 13-tag BIO entities from narrative spans.     |
|  - Role: UNDERSTAND AND EXTRACT ONLY. No legal text generation.                                  |
+---------------------------------------------------------------------------------------------------+
|                                                 |                                                 |
|                                                 v                                                 |
+---------------------------------------------------------------------------------------------------+
|  STAGE 2: SYMBOLIC DETERMINISM (Relational Knowledge Base & Rules)                                |
|  - Dynamic Question State Machine: Identifies missing slots and selects targeted questions.        |
|  - PostgreSQL Relational Database: Contains 32 audited legal provisions and 12 emergency services.|
|  - Anti-Hallucination Gating Layer: Validates that every citation originates verbatim from the DB. |
|  - Client-Side Vector PDF Engine: Compiles the 10-section dossier in browser memory via jsPDF.   |
|  - Role: VERIFY, RETRIEVE, AND ASSEMBLE. Zero generative hallucination.                           |
+---------------------------------------------------------------------------------------------------+
```

---

## 1.7 Objectives of the Project

The engineering objectives of the ABHERA project are:

1. **Develop an Accurate Multi-Label Harassment Classifier**: Train a fine-tuned `bert-base-uncased` model to recognize 6 harassment categories simultaneously, achieving greater than 95% Micro-F1 and mitigating severe class imbalance using positive-weight Binary Cross-Entropy loss.
2. **Implement Fine-Grained Legal Named Entity Recognition (NER)**: Train a token-level classifier to identify factual entities (Perpetrator Relationship, Location, Time/Frequency, Platform, Evidence, Law Section) using a 13-tag BIO schema with WordPiece offset alignment.
3. **Build an Adaptive Dynamic Question Engine**: Design a slot-filling state machine that evaluates what information is known versus what is missing for a given crime category, asking targeted follow-up inquiries without conversational loops.
4. **Construct an Audited Indian Legal Knowledge Base**: Populate a PostgreSQL relational schema with 32 statutory provisions under BNS 2023, IT Act 2000, POSH Act 2013, and PWDVA 2005, alongside a 3-tier emergency support directory.
5. **Implement an Anti-Hallucination Safeguard Layer**: Enforce programmatic validation that asserts every legal section, punishment description, and helpline number returned by the API matches a primary key in PostgreSQL verbatim.
6. **Create a Privacy-Preserving Frontend and PDF Compiler**: Build a responsive React SPA that allows survivors to interact anonymously, and compile a formal 10-section incident dossier into a downloadable PDF entirely client-side without storing personal data on the server.

---

## 1.8 Scope of the Project

### In-Scope
- **Jurisdictional Boundary**: Republic of India statutory criminal and civil legal frameworks.
- **Statutory Coverage**:
  - Bharatiya Nyaya Sanhita (BNS 2023)
  - Information Technology Act (IT Act 2000)
  - Protection of Women from Domestic Violence Act (PWDVA 2005)
  - Sexual Harassment of Women at Workplace (POSH Act 2013)
  - Dowry Prohibition Act (1961)
- **Harassment Categories**: Domestic Violence (`DV`), Sexual Harassment (`SH`), Stalking (`ST`), Cyber Abuse (`CA`), Workplace Harassment (`WH`), and Other Violence (`OV`).
- **Entity Dimensions**: Perpetrator Relationship (`PERP_REL`), Location (`LOCATION`), Time and Frequency (`TIME_FREQ`), Platform (`PLATFORM`), Evidence (`EVIDENCE`), and Statutory Sections (`LAW_SEC`).
- **Language**: English natural language input.

### Out-of-Scope (Distinguished from Planned Future Work)
- **Legal Representation**: ABHERA is an educational and forensic triage tool; it does not replace a licensed advocate or represent clients in court.
- **Automated Police Filing**: The system generates a dossier for the survivor to hand to authorities; it does not automatically submit FIRs into police databases (which requires government CCTNS API credentials).
- **Audio / Voice Intake**: Currently processes written text; voice-to-text input is a planned future enhancement.
- **Vernacular Indian Languages**: The initial production release operates on English text; multilingual fine-tuning (e.g., via IndicBERT) is planned for future iterations.

---

## 1.9 Advantages of ABHERA

1. **Absolute Legal Truth (Zero Hallucination)**: Because legal provisions are retrieved deterministically from PostgreSQL and gated by an Anti-Hallucination layer, it is mathematically impossible for ABHERA to invent fake penal sections.
2. **Multi-Label Reality**: Real-world crimes often overlap (e.g., an ex-partner who physically abuses a survivor while also sending harassing WhatsApp messages). ABHERA identifies both Domestic Violence and Cyber Abuse simultaneously.
3. **Trauma-Informed Interaction**: The user can pour out their thoughts in unstructured text. The system gently asks only what is legally necessary to help them.
4. **Forensic Report Generation**: Generates an organized, 10-section PDF complete with incident summary, classified categories, extracted facts, questionnaire answers, legal citations, and helpline numbers.
5. **Privacy First**: No user account creation, no password storage, no email logging, and zero server-side storage of final PDF files.
6. **Empirically Proven**: Built with rigorous machine learning standards, achieving **98.09% Micro-F1** on multi-label classification and **99.24% F1** on entity recognition across held-out test data.

---

## 1.10 Real-World Applications

- **University Internal Complaints Committees (ICC)**: Providing college students with an immediate, confidential platform to understand whether an incident constitutes workplace/campus harassment under the POSH Act.
- **Corporate HR Grievance Redressal**: Helping employees structure complaints before approaching the corporate POSH committee, ensuring all necessary evidence (emails, timestamps, witnesses) is documented.
- **Legal Aid Clinics & NGOs**: Assisting paralegals and social workers in triaging intake narratives and preparing initial complaint drafts for underprivileged survivors.
- **Cybercrime Awareness**: Helping victims of online stalking, non-consensual image sharing, and cyberbullying identify relevant IT Act provisions (Section 66E, 67, 67A) and emergency cyber reporting channels.
- **Personal Evidence Preservation**: Giving survivors a timestamped, structured document of their ordeal before digital evidence (e.g., temporary messages or chat logs) is deleted or lost.

---

## 1.11 Limitations of the System

To maintain complete academic honesty during viva examination, the following verified limitations are acknowledged:

1. **Single Language Support (English Only)**: While many Indian survivors speak Hindi, Telugu, Tamil, or other regional languages, the current models are trained on English narratives.
2. **CPU Inference Latency**: Running two 110-million parameter BERT models on standard consumer CPU hardware requires approximately 200ms to 450ms per forward pass.
3. **Session Volatility**: Because user data is stored in transient browser `sessionStorage` for privacy protection, clearing browser cache or closing the browser window resets the session.
4. **Informational Nature**: ABHERA is an automated assistance tool and cannot provide formal legal advice, assess court admissibility of specific evidence items, or represent survivors before a magistrate.
'''
