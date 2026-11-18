"""
Module for Chapter 14: Viva Preparation - Part 1 (Sections A through F, Questions 1 to 81)
"""

def get_viva_part1():
    return r'''# CHAPTER 14: Complete Viva Preparation (Master Q&A Bank)

---

## 14.1 Structure of the Viva Question Bank

This master viva preparation guide contains **170 comprehensive, verified technical questions and answers** covering every layer of the ABHERA platform.

Every answer follows a three-part pedagogical structure:
1. **Beginner Answer**: Plain, simple language understandable by anyone with zero technical background.
2. **Technical Explanation**: Deep, precise computer science and engineering breakdown suitable for senior technical interviewers and university examiners.
3. **ABHERA-Specific Example**: Concrete code, database table, threshold value, or empirical test metric from the project.

---

## SECTION A: Basic Project Questions (Q1 – Q15)

### Q1: What is the full form and core meaning of ABHERA?
- **Beginner Answer**: ABHERA stands for **Adaptive BERT-based Harassment Entity Recognition & Assistance**. It is an AI-powered safety platform that helps women and victims of harassment document what happened to them, understand their legal rights in India, and get immediate help.
- **Technical Explanation**: ABHERA is a neuro-symbolic legal informatics and decision-support system. It couples fine-tuned bidirectional transformer neural networks (`bert-base-uncased`) with deterministic relational knowledge retrieval in PostgreSQL, governed by a slot-filling finite state machine and an anti-hallucination verification layer.
- **ABHERA Example**: The acronym represents the 6 architectural pillars: **A**daptive follow-up questioning, **B**ERT deep learning backbone, **H**arassment multi-label taxonomy (6 classes), **E**ntity extraction via token sequence labeling, **R**ecognition with character offset alignment, and **A**ssistance via 32 verified BNS/IT Act provisions.

### Q2: Why did you choose this project?
- **Beginner Answer**: Harassment against women in India is very common, but victims often do not report it because they are scared, confused by complicated legal terms, or afraid of going to a police station. We wanted to build a kind, private, 24/7 digital assistant that listens to their story in plain English, tells them exactly which laws protect them, and prepares a report they can show to authorities.
- **Technical Explanation**: On July 1, 2024, India replaced the Indian Penal Code (IPC 1860) with the Bharatiya Nyaya Sanhita (BNS 2023), introducing profound statutory ambiguity. Furthermore, popular generative LLMs (like ChatGPT) severely hallucinate legal section numbers, making them hazardous for safety-critical legal triage. ABHERA addresses this technical challenge by designing a dual-task deep learning perception model combined with zero-hallucination relational grounding.
- **ABHERA Example**: When a user mentions a partner coercing sexual relations by falsely promising marriage, generic chatbots often cite repealed Section 375 IPC or confuse it with rape. ABHERA deterministically retrieves the exact newly enacted **Section 69 BNS 2023** from its PostgreSQL `laws` table.

### Q3: What is the difference between ABHERA and a standard ChatGPT wrapper?
- **Beginner Answer**: ChatGPT writes text by guessing the next word probabilistically; it can make up fake laws and imaginary phone numbers. ABHERA never makes up legal advice. It uses AI only to understand what happened and who was involved, but all legal sections and phone numbers are pulled directly from a verified database.
- **Technical Explanation**: ChatGPT is an autoregressive decoder LLM prone to factual hallucination and non-deterministic text generation. In contrast, ABHERA separates perceptual classification from knowledge retrieval. The BERT models perform discriminative classification and token labeling; all downstream statutory sections and emergency services are queried via foreign-key asserts against an audited relational database.
- **ABHERA Example**: In `backend/app/services/anti_hallucination.py`, the `validate_legal_record` method performs a database lookup: `db.query(Law).filter(Law.act_name == act, Law.section_number == section)`. If a record does not exist in PostgreSQL, it is blocked from the report.

### Q4: What are the 6 harassment classes identified by ABHERA?
- **Beginner Answer**: Domestic Violence, Sexual Harassment, Stalking, Cyber Abuse, Workplace Harassment, and Other Violence.
- **Technical Explanation**: ABHERA models incident categorization as a 6-dimensional non-exclusive multi-label vector: `DV` (Domestic Violence), `SH` (Sexual Harassment), `ST` (Stalking), `CA` (Cyber Abuse), `WH` (Workplace Harassment), and `OV` (Other Violence / General Abuse).
- **ABHERA Example**: Formatted in `models/bert_multilabel_experiment_v2/label_mapping.json` as `{"0": "DV", "1": "SH", "2": "ST", "3": "CA", "4": "WH", "5": "OV"}`.

### Q5: What is the difference between multi-class and multi-label classification?
- **Beginner Answer**: Multi-class is like a multiple-choice test where only one answer can be chosen (e.g., classifying an animal as either a cat, a dog, or a bird). Multi-label is like a medical symptom checklist where multiple boxes can be checked at the same time (e.g., a patient can have both fever and cough).
- **Technical Explanation**: Multi-class classification uses a Softmax output layer where probabilities across all classes sum to 1.0 ($\sum p_i = 1$). Multi-label classification treats each category as an independent binary classification task using independent Sigmoid functions ($\sigma(z_i) \in [0, 1]$), allowing an arbitrary subset of labels to be active simultaneously.
- **ABHERA Example**: A narrative where an ex-partner stalks a woman on Instagram and threatens her is simultaneously predicted as both `ST` ($p = 0.88 \ge 0.79$) and `CA` ($p = 0.81 \ge 0.45$).

### Q6: What laws are included in ABHERA's database?
- **Beginner Answer**: Five key Indian laws: the new Bharatiya Nyaya Sanhita (BNS 2023), the Information Technology Act (2000), the Protection of Women from Domestic Violence Act (2005), the POSH Act for workplace harassment (2013), and the Dowry Prohibition Act (1961).
- **Technical Explanation**: Exactly 32 audited statutory provisions spanning criminal penal sanctions under BNS 2023, cyber offenses under IT Act 2000, civil protective remedies under PWDVA 2005, institutional grievance procedures under POSH Act 2013, and dowry prohibition sanctions under Act 28 of 1961.
- **ABHERA Example**: Querying `SELECT DISTINCT act_name FROM laws;` returns `'BNS'`, `'IT_ACT'`, `'DV_ACT'`, `'POSH_ACT'`, and `'DOWRY_ACT'`.

### Q7: How does ABHERA protect user privacy?
- **Beginner Answer**: Users do not need to sign up, create an account, or enter their name, phone number, or email. The conversation is stored temporarily in browser memory, and as soon as the browser tab is closed, everything is deleted.
- **Technical Explanation**: Privacy-by-design architecture: the `users` table records zero PII (no passwords, emails, or phone numbers), generating only an anonymous UUID. The frontend tracks state in browser `sessionStorage` (which self-destructs upon tab termination). Furthermore, PDF dossiers are compiled client-side via `jsPDF`, preventing sensitive legal documents from resting on server disks.
- **ABHERA Example**: In `backend/app/models/user.py`, the schema defines only `id`, `user_guid`, `is_anonymous = True`, and `created_at`.

### Q8: What happens if a user submits a narrative in Hindi or another Indian language?
- **Beginner Answer**: The current version of ABHERA is trained on English narratives, so a non-English narrative might result in low confidence scores or generic classification.
- **Technical Explanation**: The underlying tokenizer is `bert-base-uncased`, which has a 30,522-token vocabulary primarily composed of English words. Non-Latin characters trigger out-of-vocabulary `[UNK]` tokens or inefficient byte-level splits. Multilingual support via `IndicBERT` or `XLM-RoBERTa` is scheduled for future releases.
- **ABHERA Example**: When unmapped language text is entered, the classification probabilities remain below the class thresholds, causing the system to invoke the safe fallback: *"Incident type could not be determined from the available model output."*

### Q9: What is the Dynamic Question Engine?
- **Beginner Answer**: It is an intelligent logic system that figures out what important facts are missing from the user's story and asks gentle follow-up questions to fill in the blanks, without repeating questions.
- **Technical Explanation**: It is a slot-filling finite state machine (`backend/app/question_engine/`) that maintains a set of required forensic dimensions for each harassment category. It compares required slots against known slots (extracted by NER or user text) and selects the highest-priority missing question from a bank of prioritized inquiries.
- **ABHERA Example**: For Stalking (`ST`), required slots are `PERP_REL`, `TIME_FREQ`, `PLATFORM`, and `SAFETY_STATUS`. If the user stated *"My classmate is messaging me on Instagram"*, only `TIME_FREQ` and `SAFETY_STATUS` remain missing.

### Q10: What is the final output given to the user?
- **Beginner Answer**: A comprehensive 10-section incident report displayed on the screen and an official downloadable vector PDF document.
- **Technical Explanation**: An immutable, structured JSON dossier assembled by `ReportGeneratorService` comprising 10 conceptual sections (Incident Summary, Predicted Categories, Extracted Entities, Questionnaire Q&A, Statutory Citations, Suggested Actions, Crisis Helplines, Evidence Catalog, Model Provenance, and Legal Disclaimer), compiled client-side into a vector PDF via `jsPDF`.
- **ABHERA Example**: The generated PDF contains the official heading, metadata table, BNS legal sections with imprisonment terms, and emergency numbers (112, 181, 1090).

### Q11: What are the emergency numbers provided by ABHERA?
- **Beginner Answer**: 112 (All-in-one Emergency Police/Ambulance), 181 (Women's Helpline), 1090 (Women Power Line), and 1930 (National Cybercrime Helpline).
- **Technical Explanation**: A 3-tiered emergency directory: Tier 1 comprises National immediate response hotlines (112, 181, 1090); Tier 2 comprises specialized digital portals (`cybercrime.gov.in`, NCW cell); Tier 3 comprises local District Legal Services Authorities (DLSA) and Sakhi One-Stop Centres.
- **ABHERA Example**: Defined in `data/support_services.csv` and retrieved via `backend/app/services/support_mapping.py`.

### Q12: Is ABHERA a mobile app or a web app?
- **Beginner Answer**: It is a responsive web application that works on both mobile phones and desktop computers through any web browser.
- **Technical Explanation**: It is a cloud-hosted Single Page Application (SPA) built with React 18, Vite, and Tailwind CSS, communicating with a FastAPI REST backend. Its responsive CSS grid and flexbox layout adapt to any viewport size without requiring native mobile installation.
- **ABHERA Example**: Hosted publicly at `https://abhera.vercel.app`.

### Q13: What dataset was used to train the machine learning models?
- **Beginner Answer**: A curated dataset of 1,086 realistic women's safety and harassment incident narratives categorized across Indian criminal law dimensions.
- **Technical Explanation**: An audited dataset of 1,086 multi-label English records partitioned into 760 Train (70.0%), 162 Validation (14.9%), and 164 Test (15.1%) splits using fixed random seed 42 (`ml/preprocessing/`).
- **ABHERA Example**: The distribution is documented in `ml/evaluation/FINAL_EVALUATION_RESULTS.json`.

### Q14: How accurate is ABHERA's machine learning model?
- **Beginner Answer**: It is over 98% accurate, meaning out of 100 incident disclosures, it correctly identifies almost every single legal offense.
- **Technical Explanation**: Promoted BERT v2 achieved a **Micro-F1 of 98.09%**, **Macro-F1 of 97.58%**, **Exact Match Accuracy of 95.73%**, and **Multi-Label Exact Match of 94.12%** across 164 held-out test cases, with only 1 False Negative and 6 False Positives out of 984 total binary decisions (99.29% elementwise accuracy).
- **ABHERA Example**: The single false negative was `TEST_0050` where an indirect stalking narrative missed the secondary sexual harassment flag by 0.0451.

### Q15: What happens if the backend server stops running?
- **Beginner Answer**: The frontend webpage will stay visible on the user's phone, but when they try to send a message, it will show an error saying the backend is offline.
- **Technical Explanation**: Because the architecture is completely decoupled, the React frontend hosted on Vercel remains accessible. However, Axios API requests to `POST /api/chat/message` will fail with an HTTP network error or `502 Bad Gateway` from the Cloudflare Tunnel.
- **ABHERA Example**: The frontend `LoadingIndicator` catches the Axios exception and renders: *"Unable to connect to backend server. Please verify network connection or server status."*

---

## SECTION B: Programming Fundamentals (Q16 – Q27)

### Q16: What is the difference between synchronous and asynchronous code in Python?
- **Beginner Answer**: Synchronous code does one task at a time and freezes while waiting. Asynchronous code can pause a waiting task (like waiting for a database response) and work on other tasks in the meantime.
- **Technical Explanation**: Synchronous execution blocks the current thread until an I/O operation returns. Asynchronous programming in Python uses an event loop (`asyncio`) where tasks yield execution control (`await`) during non-blocking socket I/O, allowing a single thread to handle thousands of concurrent network connections.
- **ABHERA Example**: FastAPI route handlers utilize asynchronous concurrency, allowing incoming user chat requests to be received without blocking on ongoing database transactions.

### Q17: What are Python decorators and how are they used in FastAPI?
- **Beginner Answer**: A decorator is a special tag prefixed with `@` that wraps around a function to give it extra powers without modifying its internal code.
- **Technical Explanation**: A decorator is a higher-order function that takes another function as an argument, extends or alters its behavior, and returns a new callable. In FastAPI, route decorators register path endpoints with the internal URL router.
- **ABHERA Example**: In `backend/app/routes/chat.py`:
  ```python
  @router.post("/message", response_model=ChatMessageResponse)
  def send_chat_message(...):
  ```
  The `@router.post` decorator registers the function to handle HTTP POST requests at `/api/chat/message` and binds Pydantic response serialization.

### Q18: What is Pydantic and why is it used?
- **Beginner Answer**: Pydantic is a data security tool in Python that makes sure incoming data matches the exact format and types required. If someone sends text where a number is expected, Pydantic catches it immediately.
- **Technical Explanation**: Pydantic is a data parsing, validation, and serialization library based on Python type annotations. It enforces strict schema compliance at runtime, casting types and generating detailed validation error messages (`422 Unprocessable Entity`) if constraints are violated.
- **ABHERA Example**: In `backend/app/schemas/chat.py`:
  ```python
  class ChatMessageRequest(BaseModel):
      session_id: str
      message: str
  ```

### Q19: What is dependency injection and how does FastAPI implement it?
- **Beginner Answer**: Instead of a function creating its own tools (like opening a new database connection every time), the system automatically creates the tool and hands it to the function when needed.
- **Technical Explanation**: Dependency Injection (DI) is a software design pattern where an object receives its dependencies from an external assembler. FastAPI implements DI via the `Depends()` keyword in endpoint function signatures, automatically managing lifecycle setup and teardown.
- **ABHERA Example**: In `backend/app/routes/report.py`:
  ```python
  def generate_report_endpoint(payload: ReportGenerateRequest, db: Session = Depends(get_db)):
  ```
  FastAPI calls `get_db()`, yields a clean SQLAlchemy session, executes the route, and closes the session in the `finally:` block.

### Q20: What is a Python Virtual Environment (`venv`)?
- **Beginner Answer**: It is an isolated sandbox folder on your computer where you install specific library versions for one project without affecting other projects.
- **Technical Explanation**: A virtual environment is an isolated directory tree containing a dedicated Python binary and independent `site-packages` directory. This prevents version conflicts between competing library dependencies across different applications.
- **ABHERA Example**: ABHERA's dedicated virtual environment is located at `D:\Abhera(Mini)\backend\venv`, containing pinned versions of `torch 2.14.0`, `transformers 5.17.0`, and `fastapi 0.141.1`.

### Q21: What is the purpose of `__init__.py` in Python?
- **Beginner Answer**: It tells Python that a folder should be treated as a code package so other files can import functions from it.
- **Technical Explanation**: In Python, a directory containing an `__init__.py` file is recognized as a regular package, allowing dotted module imports (e.g., `from app.services import chat_service`). It can also define package-level initialization and specify `__all__` exports.
- **ABHERA Example**: In `backend/app/models/__init__.py`, all 11 SQLAlchemy models are imported and exported in `__all__ = ["User", "Session", "Law", ...]`.

### Q22: What is the difference between `==` and `is` in Python?
- **Beginner Answer**: `==` checks if two things have the same value (like two identical \$10 bills). `is` checks if they are the exact same physical object in memory.
- **Technical Explanation**: `==` invokes the `__eq__()` magic method to evaluate value equality. `is` evaluates reference identity, returning `True` only if `id(a) == id(b)` (both variables point to the exact same memory address).
- **ABHERA Example**: In `backend/app/services/chat_service.py`: `if cls._bert_predictor is None:` checks whether the singleton model pointer has been allocated yet.

### Q23: What are Python Type Hints and how do they benefit ABHERA?
- **Beginner Answer**: Type hints are labels you add to variables (like saying a variable must be a `str` or an `int`) so that developers and tools catch mistakes before the program runs.
- **Technical Explanation**: Introduced in PEP 484, type annotations allow static analysis tools (MyPy) and runtime engines (Pydantic, FastAPI) to validate function contracts, generate OpenAPI documentation, and provide IDE autocompletion.
- **ABHERA Example**: `def process_message(db: DBSession, session_id: str, user_message_text: str) -> Dict[str, Any]:`

### Q24: What is the Global Interpreter Lock (GIL) in Python?
- **Beginner Answer**: A built-in lock in standard Python that ensures only one CPU thread executes Python instructions at any given microsecond.
- **Technical Explanation**: The GIL is a mutex used by the CPython interpreter to synchronize thread execution and prevent race conditions in memory management (reference counting). While it restricts CPU-bound multithreading in pure Python, C-extension libraries like PyTorch and NumPy release the GIL during heavy tensor matrix multiplication.
- **ABHERA Example**: When PyTorch computes transformer attention inside `models/bert_multilabel_experiment_v2/`, the underlying C++ libraries bypass the Python GIL and utilize multiple CPU cores.

### Q25: How does JavaScript handle asynchronous code (Promises, async/await)?
- **Beginner Answer**: Instead of freezing the screen while waiting for the server to reply, JavaScript creates a "Promise" (a promise to deliver the data later) and continues letting the user scroll and click.
- **Technical Explanation**: JavaScript runs on a single-threaded non-blocking event loop. Asynchronous operations (like Axios HTTP calls) are delegated to the browser's Web APIs. When the network response returns, a callback is pushed to the Microtask Queue and executed by the event loop via `async`/`await` syntactic sugar.
- **ABHERA Example**: In `frontend/src/services/api.js`:
  ```javascript
  export const startSession = async () => {
    const response = await api.post('/api/session/start', {});
    return response.data;
  };
  ```

### Q26: What is the difference between `let`, `const`, and `var` in JavaScript?
- **Beginner Answer**: `const` is for variables that never change. `let` is for variables that can change. `var` is an old, outdated way that causes bugs.
- **Technical Explanation**: `var` is function-scoped and hoisted to the top of its scope. `let` and `const` are block-scoped (constrained to `{ ... }`) and temporal-dead-zone safe. `const` prevents re-assignment of the variable identifier.
- **ABHERA Example**: Throughout `frontend/src/pages/Report.jsx`, `const doc = new jsPDF(...)` guarantees the PDF document instance is immutable.

### Q27: What is JSON and why is it used everywhere in ABHERA?
- **Beginner Answer**: JSON (JavaScript Object Notation) is a simple, human-readable text format for sending structured data (like key-value pairs and lists) between different programming languages.
- **Technical Explanation**: JSON is a lightweight, language-agnostic data-interchange format. Because both Python and JavaScript have native, lightning-fast JSON parsers, it serves as the universal wire format for REST API communication and relational storage in PostgreSQL `JSON` columns.
- **ABHERA Example**: `{"session_id": "SESS-123", "predicted_labels": ["DV", "OV"]}`.

---

## SECTION C: Frontend Questions (Q28 – Q39)

### Q28: What is React and what is a Single Page Application (SPA)?
- **Beginner Answer**: React is a tool for building interactive websites. An SPA is a website that loads only once; when you click around, it swaps out content instantly without reloading the entire webpage.
- **Technical Explanation**: React is a declarative component-based UI library developed by Meta. A Single Page Application serves a single `index.html` file; page navigation and DOM updates are handled entirely on the client side via JavaScript without round-trip HTML requests to a web server.
- **ABHERA Example**: When a user transitions from `/chatbot` to `/report`, the screen updates instantly without a browser page reload.

### Q29: What is the Virtual DOM and how does it work?
- **Beginner Answer**: It is a lightweight copy of the webpage kept in computer memory. React makes changes to this copy first, finds the fastest way to update the real screen, and only updates the parts that actually changed.
- **Technical Explanation**: The Virtual DOM is an in-memory representation of the real DOM tree. When component state changes, React constructs a new Virtual DOM tree, runs a tree reconciliation algorithm ("diffing"), and batches the minimal set of real DOM operations to maximize rendering performance (60 FPS).
- **ABHERA Example**: When a new chat bubble appears in `Chatbot.jsx`, only that single message `<div>` is inserted into the real DOM; the navigation bar, emblem, and chat input are not re-rendered.

### Q30: What is the React Context API and why is it used in ABHERA?
- **Beginner Answer**: It is a global storage box where any component in the app can grab or save data (like session IDs and chat messages) without having to pass it down through parent and child components one by one.
- **Technical Explanation**: The Context API provides a way to pass data through the component tree without having to pass props down manually at every level ("prop drilling"). It implements a Publisher-Subscriber pattern where child components re-render when the context value changes.
- **ABHERA Example**: `ConversationContext.jsx` provides `sessionId`, `messages`, `predictedLabels`, and `currentQuestion` to all views (`Chatbot.jsx`, `Report.jsx`).

### Q31: What is the difference between `useState` and `useEffect`?
- **Beginner Answer**: `useState` is used to remember data that changes on the screen (like what the user is typing). `useEffect` is used to run an action automatically when something happens (like fetching data when a page first opens).
- **Technical Explanation**: `useState` declares a state variable and a setter function that triggers component re-rendering upon modification. `useEffect` declares a side-effect callback that executes after component mounting, updating, or unmounting, governed by a dependency array.
- **ABHERA Example**: In `frontend/src/pages/Chatbot.jsx`:
  ```jsx
  const [inputText, setInputText] = useState(''); // Tracks user keystrokes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); // Scrolls on new message
  }, [messages]);
  ```

### Q32: Why does ABHERA use `sessionStorage` instead of `localStorage`?
- **Beginner Answer**: `localStorage` keeps data forever until manually deleted, which could allow an abusive partner or family member to see what the user did days later. `sessionStorage` automatically deletes everything the second the browser tab is closed.
- **Technical Explanation**: `localStorage` persists data across browser sessions indefinitely on disk. `sessionStorage` maintains a separate storage area for each given origin that is scoped strictly to the lifetime of the browser tab. When the tab is closed, the storage is purged from RAM, preventing forensic discovery.
- **ABHERA Example**: In `frontend/src/context/ConversationContext.jsx`:
  ```javascript
  sessionStorage.setItem('abhera_session_id', id);
  ```

### Q33: How does client-side PDF generation work with `jsPDF`?
- **Beginner Answer**: Instead of asking a server to create a PDF file and download it, JavaScript uses mathematical drawing commands directly inside the user's browser to draw lines, boxes, and text into a PDF in memory.
- **Technical Explanation**: `jsPDF` is a client-side vector graphics and PDF document compiler. It writes raw PDF 1.7 specification byte streams in browser RAM, constructing page streams with precise coordinates (`pt`), fonts, line wraps, and color fills, triggering an automatic browser blob download (`doc.save()`).
- **ABHERA Example**: In `frontend/src/pages/Report.jsx`:
  ```javascript
  const doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  doc.setFillColor(30, 58, 138); // Deep navy header
  doc.rect(0, 0, 595, 70, 'F');
  doc.text("OFFICIAL INCIDENT REPORT & EVIDENTIARY DOSSIER", 40, 42);
  doc.save("ABHERA_Incident_Report.pdf");
  ```

### Q34: What is Vite and why was it chosen over Webpack / Create-React-App?
- **Beginner Answer**: Vite is a modern development tool that starts your website in less than a second, while older tools like Webpack take nearly a minute to load.
- **Technical Explanation**: Vite leverages native browser ES Modules (ESM) and an ultra-fast Go-based compiler (`esbuild`) during development, serving unbundled source files on demand. For production, it uses Rollup for tree-shaking and minification, achieving dramatically faster build times than Webpack.
- **ABHERA Example**: Configured in `frontend/vite.config.js`, starting local dev server on `http://localhost:5173`.

### Q35: What is Tailwind CSS and what are its benefits?
- **Beginner Answer**: It is a collection of pre-made styling tags (classes) you can add directly to your HTML (like `text-center`, `p-4`, `bg-blue-600`) so you don't have to write thousands of lines of custom CSS.
- **Technical Explanation**: Tailwind is a utility-first CSS framework that compiles down to a minimal, purged production stylesheet by scanning source code for used class names, preventing CSS bloat and enforcing a consistent design system.
- **ABHERA Example**: `<div className="flex flex-col h-full bg-slate-50 p-6 rounded-xl shadow-md">`

### Q36: How does the auto-resizing chat textarea work?
- **Beginner Answer**: As you type more lines of text, the box stretches downward automatically so you can read everything without struggling with a tiny scrollbar.
- **Technical Explanation**: A React `ref` listens to the textarea's `input` event, resets `textarea.style.height = 'auto'`, and recalculates `textarea.style.height = textarea.scrollHeight + 'px'`, bounded by a maximum CSS height.
- **ABHERA Example**: Implemented in `frontend/src/components/ChatInput.jsx`.

### Q37: How does ABHERA handle network errors gracefully on the frontend?
- **Beginner Answer**: If the server goes down, the app doesn't crash; instead, it shows a gentle notification asking the user to check their connection and try again.
- **Technical Explanation**: Axios API calls are wrapped in `try/catch` blocks. When an error is caught (e.g., `ERR_NETWORK`), the catch block extracts `error.response?.data?.detail` or returns a fallback error string, setting `error: str` in state and displaying an alert banner.
- **ABHERA Example**: In `frontend/src/pages/Chatbot.jsx`:
  ```jsx
  try {
    await sendChatMessage(sessionId, text);
  } catch (err) {
    setError("Unable to reach ABHERA server. Please ensure the backend is running.");
  }
  ```

### Q38: What are Lucide React icons and why are they used?
- **Beginner Answer**: They are clean, modern vector icons (like shield, download, phone, and info symbols) that make the app look professional, trustworthy, and easy to navigate.
- **Technical Explanation**: Lucide React is an open-source, highly customizable SVG icon library with zero runtime dependencies. Icons are rendered as lightweight SVG DOM elements that scale crisply across high-DPI screens without pixelation.
- **ABHERA Example**: `import { Shield, FileText, Phone, Download } from 'lucide-react';`

### Q39: What is CORS and why did you have to configure it?
- **Beginner Answer**: Web browsers block a website from talking to a different server unless that server explicitly gives permission. We configured CORS so our frontend on port 5173 is allowed to talk to our backend on port 8000.
- **Technical Explanation**: Cross-Origin Resource Sharing (CORS) is a browser security mechanism that uses HTTP headers to restrict resources on a web page from being requested from another domain. Because Vite runs on `localhost:5173` and FastAPI on `localhost:8000` (different ports $\implies$ different origins), the browser sends preflight `OPTIONS` requests that FastAPI must authorize via `CORSMiddleware`.
- **ABHERA Example**: In `backend/app/config.py`: `CORS_ORIGINS = ["http://localhost:5173", "https://abhera.vercel.app"]`.

---

## SECTION D: Backend and REST API Questions (Q40 – Q54)

### Q40: What is a REST API and what does it stand for?
- **Beginner Answer**: REST stands for **Representational State Transfer**. It is a standard set of rules that allows web applications to talk to backend servers over the internet by exchanging simple text messages formatted as JSON.
- **Technical Explanation**: REST is a stateless, client-server, cacheable architectural style utilizing standard HTTP methods (`GET`, `POST`, `PUT`, `DELETE`). Resources are identified by URIs, and representations are exchanged via MIME types like `application/json`.
- **ABHERA Example**: `POST /api/chat/message` accepts a JSON message and returns structured bot replies.

### Q41: Why did you choose FastAPI over Flask or Django?
- **Beginner Answer**: FastAPI is much faster, automatically checks that all data is correct using type hints, handles many users at once using asynchronous code, and writes its own web documentation automatically.
- **Technical Explanation**: FastAPI offers native ASGI asynchronous concurrency, automatic OpenAPI/Swagger documentation generation, dependency injection, and Pydantic v2 data validation that executes up to 5x faster than traditional serialization libraries.
- **ABHERA Example**: FastAPI automatically generates interactive documentation accessible at `http://localhost:8000/docs`.

### Q42: What is Uvicorn?
- **Beginner Answer**: Uvicorn is the server engine that runs our Python FastAPI application so it can listen for web traffic from the internet.
- **Technical Explanation**: Uvicorn is a lightning-fast ASGI (Asynchronous Server Gateway Interface) web server implementation for Python, built on `uvloop` (an ultra-fast C-based event loop) and `httptools`.
- **ABHERA Example**: Started via: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.

### Q43: What HTTP status codes are used in ABHERA?
- **Beginner Answer**: `200 OK` means success; `400 Bad Request` means missing data; `404 Not Found` means item doesn't exist; `500 Server Error` means something broke on the server.
- **Technical Explanation**: Standardized RFC 9110 status codes:
  - `200 OK`: Successful intake, session retrieval, or report generation.
  - `400 Bad Request`: Empty `session_id` or whitespace-only message payload.
  - `404 Not Found`: Non-existent `session_id` or `submission_id`.
  - `422 Unprocessable Entity`: Pydantic schema validation failure.
  - `500 Internal Server Error`: Unhandled backend exception.
- **ABHERA Example**: Raised in `backend/app/routes/chat.py`: `raise HTTPException(status_code=400, detail="Message content cannot be empty.")`.

### Q44: What is the purpose of the `/health` endpoint?
- **Beginner Answer**: It is a quick check to see if the backend is awake and working properly.
- **Technical Explanation**: A lightweight, unauthenticated health-check endpoint that returns system status without invoking heavy ML models or database queries. Used by cloud orchestrators, uptime monitors, and frontend pre-flight checks.
- **ABHERA Example**: In `backend/app/routes/health.py`: returns `{"status": "online", "service": "FastAPI Backend"}`.

### Q45: How does session handling work in ABHERA?
- **Beginner Answer**: When you first visit, the backend gives you a unique session code. All your messages and answers are tied to this code in the database so the system remembers your conversation.
- **Technical Explanation**: Stateless REST requests are correlated using an application-level `session_id` (format: `SESS-<uuid[:8]>`) stored in the `sessions` table. The `SessionService` manages creation, state transitions, and retrieval.
- **ABHERA Example**: `POST /api/session/start` creates a new session in PostgreSQL with status `ACTIVE`.

### Q46: What are the 4 states in ABHERA's conversation state machine?
- **Beginner Answer**: `ACTIVE` (started), `QUESTIONING` (asking follow-up questions), `PROCESSING` (analyzing), and `COMPLETED` (done and ready for report).
- **Technical Explanation**: A finite state machine (FSM) defined in `backend/app/models/session.py` as `ALLOWED_CONVERSATION_STATUSES = {"ACTIVE", "QUESTIONING", "PROCESSING", "COMPLETED"}`.
- **ABHERA Example**: Transitioning to `COMPLETED` signals `Chatbot.jsx` to render the "View Report" button.

### Q47: What is the `Depends(get_db)` pattern in FastAPI?
- **Beginner Answer**: It is an automatic helper that gives the database connection to the function when a request arrives, and automatically closes it when the request is finished.
- **Technical Explanation**: A generator-based dependency injection function that yields a database session from `SessionLocal()` and guarantees session termination in a `finally` block, preventing database connection leaks.
- **ABHERA Example**: In `app/routes/session.py`: `db: DBSession = Depends(get_db)`.

### Q48: How does the Anti-Hallucination Service validate data?
- **Beginner Answer**: Before any legal section or helpline is added to a report, the service checks whether that exact section exists in our PostgreSQL database. If it doesn't match a real row, it is blocked.
- **Technical Explanation**: The service executes exact-match primary-key and foreign-key queries against the `laws` and `support_services` tables, verifying that `act_name`, `section_number`, and `applicable_label` match verbatim.
- **ABHERA Example**: In `backend/app/services/anti_hallucination.py`, `validate_legal_record()` rejects any section not empirically in the database.

### Q49: What is the Report Generator Service?
- **Beginner Answer**: A program that takes all the facts from the conversation, organizes them into 10 neat sections, adds the right laws and phone numbers, and formats them into a final report.
- **Technical Explanation**: A deterministic, non-generative document compiler (`backend/app/services/report_generator.py`) that aggregates the raw narrative, ML classifications, extracted entities, questionnaire answers, and anti-hallucination validated statutory records into an immutable JSON dictionary.
- **ABHERA Example**: Assembles Sections 1 through 10 and writes the result to the `reports` table.

### Q50: How does ABHERA prevent conversational loops in the chatbot?
- **Beginner Answer**: The bot keeps a checklist of questions it has already asked. Once a question is asked or answered, it is crossed off and can never be asked again.
- **Technical Explanation**: The `DynamicQuestionEngine` tracks answered question IDs in `submission_question_answers` and in the `sessions.answers` JSON array. The `question_selector.py` filters candidates with `candidate_id not in asked_ids`, mathematically guaranteeing loop termination.
- **ABHERA Example**: In `backend/app/question_engine/question_selector.py`: `if q.question_id in previously_asked_ids: continue`.

### Q51: How are models loaded into memory on the backend?
- **Beginner Answer**: They are loaded once when the server starts up and kept in memory so they are instantly ready to answer questions without reloading every time.
- **Technical Explanation**: Models are implemented using the **Singleton Pattern** in `ChatService`. The `get_bert_predictor()` method instantiates `BERTIncidentPredictor` once upon first invocation and caches the reference in a class variable (`cls._bert_predictor`), avoiding multi-second reload penalties per turn.
- **ABHERA Example**: In `backend/app/services/chat_service.py`: `if cls._bert_predictor is None: cls._bert_predictor = BERTIncidentPredictor()`.

### Q52: What is an ORM (Object-Relational Mapping)?
- **Beginner Answer**: An ORM lets you talk to your database using normal Python objects instead of writing raw SQL text by hand.
- **Technical Explanation**: An ORM maps relational database tables to object-oriented classes. Instances of the class represent table rows, and attributes represent columns. SQLAlchemy translates Python operations (`session.query(Law).filter(...)`) into optimized SQL queries.
- **ABHERA Example**: `law = db.query(Law).filter(Law.applicable_label == "ST").all()`.

### Q53: How does the backend handle exceptions?
- **Beginner Answer**: Any unexpected problem is caught by a safety net (`try/catch`), logged into a file for developers to see, and a polite error message is sent back to the user without crashing the server.
- **Technical Explanation**: Route handlers wrap business logic in `try/except` blocks. Expected errors raise `HTTPException(status_code, detail)`. Unhandled exceptions are logged with full stack traces (`exc_info=True`) and caught by global exception handlers returning `500 Internal Server Error`.
- **ABHERA Example**: In `backend/app/routes/report.py`:
  ```python
  except Exception as e:
      logger.error(f"Error generating report: {e}", exc_info=True)
      raise HTTPException(status_code=500, detail="Internal error generating report.")
  ```

### Q54: What is the purpose of `data_importer.py`?
- **Beginner Answer**: It is a setup script that reads our CSV files containing 32 laws and 12 helplines and loads them cleanly into PostgreSQL.
- **Technical Explanation**: An idempotent ETL utility (`backend/app/services/data_importer.py`) that validates CSV columns, clears legacy knowledge tables, and imports verified rows into the `laws` and `support_services` tables.
- **ABHERA Example**: Executed via: `python -m app.services.data_importer`.

---

## SECTION E: Database and SQL Questions (Q55 – Q66)

### Q55: What is PostgreSQL and how does it differ from MySQL?
- **Beginner Answer**: Both are popular databases, but PostgreSQL is known for strict data accuracy, advanced features (like JSON storage), and adherence to international SQL standards.
- **Technical Explanation**: PostgreSQL is an advanced object-relational database emphasizing extensibility, ACID compliance, complex queries, and native JSONB indexing, whereas MySQL was historically optimized for simple read-heavy web applications.
- **ABHERA Example**: ABHERA uses PostgreSQL 16 to leverage native `JSON` columns for storing conversation messages and entities.

### Q56: What is a Primary Key and Foreign Key?
- **Beginner Answer**: A Primary Key is a unique ID badge for every single row in a table. A Foreign Key is a link in one table that points to the ID badge in another table.
- **Technical Explanation**: A Primary Key (PK) uniquely identifies each record in a relational table and enforces uniqueness and non-null constraints. A Foreign Key (FK) establishes a referential constraint between a column in a child table and the Primary Key of a parent table.
- **ABHERA Example**: In `incident_submissions`, `id` is the Primary Key, while `session_id` is a Foreign Key referencing `sessions.session_id`.

### Q57: What does `ON DELETE CASCADE` mean?
- **Beginner Answer**: If you delete a parent item (like a user account), all the related child items (like their messages and reports) are automatically cleaned up and deleted too so no junk is left behind.
- **Technical Explanation**: A referential action on a foreign key constraint. When a row in the parent table is deleted, PostgreSQL automatically deletes all referencing child rows in the dependent table, maintaining referential integrity.
- **ABHERA Example**: `ForeignKey("incident_submissions.submission_id", ondelete="CASCADE")` in `conversation_messages`.

### Q58: Why does the `sessions` table use a JSON column for messages?
- **Beginner Answer**: Because a chat conversation has many back-and-forth messages that are easiest to read and write all together in a single quick package.
- **Technical Explanation**: Storing the message sequence as a `JSON` array inside `sessions` enables atomic, single-query retrieval of the entire conversation history during active chat turns, avoiding repeated expensive multi-table joins.
- **ABHERA Example**: `messages = Column(JSON, nullable=True, default=list)`.

### Q59: What is database normalization?
- **Beginner Answer**: It is organizing database tables neatly so that data is not repeated unnecessarily, saving space and preventing mistakes.
- **Technical Explanation**: A database design technique that organizes tables to reduce data redundancy and improve data integrity (typically up to Third Normal Form - 3NF) by separating distinct entities into related tables.
- **ABHERA Example**: Legal statutory text is stored once in the `laws` table, rather than duplicating the entire text inside every incident submission row.

### Q60: What is an index in a database and which columns are indexed in ABHERA?
- **Beginner Answer**: An index is like the alphabetical index at the back of a textbook. Instead of scanning through every single page, the computer jumps straight to the page you want.
- **Technical Explanation**: A B-Tree data structure maintained on a table column that allows search queries to execute in $O(\log N)$ logarithmic time instead of $O(N)$ full table scans.
- **ABHERA Example**: B-Tree indexes exist on `laws.applicable_label`, `sessions.session_id`, `incident_submissions.submission_id`, and `users.user_guid`.

### Q61: What are ACID properties in a database?
- **Beginner Answer**: Four safety rules: **A**ll-or-nothing (Atomicity), **C**orrect rules followed (Consistency), **I**solated transactions (Isolation), and **D**urable permanent saves (Durability).
- **Technical Explanation**:
  - **Atomicity**: Transactions execute completely or abort entirely.
  - **Consistency**: Database transitions only between valid states satisfying constraints.
  - **Isolation**: Concurrent transactions execute as if sequential.
  - **Durability**: Committed data survives system crashes and power loss.
- **ABHERA Example**: When saving a report, if an error occurs mid-write, SQLAlchemy triggers `db.rollback()`, ensuring no partial records are written.

### Q62: What is the schema of the `laws` table?
- **Beginner Answer**: It has columns for the Act Name (e.g., BNS), Section Number (e.g., Section 78), the official Legal Text, and the Harassment Code it applies to (e.g., ST).
- **Technical Explanation**: Defined in `backend/app/models/law.py`: `id` (Integer PK), `act_name` (VARCHAR 100), `section_number` (VARCHAR 50), `section_text` (TEXT), `applicable_label` (VARCHAR 50), `created_at` (TIMESTAMP).
- **ABHERA Example**: A row has `act_name = "BNS"`, `section_number = "Section 78"`, `applicable_label = "ST"`.

### Q63: How is a final report stored in PostgreSQL?
- **Beginner Answer**: The complete 10-section report is saved as a structured JSON document in the `reports` table linked to the submission ID.
- **Technical Explanation**: The `reports` table contains a unique Foreign Key `submission_id` and an immutable `report_data` JSON column containing all 10 assembled sections.
- **ABHERA Example**: `new_report = Report(submission_id=submission_id, report_data=report_dict)`.

### Q64: Write the SQL query to retrieve all cyber harassment laws in the database.
- **Beginner Answer**: `SELECT * FROM laws WHERE applicable_label = 'CA';`
- **Technical Explanation**:
  ```sql
  SELECT id, act_name, section_number, section_text, applicable_label
  FROM laws
  WHERE applicable_label = 'CA'
  ORDER BY id ASC;
  ```
- **ABHERA Example**: Returns IT Act Section 66E, 67, and 67A.

### Q65: What is connection pooling in SQLAlchemy?
- **Beginner Answer**: Keeping a group of open database connections ready to use so the server doesn't waste time opening and closing a new connection for every single click.
- **Technical Explanation**: A cache of active database connections maintained by `create_engine` (`pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`) that recycles TCP connections to PostgreSQL, eliminating socket handshake overhead.
- **ABHERA Example**: Configured in `backend/app/database.py`.

### Q66: What is the difference between `autocommit=False` and `autocommit=True`?
- **Beginner Answer**: `autocommit=True` saves every single change instantly without asking. `autocommit=False` waits until you explicitly say "commit", letting you cancel if something goes wrong.
- **Technical Explanation**: In `autocommit=False`, all database modifications occur within an explicit transaction block. Changes are permanent only when `db.commit()` is called; if an exception occurs, `db.rollback()` undoes all staged modifications.
- **ABHERA Example**: In `SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)`.

---

## SECTION F: Machine Learning Fundamentals (Q67 – Q81)

### Q67: What is supervised learning?
- **Beginner Answer**: Teaching a computer by showing it examples that already have the correct answers attached (like showing flashcards with pictures of animals and their names on the back).
- **Technical Explanation**: A machine learning paradigm where an algorithm learns a mapping function $f: X \to Y$ from labeled training pairs $(x_i, y_i)$, optimizing parameters to minimize an empirical loss function.
- **ABHERA Example**: Training BERT on 760 pairs of `(incident_narrative_text, ground_truth_harassment_labels)`.

### Q68: What is the difference between training, validation, and testing sets?
- **Beginner Answer**:
  - Training set = Textbooks you study from.
  - Validation set = Practice exams you take to tune your study habits.
  - Testing set = Final board exam you take once at the end without looking at answers.
- **Technical Explanation**:
  - Training set (70%): Used to compute gradients and update model weights via backpropagation.
  - Validation set (15%): Used to evaluate generalization during training, trigger early stopping, and tune hyper-parameters (decision thresholds).
  - Testing set (15%): Held out completely; used only for unbiased evaluation of the final promoted model.
- **ABHERA Example**: In `ml/preprocessing/`: 760 train, 162 val, 164 test records (`seed = 42`).

### Q69: What is overfitting and how do you prevent it?
- **Beginner Answer**: Overfitting is when a student memorizes the exact practice questions by heart instead of understanding the concepts. When given a slightly different question on the real exam, they fail.
- **Technical Explanation**: Overfitting occurs when a model learns noise and specific idiosyncrasies of the training data rather than underlying patterns, resulting in low training loss but high validation loss. Prevented by dropout layers ($p=0.1$), weight decay ($L_2$ regularization = 0.01), early stopping, and small learning rates ($3 \times 10^{-5}$).
- **ABHERA Example**: Monitored in `ml/training/train_bert_experiment_v2.py` across 6 epochs; training was stopped when validation F1 peaked at epoch 6.

### Q70: What is Binary Cross-Entropy Loss?
- **Beginner Answer**: A mathematical penalty score that measures how wrong the model was on a yes-or-no question. If the model was confident and wrong, the penalty is huge.
- **Technical Explanation**: The standard loss function for binary and multi-label classification:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
  where $y_i \in \{0, 1\}$ is the ground-truth and $\hat{y}_i \in [0, 1]$ is the predicted probability.
- **ABHERA Example**: Used in PyTorch via `torch.nn.BCEWithLogitsLoss`.

### Q71: What is the Sigmoid activation function and why is it used in multi-label classification?
- **Beginner Answer**: It is a mathematical curve that squashes any positive or negative number into a clean percentage probability between 0% and 100%.
- **Technical Explanation**: The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps real-valued logits $(-\infty, +\infty)$ monotonically into $[0, 1]$. In multi-label classification, it is applied independently to each output logit so that each class probability is evaluated independently.
- **ABHERA Example**: In `predict_bert.py`: `probabilities = torch.sigmoid(logits).cpu().numpy()`.

### Q72: What is Precision and Recall?
- **Beginner Answer**:
  - **Precision**: When the model flags Domestic Violence, how often is it actually right?
  - **Recall**: Out of all real Domestic Violence cases that occurred, how many did the model successfully catch?
- **Technical Explanation**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **ABHERA Example**: On the test set, ABHERA achieved **96.77% Micro Precision** and **99.45% Micro Recall**.

### Q73: What is the F1-Score and why is it preferred over simple accuracy?
- **Beginner Answer**: If 95% of people are healthy and 5% have a disease, a dumb model that says "everyone is healthy" gets 95% accuracy while being completely useless. F1-Score combines precision and recall so cheating the test is impossible.
- **Technical Explanation**: F1-score is the harmonic mean of precision and recall:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  It gives a reliable metric on imbalanced datasets because it penalizes models that sacrifice recall for precision or vice versa.
- **ABHERA Example**: ABHERA v2 achieved a **Micro-F1 of 98.09%**.

### Q74: What is the difference between Micro-F1 and Macro-F1?
- **Beginner Answer**: Micro-F1 counts all right and wrong answers together in one big pile. Macro-F1 calculates the score for each category separately and takes the average, giving equal importance to rare categories.
- **Technical Explanation**:
  - **Micro-F1**: Aggregates globally across all classes ($\sum TP$, $\sum FP$, $\sum FN$) before computing F1; weighted by class frequency.
  - **Macro-F1**: Computes F1 independently for each class and calculates the unweighted arithmetic mean; treats minority classes with equal weight to majority classes.
- **ABHERA Example**: ABHERA achieved **Micro-F1 = 98.09%** and **Macro-F1 = 97.58%**.

### Q75: What is Exact Match Accuracy?
- **Beginner Answer**: The student gets the question right only if they check every single correct box and zero wrong boxes. If they miss even one box, the entire question scores zero.
- **Technical Explanation**: Also known as Subset Accuracy. For a multi-label sample, it requires the entire predicted binary vector $\mathbf{\hat{y}}$ to match the ground truth vector $\mathbf{y}$ simultaneously:
  $$\text{Exact Match} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\mathbf{\hat{y}}_i == \mathbf{y}_i)$$
- **ABHERA Example**: ABHERA achieved **95.73% Exact Match Accuracy** (157 out of 164 test cases matched perfectly across all 6 labels).

### Q76: What is Multi-Label Exact Match?
- **Beginner Answer**: It is the exact match score restricted strictly to complicated cases where two or more crimes happened at the same time.
- **Technical Explanation**: Exact match accuracy evaluated solely on the subset of samples having $\sum y_{i,c} \ge 2$.
- **ABHERA Example**: BERT v1 scored **0.0%** (0 / 17 cases). ABHERA v2 achieved **94.12%** (16 out of 17 multi-label test cases matched ground truth perfectly).

### Q77: What is class imbalance and how did it affect BERT v1?
- **Beginner Answer**: When one crime has 300 examples but another has only 90, the AI gets lazy and ignores the rare crime. BERT v1 did this and missed 23 real crimes.
- **Technical Explanation**: Skewed class priors cause the loss gradient to be dominated by the majority negative class (`0`), driving output logits toward large negative numbers. In BERT v1, this caused 23 False Negatives on multi-label combinations.
- **ABHERA Example**: `OV` represented only 9.0% of the dataset; resolved in v2 using `pos_weight = 10.08`.

### Q78: How does `pos_weight` in PyTorch fix class imbalance?
- **Beginner Answer**: It acts like a multiplier that makes the penalty 10 times bigger if the AI accidentally misses a rare crime.
- **Technical Explanation**: Scales the loss contributed by positive errors by a factor $w_c = \frac{N - N_c}{N_c}$ inside `BCEWithLogitsLoss`. False negative errors produce gradients multiplied by $w_c$, forcing the optimizer to achieve high recall on minority classes.
- **ABHERA Example**: In `training_config.json`: `pos_weights = {"DV": 3.344, "ST": 5.101, "OV": 10.082}`.

### Q79: What is the AdamW optimizer?
- **Beginner Answer**: It is the mathematical algorithm that steers the AI toward the right answers during training, making sure it learns smoothly without making wild guesses.
- **Technical Explanation**: AdamW is an extension of the Adam (Adaptive Moment Estimation) optimizer that decouples weight decay ($L_2$ regularization) from the gradient momentum updates, leading to superior generalization in deep transformers.
- **ABHERA Example**: Used in `train_bert_experiment_v2.py` with base learning rate $\alpha = 3 \times 10^{-5}$ and weight decay $\lambda = 0.01$.

### Q80: What is a learning rate warmup?
- **Beginner Answer**: Starting with very small baby steps for the first few minutes so the AI doesn't get overwhelmed and wreck its pre-trained knowledge, then speeding up once it gets oriented.
- **Technical Explanation**: A learning rate scheduling technique where the learning rate increases linearly from $0$ to the maximum learning rate over a specified number of initial steps (warmup phase) before decaying. This prevents large initial gradients from destabilizing pre-trained transformer embeddings.
- **ABHERA Example**: Configured over the first 10% of training steps in `ml/training/train_bert_experiment_v2.py`.

### Q81: What are True Positives, False Positives, True Negatives, and False Negatives?
- **Beginner Answer**:
  - **True Positive (TP)**: Crime happened, AI correctly flagged it.
  - **False Positive (FP)**: Crime did NOT happen, but AI incorrectly flagged it (false alarm).
  - **True Negative (TN)**: Crime did NOT happen, AI correctly stayed quiet.
  - **False Negative (FN)**: Crime happened, but AI missed it completely (dangerous error).
- **Technical Explanation**: The four outcomes of a binary decision matrix. In safety-critical systems, False Negatives carry high human cost, whereas False Positives are easily clarified during follow-up questioning.
- **ABHERA Example**: Test results: **TP = 180, FP = 6, FN = 1, TN = 797**.
'''
