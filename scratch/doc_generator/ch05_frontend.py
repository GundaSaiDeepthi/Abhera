"""
Module for Chapter 5: Frontend Development
"""

def get_chapter_05():
    return r'''# CHAPTER 5: Frontend Development

---

## 5.1 Architecture of the React Application

The frontend of ABHERA is built as a **Single Page Application (SPA)** using **React 18** and **Vite**. 

In traditional multi-page web applications, every click or form submission triggers a full browser reload, fetching a new HTML document from the server. For a traumatized survivor disclosing an intimate or violent ordeal, a reloading screen that flashes white and loses scroll position creates friction, distress, and distrust.

React solves this through its **component-based declarative architecture**. The page is split into small, reusable, isolated building blocks (components). When new data arrives from the backend (such as a predicted harassment category or a follow-up question), React calculates the exact differences using its **Virtual DOM** and updates only the necessary DOM elements on the screen in milliseconds.

```
+---------------------------------------------------------------------------------------------------+
|                                  REACT COMPONENT TREE HIERARCHY                                   |
+---------------------------------------------------------------------------------------------------+
                                            [ index.html ]
                                                  |
                                             [ main.jsx ]
                                                  |
                                      [ ConversationProvider ]
                                   (Global State in Context API)
                                                  |
                                              [ App.jsx ]
                                     (BrowserRouter & Route Engine)
                                                  |
             +--------------------+---------------+--------------------+--------------------+
             |                    |                                    |                    |
             v                    v                                    v                    v
      [ Navbar.jsx ]        [ Home.jsx ]                         [ Chatbot.jsx ]     [ Report.jsx ]
   (Persistent Header)   (Hero & Educational)                 (Conversational Intake) (Evidentiary Dossier)
                                                                       |                    |
                                                              +--------+--------+           |
                                                              |                 |           v
                                                       [ ChatMessage ]    [ ChatInput ] [ jsPDF Vector ]
                                                       (Speech Bubbles)   (Auto-resize) (Browser Engine)
```

---

## 5.2 Application Routing (`frontend/src/App.jsx`)

ABHERA implements client-side routing using **React Router v6** (`react-router-dom`). 

Client-side routing means the browser URL changes (e.g., from `/` to `/chatbot` or `/report`), but the browser does not make a network request for a new HTML page. Instead, React Router intercepts the URL change and mounts the corresponding page component dynamically.

### Actual Source Code: `frontend/src/App.jsx`
```jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Chatbot from './pages/Chatbot';
import Processing from './pages/Processing';
import Report from './pages/Report';

function App() {
  return (
    <Router>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chatbot" element={<Chatbot />} />
            <Route path="/processing" element={<Processing />} />
            <Route path="/report" element={<Report />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
```

### Line-by-Line Code Explanation:
- **Lines 1-2**: Imports the React core library and the routing primitives (`BrowserRouter as Router`, `Routes`, and `Route`) from `react-router-dom`.
- **Lines 3-7**: Imports the persistent navigation header (`Navbar`) and the four primary application views (`Home`, `Chatbot`, `Processing`, and `Report`).
- **Line 9**: Declares the root functional component `App`.
- **Line 11**: Wraps the entire application tree in `<Router>`. This provides the HTML5 History API context so child components can navigate programmatically using `useNavigate()`.
- **Line 12**: Creates a wrapper `<div>` with CSS class `app-layout` that applies a full-height flex column layout.
- **Line 13**: `<Navbar />` is placed *outside* the `<Routes>` container. This ensures that the navigation bar stays pinned to the top of the viewport across all page navigations without re-rendering.
- **Lines 15-20**: The `<Routes>` container evaluates the current browser URL and matches it against defined `<Route>` paths:
  - Path `"/"`: Displays the landing page (`Home.jsx`).
  - Path `"/chatbot"`: Displays the conversational intake interface (`Chatbot.jsx`).
  - Path `"/processing"`: Displays a transitional loading animation while ML predictions finalize (`Processing.jsx`).
  - Path `"/report"`: Displays the final evidentiary report and PDF download button (`Report.jsx`).

---

## 5.3 Global State Management (`frontend/src/context/ConversationContext.jsx`)

In React, passing data deeply through multiple levels of components (known as "prop drilling") is error-prone. ABHERA manages all session data, message histories, ML predictions, and extracted entities using the **React Context API**.

Furthermore, to protect against accidental page refreshes, the context provider synchronizes active session identifiers with the browser's native `sessionStorage`.

### Key State Properties Managed:
1. `sessionId`: The unique UUID of the active conversation session (e.g., `SESS-3f8a12bc`).
2. `submissionId`: The unique ID of the incident submission record (e.g., `SUB-9d7e41fa`).
3. `messages`: Array of chat message objects (`{ sender: 'user' | 'assistant', text: '...' }`).
4. `predictedLabels`: Array of harassment category codes returned by BERT (e.g., `['DV', 'CA']`).
5. `entities`: Dictionary of extracted named entities (e.g., `{ PERP_REL: ['husband'], PLATFORM: ['WhatsApp'] }`).
6. `currentQuestion`: The active follow-up question selected by the Dynamic Question Engine.
7. `conversationStatus`: The finite state machine status (`'ACTIVE'`, `'QUESTIONING'`, `'PROCESSING'`, `'COMPLETED'`).
8. `loading`: Boolean flag controlling UI spinners during asynchronous API calls.

### Actual Source Code Snippet:
```jsx
export const ConversationProvider = ({ children }) => {
  const [sessionId, setSessionIdState] = useState(getStoredSessionId);
  const [submissionId, setSubmissionIdState] = useState(getStoredSubmissionId);
  const [messages, setMessages] = useState([]);
  const [predictedLabels, setPredictedLabels] = useState([]);
  const [entities, setEntities] = useState({});
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [conversationStatus, setConversationStatus] = useState('ACTIVE');
  const [loading, setLoading] = useState(false);

  const setSessionId = useCallback((id) => {
    setSessionIdState(id);
    if (id) {
      sessionStorage.setItem('abhera_session_id', id);
    } else {
      sessionStorage.removeItem('abhera_session_id');
    }
  }, []);

  // ... provides state and helper functions to child components
```

---

## 5.4 REST API Service Client (`frontend/src/services/api.js`)

All asynchronous network communication between the React frontend and the FastAPI backend is managed by an **Axios HTTP client** instance configured in `frontend/src/services/api.js`.

### Why Axios is Used Instead of Standard `fetch()`:
1. **Automatic JSON Transformation**: Axios automatically parses incoming JSON response strings into JavaScript objects, eliminating the need to write `.then(res => res.json())`.
2. **Environment Base URL Configuration**: Axios allows creating a centralized client instance with a pre-configured `baseURL`.
3. **Robust Error Handling**: Axios throws structured errors for all non-2xx HTTP status codes (such as `400 Bad Request` or `500 Internal Server Error`), making it easy to display clear user-facing error notifications.

### Actual Source Code: `frontend/src/services/api.js`
```javascript
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const startSession = async () => {
  const response = await api.post('/api/session/start', {});
  return response.data;
};

export const sendChatMessage = async (sessionId, message) => {
  const response = await api.post('/api/chat/message', {
    session_id: sessionId,
    message: message,
  });
  return response.data;
};

export const generateReport = async (data) => {
  const response = await api.post('/api/report/generate', data);
  return response.data;
};

export const getReport = async (submissionId) => {
  const response = await api.get(`/api/report/${submissionId}`);
  return response.data;
};

export default api;
```

---

## 5.5 Conversational Intake View (`frontend/src/pages/Chatbot.jsx`)

The chatbot page is the core user-facing interaction environment. It is designed to be gentle, reassuring, and responsive:

1. **Auto-Scrolling Chat Window**: As messages are exchanged, a React `useRef` automatically scrolls the conversation view to the latest message.
2. **Dynamic Question Integration**: When the backend returns a follow-up question (status: `QUESTIONING`), the question is displayed with distinctive styling, and the input placeholder changes to prompt the survivor for that specific information.
3. **Completion Transition**: When all required information slots are collected, the backend sets status to `COMPLETED`. The chatbot displays a success alert and an "Analyze & Generate Report" button that routes the survivor to `/report`.

---

## 5.6 Client-Side PDF Generation (`frontend/src/pages/Report.jsx`)

A standout architectural achievement in ABHERA is **client-side vector PDF generation**. 

In most traditional web applications, PDF generation is executed on the backend using tools like wkhtmltopdf or headless Chrome (Puppeteer). However, generating incident reports on a shared server creates severe privacy vulnerabilities: temporary files containing harrowing personal disclosures sit unencrypted on server hard disks.

In ABHERA, the entire PDF is generated **directly in the survivor's browser memory** using `jsPDF`. The server never generates or stores a PDF file.

### How Client-Side Vector PDF Generation Works:
1. When the survivor clicks **"Download PDF Report"**, the `handleDownloadPDF()` function in `Report.jsx` executes.
2. An instance of `new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' })` is created.
3. The generator programmatically draws:
   - Deep navy corporate header banner (`#1e3a8a`).
   - Project emblem and title ("OFFICIAL INCIDENT REPORT & EVIDENTIARY DOSSIER").
   - 10 distinct structured sections with custom border boxes, section titles, and typography.
   - Verified statutory citations (BNS 2023, IT Act 2000) and helpline hotlines.
   - Dynamic page numbering ("Page X of Y") and confidentiality footers on every page.
4. Finally, `doc.save('ABHERA_Incident_Report.pdf')` instructs the browser to download the file directly to the user's local Downloads folder.

---

## 5.7 CSS and Responsive Design (`frontend/src/index.css`)

The application's visual styling is defined in `frontend/src/index.css` using modern CSS variables, flexbox, grid layouts, and utility classes:
- **Color Palette**: Calming deep blues (`#1e3a8a`), slate grays (`#f8fafc`, `#334155`), and supportive teals (`#0284c7`), avoiding harsh reds or high-contrast warning colors that heighten survivor stress.
- **Responsive Layout**: Fluid breakpoints ensure that textareas, buttons, and report tables automatically resize gracefully whether viewed on a 6-inch mobile smartphone or a 27-inch desktop monitor.
'''
