# Frontend Architecture - React.js Conversational Interface

## Overview
The frontend is a lightweight, accessible React.js single-page web app. It provides a chatbot experience for reporting women's safety incidents and reviewing structured legal/support assistance reports.

## Planned Structure
- `src/components/`: Conversational chat box, message bubbles, follow-up prompt cards, and final structured summary report visualizer.
- `src/services/`: API client interfacing with FastAPI REST endpoints (`/api/v1/sessions`, `/api/v1/messages`, `/api/v1/report`).
- `src/styles/`: Theme variables, responsive layouts, accessible typography, and smooth messaging transition animations.
