"""
Module for Chapter 13: Deployment
"""

def get_chapter_13():
    return r'''# CHAPTER 13: Deployment

---

## 13.1 Production Deployment Reality: The Free Hybrid Architecture

A crucial distinction must be stated clearly for academic and viva review:

> [!IMPORTANT]
> **Free Hybrid Architecture Disclosure**:
> ABHERA's current free live deployment is a **hybrid cloud-and-local edge setup**. 
> - The **React 18 frontend** is hosted publicly on **Vercel's Global Edge Network** (`https://abhera.vercel.app`).
> - The **FastAPI backend, PyTorch dual BERT models, and PostgreSQL database** run on the **local workstation computer**, bridged securely to the internet via a **Cloudflare Quick Tunnel** (`cloudflared`).
> - **This is NOT permanent 24/7 cloud container hosting.** If the local workstation is shut down, enters sleep mode, or closes the terminal, the tunnel collapses, and the public frontend will display an API connection error (`502 Bad Gateway` / Network Error).

### Why Pure Free-Tier Cloud Hosting is Unfeasible for BERT:
Most free-tier cloud container providers (such as Render Free, Railway Free, or Vercel Serverless Functions) impose strict memory limits of **512 MB of RAM**.
However, ABHERA runs two 110-million parameter PyTorch transformer encoders:
- BERT Multi-Label Classifier: ~480 MB in RAM
- BERT NER Token Classifier: ~475 MB in RAM
- Python Runtime & FastAPI: ~120 MB in RAM
- SQLAlchemy & OS Buffers: ~260 MB in RAM
- **Total Minimum Memory Footprint: ~1.35 GB of Host RAM**

Attempting to run ABHERA on a 512 MB free container results in an immediate `OOMKilled` (Out of Memory Kernel Panic) during model loading. A paid container with at least 2.0 GB of RAM is required for permanent 24/7 cloud hosting. The hybrid Cloudflare Tunnel setup is the standard, zero-cost engineering solution for demonstrating deep learning systems to examiners and panels.

```
+===================================================================================================+
|                                 ABHERA FREE HYBRID DEPLOYMENT TOPOLOGY                            |
+===================================================================================================+

   [ Public Web Browser / Examiner Phone ]
                     |
                     | (1) HTTPS Request to Vercel Domain
                     v
   +----------------------------------------------------+
   | CLOUD TIER: VERCEL GLOBAL EDGE CDN                 |
   | - Hosts React 18 SPA (`frontend/dist/`)            |
   | - Domain: `https://abhera.vercel.app`              |
   | - Environment Variable: `VITE_API_URL`             |
   +----------------------------------------------------+
                     |
                     | (2) REST API Calls (Axios HTTPS)
                     v
   +----------------------------------------------------+
   | CLOUDFLARE EDGE NETWORK (Inbound Proxy)            |
   | - Public Ingress: `https://*.trycloudflare.com`    |
   | - Zero Public IP Exposure / No Port Forwarding     |
   +----------------------------------------------------+
                     |
                     | (3) Outbound Encrypted Cloudflare Quick Tunnel
                     v
+===================================================================================================+
| LOCAL WORKSTATION RUNTIME (Must be Powered On during Viva / Demo)                                  |
| Location: `D:\Abhera(Mini)`                                                                       |
|                                                                                                   |
|   [ cloudflared daemon ]  <--->  [ Uvicorn ASGI Server ]  <--->  [ Dual BERT Transformers in RAM ]|
|   (Tunnel Client)                (localhost:8000)                (417MB + 415MB safetensors)      |
|                                         |                                                         |
|                                         v                                                         |
|                             [ PostgreSQL 16 Service ]                                             |
|                             (localhost:5432 / womens_safety_db)                                   |
+===================================================================================================+
```

---

## 13.2 Step-by-Step Deployment Playbook

### Step 1: Git Repository & GitHub Setup
1. Initialize Git in the project root if not already tracked.
2. Verify that sensitive files (`.env`, `backend/venv/`, `node_modules/`, `database/*.sql`) are strictly ignored in `.gitignore`.
3. Push the codebase to a GitHub repository:
```powershell
# Check git status and staged files
git status

# Commit any verified updates
git commit -m "Production release commit"

# Push to private GitHub repository
git push origin main
```

---

### Step 2: Vercel Frontend Project Setup
1. Log in to [Vercel](https://vercel.com) using your GitHub account.
2. Click **"Add New Project"** and import the `Abhera(Mini)` repository.
3. Configure Project Settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Configure Environment Variables:
   - Add `VITE_API_URL`: Set temporarily to your Cloudflare Tunnel URL (e.g., `https://example-tunnel.trycloudflare.com`).
5. Click **"Deploy"**. Vercel compiles the React application and deploys it to `https://abhera.vercel.app`.

---

### Step 3: Local Workstation Preparation & Database Verification
Before starting the backend, verify that PostgreSQL 16 is running locally:

```powershell
# 1. Start PostgreSQL Service in Windows PowerShell (if stopped)
net start postgresql-x64-16

# 2. Test TCP Connection to PostgreSQL port 5432
Test-NetConnection -Port 5432 -ComputerName localhost
```
*Expected Output*: `TcpTestSucceeded : True`.

3. Ensure all 11 database tables and initial questions are seeded:
```powershell
$env:PYTHONPATH="D:\Abhera(Mini)\backend;D:\Abhera(Mini)"
& "D:\Abhera(Mini)\backend\venv\Scripts\python.exe" -m app.db.init_db
& "D:\Abhera(Mini)\backend\venv\Scripts\python.exe" -m app.db.seed_questions
& "D:\Abhera(Mini)\backend\venv\Scripts\python.exe" -m app.services.data_importer
```

---

### Step 4: Starting the FastAPI Backend
Launch the Uvicorn ASGI application server inside the virtual environment:

```powershell
# Set Python path and launch Uvicorn
$env:PYTHONPATH="D:\Abhera(Mini)\backend;D:\Abhera(Mini)"
& "D:\Abhera(Mini)\backend\venv\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Explanation of Command Flags:
- `app.main:app`: Instructs Uvicorn to look inside module `app/main.py` and run the `app = FastAPI()` object.
- `--host 0.0.0.0`: Binds the server to all network adapters (both localhost `127.0.0.1` and local LAN IP), allowing Cloudflare Tunnel to connect.
- `--port 8000`: Specifies TCP listening port 8000.
- `--reload`: Automatically reloads the server if any Python file is edited.

---

### Step 5: Launching the Cloudflare Quick Tunnel
In a second Windows PowerShell terminal window, launch the `cloudflared` tunnel:

```powershell
# Start Cloudflare Quick Tunnel targeting the local FastAPI port
cloudflared tunnel --url http://localhost:8000
```

### Explanation of Output:
`cloudflared` connects to Cloudflare edge data centers and prints:
```
+--------------------------------------------------------------------------------------------+
| Your quick Tunnel has been created! Visit it at:                                           |
| https://random-subdomain-name.trycloudflare.com                                            |
+--------------------------------------------------------------------------------------------+
```
Copy this generated HTTPS URL.

---

### Step 6: Linking Frontend and Backend
1. In the Vercel Dashboard for the ABHERA project, navigate to **Settings $\to$ Environment Variables**.
2. Update `VITE_API_URL` with the copied Cloudflare Tunnel URL:
   ```
   VITE_API_URL=https://random-subdomain-name.trycloudflare.com
   ```
3. Click **"Save"**, then go to the **Deployments** tab and click **"Redeploy"** so Vite rebuilds the frontend with the updated API endpoint.
4. In `backend/app/config.py`, verify that `settings.CORS_ORIGINS` includes your Vercel domain or `*` for testing.

---

### Step 7: Live Production Testing
1. Open a mobile phone or secondary laptop not connected to your home Wi-Fi (e.g., using mobile 4G/5G data).
2. Open the browser and visit `https://abhera.vercel.app`.
3. Open the browser developer tools (F12) or observe the mobile screen:
   - The status indicator should show **"API Online"** (calling `GET /health` through the tunnel).
   - Enter a test narrative: *"Someone is stalking me on Instagram."*
   - Verify that the message reaches your local terminal running Uvicorn, BERT outputs predictions, and the follow-up question appears on the mobile phone screen.
   - Click "Download Official PDF Report" and verify that the vector PDF is compiled and saved on the mobile device.
'''
