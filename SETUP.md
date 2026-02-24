# Setup Guide

Complete setup instructions for the RAG PDF Chatbot.

## Table of Contents

- [First-Time Setup](#first-time-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Verification](#verification)
- [Common Issues](#common-issues)

---

## First-Time Setup

### 1. Clone the Repository

```bash
git clone https://github.com/om-gupta-30/YNM-RAG-CHATBOT.git
cd YNM-RAG-CHATBOT
```

### 2. Set Up Environment Variables

```bash
make setup-env
```

This creates `.env` from `.env.example`. Now edit `.env`:

```bash
# macOS/Linux
nano .env

# Or use your preferred editor
code .env
```

Add your Gemini API key:
```
GEMINI_API_KEY=your_actual_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

### 3. Install Dependencies

**Option A: Quick install (recommended)**
```bash
make install
```

**Option B: With virtual environment (recommended for Python)**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
make install
```

**Option C: Manual install**
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 4. Verify Setup

```bash
make check-env
```

This verifies your environment variables are configured correctly.

---

## Environment Configuration

### Required Variables

| Variable | Description | Where to Get |
|----------|-------------|--------------|
| `GEMINI_API_KEY` | Google Gemini API key | [Get key](https://aistudio.google.com/app/apikey) |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Backend server port | `8000` |
| `HOST` | Backend server host | `0.0.0.0` |

### Frontend Environment Variables

The frontend uses Vite environment variables. Create `frontend/.env.local` (optional):

```bash
# Backend API URL (defaults to http://127.0.0.1:8000)
VITE_API_URL=http://localhost:8000
```

---

## Running the Application

### Development Mode (Recommended)

Run both backend and frontend concurrently:

```bash
make dev
```

This starts:
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

### Run Separately

**Backend only:**
```bash
make dev-backend
```

**Frontend only:**
```bash
make dev-frontend
```

### Production Build

```bash
# Build frontend
make build

# Run backend in production mode
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Verification

### 1. Check Backend Health

```bash
make health
```

Or manually:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "env": {
    "GEMINI_API_KEY": "set",
    "GEMINI_ENABLED": true,
    "PORT": "not set"
  }
}
```

### 2. Test the Application

1. Open http://localhost:5173 in your browser
2. Type a test question: "What is this document about?"
3. Verify you get a response with sources

### 3. Check Running Processes

```bash
make status
```

This shows which ports are active.

---

## Common Issues

### Issue: "GEMINI_API_KEY not set"

**Solution:**
```bash
make setup-env
# Edit .env and add your API key
make check-env
```

### Issue: "Port already in use"

**Solution:**
```bash
make kill
make dev
```

### Issue: "Module not found"

**Solution:**
```bash
# Reinstall dependencies
make clean
make install
```

### Issue: "faiss.index not found"

**Solution:**
```bash
make rebuild-index
```

### Issue: Frontend can't connect to backend

**Solution:**
1. Check backend is running: `make status`
2. Verify backend health: `make health`
3. Check CORS settings in `app.py`
4. Update `VITE_API_URL` in `frontend/.env.local` if needed

### Issue: "Command not found: make"

**Solution:**

**macOS:**
```bash
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install build-essential
```

**Windows:**
- Use WSL (Windows Subsystem for Linux)
- Or install Make for Windows

---

## Development Tools (Optional)

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

This includes:
- `black` — Code formatter
- `flake8` — Linter
- `pytest` — Testing framework
- `mypy` — Type checker

### Set Up Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This automatically checks your code before each commit.

### Run Linters

```bash
make lint              # Frontend
make lint-backend      # Backend (requires dev dependencies)
```

---

## Next Steps

1. ✅ Verify setup: `make check-env`
2. ✅ Run application: `make dev`
3. ✅ Test in browser: http://localhost:5173
4. ✅ Review documentation: [README.md](README.md)
5. ✅ Deploy: See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Getting Help

- [README.md](README.md) — Project overview and quick start
- [DEPLOYMENT.md](DEPLOYMENT.md) — Platform-specific deployment guides
- [SECURITY.md](SECURITY.md) — Security policy
- [GitHub Issues](https://github.com/om-gupta-30/YNM-RAG-CHATBOT/issues) — Bug reports and questions
