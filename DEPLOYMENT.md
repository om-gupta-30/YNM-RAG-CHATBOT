# Deployment Guide

This guide covers deploying the RAG PDF Chatbot to various platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Vercel Deployment](#vercel-deployment)
- [Google Cloud Platform (GCP)](#google-cloud-platform-gcp)
- [Railway](#railway)
- [Render](#render)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)

---

## Prerequisites

Before deploying, ensure:

1. ✅ `.env` is **NOT** committed to git
2. ✅ All secrets are configured as environment variables
3. ✅ `faiss.index` and `metadata.json` are generated
4. ✅ Frontend builds successfully (`make build`)
5. ✅ Backend health check passes (`make health`)

---

## Vercel Deployment

### Option 1: Full-Stack (Serverless)

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel
   ```

3. **Set Environment Variables:**
   ```bash
   vercel env add GEMINI_API_KEY
   # Paste your API key when prompted
   ```

4. **Deploy to production:**
   ```bash
   vercel --prod
   ```

### Option 2: Frontend Only (Backend Elsewhere)

1. **Update API endpoint** in `frontend/src/api.js`:
   ```javascript
   const API_BASE_URL = 'https://your-backend-url.com';
   ```

2. **Deploy frontend:**
   ```bash
   cd frontend
   vercel
   ```

---

## Google Cloud Platform (GCP)

### Cloud Run (Recommended)

1. **Install gcloud CLI:**
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Deploy:**
   ```bash
   gcloud run deploy rag-chatbot \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GEMINI_API_KEY=your_key_here \
     --memory 2Gi \
     --cpu 2 \
     --timeout 300
   ```

4. **Get the URL:**
   ```bash
   gcloud run services describe rag-chatbot --region us-central1 --format 'value(status.url)'
   ```

### App Engine

1. **Create `app.yaml`:**
   ```yaml
   runtime: python311
   entrypoint: uvicorn app:app --host 0.0.0.0 --port $PORT
   
   env_variables:
     GEMINI_API_KEY: "your_key_here"
   
   automatic_scaling:
     min_instances: 0
     max_instances: 10
   ```

2. **Deploy:**
   ```bash
   gcloud app deploy
   ```

---

## Railway

1. **Connect GitHub repository** at [railway.app](https://railway.app)

2. **Configure build:**
   - Build Command: `pip install -r requirements.txt && cd frontend && npm install && npm run build`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables:**
   - `GEMINI_API_KEY`: Your API key
   - `PORT`: Auto-assigned by Railway

4. **Deploy** — Railway auto-deploys on git push

---

## Render

1. **Create new Web Service** at [render.com](https://render.com)

2. **Connect repository:** `https://github.com/om-gupta-30/YNM-RAG-CHATBOT`

3. **Configure:**
   - **Environment:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt && cd frontend && npm install && npm run build && cd ..
     ```
   - **Start Command:**
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```

4. **Environment Variables:**
   - `GEMINI_API_KEY`: Your API key
   - `PYTHON_VERSION`: `3.11.0`

5. **Deploy** — Render auto-deploys on git push

---

## Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  rag-chatbot
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  rag-chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./images:/app/images:ro
      - ./faiss.index:/app/faiss.index:ro
      - ./metadata.json:/app/metadata.json:ro
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Push to Docker Hub

```bash
docker tag rag-chatbot your-username/rag-chatbot:latest
docker push your-username/rag-chatbot:latest
```

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `CORS_ORIGINS` | Allowed origins | `*` |

---

## Post-Deployment Checklist

- [ ] Test health endpoint: `curl https://your-app.com/health`
- [ ] Verify API endpoints work
- [ ] Check frontend loads correctly
- [ ] Test a sample query
- [ ] Monitor logs for errors
- [ ] Set up monitoring/alerts (optional)
- [ ] Configure custom domain (optional)
- [ ] Enable HTTPS (most platforms do this automatically)

---

## Troubleshooting

### Build fails on deployment platform

- Ensure `requirements.txt` is up to date
- Check Python version compatibility (3.10+)
- Verify Node.js version (18+)

### Backend returns 500 errors

- Check `GEMINI_API_KEY` is set correctly
- Verify `faiss.index` and `metadata.json` exist
- Check logs for specific error messages

### Frontend can't connect to backend

- Update `API_BASE_URL` in `frontend/src/api.js`
- Verify CORS settings allow your frontend domain
- Check network/firewall rules

### Large file size issues

- The `images/` directory (27MB) and `faiss.index` (~2MB) are required
- Consider using cloud storage for images if size is an issue
- Some platforms have file size limits (e.g., Vercel 250MB)

---

## Monitoring & Maintenance

### Health Checks

Most platforms support health check endpoints:
- **Endpoint:** `/health`
- **Expected Response:** `{"status": "ok", "env": {...}}`

### Logs

Access logs via your platform:
- **Vercel:** `vercel logs`
- **GCP:** `gcloud run logs read`
- **Railway:** Dashboard → Logs tab
- **Render:** Dashboard → Logs tab

### Updates

To deploy updates:
```bash
git add .
git commit -m "Update: description"
git push origin main
```

Most platforms auto-deploy on push to main branch.

---

## Cost Considerations

### Gemini API
- Embedding: ~$0.00001 per 1K tokens
- Generation: ~$0.00025 per 1K tokens
- Monitor usage: https://console.cloud.google.com/

### Platform Costs
- **Vercel:** Free tier available (hobby projects)
- **GCP Cloud Run:** Pay per request (free tier: 2M requests/month)
- **Railway:** $5/month (500 hours)
- **Render:** Free tier available (limited)

---

## Support

For deployment issues:
- Check platform-specific documentation
- Review logs for error messages
- Open an issue on GitHub
- See [CONTRIBUTING.md](CONTRIBUTING.md)
