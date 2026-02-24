# Quick Reference Card

One-page reference for common tasks.

---

## 🚀 Common Commands

```bash
# Setup (first time)
make setup-env        # Create .env
make install          # Install dependencies
make check-env        # Verify configuration

# Development
make dev              # Run full stack
make dev-backend      # Backend only
make dev-frontend     # Frontend only
make health           # Check backend
make status           # Show processes

# Deployment
make verify-deploy    # Security check
make build            # Build frontend

# Maintenance
make kill             # Kill dev servers
make clean            # Clean artifacts
make rebuild-index    # Rebuild FAISS
```

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Overview & quick start |
| `SETUP.md` | Detailed setup |
| `DEPLOYMENT.md` | Deploy to platforms |
| `SECURITY.md` | Security practices |

---

## 🔒 Security Checklist

Before pushing to GitHub:

```bash
✓ make verify-deploy    # Must pass
✓ git status            # Check .env not staged
✓ git diff              # Review changes
```

---

## 🌐 Deployment

### Vercel (Easiest)
```bash
npm i -g vercel
vercel
vercel env add GEMINI_API_KEY
vercel --prod
```

### GCP Cloud Run
```bash
gcloud run deploy rag-chatbot \
  --source . \
  --set-env-vars GEMINI_API_KEY=your_key
```

### Railway/Render
1. Connect GitHub repo
2. Set `GEMINI_API_KEY` env var
3. Auto-deploys on push

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | `make kill` |
| Missing .env | `make setup-env` |
| Backend won't start | `make check-env` |
| Frontend build fails | `make clean && make install-frontend` |

---

## 📊 Ports

- Backend: `8000`
- Frontend: `5173`
- Preview: `5174`

---

## 🔑 Environment Variables

Required:
- `GEMINI_API_KEY` — Get from https://aistudio.google.com/app/apikey

Optional:
- `PORT` — Default: 8000
- `HOST` — Default: 0.0.0.0

---

## 📞 Help

- Docs: See `*.md` files
- Issues: GitHub Issues
- Email: Check GitHub profile
