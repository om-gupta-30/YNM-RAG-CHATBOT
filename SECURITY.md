# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by:

1. **DO NOT** open a public issue
2. Email the maintainer directly (check GitHub profile for contact)
3. Include detailed steps to reproduce the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Security Best Practices

### API Keys & Secrets

- **NEVER commit `.env` files** to version control
- Always use `.env.example` as a template (without actual values)
- Rotate API keys immediately if accidentally exposed
- Use environment variables for all sensitive configuration

### Deployment Security

- Set `GEMINI_API_KEY` as an environment variable in your deployment platform
- Use HTTPS in production
- Configure CORS appropriately (don't use `*` in production)
- Implement rate limiting for API endpoints
- Use authentication/authorization for production deployments

### Files That Should NEVER Be Committed

- `.env` (contains actual secrets)
- `*.key`, `*.pem`, `*.p12`, `*.pfx` (certificates/keys)
- `credentials.json`, `service-account*.json` (GCP credentials)
- Any file containing API keys or passwords

### Checking for Exposed Secrets

Before pushing to GitHub:

```bash
# Verify .env is not tracked
git status

# Check if .env was ever committed
git log --all --full-history -- .env

# Verify environment setup
make check-env

# Review what will be pushed
git diff origin/main
```

### If You Accidentally Commit Secrets

1. **Immediately revoke/rotate the exposed key**
2. Generate a new API key
3. Update your `.env` with the new key
4. Remove the secret from git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   ```
5. Force push (⚠️ coordinate with team first):
   ```bash
   git push origin --force --all
   ```

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Dependencies

This project uses:
- Google Gemini API (ensure you comply with Google's terms of service)
- FAISS (Meta, MIT License)
- FastAPI (MIT License)
- React (MIT License)

Keep dependencies updated to patch security vulnerabilities:

```bash
# Backend
pip list --outdated

# Frontend
npm outdated --prefix frontend
```
