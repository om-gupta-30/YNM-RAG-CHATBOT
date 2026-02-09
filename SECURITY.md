# Security

## Reporting a Vulnerability

If you discover a security issue, please open a GitHub Issue or contact the maintainer directly. Do not disclose secrets in public issues.

## Secrets and Environment Variables

- **Never commit** `.env` or any file containing API keys, tokens, or credentials.
- Use `.env.example` as a template only; it contains no secrets.
- For **GitHub**: Ensure `.env` is listed in `.gitignore`. Before every push, run `git status` and confirm no `.env` or secret files are staged.
- For **Vercel**: Set `GEMINI_API_KEY` and `VITE_API_URL` in Project Settings → Environment Variables. Never put secrets in repository code or config.
- For **Google Cloud Run**: Set `GEMINI_API_KEY` via the deploy script (`--set-env-vars`) or Cloud Console. The key is never stored in the image or source.

## If You Accidentally Committed a Secret

1. **Rotate the key immediately** (revoke and create a new one in Google AI Studio).
2. Remove the secret from git history (e.g. `git filter-branch`, BFG Repo-Cleaner, or GitHub’s secret scanning and guidance).
3. Force-push only if you understand the impact; prefer creating a new repo if the repo was never shared.

## Application Security

- The backend uses `GEMINI_API_KEY` only on the server; it is never sent to the browser or exposed in API responses.
- The `/health` endpoint returns only whether the key is "set" or "missing", not the key value.
