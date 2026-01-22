# ---- UI build ----
# Build-time env: VITE_API_URL (empty = same-origin API in production)
FROM node:20-alpine AS ui
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
ENV VITE_API_URL=
RUN npm run build

# ---- Backend + runtime ----
# Runtime env (Cloud Run): GEMINI_API_KEY (required), PORT (set by Cloud Run)
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=ui /app/ui/dist ./ui/dist

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
