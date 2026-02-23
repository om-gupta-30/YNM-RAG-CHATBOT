.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend \
        build lint kill rebuild-index clean

help:
	@echo ""
	@echo "RAG Chatbot – available commands:"
	@echo ""
	@echo "  make install          Install all backend + frontend dependencies"
	@echo "  make install-backend  Install Python dependencies only"
	@echo "  make install-frontend Install Node dependencies only"
	@echo ""
	@echo "  make dev              Run backend + frontend concurrently (recommended)"
	@echo "  make dev-backend      Run FastAPI backend only  (http://localhost:8000)"
	@echo "  make dev-frontend     Run Vite dev server only  (http://localhost:5173)"
	@echo ""
	@echo "  make build            Build the frontend for production"
	@echo "  make lint             Lint the frontend"
	@echo ""
	@echo "  make kill             Kill all processes on dev ports"
	@echo "  make rebuild-index    Re-chunk + re-embed → new faiss.index"
	@echo "  make clean            Remove build artefacts and caches"
	@echo ""

install: install-backend install-frontend

install-backend:
	pip install -r requirements.txt

install-frontend:
	npm install --prefix frontend

dev:
	@echo "Starting backend and frontend..."
	@trap 'kill 0' INT; \
	  uvicorn app:app --reload --host 0.0.0.0 --port 8000 & \
	  npm run dev --prefix frontend & \
	  wait

dev-backend:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	npm run dev --prefix frontend

build:
	npm run build --prefix frontend

lint:
	npm run lint --prefix frontend

kill:
	@for port in 8000 5173 5174; do \
	  pids=$$(lsof -ti:$$port 2>/dev/null); \
	  if [ -n "$$pids" ]; then \
	    echo "Killing port $$port (pids: $$pids)"; \
	    echo "$$pids" | xargs kill -9; \
	  fi; \
	done
	@echo "All dev ports cleared."

rebuild-index:
	python3 rebuild_index.py

clean:
	rm -rf frontend/dist frontend/node_modules/__vite_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
