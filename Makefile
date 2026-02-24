.PHONY: help install install-backend install-frontend setup-env check-env \
        dev dev-backend dev-frontend build lint lint-backend test \
        kill rebuild-index clean clean-all health status verify-deploy

help:
	@echo ""
	@echo "RAG Chatbot – available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install all backend + frontend dependencies"
	@echo "  make install-backend  Install Python dependencies only"
	@echo "  make install-frontend Install Node dependencies only"
	@echo "  make setup-env        Copy .env.example to .env (if not exists)"
	@echo "  make check-env        Verify environment variables are set"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Run backend + frontend concurrently (recommended)"
	@echo "  make dev-backend      Run FastAPI backend only  (http://localhost:8000)"
	@echo "  make dev-frontend     Run Vite dev server only  (http://localhost:5173)"
	@echo "  make health           Check backend health endpoint"
	@echo "  make status           Show running processes on dev ports"
	@echo ""
	@echo "Build & Quality:"
	@echo "  make build            Build the frontend for production"
	@echo "  make lint             Lint the frontend code"
	@echo "  make lint-backend     Lint Python code with flake8/black (if installed)"
	@echo "  make test             Run tests (placeholder for future tests)"
	@echo ""
	@echo "Utilities:"
	@echo "  make kill             Kill all processes on dev ports (8000, 5173, 5174)"
	@echo "  make rebuild-index    Re-chunk + re-embed → new faiss.index"
	@echo "  make clean            Remove build artifacts and caches"
	@echo "  make clean-all        Deep clean (includes node_modules, venv, indexes)"
	@echo "  make verify-deploy    Verify project is safe to deploy (checks for secrets)"
	@echo ""

# Installation targets
install: install-backend install-frontend
	@echo "✓ All dependencies installed successfully!"

install-backend:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Backend dependencies installed"

install-frontend:
	@echo "Installing Node dependencies..."
	npm install --prefix frontend
	@echo "✓ Frontend dependencies installed"

# Environment setup
setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example"; \
		echo "⚠ Please edit .env and add your GEMINI_API_KEY"; \
	else \
		echo "✓ .env already exists"; \
	fi

check-env:
	@if [ ! -f .env ]; then \
		echo "✗ .env file not found. Run 'make setup-env' first."; \
		exit 1; \
	fi
	@if ! grep -q "GEMINI_API_KEY=.*[^[:space:]]" .env 2>/dev/null; then \
		echo "✗ GEMINI_API_KEY not set in .env"; \
		exit 1; \
	fi
	@echo "✓ Environment variables configured"

# Development targets
dev: check-env
	@echo "Starting backend and frontend..."
	@trap 'kill 0' INT; \
	  uvicorn app:app --reload --host 0.0.0.0 --port 8000 & \
	  npm run dev --prefix frontend & \
	  wait

dev-backend: check-env
	@echo "Starting FastAPI backend on http://localhost:8000"
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	@echo "Starting Vite dev server on http://localhost:5173"
	npm run dev --prefix frontend

# Health and status checks
health:
	@echo "Checking backend health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "✗ Backend not responding"

status:
	@echo "Checking processes on dev ports..."
	@for port in 8000 5173 5174; do \
		pids=$$(lsof -ti:$$port 2>/dev/null); \
		if [ -n "$$pids" ]; then \
			echo "✓ Port $$port: Active (PIDs: $$pids)"; \
		else \
			echo "  Port $$port: Inactive"; \
		fi; \
	done

# Build and quality targets
build:
	@echo "Building frontend for production..."
	npm run build --prefix frontend
	@echo "✓ Frontend built successfully (frontend/dist)"

lint:
	@echo "Linting frontend code..."
	npm run lint --prefix frontend

lint-backend:
	@echo "Linting Python code..."
	@if command -v black >/dev/null 2>&1; then \
		black --check *.py 2>/dev/null || echo "⚠ black not configured"; \
	else \
		echo "⚠ black not installed (pip install black)"; \
	fi
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 *.py 2>/dev/null || echo "⚠ flake8 not configured"; \
	else \
		echo "⚠ flake8 not installed (pip install flake8)"; \
	fi

test:
	@echo "Running tests..."
	@echo "⚠ No tests configured yet. Add pytest tests in tests/ directory."

# Utility targets
kill:
	@echo "Killing processes on dev ports..."
	@for port in 8000 5173 5174; do \
	  pids=$$(lsof -ti:$$port 2>/dev/null); \
	  if [ -n "$$pids" ]; then \
	    echo "Killing port $$port (PIDs: $$pids)"; \
	    echo "$$pids" | xargs kill -9; \
	  fi; \
	done
	@echo "✓ All dev ports cleared"

rebuild-index: check-env
	@echo "Rebuilding FAISS index..."
	python3 rebuild_index.py
	@echo "✓ Index rebuilt successfully"

clean:
	@echo "Cleaning build artifacts and caches..."
	rm -rf frontend/dist frontend/node_modules/.vite
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete"

clean-all: clean
	@echo "Deep cleaning (node_modules, venv, indexes)..."
	rm -rf frontend/node_modules
	rm -rf .venv
	rm -f faiss.index
	rm -f vision_captions.json
	@echo "✓ Deep clean complete"

# Security verification
verify-deploy:
	@echo "Running deployment verification..."
	@chmod +x scripts/verify-deployment.sh
	@./scripts/verify-deployment.sh
