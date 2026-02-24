# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive deployment support (Vercel, GCP, Railway, Render, Docker)
- GitHub Actions CI/CD workflows (linting, building, security checks)
- Security verification script (`make verify-deploy`)
- Pre-commit hooks configuration
- Docker support with multi-stage builds
- Deployment documentation (DEPLOYMENT.md)
- Contributing guidelines (CONTRIBUTING.md)
- Security policy (SECURITY.md)
- Development dependencies (requirements-dev.txt)
- Enhanced Makefile with new commands (setup-env, check-env, health, status, verify-deploy)

### Changed
- Updated README.md with deployment instructions and security notices
- Enhanced .gitignore with comprehensive patterns
- Improved .env.example with detailed comments
- Updated Makefile with better organization and feedback

### Security
- Added secret scanning in CI/CD pipeline
- Created deployment verification script
- Enhanced .gitignore to prevent secret leaks
- Added .dockerignore and .gcloudignore

## [1.0.0] - 2024-02-23

### Added
- Initial release
- RAG pipeline with FAISS vector search
- Intent-based retrieval (figure, table, page, section, general, comparison)
- FastAPI backend with structured responses
- React frontend with dark/light theme
- Multi-chat support with PDF export
- Confidence scoring system
- Vision caption support for images
- Source citations with context expansion

### Features
- Intelligent Q&A over PDF documents
- Support for figures, tables, and page references
- Semantic search with Gemini embeddings
- Structured answers (paragraphs and lists)
- Modern chat interface
- PDF export functionality
