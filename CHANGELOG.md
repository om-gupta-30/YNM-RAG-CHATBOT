# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-24

### Added
- Deployment support for Vercel, GCP Cloud Run, Railway, Render, and Docker
- GitHub Actions CI/CD workflow with linting, build, and security checks
- Pre-deployment security verification script (`make verify-deploy`)
- Pre-commit hooks configuration for code quality and secret detection
- Multi-stage Dockerfile
- Documentation: SETUP.md, DEPLOYMENT.md, CONTRIBUTING.md, SECURITY.md
- Development dependencies (requirements-dev.txt)
- Enhanced Makefile with setup-env, check-env, health, status, verify-deploy commands

### Changed
- Rewrote README.md for clarity and completeness
- Expanded .gitignore with comprehensive patterns for secrets, caches, and build artifacts
- Improved .env.example with detailed comments and links

### Security
- Secret scanning in CI/CD pipeline
- .dockerignore and .gcloudignore exclude secrets from builds
- Verified .env was never committed to git history

## [1.0.0] - 2026-02-23

### Added
- RAG pipeline with FAISS vector search and Google Gemini
- Intent-based retrieval (figure, table, page, section, general, comparison)
- FastAPI backend with structured JSON responses
- React 19 frontend with dark/light theme, multi-chat, and PDF export
- Confidence scoring system
- Vision caption support for page images
- Source citations with context expansion
