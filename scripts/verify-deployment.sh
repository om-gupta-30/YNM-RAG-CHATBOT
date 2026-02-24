#!/bin/bash
# Deployment verification script
# Run before pushing to GitHub or deploying

set -e

echo "🔍 RAG Chatbot - Deployment Verification"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

errors=0
warnings=0

# Check 1: .env not tracked
echo "1. Checking .env is not tracked..."
if git ls-files | grep -E "^\.env$" > /dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: .env is tracked in git!${NC}"
    errors=$((errors + 1))
else
    echo -e "${GREEN}✅ .env is not tracked${NC}"
fi

# Check 2: No secrets in code (excluding .env which is already verified as not tracked)
echo "2. Scanning for hardcoded secrets..."
if grep -r "AIza[0-9A-Za-z_-]\{35\}" . \
    --exclude-dir=.git \
    --exclude-dir=node_modules \
    --exclude-dir=.github \
    --exclude-dir=frontend/node_modules \
    --exclude-dir=scripts \
    --exclude=".env" \
    --exclude="*.md" > /dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: Gemini API key pattern detected in tracked files!${NC}"
    errors=$((errors + 1))
else
    echo -e "${GREEN}✅ No API keys found in tracked code${NC}"
fi

# Check 3: .gitignore exists and is comprehensive
echo "3. Checking .gitignore..."
if [ ! -f .gitignore ]; then
    echo -e "${RED}❌ ERROR: .gitignore not found!${NC}"
    errors=$((errors + 1))
else
    required_patterns=(".env" "__pycache__" "node_modules" "*.pyc" "faiss.index")
    missing=0
    for pattern in "${required_patterns[@]}"; do
        if ! grep -q "$pattern" .gitignore; then
            echo -e "${YELLOW}⚠️  WARNING: '$pattern' not in .gitignore${NC}"
            warnings=$((warnings + 1))
            missing=1
        fi
    done
    if [ $missing -eq 0 ]; then
        echo -e "${GREEN}✅ .gitignore is comprehensive${NC}"
    fi
fi

# Check 4: .env.example exists
echo "4. Checking .env.example..."
if [ ! -f .env.example ]; then
    echo -e "${YELLOW}⚠️  WARNING: .env.example not found${NC}"
    warnings=$((warnings + 1))
else
    if grep -q "GEMINI_API_KEY=AIza" .env.example; then
        echo -e "${RED}❌ ERROR: .env.example contains actual API key!${NC}"
        errors=$((errors + 1))
    else
        echo -e "${GREEN}✅ .env.example is safe${NC}"
    fi
fi

# Check 5: Required files exist
echo "5. Checking required files..."
required_files=("app.py" "requirements.txt" "Makefile" "README.md")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ ERROR: Required file '$file' not found!${NC}"
        errors=$((errors + 1))
    fi
done
echo -e "${GREEN}✅ All required files present${NC}"

# Check 6: Git history
echo "6. Checking git history for .env..."
if git log --all --full-history -- .env 2>/dev/null | grep -q "commit"; then
    echo -e "${RED}❌ ERROR: .env was committed in git history!${NC}"
    echo -e "${YELLOW}   Run: git filter-branch to remove it${NC}"
    errors=$((errors + 1))
else
    echo -e "${GREEN}✅ .env never committed${NC}"
fi

# Check 7: Staged files
echo "7. Checking staged files..."
if git diff --cached --name-only | grep -E "\.env$|\.key$|\.pem$|credentials\.json" > /dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: Secret files are staged!${NC}"
    errors=$((errors + 1))
else
    echo -e "${GREEN}✅ No secret files staged${NC}"
fi

# Summary
echo ""
echo "========================================"
echo "Summary:"
echo "  Errors: $errors"
echo "  Warnings: $warnings"
echo ""

if [ $errors -gt 0 ]; then
    echo -e "${RED}❌ VERIFICATION FAILED${NC}"
    echo "Fix errors before deploying!"
    exit 1
elif [ $warnings -gt 0 ]; then
    echo -e "${YELLOW}⚠️  VERIFICATION PASSED WITH WARNINGS${NC}"
    echo "Review warnings before deploying."
    exit 0
else
    echo -e "${GREEN}✅ VERIFICATION PASSED${NC}"
    echo "Safe to deploy!"
    exit 0
fi
