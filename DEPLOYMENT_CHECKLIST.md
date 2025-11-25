# GitHub Deployment Checklist

Use this checklist before pushing to GitHub and deploying to production.

## Pre-Commit Security Checklist

### 1. Secrets & Credentials
- [ ] No API keys hardcoded in source files
- [ ] No passwords in configuration files
- [ ] No database credentials in code
- [ ] All secrets use environment variables
- [ ] `.env` files are in `.gitignore`
- [ ] `.env.local` files are in `.gitignore`
- [ ] `frontend/.env.local` is gitignored
- [ ] No AWS/GCP/Azure credentials in code
- [ ] No private SSH keys committed

**Verify**:
```bash
# Search for potential secrets
git grep -E "(api[_-]?key|password|secret|token|credential)" --and --not -E "\.example|README"
```

### 2. Environment Files
- [ ] `.env.example` is up to date with all required variables
- [ ] `frontend/.env.local.example` is up to date
- [ ] No actual values in example files (only placeholders)
- [ ] README documents all environment variables

### 3. .gitignore Configuration
- [ ] `venv/` and `venv_ml/` are ignored
- [ ] `node_modules/` is ignored
- [ ] `.env` and `.env.local` are ignored
- [ ] Database files (`.db`, `.sqlite`) are ignored
- [ ] `__pycache__/` is ignored
- [ ] IDE files (`.vscode/`, `.idea/`) are ignored
- [ ] Logs (`*.log`, `logs/`) are ignored
- [ ] ML artifacts (`mlruns/`, `models/`) are ignored

**Verify**:
```bash
git status --ignored
```

### 4. Documentation
- [ ] README.md is complete and accurate
- [ ] Installation instructions are clear
- [ ] API documentation is updated
- [ ] Architecture diagrams are current
- [ ] CONTRIBUTING.md exists
- [ ] LICENSE file exists
- [ ] SECURITY.md exists with contact info
- [ ] DEPLOYMENT.md has deployment instructions

### 5. Configuration Files
- [ ] `docker-compose.yml` uses environment variables for secrets
- [ ] No hardcoded ports that conflict with common services
- [ ] Health checks are configured
- [ ] Resource limits are set appropriately

## Code Quality Checklist

### 1. Python Backend
- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are used
- [ ] No print statements (use logging)
- [ ] Error handling is implemented
- [ ] Input validation on all endpoints

**Run**:
```bash
# Linting
flake8 backend/ --max-line-length=100

# Formatting
black backend/ --check --line-length=100

# Type checking
mypy backend/ --ignore-missing-imports
```

### 2. Frontend
- [ ] ESLint passes
- [ ] Prettier formatting applied
- [ ] TypeScript types are defined
- [ ] No `console.log` in production code
- [ ] Components are properly typed

**Run**:
```bash
cd frontend
npm run lint
npm run type-check
```

### 3. Testing
- [ ] All tests pass
- [ ] New features have tests
- [ ] Code coverage > 80%
- [ ] Integration tests pass

**Run**:
```bash
# Backend tests
pytest tests/ -v --cov=backend

# Frontend tests
cd frontend && npm test
```

## GitHub Repository Setup

### 1. Repository Settings
- [ ] Repository is public/private as intended
- [ ] Repository description is set
- [ ] Topics/tags are added for discoverability
- [ ] Default branch is set to `main`
- [ ] Branch protection rules are configured (for production)

### 2. GitHub Actions
- [ ] Workflow files are in `.github/workflows/`
- [ ] CI workflow runs on pull requests
- [ ] Secrets are configured in repository settings:
  - `GITHUB_TOKEN` (automatically available)
  - Add any additional secrets needed

### 3. Repository Files
- [ ] README.md in root
- [ ] LICENSE file
- [ ] CONTRIBUTING.md
- [ ] SECURITY.md
- [ ] .gitignore is comprehensive
- [ ] Issue templates (optional)
- [ ] Pull request template (optional)

## First Commit Checklist

### 1. Initialize Repository
```bash
git init
git add .
git commit -m "Initial commit: ML Monitoring System"
```

### 2. Add Remote
```bash
git remote add origin https://github.com/YOUR_USERNAME/ml-monitoring-system.git
```

### 3. Push to GitHub
```bash
git push -u origin main
```

### 4. Verify
- [ ] All files are visible on GitHub
- [ ] No sensitive files were pushed
- [ ] GitHub Actions workflows trigger
- [ ] README renders correctly

## Docker Image Checklist

### 1. Dockerfile Security
- [ ] Uses specific base image versions (not `latest`)
- [ ] Runs as non-root user
- [ ] Multi-stage build to reduce image size
- [ ] No secrets in Docker layers
- [ ] `.dockerignore` is configured

### 2. Build & Test
```bash
# Build images
docker-compose build

# Test locally
docker-compose up -d

# Verify services
docker-compose ps
docker-compose logs backend
curl http://localhost:8000/health
```

### 3. Image Scanning
```bash
# Scan for vulnerabilities
docker scan backend:latest
```

## Production Deployment Checklist

### 1. Environment Configuration
- [ ] Production environment variables set
- [ ] Strong, unique API keys generated
- [ ] Database password is strong (16+ chars)
- [ ] SSL/TLS certificates configured
- [ ] CORS origins restricted to production domains
- [ ] Rate limiting configured

### 2. Infrastructure
- [ ] Database backups configured
- [ ] Monitoring/alerting set up
- [ ] Log aggregation configured
- [ ] CDN configured for frontend (optional)
- [ ] DDoS protection enabled

### 3. Security
- [ ] HTTPS enabled
- [ ] Security headers configured
- [ ] Firewall rules set
- [ ] Database access restricted
- [ ] API key rotation policy in place

### 4. Performance
- [ ] Caching configured (Redis)
- [ ] Database indexes created
- [ ] Resource limits appropriate
- [ ] Auto-scaling configured (if applicable)

### 5. Monitoring
- [ ] Health check endpoints working
- [ ] Prometheus metrics exposed
- [ ] Grafana dashboards configured
- [ ] Alerting rules defined
- [ ] Uptime monitoring set up

## Post-Deployment Checklist

### 1. Verification
- [ ] All services are running
- [ ] API endpoints respond correctly
- [ ] Frontend loads successfully
- [ ] Database migrations completed
- [ ] Monitoring dashboards show data

### 2. Testing
- [ ] Smoke tests pass
- [ ] API authentication works
- [ ] Predictions can be made
- [ ] Model loading successful
- [ ] Metrics are being collected

### 3. Documentation
- [ ] Update README with production URLs
- [ ] Document any manual steps taken
- [ ] Update API documentation if needed
- [ ] Create runbook for common operations

## Common Mistakes to Avoid

1. **Don't commit**:
   - `.env` files
   - API keys or secrets
   - Database files
   - `node_modules/`
   - `__pycache__/`
   - Personal IDE settings

2. **Don't use**:
   - Default passwords
   - `localhost` URLs in production configs
   - Debug mode in production
   - Weak API keys

3. **Always**:
   - Test locally before pushing
   - Review diffs before committing
   - Use meaningful commit messages
   - Update documentation
   - Run security scans

## Emergency Procedures

### If Secrets Are Committed

1. **Immediately rotate all exposed secrets**
2. **Remove from git history**:
```bash
# Use BFG Repo Cleaner or git filter-branch
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch PATH_TO_SECRET_FILE" \
  --prune-empty --tag-name-filter cat -- --all
```
3. **Force push** (only if no one else has pulled):
```bash
git push origin --force --all
```
4. **Notify team** if repository is shared
5. **Review access logs** for unauthorized use

### If Vulnerability Discovered

1. **Follow SECURITY.md procedures**
2. **Assess severity**
3. **Develop and test fix**
4. **Deploy patch immediately** (for critical issues)
5. **Notify users** if data breach occurred

## Final Verification

Before deployment, run:

```bash
# Check for secrets
git secrets --scan-history  # If git-secrets is installed

# Verify gitignore
git status --ignored

# Test Docker build
docker-compose build

# Run all tests
pytest tests/
cd frontend && npm test

# Security scan
npm audit
pip audit
```

---

**Remember**: It's better to spend extra time on security before deployment than to deal with a breach after!

## Checklist Summary

- [ ] All secrets removed/secured
- [ ] .gitignore configured
- [ ] Documentation complete
- [ ] Tests passing
- [ ] Code quality checks pass
- [ ] Docker images built
- [ ] GitHub repository configured
- [ ] Security review complete
- [ ] Ready for deployment

**Date**: ___________
**Reviewer**: ___________
**Approved**: [ ] Yes [ ] No
