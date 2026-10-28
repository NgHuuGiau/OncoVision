.PHONY: help install install-dev install-monai test test-fast test-cov lint fmt type-check clean smoke smoke-ci doctor chat medical run-app run-chat run-menu doctor smoke test test-fast test-cov lint fmt type-check clean docker-build docker-run docker-stop docker-logs docker-shell install-monai docker-run-monai

.PHONY: help install install-dev install-monai test test-fast test-cov lint fmt type-check clean smoke smoke-ci doctor chat medical run-app run-chat run-menu

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

help: ## Show this help message
	@echo "$(CYAN)OncoVision - Medical AI Screening System$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  install-monai  Install MONAI/nnUNet dependencies (for pancreas/stomach)"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  fmt            Format code with ruff"
	@echo "  lint           Lint code with ruff"
	@echo "  type-check     Run mypy type checking"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  test           Run all tests"
	@echo "  test-fast      Run tests excluding slow/integration tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run ruff linting"
	@echo "  type-check     Run mypy type checking"
	@echo ""
	@echo "$(GREEN)Runtime Commands:$(NC)"
	@echo "  run-app        Run camera application (run_app.py)"
	@echo "  run-chat       Run desktop chat (run_chat.py)"
	@echo "  run-menu       Run menu system (run_menu.py)"
	@echo "  doctor         Run system check (run_doctor.py)"
	@echo "  medical        Run medical CLI (run_medical.py)"
	@echo "  chat           Run chat web app"
	@echo "  chat-desktop   Run desktop chat UI"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  test           Run all tests"
	@echo "  test-fast      Run fast tests (exclude slow/integration)"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run ruff linting"
	@echo "  type-check     Run mypy type checking"
	@echo ""
	@echo "$(GREEN)Quality Gates:$(NC)"
	@echo "  smoke          Run smoke test (run_smoke.py)"
	@echo "  smoke-ci       Run smoke test for CI"
	@echo "  doctor         Run system check (run_doctor.py)"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run container"
	@echo "  docker-stop    Stop container"
	@echo "  docker-logs    View container logs"
	@echo "  docker-shell   Open shell in container"
	@echo "  install-monai  Install MONAI/nnUNet deps"
	@echo "  docker-run-monai Run with MONAI support"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  clean          Clean cache, build artifacts, temp files"
	@echo ""
	@echo "$(GREEN)Database:$(NC)"
	@echo "  db-init        Initialize database"
	@echo "  db-migrate     Run migrations"
	@echo "  db-shell       Open SQLite shell"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install-dev"
	@echo "  make test"
	@echo "  make run-chat"
	@echo "  make smoke"

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install ruff==0.16.3 mypy pytest pytest-cov pytest-asyncio pre-commit

install-monai: ## Install MONAI/nnUNet for pancreas/stomach segmentation
	pip install monai>=1.3.0 nnunetv2>=2.0

install-dev-full: install-dev install-monai ## Install all development dependencies including MONAI

# Testing
test: ## Run all tests
	pytest tests/ -v --tb=short

test-fast: ## Run tests excluding slow/integration tests
	pytest tests/ -q -m "not slow and not integration" --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=app --cov=core --cov=medical --cov=utils --cov=config --cov=scripts --cov-report=term-missing --cov-report=html:htmlcov

test-unit: ## Run unit tests only
	pytest tests/ -q -m "unit" --tb=short

test-integration: ## Run integration tests
	pytest tests/ -q -m "integration" --tb=short

test-medical: ## Run medical tests only
	pytest tests/test_medical_pipeline.py tests/test_cancer_registry.py tests/test_volume_inference.py tests/test_fetch_model.py -v

test-web: ## Run web tests only
	pytest tests/test_web_cases.py tests/test_web_auth.py -v

test-model: ## Run model selector tests
	pytest tests/test_model_selector.py -v

test-hardware: ## Run hardware detector tests
	pytest tests/test_hardware_detector.py -v

# Linting & Formatting
lint: ## Run ruff linting
	ruff check .

fmt: ## Format code with ruff
	ruff check . --fix

fmt-check: ## Check formatting without fixing
	ruff format --check .

type-check: ## Run mypy type checking
	mypy .

# Linting combined
quality: fmt lint type-check ## Run all quality checks

# Runtime commands
run-app: ## Run camera application
	python run_app.py

run-chat: ## Run desktop chat UI
	python run_chat.py

run-chat-web: ## Run web chat app
	python -m uvicorn web_app:app --host 127.0.0.1 --port 8000

run-chat-desktop: ## Run desktop chat UI
	python run_chat.py

run-menu: ## Run menu system
	python run_menu.py

run-doctor: ## Run system check
	python run_doctor.py --skip-camera-check

run-medical: ## Run medical CLI
	python run_medical.py

run-smoke: ## Run smoke test
	python run_smoke.py --ci-safe --stop-on-fail

doctor: run-doctor ## Alias for run-doctor

smoke: run-smoke ## Alias for run-smoke

smoke-ci: ## Run smoke test for CI
	python run_smoke.py --ci-safe --stop-on-fail

chat: run-chat-web ## Alias for run-chat-web

chat-desktop: run-chat-desktop ## Alias for run-chat-desktop

run-menu: ## Run menu system
	python run_menu.py

# Database
db-init: ## Initialize database
	python -m app.web_auth create-admin --username admin --email admin@hospital.com

db-migrate: ## Run migrations
	@echo "No migrations needed (SQLite auto-migration)"

db-shell: ## Open SQLite shell
	sqlite3 output/onco.db

db-backup: ## Backup database
	python run_medical.py backup-db

# Docker
docker-build: ## Build Docker image
	docker build -t onkovision:latest .

docker-run: ## Run container
	docker run -d --name onkovision \
		-p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/output:/app/output \
		-v $(PWD)/data:/app/data \
		onkovision:latest

docker-run-monai: ## Run container with MONAI support
	docker run -d --name onkovision-monai \
		-p 8000:8000 \
		--gpus all \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/output:/app/output \
		-v $(PWD)/data:/app/data \
		onkovision:latest-monai

docker-stop: ## Stop container
	docker stop onkovision || true
	docker rm onkovision || true

docker-logs: ## View container logs
	docker logs -f onkovision

docker-shell: ## Open shell in container
	docker exec -it onkovision bash

docker-build-monai: ## Build MONAI-enabled image
	docker build -f Dockerfile.monai -t onkovision:latest-monai .

# Cleanup
clean: ## Clean cache, build artifacts, temp files
	@echo "$(YELLOW)Cleaning...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ 2>/dev/null || true
	@rm -rf htmlcov/ .coverage 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@find . -type f -name "*.tmp" -delete 2>/dev/null || true
	@find . -type f -name "*.bak" -delete 2>/dev/null || true
	@rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)Clean complete$(NC)"

clean-deep: clean ## Deep clean including output and runs
	@rm -rf output/ runs/ .pytest_cache/ 2>/dev/null || true

# Help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'