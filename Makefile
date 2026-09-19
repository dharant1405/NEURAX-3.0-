.PHONY: setup data generate run test clean

PYTHON := python
PIP    := pip
VENV   := .venv

# ─────────────────────────────────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m pytest --version

# ─────────────────────────────────────────────────────────────────────────────
generate:
	@echo "Generating synthetic dataset in data/synthetic/..."
	$(PYTHON) data/synthetic/generator.py

# ─────────────────────────────────────────────────────────────────────────────
data: generate

# ─────────────────────────────────────────────────────────────────────────────
run:
	@echo "Starting InspectIQ dashboard..."
	streamlit run dashboard/app.py

# ─────────────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ─────────────────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning cache and generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/ || true
