# CDMConnector (Python) - developer targets
# Override port if in use: PORT=8080 make docs
PORT ?= 4522

.PHONY: install install-dev lint test test-integration test-live test-cov cov coverage docs serve site clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

lint:
	ruff check src tests

test:
	pytest tests -v --tb=short

test-integration:
	pytest tests -m integration -v --tb=short

# Run live DB tests against specific backends (default: duckdb).
# Example: CDMCONNECTOR_TEST_DB=duckdb,postgres make test-live
test-live:
	pytest tests/test_cdm_live.py tests/test_generate_cohort_set_live.py -m integration -v --tb=short

test-cov:
	pytest tests -v --cov=src/cdmconnector --cov-report=term-missing

cov:
	pytest tests -v --cov=src/cdmconnector --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Overall and per-file coverage above; HTML report in htmlcov/index.html"

coverage:
	pytest tests -v --cov=src/cdmconnector --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo ""
	@echo "Coverage reports: terminal (above), HTML (htmlcov/index.html), XML (coverage.xml)"

docs:
	cd docs-src && quartodoc build
	cd docs-src && quarto render .
	touch docs/.nojekyll
	mkdir -p docs/.well-known/appspecific && cp docs-src/.well-known/appspecific/com.chrome.devtools.json docs/.well-known/appspecific/ 2>/dev/null || echo '{}' > docs/.well-known/appspecific/com.chrome.devtools.json
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null || true
	@echo "Serving docs at http://localhost:$(PORT)/ (Ctrl+C to stop)"
	@cd docs && python -m http.server $(PORT) & \
	sleep 2 && open http://localhost:$(PORT)/ && wait

serve:
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null || true
	@echo "Serving docs at http://localhost:$(PORT)/"
	cd docs && python -m http.server $(PORT)

site:
	quarto preview docs-src/

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
