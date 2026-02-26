# Run 'make test-unit' during development for fast feedback.
# Run 'make test-all' before committing.
# Run 'make test-contract' whenever you update MockDatabricksClient
# or the Databricks model's predict method return schema.

.PHONY: test-unit test-integration test-contract test-all test-fast

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-contract:
	pytest tests/contracts/ -v -m contract

test-all:
	pytest tests/ -v --cov=app --cov-report=term-missing \
		--cov-report=html:coverage_html --cov-fail-under=70

test-fast:
	pytest tests/unit/ tests/contracts/ -v
