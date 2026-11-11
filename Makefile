.PHONY: install lint format typecheck test check smoke clean

PYTHON ?= python

install:  ## Editable install with dev tools and the MCP extra
	$(PYTHON) -m pip install -e ".[dev,mcp]"

lint:  ## Lint and verify formatting (must be clean in CI)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-format and apply safe lint fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:  ## mypy in strict mode over src/
	$(PYTHON) -m mypy

test:  ## Unit and integration tests (no GPU, no model downloads)
	$(PYTHON) -m pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs

smoke:  ## Full pipeline with fake models and real ffmpeg, in seconds
	./scripts/smoke.sh

clean:
	rm -rf build dist .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov
