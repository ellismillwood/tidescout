PY := $(HOME)/.venvs/tidescout/bin/python

.PHONY: install check test lint

install:
	uv pip install -p $(PY) -e './backend[dev]'

lint:
	cd backend && $(PY) -m ruff check .

test:
	cd backend && $(PY) -m pytest -q

check: lint test
