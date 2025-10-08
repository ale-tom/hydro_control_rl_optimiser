# Makefile for a template pyproject.toml (setuptools) project using a "src" layout.
# Defaults to Python 3.12; override with: make venv PYTHON_BIN=/path/to/python

PYTHON_BIN ?= python3.12
VENV_DIR ?= .venv
BIN := $(VENV_DIR)/bin

PIP := $(BIN)/pip
PY := $(BIN)/python
PYTEST := $(BIN)/pytest
RUFF := $(BIN)/ruff
BLACK := $(BIN)/black
MYPY := $(BIN)/mypy

.DEFAULT_GOAL := help

.PHONY: help venv install fmt lint test build clean clean-venv run

help:
	@echo "Targets:"
	@echo "  venv        - Create virtual environment using $(PYTHON_BIN)"
	@echo "  install     - Install project (editable) + dev extras"
	@echo "  fmt         - Black + Ruff --fix"
	@echo "  lint        - Ruff + mypy"
	@echo "  test        - Pytest"
	@echo "  build       - Build wheel and sdist"
	@echo "  run         - Run src/main.py"
	@echo "  clean       - Remove build & cache artifacts"
	@echo "  clean-venv  - Remove .venv"

venv:
	$(PYTHON_BIN) -m venv $(VENV_DIR)
	$(PIP) install -U pip

install: venv
	$(PIP) install -e ".[dev]"

fmt:
	$(BLACK) .
	$(RUFF) check . --fix

lint:
	$(RUFF) check .
	$(MYPY) src tests

test:
	$(PYTEST) -q

build:
	$(PY) -m build

clean:
	rm -rf build dist .pytest_cache .mypy_cache *.egg-info **/*.egg-info

clean-venv:
	rm -rf $(VENV_DIR)

run:
	$(PY) src/main.py
