.DEFAULT_GOAL := help
SHELL := /bin/bash
EXECUTABLES = poetry
K := $(foreach exec,$(EXECUTABLES),\
        $(if $(shell which $(exec)),some string,$(error "You must install $(exec) to use this Makefile")))

.PHONY: all
all: help lint tests ## Run all

.PHONY: help
help: ## Install make to use this file, then write "make [command]".
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# LINT

.PHONY: lint
lint: toml_sort isort black flake8 pylint mypy ## Run lint

.PHONY: toml_sort
toml_sort: ## Run toml_sort
	poetry run toml-sort pyproject.toml --all --in-place

.PHONY: isort
isort: ## Run isort
	poetry run isort .

.PHONY: black
black: ## Run black
	poetry run black .

.PHONY: flake8
flake8: ## Run flake8
	poetry run flake8 .

.PHONY: pylint
pylint: ## Run pylint
	poetry run pylint src

.PHONY: mypy
mypy: ## Run mypy
	poetry run mypy --install-types --non-interactive .

# TESTS

.PHONY: test
test: ## Run test
	poetry run pytest

.PHONY: tests
tests: test ## Run tests

# POETRY

.PHONY: install_dependencies
install_dependencies: ## Install dependencies
	poetry install --no-root

.PHONY: update_dependencies
update_dependencies: ## Update dependencies
	poetry update
	poetry install --no-root

.PHONY: shell
shell: ## Enter poetry virtualenv
	poetry shell

.PHONY: install
install: ## Install package into virtualenv
	poetry install
