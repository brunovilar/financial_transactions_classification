#!/usr/bin/make -f


# Base Commands
##############################################

MAKE += --no-print-directory RECURSIVE=1

COMPOSE := docker-compose


# Docker Compose Basics
##############################################

build:
	@$(COMPOSE) build application
	@$(COMPOSE) build mlflow-server

clean:
	@$(COMPOSE) down -v

up:
	@$(COMPOSE) up


# Lint and Format
##############################################

isort:
	@$(COMPOSE) run -T --rm --entrypoint isort application .

black:
	@$(COMPOSE) run -T --rm --entrypoint black application .

autoflake:
	@$(COMPOSE) run -T --rm --entrypoint autoflake application \
		--in-place --remove-all-unused-imports --remove-unused-variables \
		--ignore-init-module-imports --expand-star-imports --recursive \
		.

mypy:
	@$(COMPOSE) run -T --rm --entrypoint mypy application --show-error-codes ./application ./tests

flake8:
	@$(COMPOSE) run -T --rm --entrypoint flake8 application --ignore=F811 ./application ./tests

lint:
	@$(MAKE) mypy
	@$(MAKE) flake8

format:
	@$(MAKE) black
	@$(MAKE) isort
	@$(MAKE) autoflake


# MLFlow
##############################################

start-mlflow:
	@$(COMPOSE) up -d mlflow-server


# Tests
##############################################

unit-tests:
	@$(COMPOSE) run --rm --entrypoint pytest application -v tests/unit -s -vv

integration-tests:
	@$(MAKE) start-mlflow
	@$(COMPOSE) run --rm --entrypoint pytest application -v tests/integration -s -vv

all-tests:
	@$(MAKE) clean
	@$(MAKE) build
	@$(MAKE) unit-tests
	@$(MAKE) integration-tests
