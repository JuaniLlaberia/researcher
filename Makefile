# Project tasks. `make up` is the single command to bring up the database and
# ensure the schema exists (idempotent — safe to run repeatedly).

PYTHON ?= .venv/bin/python

.PHONY: up down db init-db logs

## up: start containers, wait for the DB, then create any missing tables
up:
	docker compose up -d
	./scripts/wait_for_db.sh
	$(PYTHON) scripts/init_db.py

## db: start only the database container, wait, then ensure schema
db:
	docker compose up -d db
	./scripts/wait_for_db.sh
	$(PYTHON) scripts/init_db.py

## init-db: create any missing tables (assumes DB already running)
init-db:
	$(PYTHON) scripts/init_db.py

## down: stop containers (keeps the data volume)
down:
	docker compose down

## logs: tail database logs
logs:
	docker compose logs -f db
