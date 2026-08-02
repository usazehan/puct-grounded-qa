.PHONY: up down probe test fmt

up:
	docker compose up -d --build

down:
	docker compose down

probe:
	python scripts/probe_extraction.py $(or $(DIR),data/raw)

test:
	PYTHONPATH=src pytest -q

psql:
	docker compose exec db psql -U puctqa -d puctqa
