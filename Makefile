# Variables
UVX = uvx
RUFF = $(UVX) ruff@0.8.6


.PHONY: check
check:
	$(RUFF) check .

.PHONY: fix
fix:
	$(RUFF) check . --fix

.PHONY: fmt
fmt:
	$(RUFF) format .

.PHONY: docker-up
docker-up:
	docker compose up -d
	./scripts/healthcheck.sh

.PHONY: docker-down
docker-down:
	docker compose down --rmi local

.PHONY: integration
integration:
	uv run pytest ./tests/integration_tests/

.PHONY: unit
unit:
	uv run pytest ./tests/unit_tests/
