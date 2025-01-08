# Variables
UVX = uvx
RUFF = $(UVX) ruff@0.8.6


.PHONY: check
check:
	$(RUFF) check .

.PHONY: check-fix
check-fix:
	$(RUFF) check . --fix

.PHONY: fmt
fmt:
	$(RUFF) format .
