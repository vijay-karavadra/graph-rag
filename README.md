# graph-pancake

## Development

### UV

This project is managed with [UV](https://docs.astral.sh/uv/)

Make sure you have > 0.5.0 of UV installed.

* Use `uv python install` to install the project python version
* Use `uv sync` to install the dependencies into a virtual environment

### Testing

Before running integration tests, copy the `.env.template` file to `.env` and fill it out.

## Helpers

There is a Makefile with some standard commands:

### Linting

* `make fmt` Uses Ruff to format all code
* `make check` Uses Ruff to check all code
* `make fix` Uses Ruff to check and attempt to fix all code
* `make mypy` Runs the Mypy static type check.

`make lint`: Does `fmt`, `fix`, and `mypy` in a single command

### Testing

* `make docker-up` Launch containers for integration tests
* `make docker-down` Shutdown containers for integration tests
* `make integration` Run integration tests
* `make unit` Run unit tests
