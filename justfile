set dotenv-load

default:
    @just --list

[group('setup')]
[doc('install project dependencies')]
setup:
    mise install
    mise reshim
    uv sync
    prek install

[group('hooks')]
[doc('update prek hooks to latest versions')]
hooks-update:
    prek update

[group('hooks')]
[doc('run hooks against staged files')]
hooks-run:
    prek run

[group('hooks')]
[doc('run hooks against all files')]
hooks-run-all:
    prek run --all-files --verbose

[group('format')]
[doc('format all code')]
format:
    ruff format src/ tests/ scripts/
    biome format --write src/
    just --fmt

[group('lint')]
[doc('lint all code')]
lint:
    ruff check src/ tests/ scripts/
    ty check src/
    biome ci src/
    just --fmt --check

[group('test')]
[doc('run fast tests')]
test:
    uv run pytest

[group('test')]
[doc('run all tests including slow')]
test-all:
    uv run pytest --run-slow

[group('checks')]
[doc('run full CI pipeline: format, lint, test')]
check-all: format lint test

[group('clean')]
[doc('clean cache')]
clean:
    uv cache clean
