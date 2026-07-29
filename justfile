set dotenv-load

default:
    @just --list

[doc('install project dependencies')]
[group('setup')]
setup:
    mise install
    mise reshim
    uv sync
    vale sync
    prek install

[doc('update prek hooks to latest versions')]
[group('hooks')]
hooks-update:
    prek update

[doc('run hooks against staged files')]
[group('hooks')]
hooks-run:
    prek run

[doc('run hooks against all files')]
[group('hooks')]
hooks-run-all:
    prek run --all-files --verbose

[doc('format all code')]
[group('format')]
format:
    ruff format src/ tests/ scripts/
    biome format --write src/
    just --fmt

[doc('lint all code')]
[group('lint')]
lint:
    ruff check src/ tests/ scripts/
    ty check src/
    biome ci src/
    vale --no-global $(git ls-files --cached --others --exclude-standard '*.md') 2>&1
    just --fmt --check

[doc('run fast tests')]
[group('test')]
test:
    uv run pytest

[doc('run all tests including slow')]
[group('test')]
test-all:
    uv run pytest --run-slow

[doc('run spektrafilm GUI')]
[group('run')]
run:
    uv run spektrafilm

[doc('run full CI pipeline: format, lint, test')]
[group('checks')]
check-all: format lint test

[doc('clean cache')]
[group('clean')]
clean:
    uv cache clean
