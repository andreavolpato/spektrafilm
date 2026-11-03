"""GUI test configuration.

The GUI ships with Chinese as the default interface language; the test
suite asserts the original English strings, so pin the language here.
"""

import pytest


@pytest.fixture(autouse=True)
def _english_ui(monkeypatch):
    monkeypatch.setenv('SPEKTRAFILM_LANG', 'en')
