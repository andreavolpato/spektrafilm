"""Minimal translation layer for the desktop GUI.

Catalogs live in ``locales/<language>.json`` and map the English source text
to its translation. English is the source language and needs no catalog; a
string missing from a catalog falls back to English.

The language follows the system locale. Set ``SPEKTRAFILM_LANG`` (``en`` or
``zh_CN``) to override it.

Only displayed text goes through ``tr``. Values stored in the runtime
parameters and the saved GUI state stay in English.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from qtpy import QtCore

LANGUAGE_ENV_VAR = 'SPEKTRAFILM_LANG'
DEFAULT_LANGUAGE = 'en'
_LOCALES_DIR = Path(__file__).with_name('locales')


def _resolve_language(locale_name: str) -> str:
    if locale_name.lower().replace('-', '_').startswith('zh'):
        return 'zh_CN'
    return DEFAULT_LANGUAGE


def current_language() -> str:
    requested = os.environ.get(LANGUAGE_ENV_VAR, '').strip()
    return _resolve_language(requested or QtCore.QLocale.system().name())


def _key(text: str) -> str:
    return text.strip().lower()


@lru_cache(maxsize=1)
def _catalog() -> dict[str, str]:
    language = current_language()
    if language == DEFAULT_LANGUAGE:
        return {}
    catalog_path = _LOCALES_DIR / f'{language}.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    return {_key(source): translation for source, translation in catalog.items()}


def tr(text: str, **fields: object) -> str:
    """Translate ``text``, matching it regardless of case, and fill ``{fields}``."""
    translated = _catalog().get(_key(text), text)
    return translated.format(**fields) if fields else translated
