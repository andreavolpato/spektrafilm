"""Render the QA ``report.md`` as a self-contained ``report.html``.

A colorist receiving a bundle drops it on their desktop and double-
clicks: ``report.md`` opens in a text editor; ``report.html`` opens in
a browser with figures rendered inline. The two files coexist; the
HTML references the same ``figures/*.png`` via relative paths the
markdown already uses, so the bundle's on-disk layout is unchanged.

Implementation is intentionally minimal:

- Single dependency: ``markdown`` (Python-Markdown, the de-facto stdlib
  for md->html in Python — pure-Python, ~200KB, zero transitive deps).
- Three extensions enabled: ``tables`` (the report's summary table),
  ``fenced_code`` (defensive — the current report doesn't use code
  fences but future tests might), and ``toc`` (gives every heading an
  ``id="..."`` attribute so the in-page anchor links in the summary
  table resolve).
- A single inline ``<style>`` block (~40 lines of CSS) ships the
  typography, image scaling, and table styling. No external CSS file;
  the HTML stays portable for emailing or zipping.

See studies/a40_lut_system/n120_ocio_config_emission.md for the
discussion that led here (we picked Python-Markdown over pandoc /
LaTeX for the dependency-light path).
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown


# Anchor scheme used by the QA report: lowercase, ``_`` -> ``-``. Python-
# Markdown's ``toc`` extension uses a different default slug, so we pass
# a custom slugifier that matches what ``suite._anchor`` produces in the
# rendered markdown's in-page links.
def _slugify(text: str, _sep: str) -> str:
    return text.replace("_", "-").lower()


_EXTENSIONS = ["tables", "fenced_code", "toc"]
_EXT_CONFIGS = {"toc": {"slugify": _slugify}}


# Embedded CSS — kept short, readable, and palette-neutral so it
# doesn't fight the figures' own colors. Max-width container so the
# report stays readable on wide screens; tables get light borders;
# images scale to the column.
_INLINE_CSS = """
:root {
  color-scheme: light dark;
  --fg: #1f1f1f;
  --bg: #fbfbfb;
  --muted: #585858;
  --accent: #2a5d9e;
  --border: #d8d8d8;
  --code-bg: #f1f1f1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --fg: #ececec;
    --bg: #161616;
    --muted: #a8a8a8;
    --accent: #79a6e0;
    --border: #2e2e2e;
    --code-bg: #232323;
  }
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  color: var(--fg);
  background: var(--bg);
  max-width: 960px;
  margin: 2rem auto;
  padding: 0 1.5rem 4rem;
}
h1, h2, h3, h4 { color: var(--fg); line-height: 1.25; margin-top: 1.6em; }
h1 { font-size: 1.85em; }
h2 { font-size: 1.4em; border-bottom: 1px solid var(--border); padding-bottom: 0.2em; }
h3 { font-size: 1.15em; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
img { max-width: 100%; height: auto; display: block; margin: 1em 0; }
table { border-collapse: collapse; margin: 1em 0; font-size: 0.95em; }
th, td { border: 1px solid var(--border); padding: 0.45em 0.85em; text-align: left; }
th { background: var(--code-bg); font-weight: 600; }
code { background: var(--code-bg); padding: 0.12em 0.35em; border-radius: 3px;
       font-size: 0.92em; font-family: "SF Mono", Consolas, Menlo, monospace; }
pre { background: var(--code-bg); padding: 0.85em 1em; border-radius: 5px;
      overflow-x: auto; }
pre code { background: transparent; padding: 0; }
hr { border: none; border-top: 1px solid var(--border); margin: 2em 0; }
ul, ol { padding-left: 1.6em; }
"""


def report_md_to_html(
    md_path: Path,
    html_path: Path | None = None,
    *,
    title: str | None = None,
) -> Path:
    """Convert ``md_path`` to a self-contained HTML file.

    Returns the resolved ``html_path``. Image references in the
    markdown are emitted as-is in the HTML, so they continue to
    resolve against the directory containing ``html_path``
    (typically the same directory as ``md_path``).

    ``title`` defaults to the first H1 in the markdown, or the
    markdown file's stem if no H1 is found.
    """
    md_path = Path(md_path)
    if html_path is None:
        html_path = md_path.with_suffix(".html")
    md_text = md_path.read_text(encoding="utf-8")
    if title is None:
        title = _extract_title(md_text) or md_path.stem

    body_html = markdown.markdown(
        md_text,
        extensions=_EXTENSIONS,
        extension_configs=_EXT_CONFIGS,
        output_format="html",
    )
    html_text = _wrap(title=title, body=body_html)
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def _extract_title(md_text: str) -> str | None:
    """Return the first ATX-style H1 heading's text, or None."""
    for line in md_text.splitlines():
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            # Strip backticks (code-quote markers) from the title so it's
            # readable as a browser tab name. Leave underscores alone —
            # they belong inside identifiers like `test_bundle`.
            return m.group(1).replace("`", "").strip()
    return None


def _wrap(*, title: str, body: str) -> str:
    """Assemble the HTML5 document with inline CSS."""
    # Escape the title for the <title> element only — body HTML is
    # trusted markdown output.
    escaped_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{escaped_title}</title>\n"
        "  <style>\n"
        f"{_INLINE_CSS}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )
