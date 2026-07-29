"""Tests for the QA report.md -> report.html conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from spektrafilm_lut_creator.qa.html_export import report_md_to_html


@pytest.fixture
def sample_report_md(tmp_path: Path) -> Path:
    """A minimal report.md exercising the structural features the real
    suite emits: H1/H2 headings, in-page anchor links, summary tables,
    image references, code-quoted text."""
    md = tmp_path / "report.md"
    md.write_text(
        "# QA report — `test_bundle`\n"
        "\n"
        "- **Print**: `kodak_portra_endura`\n"
        "- **Topology**: `1lut`\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Test | Status | Headline numbers |\n"
        "|---|---|---|\n"
        "| [off_grid_identity](#off-grid-identity) | ✅ PASS | max ΔE = 0.84 |\n"
        "| [monotonicity](#monotonicity) | ❌ FAIL | 3 violations |\n"
        "\n"
        "## off_grid_identity\n"
        "\n"
        "**Status**: ✅ PASS\n"
        "\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        "| `max_delta_e_2000` | 0.84 |\n"
        "\n"
        "![off_grid_identity](figures/off_grid_identity.png)\n"
        "\n"
        "Interpretation paragraph.\n"
        "\n"
        "## monotonicity\n"
        "\n"
        "**Status**: ❌ FAIL\n",
        encoding="utf-8",
    )
    return md


class TestReportMdToHtml:
    def test_writes_sibling_html_file(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        assert html == sample_report_md.with_suffix(".html")
        assert html.is_file()

    def test_html_starts_with_doctype(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        text = html.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")
        assert "<title>" in text
        assert "</html>" in text.rstrip()

    def test_title_extracted_from_first_h1(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        text = html.read_text(encoding="utf-8")
        # The backticks in the H1 are stripped for the <title> element.
        assert "<title>QA report — test_bundle</title>" in text

    def test_inline_css_block_present(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        text = html.read_text(encoding="utf-8")
        # A few load-bearing CSS rules — if any of these disappear the
        # report stops looking like a report.
        assert "<style>" in text
        assert "max-width:" in text
        assert "table {" in text or "table{" in text

    def test_summary_table_is_rendered(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        text = html.read_text(encoding="utf-8")
        assert "<table>" in text
        assert "<th>Test</th>" in text
        assert "off_grid_identity" in text
        assert "monotonicity" in text

    def test_image_path_preserved_relative(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        text = html.read_text(encoding="utf-8")
        # The image src must stay relative so the HTML resolves figures
        # from its sibling figures/ directory.
        assert 'src="figures/off_grid_identity.png"' in text
        assert "http" not in text.split('src="figures/')[1].split('"')[0]

    def test_anchor_link_resolves_to_heading_id(self, sample_report_md):
        html = report_md_to_html(sample_report_md)
        text = html.read_text(encoding="utf-8")
        # The summary table's [link](#off-grid-identity) and the
        # corresponding h2's id="..." must agree, otherwise the
        # in-page navigation breaks.
        assert 'href="#off-grid-identity"' in text
        assert 'id="off-grid-identity"' in text

    def test_custom_html_path_honored(self, sample_report_md, tmp_path):
        custom = tmp_path / "out" / "renamed.html"
        custom.parent.mkdir()
        returned = report_md_to_html(sample_report_md, html_path=custom)
        assert returned == custom
        assert custom.is_file()

    def test_custom_title_overrides_h1(self, sample_report_md):
        html = report_md_to_html(sample_report_md, title="Overridden")
        text = html.read_text(encoding="utf-8")
        assert "<title>Overridden</title>" in text
