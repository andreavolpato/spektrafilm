"""Tests for the spektrafilm-lut CLI."""

from __future__ import annotations

import json
import textwrap

import pytest

from spektrafilm_lut_creator import cli

# ---------------------------------------------------------------------------
# Slug resolution.
# ---------------------------------------------------------------------------


class TestSlugResolution:
    def test_canonical_name_round_trips(self):
        assert (
            cli.resolve_color_space("Panasonic V-Log", role="input")
            == "Panasonic V-Log"
        )
        assert cli.resolve_color_space("sRGB", role="output") == "sRGB"

    def test_slug_resolves_to_canonical(self):
        assert cli.resolve_color_space("vlog", role="input") == "Panasonic V-Log"
        assert cli.resolve_color_space("srgb", role="output") == "sRGB"
        assert cli.resolve_color_space("acescg", role="input") == "ACEScg"
        assert cli.resolve_color_space("rec2020", role="output") == "Rec.2020"

    def test_unknown_name_errors_with_role_hint(self):
        with pytest.raises(ValueError, match="unknown input color space 'nope'"):
            cli.resolve_color_space("nope", role="input")


# ---------------------------------------------------------------------------
# `list` subcommand.
# ---------------------------------------------------------------------------


class TestListSubcommand:
    def test_list_input_includes_known_spaces(self, capsys):
        rc = cli.main(["list", "input"])
        assert rc == 0
        lines = capsys.readouterr().out.splitlines()
        # Each line is "<canonical name>  <slug>"; check both forms
        # appear so the user sees what they can paste into --input.
        assert any(
            line.startswith("Panasonic V-Log") and line.endswith("vlog")
            for line in lines
        )
        assert any(
            line.startswith("ACEScct") and line.endswith("acescct") for line in lines
        )
        assert any(line.startswith("sRGB") and line.endswith("srgb") for line in lines)

    def test_list_output_excludes_input_only_spaces(self, capsys):
        rc = cli.main(["list", "output"])
        assert rc == 0
        lines = capsys.readouterr().out.splitlines()
        assert any(line.startswith("sRGB") for line in lines)
        # ACEScg is input-only; must not appear under outputs.
        assert not any(line.startswith("ACEScg ") or line == "ACEScg" for line in lines)

    def test_list_target_includes_lumix(self, capsys):
        rc = cli.main(["list", "target"])
        assert rc == 0
        names = capsys.readouterr().out.splitlines()
        assert "lumix_realtime_vlog" in names

    def test_list_film_includes_known_stock(self, capsys):
        rc = cli.main(["list", "film"])
        assert rc == 0
        names = capsys.readouterr().out.splitlines()
        assert "kodak_portra_400" in names
        # A print must not appear under films.
        assert "kodak_portra_endura" not in names

    def test_list_print_includes_known_print(self, capsys):
        rc = cli.main(["list", "print"])
        assert rc == 0
        names = capsys.readouterr().out.splitlines()
        assert "kodak_portra_endura" in names
        assert "kodak_portra_400" not in names


# ---------------------------------------------------------------------------
# `build` subcommand — argument plumbing and BundleSpec construction.
#
# These tests build at resolution=5 to keep the suite fast; correctness
# of the bake is covered elsewhere (test_builders.py).
# ---------------------------------------------------------------------------


class TestBuildSubcommand:
    def test_build_with_slugs_creates_bundle(self, tmp_path, capsys):
        rc = cli.main(
            [
                "build",
                "--film",
                "kodak_portra_400",
                "--print",
                "kodak_portra_endura",
                "--input",
                "vlog",
                "--output",
                "srgb",
                "--resolution",
                "5",
                "--out",
                str(tmp_path),
            ]
        )
        assert rc == 0
        # Auto-named bundle directory landed at out/<name>/.
        children = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(children) == 1
        bundle_dir = children[0]
        assert (bundle_dir / "bundle.json").exists()
        assert (bundle_dir / "SPEKTRAFILM_LICENSE.txt").exists()
        meta = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        assert meta["color_spaces"]["input"]["name"] == "Panasonic V-Log"
        assert meta["color_spaces"]["output"]["name"] == "sRGB"
        out = capsys.readouterr().out
        assert "[done]" in out

    def test_build_with_canonical_names_works(self, tmp_path):
        rc = cli.main(
            [
                "build",
                "--film",
                "kodak_portra_400",
                "--print",
                "kodak_portra_endura",
                "--input",
                "Panasonic V-Log",
                "--output",
                "sRGB",
                "--resolution",
                "5",
                "--out",
                str(tmp_path),
            ]
        )
        assert rc == 0

    def test_build_unknown_color_space_returns_error(self, tmp_path, capsys):
        rc = cli.main(
            [
                "build",
                "--film",
                "kodak_portra_400",
                "--print",
                "kodak_portra_endura",
                "--input",
                "vlogggg",
                "--output",
                "srgb",
                "--resolution",
                "5",
                "--out",
                str(tmp_path),
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "unknown input color space" in err

    def test_build_missing_required_field_reports_clearly(self, tmp_path, capsys):
        rc = cli.main(
            [
                "build",
                "--film",
                "kodak_portra_400",
                # missing --print
                "--input",
                "vlog",
                "--output",
                "srgb",
                "--out",
                str(tmp_path),
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "missing --print" in err

    def test_build_repeats_print_for_multi_print(self, tmp_path):
        rc = cli.main(
            [
                "build",
                "--film",
                "kodak_portra_400",
                "--print",
                "kodak_portra_endura",
                "--print",
                "fujifilm_crystal_archive_typeii",
                "--input",
                "vlog",
                "--output",
                "srgb",
                "--topology",
                "1lut",
                "--resolution",
                "5",
                "--out",
                str(tmp_path),
            ]
        )
        assert rc == 0

    def test_build_from_toml(self, tmp_path):
        spec_file = tmp_path / "spec.toml"
        spec_file.write_text(
            textwrap.dedent("""
            film_profile = "kodak_portra_400"
            print_profiles = ["kodak_portra_endura"]
            input_color_space = "vlog"
            output_color_space = "srgb"
            topology = "1lut"
            resolution = 5
        """).strip(),
            encoding="utf-8",
        )
        rc = cli.main(
            [
                "build",
                "--from",
                str(spec_file),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 0
        bundle_dirs = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
        assert len(bundle_dirs) == 1

    def test_cli_overrides_toml(self, tmp_path):
        """`--resolution 7` on the CLI wins over `resolution = 5` in TOML."""
        spec_file = tmp_path / "spec.toml"
        spec_file.write_text(
            textwrap.dedent("""
            film_profile = "kodak_portra_400"
            print_profiles = ["kodak_portra_endura"]
            input_color_space = "vlog"
            output_color_space = "srgb"
            resolution = 5
        """).strip(),
            encoding="utf-8",
        )
        rc = cli.main(
            [
                "build",
                "--from",
                str(spec_file),
                "--resolution",
                "7",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 0
        bundle_dirs = list((tmp_path / "out").iterdir())
        meta = json.loads((bundle_dirs[0] / "bundle.json").read_text(encoding="utf-8"))
        assert meta["resolution"] == 7

    def test_build_toml_with_input_table_and_output_off_string(self, tmp_path):
        spec_file = tmp_path / "spec.toml"
        spec_file.write_text(
            textwrap.dedent("""
            film_profile = "kodak_portra_400"
            print_profiles = ["kodak_portra_endura"]
            input_color_space = "vlog"
            output_color_space = "srgb"
            resolution = 5
            output_gamut_compress = "off"

            [input_gamut_compress]
            active = false
        """).strip(),
            encoding="utf-8",
        )
        rc = cli.main(
            [
                "build",
                "--from",
                str(spec_file),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 0
