"""Tests for the delivery-target registry and its integration with
:class:`BundleSpec` and :class:`BundleBuilder`.

The registry is the single source of truth for vendor / camera-specific
LUT delivery requirements. These tests verify that the catalog is
well-formed, that BundleSpec validates against the target's allowed
inputs and outputs, and that the builder emits the target-specific file
in addition to the generic .cube.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spektrafilm_lut_creator.delivery_targets import (
    DeliveryTarget,
    list_targets,
    register,
)
from spektrafilm_lut_creator.delivery_targets import get as get_target

from .factories import make_bundle_spec

_LUT_LICENSE_PATH = Path(__file__).resolve().parents[2] / "SPEKTRAFILM_LICENSE.txt"


class TestRegistry:
    def test_lumix_realtime_vlog_is_registered(self):
        target = get_target("lumix_realtime_vlog")
        assert target.format == "lumix"
        assert "Panasonic V-Log" in target.valid_inputs
        assert "sRGB" in target.valid_outputs
        assert target.writer_kwargs.get("photo_style_tag") == "VLOG"
        assert target.cameras  # at least one model listed
        assert target.verified  # has a verification note

    def test_list_targets_returns_sorted(self):
        targets = list_targets()
        assert "lumix_realtime_vlog" in targets
        assert targets == sorted(targets)

    def test_unknown_target_raises_keyerror(self):
        with pytest.raises(KeyError, match="lumix_realtime_vlog"):
            get_target("does_not_exist")

    def test_register_rejects_unknown_format(self):
        with pytest.raises(KeyError, match="unknown format"):
            register(
                DeliveryTarget(
                    name="invalid_test",
                    description="x",
                    format="not_a_real_format",
                    valid_inputs=("sRGB",),
                    valid_outputs=("sRGB",),
                )
            )

    def test_register_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="valid_inputs"):
            register(
                DeliveryTarget(
                    name="invalid_test_inputs",
                    description="x",
                    format="cube",
                    valid_inputs=(),
                    valid_outputs=("sRGB",),
                )
            )


class TestBundleSpecTargetValidation:
    def test_accepts_valid_combination(self):
        # No error.
        make_bundle_spec(
            name="ok",
            input_color_space="Panasonic V-Log",
            target="lumix_realtime_vlog",
        )

    def test_rejects_input_outside_target(self):
        with pytest.raises(ValueError, match="requires input"):
            make_bundle_spec(
                name="bad_input",
                input_color_space="ACEScg",  # not in valid_inputs for Lumix target
                target="lumix_realtime_vlog",
            )

    def test_rejects_output_outside_target(self):
        with pytest.raises(ValueError, match="requires output"):
            make_bundle_spec(
                name="bad_output",
                input_color_space="Panasonic V-Log",
                output_color_space="Adobe RGB",  # not in valid_outputs for Lumix
                target="lumix_realtime_vlog",
            )

    def test_unknown_target_name_raises(self):
        with pytest.raises(KeyError, match="not_a_target"):
            make_bundle_spec(
                name="x",
                input_color_space="sRGB",
                target="not_a_target",
            )

    def test_none_target_is_unconstrained(self):
        # No target set: any input/output combination from the color-space
        # registry is allowed (only registry-level validation applies).
        make_bundle_spec(
            name="any",
            input_color_space="ACEScg",
            output_color_space="Rec.2020",
            target=None,
        )


class TestBuilderEmitsTargetVariant:
    """A bundle built with a target also writes the target's file format."""

    def test_target_set_writes_only_target_format(self, tmp_path):
        """When a target is set, the bundle contains only the target's
        format file (no generic Adobe sibling). Filename is canonical."""
        from spektrafilm_lut_creator.builders import BundleBuilder

        spec = make_bundle_spec(
            name="target_test",
            input_color_space="Panasonic V-Log",
            target="lumix_realtime_vlog",
            resolution=5,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out_dir = builder.write(bundle, tmp_path / "out")

        rel_path, _ = bundle.luts[0]
        assert (out_dir / rel_path).exists()
        assert (out_dir / "bundle.json").exists()
        readme = out_dir / "README.md"
        assert readme.exists()
        readme_text = readme.read_text(encoding="utf-8")
        assert "Panasonic V-Log" in readme_text
        assert "sRGB" in readme_text
        assert "lumix_realtime_vlog" in readme_text

        # Exactly one cube file (target-only emission).
        cubes = list(out_dir.glob("*.cube"))
        assert len(cubes) == 1, f"expected one cube file, got {cubes}"

        # File has the Lumix-strict layout with #LUMIXPHOTOSTYLE.
        text = (out_dir / rel_path).read_text(encoding="utf-8")
        assert "#LUMIXPHOTOSTYLE VLOG" in text
        # No rich provenance comment block in Lumix-strict files.
        assert "spektrafilm LUT" not in text

        import json

        payload = json.loads((out_dir / "bundle.json").read_text(encoding="utf-8"))
        assert payload["target"] == "lumix_realtime_vlog"

    def test_no_target_writes_generic_cube_with_provenance(self, tmp_path):
        """When target is None, the bundle uses the generic Adobe .cube
        format with the rich provenance comment header."""
        from spektrafilm_lut_creator.builders import BundleBuilder

        spec = make_bundle_spec(
            name="no_target",
            target=None,
            resolution=5,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out_dir = builder.write(bundle, tmp_path / "out")

        rel_path, _ = bundle.luts[0]
        assert (out_dir / rel_path).exists()
        assert (out_dir / "bundle.json").exists()

        text = (out_dir / rel_path).read_text(encoding="utf-8")
        # Generic format carries the rich provenance header.
        assert "spektrafilm LUT" in text
        assert "CC BY-SA 4.0" in text

    def test_target_write_copies_lut_license(self, tmp_path):
        from spektrafilm_lut_creator.builders import BundleBuilder

        spec = make_bundle_spec(
            name="target_license",
            input_color_space="Panasonic V-Log",
            target="lumix_realtime_vlog",
            resolution=5,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out_dir = builder.write(bundle, tmp_path / "target_license")

        assert (out_dir / "SPEKTRAFILM_LICENSE.txt").read_text(
            encoding="utf-8"
        ) == _LUT_LICENSE_PATH.read_text(encoding="utf-8")

    def test_no_target_write_copies_lut_license(self, tmp_path):
        from spektrafilm_lut_creator.builders import BundleBuilder

        spec = make_bundle_spec(
            name="generic_license",
            target=None,
            resolution=5,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out_dir = builder.write(bundle, tmp_path / "generic_license")

        assert (out_dir / "SPEKTRAFILM_LICENSE.txt").read_text(
            encoding="utf-8"
        ) == _LUT_LICENSE_PATH.read_text(encoding="utf-8")
