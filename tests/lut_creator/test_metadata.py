"""Tests for the bundle.json schema dataclasses (shape only; I/O lands later)."""

from __future__ import annotations

from spektrafilm_lut_creator.metadata import (
    SCHEMA_VERSION,
    BundleMeta,
    ColorSpaceMeta,
    LutFileMeta,
    StocksMeta,
    WiresMeta,
)
from spektrafilm_lut_creator.wires import DensityWire, LogEWire


def test_default_bundle_meta_has_current_schema_version():
    meta = BundleMeta()
    assert meta.schema_version == SCHEMA_VERSION


def test_two_lut_bundle_construction():
    meta = BundleMeta(
        name="portra400_5prints",
        topology="2lut",
        resolution=33,
        stocks=StocksMeta(
            film="kodak_portra_400", prints=("kodak_endura", "fuji_crystal_archive")
        ),
        color_spaces={
            "input": ColorSpaceMeta(name="ACEScg", cctf=False),
            "output": ColorSpaceMeta(name="sRGB", cctf=True),
        },
        wires=WiresMeta(
            cmy_film=DensityWire(d_max=(3.8, 4.1, 3.6)),
        ),
        luts=(
            LutFileMeta(
                role="film", path="film.cube", domain="input_rgb", range="cmy_film"
            ),
            LutFileMeta(
                role="print",
                print_profile="kodak_endura",
                path="prints/kodak_endura/print.cube",
                domain="cmy_film",
                range="output_rgb",
            ),
        ),
    )

    assert meta.topology == "2lut"
    assert meta.stocks.prints == ("kodak_endura", "fuji_crystal_archive")
    assert meta.color_spaces["input"].name == "ACEScg"
    assert meta.color_spaces["input"].cctf is False
    assert meta.wires.cmy_film.d_max == (3.8, 4.1, 3.6)
    assert meta.wires.log_e_film is None
    assert len(meta.luts) == 2
    assert meta.luts[1].print_profile == "kodak_endura"


def test_four_lut_bundle_carries_all_intermediate_wires():
    meta = BundleMeta(
        topology="4lut",
        wires=WiresMeta(
            log_e_film=LogEWire(min=-3.1, max=2.4),
            cmy_film=DensityWire(d_max=(3.8, 4.1, 3.6)),
            log_e_print=LogEWire(min=-2.7, max=2.1),
            cmy_print=DensityWire(d_max=(2.4, 2.6, 2.5)),
        ),
    )
    assert meta.wires.log_e_film.max == 2.4
    assert meta.wires.cmy_print.d_max == (2.4, 2.6, 2.5)
