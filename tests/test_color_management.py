from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.color_management import ColorEncoding, input_encoding_from_io, output_encoding_from_io
from spektrafilm.runtime.params_schema import IOParams
from spektrafilm.runtime.stages.scanning import ScanningStage


def test_input_encoding_from_io_maps_decoding_flag_to_transfer() -> None:
    linear_io = IOParams(input_color_space="ProPhoto RGB", input_cctf_decoding=False)
    cctf_io = IOParams(input_color_space="Display P3", input_cctf_decoding=True)

    linear_encoding = input_encoding_from_io(linear_io)
    cctf_encoding = input_encoding_from_io(cctf_io)

    assert linear_encoding.color_space == "ProPhoto RGB"
    assert linear_encoding.transfer == "linear"
    assert linear_encoding.role == "scene"
    assert cctf_encoding.color_space == "Display P3"
    assert cctf_encoding.transfer == "cctf"
    assert cctf_encoding.role == "scene"


def test_output_encoding_from_io_maps_sdr_png_jpeg_contract() -> None:
    io = IOParams(
        output_color_space="Display P3",
        output_cctf_encoding=True,
        output_clip_min=True,
        output_clip_max=True,
    )

    encoding = output_encoding_from_io(io)

    assert encoding.color_space == "Display P3"
    assert encoding.transfer == "cctf"
    assert encoding.role == "display"
    assert encoding.clip_negatives is True
    assert encoding.clip_highlights is True


def test_output_encoding_from_io_maps_hdr_exr_contract() -> None:
    io = IOParams(
        output_color_space="ACES2065-1",
        output_cctf_encoding=False,
        output_clip_min=True,
        output_clip_max=False,
    )

    encoding = output_encoding_from_io(io)

    assert encoding.color_space == "ACES2065-1"
    assert encoding.transfer == "linear"
    assert encoding.role == "scene"
    assert encoding.clip_negatives is True
    assert encoding.clip_highlights is False


def test_color_encoding_rejects_unknown_color_space() -> None:
    with pytest.raises(ValueError, match="Unknown RGB colourspace"):
        ColorEncoding(color_space="srgb", transfer="cctf")


def test_scanning_stage_clip_contract_preserves_hdr_highlights_when_requested() -> None:
    stage = object.__new__(ScanningStage)
    rgb = np.array([[[-0.25, 0.5, 2.0]]], dtype=np.float32)
    encoding = ColorEncoding(
        color_space="ACES2065-1",
        transfer="linear",
        role="scene",
        clip_negatives=True,
        clip_highlights=False,
    )

    result = stage._apply_cctf_encoding_and_clip(rgb, encoding)

    np.testing.assert_allclose(result, np.array([[[0.0, 0.5, 2.0]]], dtype=np.float32))
