from __future__ import annotations

import numpy as np
import OpenImageIO as oiio
import exiv2
import pytest

from spektrafilm.color_management import ColorEncoding
from spektrafilm.utils.io import (
    ImageMetadata,
    _known_color_space_from_chromaticities,
    colorspace_chromaticities,
    read_image_color_encoding,
    save_image_oiio,
    write_image_metadata,
)


def _image_spec(path):
    image_input = oiio.ImageInput.open(str(path))
    assert image_input is not None
    try:
        return image_input.spec()
    finally:
        image_input.close()


def _icc_bytes_from_spec(spec) -> bytes | None:
    value = spec.getattribute("ICCProfile")
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.uint8).tobytes()
    return bytes(value)


def test_png_embeds_srgb_icc_and_reads_cctf_encoding(tmp_path) -> None:
    path = tmp_path / "srgb.png"
    image = np.full((2, 3, 3), 0.5, dtype=np.float32)

    save_image_oiio(
        str(path),
        image,
        encoding=ColorEncoding(color_space="sRGB", transfer="cctf", role="display"),
    )

    spec = _image_spec(path)
    assert _icc_bytes_from_spec(spec)
    encoding = read_image_color_encoding(str(path))
    assert encoding is not None
    assert encoding.color_space == "sRGB"
    assert encoding.transfer == "cctf"


def test_png_default_export_uses_16_bit_rgb_and_embeds_icc(tmp_path) -> None:
    path = tmp_path / "srgb-16.png"
    image = np.array([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]]], dtype=np.float32)

    save_image_oiio(
        str(path),
        image,
        encoding=ColorEncoding(color_space="sRGB", transfer="cctf", role="display"),
    )

    spec = _image_spec(path)
    assert spec.format == oiio.TypeDesc("uint16")
    assert spec.nchannels == 3
    assert _icc_bytes_from_spec(spec)

    image_input = oiio.ImageInput.open(str(path))
    assert image_input is not None
    try:
        pixels = np.asarray(image_input.read_image(oiio.TypeDesc("uint16"))).reshape(image.shape)
    finally:
        image_input.close()
    np.testing.assert_array_equal(pixels[0, 0], np.array([0, 32768, 65535], dtype=np.uint16))


def test_jpeg_embeds_display_p3_icc_and_reads_cctf_encoding(tmp_path) -> None:
    path = tmp_path / "display-p3.jpg"
    image = np.full((2, 3, 3), 0.5, dtype=np.float32)

    save_image_oiio(
        str(path),
        image,
        encoding=ColorEncoding(color_space="Display P3", transfer="cctf", role="display"),
    )

    spec = _image_spec(path)
    assert _icc_bytes_from_spec(spec)
    encoding = read_image_color_encoding(str(path))
    assert encoding is not None
    assert encoding.color_space == "Display P3"
    assert encoding.transfer == "cctf"


def test_exr_writes_chromaticities_colorspace_and_preserves_hdr_values(tmp_path) -> None:
    path = tmp_path / "hdr.exr"
    image = np.array([[[0.25, 1.5, 4.0], [0.1, 0.2, 0.3]]], dtype=np.float32)

    save_image_oiio(
        str(path),
        image,
        encoding=ColorEncoding(
            color_space="Display P3",
            transfer="linear",
            role="scene",
            clip_highlights=False,
        ),
    )

    spec = _image_spec(path)
    np.testing.assert_allclose(spec.getattribute("chromaticities"), colorspace_chromaticities("Display P3"))
    assert spec.get_string_attribute("oiio:ColorSpace") == "Display P3"
    assert spec.get_string_attribute("colorInteropID") == "Display P3"

    image_input = oiio.ImageInput.open(str(path))
    assert image_input is not None
    try:
        pixels = np.asarray(image_input.read_image(oiio.TypeDesc("float")), dtype=np.float32)
    finally:
        image_input.close()
    pixels = pixels.reshape(image.shape)
    np.testing.assert_allclose(pixels, image, rtol=1e-6, atol=1e-6)
    assert float(np.max(pixels)) > 1.0

    encoding = read_image_color_encoding(str(path))
    assert encoding is not None
    assert encoding.color_space == "Display P3"
    assert encoding.transfer == "linear"
    assert encoding.clip_highlights is False


def test_linear_png_without_linear_icc_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="PNG export requires CCTF-encoded data"):
        save_image_oiio(
            str(tmp_path / "linear.png"),
            np.full((1, 1, 3), 0.5, dtype=np.float32),
            encoding=ColorEncoding(color_space="Display P3", transfer="linear", role="scene"),
        )


def test_cctf_exr_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="EXR export requires linear data"):
        save_image_oiio(
            str(tmp_path / "encoded.exr"),
            np.full((1, 1, 3), 0.5, dtype=np.float32),
            encoding=ColorEncoding(color_space="sRGB", transfer="cctf", role="display"),
        )


def test_metadata_copy_keeps_png_icc_profile(tmp_path) -> None:
    destination = tmp_path / "destination.png"
    image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    encoding = ColorEncoding(color_space="Display P3", transfer="cctf", role="display")

    save_image_oiio(str(destination), image, encoding=encoding)
    metadata = ImageMetadata(exif=exiv2.ExifData(), iptc=exiv2.IptcData(), xmp=exiv2.XmpData())

    write_image_metadata(str(destination), metadata)

    spec = _image_spec(destination)
    assert _icc_bytes_from_spec(spec)
    roundtrip_encoding = read_image_color_encoding(str(destination))
    assert roundtrip_encoding is not None
    assert roundtrip_encoding.color_space == "Display P3"


def test_metadata_copy_keeps_jpeg_icc_profile(tmp_path) -> None:
    destination = tmp_path / "destination.jpg"
    image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    encoding = ColorEncoding(color_space="Display P3", transfer="cctf", role="display")

    save_image_oiio(str(destination), image, encoding=encoding)
    metadata = ImageMetadata(exif=exiv2.ExifData(), iptc=exiv2.IptcData(), xmp=exiv2.XmpData())

    write_image_metadata(str(destination), metadata)

    spec = _image_spec(destination)
    assert _icc_bytes_from_spec(spec)
    roundtrip_encoding = read_image_color_encoding(str(destination))
    assert roundtrip_encoding is not None
    assert roundtrip_encoding.color_space == "Display P3"


def test_chromaticities_matching_rejects_standard_primaries_with_wrong_whitepoint() -> None:
    chromaticities = np.asarray(colorspace_chromaticities("Display P3"), dtype=float)
    chromaticities[6:8] += np.array([0.001, -0.001], dtype=float)

    class FakeSpec:
        def getattribute(self, name):
            assert name == "chromaticities"
            return chromaticities

    assert _known_color_space_from_chromaticities(FakeSpec()) is None
