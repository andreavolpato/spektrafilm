import datetime
import os

import exiv2
import numpy as np
import pytest

from spektrafilm.utils import io as io_module
from spektrafilm.utils.io import ImageMetadata, read_image_metadata, save_image_oiio, write_image_metadata


def _build_source_metadata():
    exif = exiv2.ExifData()

    exif["Exif.Image.Make"] = "Canon"
    exif["Exif.Image.Model"] = "Canon EOS 5D Mark IV"
    exif["Exif.Photo.FocalLength"] = "50/1"
    exif["Exif.Photo.LensModel"] = "Canon EF 50mm f/1.8 STM"

    iptc = exiv2.IptcData()
    iptc["Iptc.Application2.Keywords"] = "dummy"

    xmp = exiv2.XmpData()
    xmp["Xmp.dc.creator"] = "Dummy Creator"

    return ImageMetadata(exif=exif, iptc=iptc, xmp=xmp)


def _read_tags(metadata_field):
    return {datum.key(): str(datum.value()) for datum in metadata_field}


def test_read_metadata_returns_none_for_missing_file(tmp_path):
    assert read_image_metadata(str(tmp_path / "does_not_exist.jpg")) is None


def test_write_metadata_carries_source_tags_and_sets_overrides(tmp_path, monkeypatch):
    fixed_now = datetime.datetime(2025, 6, 15, 12, 30, 45)

    class _MonkeyDatetime(datetime.datetime):
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr(io_module.datetime, "datetime", _MonkeyDatetime)

    source_metadata = _build_source_metadata()

    destination_path = tmp_path / "dst.jpg"

    save_image_oiio(str(destination_path), np.random.rand(12, 16, 3))

    write_image_metadata(str(destination_path), source_metadata)

    result = read_image_metadata(str(destination_path))

    exif = _read_tags(result.exif)
    iptc = _read_tags(result.iptc)
    xmp = _read_tags(result.xmp)

    # Copied tags
    assert exif["Exif.Image.Make"] == "Canon"
    assert exif["Exif.Image.Model"] == "Canon EOS 5D Mark IV"
    assert exif["Exif.Photo.FocalLength"] == "50/1"
    assert exif["Exif.Photo.LensModel"] == "Canon EF 50mm f/1.8 STM"

    assert iptc["Iptc.Application2.Keywords"] == "dummy"

    assert xmp["Xmp.dc.creator"] == "Dummy Creator"

    # Overridden tags
    assert exif["Exif.Image.Orientation"] == "1"
    assert exif["Exif.Image.Software"] == "spektrafilm"
    assert exif["Exif.Image.DateTime"] == "2025:06:15 12:30:45"
    assert exif["Exif.Photo.PixelXDimension"] == "16"
    assert exif["Exif.Photo.PixelYDimension"] == "12"


def test_save_without_metadata_has_no_exif(tmp_path):
    destination_path = tmp_path / "plain.jpg"

    save_image_oiio(str(destination_path), np.random.rand(4, 4, 3))

    assert os.path.isfile(destination_path)

    result = read_image_metadata(str(destination_path))

    assert "Exif.Image.Software" not in _read_tags(result.exif)


@pytest.mark.parametrize(
    ("saving_color_space", "saving_cctf_encoding", "expected_colorspace", "expected_iop", "expected_profile_name"),
    [
        ("sRGB", True, "1", "R98", "sRGB"),
        ("Adobe RGB (1998)", True, "65535", "R03", "Adobe RGB (1998)"),
        ("Display P3", True, "65535", None, "Display P3"),
        ("ProPhoto RGB", False, "65535", None, "ProPhoto RGB (linear)"),
        ("sRGB", False, "65535", None, "sRGB (linear)"),
    ],
)
def test_write_metadata_records_saving_color_space(
    tmp_path,
    saving_color_space,
    saving_cctf_encoding,
    expected_colorspace,
    expected_iop,
    expected_profile_name,
):
    destination_path = tmp_path / "tagged.jpg"

    save_image_oiio(str(destination_path), np.random.rand(8, 8, 3))
    write_image_metadata(
        str(destination_path),
        saving_color_space=saving_color_space,
        saving_cctf_encoding=saving_cctf_encoding,
    )

    result = read_image_metadata(str(destination_path))
    exif = _read_tags(result.exif)
    xmp = _read_tags(result.xmp)

    assert exif["Exif.Photo.ColorSpace"] == expected_colorspace
    if expected_iop is None:
        assert "Exif.Iop.InteroperabilityIndex" not in exif
    else:
        assert exif["Exif.Iop.InteroperabilityIndex"] == expected_iop
    assert xmp["Xmp.photoshop.ICCProfile"] == expected_profile_name


@pytest.mark.parametrize("ext", ["jpg", "png"])
def test_save_image_oiio_embeds_icc_profile_when_available(tmp_path, monkeypatch, ext):
    import OpenImageIO as oiio
    from PIL import ImageCms

    # A real (small) ICC profile so libpng's validation accepts it.
    real_icc_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    monkeypatch.setattr(
        io_module,
        "_load_icc_profile",
        lambda color_space, cctf_encoding: real_icc_bytes,
    )

    destination_path = tmp_path / f"with_icc.{ext}"
    save_image_oiio(
        str(destination_path),
        np.random.rand(8, 8, 3),
        color_space="Display P3",
        cctf_encoding=True,
    )

    in_img = oiio.ImageInput.open(str(destination_path))
    try:
        embedded = in_img.spec().getattribute("ICCProfile")
    finally:
        in_img.close()

    assert embedded is not None
    assert bytes(embedded) == real_icc_bytes


def test_save_image_oiio_skips_icc_when_profile_missing(tmp_path, monkeypatch):
    import OpenImageIO as oiio

    monkeypatch.setattr(io_module, "_load_icc_profile", lambda color_space, cctf_encoding: None)

    destination_path = tmp_path / "no_icc.jpg"
    save_image_oiio(
        str(destination_path),
        np.random.rand(8, 8, 3),
        color_space="ProPhoto RGB",
        cctf_encoding=True,
    )

    in_img = oiio.ImageInput.open(str(destination_path))
    try:
        embedded = in_img.spec().getattribute("ICCProfile")
    finally:
        in_img.close()

    assert embedded is None


def test_write_metadata_without_source_still_writes_overrides(tmp_path):
    destination_path = tmp_path / "no_source.jpg"

    save_image_oiio(str(destination_path), np.random.rand(4, 4, 3))
    write_image_metadata(str(destination_path), saving_color_space="sRGB")

    result = read_image_metadata(str(destination_path))
    exif = _read_tags(result.exif)

    assert exif["Exif.Image.Software"] == "spektrafilm"
    assert exif["Exif.Photo.ColorSpace"] == "1"
