import datetime
import os

import exiv2
import numpy as np

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
    assert exif["Exif.Image.Software"] == "Spektrafilm"
    assert exif["Exif.Image.DateTime"] == "2025:06:15 12:30:45"
    assert exif["Exif.Photo.PixelXDimension"] == "16"
    assert exif["Exif.Photo.PixelYDimension"] == "12"


def test_save_without_metadata_has_no_exif(tmp_path):
    destination_path = tmp_path / "plain.jpg"

    save_image_oiio(str(destination_path), np.random.rand(4, 4, 3))

    assert os.path.isfile(destination_path)

    result = read_image_metadata(str(destination_path))

    assert "Exif.Image.Software" not in _read_tags(result.exif)
