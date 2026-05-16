from __future__ import annotations

import binascii
import datetime
import importlib.resources as pkg_resources
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import colour
import exiv2
from exiv2.types import DataBuf
import numpy as np
import OpenImageIO as oiio
import PIL.Image
import PIL.ImageCms
import scipy.interpolate

from spektrafilm.color_management import ColorEncoding

################################################################################
# Image metadata
################################################################################


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    exif: exiv2.ExifData
    iptc: exiv2.IptcData
    xmp: exiv2.XmpData


@dataclass(frozen=True, slots=True)
class ImagePayload:
    pixels: np.ndarray
    color_encoding: ColorEncoding | None
    source_metadata: ImageMetadata | None = None


def read_image_metadata(filename: str) -> ImageMetadata | None:
    """Read content metadata (EXIF, IPTC, XMP) from an image file.

    Uses the Exiv2 library to read metadata from any format including RAW files.

    Parameters
    ----------
    filename : str
        Path to the image file.

    Returns
    -------
    ImageMetadata or None
        The image metadata, or ``None`` if the file cannot be opened.
    """
    try:
        image = exiv2.ImageFactory.open(filename)
        image.readMetadata()
    except Exception:
        return None

    return ImageMetadata(
        exif=image.exifData(),
        iptc=image.iptcData(),
        xmp=image.xmpData(),
    )


def write_image_metadata(
    filename: str,
    source_metadata: ImageMetadata | None = None,
    *,
    saving_color_space: str | None = None,
    saving_cctf_encoding: bool = True,
) -> None:
    """Write metadata to an image file after pixel data has been saved.

    Copies any source EXIF, IPTC and XMP tags, then sets overridden tags
    (Orientation, DateTime, Software, pixel dimensions). When
    ``saving_color_space`` is given, also tags the file with the EXIF
    ColorSpace / Interoperability fields that match the saved color space and
    records the human-readable profile name in ``Xmp.photoshop.ICCProfile``.

    Parameters
    ----------
    filename : str
        Path to the output image file (must already exist on disk).
    source_metadata : ImageMetadata, optional
        Metadata returned by ``read_image_metadata`` to copy from the original
        file. Pass ``None`` when there is no source file.
    saving_color_space : str, optional
        Human-readable name of the color space the pixels were encoded in
        (e.g. ``"sRGB"``, ``"Adobe RGB (1998)"``, ``"Display P3"``).
    saving_cctf_encoding : bool, default True
        Whether the saved pixels carry the color space's encoding transfer
        function. ``False`` (linear data) is appended to the recorded profile
        name so downstream tools can flag it.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "exr":
        return

    image_input = oiio.ImageInput.open(filename)
    if image_input is None:
        raise RuntimeError(f"Could not open image file with OpenImageIO: {filename}")

    try:
        spec = image_input.spec()
        icc_before = _icc_profile_bytes_from_spec(spec)
    finally:
        image_input.close()
    destination = exiv2.ImageFactory.open(filename)
    destination.readMetadata()

    if source_metadata is not None:
        destination.setExifData(source_metadata.exif)
        destination.setIptcData(source_metadata.iptc)
        destination.setXmpData(source_metadata.xmp)

    destination_exif = destination.exifData()

    destination_exif["Exif.Image.Orientation"] = 1
    destination_exif["Exif.Image.DateTime"] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    destination_exif["Exif.Image.Software"] = "spektrafilm"
    destination_exif["Exif.Photo.PixelXDimension"] = spec.width
    destination_exif["Exif.Photo.PixelYDimension"] = spec.height

    if saving_color_space is not None:
        _set_color_space_tags(
            destination_exif,
            destination.xmpData(),
            saving_color_space,
            saving_cctf_encoding,
        )

    if icc_before is not None:
        destination.setIccProfile(DataBuf(icc_before))

    destination.writeMetadata()

    if icc_before is not None:
        icc_after = _icc_profile_bytes_from_file(filename)
        if icc_after != icc_before:
            raise RuntimeError("metadata copy did not preserve the output ICC profile")


# EXIF Photo.ColorSpace values per EXIF 2.32 spec.
_EXIF_COLORSPACE_SRGB = 1
_EXIF_COLORSPACE_UNCALIBRATED = 65535


# Maps (color_space_name, cctf_encoded) -> path inside spektrafilm/data/icc/.
# Filenames preserve the upstream names so they stay traceable to the source
# repos (see data/icc/README.md). Missing entries / files are silently skipped.
_ICC_FILENAMES: dict[tuple[str, bool], str] = {
    # Elle Stone — established RGB working spaces, V2 for broad compatibility.
    ("sRGB", True): "ellelstone/sRGB-elle-V2-srgbtrc.icc",
    ("sRGB", False): "ellelstone/sRGB-elle-V2-g10.icc",
    ("Adobe RGB (1998)", True): "ellelstone/ClayRGB-elle-V2-g22.icc",
    ("Adobe RGB (1998)", False): "ellelstone/ClayRGB-elle-V2-g10.icc",
    ("ProPhoto RGB", True): "ellelstone/LargeRGB-elle-V2-g18.icc",
    ("ProPhoto RGB", False): "ellelstone/LargeRGB-elle-V2-g10.icc",
    ("ITU-R BT.2020", True): "ellelstone/Rec2020-elle-V2-rec709.icc",
    ("ITU-R BT.2020", False): "ellelstone/Rec2020-elle-V2-g10.icc",
    # ACES2065-1 is scene-linear; both flags map to the linear ACES (AP0) file.
    ("ACES2065-1", True): "ellelstone/ACES-elle-V2-g10.icc",
    ("ACES2065-1", False): "ellelstone/ACES-elle-V2-g10.icc",
    # Saucecontrol — P3 variants Elle Stone's set doesn't cover.
    # No compact linear P3 ICC ships upstream; linear variants fall through.
    ("Display P3", True): "saucecontrol/DisplayP3-v2-micro.icc",
    ("DCI-P3", True): "saucecontrol/DCI-P3-v4.icc",
}


def _load_icc_profile(color_space: str, cctf_encoding: bool) -> bytes | None:
    relative_path = _ICC_FILENAMES.get((color_space, cctf_encoding))
    if relative_path is None:
        return None
    resource = pkg_resources.files("spektrafilm.data.icc").joinpath(*relative_path.split("/"))
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError):
        return None


def _set_color_space_tags(
    exif_data: "exiv2.ExifData",
    xmp_data: "exiv2.XmpData",
    saving_color_space: str,
    saving_cctf_encoding: bool,
) -> None:
    if saving_color_space == "sRGB" and saving_cctf_encoding:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_SRGB
        exif_data["Exif.Iop.InteroperabilityIndex"] = "R98"
    elif saving_color_space == "Adobe RGB (1998)" and saving_cctf_encoding:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_UNCALIBRATED
        exif_data["Exif.Iop.InteroperabilityIndex"] = "R03"
    else:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_UNCALIBRATED

    profile_name = saving_color_space if saving_cctf_encoding else f"{saving_color_space} (linear)"
    xmp_data["Xmp.photoshop.ICCProfile"] = profile_name


################################################################################
# 16-bit PNG I/O
################################################################################

_KNOWN_METADATA_COLOR_SPACES = (
    "sRGB",
    "Display P3",
    "DCI-P3",
    "Adobe RGB (1998)",
    "ITU-R BT.2020",
    "ProPhoto RGB",
    "ACES2065-1",
)


def resolve_icc_profile_bytes(color_space: str, cctf_encoding: bool = True) -> bytes | None:
    """Return ICC profile bytes for a named RGB colour-space, or ``None``."""

    profile_bytes = _load_icc_profile(color_space, cctf_encoding)
    if profile_bytes is not None:
        return profile_bytes

    if color_space == "sRGB":
        from PIL.ImageCms import ImageCmsProfile
        return ImageCmsProfile(PIL.ImageCms.createProfile("sRGB")).tobytes()

    return None


def colorspace_chromaticities(color_space: str) -> tuple[float, ...] | None:
    """Return (Rx, Ry, Gx, Gy, Bx, By, Wx, Wy) for an EXR *chromaticities* attribute."""
    try:
        cs = colour.RGB_COLOURSPACES[color_space]
    except KeyError:
        return None
    return (
        float(cs.primaries[0][0]), float(cs.primaries[0][1]),
        float(cs.primaries[1][0]), float(cs.primaries[1][1]),
        float(cs.primaries[2][0]), float(cs.primaries[2][1]),
        float(cs.whitepoint[0]), float(cs.whitepoint[1]),
    )


_OIIO_COLORSPACE_ALIASES = {
    "aces2065-1": "ACES2065-1",
    "lin_ap0_scene": "ACES2065-1",
    "display p3": "Display P3",
    "srgb": "sRGB",
    "srgb_rec709_scene": "sRGB",
    "adobe rgb (1998)": "Adobe RGB (1998)",
    "prophoto rgb": "ProPhoto RGB",
    "itu-r bt.2020": "ITU-R BT.2020",
    "bt2020": "ITU-R BT.2020",
}


_ICC_PROFILE_DESCRIPTION_ALIASES = {
    "display p3": "Display P3",
    "srgb": "sRGB",
    "srgb iec61966-2.1": "sRGB",
    "adobe rgb (1998)": "Adobe RGB (1998)",
    "prophoto rgb": "ProPhoto RGB",
    "bt.2020": "ITU-R BT.2020",
    "rec.2020": "ITU-R BT.2020",
}

_CHROMATICITY_PRIMARY_ERROR_THRESHOLD = 2e-4
_CHROMATICITY_WHITEPOINT_ERROR_THRESHOLD = 5e-4
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_image_color_encoding(filename: str) -> ColorEncoding | None:
    """Read known ICC/EXR colour metadata and map it to a runtime encoding."""

    in_img = oiio.ImageInput.open(filename)
    if not in_img:
        oiio.geterror()
        raise IOError("Could not open image file: " + filename)

    try:
        spec = in_img.spec()
    finally:
        in_img.close()

    ext = Path(filename).suffix.lower()
    oiio_color_space = _known_color_space_from_oiio(spec)
    color_space = (
        _known_color_space_from_icc_profile(spec)
        or oiio_color_space
        or _known_color_space_from_chromaticities(spec)
    )
    if color_space is None:
        return None

    transfer = "linear" if ext == ".exr" or _oiio_colorspace_is_linear(spec) else "cctf"
    return ColorEncoding(
        color_space=color_space,
        transfer=transfer,
        role="scene" if transfer == "linear" else "display",
        clip_negatives=True,
        clip_highlights=ext != ".exr",
    )


def load_image_payload(filename: str) -> ImagePayload:
    """Load pixels plus best-effort colour and content metadata."""

    pixels = load_image_oiio(filename)
    try:
        color_encoding = read_image_color_encoding(filename)
    except (OSError, RuntimeError, TypeError, ValueError):
        color_encoding = None
    return ImagePayload(
        pixels=pixels,
        color_encoding=color_encoding,
        source_metadata=read_image_metadata(filename),
    )


def _known_color_space_from_oiio(spec) -> str | None:
    color_space = spec.get_string_attribute("oiio:ColorSpace") or spec.get_string_attribute("colorInteropID")
    if not color_space:
        return None
    return _OIIO_COLORSPACE_ALIASES.get(" ".join(color_space.lower().split()))


def _oiio_colorspace_is_linear(spec) -> bool:
    color_space = spec.get_string_attribute("oiio:ColorSpace") or spec.get_string_attribute("colorInteropID")
    if not color_space:
        return False
    normalized = color_space.strip().lower()
    return normalized.startswith("lin_") or normalized in {"aces2065-1"}


def _known_color_space_from_icc_profile(spec) -> str | None:
    icc_bytes = _icc_profile_bytes_from_spec(spec)
    if icc_bytes:
        for color_space in _KNOWN_METADATA_COLOR_SPACES:
            known_profile = resolve_icc_profile_bytes(color_space)
            if known_profile is not None and icc_bytes == known_profile:
                return color_space

    description = spec.get_string_attribute("ICCProfile:profile_description")
    if description:
        return _ICC_PROFILE_DESCRIPTION_ALIASES.get(" ".join(description.lower().split()))
    return None


def _icc_profile_bytes_from_spec(spec) -> bytes | None:
    try:
        value = spec.getattribute("ICCProfile")
    except (AttributeError, TypeError, ValueError):
        return None
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.uint8).tobytes()
    try:
        return bytes(value)
    except (TypeError, ValueError):
        return None


def _icc_profile_bytes_from_file(filename: str) -> bytes | None:
    image_input = oiio.ImageInput.open(filename)
    if image_input is None:
        return None
    try:
        return _icc_profile_bytes_from_spec(image_input.spec())
    finally:
        image_input.close()


def _known_color_space_from_chromaticities(spec) -> str | None:
    try:
        chromaticities = spec.getattribute("chromaticities")
    except (AttributeError, TypeError, ValueError):
        return None
    if chromaticities is None:
        return None
    chromaticities_array = np.asarray(chromaticities, dtype=float)
    if chromaticities_array.shape != (8,):
        return None

    best_color_space: str | None = None
    best_primary_error = float("inf")
    best_whitepoint_error = float("inf")
    for color_space in _KNOWN_METADATA_COLOR_SPACES:
        reference = colorspace_chromaticities(color_space)
        if reference is None:
            continue
        reference_array = np.asarray(reference, dtype=float)
        primary_error = float(np.max(np.abs(chromaticities_array[:6] - reference_array[:6])))
        whitepoint_error = float(np.max(np.abs(chromaticities_array[6:8] - reference_array[6:8])))
        if (primary_error, whitepoint_error) < (best_primary_error, best_whitepoint_error):
            best_primary_error = primary_error
            best_whitepoint_error = whitepoint_error
            best_color_space = color_space
    if (
        best_color_space is not None
        and best_primary_error <= _CHROMATICITY_PRIMARY_ERROR_THRESHOLD
        and best_whitepoint_error <= _CHROMATICITY_WHITEPOINT_ERROR_THRESHOLD
    ):
        return best_color_space
    return None


def load_image_oiio(filename):
    # Open the image file
    in_img = oiio.ImageInput.open(filename)
    if not in_img:
        oiio.geterror()
        raise IOError("Could not open image file: " + filename)

    try:
        spec = in_img.spec()

        # Determine the native pixel format:
        # Use "uint16" for PNG and "half" for EXR if applicable.
        if spec.format == oiio.TypeDesc("uint8"): # for compatibility
            read_type = oiio.TypeDesc("uint8")
        elif spec.format == oiio.TypeDesc("uint16"):
            read_type = oiio.TypeDesc("uint16")
        elif spec.format == oiio.TypeDesc("half"):
            read_type = oiio.TypeDesc("half")
        elif spec.format == oiio.TypeDesc("float"):
            read_type = oiio.TypeDesc("float")
        else:
            # Fallback: use "uint16" by default. You might choose "float" if desired.
            read_type = oiio.TypeDesc("uint16")

        # Read the image data using the chosen type
        pixels = in_img.read_image(read_type)
        if pixels is None:
            raise Exception("Failed to read image data from " + filename)

        # Convert the raw data to a NumPy array and reshape it
        np_pixels = np.array(pixels)
        np_pixels = np_pixels.reshape(spec.height, spec.width, spec.nchannels)

        if spec.format == oiio.TypeDesc("uint16"):
            np_pixels = np.double(np_pixels) / (2**16 - 1)
        if spec.format == oiio.TypeDesc("uint8"):
            np_pixels = np.double(np_pixels) / (2**8 - 1)

        return np_pixels
    finally:
        in_img.close()

def save_image_oiio(
    filename,
    image_data,
    bit_depth: int | None = None,
    *,
    color_space: str | None = None,
    cctf_encoding: bool = True,
    encoding: ColorEncoding | None = None,
    white_luminance: float | None = None,
):
    """Save a 3-channel image to disk via OpenImageIO.

    Pixel format per extension:

    - ``.jpg`` / ``.jpeg``: clipped to [0, 1] and written as uint8.
      ``bit_depth`` is ignored.
    - ``.png``: clipped to [0, 1] and written as uint16 by default.
      Pass ``bit_depth=8`` for uint8.
    - ``.tif`` / ``.tiff``: ``bit_depth`` selects the encoding —
      8 → uint8 (clipped, scaled to [0, 255]),
      16 → uint16 (clipped, scaled to [0, 65535]),
      32 → float32 (raw, no clip/scale). Written with ZIP/deflate
      compression.
    - ``.exr``: ``bit_depth`` selects the encoding —
      16 → half (float16), 32 → float32. Always raw, no clip/scale.

    With the default ``bit_depth=None`` this gives float32 EXR and uint16
    PNG/TIFF. Pass ``bit_depth=16`` for half EXR, ``bit_depth=32`` for
    float32 TIFF, or ``bit_depth=8`` for uint8 TIFF.

    When ``color_space`` is provided and a matching ICC profile exists in
    ``spektrafilm/data/icc/`` (see the table in ``_ICC_FILENAMES``), the
    profile bytes are embedded into the file's native ICC slot:
    JPEG APP2 marker, PNG iCCP chunk, or TIFF ICCProfile tag. EXR carries
    its own color metadata so ICC embedding is skipped there. Missing
    profiles fall back to no embedding — the EXIF/XMP color-space tagging
    written by ``write_image_metadata`` still labels the file.

    Parameters
    ----------
    filename : str
        Output path; the extension selects the file format.
    image_data : np.ndarray
        Image data with shape ``(height, width, 3)``. Floating-point input
        is assumed to be in [0, 1] for integer-encoded formats.
    bit_depth : int, optional
        Precision selector for TIFF and EXR (see above). Ignored for JPEG.
        Defaults to 16 for PNG/TIFF and 32 for EXR, preserving float HDR by
        default while keeping compact integer output for display formats.
    color_space : str, optional
        Name of the color space the pixels are encoded in (e.g. ``"sRGB"``,
        ``"Display P3"``). Used to look up the ICC profile to embed.
    cctf_encoding : bool, default True
        Whether the pixels carry the color space's encoding transfer
        function. Affects which ICC variant is embedded (encoded vs linear).
    encoding : ColorEncoding, optional
        Full colour encoding contract for the file output. Takes precedence
        over ``color_space`` and ``cctf_encoding``.
    white_luminance : float, optional
        Optional EXR whiteLuminance metadata, in cd/m².
    """
    # Determine file type based on extension before deriving the default encoding.
    ext = filename.split('.')[-1].lower()
    if bit_depth is None:
        bit_depth = 32 if ext == "exr" else 16

    if encoding is None and color_space is not None:
        if ext == "exr":
            encoding = ColorEncoding(
                color_space=color_space,
                transfer="linear",
                role="scene",
                clip_highlights=False,
            )
        else:
            encoding = ColorEncoding(
                color_space=color_space,
                transfer="cctf" if cctf_encoding else "linear",
                role="display" if cctf_encoding else "scene",
            )

    if encoding is not None:
        color_space = encoding.color_space
        cctf_encoding = encoding.is_cctf_encoded

    # Extract image dimensions and number of channels
    height, width, nchannels = image_data.shape

    if ext in {"png", "jpg", "jpeg"}:
        if encoding is not None and encoding.is_linear:
            raise ValueError(
                f"{ext.upper()} export requires CCTF-encoded data; linear {encoding.color_space} should be saved as EXR."
            )

        save_kwargs: dict[str, object] = {}
        if color_space is not None:
            icc_bytes = _load_icc_profile(color_space, cctf_encoding)
            if icc_bytes is not None:
                save_kwargs["icc_profile"] = icc_bytes

        if ext == "png" and bit_depth >= 16:
            img_uint16 = np.rint(np.clip(image_data, 0, 1) * 65535.0).astype(np.uint16)
            _write_png_rgb16(filename, img_uint16, icc_profile=save_kwargs.get("icc_profile"))
        else:
            img_uint8 = np.clip(image_data, 0, 1) * 255.0
            img_uint8 = img_uint8.astype(np.uint8)
            pil_image = PIL.Image.fromarray(img_uint8, mode="RGB")
            if ext == "png":
                pil_image.save(filename, **save_kwargs)
            else:
                pil_image.save(filename, quality=95, **save_kwargs)
        return

    if ext == "exr" and encoding is not None and encoding.is_cctf_encoded:
        raise ValueError(f"EXR export requires linear data; CCTF-encoded {encoding.color_space} should be saved as PNG/JPEG.")

    # Create an ImageSpec with the proper data type
    if ext == "exr" and bit_depth == 16:
        # Convert the image data to 16-bit half precision.
        # Note: numpy's float16 is used here; OpenImageIO accepts "half" for 16-bit floats.
        img_half = image_data.astype(np.float16)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("half"))
        data_to_write = img_half
    elif ext=='exr' and bit_depth==32:
        # Convert the image data to 32-bit float precision.
        # Note: numpy's float32 is used here; OpenImageIO accepts "float" for 32-bit floats.
        img_float = image_data.astype(np.float32)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("float"))
        data_to_write = img_float
    elif ext in {"tif", "tiff"}:
        if bit_depth == 8:
            img = (np.clip(image_data, 0, 1) * 255.0).astype(np.uint8)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint8"))
        elif bit_depth == 16:
            img = (np.clip(image_data, 0, 1) * 65535.0).astype(np.uint16)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint16"))
        elif bit_depth == 32:
            img = image_data.astype(np.float32)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("float"))
        else:
            raise ValueError(f"Unsupported bit_depth for TIFF: {bit_depth}")
        # ZIP/deflate is lossless and works for all bit depths (LZW is faster but
        # integer-only); a TIFF of a 4K float image is ~100 MB uncompressed.
        spec.attribute("Compression", "zip")
        data_to_write = img
    else:
        raise ValueError("Unsupported file extension: " + ext)

    if color_space is not None:
        chromaticities = colorspace_chromaticities(color_space)
        if chromaticities is not None:
            spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), chromaticities)
        spec.attribute("colorInteropID", color_space)
        spec.attribute("oiio:ColorSpace", color_space)
    if white_luminance is not None:
        spec.attribute("whiteLuminance", float(white_luminance))

    if color_space is not None and ext != "exr":
        icc_bytes = _load_icc_profile(color_space, cctf_encoding)
        if icc_bytes is not None:
            icc_array = np.frombuffer(icc_bytes, dtype=np.uint8)
            spec.attribute(
                "ICCProfile",
                oiio.TypeDesc(f"uint8[{icc_array.size}]"),
                icc_array,
            )

    # Create an ImageOutput for writing the file
    out = oiio.ImageOutput.create(filename)
    if not out:
        raise IOError("Could not create output image: " + filename)

    try:
        out.open(filename, spec)
        # Write the image data; write_image accepts the NumPy array directly.
        out.write_image(data_to_write)
    finally:
        out.close()


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_png_rgb16(filename: str, image_data: np.ndarray, *, icc_profile: bytes | None) -> None:
    image = np.asarray(image_data)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("16-bit PNG export requires an RGB image with shape (height, width, 3).")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot save an empty PNG image.")

    image_be = image.astype(">u2", copy=False)
    raw_scanlines = b"".join(b"\x00" + image_be[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)

    chunks = [_png_chunk(b"IHDR", ihdr)]
    if icc_profile is not None:
        chunks.append(_png_chunk(b"iCCP", b"ICC profile\x00\x00" + zlib.compress(icc_profile)))
    chunks.append(_png_chunk(b"IDAT", zlib.compress(raw_scanlines)))
    chunks.append(_png_chunk(b"IEND", b""))

    Path(filename).write_bytes(_PNG_SIGNATURE + b"".join(chunks))

################################################################################
# Neutral filter values
################################################################################

NEUTRAL_PRINT_FILTERS_FILENAME = 'neutral_print_filters.json'


def save_neutral_print_filters(neutral_print_filters):
    package = pkg_resources.files('spektrafilm.data.filters')
    resource = package / NEUTRAL_PRINT_FILTERS_FILENAME
    with resource.open("w") as file:
        json.dump(neutral_print_filters, file, indent=4)


def read_neutral_print_filters():
    package = pkg_resources.files('spektrafilm.data.filters')
    resource = package / NEUTRAL_PRINT_FILTERS_FILENAME
    with resource.open("r") as file:
        return json.load(file)

################################################################################
# Profiles
################################################################################

def load_dichroic_filters(wavelengths, brand='thorlabs'):
    channels = ['c','m','y']
    filters = np.zeros((np.size(wavelengths), 3))
    for i, channel in enumerate(channels):
        package = pkg_resources.files('spektrafilm.data.filters.dichroics')
        filename = brand+'/filter_'+channel+'.csv'
        resource = package / filename
        with resource.open("r") as file:
            data = np.loadtxt(file, delimiter=',')
            unique_index = np.unique(data[:,0], return_index=True)[1]
            data = data[unique_index,:]
            # filters[:,i] = scipy.interpolate.CubicSpline(data[:,0], data[:,1]/100)(wavelengths)
            filters[:,i] = scipy.interpolate.Akima1DInterpolator(data[:,0], data[:,1]/100)(wavelengths)
    return filters

def load_filter(wavelengths, name='KG3', brand='schott', filter_type='heat_absorbing', percent_transmittance=False):
    transmittance = np.zeros_like(wavelengths)
    package = pkg_resources.files('spektrafilm.data.filters.'+filter_type)
    filename = brand+'/'+name+'.csv'
    resource = package / filename
    if percent_transmittance: scale = 100
    else: scale = 1
    with resource.open("r") as file:
        data = np.loadtxt(file, delimiter=',')
        unique_index = np.unique(data[:,0], return_index=True)[1]
        data = data[unique_index,:]
        # transmittance = scipy.interpolate.CubicSpline(data[:,0], data[:,1]/scale)(wavelengths)
        transmittance = scipy.interpolate.Akima1DInterpolator(data[:,0], data[:,1]/scale)(wavelengths)
    return transmittance
