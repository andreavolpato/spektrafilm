from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import colour
import exiv2
import lensfunpy
import numpy as np
import rawpy
from scipy.ndimage import map_coordinates

_TUNGSTEN_TEMPERATURE = 2850.0
_DAYLIGHT_REFERENCE_TEMPERATURE = 6504.0
_ACES_COLOURSPACE = colour.RGB_COLOURSPACES["ACES2065-1"]


@dataclass(frozen=True, slots=True)
class ExifData:
    make: str
    model: str
    lens_make: str
    lens_model: str
    focal_length: float
    f_number: float


def _whitepoint_xyz_from_temperature(temperature: float) -> np.ndarray:
    """Convert a colour temperature to an XYZ whitepoint.

    Daylight whitepoints above 4000 K are better approximated by the CIE
    daylight locus than by a pure Planckian radiator. Warmer illuminants such as
    tungsten are modelled with the Kang 2002 Planckian approximation.
    """

    method = 'CIE Illuminant D Series' if temperature >= 4000.0 else 'Kang 2002'
    xy = colour.CCT_to_xy(np.float64(temperature), method=method)
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float64)


def _apply_white_balance_adaptation(
    rgb: np.ndarray,
    source_white_xyz: np.ndarray,
    target_white_xyz: np.ndarray,
) -> np.ndarray:
    """Apply a colour-science chromatic adaptation in linear ACES RGB."""

    source_white_xyz = np.asarray(source_white_xyz, dtype=np.float64)
    target_white_xyz = np.asarray(target_white_xyz, dtype=np.float64)
    source_white_xyz = source_white_xyz / source_white_xyz[1]
    target_white_xyz = target_white_xyz / target_white_xyz[1]

    xyz = colour.RGB_to_XYZ(
        rgb,
        colourspace=_ACES_COLOURSPACE,
        chromatic_adaptation_transform=None,
        apply_cctf_decoding=False,
    )
    xyz = colour.chromatic_adaptation(
        xyz,
        source_white_xyz,
        target_white_xyz,
        method='Von Kries',
    )
    return colour.XYZ_to_RGB(
        xyz,
        colourspace=_ACES_COLOURSPACE,
        chromatic_adaptation_transform=None,
        apply_cctf_encoding=False,
    ).astype(np.float32)


def _apply_tint_adjustment(rgb: np.ndarray, tint: float | None) -> np.ndarray:
    """Apply a simple green-magenta tint adjustment in linear ACES RGB."""

    if tint is None or np.isclose(tint, 1.0):
        return rgb

    tint_scale = np.array([1.0, float(tint), 1.0], dtype=np.float32)
    return (rgb * tint_scale).astype(np.float32)


def _postprocess_params(
    white_balance: str | tuple[float, float],
    temperature: float | None,
    tint: float | None,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray] | None, float | None]:
    """Build the ``rawpy.postprocess`` parameters for the requested settings.

    The output is always configured as linear 16-bit ACES RGB. White balance is
    handled in one of two ways:

    - ``'as_shot'`` uses LibRaw camera white balance directly during demosaic.
    - Other modes use LibRaw's daylight-balanced default output as the base and
      apply colour-science chromatic adaptation in linear ACES RGB.
    """

    params: dict[str, object] = {
        'output_color': getattr(rawpy, 'ColorSpace').ACES,
        'output_bps': 16,
        'no_auto_bright': True,
        'gamma': (1, 1),
    }
    postprocess_adaptation: tuple[np.ndarray, np.ndarray] | None = None
    tint_multiplier: float | None = None
    reference_white_xyz = _whitepoint_xyz_from_temperature(_DAYLIGHT_REFERENCE_TEMPERATURE)

    def set_colour_science_adjustment(target_temperature: float, target_tint: float | None) -> None:
        nonlocal postprocess_adaptation, tint_multiplier

        scene_white_xyz = _whitepoint_xyz_from_temperature(target_temperature)
        if not np.allclose(reference_white_xyz, scene_white_xyz):
            postprocess_adaptation = (scene_white_xyz, reference_white_xyz)
        tint_multiplier = target_tint

    if white_balance == 'as_shot':
        params['use_camera_wb'] = True
    elif white_balance == 'daylight':
        pass
    elif white_balance == 'tungsten':
        set_colour_science_adjustment(_TUNGSTEN_TEMPERATURE, 1.0)
    elif white_balance == 'custom':
        if temperature is None:
            raise ValueError('A custom raw white balance requires a temperature value.')
        set_colour_science_adjustment(temperature, tint)
    else:
        custom_temperature, custom_tint = white_balance
        set_colour_science_adjustment(custom_temperature, custom_tint)

    return params, postprocess_adaptation, tint_multiplier


def _read_exif_metadata(raw_path: str | PathLike[str]) -> ExifData:
    """Read basic EXIF data from a RAW file for lens correction.

    The function uses ``exiv2`` to extract standard EXIF tags for camera make
    and model, lens make and model, as well as focal length and aperture.

    Parameters
    ----------
    raw_path
        Path to the RAW image file.

    Returns:
    -------
    ExifData
        Frozen dataclass with camera and lens metadata. Fields that could not be
        read default to ``""`` for strings and ``0.0`` for numeric values.

    """
    try:
        image = exiv2.ImageFactory.open(str(raw_path))
        image.readMetadata()
        exif = image.exifData()
    except Exception:
        return ExifData(
            make="",
            model="",
            lens_make="",
            lens_model="",
            focal_length=0.0,
            f_number=0.0,
        )

    def _str(key: str) -> str:
        if key not in exif:
            return ""

        return str(exif[key].value()).strip()

    def _float(key: str) -> float:
        if key not in exif:
            return 0.0
        return exif[key].toFloat()

    return ExifData(
        make=_str("Exif.Image.Make"),
        model=_str("Exif.Image.Model"),
        lens_make=_str("Exif.Photo.LensMake"),
        lens_model=_str("Exif.Photo.LensModel"),
        focal_length=_float("Exif.Photo.FocalLength"),
        f_number=_float("Exif.Photo.FNumber"),
    )


def _normalize_lens_text(value: object) -> str:
    """Normalize lens metadata strings for case-insensitive comparisons."""

    return ' '.join(str(value).strip().lower().split())


def _compact_lens_text(value: object) -> str:
    """Collapse lens metadata to alphanumeric characters for loose matching."""

    return ''.join(character.lower() for character in str(value) if character.isalnum())


def _lens_model_score(lens_model: object, exif_lens_model: str) -> tuple[int, int, int]:
    """Score how closely a database lens model matches the EXIF lens model."""

    normalized_model = _normalize_lens_text(exif_lens_model)
    compact_model = _compact_lens_text(exif_lens_model)
    normalized_lens_model = _normalize_lens_text(lens_model)
    compact_lens_model = _compact_lens_text(lens_model)
    return (
        int(normalized_lens_model == normalized_model),
        int(bool(compact_model) and compact_model in compact_lens_model),
        len(set(normalized_model.split()) & set(normalized_lens_model.split())),
    )


def _lens_matches_focal(lens: object, focal_length: float) -> bool:
    """Return whether the focal length falls within the lens calibration range."""

    min_focal = getattr(lens, "min_focal", None)
    max_focal = getattr(lens, "max_focal", None)
    if min_focal is None or max_focal is None or focal_length <= 0:
        return False
    return float(min_focal) <= focal_length <= float(max_focal)


def _lens_matches_aperture(lens: object, f_number: float) -> bool:
    """Return whether the aperture falls within the lens calibration range."""

    min_aperture = getattr(lens, "min_aperture", None)
    max_aperture = getattr(lens, "max_aperture", None)
    if min_aperture is None or f_number <= 0:
        return False
    upper_bound = float(max_aperture) if max_aperture is not None else float('inf')
    return float(min_aperture) <= f_number <= upper_bound


def _find_lens_candidates(db: object, camera: object, exif_metadata: ExifData) -> list[object]:
    """Find lensfun candidates with direct queries before a broad fallback search."""

    queries = [
        (exif_metadata.lens_make or None, exif_metadata.lens_model, False),
        (exif_metadata.lens_make or None, exif_metadata.lens_model, True),
        (None, exif_metadata.lens_model, True),
    ]

    def dedupe(lenses: list[object]) -> list[object]:
        candidates: list[object] = []
        seen: set[tuple[object, ...]] = set()
        for lens in lenses:
            identity = (
                _normalize_lens_text(getattr(lens, 'maker', '')),
                _normalize_lens_text(getattr(lens, 'model', '')),
                getattr(lens, 'min_focal', None),
                getattr(lens, 'max_focal', None),
                getattr(lens, 'min_aperture', None),
                getattr(lens, 'max_aperture', None),
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(lens)
        return candidates

    for lens_make, lens_model, loose_search in queries:
        direct_candidates = dedupe(db.find_lenses(camera, lens_make, lens_model, loose_search=loose_search))
        if direct_candidates:
            return direct_candidates

    broader_candidates = dedupe(db.find_lenses(camera, None, None, loose_search=True))
    return [
        lens for lens in broader_candidates
        if _lens_model_score(getattr(lens, 'model', ''), exif_metadata.lens_model) > (0, 0, 0)
    ]


def _select_lens_candidate(lenses: list[object], exif_metadata: ExifData) -> object:
    """Pick the best lens candidate using model, maker, focal and aperture cues."""

    focal_matches = [lens for lens in lenses if _lens_matches_focal(lens, exif_metadata.focal_length)]
    candidates = focal_matches if focal_matches else lenses
    normalized_maker = _normalize_lens_text(exif_metadata.lens_make)

    def lens_score(lens: object) -> tuple[tuple[int, int, int], int, int, int, float]:
        return (
            _lens_model_score(getattr(lens, 'model', ''), exif_metadata.lens_model),
            int(bool(normalized_maker) and _normalize_lens_text(getattr(lens, 'maker', '')) == normalized_maker),
            int(_lens_matches_focal(lens, exif_metadata.focal_length)),
            int(_lens_matches_aperture(lens, exif_metadata.f_number)),
            float(getattr(lens, 'score', 0.0) or 0.0),
        )

    candidates.sort(key=lens_score, reverse=True)
    return candidates[0]


def _apply_lens_correction(
    rgb: np.ndarray,
    exif_metadata: ExifData,
) -> tuple[np.ndarray, str]:
    """Apply ``lensfun`` lens corrections to the image.

    Camera and lens are looked up in the ``lensfun`` database using the
    supplied EXIF metadata. If either the camera or lens is not found, no
    correction is applied and the original image is returned.

    The following corrections are applied in sequence, if supported by the
    identified lens: vignetting, chromatic aberration, distortion,
    geometry and scale.

    Parameters
    ----------
    rgb
        Linear ``float32`` RGB image array with shape ``(H, W, 3)``.
    exif_metadata
        Camera and lens metadata read from the RAW file.

    Returns:
    -------
    numpy.ndarray
        Corrected (or original) ``float32`` RGB image.
    str
        Human readable summary of the camera, lens, focal length and aperture
        used for the correction. Empty when the camera or lens was not found in
        the ``lensfun`` database.

    """
    if not exif_metadata.lens_model.strip():
        return rgb, ""

    db = lensfunpy.Database()
    cameras = db.find_cameras(exif_metadata.make, exif_metadata.model, loose_search=True)

    if not cameras:
        return rgb, ""

    camera = cameras[0]

    lenses = _find_lens_candidates(db, camera, exif_metadata)

    if not lenses:
        return rgb, ""

    lens = _select_lens_candidate(lenses, exif_metadata)
    lens_label = getattr(lens, 'model', None) or exif_metadata.lens_model or str(lens)
    lens_info = f"{lens_label} @ {exif_metadata.focal_length}mm f/{exif_metadata.f_number}"

    height, width = rgb.shape[:2]

    mod = lensfunpy.Modifier(lens, camera.crop_factor, width, height)
    mod.initialize(
        exif_metadata.focal_length,
        exif_metadata.f_number,
        pixel_format=np.float32,
        flags=lensfunpy.ModifyFlags.ALL,
    )

    mod.apply_color_modification(rgb)
    undist_coords = mod.apply_subpixel_geometry_distortion()

    if undist_coords is not None:
        corrected = np.empty_like(rgb)

        for c in range(rgb.shape[2]):
            corrected[:, :, c] = map_coordinates(
                rgb[:, :, c],
                [undist_coords[:, :, c, 1], undist_coords[:, :, c, 0]],
                order=1,
                mode="nearest",
            )

        rgb = corrected

    return rgb, lens_info


def load_and_process_raw_file(
    raw_path: str | PathLike[str],
    white_balance='as_shot',
    temperature: float | None = None,
    tint: float | None = None,
    lens_correction: bool = False,
    output_colorspace: str = "ACES2065-1",
    output_cctf_encoding: bool = False,
    lens_info_out: dict[str, str] | None = None,
) -> np.ndarray:
    """Load a RAW file into linear RGB and optionally convert its colourspace.

    The RAW is demosaiced by ``rawpy`` into linear 16-bit ACES RGB with auto
    brightening disabled. ``'as_shot'`` white balance comes from the camera
    metadata; the other white-balance modes use LibRaw's daylight-balanced base
    output and colour-science chromatic adaptation in linear ACES RGB.

    Parameters
    ----------
    raw_path
        Path to the RAW image.
    white_balance
        White-balance mode. Supported string values are ``'as_shot'``,
        ``'daylight'``, ``'tungsten'``, and ``'custom'``. A
        ``(temperature, tint)`` tuple is also accepted.
    temperature
        Correlated colour temperature in kelvin for ``'custom'`` mode.
    tint
        Multiplicative adjustment applied to both green channels for
        temperature-derived white balance.
    lens_correction
        When ``True``, apply all lensfun lens corrections (vignetting,
        TCA, distortion, geometry and scale). Camera and lens are identified
        from the EXIF metadata.
    output_colorspace
        Output RGB colourspace name understood by ``colour.RGB_COLOURSPACES``.
    output_cctf_encoding
        Whether to apply the output colourspace transfer function when a
        colourspace conversion is requested.

    Returns
    -------
    numpy.ndarray
        RGB image as ``float32`` in the requested output colourspace.
    """

    with rawpy.imread(str(raw_path)) as raw:
        params, postprocess_adaptation, tint_multiplier = _postprocess_params(white_balance, temperature, tint)
        rgb = raw.postprocess(**params).astype(np.float32) / np.float32(65535.0)

    if lens_correction:
        exif_metadata = _read_exif_metadata(raw_path)
        rgb, lens_info = _apply_lens_correction(rgb, exif_metadata)

        if lens_info_out is not None and lens_info:
            lens_info_out["summary"] = lens_info

    if postprocess_adaptation is not None:
        rgb = _apply_white_balance_adaptation(rgb, *postprocess_adaptation)

    rgb = _apply_tint_adjustment(rgb, tint_multiplier)

    if output_colorspace != 'ACES2065-1':
        rgb = colour.RGB_to_RGB(
            rgb,
            input_colourspace=_ACES_COLOURSPACE,
            output_colourspace=colour.RGB_COLOURSPACES[output_colorspace],
            apply_cctf_decoding=False,
            apply_cctf_encoding=output_cctf_encoding,
        )

    return rgb


__all__ = ['load_and_process_raw_file']