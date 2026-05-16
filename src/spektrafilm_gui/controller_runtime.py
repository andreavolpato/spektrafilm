from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qtpy import QtCore

from spektrafilm.color_management import ColorEncoding
from spektrafilm.utils.io import resolve_icc_profile_bytes


DISPLAY_PREVIEW_COLOR_SPACE = 'sRGB'
QObject = getattr(QtCore, 'QObject')
QRunnable = getattr(QtCore, 'QRunnable')
Signal = getattr(QtCore, 'Signal')


@dataclass(slots=True)
class SimulationRequest:
    mode_label: str
    image: np.ndarray
    params: object
    use_display_transform: bool
    output_encoding: ColorEncoding | None = None
    output_color_space: str | None = None


@dataclass(slots=True)
class SimulationResult:
    mode_label: str
    display_image: np.ndarray
    float_image: np.ndarray
    use_display_transform: bool
    status_message: str
    output_encoding: ColorEncoding | None = None
    output_color_space: str | None = None


class SimulationWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class SimulationWorker(QRunnable):
    def __init__(self, request: SimulationRequest, *, execute_request: Callable[[SimulationRequest], SimulationResult]):
        super().__init__()
        self._request = request
        self._execute_request = execute_request
        self.signals = SimulationWorkerSignals()

    def run(self) -> None:
        try:
            result = self._execute_request(self._request)
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self.signals.finished.emit(result)


def normalized_image_data(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image, 0.0, 1.0)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        if max_value == 0:
            return image.astype(np.float32)
        return image.astype(np.float32) / max_value
    return image.astype(np.float32)


def apply_white_padding(image_data: np.ndarray, padding_pixels: float) -> np.ndarray:
    padding = max(0, int(round(padding_pixels)))
    if padding == 0:
        return np.asarray(image_data)

    image = np.asarray(image_data)
    if image.ndim < 2:
        return image

    fill_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    pad_width = [(padding, padding), (padding, padding)]
    pad_width.extend((0, 0) for _ in range(image.ndim - 2))
    return np.pad(image, pad_width, mode='constant', constant_values=fill_value)


def padding_pixels_for_image(image_data: np.ndarray, padding_fraction: float) -> int:
    image = np.asarray(image_data)
    if image.ndim < 2:
        return 0

    padding_fraction = max(0.0, float(padding_fraction))
    long_edge = max(int(image.shape[0]), int(image.shape[1]))
    return int(np.floor(long_edge * padding_fraction))


def display_profile_name(display_profile: object, *, imagecms_module: Any) -> str:
    try:
        profile_name = imagecms_module.getProfileName(display_profile)
    except (AttributeError, OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        profile_name = None

    if isinstance(profile_name, str):
        cleaned_name = profile_name.replace('\x00', ' ').strip()
        if cleaned_name:
            return ' '.join(cleaned_name.split())

    profile_filename = getattr(display_profile, 'filename', None)
    if isinstance(profile_filename, str) and profile_filename.strip():
        return Path(profile_filename).stem

    return type(display_profile).__name__


def display_profile_details(*, imagecms_module: Any) -> tuple[object | None, str | None]:
    try:
        display_profile = imagecms_module.get_display_profile()
    except (OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        return None, None
    if display_profile is None:
        return None, None
    return display_profile, display_profile_name(display_profile, imagecms_module=imagecms_module)


def display_profile_available(*, imagecms_module: Any) -> bool:
    try:
        return imagecms_module.get_display_profile() is not None
    except (OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        return False


def display_transform_status_message(enabled: bool, *, imagecms_module: Any) -> str:
    if not enabled:
        return 'Display transform: disabled'
    display_profile, profile_name = display_profile_details(imagecms_module=imagecms_module)
    if display_profile is None:
        return 'Display transform: no display profile, using raw preview'
    return f'Display transform: display profile found ({profile_name})'


def prepare_input_color_preview_image(
    image_data: np.ndarray,
    *,
    input_color_space: str,
    apply_cctf_decoding: bool,
    colour_module: Any,
) -> np.ndarray:
    normalized_image = normalized_image_data(np.asarray(image_data)[..., :3])
    try:
        srgb_preview = colour_module.RGB_to_RGB(
            normalized_image,
            input_color_space,
            DISPLAY_PREVIEW_COLOR_SPACE,
            apply_cctf_decoding=apply_cctf_decoding,
            apply_cctf_encoding=True,
        )
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
        return np.asarray(np.clip(normalized_image, 0.0, 1.0), dtype=np.float32)
    return np.asarray(np.clip(srgb_preview, 0.0, 1.0), dtype=np.float32)


def apply_display_transform(
    image_data: np.ndarray,
    *,
    output_encoding: ColorEncoding | None = None,
    output_color_space: str | None = None,
    colour_module: Any,
    imagecms_module: Any,
    pil_image_module: Any,
) -> tuple[np.ndarray, str]:
    output_encoding = _resolve_output_encoding(output_encoding, output_color_space)
    display_profile, profile_name = display_profile_details(imagecms_module=imagecms_module)
    if display_profile is None:
        fallback_pixels = _encode_pixels_for_output_profile(
            image_data,
            output_encoding=output_encoding,
            colour_module=colour_module,
        )
        return np.uint8(np.clip(fallback_pixels, 0.0, 1.0) * 255), 'Display transform: no display profile, using raw preview'

    source_profile = _imagecms_profile_for_color_space(output_encoding.color_space, imagecms_module=imagecms_module)
    fallback_status = None
    if source_profile is None:
        source_pixels = colour_module.RGB_to_RGB(
            image_data,
            output_encoding.color_space,
            DISPLAY_PREVIEW_COLOR_SPACE,
            apply_cctf_decoding=output_encoding.is_cctf_encoded,
            apply_cctf_encoding=True,
        )
        source_profile = imagecms_module.createProfile(DISPLAY_PREVIEW_COLOR_SPACE)
        if output_encoding.is_linear:
            fallback_status = (
                f'Display transform: active ({profile_name}); {output_encoding.color_space} has no ICC profile, '
                'using colorimetric sRGB preview without a scene-linear view transform'
            )
        else:
            fallback_status = (
                f'Display transform: active ({profile_name}); {output_encoding.color_space} has no ICC profile, '
                'using sRGB preview fallback'
            )
    else:
        source_pixels = _encode_pixels_for_output_profile(
            image_data,
            output_encoding=output_encoding,
            colour_module=colour_module,
        )

    source_uint8 = np.uint8(np.clip(source_pixels, 0.0, 1.0) * 255)
    source_image = pil_image_module.fromarray(source_uint8, mode='RGB')
    transformed_image = imagecms_module.profileToProfile(source_image, source_profile, display_profile, outputMode='RGB')
    return np.asarray(transformed_image, dtype=np.uint8), fallback_status or f'Display transform: active ({profile_name})'


def _imagecms_profile_for_color_space(color_space: str, *, imagecms_module: Any) -> object | None:
    icc_bytes = resolve_icc_profile_bytes(color_space)
    if icc_bytes is not None:
        try:
            return imagecms_module.ImageCmsProfile(BytesIO(icc_bytes))
        except (AttributeError, OSError, TypeError, ValueError, imagecms_module.PyCMSError):
            return None

    if color_space == DISPLAY_PREVIEW_COLOR_SPACE:
        try:
            return imagecms_module.createProfile(DISPLAY_PREVIEW_COLOR_SPACE)
        except (AttributeError, OSError, TypeError, ValueError, imagecms_module.PyCMSError):
            return None

    return None


def _encode_pixels_for_output_profile(
    image_data: np.ndarray,
    *,
    output_encoding: ColorEncoding,
    colour_module: Any,
) -> np.ndarray:
    if not output_encoding.is_linear:
        return image_data
    return colour_module.RGB_to_RGB(
        image_data,
        output_encoding.color_space,
        output_encoding.color_space,
        apply_cctf_decoding=False,
        apply_cctf_encoding=True,
    )


def prepare_output_display_image(
    image_data: np.ndarray,
    *,
    output_encoding: ColorEncoding | None = None,
    output_color_space: str | None = None,
    use_display_transform: bool,
    padding_pixels: float = 0.0,
    imagecms_module: Any,
    colour_module: Any,
    pil_image_module: Any,
) -> tuple[np.ndarray, str]:
    del padding_pixels
    output_encoding = _resolve_output_encoding(output_encoding, output_color_space)
    normalized_image = normalized_image_data(np.asarray(image_data)[..., :3])
    preview_source = normalized_image
    if output_encoding.is_linear:
        try:
            preview_source = colour_module.RGB_to_RGB(
                normalized_image,
                output_encoding.color_space,
                output_encoding.color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True,
            )
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
            preview_source = normalized_image
    preview_image = np.uint8(np.clip(preview_source, 0.0, 1.0) * 255)
    if not use_display_transform:
        return preview_image, display_transform_status_message(False, imagecms_module=imagecms_module)
    try:
        transformed_image, status = apply_display_transform(
            normalized_image,
            output_encoding=output_encoding,
            colour_module=colour_module,
            imagecms_module=imagecms_module,
            pil_image_module=pil_image_module,
        )
        return transformed_image, status
    except (AttributeError, LookupError, OSError, RuntimeError, ValueError, TypeError, imagecms_module.PyCMSError):
        return preview_image, 'Display transform: transform failed, using raw preview'


def execute_simulation_request(
    request: SimulationRequest,
    *,
    run_simulation_fn: Callable[[np.ndarray, object], np.ndarray],
    prepare_output_display_image_fn: Callable[..., tuple[np.ndarray, str]],
) -> SimulationResult:
    output_encoding = _resolve_output_encoding(request.output_encoding, request.output_color_space)
    scan = run_simulation_fn(request.image, request.params)
    scan_display, display_status = prepare_output_display_image_fn(
        scan,
        output_encoding=output_encoding,
        use_display_transform=request.use_display_transform,
    )
    return SimulationResult(
        mode_label=request.mode_label,
        display_image=scan_display,
        float_image=np.asarray(scan),
        use_display_transform=request.use_display_transform,
        status_message=display_status,
        output_encoding=output_encoding,
        output_color_space=output_encoding.color_space,
    )


def _resolve_output_encoding(
    output_encoding: ColorEncoding | None,
    output_color_space: str | None,
) -> ColorEncoding:
    if output_encoding is not None:
        return output_encoding
    return ColorEncoding(
        color_space=output_color_space or DISPLAY_PREVIEW_COLOR_SPACE,
        transfer="cctf",
        role="display",
    )
