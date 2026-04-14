from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qtpy import QtCore


DISPLAY_PREVIEW_COLOR_SPACE = 'sRGB'
QObject = getattr(QtCore, 'QObject')
QRunnable = getattr(QtCore, 'QRunnable')
Signal = getattr(QtCore, 'Signal')


@dataclass(slots=True)
class SimulationRequest:
    mode_label: str
    image: np.ndarray
    params: object
    output_color_space: str
    use_display_transform: bool


@dataclass(slots=True)
class SimulationResult:
    mode_label: str
    display_image: np.ndarray
    float_image: np.ndarray
    output_color_space: str
    use_display_transform: bool
    status_message: str


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
    output_color_space: str,
    colour_module: Any,
    imagecms_module: Any,
    pil_image_module: Any,
) -> tuple[np.ndarray, str]:
    display_profile, profile_name = display_profile_details(imagecms_module=imagecms_module)
    if display_profile is None:
        return np.uint8(np.clip(image_data, 0.0, 1.0) * 255), 'Display transform: no display profile, using raw preview'

    srgb_preview = colour_module.RGB_to_RGB(
        image_data,
        output_color_space,
        DISPLAY_PREVIEW_COLOR_SPACE,
        apply_cctf_decoding=True,
        apply_cctf_encoding=True,
    )
    srgb_preview_uint8 = np.uint8(np.clip(srgb_preview, 0.0, 1.0) * 255)
    source_profile = imagecms_module.createProfile(DISPLAY_PREVIEW_COLOR_SPACE)
    source_image = pil_image_module.fromarray(srgb_preview_uint8, mode='RGB')
    transformed_image = imagecms_module.profileToProfile(source_image, source_profile, display_profile, outputMode='RGB')
    return np.asarray(transformed_image, dtype=np.uint8), f'Display transform: active ({profile_name})'


def prepare_output_display_image(
    image_data: np.ndarray,
    *,
    output_color_space: str,
    use_display_transform: bool,
    padding_pixels: float = 0.0,
    imagecms_module: Any,
    colour_module: Any,
    pil_image_module: Any,
) -> tuple[np.ndarray, str]:
    del padding_pixels
    normalized_image = normalized_image_data(np.asarray(image_data)[..., :3])
    preview_image = np.uint8(np.clip(normalized_image, 0.0, 1.0) * 255)
    if not use_display_transform:
        return preview_image, display_transform_status_message(False, imagecms_module=imagecms_module)
    try:
        transformed_image, status = apply_display_transform(
            normalized_image,
            output_color_space=output_color_space,
            colour_module=colour_module,
            imagecms_module=imagecms_module,
            pil_image_module=pil_image_module,
        )
        return transformed_image, status
    except (OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        return preview_image, 'Display transform: transform failed, using raw preview'


def execute_simulation_request(
    request: SimulationRequest,
    *,
    run_simulation_fn: Callable[[np.ndarray, object], np.ndarray],
    prepare_output_display_image_fn: Callable[..., tuple[np.ndarray, str]],
) -> SimulationResult:
    scan = run_simulation_fn(request.image, request.params)
    scan_display, display_status = prepare_output_display_image_fn(
        scan,
        output_color_space=request.output_color_space,
        use_display_transform=request.use_display_transform,
    )
    return SimulationResult(
        mode_label=request.mode_label,
        display_image=scan_display,
        float_image=np.asarray(scan),
        output_color_space=request.output_color_space,
        use_display_transform=request.use_display_transform,
        status_message=display_status,
    )