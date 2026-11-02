from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class BasicAdjustmentsState:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0


ADJUSTMENT_LABELS = {
    'exposure': ('Exposure', -5.0, 5.0, 0.1, 1, 'EV'),
    'contrast': ('Contrast', -100.0, 100.0, 1.0, 0, ''),
    'highlights': ('Highlights', -100.0, 100.0, 1.0, 0, ''),
    'shadows': ('Shadows', -100.0, 100.0, 1.0, 0, ''),
    'whites': ('Whites', -100.0, 100.0, 1.0, 0, ''),
    'blacks': ('Blacks', -100.0, 100.0, 1.0, 0, ''),
    'texture': ('Texture', -100.0, 100.0, 1.0, 0, ''),
    'clarity': ('Clarity', -100.0, 100.0, 1.0, 0, ''),
    'dehaze': ('Dehaze', -100.0, 100.0, 1.0, 0, ''),
    'vibrance': ('Vibrance', -100.0, 100.0, 1.0, 0, ''),
    'saturation': ('Saturation', -100.0, 100.0, 1.0, 0, ''),
}


def has_basic_adjustments(state: BasicAdjustmentsState) -> bool:
    return any(getattr(state, name) != 0.0 for name in ADJUSTMENT_LABELS)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _scale_about_luminance(rgb: np.ndarray, luminance: np.ndarray, new_luminance: np.ndarray) -> np.ndarray:
    scale = np.divide(new_luminance, luminance, out=np.zeros_like(new_luminance), where=luminance > 1e-8)
    scaled = rgb * scale[..., None]
    black_pixels = luminance <= 1e-8
    if np.any(black_pixels):
        scaled[black_pixels] = new_luminance[black_pixels][:, None]
    return scaled


def _apply_saturation(rgb: np.ndarray, amount: float, *, vibrance: bool) -> np.ndarray:
    if amount == 0.0:
        return rgb
    luminance = np.sum(rgb * np.array((0.2126, 0.7152, 0.0722)), axis=-1)
    saturation = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    scale = amount / 100.0
    if vibrance:
        scale *= 1.0 - np.clip(saturation / np.maximum(np.abs(luminance), 1e-5), 0.0, 1.0)
    return luminance[..., None] + (rgb - luminance[..., None]) * (1.0 + scale)


def apply_basic_adjustments(image: np.ndarray, state: BasicAdjustmentsState) -> np.ndarray:
    """Apply Lightroom-style global and detail adjustments to sRGB-encoded RGB."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] < 3:
        raise ValueError(f'Expected an RGB image, got shape {source.shape}.')
    if not has_basic_adjustments(state):
        return np.array(source[..., :3], copy=True)

    encoded = np.maximum(source[..., :3], 0.0)
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    linear *= 2.0 ** state.exposure

    luminance = np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1)
    if state.contrast:
        factor = 1.0 + state.contrast / 100.0
        linear = 0.18 + (linear - 0.18) * factor
        luminance = np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1)

    if state.highlights:
        mask = _smoothstep((luminance - 0.15) / 0.85)
        luminance = luminance * np.exp2((state.highlights / 100.0) * mask * 1.5)
        linear = _scale_about_luminance(linear, np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1), luminance)
    if state.shadows:
        mask = 1.0 - _smoothstep(luminance / 0.65)
        luminance = luminance * np.exp2((state.shadows / 100.0) * mask * 1.5)
        linear = _scale_about_luminance(linear, np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1), luminance)
    if state.whites:
        mask = _smoothstep((luminance - 0.55) / 0.45)
        luminance = luminance + (state.whites / 100.0) * mask * 0.25
        linear = _scale_about_luminance(linear, np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1), np.maximum(luminance, 0.0))
    if state.blacks:
        mask = 1.0 - _smoothstep(luminance / 0.35)
        luminance = luminance + (state.blacks / 100.0) * mask * 0.08
        linear = _scale_about_luminance(linear, np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1), np.maximum(luminance, 0.0))

    luminance = np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1)
    for amount, sigma, strength in ((state.texture, 1.2, 1.5), (state.clarity, 8.0, 1.0), (state.dehaze, 20.0, 0.8)):
        if amount:
            local = gaussian_filter(luminance, sigma=sigma, mode='reflect')
            linear += (luminance - local)[..., None] * (amount / 100.0 * strength)
            luminance = np.sum(linear * np.array((0.2126, 0.7152, 0.0722)), axis=-1)

    linear = _apply_saturation(linear, state.vibrance, vibrance=True)
    linear = _apply_saturation(linear, state.saturation, vibrance=False)
    linear = np.maximum(linear, 0.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
