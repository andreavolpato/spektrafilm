from dataclasses import dataclass

import numpy as np

GAMMA = 2.2
EPSILON = 1e-4
INPUT_DENSITY_MAX = 6.0
RENDER_DENSITY_MAX = 8.0

SHADOW_RGB = np.array([0.34, 0.39, 0.27], dtype=np.float32)
RECEIVER_RGB = np.array([0.04, 0.08, 0.15], dtype=np.float32)
SILVER_RGB = np.array([0.18, 0.19, 0.20], dtype=np.float32)
SILVER_CROSS_RGB = np.array([0.03, 0.02, 0.04], dtype=np.float32)
ANTI_HIGHLIGHT_RGB = np.array([0.10, 0.06, 0.02], dtype=np.float32)
ANTI_INVERT_RGB = np.array([0.13, 0.07, 0.18], dtype=np.float32)
ANTI_LATENT_RGB = np.array([0.04, 0.02, 0.06], dtype=np.float32)
WARM_FLASH_RGB = np.array([0.02, 0.08, 0.19], dtype=np.float32)
STAIN_RGB = np.array([0.03, 0.06, 0.11], dtype=np.float32)

@dataclass(slots=True)
class PolaroidState:
    layer_stack: np.ndarray


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def normalized_sigmoid(t, center, slope):
    start = _expit(-center * slope)
    end = _expit((1.0 - center) * slope)
    value = _expit((t - center) * slope)
    return np.clip((value - start) / (end - start + EPSILON), 0.0, 1.0)


def delayed_rise(t, center, slope, tail_power):
    rise = normalized_sigmoid(t, center, slope)
    return rise + (1.0 - rise) * (t**tail_power)


def windowed_pulse(t, on_center, on_slope, off_center, off_slope, decay_power):
    rising = normalized_sigmoid(t, on_center, on_slope)
    falling = 1.0 - normalized_sigmoid(t, off_center, off_slope)
    return rising * falling * ((1.0 - t) ** decay_power)


def normalize_image(img):
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("img must be an HxWx3 image")

    img = img[..., :3]
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

    return np.clip(img, EPSILON, 1.0)


def normalize_per_channel(values):
    return values / (values.max(axis=(0, 1), keepdims=True) + EPSILON)


def split_chroma_layers(chroma):
    zero = np.zeros_like(chroma[..., 0], dtype=np.float32)
    cyan_layer = np.stack([chroma[..., 0], zero, zero], axis=2)
    magenta_layer = np.stack([zero, chroma[..., 1], zero], axis=2)
    yellow_layer = np.stack([zero, zero, chroma[..., 2]], axis=2)
    return cyan_layer, magenta_layer, yellow_layer


def build_layer_stack(img):
    img = normalize_image(img)
    density = np.clip(-np.log(img**GAMMA), 0.0, INPUT_DENSITY_MAX).astype(np.float32)

    neutral = density.min(axis=2, keepdims=True)
    chroma = density - neutral
    density_latent = normalize_per_channel(density)
    chroma_latent = normalize_per_channel(chroma)
    neutral_latent = neutral / (neutral.max() + EPSILON)

    shadow = np.clip(np.sum(density * SHADOW_RGB, axis=2, keepdims=True), 0.0, 1.75) / 1.75
    highlight = 1.0 - shadow
    chroma_strength = chroma_latent.max(axis=2, keepdims=True)
    warmth = np.clip(
        (chroma[..., 1:2] + 1.25 * chroma[..., 2:3] - 0.85 * chroma[..., 0:1])
        / (chroma.sum(axis=2, keepdims=True) + EPSILON),
        0.0,
        1.0,
    )

    cyan_layer, magenta_layer, yellow_layer = split_chroma_layers(chroma)
    neutral_layer = np.repeat(neutral, 3, axis=2)
    receiver_layer = (0.30 + 0.70 * highlight) * RECEIVER_RGB

    silver_map = np.clip(0.24 + 0.46 * shadow + 0.30 * neutral_latent, 0.0, 1.0)
    silver_layer = np.clip(
        silver_map * SILVER_RGB + chroma_latent[..., [1, 2, 0]] * SILVER_CROSS_RGB,
        0.0,
        0.28,
    )

    anti_layer = (
        highlight * ANTI_HIGHLIGHT_RGB
        + (1.0 - density_latent[..., [2, 0, 1]]) * ANTI_INVERT_RGB
        + chroma_latent[..., [1, 2, 0]] * ANTI_LATENT_RGB
    )
    anti_layer *= 0.40 + 0.60 * highlight

    warm_layer = (
        0.24 + 0.42 * highlight + 0.34 * np.maximum(warmth, chroma_strength)
    ) * WARM_FLASH_RGB
    stain_layer = highlight * STAIN_RGB

    return np.stack(
        [
            neutral_layer,
            yellow_layer,
            magenta_layer,
            cyan_layer,
            receiver_layer,
            silver_layer,
            anti_layer,
            warm_layer,
            stain_layer,
        ],
        axis=0,
    ).astype(np.float32)


def prepare_polaroid_state(img):
    return PolaroidState(layer_stack=build_layer_stack(img))


def layer_coefficients(t):
    t = float(np.clip(t, 0.0, 1.0))
    body = delayed_rise(t, center=0.19, slope=7.4, tail_power=3.8)

    neutral = body * delayed_rise(t, center=0.12, slope=8.0, tail_power=2.8)
    yellow = body * delayed_rise(t, center=0.22, slope=11.0, tail_power=3.0)
    magenta = body * delayed_rise(t, center=0.31, slope=10.2, tail_power=3.4)
    cyan = body * delayed_rise(t, center=0.52, slope=9.4, tail_power=5.4)
    receiver = 1.08 * ((1.0 - t) ** 0.46)
    silver = 1.00 * windowed_pulse(t, 0.05, 32.0, 0.44, 8.0, 0.24)
    anti = 1.22 * windowed_pulse(t, 0.10, 54.0, 0.28, 28.0, 0.18)
    warm = 0.96 * windowed_pulse(t, 0.12, 20.0, 0.80, 6.5, 0.34)
    stain = 0.68 * windowed_pulse(t, 0.03, 24.0, 0.98, 4.0, 0.08)

    return np.array(
        [neutral, yellow, magenta, cyan, receiver, silver, anti, warm, stain],
        dtype=np.float32,
    )


def mix_layers(layer_stack, coefficients):
    return np.tensordot(coefficients, layer_stack, axes=(0, 0))


def density_to_rgb(density):
    density = np.clip(density, 0.0, RENDER_DENSITY_MAX)
    return np.exp(-density / GAMMA)


def render_polaroid_frame(img_or_state, t):
    state = img_or_state if isinstance(img_or_state, PolaroidState) else prepare_polaroid_state(img_or_state)
    density = mix_layers(state.layer_stack, layer_coefficients(t))
    return density_to_rgb(density)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy import ndimage

    DEMO_IMAGE_PATH = Path("img/readme/banner.png")
    
    def show_polaroid_animation(img, frames=60, interval=16):
        state = prepare_polaroid_state(img)
        frame_times = np.linspace(0.0, 1.0, num=max(frames, 1), dtype=np.float32)

        fig, ax = plt.subplots()
        artist = ax.imshow(render_polaroid_frame(state, 0.0), animated=True)
        ax.set_axis_off()

        def update(t):
            artist.set_data(render_polaroid_frame(state, float(t)))
            return (artist,)

        animation = FuncAnimation(fig, update, frames=frame_times, interval=interval, blit=True)
        plt.show()
        return animation



    def _resize_image_nearest(img, target_width, target_height):
        zoom_factors = (target_height / img.shape[0], target_width / img.shape[1], 1)
        return ndimage.zoom(img, zoom_factors, order=0)

    def load_demo_image(path=DEMO_IMAGE_PATH, target_width=768, target_height=512):
        demo_img = plt.imread(path)
        return _resize_image_nearest(demo_img, target_width=target_width, target_height=target_height)


    def main():
        show_polaroid_animation(load_demo_image())

    main()