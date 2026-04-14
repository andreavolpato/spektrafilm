from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.util import img_as_float32


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEST_TILE_PATH = ASSETS_DIR / "virtual_paper_watermark.png"

DEFAULT_CANVAS_SIZE = (860, 1280)  # (width, height) in px
GRID_SCALE = 0.22  # logo long edge / canvas long edge
OVERLAP_FRACTION = 0.25  # only used if center_distance is None
DEFAULT_CENTER_DISTANCE = 0.25  # tile-center spacing / canvas long edge
ROMBIC_GRID_ANGLE = 72.0  # degrees
GLOBAL_ROTATION = 16.0  # degrees
PHASE_HORIZONTAL = 0.0  # long-edge fraction
PHASE_VERTICAL = 0.0  # long-edge fraction
DEFAULT_LOGO_RGB = np.array((215,215,215))/255  # RGB 0..1
DEFAULT_BACKGROUND_RGB = np.array((242,242,242))/255  # RGB 0..1
DEFAULT_GLARE = 0.25  # paper glare strength


@lru_cache(maxsize=1)
def load_logo_alpha():
    return np.ascontiguousarray(img_as_float32(io.imread(TEST_TILE_PATH)).astype(np.float32)[..., 3])


@lru_cache(maxsize=32)
def prepare_tile_stamp(scale, angle):
    alpha = load_logo_alpha()
    if scale < 1.0:
        alpha = ndimage.gaussian_filter(alpha, sigma=max(0.0, 0.5 / scale - 0.5), mode="nearest")
    if not np.isclose(scale, 1.0):
        alpha = ndimage.zoom(alpha, zoom=scale, order=1, mode="constant", cval=0.0)
    scaled_width = alpha.shape[1]
    if not np.isclose(angle, 0.0):
        alpha = ndimage.rotate(alpha, angle=angle, axes=(1, 0), reshape=True, order=1, mode="constant", cval=0.0)
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    ij = np.argwhere(alpha > 1e-4)
    y0, x0 = ij.min(0)
    y1, x1 = ij.max(0) + 1
    alpha = np.ascontiguousarray(alpha[y0:y1, x0:x1])
    return alpha, np.ascontiguousarray(1.0 - alpha), scaled_width


@lru_cache(maxsize=32)
def get_cached_glare_map(width, height, seed):
    rng = np.random.default_rng(seed)
    a = rng.random((height, width), dtype=np.float32)
    b = np.power(rng.random((height, width), dtype=np.float32), 10.0)
    return np.ascontiguousarray((0.88 * a + 0.12 * b).astype(np.float32))


def build_glare_map(shape, glare, seed):
    if glare == 0.0:
        return None
    height, width = shape
    if seed is None:
        rng = np.random.default_rng()
        a = rng.random((height, width), dtype=np.float32)
        b = np.power(rng.random((height, width), dtype=np.float32), 10.0)
        return np.ascontiguousarray((0.88 * a + 0.12 * b).astype(np.float32))
    return get_cached_glare_map(width, height, int(seed))


def build_rhombic_basis(
    tile_width,
    overlap,
    grid_angle,
    rotation,
    center_distance,
):
    half = np.deg2rad(0.5 * grid_angle)
    distance = tile_width * (1.0 - overlap) / np.cos(half) if center_distance is None else center_distance
    basis = np.array(
        [[distance * np.cos(half), distance * np.sin(half)], [distance * np.cos(half), -distance * np.sin(half)]],
        dtype=np.float32,
    )
    theta = np.deg2rad(rotation)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], dtype=np.float32)
    return basis @ rot.T


def generate_lattice_centers(
    canvas_shape,
    stamp_shape,
    basis,
    phase_x,
    phase_y,
):
    height, width = canvas_shape
    stamp_h, stamp_w = stamp_shape
    origin = np.array([0.5 * width + phase_x, 0.5 * height + phase_y], dtype=np.float32)
    radius = 0.5 * np.hypot(stamp_w, stamp_h)
    limit = int(np.ceil((np.hypot(width, height) + 2.0 * radius) / np.min(np.linalg.norm(basis, axis=1)))) + 2
    i, j = np.meshgrid(np.arange(-limit, limit + 1), np.arange(-limit, limit + 1), indexing="ij")
    centers = origin + i.reshape(-1, 1) * basis[0] + j.reshape(-1, 1) * basis[1]
    visible = (
        (centers[:, 0] >= -radius)
        & (centers[:, 0] <= width + radius)
        & (centers[:, 1] >= -radius)
        & (centers[:, 1] <= height + radius)
    )
    centers = centers[visible]
    return centers[np.lexsort((centers[:, 1], centers[:, 0]))]


def multiply_tile_transmittance(transmittance, inverse_alpha, center):
    stamp_h, stamp_w = inverse_alpha.shape
    center_x, center_y = center
    x0 = int(np.round(center_x - 0.5 * stamp_w))
    y0 = int(np.round(center_y - 0.5 * stamp_h))
    x1 = x0 + stamp_w
    y1 = y0 + stamp_h
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(transmittance.shape[1], x1)
    dy1 = min(transmittance.shape[0], y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    sx0 = dx0 - x0
    sy0 = dy0 - y0
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    region = transmittance[dy0:dy1, dx0:dx1]
    np.multiply(region, inverse_alpha[sy0:sy1, sx0:sx1], out=region)


def render_virtual_photo_paper_back(
    canvas_size=DEFAULT_CANVAS_SIZE,
    zoom=GRID_SCALE,
    angle=GLOBAL_ROTATION,
    overlap=OVERLAP_FRACTION,
    center_distance=DEFAULT_CENTER_DISTANCE,
    logo_rgb=DEFAULT_LOGO_RGB,
    background_rgb=DEFAULT_BACKGROUND_RGB,
    glare=DEFAULT_GLARE,
    rombic_grid_angle=ROMBIC_GRID_ANGLE,
    phase_horizontal=PHASE_HORIZONTAL,
    phase_vertical=PHASE_VERTICAL,
    seed=0,
    measure_timing=False,
):
    t0 = perf_counter() if measure_timing else 0.0
    canvas_width, canvas_height = (canvas_size, canvas_size) if isinstance(canvas_size, int) else canvas_size
    canvas_width = int(canvas_width)
    canvas_height = int(canvas_height)
    canvas_shape = (canvas_height, canvas_width)
    canvas_long_edge = float(max(canvas_width, canvas_height))
    background = np.asarray(background_rgb, dtype=np.float32)
    logo = np.asarray(DEFAULT_LOGO_RGB if logo_rgb is None else logo_rgb, dtype=np.float32)

    source_alpha = load_logo_alpha()
    tile_scale = float(zoom) * canvas_long_edge / float(max(source_alpha.shape))
    tile_alpha, inverse_alpha, scaled_width = prepare_tile_stamp(tile_scale, float(angle))
    t1 = perf_counter() if measure_timing else 0.0

    spacing = None if center_distance is None else float(center_distance) * canvas_long_edge
    basis = build_rhombic_basis(scaled_width, overlap, rombic_grid_angle, angle, spacing)
    centers = generate_lattice_centers(
        canvas_shape,
        tile_alpha.shape,
        basis,
        float(phase_horizontal) * canvas_long_edge,
        float(phase_vertical) * canvas_long_edge,
    )
    t2 = perf_counter() if measure_timing else 0.0

    transmittance = np.ones(canvas_shape, dtype=np.float32)
    for center in centers:
        multiply_tile_transmittance(transmittance, inverse_alpha, center)
    t3 = perf_counter() if measure_timing else 0.0

    glare_map = build_glare_map(canvas_shape, glare, seed)
    delta = background - logo
    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.float32)
    if glare_map is None:
        for channel in range(3):
            np.multiply(transmittance, delta[channel], out=canvas[..., channel])
            canvas[..., channel] += logo[channel]
    else:
        paper = np.empty(canvas_shape, dtype=np.float32)
        np.multiply(transmittance, 0.70, out=paper)
        paper += 0.30

        variation = np.empty(canvas_shape, dtype=np.float32)
        np.multiply(glare_map, 2.0, out=variation)
        variation -= 1.0
        variation *= paper
        variation *= 0.07 * glare
        variation += 1.0

        lift = np.power(glare_map, 6.0)
        lift *= paper
        lift *= 0.42 * glare

        for channel in range(3):
            np.multiply(transmittance, delta[channel], out=canvas[..., channel])
            canvas[..., channel] += logo[channel]
            canvas[..., channel] *= variation
            canvas[..., channel] += (1.0 - canvas[..., channel]) * lift
    np.clip(canvas, 0.0, 1.0, out=canvas)
    t4 = perf_counter() if measure_timing else 0.0

    timings = None
    if measure_timing:
        timings = {
            "stamp_ms": 1000.0 * (t1 - t0),
            "grid_ms": 1000.0 * (t2 - t1),
            "transmittance_ms": 1000.0 * (t3 - t2),
            "finalize_ms": 1000.0 * (t4 - t3),
            "render_ms": 1000.0 * (t4 - t0),
        }

    return canvas, centers, timings


def virtual_photo_paper_back(
    canvas_size=DEFAULT_CANVAS_SIZE,
    zoom=GRID_SCALE,
    angle=GLOBAL_ROTATION,
    overlap=OVERLAP_FRACTION,
    center_distance=DEFAULT_CENTER_DISTANCE,
    logo_rgb=DEFAULT_LOGO_RGB,
    background_rgb=DEFAULT_BACKGROUND_RGB,
    glare=DEFAULT_GLARE,
    rombic_grid_angle=ROMBIC_GRID_ANGLE,
    phase_horizontal=PHASE_HORIZONTAL,
    phase_vertical=PHASE_VERTICAL,
    seed=0,
    print_timing=False,
):
    canvas, centers, timings = render_virtual_photo_paper_back(
        canvas_size=canvas_size,
        zoom=zoom,
        angle=angle,
        overlap=overlap,
        center_distance=center_distance,
        logo_rgb=logo_rgb,
        background_rgb=background_rgb,
        glare=glare,
        rombic_grid_angle=rombic_grid_angle,
        phase_horizontal=phase_horizontal,
        phase_vertical=phase_vertical,
        seed=seed,
        measure_timing=print_timing,
    )
    if print_timing and timings is not None:
        canvas_width, canvas_height = (canvas_size, canvas_size) if isinstance(canvas_size, int) else canvas_size
        print(
            "virtual_photo_paper_back"
            f" | canvas={int(canvas_width)}x{int(canvas_height)}"
            f" | tiles={len(centers)}"
            f" | stamp={timings['stamp_ms']:.2f} ms"
            f" | grid={timings['grid_ms']:.2f} ms"
            f" | transmittance={timings['transmittance_ms']:.2f} ms"
            f" | finalize={timings['finalize_ms']:.2f} ms"
            f" | render={timings['render_ms']:.2f} ms"
        )
    return np.ascontiguousarray(np.rint(canvas * 255.0).astype(np.uint8))


_source_alpha = load_logo_alpha()
prepare_tile_stamp(float(GRID_SCALE) * float(max(DEFAULT_CANVAS_SIZE)) / float(max(_source_alpha.shape)), float(GLOBAL_ROTATION))
get_cached_glare_map(DEFAULT_CANVAS_SIZE[0], DEFAULT_CANVAS_SIZE[1], 0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_virtual_photo_paper_back(canvas_rgb):
        canvas_height, canvas_width = canvas_rgb.shape[:2]
        figure = plt.figure(figsize=(canvas_width / 100.0, canvas_height / 100.0), dpi=100)
        axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axis.imshow(canvas_rgb)
        axis.set_axis_off()
        return figure, axis

    canvas = virtual_photo_paper_back(print_timing=True)
    plot_virtual_photo_paper_back(canvas)
    plt.show()