import numpy as np
import colour


def _luminance_y(image, color_space, apply_cctf_decoding):
    image_XYZ = colour.RGB_to_XYZ(image, color_space, apply_cctf_decoding=apply_cctf_decoding)
    return image_XYZ[:, :, 1]


def _normalized_coords(image):
    """Return x, y coordinate arrays normalized so the long edge spans [-0.5, 0.5]."""
    norm_shape = image.shape[0:2] / np.max(image.shape[0:2])
    x = (np.arange(image.shape[1]) / image.shape[1] - 0.5) * norm_shape[1]
    y = (np.arange(image.shape[0]) / image.shape[0] - 0.5) * norm_shape[0]
    return x, y


def measure_autoexposure_ev(image, color_space='sRGB', apply_cctf_decoding=True, method='center_weighted'):
    image_Y = _luminance_y(image, color_space, apply_cctf_decoding)

    if method == 'average':
        # Uniform average over the entire frame.
        exposure = np.mean(image_Y) / 0.184

    elif method == 'median':
        exposure = np.median(image_Y) / 0.184

    elif method == 'center_weighted':
        # Gaussian falloff from center; sigma ~0.2–0.3 of the long edge.
        x, y = _normalized_coords(image)
        sigma = 0.2
        mask = np.exp(-(x ** 2 + y[:, None] ** 2) / (2 * sigma ** 2))
        mask /= np.sum(mask)
        exposure = np.sum(image_Y * mask) / 0.184

    elif method == 'partial':
        # Hard circular region covering ~15% radius (Canon Partial).
        x, y = _normalized_coords(image)
        radius = np.sqrt(x ** 2 + y[:, None] ** 2)
        mask = radius < 0.15
        if mask.sum() == 0:
            mask = np.ones_like(radius, dtype=bool)
        exposure = np.mean(image_Y[mask]) / 0.184

    elif method == 'matrix':
        # Divide into a 5×5 grid; weight each cell by a raised-cosine distance
        # from center so corner zones contribute less.
        h, w = image_Y.shape
        n_rows, n_cols = 5, 5
        cell_h = h // n_rows
        cell_w = w // n_cols
        zone_means = []
        zone_weights = []
        for r in range(n_rows):
            for c in range(n_cols):
                cell = image_Y[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                if cell.size == 0:
                    continue
                zone_means.append(np.mean(cell))
                dy = (r - (n_rows - 1) / 2) / ((n_rows - 1) / 2)
                dx = (c - (n_cols - 1) / 2) / ((n_cols - 1) / 2)
                dist = np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2)
                zone_weights.append(0.5 * (1.0 + np.cos(np.pi * dist)))
        zone_weights = np.array(zone_weights)
        zone_weights /= zone_weights.sum()
        exposure = float(np.dot(zone_weights, zone_means)) / 0.184

    elif method == 'multi_zone':
        # Three concentric rings: spot (0–5%), mid (5–25%), outer (25–50%),
        # weighted 50 / 30 / 20 to mimic a multi-pattern meter.
        x, y = _normalized_coords(image)
        radius = np.sqrt(x ** 2 + y[:, None] ** 2)
        ring_bounds = [(0.00, 0.05), (0.05, 0.25), (0.25, 0.50)]
        ring_weights = [0.50, 0.30, 0.20]
        weighted_sum = 0.0
        weight_total = 0.0
        for (r_min, r_max), w in zip(ring_bounds, ring_weights):
            mask = (radius >= r_min) & (radius < r_max)
            if mask.sum() == 0:
                continue
            weighted_sum += w * np.mean(image_Y[mask])
            weight_total += w
        exposure = (weighted_sum / weight_total if weight_total > 0 else np.mean(image_Y)) / 0.184

    elif method == 'highlight_weighted':
        # Bias toward brighter pixels to protect film highlights from clipping.
        # Weight = Y^2 so bright areas dominate; renormalize.
        weights = image_Y ** 2
        total = weights.sum()
        if total < 1e-12:
            weights = np.ones_like(image_Y)
            total = weights.sum()
        exposure = float(np.sum(image_Y * weights) / total) / 0.184

    else:
        exposure = 1.0

    exposure_compensation_ev = -np.log2(exposure)
    if np.isinf(exposure_compensation_ev):
        exposure_compensation_ev = 0.0
        print('Warning: Autoexposure is Inf. Setting autoexposure compensation to 0 EV.')
    return exposure_compensation_ev


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = np.random.uniform(0, 1, (3000, 2000, 3))
    exposure_ev = measure_autoexposure_ev(image)
    print(exposure_ev)
    plt.imshow(image)
    plt.show()
