from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PLOT_PATCH_ORDER = [
    'neg_print_midgray',
    'neg_print_red',
    'neg_print_green',
    'neg_print_blue',
    'neg_scan_midgray',
    'pos_scan_midgray',
    'pos_scan_red',
]
INPUT_PATCHES = {
    'neg_print_midgray': (0.184, 0.184, 0.184),
    'neg_print_red': (0.5, 0.05, 0.05),
    'neg_print_green': (0.05, 0.5, 0.05),
    'neg_print_blue': (0.05, 0.05, 0.5),
    'neg_scan_midgray': (0.184, 0.184, 0.184),
    'pos_scan_midgray': (0.184, 0.184, 0.184),
    'pos_scan_red': (0.5, 0.05, 0.05),
}


def _ensure_src_on_path(repo_root: Path) -> None:
    src_path = repo_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _patch(rgb: tuple[float, float, float], size: int = 4) -> np.ndarray:
    return np.ones((size, size, 3), dtype=np.float64) * np.asarray(rgb, dtype=np.float64)


def _configure_params(params):
    params.io.input_color_space = 'sRGB'
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    params.io.full_image = True
    params.io.preview_resize_factor = 1.0
    params.io.upscale_factor = 1.0
    params.io.crop = False
    params.camera.auto_exposure = False
    params.camera.exposure_compensation_ev = 0.0
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.print_render.glare.active = False
    params.settings.use_enlarger_lut = False
    params.settings.use_scanner_lut = False
    return params


def evaluate_current_worktree(
    repo_root: Path,
    *,
    film_profile: str,
    print_profile: str,
    positive_film_profile: str,
) -> dict[str, list[float]]:
    _ensure_src_on_path(repo_root)

    from spektrafilm import create_params, simulate

    results: dict[str, list[float]] = {}
    params = _configure_params(create_params(film_profile=film_profile, print_profile=print_profile))
    results['neutral_filters'] = [
        float(params.enlarger.c_filter_neutral),
        float(params.enlarger.m_filter_neutral),
        float(params.enlarger.y_filter_neutral),
    ]
    for name, rgb in {
        'neg_print_midgray': (0.184, 0.184, 0.184),
        'neg_print_red': (0.5, 0.05, 0.05),
        'neg_print_green': (0.05, 0.5, 0.05),
        'neg_print_blue': (0.05, 0.05, 0.5),
    }.items():
        results[name] = np.mean(simulate(_patch(rgb), params), axis=(0, 1)).tolist()

    params.io.scan_film = True
    results['neg_scan_midgray'] = np.mean(simulate(_patch((0.184, 0.184, 0.184)), params), axis=(0, 1)).tolist()

    params = _configure_params(create_params(film_profile=positive_film_profile, print_profile=print_profile))
    params.io.scan_film = True
    for name, rgb in {
        'pos_scan_midgray': (0.184, 0.184, 0.184),
        'pos_scan_red': (0.5, 0.05, 0.05),
    }.items():
        results[name] = np.mean(simulate(_patch(rgb), params), axis=(0, 1)).tolist()
    return results


def _run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _evaluate_revision(
    revision: str,
    *,
    film_profile: str,
    print_profile: str,
    positive_film_profile: str,
) -> dict[str, list[float]]:
    with tempfile.TemporaryDirectory(prefix='spektrafilm-rev-') as temp_dir:
        worktree_path = Path(temp_dir) / revision.replace('/', '_').replace('\\', '_')
        subprocess.run(
            ['git', 'worktree', 'add', '--detach', str(worktree_path), revision],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(worktree_path / 'src')
            output = _run_subprocess(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    '--evaluate-current-worktree',
                    '--film-profile', film_profile,
                    '--print-profile', print_profile,
                    '--positive-film-profile', positive_film_profile,
                ],
                cwd=worktree_path,
                env=env,
            )
            return json.loads(output)
        finally:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', str(worktree_path)],
                cwd=str(REPO_ROOT),
                check=True,
                capture_output=True,
                text=True,
            )


def _vector_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _print_summary(results_by_revision: dict[str, dict[str, list[float]]]) -> None:
    print(json.dumps(results_by_revision, indent=2, sort_keys=True))
    revisions = list(results_by_revision)
    if len(revisions) < 2:
        return

    print('\nPairwise distances')
    keys = list(results_by_revision[revisions[0]].keys())
    for previous, current in zip(revisions, revisions[1:]):
        print(f'{previous} -> {current}')
        for key in keys:
            distance = _vector_distance(results_by_revision[previous][key], results_by_revision[current][key])
            print(f'  {key}: {distance:.6f}')


def _format_patch_label(patch_name: str) -> str:
    return patch_name.replace('_', ' ')


def _format_revision_label(revision: str) -> str:
    if len(revision) >= 7 and all(character in '0123456789abcdef' for character in revision.lower()):
        return revision[:7]
    return revision


def _encode_srgb_for_display(rgb: np.ndarray) -> np.ndarray:
    import colour

    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, None)
    return colour.cctf_encoding(rgb, function='sRGB')


def _plot_patch_grid(results_by_revision: dict[str, dict[str, list[float]]], output_path: Path | None, *, show_plot: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb_to_hsv

    revisions = list(results_by_revision)
    patch_names = [name for name in PLOT_PATCH_ORDER if name in results_by_revision[revisions[0]]]
    if not patch_names:
        return

    nrows = len(patch_names)
    ncols = len(revisions) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.6 * ncols, 1.8 * nrows))
    axes = np.atleast_2d(axes)
    if axes.shape[0] != nrows:
        axes = axes.T

    for row_index, patch_name in enumerate(patch_names):
        input_rgb_linear = np.asarray(INPUT_PATCHES[patch_name], dtype=np.float64)
        input_rgb_display = np.clip(_encode_srgb_for_display(input_rgb_linear), 0.0, 1.0)
        input_tile = np.ones((32, 32, 3), dtype=np.float64) * input_rgb_display
        input_ax = axes[row_index, 0]
        input_ax.imshow(input_tile)
        input_ax.set_xticks([])
        input_ax.set_yticks([])
        if row_index == 0:
            input_ax.set_title('Input sRGB')
        input_ax.set_ylabel(_format_patch_label(patch_name))
        input_hsv = rgb_to_hsv(input_rgb_display[None, None, :])[0, 0]
        input_ax.text(
            0.5,
            -0.12,
            f'Lin {input_rgb_linear[0]:.2f} {input_rgb_linear[1]:.2f} {input_rgb_linear[2]:.2f}\nsRGB {input_rgb_display[0]:.2f} {input_rgb_display[1]:.2f} {input_rgb_display[2]:.2f} S {input_hsv[1]:.2f}',
            transform=input_ax.transAxes,
            ha='center',
            va='top',
            fontsize=8,
        )

        for col_index, revision in enumerate(revisions, start=1):
            rgb_linear = np.asarray(results_by_revision[revision][patch_name], dtype=np.float64)
            rgb_display = np.clip(_encode_srgb_for_display(rgb_linear), 0.0, 1.0)
            tile = np.ones((32, 32, 3), dtype=np.float64) * rgb_display
            ax = axes[row_index, col_index]
            ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(f'{_format_revision_label(revision)} sRGB')
            hsv = rgb_to_hsv(rgb_display[None, None, :])[0, 0]
            ax.text(
                0.5,
                -0.12,
                f'Lin {rgb_linear[0]:.2f} {rgb_linear[1]:.2f} {rgb_linear[2]:.2f}\nsRGB {rgb_display[0]:.2f} {rgb_display[1]:.2f} {rgb_display[2]:.2f} S {hsv[1]:.2f}',
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=8,
            )

    fig.suptitle('Simulation Patch Comparison', fontsize=14)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare deterministic simulation outputs across git revisions.')
    parser.add_argument('revisions', nargs='*', help='Git revisions to compare in order.')
    parser.add_argument('--film-profile', default='kodak_portra_400')
    parser.add_argument('--print-profile', default='kodak_portra_endura')
    parser.add_argument('--positive-film-profile', default='kodak_ektachrome_100')
    parser.add_argument('--plot-file', type=Path, help='Optional output path for a visual patch comparison plot.')
    parser.add_argument('--show-plot', action='store_true', help='Display the patch comparison plot interactively.')
    parser.add_argument('--evaluate-current-worktree', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluate_current_worktree:
        results = evaluate_current_worktree(
            Path.cwd(),
            film_profile=args.film_profile,
            print_profile=args.print_profile,
            positive_film_profile=args.positive_film_profile,
        )
        print(json.dumps(results, sort_keys=True))
        return

    if not args.revisions:
        raise SystemExit('Provide at least one revision or use --evaluate-current-worktree.')

    results_by_revision = {
        revision: _evaluate_revision(
            revision,
            film_profile=args.film_profile,
            print_profile=args.print_profile,
            positive_film_profile=args.positive_film_profile,
        )
        for revision in args.revisions
    }
    _print_summary(results_by_revision)
    if args.plot_file is not None or args.show_plot:
        _plot_patch_grid(results_by_revision, args.plot_file, show_plot=args.show_plot)


if __name__ == '__main__':
    main()