"""Plot a film gray ramp before and after refinement.

Negative film inputs are rendered through the print path, while positive film
inputs are rendered by scanning the film directly.
"""

from __future__ import annotations

import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np

from spektrafilm import load_profile
from spektrafilm.runtime.api import Simulator, digest_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm_profile_creator.core.balancing import (
    balance_film_sensitivity,
    prelminary_neutral_shift,
    reconstruct_metameric_neutral,
)
from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.densitometer import densitometer_normalization, unmix_density
from spektrafilm_profile_creator.core.profile_transforms import remove_density_min
from spektrafilm_profile_creator.data.loader import load_raw_profile
from spektrafilm_profile_creator.neutral_print_filters import fit_neutral_filters
from spektrafilm_profile_creator.reconstruction.dye_reconstruction import reconstruct_dye_density
from spektrafilm_profile_creator.refinement.common import MIDGRAY_RGB, MIDGRAY_RGB_VECTOR
from spektrafilm_profile_creator.workflows import process_raw_profile


DEFAULT_FILM = 'fujifilm_velvia_100'
DEFAULT_EV_RAMP = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Plot a neutral gray ramp for a film profile, comparing the '
            'profile before and after refinement. Negative film is printed '
            'to paper, while positive film is scanned directly.'
        )
    )
    parser.add_argument(
        'film',
        nargs='?',
        default=DEFAULT_FILM,
        help='Film profile to inspect.',
    )
    parser.add_argument(
        '--paper',
        help='Print paper profile. If omitted, the script tries to infer the target paper from profile metadata.',
    )
    parser.add_argument(
        '--ev',
        nargs='*',
        type=float,
        default=list(DEFAULT_EV_RAMP),
        help='Exposure-compensation EV samples to render.',
    )
    parser.add_argument(
        '--save',
        help='Optional output path for the figure image.',
    )
    return parser.parse_args()


def infer_target_paper(raw_profile, film_profile, explicit_paper: str | None) -> str:
    if explicit_paper:
        return explicit_paper
    if film_profile.info.target_print:
        return str(film_profile.info.target_print)
    if raw_profile is not None and raw_profile.info.target_print:
        return str(raw_profile.info.target_print)
    raise ValueError(
        f'Unable to infer a target paper for {raw_profile.info.stock!r}. Pass --paper explicitly.'
    )


def process_film_profiles(film_name: str):
    raw_profile = load_raw_profile(film_name)
    if not raw_profile.info.is_film:
        raise ValueError(
            f'{film_name!r} must be a film profile. Got '
            f'type={raw_profile.info.type!r}, support={raw_profile.info.support!r}.'
        )

    if raw_profile.info.type == 'negative':
        source_profile = reconstruct_dye_density(
            raw_profile.as_profile(),
            model=raw_profile.recipe.dye_density_reconstruct_model,
        )
        source_profile = densitometer_normalization(source_profile)
        source_profile = balance_film_sensitivity(source_profile)
        source_profile = remove_density_min(source_profile)
        source_profile = prelminary_neutral_shift(source_profile)
        source_profile = unmix_density(source_profile)
    elif raw_profile.info.type == 'positive':
        source_profile = remove_density_min(raw_profile.as_profile(), reconstruct_base_density=True)
        source_profile = reconstruct_metameric_neutral(source_profile)
        source_profile = densitometer_normalization(source_profile)
        source_profile = balance_film_sensitivity(source_profile)
        source_profile = prelminary_neutral_shift(source_profile, per_channel_shift=False)
        source_profile = unmix_density(source_profile)
    else:
        raise ValueError(
            f'{film_name!r} must be a negative or positive film profile. Got '
            f'type={raw_profile.info.type!r}, support={raw_profile.info.support!r}.'
        )

    refined_profile = process_raw_profile(raw_profile)
    if not refined_profile.info.is_film or refined_profile.info.type != raw_profile.info.type:
        raise ValueError(
            f'{film_name!r} refinement returned an unexpected profile type. Got '
            f'type={refined_profile.info.type!r}, support={refined_profile.info.support!r}.'
        )
    return raw_profile, source_profile, refined_profile


def build_negative_film_params(film_profile, paper_name: str) -> tuple[RuntimePhotoParams, object]:

    params = RuntimePhotoParams(
        film=film_profile.clone(),
        print=load_profile(paper_name),
    )
    params.film = replace_fitted_density_curves(params.film)
    params.camera.auto_exposure = False
    params.enlarger.normalize_print_exposure = False
    params.enlarger.print_exposure_compensation = True
    params.settings.rgb_to_raw_method = 'hanatos2025'
    params.settings.neutral_print_filters_from_database = False

    film_density_cmy = getattr(film_profile.info, 'fitted_cmy_midscale_neutral_density', None)
    if film_density_cmy is not None:
        film_density_cmy = np.asarray(film_density_cmy, dtype=np.float64).reshape((3,))

    neutral_fit = fit_neutral_filters(
        params,
        normalize_print_exposure=False,
    )
    params.enlarger.c_filter_neutral = float(neutral_fit.c_filter)
    params.enlarger.m_filter_neutral = float(neutral_fit.m_filter)
    params.enlarger.y_filter_neutral = float(neutral_fit.y_filter)
    params.enlarger.print_exposure = float(neutral_fit.print_exposure)
    return params, neutral_fit


def build_positive_film_params(film_profile) -> RuntimePhotoParams:
    params = RuntimePhotoParams(
        film=film_profile.clone(),
        print=film_profile.clone(),
    )
    params.film = replace_fitted_density_curves(params.film)
    params.camera.auto_exposure = False
    params.enlarger.print_exposure_compensation = False
    params.enlarger.normalize_print_exposure = False
    params.io.scan_film = True
    params.settings.rgb_to_raw_method = 'hanatos2025'
    params.settings.neutral_print_filters_from_database = False
    return params


def render_gray_ramp(
    params: RuntimePhotoParams,
    ev_ramp: np.ndarray,
) -> np.ndarray:
    working_params = copy.deepcopy(params)
    working_params.io.input_color_space = 'sRGB'
    working_params.io.input_cctf_decoding = False
    working_params.io.output_color_space = 'sRGB'
    working_params.io.output_cctf_encoding = True
    working_params.debug.deactivate_spatial_effects = True
    working_params.debug.deactivate_stochastic_effects = True
    working_params.debug.debug_mode = 'off'
    working_params.debug.output_film_log_raw = False
    working_params.debug.output_film_density_cmy = False
    working_params.debug.output_print_density_cmy = False
    working_params.debug.inject_film_density_cmy = False
    working_params.settings.use_enlarger_lut = False
    working_params.settings.use_scanner_lut = False
    working_params.print_render.glare.active = False

    simulator = Simulator(digest_params(working_params))
    simulator.soft_update(print_exposure=float(working_params.enlarger.print_exposure))

    gray = np.zeros((np.size(ev_ramp), 3), dtype=np.float64)
    for index, exposure_compensation_ev in enumerate(ev_ramp):
        simulator.soft_update(exposure_compensation_ev=float(exposure_compensation_ev))
        gray[index] = simulator.process(MIDGRAY_RGB).reshape((3,))
    return np.clip(gray, 0.0, 1.0)


def make_patch_strip(rgb_values: np.ndarray, *, patch_height: int = 42, patch_width: int = 72) -> np.ndarray:
    patches = np.asarray(rgb_values, dtype=np.float64)[None, :, :]
    strip = np.repeat(patches, patch_height, axis=0)
    strip = np.repeat(strip, patch_width, axis=1)
    return np.clip(strip, 0.0, 1.0)


def mean_abs_neutrality_error(rgb_values: np.ndarray) -> np.ndarray:
    ramp = np.asarray(rgb_values, dtype=np.float64)
    return np.mean(np.abs(ramp - np.mean(ramp, axis=1, keepdims=True)), axis=1)


def mean_abs_midgray_error(rgb_values: np.ndarray) -> float:
    ramp = np.asarray(rgb_values, dtype=np.float64)
    return float(np.mean(np.abs(ramp - MIDGRAY_RGB_VECTOR.reshape((1, 3)))))


def plot_gray_ramps(
    *,
    film_name: str,
    paper_name: str | None,
    ev_ramp: np.ndarray,
    before_refinement: np.ndarray,
    refined: np.ndarray,
    before_fit=None,
    refined_fit=None,
    scan_film: bool = False,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        constrained_layout=True,
        height_ratios=(1.0, 2.2),
    )

    strip_before = make_patch_strip(before_refinement)
    strip_refined = make_patch_strip(refined)
    patch_centers = np.arange(len(ev_ramp)) * 72 + 35.5

    axes[0, 0].imshow(strip_before)
    if scan_film:
        axes[0, 0].set_title('Before refinement\nscan film')
    else:
        axes[0, 0].set_title(
            'Before refinement\n'
            f'C/M/Y={before_fit.c_filter:.1f}/{before_fit.m_filter:.1f}/{before_fit.y_filter:.1f} '
            f'| exp={before_fit.print_exposure:.4f}'
        )
    axes[0, 0].set_xticks(patch_centers, [f'{value:+.1f}' for value in ev_ramp])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xlabel('Camera exposure compensation [EV]')

    axes[0, 1].imshow(strip_refined)
    if scan_film:
        axes[0, 1].set_title('After refinement\nscan film')
    else:
        axes[0, 1].set_title(
            'After refinement\n'
            f'C/M/Y={refined_fit.c_filter:.1f}/{refined_fit.m_filter:.1f}/{refined_fit.y_filter:.1f} '
            f'| exp={refined_fit.print_exposure:.4f}'
        )
    axes[0, 1].set_xticks(patch_centers, [f'{value:+.1f}' for value in ev_ramp])
    axes[0, 1].set_yticks([])
    axes[0, 1].set_xlabel('Camera exposure compensation [EV]')

    color_labels = ('R', 'G', 'B')
    color_values = ('tab:red', 'tab:green', 'tab:blue')
    for channel_index, (label, color) in enumerate(zip(color_labels, color_values, strict=True)):
        axes[1, 0].plot(
            ev_ramp,
            refined[:, channel_index],
            color=color,
            linewidth=2.0,
            label=f'{label} refined',
        )
        axes[1, 0].plot(
            ev_ramp,
            before_refinement[:, channel_index],
            color=color,
            linewidth=1.2,
            linestyle='--',
            alpha=0.45,
            label=f'{label} before refinement',
        )
    axes[1, 0].set_title('Per-channel output')
    axes[1, 0].set_xlabel('Camera exposure compensation [EV]')
    axes[1, 0].set_ylabel('Output RGB')
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(ncol=2, fontsize=8)

    neutrality_before = mean_abs_neutrality_error(before_refinement)
    neutrality_refined = mean_abs_neutrality_error(refined)
    axes[1, 1].plot(ev_ramp, neutrality_refined, color='black', linewidth=2.0, label='Refined')
    axes[1, 1].plot(
        ev_ramp,
        neutrality_before,
        color='gray',
        linewidth=1.5,
        linestyle='--',
        label='Before refinement',
    )
    axes[1, 1].set_title('Neutrality error')
    axes[1, 1].set_xlabel('Camera exposure compensation [EV]')
    axes[1, 1].set_ylabel('Mean absolute channel deviation')
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.suptitle(
        (
            f'{film_name} | scan film direct'
            if scan_film
            else f'{film_name} -> {paper_name} | each side uses its own neutral-filter fit'
        ),
        fontsize=12,
    )


def main() -> None:
    args = parse_args()
    ev_ramp = np.asarray(args.ev, dtype=np.float64)
    if ev_ramp.ndim != 1 or ev_ramp.size == 0:
        raise ValueError('The EV ramp must contain at least one value.')

    raw_profile, source_profile, refined_profile = process_film_profiles(args.film)

    if refined_profile.info.type == 'negative':
        paper_name = infer_target_paper(raw_profile, refined_profile, args.paper)
        before_params, before_fit = build_negative_film_params(source_profile, paper_name)
        refined_params, refined_fit = build_negative_film_params(refined_profile, paper_name)
        scan_film = False
    else:
        paper_name = None
        before_params = build_positive_film_params(source_profile)
        refined_params = build_positive_film_params(refined_profile)
        before_fit = None
        refined_fit = None
        scan_film = True

    before_refinement = render_gray_ramp(before_params, ev_ramp)
    refined = render_gray_ramp(refined_params, ev_ramp)

    print(f'film={args.film}')
    print(f'film_type={refined_profile.info.type}')
    print('profile_source=before_and_after_refinement')
    if paper_name is not None:
        print(f'paper={paper_name}')
    print('output_color_space=sRGB')
    print('output_cctf_encoding=True')
    print(f'scan_film={scan_film}')
    if not scan_film:
        print('print_exposure_compensation=True')
    print(f'ev_ramp={ev_ramp.tolist()}')
    if not scan_film:
        print(
            'before_refinement_neutral_filters_cmy='
            f'[{before_fit.c_filter:.4f}, {before_fit.m_filter:.4f}, {before_fit.y_filter:.4f}]'
        )
        print(
            'refined_neutral_filters_cmy='
            f'[{refined_fit.c_filter:.4f}, {refined_fit.m_filter:.4f}, {refined_fit.y_filter:.4f}]'
        )
        print(f'before_refinement_print_exposure={before_fit.print_exposure:.6f}')
        print(f'refined_print_exposure={refined_fit.print_exposure:.6f}')
    print(f'before_refinement_midgray_mae={mean_abs_midgray_error(before_refinement):.6f}')
    print(f'refined_midgray_mae={mean_abs_midgray_error(refined):.6f}')
    print(
        'before_refinement_neutrality_mae='
        f'{float(np.mean(mean_abs_neutrality_error(before_refinement))):.6f}'
    )
    print(
        'refined_neutrality_mae='
        f'{float(np.mean(mean_abs_neutrality_error(refined))):.6f}'
    )

    plot_gray_ramps(
        film_name=args.film,
        paper_name=paper_name,
        ev_ramp=ev_ramp,
        before_refinement=before_refinement,
        refined=refined,
        before_fit=before_fit,
        refined_fit=refined_fit,
        scan_film=scan_film,
    )

    if args.save:
        plt.savefig(args.save, dpi=160)
    if plt.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':
    main()