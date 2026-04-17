import matplotlib.pyplot as plt

from spektrafilm import init_params, simulate
from spektrafilm.profiles.io import save_profile
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.utils.io import load_image_oiio
from spektrafilm_profile_creator import (
    process_profile,
    regenerate_neutral_filter_entry,
)


FILM_STOCK = 'kodak_gold_200'
PRINT_PAPER = 'kodak_portra_endura' 
ILLUMINANT = Illuminants.lamp.value
REFERENCE_IMAGE = 'img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif'


def _process_target_profiles() -> None:
    # Process the print first so the negative-film refinement reads the updated paper profile.
    for stock in (PRINT_PAPER, FILM_STOCK):
        profile = process_profile(stock)
        save_profile(profile)
        print(f'Processed profile: {stock}')


def _plot_reference_simulation() -> None:
    image = load_image_oiio(REFERENCE_IMAGE)
    params = init_params(film_profile=FILM_STOCK, print_profile=PRINT_PAPER)
    params.film_render.grain.sublayers_active = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.camera.exposure_compensation_ev = 0
    params.enlarger.print_exposure = 1.0
    params.camera.film_format_mm = 35
    params.print_render.glare.active = True

    print_scan = simulate(image, params)
    params.io.scan_film = True
    negative_scan = simulate(image, params)

    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(negative_scan)
    axes[0].axis('off')
    axes[0].set_title(f'{FILM_STOCK} negative')
    axes[1].imshow(print_scan)
    axes[1].axis('off')
    axes[1].set_title(f'{FILM_STOCK} on {PRINT_PAPER}')
    plt.tight_layout()
    plt.show()


def main() -> None:
    _process_target_profiles()

    fitted_filters, fitted_residue = regenerate_neutral_filter_entry(
        stock=FILM_STOCK,
        paper=PRINT_PAPER,
        illuminant=ILLUMINANT,
    )
    print(
        f'Updated neutral print filters for {PRINT_PAPER} / {ILLUMINANT} / {FILM_STOCK}: '
        f'C={fitted_filters[0]:.6f}, M={fitted_filters[1]:.6f}, Y={fitted_filters[2]:.6f}, '
        f'residue={fitted_residue:.6e}'
    )
    _plot_reference_simulation()


if __name__ == '__main__':
    main()