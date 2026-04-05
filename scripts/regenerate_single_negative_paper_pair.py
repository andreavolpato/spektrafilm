from spektrafilm.profiles.io import save_profile
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.utils.io import save_neutral_print_filters
from spektrafilm_profile_creator import (
    NeutralPrintFilterRegenerationConfig,
    fit_neutral_print_filter_entry,
    process_profile,
)


FILM_STOCK = 'kodak_portra_400'
PRINT_PAPER = 'kodak_supra_endura' 
ILLUMINANT = Illuminants.lamp.value


def _process_target_profiles() -> None:
    # Process the print first so the negative-film refinement reads the updated paper profile.
    for stock in (PRINT_PAPER, FILM_STOCK):
        profile = process_profile(stock)
        save_profile(profile)
        print(f'Processed profile: {stock}')


def main() -> None:
    _process_target_profiles()

    result = fit_neutral_print_filter_entry(
        stock=FILM_STOCK,
        paper=PRINT_PAPER,
        illuminant=ILLUMINANT,
        config=NeutralPrintFilterRegenerationConfig(),
    )
    save_neutral_print_filters(result.filters)

    fitted_filters = result.filters[PRINT_PAPER][ILLUMINANT][FILM_STOCK]
    fitted_residue = result.residues[PRINT_PAPER][ILLUMINANT][FILM_STOCK]
    print(
        f'Updated neutral print filters for {PRINT_PAPER} / {ILLUMINANT} / {FILM_STOCK}: '
        f'C={fitted_filters[0]:.6f}, M={fitted_filters[1]:.6f}, Y={fitted_filters[2]:.6f}, '
        f'residue={fitted_residue:.6e}'
    )


if __name__ == '__main__':
    main()