import matplotlib.pyplot as plt
from spektrafilm_profile_creator.plotting import plot_profile
from spektrafilm.profiles.io import save_profile
from spektrafilm_profile_creator import (
    load_raw_profile,
    load_stock_catalog,
    process_raw_profile,
)
from spektrafilm_profile_creator import regenerate_neutral_filter_database



PRINT_DATA_PACKAGE_PREFIX = 'spektrafilm_profile_creator.data.print.'


def _is_print_stock(data_package: str) -> bool:
    return data_package.startswith(PRINT_DATA_PACKAGE_PREFIX)


def _collect_raw_profiles():
    grouped_raw_profiles = {
        'print_paper': [],
        'print_film': [],
        'negative_film': [],
        'positive_film': [],
    }

    for stock, data_package in load_stock_catalog().items():
        raw_profile = load_raw_profile(stock)
        if not raw_profile.recipe.should_process:
            continue

        if _is_print_stock(data_package):
            if raw_profile.info.support == 'paper':
                grouped_raw_profiles['print_paper'].append(raw_profile)
            elif raw_profile.info.support == 'film':
                grouped_raw_profiles['print_film'].append(raw_profile)
            continue

        if raw_profile.info.support == 'film' and raw_profile.info.type == 'negative':
            grouped_raw_profiles['negative_film'].append(raw_profile)
        elif raw_profile.info.support == 'film' and raw_profile.info.type == 'positive':
            grouped_raw_profiles['positive_film'].append(raw_profile)

    return grouped_raw_profiles


def _process_profiles(raw_profiles):
    for raw_profile in raw_profiles:
        profile = process_raw_profile(raw_profile)
        save_profile(profile)
        plot_profile(profile)


process_print_paper = True
process_print_film = True
process_negative = True
process_positive = True
fit_neutral_filters = True

grouped_raw_profiles = _collect_raw_profiles()

print('----------------------------------------')
print('Print paper profiles')
if process_print_paper:
    _process_profiles(grouped_raw_profiles['print_paper'])


print('----------------------------------------')
print('Print film profiles')
if process_print_film:
    _process_profiles(grouped_raw_profiles['print_film'])


print('----------------------------------------')
print('Negative profiles')
if process_negative:
    _process_profiles(grouped_raw_profiles['negative_film'])


print('----------------------------------------')
print('Positive profiles')
if process_positive:
    _process_profiles(grouped_raw_profiles['positive_film'])


if fit_neutral_filters:
    regenerate_neutral_filter_database()

plt.show()
