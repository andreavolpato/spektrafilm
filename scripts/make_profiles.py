import matplotlib.pyplot as plt
from spektrafilm_profile_creator.plotting import plot_profile
from spektrafilm.profiles.io import save_profile
from spektrafilm_profile_creator import (
    load_raw_profile,
    load_stock_catalog,
    process_raw_profile,
)

process_print_paper = False
process_negative = False
process_positive = True

raw_profile_stocks = load_stock_catalog()
paper_raw_profiles = []
negative_film_raw_profiles = []
positive_film_raw_profiles = []
for stock in raw_profile_stocks:
    raw_profile = load_raw_profile(stock)
    if not raw_profile.recipe.should_process:
        continue
    if raw_profile.info.support == 'paper':
        paper_raw_profiles.append(raw_profile)
    elif raw_profile.info.support == 'film' and raw_profile.info.type == 'negative':
        negative_film_raw_profiles.append(raw_profile)
    elif raw_profile.info.support == 'film' and raw_profile.info.type == 'positive':
        positive_film_raw_profiles.append(raw_profile)

print('----------------------------------------')
print('Paper profiles')

if process_print_paper:
    for raw_profile in paper_raw_profiles:
        profile = process_raw_profile(raw_profile)
        save_profile(profile)
        plot_profile(profile)


print('----------------------------------------')
print('Negative profiles')

if process_negative:
    for raw_profile in negative_film_raw_profiles:
        profile = process_raw_profile(raw_profile)
        save_profile(profile)
        plot_profile(profile)


print('----------------------------------------')
print('Positive profiles')

if process_positive:
    for raw_profile in positive_film_raw_profiles:
        profile = process_raw_profile(raw_profile)
        save_profile(profile)
        plot_profile(profile)

plt.show()
