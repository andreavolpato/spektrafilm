from spectral_film_lab.utils.fast_stats import warmup_fast_stats
from spectral_film_lab.utils.lut import warmup_luts
from spectral_film_lab.utils.fast_interp import warmup_fast_interp
# from spectral_film_lab.utils.fast_gaussian_filter import warmup_fast_gaussian_filter

# precompile numba functions
def warmup():
    warmup_fast_stats()
    warmup_luts()
    warmup_fast_interp()
    # warmup_fast_gaussian_filter()

