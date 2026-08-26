from __future__ import annotations

from spektrafilm_lut_creator.bundles import BundleSpec

DEFAULT_FILM_PROFILE = "kodak_portra_400"
DEFAULT_PRINT_PROFILES = ("kodak_portra_endura",)
DEFAULT_INPUT_COLOR_SPACE = "ACEScct"
DEFAULT_OUTPUT_COLOR_SPACE = "sRGB"
DEFAULT_TOPOLOGY = "1lut"
DEFAULT_RESOLUTION = 5


def make_bundle_spec(**overrides) -> BundleSpec:
    spec_kwargs = dict(
        name="test_bundle",
        film_profile=DEFAULT_FILM_PROFILE,
        print_profiles=DEFAULT_PRINT_PROFILES,
        input_color_space=DEFAULT_INPUT_COLOR_SPACE,
        output_color_space=DEFAULT_OUTPUT_COLOR_SPACE,
        topology=DEFAULT_TOPOLOGY,
        resolution=DEFAULT_RESOLUTION,
    )
    spec_kwargs.update(overrides)
    return BundleSpec(**spec_kwargs)
