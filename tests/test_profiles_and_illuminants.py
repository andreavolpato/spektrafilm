import numpy as np
from agx_emulsion.profiles.io import load_profile


class TestLoadProfile:
    def test_profile_has_required_fields(self, portra_400_profile):
        p = portra_400_profile
        assert hasattr(p, 'info')
        assert hasattr(p, 'data')
        assert hasattr(p.data, 'log_sensitivity')
        assert hasattr(p.data, 'density_curves')
        assert hasattr(p.data, 'dye_density')
        assert hasattr(p.data, 'log_exposure')
        assert hasattr(p.data, 'wavelengths')

    def test_density_curves_have_three_channels(self, portra_400_profile):
        dc = portra_400_profile.data.density_curves
        assert dc.ndim == 2
        assert dc.shape[1] == 3
