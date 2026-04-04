from __future__ import annotations

import numpy as np

from spektrafilm.model.color_filters import color_enlarger


class EnlargerService:
    def __init__(self, enlarger_params):
        self.density_spectral_midgray = None
        self.print_exposure_compensation = enlarger_params.print_exposure_compensation
        self._enlarger = enlarger_params

    def enlarger_filtered_illuminant(self, light_source):
        c_filter = self._enlarger.c_filter_neutral
        m_filter = self._enlarger.m_filter_neutral + self._enlarger.m_filter_shift
        y_filter = self._enlarger.y_filter_neutral + self._enlarger.y_filter_shift
        filter_cc_values = np.array([c_filter, m_filter, y_filter])
        return color_enlarger(light_source, filter_cc_values)

    def preflash_filtered_illuminant(self, light_source):
        c_filter = self._enlarger.c_filter_neutral
        m_filter = self._enlarger.m_filter_neutral + self._enlarger.preflash_m_filter_shift
        y_filter = self._enlarger.y_filter_neutral + self._enlarger.preflash_y_filter_shift
        filter_cc_values = np.array([c_filter, m_filter, y_filter])
        return color_enlarger(light_source, filter_cc_values)
