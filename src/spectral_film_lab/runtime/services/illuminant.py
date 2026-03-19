from __future__ import annotations

from spectral_film_lab.config import ENLARGER_STEPS
from spectral_film_lab.model.color_filters import color_enlarger


class EnlargerIlluminant:
    def __init__(self, enlarger_params):
        self._enlarger = enlarger_params

    def print_illuminant(self, light_source):
        y_filter = self._enlarger.y_filter_neutral * ENLARGER_STEPS + self._enlarger.y_filter_shift
        m_filter = self._enlarger.m_filter_neutral * ENLARGER_STEPS + self._enlarger.m_filter_shift
        c_filter = self._enlarger.c_filter_neutral * ENLARGER_STEPS
        return color_enlarger(light_source, y_filter, m_filter, c_filter)

    def preflash_illuminant(self, light_source):
        y_filter = self._enlarger.y_filter_neutral * ENLARGER_STEPS + self._enlarger.preflash_y_filter_shift
        m_filter = self._enlarger.m_filter_neutral * ENLARGER_STEPS + self._enlarger.preflash_m_filter_shift
        c_filter = self._enlarger.c_filter_neutral * ENLARGER_STEPS
        return color_enlarger(light_source, y_filter, m_filter, c_filter)
