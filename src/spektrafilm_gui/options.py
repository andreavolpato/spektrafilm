from __future__ import annotations

from enum import Enum


class RGBColorSpaces(Enum):
    sRGB = "sRGB"
    DCI_P3 = "DCI-P3"
    DisplayP3 = "Display P3"
    AdobeRGB = "Adobe RGB (1998)"
    ITU_R_BT2020 = "ITU-R BT.2020"
    ProPhotoRGB = "ProPhoto RGB"
    ACES2065_1 = "ACES2065-1"


class RGBtoRAWMethod(Enum):
    hanatos2025 = "hanatos2025"
    mallett2019 = "mallett2019"


class RawWhiteBalance(Enum):
    as_shot = "as_shot"
    daylight = "daylight"
    tungsten = "tungsten"
    custom = "custom"


class AutoExposureMethods(Enum):
    median = "median"
    center_weighted = "center_weighted"


class NapariInterpolationModes(Enum):
    nearest = "nearest"
    linear = "linear"
    cubic = "cubic"
    spline16 = "spline16"
    spline36 = "spline36"
    lanczos = "lanczos"
    blackman = "blackman"


class DiffusionFilterFamilies(Enum):
    glimmerglass = "glimmerglass"
    black_pro_mist = "black_pro_mist"
    pro_mist = "pro_mist"
    cinebloom = "cinebloom"