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
    DaVinciWideGamut = "DaVinci Wide Gamut"
    V_Gamut = "V-Gamut"


class WorkflowRoutes(Enum):
    input_passthrough = "input"
    input_film_scan = "input > film > scan"
    input_film_print_scan = "input > film > print > scan"
    input_convert_film_print_scan = "input > convert-film > print > scan"
    input_convert_film_scan_minus_base = "input > convert-film > scan-minus-base"
    input_convert_film_scan = "input > convert-film > scan"


class ScanIlluminants(Enum):
    # Curated capture-rig light sources for the convert (scanned-negative ->
    # film density) workflow. Values are names standard_illuminant() resolves.
    D50 = "D50"
    D55 = "D55"
    D65 = "D65"
    A = "A"
    BB3200 = "BB3200"
    BB5000 = "BB5000"
    LED_B3 = "LED-B3"
    LED_B5 = "LED-B5"
    LED_RGB1 = "LED-RGB1"
    LED_V2 = "LED-V2"
    FL2 = "FL2"


class RGBtoRAWMethod(Enum):
    arctic2026beta04 = "arctic2026beta04" # 2026-06-21
    hanatos2025 = "hanatos2025"
    jakob2019 = "jakob2019"
    otsu2018 = "otsu2018"
    mallett2019 = "mallett2019"
    gauss_lasers = "gauss-lasers" # 2026-06-19


class RawWhiteBalance(Enum):
    as_shot = "as_shot"
    daylight = "daylight"
    tungsten = "tungsten"
    custom = "custom"


class AutoExposureMethods(Enum):
    center_weighted = "center_weighted"
    matrix = "matrix"
    multi_zone = "multi_zone"
    partial = "partial"
    highlight_weighted = "highlight_weighted"
    median = "median"
    average = "average"


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


class InputGamutCompressAlgorithms(Enum):
    xy = "xy"
    oklch = "oklch"


class OutputGamutCompressAlgorithms(Enum):
    off = "off"
    oklch = "oklch"
    aces_rgc = "aces_rgc"
    oklrab = "oklrab"
    jzazbz = "jzazbz"
    cam16ucs = "cam16ucs"


