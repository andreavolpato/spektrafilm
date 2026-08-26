"""Curated color-space registry for the LUT exporter.

This registry is the single source of truth for color-space names
accepted anywhere in :mod:`spektrafilm_lut_creator`. Every entry is
gatekept: only names registered here are valid. Adding a space is a
one-line :func:`register` call.

Backend is `colour-science`. The :func:`to_xyz` / :func:`from_xyz` /
:func:`decode_cctf` / :func:`encode_cctf` helpers wrap the backend so
callers never need to import it directly.

See studies/a40_lut_system/n040_color_space_registry.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import colour
import numpy as np

VERSION = "0.1.0"

# Allowed kinds: linear (scene-referred), encoded_sdr (gamma/CCTF), log (camera log).
_ALLOWED_KINDS = frozenset({"linear", "encoded_sdr", "log"})
_ALLOWED_ROLES = frozenset({"input", "output"})

_BT2100_PQ_CCTF = "ITU-R BT.2100 PQ"
_BT2100_HLG_CCTF = "ITU-R BT.2100 HLG"


@dataclass(frozen=True)
class ColorSpaceEntry:
    """One curated entry in the color-space registry.

    ``primaries`` is the ``colour.RGB_COLOURSPACES`` key supplying the
    RGB gamut + white point. ``cctf`` is the ``colour.CCTF_ENCODINGS``
    key supplying the curve, or ``None`` for scene-linear spaces. ``kind``
    is the grid-sampling distribution category (see n040 §5). ``role``
    declares which sides of the bundle this space may appear on.
    ``short_tag`` is the camera-safe filename identifier (lowercase
    alphanumeric, no brand names where avoidable) used when composing
    LUT filenames.
    """

    name: str
    primaries: str
    cctf: str | None
    kind: str
    role: tuple[str, ...]
    short_tag: str = ""
    notes: str = ""
    ocio_alias: str = ""
    """ACES Studio Config-style long-name alias emitted alongside the
    short ``name`` in OCIO configs. Empty string means no alias.
    See studies/a40_lut_system/n120_ocio_config_emission.md §7."""
    midgray_linear: float = 0.18
    """Linear value (after :func:`decode_cctf`) that represents
    midgray for the purposes of LUT-input centring. ``0.18`` for
    SDR-encoded and camera-log spaces (midgray = 18% reflectance by
    design — colour-science's CCTFs decode encoded 1.0 to linear 1.0
    for these). HDR-encoded spaces (PQ / HLG) override this because
    colour-science's BT.2100 EOTF returns absolute luminance in nits.

    For Rec.2100 PQ and P3-D65 PQ this is ``100.0`` (SDR reference
    white). PQ has no canonical midgray — HDR10's 18-nit and Dolby
    Vision's 10-nit conventions would put midgray at PQ-encoded
    ~0.215 and ~0.19 respectively, which wastes 75-80% of the LUT
    cube on highlights the film just clips into its shoulder. 100
    nits puts midgray near PQ-encoded 0.508, so the film's useful
    range (~7 stops above midgray) maps to roughly the upper half
    of the input cube and the LUT samples sensibly across the
    encoded domain.

    For Rec.2100 HLG this is ``26.238`` — BT.2408's 18% grey card
    (HLG signal 38%) evaluated through the BT.2100 EOTF on the
    1000-nit reference display. That anchors midgray at HLG-encoded
    0.38, in line with the camera-log entries (0.33–0.46).

    Consumed by :func:`input_midgray_gain` and
    :func:`output_midgray_gain`."""
    scene_referred_input: bool = False
    """Mark a ``kind="linear"`` entry as a scene-referred working space
    (ACEScg, ACES2065-1, OpenEXR-style scene-linear, …) rather than
    display-linear. Only legal on ``kind="linear"`` entries; ``register``
    rejects it on encoded/log entries (where headroom is already baked
    into the curve).

    Scene-referred linear entries are silenced as bundle inputs: a
    uniform [0, 1] cube domain clips their highlights at diffuse white,
    and giving them headroom through a plain linear gain would shift
    midgray — the LUT layer pins midgray by construction and refuses
    that trade. They become inputs when the log-anchored input shaper
    lands (see n150)."""


_REGISTRY: dict[str, ColorSpaceEntry] = {}


def register(entry: ColorSpaceEntry) -> None:
    """Validate and add ``entry`` to the registry."""
    if entry.kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"{entry.name!r}: kind must be one of {sorted(_ALLOWED_KINDS)}, "
            f"got {entry.kind!r}"
        )
    for role in entry.role:
        if role not in _ALLOWED_ROLES:
            raise ValueError(
                f"{entry.name!r}: role entries must be in {sorted(_ALLOWED_ROLES)}, "
                f"got {role!r}"
            )
    if entry.primaries not in colour.RGB_COLOURSPACES:
        raise KeyError(
            f"{entry.name!r}: primaries {entry.primaries!r} not in colour.RGB_COLOURSPACES"
        )
    if entry.cctf is not None and entry.cctf not in colour.CCTF_ENCODINGS:
        raise KeyError(
            f"{entry.name!r}: cctf {entry.cctf!r} not in colour.CCTF_ENCODINGS"
        )
    if entry.kind == "linear" and entry.cctf is not None:
        raise ValueError(f"{entry.name!r}: linear spaces must have cctf=None")
    if entry.kind != "linear" and entry.cctf is None:
        raise ValueError(f"{entry.name!r}: kind={entry.kind!r} requires a cctf")
    if entry.scene_referred_input and entry.kind != "linear":
        raise ValueError(
            f"{entry.name!r}: scene_referred_input=True is only valid for "
            f"kind='linear' entries, got kind={entry.kind!r}"
        )
    if not entry.short_tag:
        raise ValueError(f"{entry.name!r}: short_tag must be non-empty")
    if not entry.short_tag.replace("_", "").isalnum() or not entry.short_tag.islower():
        raise ValueError(
            f"{entry.name!r}: short_tag {entry.short_tag!r} must be lowercase "
            "alphanumeric (underscores allowed)"
        )
    _REGISTRY[entry.name] = entry


def short_tag(name: str) -> str:
    """Return the camera-safe short tag for the registered color space."""
    return get(name).short_tag


def get(name: str) -> ColorSpaceEntry:
    """Return the registered entry for ``name`` or raise ``KeyError``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown color space {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def resolve(name_or_short_tag: str) -> str:
    """Return the canonical registry name for ``name_or_short_tag``.

    Accepts either a registered canonical name (e.g. ``"Rec.2100 PQ"``)
    or the entry's ``short_tag`` slug (e.g. ``"rec2100pq"``). Raises
    ``KeyError`` if neither matches.
    """
    if name_or_short_tag in _REGISTRY:
        return name_or_short_tag
    for entry in _REGISTRY.values():
        if entry.short_tag == name_or_short_tag:
            return entry.name
    raise KeyError(
        f"Unknown color space {name_or_short_tag!r}; registered: {sorted(_REGISTRY)}"
    )


def list_input_spaces() -> list[str]:
    """Names of spaces eligible as bundle input."""
    return sorted(name for name, entry in _REGISTRY.items() if "input" in entry.role)


def list_output_spaces() -> list[str]:
    """Names of spaces eligible as bundle output."""
    return sorted(name for name, entry in _REGISTRY.items() if "output" in entry.role)


def decode_cctf(rgb, name: str) -> np.ndarray:
    """Apply the inverse CCTF, producing linear-light RGB.

    For ``kind='linear'`` spaces this is a no-op (input returned as
    float ndarray).
    """
    entry = get(name)
    rgb = np.asarray(rgb, dtype=float)
    if entry.cctf is None:
        return rgb
    if entry.cctf == _BT2100_PQ_CCTF:
        return np.asarray(
            colour.models.eotf_BT2100_PQ(np.clip(rgb, 0.0, 1.0)), dtype=float
        )
    if entry.cctf == _BT2100_HLG_CCTF:
        return np.asarray(
            colour.models.eotf_BT2100_HLG(np.clip(rgb, 0.0, 1.0)), dtype=float
        )
    return np.asarray(colour.cctf_decoding(rgb, function=entry.cctf), dtype=float)


def encode_cctf(rgb, name: str) -> np.ndarray:
    """Apply the forward CCTF to linear-light RGB."""
    entry = get(name)
    rgb = np.asarray(rgb, dtype=float)
    if entry.cctf is None:
        return rgb
    if entry.cctf == _BT2100_PQ_CCTF:
        return np.asarray(
            colour.models.eotf_inverse_BT2100_PQ(np.clip(rgb, 0.0, None)),
            dtype=float,
        )
    if entry.cctf == _BT2100_HLG_CCTF:
        # colour-science's ARIB STD-B67 implementation evaluates both
        # branches of an internal np.where, so valid low-end samples can
        # emit a spurious invalid-log warning from the unused log branch.
        with np.errstate(invalid="ignore"):
            return np.asarray(
                colour.models.eotf_inverse_BT2100_HLG(np.clip(rgb, 0.0, None)),
                dtype=float,
            )
    return np.asarray(colour.cctf_encoding(rgb, function=entry.cctf), dtype=float)


def to_xyz(rgb, name: str) -> np.ndarray:
    """Convert RGB (in the named space, encoded as the space requires)
    to CIE XYZ tristimulus values."""
    entry = get(name)
    linear = decode_cctf(rgb, name)
    return np.asarray(
        colour.RGB_to_XYZ(
            linear, colourspace=entry.primaries, apply_cctf_decoding=False
        ),
        dtype=float,
    )


def from_xyz(xyz, name: str) -> np.ndarray:
    """Convert CIE XYZ to RGB in the named space (encoded as the space requires)."""
    entry = get(name)
    linear = np.asarray(
        colour.XYZ_to_RGB(
            np.asarray(xyz, dtype=float),
            colourspace=entry.primaries,
            apply_cctf_encoding=False,
        ),
        dtype=float,
    )
    return encode_cctf(linear, name)


# ---------------------------------------------------------------------------
# v1 entries.
#
# Skipped from v1 (documented for future addition):
#   - Nikon N-Log: colour-science ships the curve but not a Nikon gamut.
#     Will be added once a Nikon RGB colourspace is in upstream or modeled
#     manually here.
# ---------------------------------------------------------------------------

# Scene-linear working spaces. Registered but currently **silenced as bundle
# inputs** (``role=()``) — a uniform [0, 1] .cube domain can't represent
# scene-linear highlights past diffuse white without a log shaper applied
# upstream of the LUT lookup, which the bundle pipeline doesn't emit yet
# (see studies/a40_lut_system/n150_input_log_anchored_shaper.md). The
# entries stay in the registry so internal color-space math still
# resolves these names; ``list_input_spaces()`` hides them and
# ``BundleBuilder`` rejects them via the ``"input" in entry.role`` check.
# When n150 lands, restore ``role=("input",)``.
register(
    ColorSpaceEntry(
        "ACES2065-1",
        "ACES2065-1",
        None,
        "linear",
        (),
        short_tag="aces20651",
        ocio_alias="ACES - ACES2065-1",
        scene_referred_input=True,
        notes="ACES interchange (AP0 primaries). Silenced pending shaper support.",
    )
)
register(
    ColorSpaceEntry(
        "ACEScg",
        "ACEScg",
        None,
        "linear",
        (),
        short_tag="acescg",
        ocio_alias="ACES - ACEScg",
        scene_referred_input=True,
        notes="AP1 primaries; VFX rendering workhorse. Silenced pending shaper support.",
    )
)
register(
    ColorSpaceEntry(
        "Rec.709 Linear",
        "ITU-R BT.709",
        None,
        "linear",
        (),
        short_tag="rec709lin",
        scene_referred_input=True,
        notes="Silenced pending shaper support.",
    )
)
register(
    ColorSpaceEntry(
        "Rec.2020 Linear",
        "ITU-R BT.2020",
        None,
        "linear",
        (),
        short_tag="rec2020lin",
        scene_referred_input=True,
        notes="Silenced pending shaper support.",
    )
)
register(
    ColorSpaceEntry(
        "ProPhoto Linear",
        "ProPhoto RGB",
        None,
        "linear",
        (),
        short_tag="prophotolin",
        scene_referred_input=True,
        notes="Current spektrafilm default. Silenced pending shaper support.",
    )
)
register(
    ColorSpaceEntry(
        "sRGB Linear",
        "sRGB",
        None,
        "linear",
        (),
        short_tag="srgblin",
        scene_referred_input=True,
        notes="Same primaries as sRGB, no CCTF. Silenced pending shaper support.",
    )
)

# Encoded SDR (input and output).
register(
    ColorSpaceEntry(
        "sRGB",
        "sRGB",
        "sRGB",
        "encoded_sdr",
        ("input", "output"),
        short_tag="srgb",
        ocio_alias="sRGB - Display",
        notes="The web default.",
    )
)
register(
    ColorSpaceEntry(
        "Rec.709",
        "ITU-R BT.709",
        "ITU-R BT.1886",
        "encoded_sdr",
        ("input", "output"),
        short_tag="rec709",
        ocio_alias="Rec.1886 Rec.709 - Display",
        notes="Broadcast; BT.1886 EOTF.",
    )
)
register(
    ColorSpaceEntry(
        "Display P3",
        "Display P3",
        "sRGB",
        "encoded_sdr",
        ("input", "output"),
        short_tag="displayp3",
        ocio_alias="Display P3 - Display",
        notes="Apple devices.",
    )
)
register(
    ColorSpaceEntry(
        "Rec.2020",
        "ITU-R BT.2020",
        "ITU-R BT.1886",
        "encoded_sdr",
        ("input", "output"),
        short_tag="rec2020",
        ocio_alias="Rec.1886 Rec.2020 - Display",
        notes="Wide-gamut SDR.",
    )
)
register(
    ColorSpaceEntry(
        "DCI-P3",
        "DCI-P3",
        "Gamma 2.6",
        "encoded_sdr",
        ("input", "output"),
        short_tag="dcip3",
        ocio_alias="G2.6-P3-DCI - Display",
        notes="Theatrical (xenon white ~6300K, gamma 2.6).",
    )
)
register(
    ColorSpaceEntry(
        "Adobe RGB",
        "Adobe RGB (1998)",
        "Gamma 2.2",
        "encoded_sdr",
        ("input", "output"),
        short_tag="adobergb",
        notes="Standard inkjet/lab print delivery. Adobe's spec uses gamma 2.19921875; we approximate as 2.2 (error ~1e-4, below LUT precision).",
    )
)

# Camera log (input only).
register(
    ColorSpaceEntry(
        "ACEScct",
        "ACEScct",
        "ACEScct",
        "log",
        ("input",),
        short_tag="acescct",
        ocio_alias="ACEScct",
    )
)
register(
    ColorSpaceEntry(
        "ARRI LogC3 (EI800)",
        "ARRI Wide Gamut 3",
        "ARRI LogC3",
        "log",
        ("input",),
        short_tag="logc3",
        ocio_alias="ARRI LogC3 (EI800) - AWG3",
        notes="EI800 is colour-science's default LogC3 curve.",
    )
)
register(
    ColorSpaceEntry(
        "ARRI LogC4",
        "ARRI Wide Gamut 4",
        "ARRI LogC4",
        "log",
        ("input",),
        short_tag="logc4",
        ocio_alias="ARRI LogC4 - AWG4",
    )
)
register(
    ColorSpaceEntry(
        "Sony S-Log3",
        "S-Gamut3",
        "S-Log3",
        "log",
        ("input",),
        short_tag="slog3",
        ocio_alias="Sony S-Log3 - S-Gamut3",
    )
)
register(
    ColorSpaceEntry(
        "Sony S-Log3 (S-Gamut3.Cine)",
        "S-Gamut3.Cine",
        "S-Log3",
        "log",
        ("input",),
        short_tag="slog3cine",
        ocio_alias="Sony S-Log3 - S-Gamut3.Cine",
    )
)
register(
    ColorSpaceEntry(
        "Panasonic V-Log",
        "V-Gamut",
        "V-Log",
        "log",
        ("input",),
        short_tag="vlog",
        ocio_alias="Panasonic V-Log - V-Gamut",
    )
)
register(
    ColorSpaceEntry(
        "Fujifilm F-Log", "F-Gamut", "F-Log", "log", ("input",), short_tag="flog"
    )
)
register(
    ColorSpaceEntry(
        "Fujifilm F-Log2", "F-Gamut", "F-Log2", "log", ("input",), short_tag="flog2"
    )
)
register(
    ColorSpaceEntry(
        "Canon Log 3",
        "Cinema Gamut",
        "Canon Log 3",
        "log",
        ("input",),
        short_tag="canonlog3",
        ocio_alias="Canon Log 3 - Cinema Gamut",
    )
)
register(
    ColorSpaceEntry(
        "RED Log3G10",
        "REDWideGamutRGB",
        "Log3G10",
        "log",
        ("input",),
        short_tag="redlog3g10",
        ocio_alias="RED Log3G10 - REDWideGamutRGB",
    )
)
register(
    ColorSpaceEntry(
        "DaVinci Intermediate",
        "DaVinci Wide Gamut",
        "DaVinci Intermediate",
        "log",
        ("input",),
        short_tag="davinciintermediate",
    )
)
register(
    ColorSpaceEntry(
        "Apple Log",
        "ITU-R BT.2020",
        "Apple Log Profile",
        "log",
        ("input",),
        short_tag="applelog",
        ocio_alias="Apple Log",
        notes="iPhone 15+ Pro, iPhone 17 Pro, Vision Pro Immersive. Apple Log pairs with BT.2020 primaries per Apple's spec.",
    )
)
register(
    ColorSpaceEntry(
        "Blackmagic Film Gen 5",
        "Blackmagic Wide Gamut",
        "Blackmagic Film Generation 5",
        "log",
        ("input",),
        short_tag="bmfilmgen5",
        notes="Resolve-native. All current Blackmagic bodies (Pocket 4K/6K, URSA, Cine 12K, Pyxis).",
    )
)

# HDR — Rec.2100 family. Both PQ and HLG are valid in both input and output roles:
#  - Input: source camera HDR (e.g., a PQ-graded delivery being processed through spektrafilm)
#  - Output: final master deliverable (streaming HDR master = PQ; broadcast HDR = HLG)
# See studies/a40_lut_system/n060_color_space_v2_research.md for film-LUT-in-HDR gotchas
# (peak-nit assumption, never apply Rec.709 LUTs to PQ signal, etc.).
register(
    ColorSpaceEntry(
        "Rec.2100 PQ",
        "ITU-R BT.2020",
        "ITU-R BT.2100 PQ",
        "log",
        ("input", "output"),
        short_tag="rec2100pq",
        ocio_alias="Rec.2100-PQ - Display",
        midgray_linear=100.0,
        notes="HDR streaming master (Netflix/Apple TV+/Disney+/Amazon/Max). ST.2084 transfer. Domain [0,1] = 0..10,000 nits; document per-LUT peak-luminance assumption.",
    )
)
register(
    ColorSpaceEntry(
        "Rec.2100 HLG",
        "ITU-R BT.2020",
        "ITU-R BT.2100 HLG",
        "log",
        ("input", "output"),
        short_tag="rec2100hlg",
        ocio_alias="Rec.2100-HLG - Display",
        midgray_linear=26.238265737,
        notes="HDR broadcast (BBC/NHK/EBU live, YouTube HDR). HLG bakes OOTF + nominal peak luminance; not portable across display peaks the way PQ is. Midgray is BT.2408's 18% grey card: HLG signal 38% -> 26.24 nits on the 1000-nit reference display (= eotf_BT2100_HLG(0.38)).",
    )
)

# ---------------------------------------------------------------------------
# v2 Tier 2 additions (n060 §3).
#
# Each entry is registry-only — colour-science ships the underlying
# primaries + CCTF in upstream releases since n060 was written.
#
# Skipped (need manual curve implementation, deferred to a later note):
#   - DJI D-Log M (no curve in colour-science; vendor whitepaper required)
#   - ARRI LogC3 EI400 / EI1600 (only EI800 baseline in colour-science;
#     curve shifts by EI not currently exposed)
# ---------------------------------------------------------------------------

# ACEScc — pure-log sibling of ACEScct, legacy VFX working space.
register(
    ColorSpaceEntry(
        "ACEScc",
        "ACEScg",
        "ACEScc",
        "log",
        ("input",),
        short_tag="acescc",
        ocio_alias="ACEScc",
        notes="Legacy VFX log working space (AP1 primaries, ACES1.0-era). "
        "Same primaries as ACEScg; differs from ACEScct only in the "
        "removal of the small linear toe near zero.",
    )
)

# P3-D65 — the mastering-display container inside Dolby Vision / HDR10.
# Linear is a working space (input only); PQ-encoded is the actual delivery
# format used inside Rec.2020 HDR containers (input + output).
register(
    ColorSpaceEntry(
        "P3-D65 Linear",
        "P3-D65",
        None,
        "linear",
        (),
        short_tag="p3d65lin",
        scene_referred_input=True,
        notes="Linear P3-D65 working space. Used alongside the PQ-encoded "
        "variant for HDR pipelines that want a separate scene-linear stage. "
        "Silenced as bundle input pending shaper support.",
    )
)
register(
    ColorSpaceEntry(
        "P3-D65 PQ",
        "P3-D65",
        "ITU-R BT.2100 PQ",
        "log",
        ("input", "output"),
        short_tag="p3d65pq",
        ocio_alias="ST2084 P3-D65 - Display",
        midgray_linear=100.0,
        notes="The mastering-display container inside DoVi/HDR10. P3-D65 "
        "primaries with ST.2084 transfer; same curve as Rec.2100 PQ "
        "but a narrower color volume. Use when the deliverable spec "
        "calls for 'P3-limited inside Rec.2020' or when matching a "
        "1000-nit P3 mastering display directly.",
    )
)

# DJI D-Log — Ronin 4D / Inspire 3 / X9 sensor. colour-science added both
# the D-Gamut primaries and the D-Log curve since n060.
register(
    ColorSpaceEntry(
        "DJI D-Log",
        "DJI D-Gamut",
        "D-Log",
        "log",
        ("input",),
        short_tag="dlog",
        notes="DJI cinema cameras (Ronin 4D, Inspire 3, X9 sensor). "
        "D-Gamut is wider than P3, narrower than Rec.2020.",
    )
)

# ProPhoto RGB (encoded) — Lightroom histogram + ACR→Photoshop handoff,
# high-gamut stills delivery for printing. colour-science's CCTF
# implements the standard ProPhoto curve (linear toe + gamma 1.8 main
# segment) which is the Melissa-equivalent for everything outside the
# deepest shadows.
register(
    ColorSpaceEntry(
        "ProPhoto RGB",
        "ProPhoto RGB",
        "ProPhoto RGB",
        "encoded_sdr",
        ("input", "output"),
        short_tag="prophoto",
        notes="High-gamut stills working space + delivery target. Lightroom "
        "internal histogram and ACR→Photoshop interchange. D50 white "
        "point; chromatic adaptation handled by the XYZ conversion.",
    )
)

# Nikon N-Log paired with Rec.2020 — per the n060 camera-log research,
# this is the de-facto pipeline pairing in Resolve / Baselight (Nikon
# itself doesn't publish a distinct gamut, and BT.2020 is the conventional
# wrapper).
register(
    ColorSpaceEntry(
        "Nikon N-Log",
        "ITU-R BT.2020",
        "N-Log",
        "log",
        ("input",),
        short_tag="nlog",
        notes="Nikon Z 6/7/8/9 N-Log. No distinct Nikon gamut is published; "
        "ITU-R BT.2020 is the de-facto wrapper per Resolve/Baselight "
        "and the n060 camera-log research.",
    )
)


# ---------------------------------------------------------------------------
# Input gain — midgray-anchored (n200 revised).
#
# The bake applies one linear gain to post-decode input values, anchored
# on middle gray: the input's native midgray (``entry.midgray_linear``)
# always lands exactly on the film's 0.18. For SDR / camera-log / linear
# inputs (midgray_linear = 0.18) the gain is identity; for PQ / HLG
# (midgray in nits) it bridges the absolute-luminance scale down to
# reflectance. A deliberate re-exposure can be stacked on top via
# ``BundleSpec.exposure_ev`` (× 2**ev) — disclosed in the bundle, never
# implicit. White placement is whatever the curve natively encodes;
# repositioning white without moving midgray would require log shaping,
# which the LUT layer deliberately rules out (n150).
# ---------------------------------------------------------------------------


_MID_GRAY_LINEAR = 0.18


def encode_qa_input(
    linear_reflectance,
    input_color_space: str,
    exposure_ev: float = 0.0,
) -> np.ndarray:
    """Encode a reflectance-scale linear RGB stimulus into the input
    color space's encoded code values for QA, accounting for the LUT's
    input gain.

    QA stimulus generators (OkLab patches, neutral ramps, Planckian
    sweeps) produce linear RGB in 0..1 reflectance scale where 0.18 ≡
    midgray. For SDR/log inputs at ``exposure_ev=0`` this is identical
    to :func:`encode_cctf`. For HDR inputs (gain ≪ 1), this multiplies
    by ``1/gain`` first so the LUT's pre-pipeline scaling lands the
    stimulus at the intended reflectance in the film's frame —
    otherwise OkLab patches would all collapse into the deep shadow
    region of the input encoding.
    """
    gain = input_gain(input_color_space, exposure_ev)
    arr = np.asarray(linear_reflectance, dtype=float) / gain
    return encode_cctf(arr, input_color_space)


def to_xyz_qa(encoded_rgb, output_color_space: str) -> np.ndarray:
    """Decode a LUT output (encoded RGB) to reflectance-scale XYZ
    for QA analysis.

    Like :func:`to_xyz` but divides the post-CCTF linear values by
    :func:`output_midgray_gain` so HDR outputs end up on the same
    scale SDR outputs have always used (midgray ``Y ≈ 0.18``). Every
    QA convention — density curves, OkLab projections, sRGB display
    previews, ΔE comparisons — assumes this reflectance scale. For
    SDR outputs (gain=1) this is identical to :func:`to_xyz`.
    """
    entry = get(output_color_space)
    encoded_rgb = np.asarray(encoded_rgb, dtype=float)
    linear = decode_cctf(encoded_rgb, output_color_space)
    linear_reflectance = linear / output_midgray_gain(output_color_space)
    return np.asarray(
        colour.RGB_to_XYZ(
            linear_reflectance,
            colourspace=entry.primaries,
            apply_cctf_decoding=False,
        ),
        dtype=float,
    )


def effective_input_midgray_linear(
    input_color_space: str,
    exposure_ev: float = 0.0,
) -> float:
    """Linear value (in the input space's native scale) that the LUT
    treats as midgray after its input gain.

    Computed as ``0.18 / input_gain(input_cs, exposure_ev)``, which
    reduces to ``entry.midgray_linear / 2**exposure_ev``. At
    ``exposure_ev=0`` this is exactly the registry midgray (0.18 for
    SDR/log, the reference-white nits for PQ/HLG) — the value that
    lands on the film's 0.18 after the LUT's pre-pipeline scaling.
    """
    gain = input_gain(input_color_space, exposure_ev)
    return _MID_GRAY_LINEAR / gain


def output_midgray_gain(output_color_space: str) -> float:
    """Linear gain to apply before :func:`encode_cctf` so the film's
    reflectance-scale 0.18 lands at the output space's midgray.

    For SDR / camera-log outputs whose CCTFs are designed in reflectance
    scale this returns ``1.0`` (identity — the film's 0.18 encodes
    correctly). For HDR-encoded outputs (PQ / HLG) whose CCTFs return
    absolute luminance in nits, this returns
    ``entry.midgray_linear / 0.18`` so the film's midgray maps to the
    convention's midgray-nits value before encoding — e.g., ≈555.5 for
    Rec.2100 PQ under the 100-nit (SDR-ref) midgray convention, putting
    midgray at PQ-encoded ≈0.508.

    Without this gain, encoding a film output (which lives in
    reflectance scale) through a PQ EOTF inverse would treat ``0.18``
    as ``0.18 nits`` (basically black) — the LUT comes out almost
    entirely dark.
    """
    entry = get(output_color_space)
    return float(entry.midgray_linear / _MID_GRAY_LINEAR)


def input_midgray_gain(input_color_space: str) -> float:
    """Linear gain to apply after :func:`decode_cctf` so the input's
    native midgray lands on the film's reflectance-scale 0.18.

    Mirror of :func:`output_midgray_gain`: ``0.18 / entry.midgray_linear``.
    For SDR / camera-log / linear inputs whose CCTFs are designed in
    reflectance scale this is ``1.0`` (identity — a properly-exposed
    18% gray decodes to 0.18 and passes straight through). For
    HDR-encoded inputs (PQ / HLG) whose CCTFs return absolute luminance
    in nits, this bridges the scale so the convention's midgray-nits
    value (100 for PQ) lands on the film's 0.18.

    Midgray is pinned by construction: this gain never re-exposes the
    image. Deliberate re-exposure is a separate, disclosed knob —
    ``BundleSpec.exposure_ev``, composed in :func:`input_gain`.
    """
    entry = get(input_color_space)
    return float(_MID_GRAY_LINEAR / entry.midgray_linear)


def input_gain(input_color_space: str, exposure_ev: float = 0.0) -> float:
    """Total linear gain applied to post-decode input values:
    ``input_midgray_gain(input_cs) * 2**exposure_ev``.

    At ``exposure_ev=0`` (the default) midgray maps exactly to the
    film's 0.18 and the LUT is colorimetrically neutral in exposure.
    A non-zero ``exposure_ev`` is the LUT-level equivalent of the
    camera's exposure-compensation dial: +1.0 brightens the source one
    stop in the film's frame, midgray included.
    """
    return input_midgray_gain(input_color_space) * float(2.0 ** float(exposure_ev))


@dataclass(frozen=True)
class BakeFrame:
    """Bundle's gain context for a bake or QA run.

    Computed once via :meth:`from_spec` and passed through QA call
    sites instead of threading loose ``exposure_ev`` everywhere.
    Hides the scale-bridging math behind :meth:`encode_input` and
    :meth:`decode_output_to_xyz` so viz/test code never has to remember
    whether to use :func:`encode_cctf` vs :func:`encode_qa_input`, or
    :func:`to_xyz` vs :func:`to_xyz_qa`.

    For SDR specs at ``exposure_ev=0`` both gains collapse to ``1.0``
    and the methods are identical to their non-QA counterparts.
    """

    input_color_space: str
    output_color_space: str
    exposure_ev: float
    input_gain: float
    output_gain: float
    input_midgray_linear: float
    """Effective linear midgray in the input's native scale
    (``0.18 / input_gain`` = ``entry.midgray_linear / 2**exposure_ev``).
    At ``exposure_ev=0`` this is the registry's
    :attr:`ColorSpaceEntry.midgray_linear`."""

    @classmethod
    def from_spec(cls, spec) -> BakeFrame:
        """Build the frame from a :class:`BundleSpec`-shaped object
        (anything exposing ``input_color_space`` / ``output_color_space``
        / ``exposure_ev``)."""
        in_gain = input_gain(spec.input_color_space, spec.exposure_ev)
        out_gain = output_midgray_gain(spec.output_color_space)
        return cls(
            input_color_space=spec.input_color_space,
            output_color_space=spec.output_color_space,
            exposure_ev=float(spec.exposure_ev),
            input_gain=float(in_gain),
            output_gain=float(out_gain),
            input_midgray_linear=_MID_GRAY_LINEAR / float(in_gain),
        )

    def encode_input(self, linear_reflectance) -> np.ndarray:
        """Reflectance-scale linear RGB → input-encoded code values.

        Equivalent to :func:`encode_qa_input` but with the gain held
        on ``self`` so callers don't pass ``exposure_ev`` around.
        """
        arr = np.asarray(linear_reflectance, dtype=float) / self.input_gain
        return encode_cctf(arr, self.input_color_space)

    def decode_output_to_xyz(self, encoded_rgb) -> np.ndarray:
        """LUT output (encoded RGB) → reflectance-scale XYZ.

        Equivalent to :func:`to_xyz_qa` with the output color space
        bound on ``self``.
        """
        return to_xyz_qa(encoded_rgb, self.output_color_space)
