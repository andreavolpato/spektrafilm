"""Registry of field-verified manufacturer / camera delivery targets.

A :class:`DeliveryTarget` declares the constraints needed to produce a
LUT that a specific camera or workflow accepts:

- which format plugin to write through (``cube``, ``lumix``, …);
- which input/output color spaces are valid;
- the recommended grid resolution;
- any extra metadata the writer needs (e.g., Lumix's ``#LUMIXPHOTOSTYLE``
  tag, future ARRI/Sony equivalents);
- a record of *what's been field-verified* and on which cameras.

The registry is gatekept: entries land here only after a real camera has
accepted the produced file. Adding a target is one :func:`register`
call. Use ``target = None`` on a :class:`BundleSpec` for the generic
Adobe ``.cube`` path with no vendor constraints.

The current catalog ships one verified entry — ``lumix_realtime_vlog``
— mirrored from a user's working reference script. Other vendors and
workflows should be added in their own commits with the verification
context recorded in the ``verified`` field.

See [n080](../../../spektrafilm-research/studies/a40_lut_system/n080_lut_quality_and_visualization.md)
for the broader LUT-export plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeliveryTarget:
    """A camera or workflow constraint set for LUT delivery.

    Fields:
        name: registry key (snake_case).
        description: one-paragraph human-readable description.
        format: key into the LUT-format registry (``"cube"``, ``"lumix"``).
        valid_inputs: color-space registry names accepted as bundle input.
        valid_outputs: color-space registry names accepted as bundle output.
        recommended_resolution: grid size the target works best at.
        writer_kwargs: extra kwargs passed verbatim to the format
            plugin's ``write()`` (e.g., ``{"photo_style_tag": "VLOG"}``).
        cameras: tuple of camera model names known to accept this target.
        verified: how / where this target was field-tested.
        notes: anything else worth surfacing in code reviews / docs.
    """

    name: str
    description: str
    format: str
    valid_inputs: tuple[str, ...]
    valid_outputs: tuple[str, ...]
    recommended_resolution: int = 33
    writer_kwargs: Mapping[str, str] = field(default_factory=dict)
    cameras: tuple[str, ...] = ()
    verified: str = ""
    notes: str = ""


_REGISTRY: dict[str, DeliveryTarget] = {}


def register(entry: DeliveryTarget) -> None:
    """Validate and add ``entry`` to the registry."""
    # Defer the import to avoid a registration-time cycle through
    # formats/__init__.py.
    from spektrafilm_lut_creator.formats import LUT_FORMATS

    if entry.format not in LUT_FORMATS:
        raise KeyError(
            f"DeliveryTarget {entry.name!r}: unknown format "
            f"{entry.format!r}. Registered: {sorted(LUT_FORMATS)}"
        )
    if not entry.valid_inputs:
        raise ValueError(
            f"DeliveryTarget {entry.name!r}: valid_inputs must not be empty"
        )
    if not entry.valid_outputs:
        raise ValueError(
            f"DeliveryTarget {entry.name!r}: valid_outputs must not be empty"
        )
    _REGISTRY[entry.name] = entry


def get(name: str) -> DeliveryTarget:
    """Return the registered target ``name`` or raise :class:`KeyError`."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown delivery target {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def list_targets() -> list[str]:
    """Names of all registered targets."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in entries — extend only with field-verified additions.
# ---------------------------------------------------------------------------

register(
    DeliveryTarget(
        name="lumix_realtime_vlog",
        description=(
            "Panasonic Lumix Real-Time LUT, V-Log input. The LUT loads "
            "in-camera (Photo Style → Real-Time LUT) and applies live to "
            "monitor preview and recording. Output is the camera's display "
            "encoding (sRGB / Rec.709 for SDR, Rec.2020 for HDR modes)."
        ),
        format="lumix",
        valid_inputs=("Panasonic V-Log",),
        valid_outputs=("sRGB", "Rec.709", "Rec.2020"),
        recommended_resolution=33,
        writer_kwargs={"photo_style_tag": "VLOG"},
        cameras=(
            "Panasonic Lumix S5II",
            "Lumix S5IIX",
            "Lumix S9",
            "Lumix S1II",
            "Lumix S1IIE",
            "Lumix GH7",
        ),
        verified=(
            "Field-tested with a 33^3 .cube using this exact header layout "
            "(TITLE → #LUMIXPHOTOSTYLE VLOG → LUT_3D_SIZE → DOMAIN_MIN → "
            "DOMAIN_MAX → blank → data). Loads via Lumix Lab and SD-card "
            "import. Reference: user's pre-spektrafilm script."
        ),
        notes=(
            "Only the VLOG photo-style tag is currently included. HLG / STD "
            "/ other Lumix photo styles will land as separate targets once "
            "each is verified on a real camera."
        ),
    )
)
