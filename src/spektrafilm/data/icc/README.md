# ICC profiles

These ICC profiles are embedded into images saved by `save_image_oiio` so that
ICC-aware viewers (browsers, Photoshop, Lightroom, Preview, etc.) display the
pixels with the correct color interpretation. The lookup table lives in
[`spektrafilm/utils/io.py`](../../utils/io.py) (`_ICC_FILENAMES`).

Files are kept under their **original upstream names** so they stay traceable
to the source repos.

## Sources & licenses

### `ellelstone/` — Elle Stone's well-behaved ICC profiles

- Repository: <https://github.com/ellelstone/elles_icc_profiles>
- License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)
- Attribution required. See `ellelstone/LICENSE-CC-BY-SA-3.0` (if not present,
  copy from the upstream repo's `LICENSE` file).
- Naming convention: `<colorspace>-elle-V<2|4>-<trc>.icc`
  - V2 = ICC v2 (broader viewer compatibility — preferred for embedding)
  - V4 = ICC v4 (more accurate, but some viewers ignore it)
  - TRC suffixes: `g10` (linear), `g18` (γ 1.8), `g22` (γ 2.2),
    `srgbtrc` (sRGB curve), `rec709` (Rec.709/BT.2020 curve), `labl` (L\*)

Used here for: sRGB, Adobe RGB (via `ClayRGB`), ProPhoto RGB (via `LargeRGB`),
ITU-R BT.2020 (via `Rec2020`), ACES2065-1 (via `ACES`, which is AP0 primaries).

### `saucecontrol/` — Compact ICC Profiles

- Repository: <https://github.com/saucecontrol/Compact-ICC-Profiles>
- License: MIT
- Attribution required. See `saucecontrol/LICENSE-MIT` (if not present, copy
  from the upstream repo's `LICENSE` file).
- These are byte-minimal ICC profiles (~400 B – 2 KB) for the standard display
  spaces.

Used here for: Display P3, DCI-P3 — the spaces Elle Stone's set doesn't cover
directly.

## Active mapping

The current `(RGBColorSpaces enum value, cctf_encoded)` → file mapping:

| Color space        | Encoded                                         | Linear                                |
| ------------------ | ----------------------------------------------- | ------------------------------------- |
| sRGB               | `ellelstone/sRGB-elle-V2-srgbtrc.icc`           | `ellelstone/sRGB-elle-V2-g10.icc`     |
| Adobe RGB (1998)   | `ellelstone/ClayRGB-elle-V2-g22.icc`            | `ellelstone/ClayRGB-elle-V2-g10.icc`  |
| ProPhoto RGB       | `ellelstone/LargeRGB-elle-V2-g18.icc`           | `ellelstone/LargeRGB-elle-V2-g10.icc` |
| ITU-R BT.2020      | `ellelstone/Rec2020-elle-V2-rec709.icc`         | `ellelstone/Rec2020-elle-V2-g10.icc`  |
| ACES2065-1         | `ellelstone/ACES-elle-V2-g10.icc` (linear)      | `ellelstone/ACES-elle-V2-g10.icc`     |
| Display P3         | `saucecontrol/DisplayP3-v2-micro.icc`           | — (no compact linear ICC upstream)    |
| DCI-P3             | `saucecontrol/DCI-P3-v4.icc`                    | — (no compact linear ICC upstream)    |

Missing combinations are skipped silently — the image still gets the EXIF
`ColorSpace` / XMP `photoshop:ICCProfile` tagging written by
`write_image_metadata`, just no embedded ICC stream.

## Notes

- `ClayRGB` = Adobe RGB primaries under a different name (Elle uses it to
  avoid the Adobe trademark). It's a drop-in for "Adobe RGB (1998)".
- `LargeRGB` = ROMM RGB primaries (= ProPhoto RGB). The γ 1.8 TRC approximates
  ProPhoto's piecewise curve.
- ACES2065-1 uses AP0 primaries and is always scene-linear, so both the
  encoded and linear table entries point to the same `g10` file.
