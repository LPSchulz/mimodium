# Mimodium brand assets

The production identity is the **Computational Indigo** mark with a capitalized
`Mimodium` wordmark set in **Go Medium**.

## Which file to use

| Asset | Purpose |
| --- | --- |
| `mimodium-logo.svg` | Preferred horizontal logo. The wordmark is outlined, so it renders identically without an installed font. |
| `mimodium-logo-editable.svg` | Editable master with a live text wordmark. Use this when changing the name, type, layout, or color. |
| `mimodium-mark.svg` | Square icon-only mark. |

SVG is the source of truth and is sufficient for GitHub, Sphinx, and modern web
use. PNGs are generated only when a raster-only consumer requires them.

## Typography

- Typeface: **Go**
- Style: **Medium**
- SVG family name: `Go Medium`
- Weight: `500`
- Wordmark spelling: `Mimodium`
- Vendored font: `font/Go-Medium.ttf`
- Copyright: Copyright © 2016 Bigelow & Holmes Inc.
- License: BSD 3-Clause; see `GO-FONT-LICENSE.txt`
- Upstream: `go.googlesource.com/image/+/master/font/gofont/ttfs/`

The production SVG converts the wordmark to paths, while the editable SVG keeps
it as live text. This provides portable rendering without losing editability.

## Colors and opacity

| Role | Value |
| --- | --- |
| Cells and wordmark | `#29366F` |
| Blue membranes | `#5FA8FF` at `0.76` and `0.74` opacity |
| Teal membranes | `#73E0D1` at `0.72` and `0.66` opacity |
| Background and cell gutters | `#FFFFFF` |

Normal alpha compositing creates the additional overlap colors. Do not replace
those overlaps with separately drawn shapes.

## Editing and regeneration

1. Edit `mimodium-logo-editable.svg`.
2. Keep the seven cell positions and four membrane paths unchanged unless the
   mark itself is intentionally being redesigned.
3. Keep `font/Go-Medium.ttf` available when editing or exporting the live text.
4. From the project root, run:

   ```bash
   python docs/_static/brand/generate_assets.py
   ```

The generator outlines the wordmark, creates the icon-only SVG, and validates
the expected geometry and dimensions.

### Optional PNG exports

Do not commit raster exports unless a specific downstream system requires them.
Generate a horizontal PNG and square icons at 512, 256, 128, 64, and 32 pixels
on demand:

```bash
python docs/_static/brand/generate_assets.py --png
```

The PNGs are written to the git-ignored `exports/` directory.

## Dimensions

- Horizontal lockup view box: `1950 × 700`
- Square mark view box: `700 × 700`

Maintain clear space around the logo and avoid stretching, recoloring individual
cells, changing membrane opacity, adding effects, or typesetting the wordmark in
a substitute font.
