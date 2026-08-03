#!/usr/bin/env python3
"""Regenerate Mimodium production logo assets from the editable SVG."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil
from xml.etree import ElementTree as ET

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont


BRAND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BRAND_DIR.parents[2]
EDITABLE_SVG = BRAND_DIR / "mimodium-logo-editable.svg"
PRODUCTION_SVG = BRAND_DIR / "mimodium-logo.svg"
DARK_PRODUCTION_SVG = BRAND_DIR / "mimodium-logo-dark.svg"
MARK_SVG = BRAND_DIR / "mimodium-mark.svg"
DARK_MARK_PNG = BRAND_DIR / "mimodium-mark-dark.png"
FONT_DIR = BRAND_DIR / "font"
FONT_FILE = FONT_DIR / "Go-Medium.ttf"
EXPORT_DIR = BRAND_DIR / "exports"

# Bootstrap locations are used only when the tracked editable source/font are
# absent. Once generated, this script depends exclusively on files in BRAND_DIR.
DESIGN_SOURCE = (
    PROJECT_ROOT
    / "design"
    / "logo-svg"
    / "mimodium-computational-indigo"
    / "case-study"
    / "14-go-lowercase.svg"
)
SYSTEM_FONT = Path("/usr/share/fonts/fonts-go/Go-Medium.ttf")

SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}
WORD = "Mimodium"
WORD_COLOR = "#29366F"
DARK_PAGE_BACKGROUND = "#14181E"
DARK_CELL_COLOR = "#AEB8EF"
DARK_WORD_COLOR = "#F2F4FF"
LOGO_WIDTH = 1950
LOGO_HEIGHT = 700
FONT_SIZE = 260
TEXT_X = 660
TEXT_BASELINE = 416
ICON_TRANSFORM = "translate(-100 0) scale(0.9)"
PNG_ICON_SIZES = (512, 256, 128, 64, 32)
PNG_LOGO_NAME = f"mimodium-logo-{LOGO_WIDTH}x{LOGO_HEIGHT}.png"


def find(root: ET.Element, path: str) -> ET.Element:
    element = root.find(path, NS)
    if element is None:
        raise ValueError(f"Missing required SVG element: {path}")
    return element


def bootstrap_sources() -> None:
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    if not EDITABLE_SVG.exists():
        if not DESIGN_SOURCE.exists():
            raise FileNotFoundError(
                f"Editable source missing and bootstrap source unavailable: {DESIGN_SOURCE}"
            )
        shutil.copyfile(DESIGN_SOURCE, EDITABLE_SVG)
    if not FONT_FILE.exists():
        if not SYSTEM_FONT.exists():
            raise FileNotFoundError(
                f"Vendored font missing and system source unavailable: {SYSTEM_FONT}"
            )
        shutil.copyfile(SYSTEM_FONT, FONT_FILE)


def update_editable_wordmark() -> None:
    """Keep the tracked live-text master aligned with the canonical spelling."""
    tree = ET.parse(EDITABLE_SVG)
    root = tree.getroot()
    title = find(root, ".//svg:title[@id='title']")
    description = find(root, ".//svg:desc[@id='description']")
    background = find(root, ".//svg:rect[@id='background']")
    icon = find(root, ".//svg:g[@id='computational-indigo-icon']")
    wordmark = find(root, ".//svg:text[@id='wordmark']")
    title.text = "Mimodium computational indigo editable logo"
    description.text = (
        "The Mimodium cellular membrane mark with a live Go Medium wordmark."
    )
    root.set("width", str(LOGO_WIDTH))
    root.set("height", str(LOGO_HEIGHT))
    root.set("viewBox", f"0 0 {LOGO_WIDTH} {LOGO_HEIGHT}")
    background.set("width", str(LOGO_WIDTH))
    background.set("height", str(LOGO_HEIGHT))
    icon.set("transform", ICON_TRANSFORM)
    wordmark.set("x", str(TEXT_X))
    wordmark.set("y", str(TEXT_BASELINE))
    wordmark.set("font-size", str(FONT_SIZE))
    wordmark.text = WORD
    root.set("data-editable-asset", "live-go-medium-wordmark")
    ET.indent(tree, space="  ")
    tree.write(EDITABLE_SVG, encoding="utf-8", xml_declaration=True)


def wordmark_outline() -> ET.Element:
    """Convert the Go Medium wordmark to portable SVG glyph paths."""
    ttfont = TTFont(FONT_FILE)
    glyph_set = ttfont.getGlyphSet()
    cmap = ttfont.getBestCmap()
    units_per_em = ttfont["head"].unitsPerEm
    scale = FONT_SIZE / units_per_em
    pillow_font = ImageFont.truetype(FONT_FILE, FONT_SIZE)

    group = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": "wordmark-outlined",
            "fill": WORD_COLOR,
            "aria-label": WORD,
        },
    )
    for index, character in enumerate(WORD):
        glyph_name = cmap[ord(character)]
        pen = SVGPathPen(glyph_set)
        glyph_set[glyph_name].draw(pen)
        position = pillow_font.getlength(WORD[:index])
        path = ET.SubElement(
            group,
            f"{{{SVG_NS}}}path",
            {
                "id": f"wordmark-glyph-{index + 1}-{character}",
                "d": pen.getCommands(),
                "transform": (
                    f"translate({TEXT_X + position:.4f} {TEXT_BASELINE}) "
                    f"scale({scale:.8f} {-scale:.8f})"
                ),
            },
        )
        path.set("data-character", character)
    ttfont.close()
    return group


def build_production_svg() -> None:
    tree = ET.parse(EDITABLE_SVG)
    root = tree.getroot()
    title = find(root, ".//svg:title[@id='title']")
    description = find(root, ".//svg:desc[@id='description']")
    wordmark = find(root, ".//svg:text[@id='wordmark']")
    title.text = "Mimodium computational indigo logo"
    description.text = (
        "The Mimodium cellular membrane mark with a capitalized wordmark. "
        "The wordmark is outlined for portable rendering."
    )
    parent = next(parent for parent in root.iter() if wordmark in list(parent))
    wordmark_index = list(parent).index(wordmark)
    parent.remove(wordmark)
    parent.insert(wordmark_index, wordmark_outline())
    root.set("data-production-asset", "outlined-wordmark")
    ET.indent(tree, space="  ")
    tree.write(PRODUCTION_SVG, encoding="utf-8", xml_declaration=True)


def build_mark_svg() -> None:
    source = ET.parse(EDITABLE_SVG).getroot()
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": "700",
            "height": "700",
            "viewBox": "0 0 700 700",
            "role": "img",
            "aria-labelledby": "title description",
        },
    )
    title = ET.SubElement(root, f"{{{SVG_NS}}}title", {"id": "title"})
    title.text = "Mimodium computational indigo mark"
    description = ET.SubElement(root, f"{{{SVG_NS}}}desc", {"id": "description"})
    description.text = (
        "Seven indigo hexagonal cells crossed by four blue and teal membranes."
    )
    root.append(deepcopy(find(source, ".//svg:defs")))
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {
            "id": "background",
            "width": "700",
            "height": "700",
            "fill": "#FFFFFF",
        },
    )
    icon = ET.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {"id": "computational-indigo-mark", "transform": "translate(-150 0)"},
    )
    icon.append(deepcopy(find(source, ".//svg:g[@id='cells']")))
    icon.append(deepcopy(find(source, ".//svg:g[@id='membranes']")))
    icon.append(deepcopy(find(source, ".//svg:path[@id='outside-cell-cover']")))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(MARK_SVG, encoding="utf-8", xml_declaration=True)


def build_dark_svg(source: Path, destination: Path, background_color: str) -> None:
    """Create a dark-background variant while retaining the membrane colors."""
    tree = ET.parse(source)
    root = tree.getroot()
    title = find(root, ".//svg:title[@id='title']")
    description = find(root, ".//svg:desc[@id='description']")
    background = root.find(".//svg:rect[@id='background']", NS)
    if background is not None:
        background.set("fill", background_color)
    outside_cover = root.find(".//svg:path[@id='outside-cell-cover']", NS)
    if outside_cover is not None:
        outside_cover.set("fill", background_color)
    find(root, ".//svg:g[@id='cells']").set("fill", DARK_CELL_COLOR)
    wordmark = root.find(".//svg:g[@id='wordmark-outlined']", NS)
    if wordmark is not None:
        wordmark.set("fill", DARK_WORD_COLOR)
    title.text = f"{title.text} for dark backgrounds"
    description.text = f"{description.text} Adapted for dark backgrounds."
    root.set("data-color-mode", "dark")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def cubic_polygon(path_data: str, samples: int = 16) -> list[tuple[float, float]]:
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path_data)]
    if (len(values) - 2) % 6:
        raise ValueError("Unexpected membrane path structure")
    current = (values[0], values[1])
    polygon = [current]
    for index in range(2, len(values), 6):
        c1 = (values[index], values[index + 1])
        c2 = (values[index + 2], values[index + 3])
        end = (values[index + 4], values[index + 5])
        for sample in range(1, samples + 1):
            t = sample / samples
            u = 1 - t
            polygon.append(
                (
                    u**3 * current[0]
                    + 3 * u * u * t * c1[0]
                    + 3 * u * t * t * c2[0]
                    + t**3 * end[0],
                    u**3 * current[1]
                    + 3 * u * u * t * c1[1]
                    + 3 * u * t * t * c2[1]
                    + t**3 * end[1],
                )
            )
        current = end
    return polygon


def hex_points(
    center: tuple[float, float], vertices: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    return [(center[0] + x, center[1] + y) for x, y in vertices]


def parse_translate(value: str) -> tuple[float, float]:
    match = re.fullmatch(r"translate\(([-.0-9]+) ([-.0-9]+)\)", value)
    if match is None:
        raise ValueError(f"Unexpected transform: {value}")
    return float(match.group(1)), float(match.group(2))


def parse_hex(value: str) -> tuple[int, int, int]:
    color = value.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def render_icon(
    *,
    background: tuple[int, int, int, int] = (255, 255, 255, 255),
    cell_color: str | None = None,
) -> Image.Image:
    root = ET.parse(EDITABLE_SVG).getroot()
    cell_definition = find(root, ".//svg:polygon[@id='cell']")
    vertices = [
        tuple(float(coordinate) for coordinate in point.split(","))
        for point in cell_definition.attrib["points"].split()
    ]
    cells = root.findall(".//svg:g[@id='cells']/svg:use", NS)
    rendered_cell_color = parse_hex(
        cell_color or find(root, ".//svg:g[@id='cells']").attrib["fill"]
    )

    image = Image.new("RGBA", (1000, 700), background)
    cell_mask = Image.new("L", image.size, 0)
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(cell_mask)
    for cell in cells:
        polygon = hex_points(parse_translate(cell.attrib["transform"]), vertices)
        image_draw.polygon(polygon, fill=rendered_cell_color + (255,))
        mask_draw.polygon(polygon, fill=255)

    for membrane in root.findall(".//svg:g[@id='membranes']/svg:path", NS):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        color = parse_hex(membrane.attrib["fill"])
        alpha = round(float(membrane.attrib["fill-opacity"]) * 255)
        ImageDraw.Draw(layer).polygon(
            cubic_polygon(membrane.attrib["d"]),
            fill=color + (alpha,),
        )
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), cell_mask))
        image = Image.alpha_composite(image, layer)
    return image


def build_dark_mark_png() -> None:
    """Create a transparent navbar mark for dark mode."""
    icon = render_icon(background=(0, 0, 0, 0), cell_color=DARK_CELL_COLOR)
    icon.crop((150, 0, 850, 700)).save(DARK_MARK_PNG)


def render_pngs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    icon = render_icon()
    horizontal = Image.new("RGBA", (LOGO_WIDTH, LOGO_HEIGHT), (255, 255, 255, 255))
    scaled_icon = icon.resize((900, 630), Image.Resampling.LANCZOS)
    horizontal.alpha_composite(scaled_icon, (-100, 0))
    font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
    ImageDraw.Draw(horizontal).text(
        (TEXT_X, TEXT_BASELINE),
        WORD,
        fill=parse_hex(WORD_COLOR) + (255,),
        font=font,
        anchor="ls",
    )
    horizontal.convert("RGB").save(EXPORT_DIR / PNG_LOGO_NAME)

    square = icon.crop((150, 0, 850, 700))
    for size in PNG_ICON_SIZES:
        resized = square.resize((size, size), Image.Resampling.LANCZOS)
        resized.convert("RGB").save(EXPORT_DIR / f"mimodium-mark-{size}.png")


def validate(*, include_pngs: bool) -> None:
    production = ET.parse(PRODUCTION_SVG).getroot()
    dark_production = ET.parse(DARK_PRODUCTION_SVG).getroot()
    editable = ET.parse(EDITABLE_SVG).getroot()
    mark = ET.parse(MARK_SVG).getroot()
    assert production.find(".//svg:text", NS) is None
    assert len(
        production.findall(".//svg:g[@id='wordmark-outlined']/svg:path", NS)
    ) == len(WORD)
    assert find(editable, ".//svg:text[@id='wordmark']").text == WORD
    assert len(mark.findall(".//svg:g[@id='membranes']/svg:path", NS)) == 4
    assert len(mark.findall(".//svg:g[@id='cells']/svg:use", NS)) == 7
    assert find(dark_production, ".//svg:rect[@id='background']").get("fill") == (
        DARK_PAGE_BACKGROUND
    )
    assert find(dark_production, ".//svg:g[@id='cells']").get("fill") == (
        DARK_CELL_COLOR
    )
    with Image.open(DARK_MARK_PNG) as dark_mark:
        assert dark_mark.size == (700, 700)
        assert dark_mark.mode == "RGBA"
        assert dark_mark.getpixel((0, 0))[3] == 0
    if include_pngs:
        for size in PNG_ICON_SIZES:
            with Image.open(EXPORT_DIR / f"mimodium-mark-{size}.png") as image:
                assert image.size == (size, size)
        with Image.open(EXPORT_DIR / PNG_LOGO_NAME) as image:
            assert image.size == (LOGO_WIDTH, LOGO_HEIGHT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png",
        action="store_true",
        help="also create optional PNG fallbacks in the git-ignored exports directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_sources()
    ET.register_namespace("", SVG_NS)
    update_editable_wordmark()
    build_production_svg()
    build_mark_svg()
    build_dark_svg(PRODUCTION_SVG, DARK_PRODUCTION_SVG, DARK_PAGE_BACKGROUND)
    build_dark_mark_png()
    if args.png:
        render_pngs()
    validate(include_pngs=args.png)
    print("Generated production, editable, mark, and font brand assets")
    if args.png:
        print(f"Generated optional PNG exports in {EXPORT_DIR}")


if __name__ == "__main__":
    main()
