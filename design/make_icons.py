"""Rasterise the Pixforge app icon (design/pixforge-icon*.svg) to PNG and a multi-size ICO.

The geometry is simple (rounded square, stair-stepped half disc, smooth half disc, spark),
so it is drawn directly with Pillow at 1024 px and downsampled; no SVG renderer needed.
Run: uv run --project ../backend python make_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
BG, LINE, EMBER = (0x14, 0x14, 0x16, 255), (0x2B, 0x2B, 0x30, 255), (0xF2, 0xA5, 0x41, 255)
SPARK = (0xF2, 0xA5, 0x41, 179)
BIG = 1024


def draw(small: bool) -> Image.Image:
    k = BIG / 64
    im = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    radius = (12 if small else 13.5) * k
    if small:
        d.rounded_rectangle((0, 0, BIG - 1, BIG - 1), radius, fill=BG)
        r, stairs = 24, [(12, 24), (20, 16), (44, 12), (52, 16), (56, 24)]
    else:
        d.rounded_rectangle((0.5 * k, 0.5 * k, 63.5 * k, 63.5 * k), radius, fill=BG,
                            outline=LINE, width=round(k))
        r, stairs = 20, [(16, 24), (24, 16), (40, 12), (48, 16), (52, 24)]
    # left half: staircase polygon, (y where a step ends, x of that step); right: half disc
    pts = [(32, 32 - r)]
    y_prev = 32 - r
    for y_end, x in stairs:
        pts += [(x, y_prev), (x, y_end)]
        y_prev = y_end
    pts.append((32, 32 + r))
    d.polygon([(x * k, y * k) for x, y in pts], fill=EMBER)
    d.pieslice(((32 - r) * k, (32 - r) * k, (32 + r) * k, (32 + r) * k), -90, 90, fill=EMBER)
    if not small:
        spark = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(spark).rectangle((13 * k, 6 * k, 18 * k - 1, 11 * k - 1), fill=SPARK)
        im = Image.alpha_composite(im, spark)
    return im


def main() -> None:
    big, small = draw(False), draw(True)
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [(small if s < 48 else big).resize((s, s), Image.LANCZOS) for s in sizes]
    big.resize((512, 512), Image.LANCZOS).save(HERE / "pixforge-icon-512.png")
    big.resize((256, 256), Image.LANCZOS).save(HERE / "pixforge-icon-256.png")
    frames[-1].save(HERE / "pixforge.ico", format="ICO", sizes=[(s, s) for s in sizes],
                    append_images=frames[:-1])
    print("wrote", ", ".join(p.name for p in sorted(HERE.glob("pixforge*.*"))))


if __name__ == "__main__":
    main()
