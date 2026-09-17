"""Compose the four main display figures as single files for submission.

Scientific Reports asks for one image file per figure, with multi-panel figures
arranged as a single image rather than one upload per panel, and for panel
labels set in bold lower-case type. Figures 2, 3 and 4 of the manuscript are
two-panel figures, so this script stacks the panels, scales them to a common
edge, and writes the labelled result at 300 DPI.

Usage:
    python src/make_submission_figures.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

FIG = "paper/figures"
OUT = os.path.join(FIG, "submission")
DPI = 300
LABEL_PX = 64          # cap height of the panel label
PAD = 24               # padding around the label glyph
GAP = 40               # gap between panels

# (output name, [panel files], layout) for each numbered manuscript figure
FIGURES = [
    ("Figure_1", ["leakage_examples.png"], "single"),
    ("Figure_2", ["cv_per_class_f1.png", "cv_fold_spread.png"], "row"),
    ("Figure_3", ["cv_paired.png", "rand_control.png"], "row"),
    ("Figure_4", ["gradcam_examples.png", "gradcam_patient_disjoint.png"], "column"),
]


def label_font():
    """A bold sans-serif face, as the figure guidelines require."""
    for candidate in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, LABEL_PX)
        except OSError:
            continue
    return ImageFont.load_default()


def scale_to(img, *, height=None, width=None):
    if height is not None and img.height != height:
        return img.resize((round(img.width * height / img.height), height), Image.LANCZOS)
    if width is not None and img.width != width:
        return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    return img


def compose(panels, layout):
    if layout == "row":
        height = max(p.height for p in panels)
        panels = [scale_to(p, height=height) for p in panels]
        canvas = Image.new("RGB", (sum(p.width for p in panels) + GAP * (len(panels) - 1), height), "white")
        x = 0
        origins = []
        for p in panels:
            canvas.paste(p, (x, 0))
            origins.append((x, 0))
            x += p.width + GAP
        return canvas, origins
    if layout == "column":
        width = max(p.width for p in panels)
        panels = [scale_to(p, width=width) for p in panels]
        canvas = Image.new("RGB", (width, sum(p.height for p in panels) + GAP * (len(panels) - 1)), "white")
        y = 0
        origins = []
        for p in panels:
            canvas.paste(p, (0, y))
            origins.append((0, y))
            y += p.height + GAP
        return canvas, origins
    return panels[0].convert("RGB"), [(0, 0)]


def main():
    os.makedirs(OUT, exist_ok=True)
    font = label_font()
    for name, files, layout in FIGURES:
        panels = [Image.open(os.path.join(FIG, f)).convert("RGB") for f in files]
        canvas, origins = compose(panels, layout)
        if len(panels) > 1:
            draw = ImageDraw.Draw(canvas)
            for (x, y), letter in zip(origins, "abcdefg"):
                draw.text((x + PAD, y + PAD), letter, fill="black", font=font)
        path = os.path.join(OUT, f"{name}.png")
        canvas.save(path, dpi=(DPI, DPI))
        print(f"wrote {path}  {canvas.width}x{canvas.height}px  "
              f"({canvas.width / DPI:.1f}x{canvas.height / DPI:.1f} in at {DPI} DPI)")


if __name__ == "__main__":
    main()
