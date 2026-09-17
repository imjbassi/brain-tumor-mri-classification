"""Derive the revision-format manuscript from the submission manuscript.

Scientific Reports asks that revised manuscripts be single-column, unjustified,
and paginated, with line numbers to help reviewers reference the text. Rather
than maintain two copies of the paper, this rewrites paper/main.tex into
paper/main_revision.tex; main.tex stays the single source of the content.

Usage:
    python src/make_revision_tex.py
"""
import re
from pathlib import Path

SRC = Path("paper/main.tex")
DST = Path("paper/main_revision.tex")

PREAMBLE_EXTRA = r"""\emergencystretch=1.5em
\usepackage{ragged2e}
\RaggedRight
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}"""

RULES = [
    # single column, wider margins to keep the measure readable
    (r"\\documentclass\[twocolumn\]\{article\}", r"\\documentclass[11pt]{article}"),
    (r"\\usepackage\[margin=0\.75in\]\{geometry\}", r"\\usepackage[margin=1in]{geometry}"),
    # line numbers sit beside the text once there is only one column
    (r"\\usepackage\[switch,mathlines\]\{lineno\}", r"\\usepackage[mathlines]{lineno}"),
    # full-width float environments have no meaning in a one-column layout
    (r"\\begin\{figure\*\}", r"\\begin{figure}"),
    (r"\\end\{figure\*\}", r"\\end{figure}"),
    # ragged-right text, page numbers in the footer
    (r"\\emergencystretch=1\.5em", PREAMBLE_EXTRA.replace("\\", "\\\\")),
    # panels sized for a two-column page are too wide at full measure
    (r"\\includegraphics\[width=\\linewidth\]", r"\\includegraphics[width=0.85\\linewidth]"),
]


def main():
    tex = SRC.read_text(encoding="utf-8")
    for pattern, replacement in RULES:
        tex, n = re.subn(pattern, replacement, tex)
        if n == 0:
            raise SystemExit(f"pattern not found in {SRC}: {pattern}")
    DST.write_text(tex, encoding="utf-8")
    print(f"wrote {DST} from {SRC}")


if __name__ == "__main__":
    main()
