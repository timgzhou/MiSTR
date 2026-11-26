"""Shared figure style for the paper's accuracy-vs-compute plots: colours, the MiSTR pair names,
the serif font setup and a parameter-count formatter."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OURS = "#eb6834"                                  # MiSTR
COLOR = {"apt": "#2a78d6", "tldr": "#1baf7a", "anyup": "#eda100", "upa": "#e87ba4"}   # baselines
AREA_FLOOR, AREA_MAX = 22, 330          # pt^2: a zero-parameter marker, AnyUp's

QUAD_COL = "#1b7a3e"
BASE_COL = "#8c8b86"    # content-blind baselines (grid, random) on the baselines figure
# the same, one season, with cross attention in pass 2 (model.attn cross_hr: crop tokens read the
# season tokens and the sample's other crops, season tokens never read crops, so inference
# reuses the crop-free pass; ATTN=cross_hr in mixres_quadrant_rep.sh): tag, label, colour
XHR = ("-hhone-crop-score-xhr", "entropy (global threshold) + attention, one season, cross attn",
       "#00838f")


PAIR = {   # (season patch, crop geometry) -> colour: one hue per season patch, dark for the
    # pair whose crops are ~half the season patch, light for the other
    (16, "c8"): "#1565c0", (16, "c12"): "#90caf9", (14, "c14p7"): "#ad1457",
    (12, "c12p6"): "#6d4c41", (10, "c10p5"): "#2e7d32", (10, "c8"): "#a5d6a7",
    (8, "c4"): "#00838f", (8, "c6"): "#80deea", (6, "c6p3"): "#e65100", (6, "c4"): "#ffb74d",
    (4, "c4p2"): "#6a1b9a"}

def pair_name(ps, geo):
    c = re.match(r"c(\d+)(?:p(\d+))?", geo)
    return rf"$S_{{{ps}}}T_{{sea}}+S_{{{c.group(2) or c.group(1)}}}T_{{mon}}$"   # where the label sits beside its curve


def count(n):
    return f"{round(n / 1000)}k" if n >= 1000 else f"{n}"

def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 11})
