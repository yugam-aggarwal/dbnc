"""Shared publication style for DBNC paper figures.

Centralises matplotlib rcParams, a colour-blind-safe palette (Okabe-Ito), and a
consistent per-method-group colour map so that every figure in the paper (and the
internal benchmark report) shares one typographic and chromatic identity.

Design choices follow standard scientific-figure guidance:
  * vector PDF output with embedded TrueType fonts (pdf.fonttype 42 -> no Type-3,
    which IEEE PDF eXpress rejects);
  * a serif body font close to the IEEE Times text, with Computer-Modern mathtext
    (no usetex, so the figure build needs no LaTeX render toolchain);
  * high data-ink ratio (top/right spines off, grids off by default);
  * the Okabe-Ito categorical palette, which is distinguishable under the common
    forms of colour-vision deficiency, never relying on colour alone.

Import and call :func:`apply_style` at the top of any figure-generating script.
"""

from __future__ import annotations

# --- Okabe-Ito colour-blind-safe categorical palette ------------------------
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
}

# Method-group colours, reused identically across every figure.  DBNC is given
# the high-salience vermilion so the focal method stands out everywhere.
GROUP_COLORS = {
    "DBNC": "#D55E00",            # vermilion (focal)
    "Classical BNC": "#0072B2",   # blue
    "Boosted tree": "#009E73",    # green
    "Neural tabular": "#CC79A7",  # purple
}

# Perceptually-uniform sequential map for any heatmap (reversed so rank 1 = bright).
SEQ_CMAP = "cividis_r"

# Diverging map for signed quantities (deltas, win-minus-loss); colour-blind safe.
DIV_CMAP = "BrBG"

# Short labels matching the abbreviations already explained in the CD-diagram caption.
SHORT = {
    "LightGBM": "LGBM",
    "XGBoost": "XGB",
    "CatBoost": "Cat",
    "FT-Transformer": "FT-Tr",
    "DBNC": "DBNC",
    "BAN": "BAN",
    "KDB-2": "KDB-2",
    "KDB-3": "KDB-3",
    "TAN": "TAN",
    "KDB-1": "KDB-1",
    "NB": "NB",
    "Chow-Liu": "CL",
}

# IEEE column geometry (inches): single column ~252pt, full text width ~516pt.
COL_W = 3.5
FULL_W = 7.16

_BOOSTED = {"XGBoost", "LightGBM", "CatBoost"}


def group_of(method: str) -> str:
    """Map a method label to its display group."""
    if method in _BOOSTED:
        return "Boosted tree"
    if method == "FT-Transformer":
        return "Neural tabular"
    if method == "DBNC":
        return "DBNC"
    return "Classical BNC"


def color_of(method: str) -> str:
    """Consistent colour for a method, via its group."""
    return GROUP_COLORS[group_of(method)]


def short(method: str) -> str:
    return SHORT.get(method, method)


_RC = {
    # Typography: serif body + Computer-Modern math, Times-compatible.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    # Spines / ticks: minimal chrome.
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # Data primitives.
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.3,
    # Output: vector PDF, embedded TrueType, tight bounds.
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply the shared rcParams.  Call once at script start."""
    import matplotlib

    matplotlib.rcParams.update(_RC)
