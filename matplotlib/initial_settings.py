import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_dirs = [os.path.join("fonts")]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

DARK_BLUE = "#141b4d"
LIGHT_GRAY = "#8f8f99"

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": DARK_BLUE,
    "ytick.color": DARK_BLUE,
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
    "xtick.bottom": False,
    "xtick.labelbottom": False,
    "ytick.left": False,
    "ytick.labelleft": False,
    "grid.color": LIGHT_GRAY,
    "grid.linestyle": "dashed",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "axes.labelpad": 10,
    "axes.labelcolor": DARK_BLUE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "text.color": DARK_BLUE,
    "savefig.dpi": 300,
})
