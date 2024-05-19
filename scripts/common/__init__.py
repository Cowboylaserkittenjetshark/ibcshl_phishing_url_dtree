import matplotlib as mpl
import mplcatppuccin
import seaborn as sns
import logging

# Plot style
TRANSPARENT = False
sns.set_style("whitegrid")
sns.set(rc={"figure.dpi": 1200, "savefig.dpi": 1200})
mpl.style.use("mocha")

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
