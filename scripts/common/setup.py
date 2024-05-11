import matplotlib as mpl
import mplcatppuccin
import seaborn as sns
import logging

# Plot style
TRANSPARENT = True
sns.set_style("whitegrid")
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
mpl.style.use("mocha")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
