import matplotlib as mpl
import pandas as pd
import mplcatppuccin
import seaborn as sns
import paths
# Plot style
TRANSPARENT = False
sns.set_style("whitegrid")
mpl.style.use("mocha")

data = pd.read_csv(paths.data.FILE)
target_label = 'phishing'
y = data[target_label]
X = data.drop(target_label, axis = 1)
