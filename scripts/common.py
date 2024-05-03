import matplotlib as mpl
import pandas as pd
import mplcatppuccin
import seaborn as sns
from paths import data

# Plot style
TRANSPARENT = False
sns.set_style("whitegrid")
mpl.style.use("mocha")

train = pd.read_parquet(data.TRAIN_FILE)
test = pd.read_parquet(data.TEST_FILE)
data = pd.concat([train, test])
y = data['status']
X = data.drop('status', axis = 1)
