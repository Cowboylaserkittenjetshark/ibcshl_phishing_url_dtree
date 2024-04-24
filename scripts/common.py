import matplotlib as mpl
from ucimlrepo import fetch_ucirepo
import logging
import pandas as pd
import mplcatppuccin
import seaborn as sns
from paths import data

# Plot style
TRANSPARENT = False
sns.set_style("whitegrid")
mpl.style.use("mocha")

X = None
y = None

if data.FEAT_FILE.is_file() and data.TARG_FILE.is_file():
    logging.info('Using cached dataset files')
    X = pd.read_csv(data.FEAT_FILE)
    y = pd.read_csv(data.TARG_FILE)
else:
    logging.info('Dataset files not found.\nFetching ...')
    # fetch dataset
    phiusiil_phishing_url_website = fetch_ucirepo(id=967) 
  
    # data (as pandas dataframes) 
    X = phiusiil_phishing_url_website.data.features 
    y = phiusiil_phishing_url_website.data.targets

    # save to prevent redownloads
    X.to_csv(data.FEAT_FILE, encoding='utf-8', index=False)
    y.to_csv(data.TARG_FILE, encoding='utf-8', index=False)
    logging.info('Succesfully cached dataset files.')
  
    # metadata 
    print(phiusiil_phishing_url_website.metadata) 
  
    # variable information 
    print(phiusiil_phishing_url_website.variables) 
