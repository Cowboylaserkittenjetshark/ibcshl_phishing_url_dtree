import pathlib
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT = ROOT.joinpath("output/")
OUTPUT.mkdir(exist_ok=True)
DATA = ROOT.joinpath("data/")
DATA.mkdir(exist_ok=True)
FEAT_DATA_FILE = DATA.joinpath("feature.csv")
TARG_DATA_FILE = DATA.joinpath("target.csv")

X = None
y = None

if FEAT_DATA_FILE.is_file() and TARG_DATA_FILE.is_file():
    X = pd.read_csv(FEAT_DATA_FILE)
    y = pd.read_csv(TARG_DATA_FILE)
else:
    # fetch dataset
    phiusiil_phishing_url_website = fetch_ucirepo(id=967) 
  
    # data (as pandas dataframes) 
    X = phiusiil_phishing_url_website.data.features 
    y = phiusiil_phishing_url_website.data.targets

    # save to prevent redownloads
    X.to_csv(FEAT_DATA_FILE, encoding='utf-8', index=False)
    y.to_csv(TARG_DATA_FILE, encoding='utf-8', index=False)
  
    # metadata 
    print(phiusiil_phishing_url_website.metadata) 
  
    # variable information 
    print(phiusiil_phishing_url_website.variables) 
