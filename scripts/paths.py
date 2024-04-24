import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT = ROOT.joinpath("output/")
OUTPUT.mkdir(exist_ok=True)

class data:
    DATA = ROOT.joinpath("data/")
    DATA.mkdir(exist_ok=True)
    FEAT_FILE = DATA.joinpath("feature.csv")
    TARG_FILE = DATA.joinpath("target.csv")
    
