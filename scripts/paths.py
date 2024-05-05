import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT = ROOT.joinpath("output/")
OUTPUT.mkdir(exist_ok=True)

class data:
    DIR = ROOT.joinpath("data/")
    DIR.mkdir(exist_ok=True)
    # TRAIN_FILE = DIR.joinpath("Training.parquet")
    # TEST_FILE = DIR.joinpath("Testing.parquet")
    FILE = DIR.joinpath("dataset_cybersecurity_michelle.csv")
