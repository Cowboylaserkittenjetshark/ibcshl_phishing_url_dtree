import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()

class data:
    DIR = ROOT.joinpath("data/")
    DIR.mkdir(exist_ok=True)
    # TRAIN_FILE = DIR.joinpath("Training.parquet")
    # TEST_FILE = DIR.joinpath("Testing.parquet")
    FILE = DIR.joinpath("dataset_cybersecurity_michelle.csv")
    EXTRA_FILE = DIR.joinpath("malicious_phish.csv")

class output:
    DIR = ROOT.joinpath("output/")
    DIR.mkdir(exist_ok=True)
    PIE_DIR = DIR.joinpath("pie/")
    PIE_DIR.mkdir(exist_ok=True)
