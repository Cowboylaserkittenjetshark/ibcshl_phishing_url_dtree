import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from common import TRANSPARENT
from common.data import data
from paths import output


def make_graph():
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    svm = sns.heatmap(
        corr, annot=False,
        # annot_kws={"size": 8},
        # mask = mask,
    )  # Heatmap without categories that will be one hot encoded
    figure = svm.get_figure()

    plt.savefig(
        output.DIR.joinpath("heatmap.svg"),
        bbox_inches="tight",
        transparent=TRANSPARENT,
    )
    plt.savefig(
        output.DIR.joinpath("heatmap.png"),
        bbox_inches="tight",
        transparent=TRANSPARENT,
    )
    
if __name__ == "__main__":
    make_graph()
