import matplotlib.pyplot as plt
from common import TRANSPARENT
from paths import output


def make_graph(X, y):
    counts = y.value_counts()
    print(counts)
    plt.pie(counts, labels=["Phishing", "Legitimate"])

    plt.savefig(
        output.PIE_DIR.joinpath("balance.svg"),
        bbox_inches="tight",
        transparent=TRANSPARENT,
    )
    plt.savefig(
        output.PIE_DIR.joinpath("balance.png"),
        bbox_inches="tight",
        transparent=TRANSPARENT,
    )
