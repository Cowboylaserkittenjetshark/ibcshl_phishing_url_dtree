import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tldextract
from collections import defaultdict
from common.data import extra_data
import common.setup


def make_graph():
    df = extra_data
    length = len(df)
    df["tld"] = df.url.map(lambda url: tldextract.extract(url).suffix).rename(
        "tld"
    )
    # Generalize to malicious or not (was benign, phishing, defacement, etc.)
    df.rename(columns={"type": "malicious"}, inplace=True)
    df["malicious"] = df["malicious"].map(
        defaultdict(lambda: 1, {"benign": 0})
    )
    df = pd.get_dummies(df, columns=["malicious"])
    df.rename(
        columns={
            "malicious_1": "malicious",
            "malicious_0": "benign",
        },
        inplace=True,
    )
    df.drop(["url"], inplace=True, axis=1)

    df = df.groupby(by="tld").sum()
    df["total"] = df.malicious + df.benign
    df = df[df.total > length * 0.01]
    df["pct_phish"] = (df.malicious / df.total) * 100
    # print(df.loc['ru'])
    df.sort_values(
        by=["pct_phish", "total"],
        ascending=False,
        inplace=True,
    )
    ax = sns.barplot(data=df, x="pct_phish", y="tld", orient="h")
    ax.set_ylabel("Top (and Second) Level Domain")
    ax.set_xlabel("Percent Phishing")
    plt.show()


if __name__ == "__main__":
    make_graph()
