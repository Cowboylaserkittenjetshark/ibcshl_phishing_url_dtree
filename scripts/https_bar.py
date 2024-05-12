import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def make_graph(X, y):
    is_https = X.is_https
    y = pd.get_dummies(y)

    df = pd.concat([is_https, y], axis=1)
    df = df.groupby(by="is_https").sum()
    df["total"] = df.phishing + df.legitimate
    df["pct_phish"] = (df.phishing / df.total) * 100
    df.sort_values(
        by=["pct_phish", "total"],
        ascending=False,
        inplace=True,
    )
    print(df)
    ax = sns.barplot(data=df, x="is_https", y="pct_phish")
    ax.set_xlabel("HTTPS")
    ax.set_ylabel("Percent Phishing")
    plt.show()
