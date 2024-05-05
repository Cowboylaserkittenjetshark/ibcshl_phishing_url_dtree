import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_graph(X, y):
    tlds = X.tld
    y = pd.get_dummies(y)

    df = pd.concat([tlds, y], axis = 1)
    df = df.groupby(by = 'tld').sum()
    # Use [^0-9:] to drop IPs and IPs with ports specified
    df = df.filter(axis = 0, regex = "[^0-9:]")
    df['total']= df.phishing + df.legitimate
    df = df[df.total > 50]
    df["pct_phish"] = (df.phishing / df.total) * 100
    # print(df.loc['ru'])
    df.sort_values(by = ['pct_phish', 'total'], ascending = True, inplace = True)
    print(df)
    ax = sns.barplot(data=df, x='tld', y='pct_phish',orient="h")
    ax.set_xlabel('Top (and Second) Level Domain')
    ax.set_ylabel('Percent Phishing')
    plt.show()
