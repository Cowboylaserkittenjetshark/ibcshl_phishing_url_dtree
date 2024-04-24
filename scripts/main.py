import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import logging, sys
from common import X, y

features = [
    # "URL",
    # "URLLength",
    # "Domain",
    # "DomainLength",
    "IsDomainIP",
    # "TLD", # USE
    # "URLSimilarityIndex",
    # "CharContinuationRate",
    # "TLDLegitimateProb",
    # "URLCharProb",
    # "TLDLength",
    # "NoOfSubDomain",
    "HasObfuscation",
    # "NoOfObfuscatedChar",
    # "ObfuscationRatio",
    # "NoOfLettersInURL",
    # "LetterRatioInURL",
    # "NoOfDegitsInURL",
    # "DegitRatioInURL",
    # "NoOfEqualsInURL",
    # "NoOfQMarkInURL",
    # "NoOfAmpersandInURL",
    # "NoOfOtherSpecialCharsInURL",
    # "SpacialCharRatioInURL",
    "IsHTTPS",
    # "LineOfCode",
    # "LargestLineLength",
    "HasTitle",
    # "Title",
    # "DomainTitleMatchScore",
    # "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    # "NoOfURLRedirect",
    # "NoOfSelfRedirect",
    "HasDescription",
    # "NoOfPopup",
    # "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    # "NoOfImage",
    # "NoOfCSS",
    # "NoOfJS",
    # "NoOfSelfRef",
    # "NoOfEmptyRef",
    # "NoOfExternalRef",
]

logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

X.drop(columns=X.columns.difference(features), inplace=True)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# tree.plot_tree(clf)
print("model score: %.3f" % clf.score(X_test, y_test))


