import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn import tree
from sklearn.model_selection import train_test_split
import logging, sys
import paths

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

X = None
y = None

if paths.data.FEAT_FILE.is_file() and paths.data.TARG_FILE.is_file():
    logging.info('Using cached dataset files')
    X = pd.read_csv(paths.data.FEAT_FILE)
    y = pd.read_csv(paths.data.TARG_FILE)
else:
    logging.info('Dataset files not found.\nFetching ...')
    # fetch dataset
    phiusiil_phishing_url_website = fetch_ucirepo(id=967) 
  
    # data (as pandas dataframes) 
    X = phiusiil_phishing_url_website.data.features 
    y = phiusiil_phishing_url_website.data.targets

    # save to prevent redownloads
    X.to_csv(paths.data.FEAT_FILE, encoding='utf-8', index=False)
    y.to_csv(paths.data.TARG_FILE, encoding='utf-8', index=False)
    logging.info('Succesfully cached dataset files.')
  
    # metadata 
    print(phiusiil_phishing_url_website.metadata) 
  
    # variable information 
    print(phiusiil_phishing_url_website.variables) 

X.drop(columns=X.columns.difference(features), inplace=True)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# tree.plot_tree(clf)
print("model score: %.3f" % clf.score(X_test, y_test))


