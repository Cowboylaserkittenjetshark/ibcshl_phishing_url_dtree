import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
# Start KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# End KNN
from scipy.stats import randint # For randomized search
from xgboost import XGBClassifier
from common import X, y

features = [
    # "url",
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_or",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain", # Possibly arbitrary
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "path_extension",
    "nb_redirection",
    "nb_external_redirection",
    "length_words_raw",
    "char_repeat",
    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",
    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",
    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",
    "phish_hints",
    "domain_in_brand",
    "brand_in_subdomain",
    "brand_in_path",
    "suspecious_tld", # Seems arbitrary
    "statistical_report",
    "nb_hyperlinks",
    "ratio_intHyperlinks",
    "ratio_extHyperlinks",
    "ratio_nullHyperlinks",
    "nb_extCSS",
    "ratio_intRedirection",
    "ratio_extRedirection",
    "ratio_intErrors",
    "ratio_extErrors",
    "login_form",
    "external_favicon",
    "links_in_tags",
    "submit_email",
    "ratio_intMedia",
    "ratio_extMedia",
    "sfh",
    "iframe",
    "popup_window",
    "safe_anchor",
    "onmouseover",
    "right_clic",
    "empty_title",
    "domain_in_title",
    "domain_with_copyright",
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]
X = X[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

knn_clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsClassifier(
                # n_neighbors=19, metric="manhattan", weights="uniform", leaf_size=15
            ),
        ),
    ]
)

knn_clf.fit(X_train, y_train)
print("knn model score: %.3f" % knn_clf.score(X_test, y_test))
print(metrics.classification_report(y_test, knn_clf.predict(X_test)))

# clf = DecisionTreeClassifier()
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

# tree.plot_tree(clf)
print("model score: %.3f" % clf.score(X_test, y_test))
print(cross_val_score(clf, X_valid, y_valid, cv=7))

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(low=4, high=40),
    'max_leaf_nodes': randint(low=1000, high=20000),
    'min_samples_leaf': randint(low=20, high=100),
    'min_samples_split': randint(low=40, high=200)
}

def sortSecond(val):
    return val[1]
values = clf.feature_importances_
features = list(X)
importances = [(features[i], values[i]) for i in range(len(features))]
importances.sort(reverse=True, key=sortSecond)

X_train = X_train[[col[0] for col in importances[:15]]]
X_test  = X_test[[col[0] for col in importances[:15]]]
X_valid = X_valid[[col[0] for col in importances[:15]]]

# cut_clf = DecisionTreeClassifier()
cut_clf = RandomForestClassifier()
cut_clf.fit(X_train, y_train)
print("Cut: ", metrics.accuracy_score(y_valid, cut_clf.predict(X_valid)))

# clf_tuned = DecisionTreeClassifier(random_state=42)
clf_tuned = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(clf_tuned, params, cv=7)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
best_tuned_clf = random_search.best_estimator_
print("Cut & tuned: ", metrics.accuracy_score(y_valid, best_tuned_clf.predict(X_valid)))

print(metrics.classification_report(y_test, cut_clf.predict(X_test)))
print(metrics.classification_report(y_test, best_tuned_clf.predict(X_test)))

# bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# bst.fit(X_train, y_train)
# print(metrics.classification_report(y_test, bst.predict(X_test)))
