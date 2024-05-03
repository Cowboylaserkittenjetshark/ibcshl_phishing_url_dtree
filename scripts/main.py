import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split
# Start KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# End KNN
from scipy.stats import randint # For randomized search
from xgboost import XGBClassifier
import tldextract
from joblib import parallel_backend
from common import X, y

X['tld'] = X.url.map(lambda url: tldextract.extract(url).suffix).rename('tld')
X.drop(['url', 'suspecious_tld'], axis = 1, inplace = True)
X = pd.get_dummies(X, columns = ['tld'], drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

# knn_clf = Pipeline(
#     steps=[
#         ("scaler", StandardScaler()),
#         (
#             "knn",
#             KNeighborsClassifier(
#                 # n_neighbors=19, metric="manhattan", weights="uniform", leaf_size=15
#             ),
#         ),
#     ]
# )

# knn_clf.fit(X_train, y_train)
# print("knn model score: %.3f" % knn_clf.score(X_test, y_test))
# print(metrics.classification_report(y_test, knn_clf.predict(X_test)))

# clf = DecisionTreeClassifier()
# clf = RandomRandomForestClassifier()
clf = HistGradientBoostingClassifier()
clf = clf.fit(X_train, y_train)

# tree.plot_tree(clf)
print("model score: %.3f" % clf.score(X_test, y_test))
print(cross_val_score(clf, X_test, y_test, cv=7))

hgb_params = {
    'max_iter':range(20,81,10),
    # 'min_samples_leaf':range(0,21,2)
}
hgb_gs = GridSearchCV(
    estimator=HistGradientBoostingClassifier(
        learning_rate=0.1,
        # min_samples_split=500,
        # min_samples_leaf=50,
        # max_depth=8,
        # max_features='sqrt',
        # subsample=0.8,
        random_state=10
    ),
    param_grid=hgb_params,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    verbose=5
)

with parallel_backend('threading', n_jobs=6):
    hgb_gs.fit(X_train, y_train)

    print(hgb_gs.best_estimator_)
    best_tuned_clf = hgb_gs.best_estimator_
    print("Tuned: ", metrics.accuracy_score(y_test, best_tuned_clf.predict(X_test)))

    # print(metrics.classification_report(y_test, cut_clf.predict(X_test)))
    print(metrics.classification_report(y_test, best_tuned_clf.predict(X_test)))

exit(1)
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(low=4, high=40),
    'max_leaf_nodes': randint(low=1000, high=20000),
    'min_samples_leaf': randint(low=20, high=100),
    'min_samples_split': randint(low=40, high=200)
}

# def sortSecond(val):
#     return val[1]
# values = clf.feature_importances_
# features = list(X)
# importances = [(features[i], values[i]) for i in range(len(features))]
# importances.sort(reverse=True, key=sortSecond)
# print([col[0] for col in importances[:15]])

# X_train = X_train[[col[0] for col in importances[:15]]]
# X_test  = X_test[[col[0] for col in importances[:15]]]
# X_valid = X_valid[[col[0] for col in importances[:15]]]

# cut_clf = DecisionTreeClassifier()
# cut_clf = RandomForestClassifier()
# cut_clf.fit(X_train, y_train)
# print("Cut: ", metrics.accuracy_score(y_test, cut_clf.predict(X_test)))

# clf_tuned = DecisionTreeClassifier(random_state=42)
# clf_tuned = RandomForestClassifier(random_state=42)
clf_tuned = HistGradientBoostingClassifier()

random_search = RandomizedSearchCV(clf_tuned, params, cv=7)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
best_tuned_clf = random_search.best_estimator_
print("Cut & tuned: ", metrics.accuracy_score(y_test, best_tuned_clf.predict(X_test)))

# print(metrics.classification_report(y_test, cut_clf.predict(X_test)))
print(metrics.classification_report(y_test, best_tuned_clf.predict(X_test)))

# bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# bst.fit(X_train, y_train)
# print(metrics.classification_report(y_test, bst.predict(X_test)))
