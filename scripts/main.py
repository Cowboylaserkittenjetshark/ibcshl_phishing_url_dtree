import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split
from scipy.stats import randint # For randomized search
from joblib import parallel_backend
from common import X, y

X = pd.get_dummies(X, columns = ['tld'], drop_first = True)
X.drop(['google_index', 'url', 'suspecious_tld', 'domain_age', 'web_traffic'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# clf = DecisionTreeClassifier()
# clf = RandomRandomForestClassifier()
clf = HistGradientBoostingClassifier(
    learning_rate=0.1,
    random_state=42,
    max_depth=8
)
clf = clf.fit(X_train, y_train)

# tree.plot_tree(clf)
print("model score: %.3f" % clf.score(X_test, y_test))
print(cross_val_score(clf, X_test, y_test, cv=7))
print(metrics.classification_report(y_test, clf.predict(X_test)))

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
def sortSecond(val):
    return val[1]
values = clf.feature_importances_
features = list(X)
importances = [(features[i], values[i]) for i in range(len(features))]
importances.sort(reverse=True, key=sortSecond)
print(importances)
exit(1)

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
