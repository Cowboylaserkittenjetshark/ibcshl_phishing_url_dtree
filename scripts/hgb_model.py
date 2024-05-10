from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from joblib import parallel_backend
from common import X, y, fetched_features
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

def feature_importance(model):
    def sortSecond(val):
        return val[1]

    values = model.feature_importances_
    features = list(X)
    importances = [(features[i], values[i]) for i in range(len(features))]
    importances.sort(reverse=True, key=sortSecond)
    return importances

with parallel_backend('threading', n_jobs=-1):
    logging.debug('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
    test_size=0.2)

    logging.info('==== Histogram Gradient Boosting ====')
    hgb_clf = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)

    hgb_score = hgb_clf.score(X_valid, y_valid)
    hgb_cross_val_score = cross_val_score(hgb_clf, X_train, y_train, cv=7)
    hgb_report = classification_report(y_test, hgb_clf.predict(X_test))

    logging.info('Score: %s\n', hgb_score)
    logging.info('Cross val score: %s\n', hgb_cross_val_score)
    logging.info('Classification report:\n%s\n', hgb_report)
    
    logging.info('==== Histogram Gradient Boosting without fetched features ====')
    X_train = X_train.drop(fetched_features, axis = 1)
    X_test = X_test.drop(fetched_features, axis = 1)
    X_valid = X_valid.drop(fetched_features, axis = 1)
    hgb_clf = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)

    hgb_score = hgb_clf.score(X_valid, y_valid)
    hgb_cross_val_score = cross_val_score(hgb_clf, X_train, y_train, cv=7)
    hgb_report = classification_report(y_test, hgb_clf.predict(X_test))

    logging.info('Score: %s\n', hgb_score)
    logging.info('Cross val score: %s\n', hgb_cross_val_score)
    logging.info('Classification report:\n%s\n', hgb_report)
